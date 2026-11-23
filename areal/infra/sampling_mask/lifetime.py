# SPDX-License-Identifier: Apache-2.0
"""Workflow-owned mask leases, independent of policy version and queue latency.

Only small lease/source descriptors use HTTP. Candidate IDs still use GIN.
A lease is created before generation and closed after *all* batch consumers.
Absent leases reject registration, so cancelled requests cannot resurrect rows.
"""

import asyncio
import json
import secrets
from contextvars import ContextVar
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread

import requests

from areal.infra.sampling_mask.protocol import decode_endpoint, encode_endpoint
from areal.utils.network import gethostip

LEASE_KEY = "_sampling_mask_lease"
LEASES_KEY = "_sampling_mask_leases"
current_lease: ContextVar[tuple[int, int, int] | None] = ContextVar(
    "sampling_mask_lease", default=None
)


def validate_lease(value) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(type(x) is not int or x <= 0 for x in value)
    ):
        raise ValueError("Missing or malformed sampling-mask workflow lease")
    return tuple(value)


def call(source: int, nonce: int, path: str, /, **payload) -> dict:
    response = requests.post(
        decode_endpoint(source) + path, json={"nonce": nonce, **payload}, timeout=90
    )
    result = response.json()
    if response.status_code != 200:
        raise RuntimeError(f"Sampling-mask lifetime {path}: {result.get('error')}")
    return result


def release_lease(lease) -> None:
    source, nonce, lease_id = validate_lease(lease)
    call(source, nonce, "/release", lease_id=lease_id)


def release_batch(batch: list[dict | None]) -> None:
    # Keep metadata until acknowledgement, permitting retries on failure.
    trajectories = [traj for traj in batch if traj is not None]
    leases = {
        validate_lease(v) for traj in trajectories for v in traj.get(LEASES_KEY, [])
    }
    for lease in leases:
        release_lease(lease)
    for traj in trajectories:
        traj.pop(LEASES_KEY, None)


class LeaseRegistry:
    """Controller-owned admission fence and source registry for workflow leases.

    Monotonic IDs plus rejection of absent entries need no growing tombstones.
    Failed cleanup keeps a closed entry and its unacknowledged sources for retry.
    No age-based eviction is permitted while a consumer may still read a row.
    """

    def __init__(self):
        self.nonce = secrets.randbits(62) + 1
        self._lock = Lock()
        self._closed = False
        self._next = 1
        self._leases: dict[int, set[tuple[int, int]]] = {}
        self._closing: set[int] = set()
        registry = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 4096:
                        raise ValueError("Invalid sampling-mask lease message size")
                    body = json.loads(self.rfile.read(size))
                    if body.get("nonce") != registry.nonce:
                        raise ValueError(
                            "Sampling-mask lease owner generation mismatch"
                        )
                    if self.path == "/register":
                        result = {
                            "active": registry.register(
                                body["lease_id"], body["source"], body["source_nonce"]
                            )
                        }
                    elif self.path == "/release":
                        registry.release(body["lease_id"])
                        result = {"released": True}
                    else:
                        raise ValueError("Unknown sampling-mask lease operation")
                    encoded = json.dumps(result).encode()
                    self.send_response(200)
                except Exception as exc:
                    encoded = json.dumps({"error": str(exc)}).encode()
                    self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self._server = ThreadingHTTPServer((gethostip(), 0), Handler)
        self.source = encode_endpoint(*self._server.server_address)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def create(self) -> tuple[int, int, int]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Sampling-mask lease registry is closed")
            lease_id = self._next
            self._next += 1
            self._leases[lease_id] = set()
        return self.source, self.nonce, lease_id

    def register(self, lease_id: int, source: int, nonce: int) -> bool:
        with self._lock:
            if lease_id not in self._leases or lease_id in self._closing:
                return False
            self._leases[lease_id].add((source, nonce))
            return True

    def release(self, lease_id: int) -> None:
        with self._lock:
            if lease_id not in self._leases:
                return
            self._closing.add(lease_id)
            sources = tuple(self._leases[lease_id])
        # Never hold the registry lock across a producer callback: a writer
        # may concurrently register, and must observe the closed admission.
        for source, nonce in sources:
            call(
                source,
                nonce,
                "/release_lease",
                lease=[self.source, self.nonce, lease_id],
            )
            with self._lock:
                if lease_id in self._leases:
                    self._leases[lease_id].discard((source, nonce))
        with self._lock:
            self._leases.pop(lease_id, None)
            self._closing.discard(lease_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            lease_ids = list(self._leases)
            self._closing.update(lease_ids)
        errors = []
        try:
            for lease_id in lease_ids:
                try:
                    self.release(lease_id)
                except Exception as exc:
                    errors.append(exc)
        finally:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join()
        if errors:
            raise ExceptionGroup("Sampling-mask lease shutdown failed", errors)


async def arelease(lease) -> None:
    # Workflow cancellation must finish fencing even when its caller is gone.
    task = asyncio.create_task(asyncio.to_thread(release_lease, lease))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise

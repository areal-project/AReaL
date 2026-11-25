# SPDX-License-Identifier: Apache-2.0
"""Per-sampling-owner bounded CSR cache and metadata-only HTTP control plane."""

import json
import secrets
from array import array
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Lock, Semaphore, Thread

import numpy as np
import torch

from areal.infra.sampling_mask.gin import GinEndpoint, load_extension
from areal.infra.sampling_mask.lifetime import call, validate_lease
from areal.infra.sampling_mask.pages import PAGE_ROWS, SharedPageWriter
from areal.infra.sampling_mask.protocol import (
    encode_endpoint,
    pack_frame,
    unpack_row_ranges,
)
from areal.infra.sampling_mask.store import RowStore, shared_namespace
from areal.utils.network import gethostip
from areal.utils.sampling_mask import PROTOCOL_VERSION, pack_uint24

SLOT_BYTES = 8 << 20
CACHE_BYTES = 30_000_000_000


class MaskProducer:
    """Own source rows until acknowledged; no GPU history grows with rollout length.

    Registered-window storage is allocated during sampler construction, before
    SGLang model/KV-cache sizing. Communicator registration occurs once its actor
    peer is available. Each source has its own endpoint, including DP attention
    sampling owners; no assumption that inference TP0 owns all samples is made.
    """

    def __init__(self, device: int, max_sessions: int):
        self.device = device
        self.nonce = secrets.randbits(62) + 1
        self.uuid = str(torch.cuda.get_device_properties(device).uuid)
        self._condition = Condition()
        self._next_row = 1
        self._store = (
            SharedPageWriter(CACHE_BYTES)
            if shared_namespace()
            else RowStore(CACHE_BYTES)
        )
        self._fatal_error = None
        self._persisted_until = 1
        self._leases: dict[tuple[int, int, int], array] = {}
        self._pending = Semaphore(2)
        self._copy_stream = torch.cuda.Stream(device=device)
        self._writer = ThreadPoolExecutor(max_workers=1)
        self._sessions = {}
        self._session_buffers = {}
        self._transfer_locks = {}
        self._session_lock = Lock()
        # ncclMemAlloc allocation is separate from communicator registration.
        self._reserved = [
            load_extension().reserve(2 * SLOT_BYTES, device)
            for _ in range(max_sessions)
        ]
        producer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 4 << 20:
                        raise ValueError(
                            "Sampling-mask control message exceeds capacity"
                        )
                    request = json.loads(self.rfile.read(size))
                    if request.get("nonce") != producer.nonce:
                        raise ValueError("Sampling-mask producer generation mismatch")
                    if self.path == "/fetch":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/x-ndjson")
                        self.end_headers()

                        def announce(value):
                            self.wfile.write(json.dumps(value).encode() + b"\n")

                        try:
                            producer._fetch(request, announce)
                            announce(0)
                        except Exception as exc:
                            announce(f"{type(exc).__name__}: {exc}")
                        return
                    response = producer._handle(self.path, request)
                    encoded = json.dumps(response).encode()
                    self.send_response(200)
                except Exception as exc:
                    encoded = json.dumps(
                        {"error": f"{type(exc).__name__}: {exc}"}
                    ).encode()
                    self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self._server = ThreadingHTTPServer((gethostip(), 0), Handler)
        self.source = encode_endpoint(*self._server.server_address)
        Thread(target=self._server.serve_forever, daemon=True).start()

    def publish(self, ids, counts, tokens, versions, leases) -> list[list[int]]:
        """Enqueue bounded asynchronous D2H; return only CPU-generated row handles."""
        leases = [validate_lease(lease) for lease in leases]
        if len(leases) != ids.shape[0]:
            raise ValueError("Sampling-mask leases do not align with sampler batch")
        self._pending.acquire()
        with self._condition:
            if self._fatal_error is not None:
                self._pending.release()
                raise RuntimeError(
                    f"Sampling-mask producer failed: {self._fatal_error}"
                )
        batch_size = ids.shape[0]
        if len(versions) != batch_size:
            self._pending.release()
            raise ValueError("Sampling mask versions do not align with sampler batch")
        with self._condition:
            start = self._next_row
            self._next_row += batch_size
        try:
            stream = self._copy_stream
            stream.wait_stream(torch.cuda.current_stream(self.device))
            # Scheduler-owned outputs may be freed as soon as publish returns.
            for tensor in (ids, counts, tokens):
                tensor.record_stream(stream)
            with torch.cuda.stream(stream):
                packed = pack_uint24(ids)
                host_ids = torch.empty_like(packed, device="cpu", pin_memory=True)
                host_counts = torch.empty_like(counts, device="cpu", pin_memory=True)
                host_tokens = torch.empty_like(tokens, device="cpu", pin_memory=True)
                host_ids.copy_(packed, non_blocking=True)
                host_counts.copy_(counts, non_blocking=True)
                host_tokens.copy_(tokens, non_blocking=True)
                done = stream.record_event()
        except Exception as exc:
            with self._condition:
                self._fatal_error = str(exc)
                self._condition.notify_all()
            self._pending.release()
            raise

        def persist():
            try:
                done.synchronize()
                # Compact a whole sampler batch into one owned CSR allocation.
                valid = torch.arange(host_ids.shape[1])[None, :] < host_counts[:, None]
                compact = host_ids[valid]
                offsets = torch.cat([host_counts.new_zeros(1), host_counts.cumsum(0)])
                self._persist_rows(
                    np.arange(start, start + batch_size),
                    compact,
                    offsets,
                    host_tokens.long(),
                    torch.tensor(versions, dtype=torch.int64),
                    leases,
                )
                with self._condition:
                    self._persisted_until = start + batch_size
                    self._condition.notify_all()
            except Exception as exc:
                with self._condition:
                    self._fatal_error = str(exc)
                    self._condition.notify_all()
            finally:
                self._pending.release()

        try:
            self._writer.submit(persist)
        except Exception as exc:
            with self._condition:
                self._fatal_error = str(exc)
                self._condition.notify_all()
            self._pending.release()
            raise
        return [
            [self.source, self.nonce, start + i, PROTOCOL_VERSION, versions[i]]
            for i in range(batch_size)
        ]

    def _persist_rows(self, rows, packed, offsets, tokens, versions, leases):
        # Only the single writer registers sources. Install local admission
        # *before* the callback so a concurrent release can fence this writer.
        active = set()
        for lease in set(leases):
            with self._condition:
                known = lease in self._leases
                if not known:
                    self._leases[lease] = array("q")
            if not known:
                owner, nonce, lease_id = lease
                admitted = call(
                    owner,
                    nonce,
                    "/register",
                    lease_id=lease_id,
                    source=self.source,
                    source_nonce=self.nonce,
                )["active"]
                if not admitted:
                    with self._condition:
                        self._leases.pop(lease, None)
                    continue
            active.add(lease)
        with self._condition:
            # Serialize admission and append with the final release callback.
            keep = np.array(
                [lease in active and lease in self._leases for lease in leases]
            )
            if not keep.any():
                return
            if keep.all():
                self._store.add(rows, packed, offsets, tokens, versions)
            else:
                lengths = np.diff(offsets.numpy())
                self._store.add(
                    rows[keep],
                    packed[np.repeat(keep, lengths)],
                    torch.from_numpy(np.r_[np.int64(0), lengths[keep].cumsum()]),
                    tokens[keep],
                    versions[keep],
                )
            for row, lease, live in zip(rows, leases, keep, strict=True):
                if live:
                    self._leases[lease].append(int(row))

    def _release_lease(self, lease):
        with self._condition:
            rows = self._leases.pop(validate_lease(lease), None)
            if rows:
                self._store.delete(np.frombuffer(rows, dtype=np.int64))

    def _fetch(self, request, announce):
        rows = unpack_row_ranges(request["rows"])
        peer = request["peer"]
        with self._transfer_locks[peer]:
            with self._condition:
                self._wait_for_row(int(rows[-1]))
                counts = self._store.counts(rows)
            # Bound host assembly as well as the GIN windows. A single row
            # can fill a slot with IDs; its metadata then uses a second put.
            width = np.min_scalar_type(int(counts.max())).itemsize
            ends = np.r_[np.int64(0), (counts * 3 + width + 12).cumsum()]
            start = 0
            while start < len(rows):
                end = max(
                    start + 1,
                    int(ends.searchsorted(ends[start] + SLOT_BYTES - 8, side="right"))
                    - 1,
                )
                with self._condition:
                    data = self._store.read(rows[start:end])
                frame = pack_frame(*data)
                announce(frame.numel())
                for chunk in frame.split(SLOT_BYTES):
                    self._sessions[peer].transfer(chunk)
                start = end

    def _wait_for_row(self, row):
        if not 0 < row < self._next_row:
            raise ValueError("Unknown sampling-mask row")
        ready = self._condition.wait_for(
            lambda: self._fatal_error is not None or row < self._persisted_until,
            timeout=60,
        )
        if self._fatal_error is not None:
            raise RuntimeError(self._fatal_error)
        if not ready:
            raise TimeoutError("Sampling-mask source persistence timed out")

    def _handle(self, path: str, request: dict) -> dict:
        if path == "/release_lease":
            self._release_lease(request["lease"])
            return {"released": True}
        if path == "/ready":
            with self._condition:
                self._wait_for_row(request["through"])
            return {}
        if path == "/info":
            return {
                "page_directory": (
                    {"layout": 1, "prefix": self._store.prefix, "rows": PAGE_ROWS}
                    if isinstance(self._store, SharedPageWriter)
                    else None
                ),
                "uuid": self.uuid,
                "shared_namespace": shared_namespace(),
                "protocol": PROTOCOL_VERSION,
            }
        if path == "/connect":
            peer = request["peer"]
            if peer == self.uuid:
                raise ValueError(
                    "GIN communicator peers must use distinct physical GPUs"
                )
            with self._session_lock:
                if peer in self._sessions:
                    raise ValueError("Sampling-mask peer session already registered")
                if not self._reserved:
                    raise MemoryError("Sampling-mask registered-window pool exhausted")
                pointer = self._reserved.pop()
                try:
                    endpoint = GinEndpoint(
                        bytes.fromhex(request["unique_id"]),
                        0,
                        self.device,
                        SLOT_BYTES,
                        pointer,
                    )
                except Exception:
                    self._reserved.append(pointer)
                    raise
                self._sessions[peer] = endpoint
                self._session_buffers[peer] = pointer
                self._transfer_locks[peer] = Lock()
            return {"connected": True}
        if path == "/close":
            peer = request["peer"]
            with self._session_lock:
                endpoint = self._sessions[peer]
                endpoint.close()
                del self._sessions[peer]
                self._reserved.append(self._session_buffers.pop(peer))
                del self._transfer_locks[peer]
            return {"closed": True}
        raise ValueError("Unknown sampling-mask control operation")

    def close(self):
        """Stop after consumers have closed their registered GIN sessions."""
        with self._session_lock:
            if self._sessions:
                raise RuntimeError("Close sampling-mask consumers before the producer")
        self._writer.shutdown(wait=True)
        self._server.shutdown()
        self._server.server_close()
        with self._condition:
            self._store.close()
        for pointer in self._reserved:
            load_extension().free_reserved(pointer, self.device)
        self._reserved.clear()

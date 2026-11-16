# SPDX-License-Identifier: Apache-2.0
"""Per-sampling-owner bounded CSR cache and metadata-only HTTP control plane."""

import hashlib
import json
import secrets
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Condition, Lock, Semaphore, Thread

import numpy as np
import torch

from areal.infra.sampling_mask.gin import GinEndpoint, load_extension
from areal.infra.sampling_mask.protocol import encode_endpoint
from areal.infra.sampling_mask.store import RowStore
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

    def __init__(self, device: int, max_sessions: int, retained_versions: int = 2):
        self.device = device
        self.nonce = secrets.randbits(62) + 1
        self.uuid = str(torch.cuda.get_device_properties(device).uuid)
        self._condition = Condition()
        self._next_row = 1
        self._store = RowStore(CACHE_BYTES)
        self._fatal_error = None
        self._persisted_until = 1
        self._retained_versions = retained_versions
        self._expiry_watermark = -1
        self._pending = Semaphore(2)
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

    def publish(self, ids, counts, tokens, versions) -> list[list[int]]:
        """Enqueue bounded asynchronous D2H; return only CPU-generated row handles."""
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
            packed = pack_uint24(ids)
            host_ids = torch.empty_like(packed, device="cpu", pin_memory=True)
            host_counts = torch.empty_like(counts, device="cpu", pin_memory=True)
            host_tokens = torch.empty_like(tokens, device="cpu", pin_memory=True)
            host_ids.copy_(packed, non_blocking=True)
            host_counts.copy_(counts, non_blocking=True)
            host_tokens.copy_(tokens, non_blocking=True)
            done = torch.cuda.Event()
            done.record()
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
                with self._condition:
                    expiry = min(versions) - self._retained_versions
                    if expiry > self._expiry_watermark:
                        self._store.discard_before(expiry)
                        self._expiry_watermark = expiry
                    self._store.add(
                        np.arange(start, start + batch_size),
                        compact,
                        offsets,
                        host_tokens.long(),
                        torch.tensor(versions, dtype=torch.int64),
                    )
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

    def _lookup(self, row_ids):
        if len(row_ids) > 65536 or len(set(row_ids)) != len(row_ids):
            raise ValueError("Invalid sampling-mask row request")
        rows = np.asarray(row_ids, dtype=np.int64)
        with self._condition:
            if np.any(rows <= 0) or np.any(rows >= self._next_row):
                raise ValueError("Unknown sampling-mask row")
            ready = self._condition.wait_for(
                lambda: self._fatal_error is not None
                or np.all(rows < self._persisted_until),
                timeout=60,
            )
            if self._fatal_error is not None:
                raise RuntimeError(self._fatal_error)
            if not ready:
                raise TimeoutError("Sampling-mask source persistence timed out")
            self._store.compact()
            return self._store.describe(rows)

    def _handle(self, path: str, request: dict) -> dict:
        if path == "/info":
            return {
                "uuid": self.uuid,
                "protocol": PROTOCOL_VERSION,
                "slot_bytes": SLOT_BYTES,
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
        counts, tokens, versions = self._lookup(request["rows"])
        if path == "/describe":
            return {
                "counts": counts.tolist(),
                "tokens": tokens.tolist(),
                "versions": versions.tolist(),
                "protocol": PROTOCOL_VERSION,
            }
        if path == "/send":
            total = int(counts.sum()) * 3
            if total != request["bytes"] or total > SLOT_BYTES:
                raise ValueError("Sampling-mask chunk descriptor size mismatch")
            # Acquire before allocating staging tensors, so queued callers
            # cannot multiply per-peer CPU/GPU scratch usage.
            with self._transfer_locks[request["peer"]]:
                with self._condition:
                    payload = self._store.read(request["rows"])[0].reshape(-1)
                checksum = hashlib.sha256(memoryview(payload.numpy())).hexdigest()
                endpoint = self._sessions[request["peer"]]
                with torch.cuda.device(self.device):
                    device_payload = payload.to(device=self.device)
                    endpoint.transfer(device_payload)
            return {"bytes": total, "sha256": checksum}
        if path == "/release":
            # A parent prefix can appear in multiple training shards. One
            # shard's acknowledgement cannot invalidate another shard's handle.
            # Retain source rows until their policy version expires; per-shard
            # caches prevent retransmission on subsequent PPO epochs.
            return {"released": len(request["rows"])}
        raise ValueError("Unknown sampling-mask control operation")

    def close(self):
        """Stop after consumers have closed their registered GIN sessions."""
        with self._session_lock:
            if self._sessions:
                raise RuntimeError("Close sampling-mask consumers before the producer")
        self._writer.shutdown(wait=True)
        self._server.shutdown()
        self._server.server_close()
        for pointer in self._reserved:
            load_extension().free_reserved(pointer, self.device)
        self._reserved.clear()

# SPDX-License-Identifier: Apache-2.0
"""Training-shard CPU cache; each support crosses the rollout boundary once."""

import hashlib
import weakref
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace

import numpy as np
import requests
import torch
import torch.distributed as dist

from areal.infra.sampling_mask.gin import GinEndpoint, load_extension
from areal.infra.sampling_mask.protocol import REFERENCE_KEYS, decode_endpoint
from areal.infra.sampling_mask.service import CACHE_BYTES, SLOT_BYTES
from areal.infra.sampling_mask.store import RowStore, shared_namespace
from areal.utils.sampling_mask import (
    PROTOCOL_VERSION,
    UPLOAD_SLOT_BYTES,
    SamplingMask,
    unpack_uint24,
)


def control(source: int, nonce: int, path: str, **payload) -> dict:
    response = requests.post(
        decode_endpoint(source) + path, json={"nonce": nonce, **payload}, timeout=90
    )
    result = response.json()
    if response.status_code != 200:
        raise RuntimeError(
            f"Sampling-mask control {path}: {result.get('error', response.status_code)}"
        )
    body = response.request.body or b""
    result["_control_body_bytes"] = len(
        body.encode() if isinstance(body, str) else body
    ) + len(response.content)
    return result


class _UploadSlot:
    """A bounded packed H2D transfer, reusable once both Python and CUDA are done."""

    def __init__(self, device):
        self.host = torch.empty(UPLOAD_SLOT_BYTES, dtype=torch.uint8, pin_memory=True)
        self.stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(self.stream):
            self.device = torch.empty_like(self.host, device=device)
        self.done = torch.cuda.Event()
        self.owner = None

    def stage(self, mask):
        # Pipeline schedules can retain masks through backward. A busy slot
        # leaves the caller on the existing on-demand chunk upload path.
        if self.owner is not None and (
            self.owner() is not None or not self.done.query()
        ):
            return mask
        offsets = mask.offsets.view(torch.uint8)
        n = offsets.numel()
        end = n + mask.packed_ids.numel()
        if end > self.host.numel():
            return mask
        self.host[:n].copy_(offsets)
        self.host[n:end].copy_(mask.packed_ids.reshape(-1))
        with torch.cuda.stream(self.stream):
            self.device[:end].copy_(self.host[:end], non_blocking=True)
            self.done.record(self.stream)
        staged = replace(mask, slot=self)
        self.owner = weakref.ref(staged)
        return staged


@dataclass(frozen=True)
class LoadedSamplingMask:
    packed_ids: torch.Tensor
    offsets: torch.Tensor
    vocab_size: int
    slot: _UploadSlot | None = None

    def chunk(
        self, start: int, end: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        begin, finish = int(self.offsets[start]), int(self.offsets[end])
        if self.slot is not None and device == self.slot.device.device:
            slot = self.slot
            stream = torch.cuda.current_stream(device)
            stream.wait_event(slot.done)
            slot.device.record_stream(stream)
            n = self.offsets.numel() * self.offsets.element_size()
            ids = unpack_uint24(slot.device[n + 3 * begin : n + 3 * finish].view(-1, 3))
            offsets = slot.device[:n].view(torch.int64)[start : end + 1] - begin
            slot.done.record(stream)
            return ids, offsets
        ids = unpack_uint24(
            self.packed_ids[begin:finish].to(device=device, non_blocking=True)
        )
        offsets = (self.offsets[start : end + 1] - begin).to(
            device=device, non_blocking=True
        )
        return ids, offsets


class SamplingMaskConsumer:
    def __init__(self, device: torch.device, tp_group, vocab_size: int):
        self.device = device
        self.group = tp_group
        self.world = dist.get_world_size(tp_group) if tp_group is not None else 1
        self.rank = dist.get_rank(tp_group) if tp_group is not None else 0
        self.vocab_size = vocab_size
        self.uuid = str(torch.cuda.get_device_properties(device).uuid)
        peers = [(self.uuid, shared_namespace())] * self.world
        if self.world > 1:
            dist.all_gather_object(peers, peers[0], group=tp_group)
        self.uuids, namespaces = zip(*peers, strict=True)
        self.shared_namespace = namespaces[0] if len(set(namespaces)) == 1 else None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._reader = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mask-reader"
        )
        self._pending_read = None
        self._upload = _UploadSlot(self.device)
        self.shared_descriptors = 0
        self._sessions = {}
        self._stores = {}
        self._cache_bytes = 0
        self.id_bytes = 0
        self.offset_bytes = 0
        self.control_bytes = 0
        self._candidate_histogram = np.zeros(vocab_size + 1, dtype=np.int64)

    def _control(self, *args, **kwargs):
        result = control(*args, **kwargs)
        self.control_bytes += result.pop("_control_body_bytes")
        return result

    def metrics(self) -> dict[str, float]:
        """Shard-local counts; wire counters exclude transport headers/retries."""
        histogram = self._candidate_histogram
        total = int(histogram.sum())
        cumulative = histogram.cumsum()
        present = np.flatnonzero(histogram)
        values = {
            "unique_generated_tokens": total,
            "candidate_mean": float(histogram @ np.arange(len(histogram)))
            / max(total, 1),
            "candidate_p95": int(
                np.searchsorted(cumulative, max(1, np.ceil(total * 0.95)))
            )
            if total
            else 0,
            "candidate_p99": int(
                np.searchsorted(cumulative, max(1, np.ceil(total * 0.99)))
            )
            if total
            else 0,
            "candidate_max": int(present[-1]) if total else 0,
            "id_payload_bytes": self.id_bytes,
            "csr_offset_bytes": self.offset_bytes,
            "reference_tensor_bytes": total * len(REFERENCE_KEYS) * 8,
            "control_body_bytes": self.control_bytes,
            "gin_payload_bytes": sum(
                endpoint.payload_bytes for endpoint in self._sessions.values()
            ),
            "cpu_cache_bytes": self._cache_bytes,
        }
        return {
            f"sampling_mask/shard/{key}": float(value) for key, value in values.items()
        }

    def _broadcast_object(self, value, owner):
        objects = [value]
        if self.world > 1:
            dist.broadcast_object_list(
                objects, src=dist.get_global_rank(self.group, owner), group=self.group
            )
        return objects[0]

    def _owner_call(self, owner, function, *args, **kwargs):
        envelope = None
        if self.rank == owner:
            try:
                envelope = {"result": function(*args, **kwargs)}
            except Exception as exc:
                envelope = {"error": f"{type(exc).__name__}: {exc}"}
        envelope = self._broadcast_object(envelope, owner)
        if "error" in envelope:
            raise RuntimeError(f"Sampling-mask owner failed: {envelope['error']}")
        return envelope["result"]

    def _receive(self, source, nonce, rows, owner, manifest):
        key = (source, nonce)
        if manifest["protocol"] != PROTOCOL_VERSION:
            raise ValueError("Sampling-mask manifest protocol version mismatch")
        counts = manifest["counts"]
        total_bytes = sum(counts) * 3
        if total_bytes > SLOT_BYTES:
            raise ValueError(
                "Sampling-mask requested chunk exceeds registered capacity"
            )
        device_payload = torch.empty(total_bytes, dtype=torch.uint8, device=self.device)

        def receive():
            endpoint = self._sessions.get(key)
            if endpoint is None:
                unique_id = load_extension().bootstrap_id()
                connecting = self._executor.submit(
                    self._control,
                    source,
                    nonce,
                    "/connect",
                    peer=self.uuid,
                    unique_id=unique_id.hex(),
                )
                try:
                    endpoint = GinEndpoint(unique_id, 1, self.device.index, SLOT_BYTES)
                except Exception as local_error:
                    # Preserve the peer's actionable setup error, rather than
                    # reporting only a local missing-peer timeout.
                    try:
                        connecting.result()
                    except Exception as peer_error:
                        raise RuntimeError(
                            f"Sampling-mask GIN local setup: {local_error}; "
                            f"peer setup: {peer_error}"
                        ) from local_error
                    raise
                connecting.result()
                self._sessions[key] = endpoint
            sending = self._executor.submit(
                self._control,
                source,
                nonce,
                "/send",
                peer=self.uuid,
                rows=rows,
                bytes=total_bytes,
            )
            endpoint.transfer(device_payload)
            expected = sending.result()["sha256"]
            actual = hashlib.sha256(
                memoryview(device_payload.cpu().numpy())
            ).hexdigest()
            if actual != expected:
                raise ValueError("Sampling-mask GIN payload checksum mismatch")
            return True

        self._owner_call(owner, receive)
        if self.world > 1:
            dist.broadcast(
                device_payload,
                src=dist.get_global_rank(self.group, owner),
                group=self.group,
            )
        packed = device_payload.cpu().reshape(-1, 3)
        offsets = torch.tensor([0, *counts], dtype=torch.int64).cumsum(0)
        tokens = torch.tensor(manifest["tokens"], dtype=torch.int64)
        versions = torch.tensor(manifest["versions"], dtype=torch.int64)
        # Protocol/weight identities are checked separately per token below;
        # validate all CSR structure and membership in one vectorized operation.
        SamplingMask(packed, offsets, self.vocab_size, 0).validate(
            tokens, weight_version=0
        )
        store = self._stores.setdefault(key, RowStore(CACHE_BYTES))
        store.capacity = CACHE_BYTES - self._cache_bytes + store.nbytes
        previous = store.nbytes
        store.add(rows, packed, offsets, tokens, versions)
        self._cache_bytes += store.nbytes - previous
        self.id_bytes += total_bytes
        self.offset_bytes += offsets.numel() * 8
        self._candidate_histogram += np.bincount(counts, minlength=self.vocab_size + 1)
        # Only the workflow lease owner releases source rows after every
        # training consumer has finished. A shard must not delete shared prefixes.

    def _index_shared(self, store, rows, manifest):
        if manifest["protocol"] != PROTOCOL_VERSION:
            raise ValueError("Sampling-mask manifest protocol version mismatch")
        counts = np.asarray(manifest["counts"], dtype=np.int64)
        tokens, versions = (
            torch.tensor(manifest[key], dtype=torch.int64)
            for key in ("tokens", "versions")
        )
        positions = np.asarray(
            [pos for _, group, _ in manifest["blocks"] for pos in group], dtype=np.int64
        )
        if (
            counts.shape != (len(rows),)
            or tokens.shape != counts.shape
            or versions.shape != counts.shape
            or np.any(counts <= 0)
            or np.any(counts > min(SLOT_BYTES // 3, self.vocab_size))
            or not np.array_equal(np.sort(positions), np.arange(len(rows)))
        ):
            raise ValueError("Invalid shared sampling-mask row descriptors")
        self.shared_descriptors += len(manifest["blocks"])
        for name, selected, starts in manifest["blocks"]:
            selected = np.asarray(selected, dtype=np.int64)
            starts = np.asarray(starts, dtype=np.int64)
            if starts.shape != selected.shape or np.any(starts < 0):
                raise ValueError("Invalid shared sampling-mask offsets")
            offsets = torch.from_numpy(np.concatenate(([0], counts[selected].cumsum())))
            store.capacity = CACHE_BYTES - self._cache_bytes + store.nbytes
            previous = store.nbytes
            store.index_shared(
                rows[selected],
                name,
                starts,
                offsets,
                tokens[selected],
                versions[selected],
                self.vocab_size,
                SLOT_BYTES,
            )
            self._cache_bytes += store.nbytes - previous
        self.id_bytes += int(counts.sum()) * 3
        self.offset_bytes += (len(rows) + 1) * 8
        self._candidate_histogram += np.bincount(counts, minlength=self.vocab_size + 1)

    def load(
        self, inputs: dict, labels: torch.Tensor, versions: torch.Tensor | None = None
    ) -> LoadedSamplingMask:
        self._drain_reader()
        return self._assemble(*self._resolve(inputs, labels, versions))

    def _drain_reader(self):
        # Only the scheduling thread changes batches. The caller owns the result
        # (and its exception); do not keep a completed mask alive through backward.
        future = self._pending_read() if self._pending_read is not None else None
        if future is not None and not future.done():
            wait([future])
        self._pending_read = None

    def load_async(self, inputs, labels, versions) -> Future[LoadedSamplingMask]:
        """One in-flight microbatch. Keep model TP collectives on the schedule thread."""
        self._drain_reader()
        resolved = self._resolve(inputs, labels, versions) if self.world > 1 else None

        def assemble_and_upload():
            # GIN at TP=1 may also stage CUDA tensors. Ordering its work on the
            # private stream keeps its synchronizations off the compute stream.
            with torch.cuda.device(self.device), torch.cuda.stream(self._upload.stream):
                parts = (
                    resolved
                    if resolved is not None
                    else self._resolve(inputs, labels, versions)
                )
                return self._upload.stage(self._assemble(*parts))

        future = self._reader.submit(assemble_and_upload)
        self._pending_read = weakref.ref(future)
        return future

    def _resolve(self, inputs, labels, versions):
        """Resolve CPU token references after packing/CP/layout conversion."""
        if labels.device.type != "cpu":
            raise ValueError(
                "Sampling-mask cache must be prepared from CPU microbatches"
            )
        refs = np.stack(
            [inputs[key].reshape(-1).numpy() for key in REFERENCE_KEYS], axis=1
        )
        labels = labels.reshape(-1)
        versions = versions.reshape(-1) if versions is not None else None
        reads = []
        shared_manifests = []
        for source, nonce in np.unique(refs[:, :2], axis=0):
            if source == 0:
                continue
            source, nonce = int(source), int(nonce)
            selected = np.flatnonzero((refs[:, 0] == source) & (refs[:, 1] == nonce))
            row_ids = refs[selected, 2]
            store = self._stores.setdefault((source, nonce), RowStore(CACHE_BYTES))
            requested = np.unique(row_ids[store.locate(row_ids) == 0])
            if requested.size:
                info = self._owner_call(0, self._control, source, nonce, "/info")
                shared = (
                    self.shared_namespace is not None
                    and info.get("shared_namespace") == self.shared_namespace
                )
                owner = next(
                    (i for i, uuid in enumerate(self.uuids) if uuid != info["uuid"]),
                    None,
                )
                if not shared and owner is None:
                    raise ValueError(
                        "sampling_mask_transfer needs a distinct GPU in the actor TP group for colocated GIN"
                    )
                # Bound both control messages and registered payload chunks.
                for start in range(0, len(requested), 16384):
                    rows = requested[start : start + 16384].tolist()
                    if shared:
                        manifest = self._owner_call(
                            0, self._control, source, nonce, "/shared", rows=rows
                        )
                        shared_manifests.append((store, np.asarray(rows), manifest))
                        continue
                    description = self._owner_call(
                        owner, self._control, source, nonce, "/describe", rows=rows
                    )
                    chunk, size, begin = [], 0, 0

                    def receive_chunk(end):
                        manifest = {
                            key: description[key][begin:end]
                            for key in ("counts", "tokens", "versions")
                        }
                        manifest["protocol"] = description["protocol"]
                        self._receive(source, nonce, chunk, owner, manifest)

                    for index, (row, count) in enumerate(
                        zip(rows, description["counts"], strict=True)
                    ):
                        if chunk and size + 3 * count > SLOT_BYTES:
                            receive_chunk(index)
                            chunk, size = [], 0
                            begin = index
                        chunk.append(row)
                        size += 3 * count
                    if chunk:
                        receive_chunk(len(rows))
            reads.append((store, row_ids, torch.from_numpy(selected)))
        return reads, shared_manifests, labels, versions

    def _assemble(self, reads, shared_manifests, labels, versions):
        for store, rows, manifest in shared_manifests:
            self._index_shared(store, rows, manifest)
        counts = torch.zeros(len(labels), dtype=torch.int64)
        segments = []
        for store, row_ids, positions in reads:
            packed, local_offsets, tokens, model_versions = store.read(row_ids)
            if not torch.equal(tokens, labels[positions]):
                raise ValueError(
                    "Sampling-mask row is not aligned with its generated token"
                )
            if versions is not None and not torch.equal(
                model_versions, versions[positions]
            ):
                raise ValueError("Sampling-mask row model weight version mismatch")
            counts[positions] = local_offsets.diff()
            segments.append((positions, packed, local_offsets))
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        # A single source was already read in final position order. Prompt and
        # padding positions have zero counts and introduce no ID gaps.
        if len(segments) == 1:
            return LoadedSamplingMask(segments[0][1], offsets, self.vocab_size)
        packed = torch.empty((int(offsets[-1]), 3), dtype=torch.uint8)
        for positions, source, local_offsets in segments:
            indices = torch.repeat_interleave(
                offsets[positions] - local_offsets[:-1], counts[positions]
            ) + torch.arange(source.shape[0])
            packed[indices] = source
        return LoadedSamplingMask(packed, offsets, self.vocab_size)

    def release_batch(self):
        self._drain_reader()
        self._stores.clear()
        self._cache_bytes = 0

    def close(self):
        """Both GIN peers drain credits before deregistering their windows."""
        self._reader.shutdown(wait=True)
        errors = []
        try:
            for (source, nonce), endpoint in self._sessions.items():
                try:
                    closing = self._executor.submit(
                        self._control, source, nonce, "/close", peer=self.uuid
                    )
                    endpoint.close()
                    closing.result()
                except Exception as exc:
                    errors.append(exc)
        finally:
            self._executor.shutdown(wait=True)
            self._sessions.clear()
            self.release_batch()
        if errors:
            raise RuntimeError(
                f"Sampling-mask session cleanup failed: {errors}"
            ) from errors[0]


def aggregate_metrics(
    consumer: SamplingMaskConsumer | None,
    *,
    vocab_size: int,
    logical_owner: bool,
    cpu_group,
) -> dict[str, float]:
    """All actor ranks participate; count logical supports once per TP replica.

    Transport bytes belong to the actual GIN owner, which may be a different TP
    rank for every source. Counters are cumulative; cache bytes are instantaneous.
    Shared rows fetched by different DP/CP shards count as separate transfers.
    """
    histogram = np.zeros(vocab_size + 1, dtype=np.int64)
    counters = np.zeros(6, dtype=np.int64)
    if consumer is not None:
        if logical_owner:
            histogram[:] = consumer._candidate_histogram
            counters[:2] = consumer.id_bytes, consumer.offset_bytes
        counters[2:] = (
            consumer.control_bytes,
            sum(endpoint.payload_bytes for endpoint in consumer._sessions.values()),
            consumer.shared_descriptors,
            consumer._cache_bytes,
        )
    values = torch.from_numpy(np.concatenate((histogram, counters)))
    peak = values[-1:].clone()
    if cpu_group is not None:
        dist.all_reduce(values, group=cpu_group)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX, group=cpu_group)
    histogram = values[: vocab_size + 1].numpy()
    counters = values[vocab_size + 1 :].tolist()
    total = int(histogram.sum())
    cumulative = histogram.cumsum()
    present = np.flatnonzero(histogram)
    statistics = {
        "received_rows": total,
        "candidate_mean": float(histogram @ np.arange(len(histogram))) / max(total, 1),
        "candidate_p95": int(np.searchsorted(cumulative, np.ceil(total * 0.95)))
        if total
        else 0,
        "candidate_p99": int(np.searchsorted(cumulative, np.ceil(total * 0.99)))
        if total
        else 0,
        "candidate_max": int(present[-1]) if total else 0,
        "id_payload_bytes": counters[0],
        "csr_offset_bytes": counters[1],
        "reference_tensor_bytes": total * len(REFERENCE_KEYS) * 8,
        "control_body_bytes": counters[2],
        "gin_payload_bytes": counters[3],
        "shared_descriptors": counters[4],
    }
    result = {
        f"sampling_mask/lifetime/{key}": float(value)
        for key, value in statistics.items()
    }
    result["sampling_mask/cpu_cache_bytes_sum"] = float(counters[5])
    result["sampling_mask/cpu_cache_bytes_max_rank"] = float(peak[0])
    return result

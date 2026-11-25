# SPDX-License-Identifier: Apache-2.0
"""Training-shard CPU cache; each support crosses the rollout boundary once."""

import json
import weakref
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import closing
from dataclasses import dataclass, replace

import numpy as np
import requests
import torch
import torch.distributed as dist

from areal.infra.sampling_mask.gin import GinEndpoint, load_extension
from areal.infra.sampling_mask.pages import PAGE_ROWS, SharedPageReader, UnpublishedRow
from areal.infra.sampling_mask.protocol import (
    MAX_FETCH_ROWS,
    REFERENCE_KEYS,
    decode_endpoint,
    pack_row_ranges,
    unpack_frame,
)
from areal.infra.sampling_mask.service import CACHE_BYTES, SLOT_BYTES
from areal.infra.sampling_mask.store import RowStore, shared_namespace
from areal.utils.sampling_mask import (
    PROTOCOL_VERSION,
    UPLOAD_SLOT_BYTES,
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
        self.shared_index_pages_read = 0
        self._sessions = {}
        self._stores = {}
        self._seen_pages = {}
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

    def _connect(self, key):
        if key in self._sessions:
            return
        source, nonce = key
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

    def _frame_sizes(self, key, rows):
        source, nonce = key
        self._connect(key)
        request = {"nonce": nonce, "peer": self.uuid, "rows": pack_row_ranges(rows)}
        with requests.post(
            decode_endpoint(source) + "/fetch", json=request, stream=True, timeout=90
        ) as response:
            self.control_bytes += len(response.request.body)
            response.raise_for_status()
            while line := response.raw.readline():
                self.control_bytes += len(line)
                size = json.loads(line)
                if isinstance(size, str):
                    raise RuntimeError(f"Sampling-mask fetch: {size}")
                if size == 0:
                    return
                yield size
            raise EOFError("Sampling-mask fetch ended before completion")

    def _receive(self, key, rows, owner, store):
        # Only the owner opens the HTTP stream and receives GIN frames. TP
        # peers learn the actual frame length before broadcasting the bytes.
        wire = None

        def receive():
            nonlocal wire
            size = next(sizes, 0)
            if size:
                wire = torch.empty(
                    size,
                    dtype=torch.uint8,
                    device=self.device if self.world > 1 else "cpu",
                )
                endpoint = self._sessions[key]
                for chunk in wire.split(SLOT_BYTES):
                    endpoint.transfer(chunk)
            return size

        offset = 0
        with closing(self._frame_sizes(key, rows)) as sizes:
            while size := self._owner_call(owner, receive):
                if self.world > 1:
                    if self.rank != owner:
                        wire = torch.empty(size, dtype=torch.uint8, device=self.device)
                    dist.broadcast(
                        wire,
                        src=dist.get_global_rank(self.group, owner),
                        group=self.group,
                    )
                packed, offsets, tokens, versions = unpack_frame(wire.cpu())
                end = offset + len(tokens)
                previous = store.nbytes
                store.capacity = CACHE_BYTES - self._cache_bytes + previous
                store.add(rows[offset:end], packed, offsets, tokens, versions)
                self._cache_bytes += store.nbytes - previous
                self.id_bytes += packed.numel()
                self.offset_bytes += offsets.numel() * 8
                self._candidate_histogram += np.bincount(
                    offsets.diff().numpy(), minlength=self.vocab_size + 1
                )
                offset = end
        if offset != len(rows):
            raise ValueError("Sampling-mask fetch returned an incomplete row set")

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

    def _open_source(self, key):
        info = self._owner_call(0, self._control, *key, "/info")
        if info["protocol"] != PROTOCOL_VERSION:
            raise ValueError("Sampling-mask manifest protocol version mismatch")
        if (
            self.shared_namespace is not None
            and info["shared_namespace"] == self.shared_namespace
        ):
            directory = info["page_directory"]
            if directory["layout"] != 1 or directory["rows"] != PAGE_ROWS:
                raise ValueError("Sampling-mask shared page layout version mismatch")
            return SharedPageReader(directory["prefix"], self.vocab_size), None
        for owner, uuid in enumerate(self.uuids):
            if uuid != info["uuid"]:
                return RowStore(CACHE_BYTES), owner
        raise ValueError("GIN needs a distinct GPU in the actor TP group")

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
        # Trajectory-packed runs usually contain thousands of consecutive rows
        # from one producer. Sort only the run heads to find unique endpoints.
        endpoints = refs[:, :2]
        if len(endpoints):
            changed = np.any(endpoints[1:] != endpoints[:-1], axis=1)
            endpoints = endpoints[np.r_[True, changed]]
        for source, nonce in np.unique(endpoints, axis=0):
            if source == 0:
                continue
            source, nonce = int(source), int(nonce)
            selected = np.flatnonzero((refs[:, 0] == source) & (refs[:, 1] == nonce))
            row_ids = refs[selected, 2]
            key = source, nonce
            if key not in self._stores:
                self._stores[key] = self._open_source(key)
            store, owner = self._stores[key]
            if owner is not None:
                requested = np.unique(row_ids[store.locate(row_ids) == 0])
                for start in range(0, len(requested), MAX_FETCH_ROWS):
                    self._receive(
                        key, requested[start : start + MAX_FETCH_ROWS], owner, store
                    )
            reads.append((key, store, row_ids, torch.from_numpy(selected)))
        return reads, labels, versions

    def _assemble(self, reads, labels, versions):
        counts = torch.zeros(len(labels), dtype=torch.int64)
        segments = []
        for key, store, row_ids, positions in reads:
            try:
                packed, local_offsets, tokens, model_versions = store.read(row_ids)
            except UnpublishedRow:
                # A rollout may have returned while its last D2H is in flight.
                # Wait on a scalar watermark, then distinguish a released row.
                self._control(*key, "/ready", through=int(row_ids.max()))
                packed, local_offsets, tokens, model_versions = store.read(row_ids)
            if not torch.equal(tokens, labels[positions]):
                raise ValueError(
                    "Sampling-mask row is not aligned with its generated token"
                )
            if versions is not None and not torch.equal(
                model_versions, versions[positions]
            ):
                raise ValueError("Sampling-mask row model weight version mismatch")
            lengths = local_offsets.diff()
            if isinstance(store, SharedPageReader):
                self._count_shared_reads(key, row_ids, lengths.numpy())
            counts[positions] = lengths
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

    def _count_shared_reads(self, key, rows, counts):
        # Track unique logical reads for metrics only. No offsets, candidates
        # or policy data are copied into a second actor-side row index.
        rows, first = np.unique(rows, return_index=True)
        page_ids, slots = np.divmod(rows - 1, PAGE_ROWS)
        boundaries = np.r_[0, np.flatnonzero(np.diff(page_ids)) + 1, len(rows)]
        fresh = np.zeros(len(rows), dtype=bool)
        self.shared_index_pages_read += len(boundaries) - 1
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            page_key = (*key, int(page_ids[start]))
            seen = self._seen_pages.get(page_key)
            if seen is None:
                if self._cache_bytes + PAGE_ROWS // 8 > CACHE_BYTES:
                    raise MemoryError("Sampling-mask actor cache capacity exceeded")
                seen = self._seen_pages[page_key] = np.zeros(
                    PAGE_ROWS // 8, dtype=np.uint8
                )
                self._cache_bytes += seen.nbytes
            byte, bit = np.divmod(slots[start:end], 8)
            bits = np.left_shift(np.uint8(1), bit.astype(np.uint8))
            fresh[start:end] = (seen[byte] & bits) == 0
            np.bitwise_or.at(seen, byte, bits)
        new_counts = counts[first[fresh]]
        if len(new_counts):
            self.id_bytes += int(new_counts.sum()) * 3
            self.offset_bytes += (len(new_counts) + 1) * 8
            self._candidate_histogram += np.bincount(
                new_counts, minlength=self.vocab_size + 1
            )

    def release_batch(self):
        self._drain_reader()
        self._stores.clear()
        self._seen_pages.clear()
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
            consumer.shared_index_pages_read,
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
        "shared_index_pages_read": counters[4],
    }
    result = {
        f"sampling_mask/lifetime/{key}": float(value)
        for key, value in statistics.items()
    }
    result["sampling_mask/cpu_cache_bytes_sum"] = float(counters[5])
    result["sampling_mask/cpu_cache_bytes_max_rank"] = float(peak[0])
    return result

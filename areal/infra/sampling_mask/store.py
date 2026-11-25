# SPDX-License-Identifier: Apache-2.0
"""Chunk-level CSR ownership with a paged numeric index (no per-token objects)."""

import os
import re
import secrets
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import torch

from areal.utils.sampling_mask import SamplingMask

_PAGE_ROWS = 4096
_SEGMENT_ROWS = 16384
_SEGMENT_BYTES = 8 << 20
_SHM = Path("/dev/shm")


def shared_namespace() -> str | None:
    """Identify a Linux tmpfs visible to all participants, including containers."""
    try:
        mount = _SHM.stat()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None
    return f"{boot}:{mount.st_dev}:{mount.st_ino}:{os.geteuid()}"


def _map_ids(name: str) -> np.ndarray:
    if not re.fullmatch(r"areal-mask-[0-9a-f]{32}", name):
        raise ValueError("Invalid sampling-mask shared block name")
    return np.memmap(_SHM / name, mode="r", dtype=np.uint8).reshape(-1, 3)


@dataclass
class _Block:
    row_ids: np.ndarray
    packed: torch.Tensor | None
    offsets: torch.Tensor
    tokens: torch.Tensor
    versions: torch.Tensor
    live: np.ndarray
    owner: SharedMemory | None = None
    name: str | None = None
    starts: np.ndarray | None = None
    request_keys: np.ndarray | None = None

    @property
    def nbytes(self):
        return (
            (0 if self.packed is None else self.packed.numel())
            + (0 if self.owner is None else self.owner.size)
            + (0 if self.starts is None else self.starts.nbytes)
            + self.offsets.numel() * 8
            + self.tokens.numel() * 8
            + self.versions.numel() * 8
            + (0 if self.request_keys is None else self.request_keys.nbytes)
            + self.row_ids.nbytes
            + self.live.nbytes
        )


class RowStore:
    """CPU byte budget covers payload, offsets, metadata, and numeric lookup pages."""

    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.nbytes = 0
        self._next_block = 1
        self._blocks: dict[int, _Block] = {}
        self._pages: dict[int, np.ndarray] = {}
        self._small_blocks: list[int] = []
        self._pending_rows = self._pending_bytes = 0

    def locate(self, row_ids: np.ndarray) -> np.ndarray:
        row_ids = np.asarray(row_ids, dtype=np.int64)
        output = np.zeros_like(row_ids)
        page_ids = row_ids // _PAGE_ROWS
        for page in np.unique(page_ids):
            selected = page_ids == page
            values = self._pages.get(int(page))
            if values is not None:
                output[selected] = values[row_ids[selected] % _PAGE_ROWS]
        return output

    def add(
        self,
        row_ids,
        packed,
        offsets,
        tokens,
        versions,
        *,
        name=None,
        starts=None,
        request_keys=None,
    ) -> None:
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if (
            np.any(row_ids <= 0)
            or np.any(self.locate(row_ids))
            or len(np.unique(row_ids)) != len(row_ids)
        ):
            raise ValueError("Duplicate or invalid sampling-mask cache row")
        if request_keys is not None and (
            self._pending_rows + len(row_ids) > _SEGMENT_ROWS
            or self._pending_bytes + packed.numel() > _SEGMENT_BYTES
        ):
            self.compact()
        if len(row_ids) >= 2**32 or self._next_block >= 2**31:
            raise ValueError("Sampling-mask cache index capacity exceeded")
        block = _Block(
            row_ids.copy(),
            packed,
            offsets,
            tokens.long(),
            versions.long(),
            np.ones(len(row_ids), dtype=np.bool_),
            name=name,
            starts=starts,
            request_keys=request_keys,
        )
        page_ids = row_ids // _PAGE_ROWS
        new_pages = set(map(int, np.unique(page_ids))) - self._pages.keys()
        needed = block.nbytes + len(new_pages) * _PAGE_ROWS * 8
        if self.nbytes + needed > self.capacity:
            raise MemoryError(
                f"Sampling-mask CPU cache capacity exceeded: {self.nbytes + needed} > {self.capacity}"
            )
        block_id = self._next_block
        self._next_block += 1
        self._blocks[block_id] = block
        if name is None and (request_keys is not None or len(row_ids) < 256):
            self._small_blocks.append(block_id)
            self._pending_rows += len(row_ids)
            self._pending_bytes += packed.numel()
        locators = (np.int64(block_id) << 32) | np.arange(len(row_ids), dtype=np.int64)
        for page in np.unique(page_ids):
            values = self._pages.setdefault(
                int(page), np.zeros(_PAGE_ROWS, dtype=np.int64)
            )
            selected = page_ids == page
            values[row_ids[selected] % _PAGE_ROWS] = locators[selected]
        self.nbytes += needed

    def compact(self) -> None:
        """Seal bounded windows of decode batches in request order on the CPU writer.

        Row handles stay stable. The last partial window can be sealed by a reader.
        A block already exported to another process must retain its backing file.
        """
        pending = []
        row_count = byte_count = 0

        def flush():
            if len(pending) < 2:
                return
            blocks = [self._blocks[i] for i in pending]
            rows = np.concatenate([b.row_ids for b in blocks])
            counts = np.concatenate([np.diff(b.offsets.numpy()) for b in blocks])
            packed = np.concatenate([b.packed.numpy() for b in blocks])
            tokens = np.concatenate([b.tokens.numpy() for b in blocks])
            versions = np.concatenate([b.versions.numpy() for b in blocks])
            order = np.flatnonzero(np.concatenate([b.live for b in blocks]))
            if all(b.request_keys is not None for b in blocks):
                keys = np.concatenate([b.request_keys for b in blocks])
                order = order[np.argsort(keys[order], kind="stable")]
            offsets = np.concatenate(([0], counts[order].cumsum()))
            starts = counts.cumsum() - counts
            if len(order) != len(rows) or np.any(order != np.arange(len(rows))):
                indices = np.repeat(starts[order] - offsets[:-1], counts[order])
                indices += np.arange(offsets[-1])
                packed = packed[indices]
            merged = _Block(
                rows[order],
                torch.from_numpy(packed),
                torch.from_numpy(offsets),
                torch.from_numpy(tokens[order]),
                torch.from_numpy(versions[order]),
                np.ones(len(order), dtype=np.bool_),
            )
            if self._next_block >= 2**31:
                raise ValueError("Sampling-mask cache index capacity exceeded")
            block_id = self._next_block
            self._next_block += 1
            locations = (np.int64(block_id) << 32) | np.arange(len(order))
            pages = merged.row_ids // _PAGE_ROWS
            for page in np.unique(pages):
                selected = pages == page
                self._pages[int(page)][merged.row_ids[selected] % _PAGE_ROWS] = (
                    locations[selected]
                )
            self.nbytes += merged.nbytes - sum(b.nbytes for b in blocks)
            for i in pending:
                del self._blocks[i]
            self._blocks[block_id] = merged

        for i in self._small_blocks:
            block = self._blocks.get(i)
            if block is None or block.owner is not None:
                continue
            if pending and (
                row_count + len(block.row_ids) > _SEGMENT_ROWS
                or byte_count + block.packed.numel() > _SEGMENT_BYTES
            ):
                flush()
                pending.clear()
                row_count = byte_count = 0
            pending.append(i)
            row_count += len(block.row_ids)
            byte_count += block.packed.numel()
        flush()
        self._small_blocks.clear()
        self._pending_rows = self._pending_bytes = 0

    def share(self, row_ids):
        """Promote each requested immutable source block once, returning only locations."""
        locations = self.locate(row_ids)
        if np.any(locations == 0):
            raise ValueError("Unknown or already released sampling-mask row")
        blocks, indices = locations >> 32, locations & (2**32 - 1)
        groups = []
        for block_id in np.unique(blocks):
            block = self._blocks[int(block_id)]
            if block.owner is None:
                size = block.packed.numel()
                handle = SharedMemory(
                    name="areal-mask-" + secrets.token_hex(16), create=True, size=size
                )
                try:
                    with (_SHM / handle.name).open("r+b") as backing:
                        os.posix_fallocate(backing.fileno(), 0, size)
                    np.copyto(
                        np.ndarray((size // 3, 3), np.uint8, buffer=handle.buf),
                        block.packed.numpy(),
                    )
                    block.packed = None
                    block.name = handle.name
                    block.owner = handle
                except BaseException:
                    handle.unlink()
                    raise
                finally:
                    handle.close()
            selected = np.flatnonzero(blocks == block_id)
            groups.append(
                (
                    block.owner.name,
                    selected.tolist(),
                    block.offsets.numpy()[indices[selected]].tolist(),
                )
            )
        return groups

    def index_shared(
        self, rows, name, starts, offsets, tokens, versions, vocab_size, chunk_bytes
    ):
        """Validate bounded slices, then keep only row metadata on the reader."""
        source = _map_ids(name)
        lengths = offsets.diff().numpy()
        if np.any(starts > len(source) - lengths):
            raise ValueError("Sampling-mask descriptor exceeds shared block")
        first = 0
        while first < len(rows):
            limit = int(offsets[first]) + chunk_bytes // 3
            end = offsets.numpy().searchsorted(limit, side="right") - 1
            if end <= first:
                raise ValueError("Shared sampling-mask row exceeds chunk size")
            local_offsets = offsets[first : end + 1] - offsets[first]
            indices = np.repeat(
                starts[first:end] - local_offsets[:-1].numpy(), lengths[first:end]
            )
            indices += np.arange(int(local_offsets[-1]))
            SamplingMask(
                torch.from_numpy(source[indices]), local_offsets, vocab_size, 0
            ).validate(tokens[first:end], weight_version=0)
            first = end
        if np.array_equal(starts, offsets[:-1].numpy()):
            starts = None
        elif starts.max() <= np.iinfo(np.uint32).max:
            starts = starts.astype(np.min_scalar_type(int(starts.max())))
        self.add(rows, None, offsets, tokens, versions, name=name, starts=starts)

    def close(self):
        for block in self._blocks.values():
            if block.owner is not None:
                block.owner.unlink()
        self._blocks.clear()
        self._pages.clear()
        self._small_blocks.clear()
        self._pending_rows = self._pending_bytes = 0
        self.nbytes = 0

    def describe(self, row_ids):
        locations = self.locate(row_ids)
        if np.any(locations == 0):
            raise ValueError("Unknown or already released sampling-mask row")
        blocks, indices = locations >> 32, locations & (2**32 - 1)
        counts, tokens, versions = [
            np.empty(len(locations), dtype=np.int64) for _ in range(3)
        ]
        for block_id in np.unique(blocks):
            selected = np.flatnonzero(blocks == block_id)
            local = indices[selected]
            block = self._blocks[int(block_id)]
            offsets = block.offsets.numpy()
            counts[selected] = offsets[local + 1] - offsets[local]
            tokens[selected] = block.tokens.numpy()[local]
            versions[selected] = block.versions.numpy()[local]
        return tuple(torch.from_numpy(value) for value in (counts, tokens, versions))

    def read(self, row_ids):
        locations = self.locate(row_ids)
        counts, tokens, versions = self.describe(row_ids)
        counts_np = counts.numpy()
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), counts_np.cumsum()))
        packed = np.empty((int(offsets[-1]), 3), dtype=np.uint8)
        blocks, indices = locations >> 32, locations & (2**32 - 1)
        for block_id in np.unique(blocks):
            selected = np.flatnonzero(blocks == block_id)
            local = indices[selected]
            block = self._blocks[int(block_id)]
            lengths = counts_np[selected]
            cumulative = np.concatenate((np.zeros(1, dtype=np.int64), lengths.cumsum()))
            linear = np.arange(cumulative[-1])
            starts = block.offsets.numpy() if block.starts is None else block.starts
            source_indices = (
                np.repeat(starts[local] - cumulative[:-1], lengths) + linear
            )
            destination_indices = (
                np.repeat(offsets[selected] - cumulative[:-1], lengths) + linear
            )
            source = (
                block.packed.numpy() if block.name is None else _map_ids(block.name)
            )
            packed[destination_indices] = source[source_indices]
        return torch.from_numpy(packed), torch.from_numpy(offsets), tokens, versions

    def delete(self, row_ids) -> None:
        row_ids = np.asarray(row_ids, dtype=np.int64)
        locations = self.locate(row_ids)
        if np.any(locations == 0):
            raise ValueError("Sampling-mask row was released more than once")
        blocks, indices = locations >> 32, locations & (2**32 - 1)
        for block_id in np.unique(blocks):
            block = self._blocks[int(block_id)]
            block.live[indices[blocks == block_id]] = False
            if not block.live.any():
                self.nbytes -= block.nbytes
                if block.owner is not None:
                    block.owner.unlink()
                del self._blocks[int(block_id)]
        pages = row_ids // _PAGE_ROWS
        for page in np.unique(pages):
            values = self._pages[int(page)]
            values[row_ids[pages == page] % _PAGE_ROWS] = 0
            if not values.any():
                self.nbytes -= values.nbytes
                del self._pages[int(page)]

    def discard_before(self, version: int) -> None:
        """Retire discarded/overlap rows after their policy version expires."""
        for block in list(self._blocks.values()):
            expired = block.live & (block.versions.numpy() < version)
            if expired.any():
                self.delete(block.row_ids[expired])

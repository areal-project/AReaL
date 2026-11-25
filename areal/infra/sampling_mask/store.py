# SPDX-License-Identifier: Apache-2.0
"""Chunk-level CSR ownership with a paged numeric index (no per-token objects)."""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_PAGE_ROWS = 4096
_SHM = Path("/dev/shm")


def shared_namespace() -> str | None:
    """Identify a Linux tmpfs visible to all participants, including containers."""
    try:
        mount = _SHM.stat()
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None
    return f"{boot}:{mount.st_dev}:{mount.st_ino}:{os.geteuid()}"


@dataclass
class _Block:
    packed: torch.Tensor
    offsets: torch.Tensor
    tokens: torch.Tensor
    versions: torch.Tensor
    live: np.ndarray

    @property
    def nbytes(self):
        return (
            self.packed.numel()
            + self.offsets.numel() * 8
            + self.tokens.numel() * 8
            + self.versions.numel() * 8
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

    def add(self, row_ids, packed, offsets, tokens, versions) -> None:
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if (
            np.any(row_ids <= 0)
            or np.any(self.locate(row_ids))
            or len(np.unique(row_ids)) != len(row_ids)
        ):
            raise ValueError("Duplicate or invalid sampling-mask cache row")
        if len(row_ids) >= 2**32 or self._next_block >= 2**31:
            raise ValueError("Sampling-mask cache index capacity exceeded")
        block = _Block(
            packed,
            offsets,
            tokens.long(),
            versions.long(),
            np.ones(len(row_ids), dtype=np.bool_),
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
        locators = (np.int64(block_id) << 32) | np.arange(len(row_ids), dtype=np.int64)
        for page in np.unique(page_ids):
            values = self._pages.setdefault(
                int(page), np.zeros(_PAGE_ROWS, dtype=np.int64)
            )
            selected = page_ids == page
            values[row_ids[selected] % _PAGE_ROWS] = locators[selected]
        self.nbytes += needed

    def close(self):
        self._blocks.clear()
        self._pages.clear()
        self.nbytes = 0

    def _locations(self, row_ids):
        loc = self.locate(row_ids)
        if np.any(loc == 0):
            raise ValueError("Unknown or already released sampling-mask row")
        blocks = loc >> 32
        for b in np.unique(blocks):
            pos = np.flatnonzero(blocks == b)
            yield self._blocks[int(b)], pos, loc[pos] & (2**32 - 1)

    def counts(self, row_ids):
        counts = np.empty(len(row_ids), dtype=np.int64)
        for block, pos, local in self._locations(row_ids):
            offsets = block.offsets.numpy()
            counts[pos] = offsets[local + 1] - offsets[local]
        return counts

    def read(self, row_ids):
        groups = list(self._locations(row_ids))
        counts_np, tokens, versions = [
            np.empty(len(row_ids), dtype=np.int64) for _ in range(3)
        ]
        for block, pos, local in groups:
            starts = block.offsets.numpy()
            counts_np[pos] = starts[local + 1] - starts[local]
            tokens[pos] = block.tokens.numpy()[local]
            versions[pos] = block.versions.numpy()[local]
        offsets = np.r_[np.int64(0), counts_np.cumsum()]
        packed = np.empty((int(offsets[-1]), 3), dtype=np.uint8)
        for block, selected, local in groups:
            lengths = counts_np[selected]
            cumulative = np.concatenate((np.zeros(1, dtype=np.int64), lengths.cumsum()))
            linear = np.arange(cumulative[-1])
            source_indices = (
                np.repeat(block.offsets.numpy()[local] - cumulative[:-1], lengths)
                + linear
            )
            destination_indices = (
                np.repeat(offsets[selected] - cumulative[:-1], lengths) + linear
            )
            source = block.packed.numpy()
            packed[destination_indices] = source[source_indices]
        return (
            torch.from_numpy(packed),
            torch.from_numpy(offsets),
            torch.from_numpy(tokens),
            torch.from_numpy(versions),
        )

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
                del self._blocks[int(block_id)]
        pages = row_ids // _PAGE_ROWS
        for page in np.unique(pages):
            values = self._pages[int(page)]
            values[row_ids[pages == page] % _PAGE_ROWS] = 0
            if not values.any():
                self.nbytes -= values.nbytes
                del self._pages[int(page)]

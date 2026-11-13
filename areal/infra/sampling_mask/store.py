# SPDX-License-Identifier: Apache-2.0
"""Chunk-level CSR ownership with a paged numeric index (no per-token objects)."""

from dataclasses import dataclass

import numpy as np
import torch

_PAGE_ROWS = 4096


@dataclass
class _Block:
    row_ids: np.ndarray
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
            row_ids.copy(),
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
        if len(row_ids) < 256:
            self._small_blocks.append(block_id)
        locators = (np.int64(block_id) << 32) | np.arange(len(row_ids), dtype=np.int64)
        for page in np.unique(page_ids):
            values = self._pages.setdefault(
                int(page), np.zeros(_PAGE_ROWS, dtype=np.int64)
            )
            selected = page_ids == page
            values[row_ids[selected] % _PAGE_ROWS] = locators[selected]
        self.nbytes += needed

    def compact(self) -> None:
        """Coalesce decode-sized blocks at the read boundary with bounded scratch.

        A long trajectory spans thousands of sampler batches. Keeping a separate
        allocation for every decode step makes each row gather visit thousands
        of tiny blocks. Merge newly published blocks once, preserving row IDs
        and immutable supports. Scratch is bounded by 8 MiB IDs + 16k metadata
        rows; the numeric lookup pages are updated in place.
        """
        if len(self._small_blocks) < 128:
            return
        candidates, self._small_blocks = self._small_blocks, []
        pending: list[int] = []
        row_count = byte_count = 0

        def flush():
            if len(pending) < 2:
                return
            blocks = [self._blocks[index] for index in pending]
            bases = np.cumsum([0, *[block.packed.shape[0] for block in blocks]])
            merged = _Block(
                np.concatenate([block.row_ids for block in blocks]),
                torch.cat([block.packed for block in blocks]),
                torch.from_numpy(
                    np.concatenate(
                        [
                            block.offsets[:-1].numpy() + base
                            for block, base in zip(blocks, bases[:-1], strict=True)
                        ]
                        + [bases[-1:]]
                    )
                ),
                torch.cat([block.tokens for block in blocks]),
                torch.cat([block.versions for block in blocks]),
                np.ones(sum(len(block.row_ids) for block in blocks), dtype=np.bool_),
            )
            if self._next_block >= 2**31:
                raise ValueError("Sampling-mask cache index capacity exceeded")
            block_id = self._next_block
            self._next_block += 1
            self.nbytes += merged.nbytes - sum(block.nbytes for block in blocks)
            locations = (np.int64(block_id) << 32) | np.arange(
                len(merged.row_ids), dtype=np.int64
            )
            pages = merged.row_ids // _PAGE_ROWS
            for page in np.unique(pages):
                selected = pages == page
                self._pages[int(page)][merged.row_ids[selected] % _PAGE_ROWS] = (
                    locations[selected]
                )
            for index in pending:
                del self._blocks[index]
            self._blocks[block_id] = merged

        for index in candidates:
            block = self._blocks.get(index)
            if block is None or not block.live.all() or block.packed.numel() > 8 << 20:
                continue
            if pending and (
                row_count + len(block.row_ids) > 16384
                or byte_count + block.packed.numel() > 8 << 20
            ):
                flush()
                pending.clear()
                row_count = byte_count = 0
            pending.append(index)
            row_count += len(block.row_ids)
            byte_count += block.packed.numel()
        flush()

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
            source_indices = (
                np.repeat(block.offsets.numpy()[local] - cumulative[:-1], lengths)
                + linear
            )
            destination_indices = (
                np.repeat(offsets[selected] - cumulative[:-1], lengths) + linear
            )
            packed[destination_indices] = block.packed.numpy()[source_indices]
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

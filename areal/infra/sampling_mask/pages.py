# SPDX-License-Identifier: Apache-2.0
"""Producer-owned tmpfs pages addressed by the existing monotonic row handles."""

import fcntl
import hashlib
import os
import re
import secrets
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch

from areal.infra.sampling_mask.store import _SHM

PAGE_ROWS = 16384
# Candidate start (in IDs), count, sampled token, policy version. Zero count is
# unpublished/released. The prefix and layout revision identify this producer.
_RECORD = np.dtype(
    [("start", "<i8"), ("count", "<i4"), ("token", "<i4"), ("version", "<i8")]
)
PAGE_INDEX_BYTES = PAGE_ROWS * _RECORD.itemsize


class UnpublishedRow(ValueError):
    """The writer may still be draining the bounded sampler queue."""


def _groups(rows):
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or np.any(rows <= 0):
        raise ValueError("Invalid sampling-mask shared row")
    pages = (rows - 1) // PAGE_ROWS
    if len(pages) and np.all(pages == pages[0]):
        yield int(pages[0]), np.arange(len(rows)), (rows - 1) % PAGE_ROWS
        return
    order = np.argsort(pages, kind="stable")
    for selected in np.split(order, np.flatnonzero(np.diff(pages[order])) + 1):
        if len(selected):
            yield int(pages[selected[0]]), selected, (rows[selected] - 1) % PAGE_ROWS


@contextmanager
def _locked(file, write):
    fcntl.flock(file, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
    try:
        yield
    finally:
        fcntl.flock(file, fcntl.LOCK_UN)


@contextmanager
def _index(name, write=False):
    with (_SHM / name).open("r+b" if write else "rb") as handle, _locked(handle, write):
        size = os.fstat(handle.fileno()).st_size
        if size == 0 and not write:
            # shm_open publishes the name just before the creator's ftruncate.
            raise UnpublishedRow("Sampling-mask shared index is being created")
        if size != PAGE_INDEX_BYTES:
            raise ValueError("Invalid sampling-mask shared index size")
        yield np.memmap(
            handle, dtype=_RECORD, mode="r+" if write else "r", shape=(PAGE_ROWS,)
        )


class SharedPageReader:
    def __init__(self, prefix, max_candidates):
        if not isinstance(prefix, str) or not re.fullmatch("[0-9a-f]{32}", prefix):
            raise ValueError("Invalid sampling-mask shared page directory")
        self.prefix = prefix
        self.max_candidates = max_candidates

    def _names(self, page):
        key = f"{self.prefix}/{page}".encode()
        return tuple(
            "areal-mask-" + hashlib.blake2s(key + suffix, digest_size=16).hexdigest()
            for suffix in (b"/index", b"/ids")
        )

    def _lookup(self, rows, *, gather=False):
        records = np.empty(len(rows), dtype=_RECORD)
        groups = []
        for page, pos, slots in _groups(rows):
            index, payload = self._names(page)
            try:
                with _index(index) as entries:
                    local = entries[slots]
                    if np.any(local["count"] == 0):
                        raise UnpublishedRow(
                            "Unknown or already released sampling-mask row"
                        )
                    # Pin this payload incarnation under its index lock. Reads
                    # from the immutable range stay valid even after unlink.
                    source = (
                        np.memmap(_SHM / payload, dtype=np.uint8, mode="r").reshape(
                            -1, 3
                        )
                        if gather
                        else None
                    )
            except FileNotFoundError as exc:
                raise UnpublishedRow(
                    "Unknown or already released sampling-mask page"
                ) from exc
            lengths = local["count"].astype(np.int64)
            if (
                np.any(local["start"] < 0)
                or np.any(lengths < 0)
                or np.any(lengths > self.max_candidates)
            ):
                raise ValueError("Invalid sampling-mask shared index record")
            records[pos] = local
            data = None
            if source is not None:
                if np.any(local["start"] > len(source) - lengths):
                    raise ValueError(
                        "Sampling-mask shared index exceeds candidate page"
                    )
                starts = lengths.cumsum() - lengths
                linear = np.arange(lengths.sum())
                data = source[np.repeat(local["start"] - starts, lengths) + linear]
            groups.append((pos, data))
        return records, groups

    def counts(self, rows):
        records, _ = self._lookup(rows)
        return records["count"].astype(np.int64)

    def read(self, rows):
        records, groups = self._lookup(rows, gather=True)
        counts = records["count"].astype(np.int64)
        offsets = np.r_[np.int64(0), counts.cumsum()]
        if len(groups) == 1:
            packed = groups[0][1]
        else:
            packed = np.empty((offsets[-1], 3), dtype=np.uint8)
            for pos, data in groups:
                lengths = counts[pos]
                starts = lengths.cumsum() - lengths
                dst = np.repeat(offsets[pos] - starts, lengths) + np.arange(len(data))
                packed[dst] = data
        return (
            torch.from_numpy(packed),
            torch.from_numpy(offsets),
            torch.from_numpy(records["token"].astype(np.int64)),
            torch.from_numpy(records["version"].astype(np.int64)),
        )


@dataclass
class _OwnedPage:
    index: SharedMemory
    payload: SharedMemory
    used: int = 0
    live: int = 0

    def unlink(self):
        self.payload.unlink()
        self.index.unlink()


class SharedPageWriter(SharedPageReader):
    """One copy serving local reads and GIN. The producer serializes mutators."""

    def __init__(self, capacity):
        super().__init__(secrets.token_hex(16), 1 << 24)
        self.capacity = capacity
        self.nbytes = 0
        self.pages = {}
        self._tail_files = ExitStack()
        self._tail = None

    def _new_page(self, page):
        index_name, payload_name = self._names(page)
        index = SharedMemory(index_name, create=True, size=PAGE_INDEX_BYTES)
        try:
            with (_SHM / index_name).open("r+b") as file:
                os.posix_fallocate(file.fileno(), 0, PAGE_INDEX_BYTES)
            payload = SharedMemory(payload_name, create=True, size=3)
            payload.close()
        except BaseException:
            index.unlink()
            raise
        finally:
            index.close()
        self.pages[page] = _OwnedPage(index, payload)
        self.nbytes += PAGE_INDEX_BYTES + 3
        return self.pages[page]

    def _append_files(self, current):
        # Decode repeatedly appends to the same page. Map just the writable tail;
        # sealed pages retain no open file descriptors in the producer.
        if self._tail is None or self._tail[0] is not current:
            self._tail = None
            self._tail_files.close()
            with ExitStack() as stack:
                handle = stack.enter_context((_SHM / current.index.name).open("r+b"))
                ids = stack.enter_context(
                    (_SHM / current.payload.name).open("r+b", buffering=0)
                )
                index = np.memmap(handle, dtype=_RECORD, mode="r+", shape=(PAGE_ROWS,))
                self._tail_files = stack.pop_all()
                self._tail = current, handle, index, ids
        return self._tail[1:]

    def add(self, rows, packed, offsets, tokens, versions):
        rows = np.asarray(rows)
        offsets = offsets.numpy()
        counts = np.diff(offsets)
        if (
            packed.dtype != torch.uint8
            or packed.ndim != 2
            or packed.shape[1] != 3
            or offsets.shape != (len(rows) + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(packed)
            or np.any(counts <= 0)
            or np.any(counts > self.max_candidates)
            or tokens.shape != rows.shape
            or versions.shape != rows.shape
            or np.any(np.diff(rows) <= 0)
        ):
            raise ValueError("Misaligned sampling-mask source rows")
        # Single writer gets monotonic, contiguous ranges from publish. Reserve
        # before touching mmap pages (ENOSPC must be an exception, not SIGBUS).
        chunks = []
        needed = 0
        for page, selected, slots in _groups(rows):
            current = self.pages.get(page)
            begin, end = offsets[selected[0]], offsets[selected[-1] + 1]
            growth = int(end - begin) * 3
            if current is None:
                growth += PAGE_INDEX_BYTES
            needed += growth
            chunks.append((page, selected, slots, begin, end))
        if self.nbytes + needed > self.capacity:
            raise MemoryError("Sampling-mask shared page capacity exceeded")
        written = 0
        try:
            for page, selected, slots, begin, end in chunks:
                current = self.pages.get(page) or self._new_page(page)
                data = memoryview(packed[begin:end].numpy()).cast("B")
                values = np.empty(len(slots), dtype=_RECORD)
                values["start"] = current.used // 3 + offsets[selected] - begin
                values["count"] = counts[selected]
                values["token"] = tokens.numpy()[selected]
                values["version"] = versions.numpy()[selected]
                handle, index, ids = self._append_files(current)
                with _locked(handle, write=True):
                    if np.any(index["count"][slots]):
                        raise ValueError("Duplicate sampling-mask shared row")
                    # pwrite extends at the exact length; allocation failures
                    # cannot SIGBUS a writer through a sparse payload mmap.
                    try:
                        if os.pwrite(ids.fileno(), data, current.used) != len(data):
                            raise OSError("Short write to sampling-mask candidate page")
                        index[slots] = values
                    except BaseException:
                        os.ftruncate(ids.fileno(), max(3, current.used))
                        raise
                    self.nbytes += len(data) - (0 if current.used else 3)
                    current.live += len(slots)
                    current.used += len(data)
                    written += len(slots)
        except BaseException:
            if written:
                self.delete(rows[:written])
            current = self.pages.get(page)
            if current is not None and not current.live:
                self._unlink(page)
            raise

    def _unlink(self, page):
        current = self.pages.pop(page)
        if self._tail is not None and self._tail[0] is current:
            self._tail = None
            self._tail_files.close()
        current.unlink()
        self.nbytes -= PAGE_INDEX_BYTES + max(3, current.used)

    def delete(self, rows):
        for page, _, slots in _groups(rows):
            current = self.pages[page]
            with _index(current.index.name, write=True) as index:
                if np.any(index["count"][slots] == 0) or len(np.unique(slots)) != len(
                    slots
                ):
                    raise ValueError("Sampling-mask row was released more than once")
                index["count"][slots] = 0
                current.live -= len(slots)
                if not current.live:
                    self._unlink(page)

    def close(self):
        for page in list(self.pages):
            self._unlink(page)

# SPDX-License-Identifier: Apache-2.0

import ctypes
import gc
from collections.abc import Callable, Iterable


def reclaim_cpu_memory() -> bool:
    """Collect unreachable objects and return free glibc heap pages to the OS.

    Call only at batch boundaries, after releasing tensor owners and caches.
    This does not free live tensors or CUDA pinned-memory allocator caches.
    Platforms without ``malloc_trim`` still collect Python garbage.
    """
    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return False
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    return bool(malloc_trim(0))


def run_batch_cleanups(
    cleanups: Iterable[tuple[str, Callable[[], None]]],
) -> None:
    """Run every batch cleanup before propagating any ordinary failures."""
    failures: list[tuple[str, Exception]] = []
    for role, cleanup in cleanups:
        try:
            cleanup()
        except Exception as error:
            failures.append((role, error))

    if failures:
        first_role, first_error = failures[0]
        first_error.add_note(f"clear_batches failed first for role: {first_role}")
        for role, error in failures[1:]:
            first_error.add_note(
                "Additional clear_batches failure for "
                f"{role}: {type(error).__name__}: {error}"
            )
        raise first_error

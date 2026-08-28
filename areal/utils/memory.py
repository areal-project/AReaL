# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch.distributed as dist

from areal.infra.platforms import current_platform
from areal.utils import logging

logger = logging.getLogger("PeakMemory")

_GIB = 1024**3


def _peak_memory_functions() -> (
    tuple[Callable[[], None], Callable[[], int], Callable[[], int]] | None
):
    try:
        reset = current_platform.reset_peak_memory_stats
        max_allocated = current_platform.max_memory_allocated
        max_reserved = current_platform.max_memory_reserved
    except Exception:
        return None

    if not all(callable(fn) for fn in (reset, max_allocated, max_reserved)):
        return None
    return reset, max_allocated, max_reserved


@contextmanager
def report_peak_memory(phase: str) -> Iterator[None]:
    """Log rank zero's peak device memory for a non-overlapping phase.

    Unsupported accelerators run without a report. Scopes must not nest because
    resetting the process-wide allocator counter would discard an outer peak.
    """
    functions = _peak_memory_functions()
    if functions is None:
        yield
        return

    reset, max_allocated, max_reserved = functions
    try:
        reset()
    except Exception:
        logger.debug("Could not reset peak-memory statistics", exc_info=True)
        yield
        return

    try:
        yield
    finally:
        try:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                logger.info(
                    f"[PeakMemory Rank {rank}] {phase}: "
                    f"max allocated (GB): {max_allocated() / _GIB:.2f}, "
                    f"max reserved (GB): {max_reserved() / _GIB:.2f}"
                )
        except Exception:
            logger.debug("Could not report peak-memory statistics", exc_info=True)

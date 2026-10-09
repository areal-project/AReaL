# SPDX-License-Identifier: Apache-2.0

"""Parallel-state lookup for the SGLang AWEX integration."""

from typing import Any


def resolve_scheduler_parallel_attr(scheduler: Any, name: str) -> int | None:
    """Read parallel state across supported SGLang attribute layouts."""
    worker = getattr(scheduler, "tp_worker", None)
    runner = getattr(worker, "model_runner", None)
    for owner in (
        scheduler,
        getattr(scheduler, "ps", None),
        worker,
        getattr(worker, "ps", None),
        getattr(runner, "ps", None),
    ):
        value = getattr(owner, name, None)
        if value is not None:
            return int(value)
    return None

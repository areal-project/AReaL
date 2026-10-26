# SPDX-License-Identifier: Apache-2.0
"""Optional per-rank engine observations without device synchronization."""

from __future__ import annotations

import os
import time
from contextvars import ContextVar
from functools import wraps
from typing import Any

from areal.utils import logging
from areal.utils.workload_trace import append_trace

logger = logging.getLogger("WorkloadTrace")


def install_actor_observer(engine: Any) -> None:
    if not os.environ.get("AREAL_ROLLOUT_TRACE_DIR"):
        return
    import torch

    pending = []
    active: ContextVar[dict | None] = ContextVar("actor_observation", default=None)
    sequence = 0

    def drain() -> None:
        while pending and pending[0][2].query():
            row, begin, end = pending.pop(0)
            append_trace(
                "actor-phases",
                {
                    **row,
                    "timing_status": "complete",
                    "gpu_seconds": begin.elapsed_time(end) / 1000,
                },
            )

    def observe(name, original):
        @wraps(original)
        def measured(*args, **kwargs):
            nonlocal sequence
            drain()
            parent = active.get()
            sequence += 1
            row = {
                "operation": name,
                "call_id": sequence,
                "parent_call_id": None if parent is None else parent["call_id"],
                "root_phase": name if parent is None else parent["root_phase"],
                "rank": int(os.environ.get("RANK", 0)),
                "weights_version": engine.get_version(),
                "host_started": time.time(),
                "monotonic_started": time.perf_counter(),
            }
            parallel = getattr(engine, "parallel_strategy", None)
            if parallel is not None:
                row["layout"] = {
                    key: getattr(parallel, key, None)
                    for key in (
                        "data_parallel_size",
                        "tensor_parallel_size",
                        "pipeline_parallel_size",
                        "context_parallel_size",
                    )
                }
            cuda = torch.cuda.is_initialized() and name != "destroy"
            events = None
            if cuda:
                if name == "compute_logp" and parent is None:
                    torch.cuda.reset_peak_memory_stats()
                events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                events[0].record()
            token = active.set(row)
            try:
                result = original(*args, **kwargs)
                row["status"] = "returned"
                if name == "_prepare_mb_list":
                    row["microbatches"] = {
                        "real_tokens": list(result.group_lens),
                        "padded_tokens": list(result.padded_to_lengths),
                        "padding_tokens": list(result.padding_lengths),
                    }
                if name == "optimizer_step" and isinstance(result, dict):
                    row["optimizer"] = {
                        key: value
                        for key, value in result.items()
                        if isinstance(value, int | float | bool | str) or value is None
                    }
                return result
            except BaseException as error:
                row.update(status="raised", error_type=type(error).__name__)
                raise
            finally:
                active.reset(token)
                row.update(
                    host_finished=time.time(),
                    host_seconds=time.perf_counter() - row["monotonic_started"],
                )
                if events is not None:
                    events[1].record()
                    row["memory"] = {
                        "allocated": torch.cuda.memory_allocated(),
                        "reserved": torch.cuda.memory_reserved(),
                        "peak_allocated": torch.cuda.max_memory_allocated(),
                        "peak_reserved": torch.cuda.max_memory_reserved(),
                    }
                    pending.append((row, *events))
                    drain()
                else:
                    append_trace("actor-phases", {**row, "timing_status": "host_only"})
                if name == "destroy":
                    drain()
                    for unfinished, _, _ in pending:
                        append_trace(
                            "actor-phases",
                            {**unfinished, "timing_status": "incomplete_at_shutdown"},
                        )
                    pending.clear()

        return measured

    for name in (
        "compute_logp",
        "compute_advantages",
        "ppo_update",
        "forward_batch",
        "train_batch",
        "forward_backward_batch",
        "optimizer_step",
        "_prepare_mb_list",
        "full_checkpoint",
        "destroy",
    ):
        original = getattr(engine, name, None)
        if callable(original):
            setattr(engine, name, observe(name, original))

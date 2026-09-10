# SPDX-License-Identifier: Apache-2.0

"""Training work accounting, reduced before forming throughput ratios."""

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist

from areal.utils.flops import FlopsEstimator


class TrainingMetrics:
    def __init__(self, estimator: FlopsEstimator | None) -> None:
        self.estimator = estimator
        self._pending: list[tuple[torch.Tensor, float]] = []
        self._totals = [0.0, 0.0, 0.0]

    @contextmanager
    def measure(
        self, batch: dict[str, Any], synchronize: Callable[[], None]
    ) -> Iterator[None]:
        # Keep lengths on their existing device; transfer once at export.
        lengths = batch["attention_mask"].detach().bool().sum(dim=-1)
        synchronize()
        start = time.perf_counter()
        yield
        synchronize()
        self._pending.append((lengths, time.perf_counter() - start))

    def export(
        self,
        *,
        dp_group: dist.ProcessGroup | None,
        timing_group: dist.ProcessGroup | None,
    ) -> dict[str, float]:
        """Sum unique DP work; divide by MAX accumulated rank training time.

        timing_group is an explicit CPU group containing all training ranks.
        Empty intervals (e.g. evaluation-only exports) produce no metrics.
        """
        elapsed = sum(t for _, t in self._pending)
        lengths = [n for lens, _ in self._pending for n in lens.cpu().tolist()]
        tokens = sum(lengths)
        flops = sum(self.estimator(n) for n in lengths) if self.estimator else 0.0
        if not math.isfinite(flops) or flops < 0:
            raise ValueError("FLOPs estimator must return finite nonnegative FLOPs")
        device = "cpu"
        if dist.is_initialized() and dist.get_backend(dp_group) != "gloo":
            from areal.infra.platforms import current_platform

            device = current_platform.device_type
        # Ascend does not support float64 device tensors. FLOPs are estimates;
        # retain exact integer token counts in a separate reduction on all backends.
        flops_dtype = torch.float32 if device == "npu" else torch.float64
        token_work = torch.tensor(tokens, dtype=torch.int64, device=device)
        flop_work = torch.tensor(flops, dtype=flops_dtype, device=device)
        duration = torch.tensor(elapsed, dtype=torch.float64, device="cpu")
        if dist.is_initialized():
            dist.all_reduce(token_work, op=dist.ReduceOp.SUM, group=dp_group)
            dist.all_reduce(flop_work, op=dist.ReduceOp.SUM, group=dp_group)
            dist.all_reduce(duration, op=dist.ReduceOp.MAX, group=timing_group)
        elapsed = duration.item()
        self._pending.clear()
        if elapsed <= 0:
            return {}
        tokens, flops = token_work.item(), flop_work.item()
        for i, value in enumerate((tokens, flops, elapsed)):
            self._totals[i] += value
        result = {
            "train_perf/tokens": tokens,
            "train_perf/seconds": elapsed,
            "train_perf/tokens_per_second": tokens / elapsed,
            "train_perf/total_tokens": self._totals[0],
            "train_perf/total_seconds": self._totals[2],
            "train_perf/cumulative_tokens_per_second": self._totals[0]
            / self._totals[2],
        }
        if self.estimator is not None:
            result.update(
                {
                    "train_perf/estimated_flops": flops,
                    "train_perf/estimated_flops_per_second": flops / elapsed,
                    "train_perf/cumulative_estimated_flops_per_second": self._totals[1]
                    / self._totals[2],
                }
            )
        return result


@contextmanager
def record_training_batch(
    engine: Any, batch: dict[str, Any], model_config: Any
) -> Iterator[None]:
    """Shared engine integration; lazily initialize per-engine counters."""
    from contextlib import nullcontext

    from areal.infra.platforms import current_platform
    from areal.utils.flops import get_flops_estimator

    if not hasattr(engine, "_training_metrics"):
        estimator = get_flops_estimator(model_config)
        engine._training_metrics = TrainingMetrics(estimator)
    moe = getattr(engine, "_moe_metrics", None)
    with (
        engine._training_metrics.measure(batch, current_platform.synchronize),
        moe.measure() if moe is not None else nullcontext(),
    ):
        yield


def export_training_metrics(engine: Any) -> dict[str, float]:
    """Export outside the algorithm's scope to the dedicated train_perf scope."""
    metrics = getattr(engine, "_training_metrics", None)
    if metrics is None:
        return {}
    return metrics.export(
        dp_group=engine.data_parallel_group, timing_group=engine.cpu_group
    )

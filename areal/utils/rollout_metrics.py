# SPDX-License-Identifier: Apache-2.0
"""Task-local quality metrics, carried through prefetch and dynamic filtering.

Quality scalars describe attempts consumed while forming a batch, including rejected
attempts. Their ``__count`` values count observations, not final training rows.
Group metrics (e.g. pass@k) count groups; episode reward counts recorded episodes.
Tensor distributions, operational metrics and proxy PRM keep their existing scopes.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Any, Protocol, TypeVar

Metrics = dict[str, list[float]]
_CAPTURE: ContextVar[Metrics | None] = ContextVar("rollout_metrics", default=None)
_DENOMINATORS = {
    "reward": "reward_sample_count",
    "accepted": "group_attempt_count",
    "task_attempt/success": "task_attempt_count",
    "harness_success": "task_result_count",
}
_COUNTERS = {"accepted", "rejected", "filter_rejected"}


def _is_quality_metric(key: str) -> bool:
    return key.startswith(("rollout/", "debug/task/")) and not key.startswith(
        ("rollout/arena/", "rollout/prm_", "rollout/spec_accept_")
    )


def reduce_rollout_metrics(
    stats: dict[str, float], *, group: Any, is_data_parallel_head: bool
) -> dict[str, float]:
    """Merge consumed quality metrics from SPMD heads, preserving local telemetry.

    Every trainer rank participates using its CPU process group. Model-parallel
    replicas contribute no observations; averages are weighted by their own
    denominators and public counts are summed.
    """
    import torch.distributed as dist

    counts = set(_DENOMINATORS.values()) | {f"{key}_count" for key in _COUNTERS}
    quality = {}
    for key, value in stats.items():
        if not _is_quality_metric(key):
            continue
        if (
            key.endswith("__count")
            or f"{key}__count" in stats
            or key.rsplit("/", 1)[-1] in counts
        ):
            quality[key] = value
    gathered: list[dict[str, float] | None] = [None] * dist.get_world_size(group)
    dist.all_gather_object(
        gathered, quality if is_data_parallel_head else {}, group=group
    )
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}
    for rank_stats in gathered:
        for key, value in rank_stats.items():
            weight = rank_stats.get(f"{key}__count")
            if weight is not None:
                totals[key] = totals.get(key, 0.0) + value * weight
                weights[key] = weights.get(key, 0.0) + weight
            else:
                totals[key] = totals.get(key, 0.0) + value
    merged = {key: value for key, value in stats.items() if key not in quality}
    for key, value in totals.items():
        if key not in weights:
            merged[key] = value
        elif weights[key] > 0:
            merged[key] = value / weights[key]
    return merged


def rollout_stream_name(data: dict[str, Any] | None) -> str | None:
    """Use the dataset's configured name, falling back to its Arena stream ID."""
    if not data:
        return None
    name = data.get("arena_stream_name") or data.get("stream_id")
    return str(name) if name else None


def capture_scalar(key: str, value: float) -> bool:
    metrics = _CAPTURE.get()
    # Operational and proxy PRM metrics retain their wall-clock export window.
    if metrics is None or not _is_quality_metric(key):
        return False
    metrics.setdefault(key, []).append(float(value))
    return True


@contextmanager
def capture_metrics() -> Iterator[Metrics]:
    metrics: Metrics = {}
    token = _CAPTURE.set(metrics)
    try:
        yield metrics
    finally:
        _CAPTURE.reset(token)


class _MetricResult(Protocol):
    trajectory: dict[str, Any] | None
    metrics: Metrics | None


_Result = TypeVar("_Result", bound=_MetricResult)


class ConsumedMetrics:
    """Per-consumer buffer; rejected attempts count but pending results do not."""

    def __init__(self) -> None:
        self._metrics: Metrics = {}
        self._lock = Lock()

    def consume(self, result: _Result | None) -> _Result | None:
        if result is None:
            return None
        with self._lock:
            for key, values in (result.metrics or {}).items():
                self._metrics.setdefault(key, []).extend(values)
        return result if result.trajectory is not None else None

    def export(self) -> dict[str, float]:
        with self._lock:
            metrics, self._metrics = self._metrics, {}
        result = {}
        for key, values in metrics.items():
            if values:
                result[key] = sum(values) / len(values)
                result[f"{key}__count"] = len(values)
        # StatsLogger hides internal __count companions. Publish denominators
        # explicitly so dashboards can distinguish episodes, groups and Arena calls.
        for key, values in metrics.items():
            for metric in _DENOMINATORS.keys() | _COUNTERS:
                suffix = f"/{metric}"
                if not key.endswith(suffix):
                    continue
                scope = key.removesuffix(suffix)
                if scope != "rollout" and not scope.startswith("rollout/stream/"):
                    continue
                if metric in _DENOMINATORS:
                    result[f"{scope}/{_DENOMINATORS[metric]}"] = len(values)
                if metric in _COUNTERS:
                    result[f"{scope}/{metric}_count"] = sum(values)
        return result

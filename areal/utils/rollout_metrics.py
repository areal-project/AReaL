# SPDX-License-Identifier: Apache-2.0
"""Task-local quality metrics, carried through prefetch and dynamic filtering.

Quality scalars describe attempts consumed while forming a batch, including rejected
attempts. Their ``__count`` values count observations, not final training rows.
Group metrics (e.g. pass@k) count groups; episode reward counts recorded episodes.
Tensor distributions, operational metrics and proxy PRM keep their existing scopes.
"""

from collections import Counter, defaultdict
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


def _is_selfcheck_count(key: str) -> bool:
    """Identify additive Self Check counters in a rollout worker export."""
    scope, separator, metric = key.rpartition("/selfcheck/")
    if not separator or (
        scope != "rollout" and not scope.startswith("rollout/stream/")
    ):
        return False
    if metric in {"unknown_calls_count", "final_candidate/changed_count"}:
        return True
    group, _, name = metric.partition("/")
    return group in {"reward_0", "reward_1"} and (
        name == "trajectory_count"
        or (name.startswith("exact_calls_") and name.endswith("/count"))
    )


def _add_selfcheck_ratios(stats: dict[str, float]) -> None:
    """Use the merged number of known-call trajectories as each ratio's base."""
    for key, count in list(stats.items()):
        if not _is_selfcheck_count(key) or not key.endswith("/count"):
            continue
        group, _, _ = key.rpartition("/exact_calls_")
        total = stats.get(f"{group}/trajectory_count", 0)
        if total:
            stats[f"{key.removesuffix('/count')}/ratio"] = count / total


def merge_rollout_counters(
    worker_stats: list[dict[str, float]],
    consumed_stats: dict[str, float] | None = None,
) -> dict[str, float]:
    """Merge rollout counters and derive ratios from their combined totals."""
    totals: dict[str, float] = defaultdict(float)
    for stats in worker_stats:
        for key, count in stats.items():
            if _is_selfcheck_count(key):
                totals[key] += count

    for key, count in (consumed_stats or {}).items():
        if _is_selfcheck_count(key):
            totals[key] += count
        elif key.endswith("_count"):
            # Existing denominators and counters are scoped to local consumption.
            totals[key] = count

    _add_selfcheck_ratios(totals)
    return dict(totals)


def _export_selfcheck_counts(metrics: Metrics, result: dict[str, float]) -> None:
    for key, unknown_calls in metrics.items():
        if not key.endswith("/selfcheck/unknown_calls") or not _is_selfcheck_count(
            f"{key}_count"
        ):
            continue
        prefix = key.removesuffix("/unknown_calls")
        result[f"{prefix}/unknown_calls_count"] = sum(unknown_calls)
        max_calls = int(max(metrics.get(f"{prefix}/_max_calls", (0,))))
        for reward in (0, 1):
            group = f"{prefix}/reward_{reward}"
            calls = metrics.get(f"{group}/mean_calls", [])
            buckets = Counter(calls)
            result[f"{group}/trajectory_count"] = len(calls)
            counts = set(range(max_calls + 1)) | {int(count) for count in buckets}
            for count in sorted(counts):
                result[f"{group}/exact_calls_{count}/count"] = buckets[count]


def _is_quality_metric(key: str) -> bool:
    return key.startswith(("rollout/", "debug/task/")) and not key.startswith(
        ("rollout/arena/", "rollout/prm_", "rollout/spec_accept_")
    )


def get_arena_stream_name(data: dict[str, Any] | None) -> str | None:
    """Return the configured Arena stream name, falling back to its stream ID."""
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
            if key.endswith(("/selfcheck/_max_calls", "/selfcheck/unknown_calls")):
                continue
            if values:
                if key.endswith("/final_candidate/changed") and _is_selfcheck_count(
                    f"{key}_count"
                ):
                    result[f"{key}_count"] = sum(values)
                    continue
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
        _export_selfcheck_counts(metrics, result)
        _add_selfcheck_ratios(result)
        return result

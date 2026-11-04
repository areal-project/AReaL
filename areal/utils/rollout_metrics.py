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
from typing import Any, Literal, Protocol, TypeVar

Metrics = dict[str, list[float]]
_CAPTURE: ContextVar[Metrics | None] = ContextVar("rollout_metrics", default=None)
_ATTEMPT_SECONDS = "__rollout_group_attempt_seconds"
_DISCARD_SECONDS = {
    "filter_rejected": "__rollout_group_filter_rejected_seconds",
    "below_min": "__rollout_group_below_min_seconds",
    "other_discarded": "__rollout_group_other_discarded_seconds",
}


def mark_group_discard(reason: Literal["filter_rejected", "below_min"]) -> None:
    """Mark a task-local discard reason without publishing an incomplete metric."""
    metrics = _CAPTURE.get()
    if metrics is not None:
        metrics[f"__rollout_group_discard_{reason}"] = [1.0]


def capture_scalar(key: str, value: float) -> bool:
    metrics = _CAPTURE.get()
    # Operational and proxy PRM metrics retain their wall-clock export window.
    quality = key.startswith(("rollout/", "debug/task/")) and not key.startswith(
        ("rollout/arena/", "rollout/prm_", "rollout/spec_accept_")
    )
    if metrics is None or not quality:
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
            if values and not key.startswith("__rollout_group_"):
                result[key] = sum(values) / len(values)
                result[f"{key}__count"] = len(values)
        attempt_seconds = sum(metrics.get(_ATTEMPT_SECONDS, ()))
        if attempt_seconds:
            result["timeperf/rollout_group_attempt_seconds"] = attempt_seconds
            for reason, key in _DISCARD_SECONDS.items():
                seconds = sum(metrics.get(key, ()))
                result[f"timeperf/rollout_group_{reason}_seconds"] = seconds
                result[f"timeperf/rollout_group_{reason}_share"] = (
                    seconds / attempt_seconds
                )
        # StatsLogger hides internal __count companions. Publish the two main
        # denominators explicitly so dashboards can distinguish episodes/groups.
        if "rollout/reward" in metrics:
            result["rollout/reward_sample_count"] = len(metrics["rollout/reward"])
        if "rollout/accepted" in metrics:
            result["rollout/group_attempt_count"] = len(metrics["rollout/accepted"])
        return result

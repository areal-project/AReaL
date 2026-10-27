# SPDX-License-Identifier: Apache-2.0
"""Task-local quality metrics, carried through prefetch and dynamic filtering.

Captured quality scalars describe attempts consumed while forming a batch, including
rejected attempts. Their ``__count`` values count observations, not final training
rows. ``rollout/reward`` keeps its generation-side scope; its batch-collection
counterpart is ``rollout/consumed_reward``. Group metrics (e.g. pass@k) count groups;
episode reward counts recorded episodes. Tensor distributions, operational metrics and
proxy PRM keep their existing scopes.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock
from typing import Any, Protocol, TypeVar

Metrics = dict[str, list[float]]
_CAPTURE: ContextVar[Metrics | None] = ContextVar("rollout_metrics", default=None)


def capture_scalar(key: str, value: float) -> bool:
    metrics = _CAPTURE.get()
    # Operational and proxy PRM metrics retain their wall-clock export window.
    quality = key.startswith(("rollout/", "debug/task/")) and not key.startswith(
        ("rollout/arena/", "rollout/prm_", "rollout/spec_accept_")
    )
    if metrics is None or not quality:
        return False
    if key == "rollout/reward":
        # Preserve the existing generation-side reward while also recording the
        # reward of attempts dequeued by the batch collector under a separate key.
        metrics.setdefault("rollout/consumed_reward", []).append(float(value))
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
        # StatsLogger hides internal __count companions. Publish the two main
        # denominators explicitly so dashboards can distinguish episodes/groups.
        if "rollout/consumed_reward" in metrics:
            result["rollout/consumed_reward_sample_count"] = len(
                metrics["rollout/consumed_reward"]
            )
        if "rollout/accepted" in metrics:
            result["rollout/group_attempt_count"] = len(metrics["rollout/accepted"])
        return result

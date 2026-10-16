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
from typing import Protocol, TypeVar

from areal.utils.episode_turns import is_turn_distribution, publish_turn_distribution

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
    metrics: Metrics | None


_Result = TypeVar("_Result", bound=_MetricResult)


class ConsumedMetrics:
    """Per-consumer buffer; rejected attempts count but pending results do not."""

    def __init__(self, *, publish_to_tracker: bool = False) -> None:
        self._publish_to_tracker = publish_to_tracker
        self._metrics: Metrics = {}
        self._lock = Lock()

    def consume(self, result: _Result | None) -> _Result | None:
        if result is None:
            return None
        metrics = result.metrics or {}
        if self._publish_to_tracker:
            # SPMD actor export owns DP reduction. Publish only at consumption,
            # on DP heads, retaining observation counts rather than rank means.
            import torch

            from areal.utils import stats_tracker
            from areal.utils.stats_tracker import ReduceType

            tracker = stats_tracker.get()
            for key, values in metrics.items():
                if is_turn_distribution(key):
                    publish_turn_distribution(key, values)
                    continue
                if key.endswith("/num_turns_missing_count"):
                    tracker.stat_compact(
                        torch.ones(1, dtype=torch.bool),
                        ReduceType.SUM,
                        **{key: torch.tensor([sum(values)], dtype=torch.float32)},
                    )
                    continue
                for value in values:
                    tracker.scalar(**{key: value})
            for key, count_key in (
                ("rollout/reward", "rollout/reward_sample_count"),
                ("rollout/accepted", "rollout/group_attempt_count"),
            ):
                if key in metrics:
                    tracker.stat_compact(
                        torch.ones(1, dtype=torch.bool),
                        ReduceType.SUM,
                        **{
                            count_key: torch.tensor(
                                [len(metrics[key])], dtype=torch.float32
                            )
                        },
                    )
        else:
            with self._lock:
                for key, values in metrics.items():
                    self._metrics.setdefault(key, []).extend(values)
        return (
            result
            if not hasattr(result, "trajectory") or result.trajectory is not None
            else None
        )

    def export(self) -> dict[str, float]:
        with self._lock:
            metrics, self._metrics = self._metrics, {}
        result = {}
        for key, values in metrics.items():
            if values:
                if is_turn_distribution(key):
                    result.update(
                        {
                            f"{key}/avg": sum(values) / len(values),
                            f"{key}/min": min(values),
                            f"{key}/max": max(values),
                            f"{key}_count": len(values),
                        }
                    )
                    continue
                if key.endswith("/num_turns_missing_count"):
                    result[key] = sum(values)
                    continue
                result[key] = sum(values) / len(values)
                result[f"{key}__count"] = len(values)
        # StatsLogger hides internal __count companions. Publish the two main
        # denominators explicitly so dashboards can distinguish episodes/groups.
        if "rollout/reward" in metrics:
            result["rollout/reward_sample_count"] = len(metrics["rollout/reward"])
        if "rollout/accepted" in metrics:
            result["rollout/group_attempt_count"] = len(metrics["rollout/accepted"])
        return result

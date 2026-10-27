# SPDX-License-Identifier: Apache-2.0
"""Task-local quality metrics, carried through prefetch and dynamic filtering.

Quality scalars describe attempts consumed while forming a batch, including rejected
attempts. Their ``__count`` values count observations, not final training rows.
Group metrics (e.g. pass@k) count groups; episode reward counts recorded episodes.
Tensor distributions, operational metrics and proxy PRM keep their existing scopes.
"""

from threading import Lock

import torch

from areal.utils import stats_tracker
from areal.utils.episode_turns import is_turn_distribution, publish_turn_distribution
from areal.utils.scalar_capture import Metrics, suspend_scalar_capture
from areal.utils.stats_tracker import DistributedStatsTracker, ReduceType


def _publish_metrics(tracker: DistributedStatsTracker, metrics: Metrics) -> None:
    """Use the same reduction definitions for local and distributed consumers."""
    for key, values in metrics.items():
        if not values:
            continue
        if is_turn_distribution(key):
            publish_turn_distribution(key, values, tracker=tracker)
        elif key.endswith("/num_turns_missing_count"):
            tracker.stat_compact(
                torch.ones(1, dtype=torch.bool),
                ReduceType.SUM,
                **{key: torch.tensor([sum(values)], dtype=torch.float32)},
            )
        else:
            for value in values:
                tracker.scalar(**{key: value})
    # Public denominators supplement the tracker's internal __count companions.
    for key, count_key in (
        ("rollout/reward", "rollout/reward_sample_count"),
        ("rollout/accepted", "rollout/group_attempt_count"),
    ):
        if key in metrics:
            tracker.stat_compact(
                torch.ones(1, dtype=torch.bool),
                ReduceType.SUM,
                **{count_key: torch.tensor([len(metrics[key])], dtype=torch.float32)},
            )


class ConsumedMetrics:
    """Per-consumer buffer; rejected attempts count but pending results do not."""

    def __init__(self, *, publish_to_tracker: bool = False) -> None:
        # Controllers own an isolated export/reset window. SPMD publishes on DP
        # heads into the actor tracker, whose existing export owns DP reduction.
        self._tracker = None if publish_to_tracker else DistributedStatsTracker()
        self._lock = Lock()

    def record(self, metrics: Metrics | None) -> None:
        """Record consumed observations without interpreting workflow results."""
        if not metrics:
            return
        with self._lock, suspend_scalar_capture():
            tracker = (
                self._tracker if self._tracker is not None else stats_tracker.get()
            )
            _publish_metrics(tracker, metrics)

    def export(self) -> dict[str, float]:
        with self._lock:
            return self._tracker.export() if self._tracker is not None else {}

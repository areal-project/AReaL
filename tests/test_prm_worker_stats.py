# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

from areal.infra.controller.rollout_controller import (
    RolloutController,
    _merge_worker_stats,
)
from areal.utils.stats_tracker import DistributedStatsTracker, ReduceType


def test_worker_stats_preserve_weighted_prm_observations():
    base = "rollout/prm_metric/turn/scorer/accepted"
    exports = []
    for count in (2, 8):
        tracker = DistributedStatsTracker()
        for index in range(count):
            tracker.denominator(
                **{f"{base}/observed_count": torch.ones(1, dtype=torch.bool)}
            )
            tracker.scalar(**{f"{base}/rate": float(index == 0)})
            tracker.stat(
                f"{base}/observed_count",
                ReduceType.SUM,
                **{f"{base}/count": torch.tensor([float(index == 0)])},
            )
        exports.append(tracker.export())
    result = _merge_worker_stats(exports)
    assert result == pytest.approx(
        {
            f"{base}/count": 2.0,
            f"{base}/rate": 0.2,
            f"{base}/observed_count": 10,
        }
    )


def test_worker_stats_do_not_guess_custom_distribution_denominators():
    exports = []
    for values in ([5.0, 15.0], [10.0, 20.0, 30.0]):
        tracker = DistributedStatsTracker()
        tracker.denominator(n_tokens=torch.ones(len(values), dtype=torch.bool))
        tracker.stat("n_tokens", loss=torch.tensor(values))
        exports.append(tracker.export())
    assert "loss/avg" in exports[0]
    assert "n_tokens" in exports[0]
    assert _merge_worker_stats(exports) == {}


def test_rollout_controller_includes_proxy_scorer_stats():
    controller = RolloutController.__new__(RolloutController)
    controller._proxy_started = True
    controller.proxy_workers = [object()]
    controller._collective_rpc = Mock(return_value=[{}])
    controller._proxy_collective_rpc = Mock(
        return_value=[{"prm/reward": 0.25, "prm/reward__count": 4}]
    )

    assert controller.export_stats() == {"prm/reward": 0.25}
    controller._proxy_collective_rpc.assert_called_once_with(
        method="export_stats", http_timeout=60.0
    )

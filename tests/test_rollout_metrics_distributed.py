# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.utils import stats_tracker
from areal.utils.rollout_metrics import ConsumedMetrics
from areal.utils.rollout_stats import (
    ROLLOUT_HARNESS_OUTCOME,
    ROLLOUT_TURN_COUNT,
    record_rollout_turn_stats,
)


def _check_metrics(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        for empty_rank in (False, True):
            # Only consumed observations enter the usual actor DP export.
            buffer = ConsumedMetrics(publish_to_tracker=True)
            rewards = [1.0] if rank == 0 else [0.0, 0.0, 0.0]
            turns = [2.0] if rank == 0 else [4.0, 6.0, 8.0]
            outcomes = [0] if rank == 0 else [1, -1, 0]
            if empty_rank and rank == 1:
                rewards, turns, outcomes = [], [], []
            if rewards:
                buffer.record(
                    {
                        "rollout/reward": rewards,
                        "rollout/num_turns": turns,
                        "rollout/num_turns_missing_count": [float(rank)],
                        "rollout/accepted": [1.0],
                    }
                )
                record_rollout_turn_stats(
                    [
                        {
                            ROLLOUT_TURN_COUNT: torch.tensor(turns),
                            ROLLOUT_HARNESS_OUTCOME: torch.tensor(outcomes),
                        }
                    ]
                )
            else:
                record_rollout_turn_stats([])
            assert buffer.export() == {}
            result = stats_tracker.export_all(reduce_group=dist.group.WORLD)
            assert result["rollout/reward"] == (1.0 if empty_rank else 0.25)
            assert result["rollout/num_turns/avg"] == (2.0 if empty_rank else 5.0)
            assert result["rollout/num_turns_count"] == (1 if empty_rank else 4)
            assert result["rollout/num_turns_missing_count"] == (0 if empty_rank else 1)
            assert result["rollout/reward_sample_count"] == (1 if empty_rank else 4)
            assert result["rollout/group_attempt_count"] == (1 if empty_rank else 2)
            assert result["train/num_turns_count"] == (1 if empty_rank else 4)
            assert result["train/num_turns/avg"] == (2.0 if empty_rank else 5.0)
            assert result["train/num_turns/min"] == 2.0
            assert result["train/num_turns/max"] == (2.0 if empty_rank else 8.0)
            assert result["train/num_turns_no_harness_err_count"] == (
                1 if empty_rank else 2
            )
            assert stats_tracker.export_all(reduce_group=dist.group.WORLD) == {}
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_consumed_metrics_dp_uses_observations_not_rank_means(tmp_path):
    mp.spawn(
        _check_metrics, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True
    )

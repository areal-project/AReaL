# SPDX-License-Identifier: Apache-2.0

import math
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.trainer.ppo.actor import _log_version_staleness_stats
from areal.trainer.ppo.stats import log_train_inference_stats
from areal.utils.stats_tracker import DistributedStatsTracker


def _check_distributed_diagnostics(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        group = dist.group.WORLD
        tracker = DistributedStatsTracker()
        if rank == 0:
            versions = torch.tensor([0, -1])
            mask = torch.tensor([True, False])
            trainer = torch.tensor([-1.0, float("nan")])
            rollout = torch.tensor([-3.0, float("inf")])
        else:
            versions = torch.tensor([5, 4, 5])
            mask = torch.ones(3, dtype=torch.bool)
            trainer = rollout = torch.full((3,), -1.0)
        with (
            patch("areal.trainer.ppo.actor.stats_tracker", tracker),
            patch("areal.trainer.ppo.stats.stats_tracker", tracker),
        ):
            _log_version_staleness_stats(versions, 5, mask)
            log_train_inference_stats(trainer, rollout, mask)
        result = tracker.export(reduce_group=group)
        assert result["version_stats/n_valid_generated_tokens"] == 4
        assert result["version_stats/sample_staleness_theta_avg"] == 1.5
        assert result["version_stats/sample_staleness_theta_min"] == 0
        assert result["version_stats/sample_staleness_theta_max"] == 5
        assert result["version_stats/stale_token_fraction"] == 0.5
        assert result["ppo_actor/train_infer/logp_diff/avg"] == 0.5
        assert result["ppo_actor/train_infer/ratio_outside_2"] == 0.25

        # A DP shard can have no recomputed logprob statistics at all. It must
        # still participate in collectives learned through metadata exchange.
        for valid in (True, False):
            tracker = DistributedStatsTracker()
            if rank == 0:
                with patch("areal.trainer.ppo.stats.stats_tracker", tracker):
                    log_train_inference_stats(
                        torch.tensor([-1.0]),
                        torch.tensor([-3.0]),
                        torch.tensor([valid]),
                    )
            result = tracker.export(reduce_group=group)
            assert result["ppo_actor/train_infer/n_valid_tokens"] == int(valid)
            if valid:
                assert result["ppo_actor/train_infer/logp_diff/avg"] == 2
                assert result["ppo_actor/train_infer/logp_abs_diff/max"] == 2
            else:
                assert "ppo_actor/train_infer/logp_diff/avg" not in result
                assert "ppo_actor/train_infer/logp_abs_diff/max" not in result
        tracker = DistributedStatsTracker()
        with patch("areal.trainer.ppo.stats.stats_tracker", tracker):
            log_train_inference_stats(
                torch.tensor([float("nan") if rank == 0 else -1.0]),
                torch.tensor([-1.0]),
                torch.tensor([True]),
            )
        result = tracker.export(reduce_group=group)
        assert result["ppo_actor/train_infer/nonfinite_logp_fraction"] == 0.5
        assert math.isnan(result["ppo_actor/train_infer/ratio_outside_2"])
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Requires Gloo support")
def test_diagnostics_two_cpu_ranks_reduce_unequal_and_missing_populations(tmp_path):
    mp.spawn(
        _check_distributed_diagnostics,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )

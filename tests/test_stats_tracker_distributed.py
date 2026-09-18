# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.utils.stats_tracker import DistributedStatsTracker


def _weighted_stats_worker(rank: int, init_method: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        tracker = DistributedStatsTracker()
        tracker.weighted_mean(
            "loss",
            torch.tensor(2.0 if rank == 0 else 8.0),
            torch.tensor(0.125 if rank == 0 else 0.375),
        )
        if rank == 0:
            tracker.weighted_mean("one_rank", torch.tensor(3.0), torch.tensor(0.25))
        assert tracker.export(reduce_group=dist.group.WORLD) == {
            "loss": 6.5,
            "one_rank": 3.0,
        }

        # Default groups cannot discover the key on the other rank. The wider
        # key-sync group must supply its metadata and select the override group.
        local_groups = [dist.new_group([member], backend="gloo") for member in (0, 1)]
        try:
            if rank == 0:
                tracker.weighted_mean(
                    "override",
                    torch.tensor(7.0),
                    torch.tensor(0.25),
                    reduce_group=dist.group.WORLD,
                )
            assert tracker.export(
                reduce_group=local_groups[rank], key_sync_group=dist.group.WORLD
            ) == {"override": 7.0}
        finally:
            dist.destroy_process_group(local_groups[rank])
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_weighted_mean_gloo_missing_rank_and_override_group_export(tmp_path):
    mp.spawn(
        _weighted_stats_worker,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )

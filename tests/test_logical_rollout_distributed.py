# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.api.cli_args import NormConfig
from areal.infra.dist_rollout import _all_gather_ragged_trajectory_lists
from areal.utils.data import (
    Normalization,
    RolloutGroup,
    TrajBatchMeta,
    normalize_rollout_rewards,
)


def _check_logical_rollouts(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        counts = (2, 1) if rank == 0 else (1, 3)
        refs = (1.0, 3.0) if rank == 0 else (6.0, 10.0)
        rows = torch.tensor([1.0, 2.0, 3.0] if rank == 0 else [6.0, 10.0, 11.0, 12.0])
        group = RolloutGroup(counts, refs)
        trajectory = {"input_ids": torch.ones(sum(counts), 2), "rollout_group": group}
        for ragged in (False, True):
            local = [trajectory] * (rank + 1 if ragged else 1)
            gathered = _all_gather_ragged_trajectory_lists(
                local, group=dist.group.WORLD
            )
            assert [len(part) for part in gathered] == ([1, 2] if ragged else [1, 1])
            assert gathered[0][0]["rollout_group"] == RolloutGroup((2, 1), (1.0, 3.0))
            assert gathered[1][0]["rollout_group"] == RolloutGroup((1, 3), (6.0, 10.0))
        meta = TrajBatchMeta(1, [sum(counts)], [2], [group])
        for leave_out in (False, True):
            norm = Normalization(
                NormConfig(
                    mean_level="batch",
                    std_level="batch",
                    mean_leave1out=leave_out,
                    std_unbiased=False,
                    eps=0.0,
                )
            )
            actual = normalize_rollout_rewards(
                rows, norm, meta, reduce_group=dist.group.WORLD
            )
            member_refs = torch.tensor(refs).repeat_interleave(torch.tensor(counts))
            mean = (20.0 - member_refs) / 3 if leave_out else 5.0
            scale = (11.5**0.5) * (4 / 3 if leave_out else 1)
            torch.testing.assert_close(
                actual, (rows - mean) / scale, rtol=1e-6, atol=1e-6
            )
        tied_meta = TrajBatchMeta(
            1, [sum(counts)], [2], [RolloutGroup(counts, (1.0, 1.0))]
        )
        actual = normalize_rollout_rewards(
            rows, norm, tied_meta, reduce_group=dist.group.WORLD
        )
        torch.testing.assert_close(actual, rows - 1.0, rtol=0, atol=0)

        # Only one rank has ambiguous rows; both fail before statistic collectives.
        if rank == 0:
            meta.rollout_groups = [RolloutGroup(counts)]
        with pytest.raises(RuntimeError, match="explicit rollout_reward"):
            normalize_rollout_rewards(rows, norm, meta, reduce_group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
def test_logical_metadata_gathers_and_batch_statistics_count_members_across_ranks(
    tmp_path,
):
    mp.spawn(
        _check_logical_rollouts,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )

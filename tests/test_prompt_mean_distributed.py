# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.api.cli_args import MicroBatchSpec
from areal.utils.data import split_padded_tensor_dict_into_mb_list


def _check_atomic_group_failure(rank, rendezvous, scenario, explicit_group):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        group = dist.new_group(backend="gloo") if explicit_group else None
        if scenario == "oversized":
            lengths = [8] if rank == 0 else [4]
        else:
            # Both ranks can initially pack, but rank 1 needs two microbatches.
            # Rank 0 cannot match that count without splitting its prompt group.
            lengths = [6] if rank == 0 else [4, 4]
        attention_mask = (
            torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
        )
        data = {
            "attention_mask": attention_mask,
            "input_ids": torch.zeros_like(attention_mask, dtype=torch.long),
            "group_sizes": [1] * len(lengths),
        }
        with pytest.raises(RuntimeError) as exc:
            split_padded_tensor_dict_into_mb_list(
                data,
                MicroBatchSpec(n_mbs=1, max_tokens_per_mb=6),
                group=group,
            )
        messages = [None, None]
        dist.all_gather_object(messages, str(exc.value))
        assert messages[0] == messages[1]
        expected = "max_tokens_per_mb" if scenario == "oversized" else "min_groups"
        assert expected in messages[0]
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("scenario", ["oversized", "retry"])
@pytest.mark.parametrize("explicit_group", [False, True])
def test_atomic_group_failures_reach_every_rank(tmp_path, scenario, explicit_group):
    mp.spawn(
        _check_atomic_group_failure,
        args=((tmp_path / "rendezvous").as_uri(), scenario, explicit_group),
        nprocs=2,
        join=True,
    )

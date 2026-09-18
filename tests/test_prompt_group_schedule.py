# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.api.cli_args import MicroBatchSpec
from areal.engine.core.train_engine import (
    compute_microbatch_loss_weight,
    compute_total_loss_weight,
)
from areal.utils.data import (
    TRANSPORT_DUMMY_KEY,
    make_transport_microbatch,
    split_padded_tensor_dict_into_mb_list,
    split_training_batch_into_microbatches,
)
from areal.utils.functional.loss_aggregation import (
    PromptMean,
    prepare_prompt_token_weights,
)


def _prompt_batch(group_sizes: list[int]) -> dict[str, Any]:
    rows = sum(group_sizes)
    attention_mask = torch.arange(3)[None, :] <= torch.arange(rows)[:, None] % 3
    weights = torch.zeros(rows, 3)
    offset = 0
    for size in group_sizes:
        mask = attention_mask[offset : offset + size]
        weights[offset : offset + size] = mask.float() / mask.count_nonzero()
        offset += size
    return {
        "input_ids": torch.arange(rows)[:, None].expand(rows, 3).clone(),
        "attention_mask": attention_mask,
        "loss_mask": attention_mask.clone(),
        "prompt_token_weights": weights,
        "group_sizes": group_sizes,
    }


def _assert_schedule_preserves_groups(
    data: dict[str, Any], schedule: list[dict[str, Any]]
) -> None:
    expected_groups = []
    offset = 0
    for size in data["group_sizes"]:
        expected_groups.append(tuple(range(offset, offset + size)))
        offset += size

    actual_groups = []
    for microbatch in schedule:
        if microbatch.get(TRANSPORT_DUMMY_KEY):
            assert "group_sizes" not in microbatch
            assert microbatch["attention_mask"][0, 0]
            assert not microbatch["loss_mask"].any()
            assert not microbatch["prompt_token_weights"].any()
            continue
        rows = microbatch["input_ids"][:, 0]
        offset = 0
        for size in microbatch["group_sizes"]:
            actual_groups.append(tuple(rows[offset : offset + size].tolist()))
            offset += size
        assert offset == len(rows)
        torch.testing.assert_close(
            microbatch["prompt_token_weights"],
            data["prompt_token_weights"][rows],
            rtol=0,
            atol=0,
        )
    assert sorted(actual_groups) == sorted(expected_groups)


@pytest.mark.parametrize("group_sizes", [[4], [2, 1]])
@pytest.mark.parametrize("n_mbs", [1, 2, 5])
def test_prompt_schedule_fewer_groups_adapts_without_splitting(group_sizes, n_mbs):
    data = _prompt_batch(group_sizes)

    schedule = split_training_batch_into_microbatches(data, n_mbs=n_mbs)

    assert len(schedule) == min(len(group_sizes), n_mbs)
    assert all(TRANSPORT_DUMMY_KEY not in mb for mb in schedule)
    _assert_schedule_preserves_groups(data, schedule)


def test_prompt_transport_dummy_zeros_precomputed_weights():
    data = _prompt_batch([2, 1])

    dummy = make_transport_microbatch(data)

    assert dummy[TRANSPORT_DUMMY_KEY]
    assert "group_sizes" not in dummy
    assert dummy["attention_mask"][0, 0]
    assert not dummy["loss_mask"].any()
    assert not dummy["prompt_token_weights"].any()
    assert data["prompt_token_weights"].sum() > 0


def _assert_optimizer_step_objective(step: dict[str, Any], rank: int) -> bool:
    def token_values(batch: dict[str, Any]) -> torch.Tensor:
        return batch["input_ids"].float() + 10 * rank + 1 + torch.arange(3)

    # The oracle uses complete group masks, independently of precomputed weights
    # and the engine's fragment partition and normalization helpers.
    group_means = []
    offset = 0
    for size in step.get("group_sizes", []):
        mask = step["loss_mask"][offset : offset + size]
        values = token_values(step)[offset : offset + size]
        group_means.append(values[mask].mean())
        offset += size
    oracle = torch.tensor([sum(group_means), len(group_means)], dtype=torch.float32)
    dist.all_reduce(oracle)
    expected_gradient = oracle[0] / oracle[1]

    inner_data = {key: value for key, value in step.items() if key != "group_sizes"}
    microbatches = split_padded_tensor_dict_into_mb_list(
        inner_data,
        MicroBatchSpec(n_mbs=1, max_tokens_per_mb=3),
        sync_mbs=False,
    )
    reduction = PromptMean()

    def loss_weight(batch: dict[str, Any]) -> torch.Tensor:
        return reduction.normalizer(
            batch["loss_mask"], prompt_token_weights=batch["prompt_token_weights"]
        )

    total_weight = compute_total_loss_weight(
        microbatches, loss_weight, dist.group.WORLD
    )
    torch.testing.assert_close(total_weight, oracle[1], rtol=1e-6, atol=1e-6)
    parameter = torch.tensor(1.7, requires_grad=True)
    local_loss = parameter * 0
    saw_fractional_weight = False
    for batch in microbatches.mbs:
        weight = compute_microbatch_loss_weight(batch, loss_weight)
        if weight == 0:
            continue
        saw_fractional_weight |= bool((weight > 0) & (weight < 1))
        fragment_loss = reduction.aggregate(
            parameter * token_values(batch),
            batch["loss_mask"],
            prompt_token_weights=batch["prompt_token_weights"],
        )
        local_loss = local_loss + fragment_loss * weight / total_weight

    # Engines compensate for DDP's gradient averaging by multiplying each
    # rank's contribution by DP size. Reproduce that reduction on scalar grads.
    world_size = dist.get_world_size()
    gradient = torch.autograd.grad(local_loss * world_size, parameter)[0]
    dist.all_reduce(gradient)
    gradient /= world_size
    loss = local_loss.detach().clone()
    dist.all_reduce(loss)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        loss, parameter.detach() * expected_gradient, rtol=1e-6, atol=1e-6
    )
    return saw_fractional_weight


def _prompt_schedule_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        saw_fractional_weight = False
        for sizes_by_rank, requested, expected in (
            (([3], [1, 2]), 5, 3),
            (([3, 1, 2], [2]), 2, 2),
            (([4], [2]), 4, 2),
        ):
            data = _prompt_batch(sizes_by_rank[rank])
            data["prompt_token_weights"] = prepare_prompt_token_weights(
                data["loss_mask"], data["group_sizes"]
            )
            schedule = split_training_batch_into_microbatches(
                data, n_mbs=requested, group=dist.group.WORLD
            )

            assert len(schedule) == expected
            _assert_schedule_preserves_groups(data, schedule)
            local_semantic = [TRANSPORT_DUMMY_KEY not in mb for mb in schedule]
            all_semantic = [None, None]
            dist.all_gather_object(all_semantic, local_semantic)
            assert all(any(step) for step in zip(*all_semantic, strict=True))
            assert sum(local_semantic) == min(len(sizes_by_rank[rank]), requested)
            for step in schedule:
                saw_fractional_weight |= _assert_optimizer_step_objective(step, rank)
        assert saw_fractional_weight
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_prompt_schedule_uneven_ranks_preserves_groups_without_all_dummy_steps(
    tmp_path,
):
    mp.spawn(
        _prompt_schedule_worker,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )

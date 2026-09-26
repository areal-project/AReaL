# SPDX-License-Identifier: Apache-2.0

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data import (
    concat_batch,
    pack_tensor_dict,
    split_padded_tensor_dict_into_mb_list,
    split_training_batch_into_microbatches,
)


def _groups():
    groups = []
    row = 0
    for lengths in ([4, 3], [2], [1]):
        attention_mask = torch.arange(4)[None, :] < torch.tensor(lengths)[:, None]
        loss_mask = attention_mask.clone()
        loss_mask[:, 0] = False
        advantages = torch.arange(1, 4 * len(lengths) + 1).reshape(-1, 4).float()
        groups.append(
            {
                "input_ids": torch.arange(row, row + len(lengths))[:, None]
                .expand(-1, 4)
                .clone(),
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "advantages": advantages,
                "logprobs": torch.zeros_like(advantages),
                "prox_logp": torch.zeros_like(advantages),
                "kl_rewards": torch.zeros_like(advantages),
                "tot_rewards": torch.zeros_like(advantages),
                "rewards": torch.ones(len(lengths)),
            }
        )
        row += len(lengths)
    return groups


def _actor(mode, n_steps, train_batch):
    actor = object.__new__(PPOActor)
    actor.config = PPOActorConfig(
        loss_aggregation=mode,
        loss_aggregation_divisor=4.0 if mode == "constant" else None,
        ppo_n_minibatches=n_steps,
    )
    actor._mopd_loss_config = None
    actor.m2_threshold = None
    actor.engine = SimpleNamespace(
        train=lambda: None,
        train_batch=train_batch,
        get_version=lambda: 0,
        data_parallel_group=None,
        device=torch.device("cpu"),
    )
    return actor


@pytest.mark.parametrize("mode", ["token_mean", "seq_mean", "prompt_mean", "constant"])
def test_actor_modes_keep_original_response_schedule(mode):
    groups = _groups()
    batched, _ = concat_batch(groups)
    expected = [
        mb["input_ids"][:, 0].tolist()
        for mb in split_training_batch_into_microbatches(batched, n_mbs=4)
    ]
    visited = []

    def train_batch(batch, **kwargs):
        assert "group_sizes" not in batch
        visited.append(batch["input_ids"][:, 0].tolist())
        return {}

    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()):
        _actor(mode, 4, train_batch).ppo_update(groups)

    assert visited == expected
    assert len(visited) == 4  # Three groups still produce four response steps.


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("requested_steps", [1, 3, 6])
def test_actor_prompt_step_mean_matches_frozen_full_group_loss_and_gradient(
    packed, requested_steps
):
    """Fixed-parameter step averaging preserves the full original prompt objective."""
    groups = _groups()
    parameter = torch.zeros(4, requires_grad=True)
    oracle_gradient = torch.zeros_like(parameter)
    offset = 0
    active_groups = 2
    for group in groups:
        mask = group["loss_mask"]
        denominator = mask.count_nonzero()
        if denominator:
            oracle_gradient[offset : offset + len(mask)] = (
                -(group["advantages"] * mask).sum(-1) / denominator / active_groups
            )
        offset += len(mask)
    expected_full_loss = oracle_gradient.sum()
    batched, _ = concat_batch(groups)
    expected_steps = split_training_batch_into_microbatches(
        batched, n_mbs=requested_steps
    )
    step_losses = []
    saw_zero_step = False

    def train_batch(batch, *, loss_fn, loss_weight_fn):
        nonlocal saw_zero_step
        rows = batch["input_ids"][:, 0]
        expected_rows = expected_steps[len(step_losses)]["input_ids"][:, 0]
        torch.testing.assert_close(rows, expected_rows, rtol=0, atol=0)
        microbatches = split_padded_tensor_dict_into_mb_list(
            batch, MicroBatchSpec(max_tokens_per_mb=4), sync_mbs=False
        ).mbs
        losses, weights = [], []
        for microbatch in reversed(microbatches):
            if packed:
                microbatch = pack_tensor_dict(microbatch)
            logprobs = parameter[microbatch["input_ids"]]
            losses.append(
                loss_fn(
                    logprobs=logprobs,
                    entropy=torch.zeros_like(logprobs),
                    input_data=microbatch,
                )
            )
            weights.append(loss_weight_fn(microbatch))
        total_weight = sum(weights)
        # Even a real response with no original PG tokens owns a step weight.
        torch.testing.assert_close(
            total_weight, torch.tensor(1.0), rtol=1e-6, atol=1e-6
        )
        loss = (
            sum(value * weight for value, weight in zip(losses, weights)) / total_weight
        )
        expected_step = len(expected_steps) * oracle_gradient[rows].sum()
        torch.testing.assert_close(loss, expected_step, rtol=1e-6, atol=1e-6)
        if not batch["loss_mask"].any():
            saw_zero_step = True
            torch.testing.assert_close(loss, torch.tensor(0.0), rtol=0, atol=0)
        step_losses.append(loss)
        return {}

    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()):
        _actor("prompt_mean", requested_steps, train_batch).ppo_update(groups)

    mean_loss = torch.stack(step_losses).mean()
    gradient = torch.autograd.grad(mean_loss, parameter)[0]
    torch.testing.assert_close(mean_loss, expected_full_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(gradient, oracle_gradient, rtol=1e-6, atol=1e-6)
    if requested_steps >= 4:
        assert saw_zero_step


def _distributed_fixed_prompt_objective(rank, rendezvous, requested_steps=6):
    import torch.distributed as dist

    from areal.engine.core.train_engine import (
        compute_microbatch_loss_weight,
        compute_total_loss_weight,
    )
    from areal.trainer.ppo.loss_reduction import (
        prepare_policy_gradient_batch,
    )
    from areal.utils.data import TRANSPORT_DUMMY_KEY

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        groups = _groups() if rank == 0 else [_groups()[1]]
        data, meta = concat_batch(groups)

        def token_values(batch):
            return batch["input_ids"].float() + 10 * rank + 1 + torch.arange(4)

        def retained(batch):
            return batch["loss_mask"] & (batch["input_ids"] % 2 == 0)

        oracle_sum = 0.0
        active_groups = 0
        for group in groups:
            count = group["loss_mask"].count_nonzero()
            if count:
                oracle_sum += (
                    token_values(group) * retained(group)
                ).sum().item() / count.item()
                active_groups += 1
        oracle = torch.tensor([oracle_sum, active_groups])
        dist.all_reduce(oracle)
        expected_gradient = oracle[0] / oracle[1]

        prepared = prepare_policy_gradient_batch(
            data, mode="prompt_mean", group_sizes=meta.traj_group_sizes
        )
        schedule = split_training_batch_into_microbatches(
            data, n_mbs=requested_steps, group=dist.group.WORLD
        )
        steps = prepared.for_steps(
            schedule,
            dp_group=dist.group.WORLD,
            device="cpu",
        )
        assert len(steps) == min(requested_steps, 5)
        gradients, losses = [], []
        saw_zero_real = False
        saw_fractional_weight = False
        for data_step, step in zip(schedule, steps, strict=True):
            microbatches = split_padded_tensor_dict_into_mb_list(
                data_step, MicroBatchSpec(max_tokens_per_mb=4), sync_mbs=False
            )
            total_weight = compute_total_loss_weight(
                microbatches, step.loss_weight, dist.group.WORLD
            )
            torch.testing.assert_close(
                total_weight, torch.tensor(1.0), rtol=1e-6, atol=1e-6
            )
            parameter = torch.tensor(1.7, requires_grad=True)
            local_loss = parameter * 0
            for microbatch in microbatches.mbs:
                weight = compute_microbatch_loss_weight(microbatch, step.loss_weight)
                if not weight:
                    assert microbatch[TRANSPORT_DUMMY_KEY]
                    continue
                saw_fractional_weight |= bool(weight < 1)
                if not microbatch["loss_mask"].any():
                    saw_zero_real = True
                    assert TRANSPORT_DUMMY_KEY not in microbatch
                # Pack real token channels; the bound reducer does not reconstruct groups.
                values = token_values(microbatch)
                mask = retained(microbatch)
                attention = microbatch["attention_mask"]
                packed = pack_tensor_dict(microbatch)
                value = step.bind(packed).aggregate(
                    parameter * values[attention], mask[attention]
                )
                local_loss = local_loss + value * weight / total_weight
            gradient = torch.autograd.grad(local_loss * 2, parameter)[0]
            dist.all_reduce(gradient)
            gradient /= 2
            loss = local_loss.detach().clone()
            dist.all_reduce(loss)
            gradients.append(gradient)
            losses.append(loss)
        torch.testing.assert_close(
            torch.stack(gradients).mean(), expected_gradient, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            torch.stack(losses).mean(), 1.7 * expected_gradient, rtol=1e-6, atol=1e-6
        )
        if requested_steps == 6:
            assert saw_zero_real == (rank == 0)
        else:
            assert saw_fractional_weight

        # Every DP participant rejects the same globally undefined objective.
        data["loss_mask"].zero_()
        prepared = prepare_policy_gradient_batch(
            data, mode="prompt_mean", group_sizes=meta.traj_group_sizes
        )
        with pytest.raises(ValueError, match="active prompt groups"):
            prepared.for_steps(
                [data],
                dp_group=dist.group.WORLD,
                device="cpu",
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(
    not torch.distributed.is_gloo_available(), reason="Gloo is unavailable"
)
@pytest.mark.parametrize("requested_steps", [2, 6])
def test_distributed_prompt_fixed_scale_preserves_frozen_objective(
    tmp_path, requested_steps
):
    torch.multiprocessing.spawn(
        _distributed_fixed_prompt_objective,
        args=((tmp_path / "rendezvous").as_uri(), requested_steps),
        nprocs=2,
        join=True,
    )

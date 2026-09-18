# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from areal.api.cli_args import MicroBatchSpec, PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data import pack_tensor_dict, split_padded_tensor_dict_into_mb_list


@pytest.mark.parametrize("packed", [False, True])
def test_actor_splits_large_prompt_groups_inside_adaptive_optimizer_steps(packed):
    """Actual actor callbacks preserve group objectives through engine packing."""
    groups = []
    for group_id, lengths in enumerate(([4, 3, 2], [3, 2])):
        attention_mask = torch.arange(4)[None, :] < torch.tensor(lengths)[:, None]
        loss_mask = attention_mask.clone()
        loss_mask[:, 0] = False
        advantages = torch.arange(1, 4 * len(lengths) + 1).reshape(-1, 4).float()
        groups.append(
            {
                "input_ids": torch.full_like(advantages, group_id, dtype=torch.long),
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
    expected = {
        group_id: -group["advantages"][group["loss_mask"]].mean()
        for group_id, group in enumerate(groups)
    }
    visited = []

    def train_batch(batch, *, loss_fn, loss_weight_fn):
        assert "group_sizes" not in batch
        group_ids = batch["input_ids"][:, 0].unique().tolist()
        assert len(group_ids) == 1
        visited.extend(group_ids)
        microbatches = split_padded_tensor_dict_into_mb_list(
            batch, MicroBatchSpec(max_tokens_per_mb=4), sync_mbs=False
        ).mbs
        assert len(microbatches) > 1  # Every whole group exceeds the token cap.
        parameter = torch.tensor(0.0, requires_grad=True)
        losses, weights = [], []
        for microbatch in microbatches:
            if packed:
                microbatch = pack_tensor_dict(microbatch)
            logprobs = parameter.expand_as(microbatch["logprobs"])
            losses.append(
                loss_fn(
                    logprobs=logprobs,
                    entropy=torch.zeros_like(logprobs),
                    input_data=microbatch,
                )
            )
            weights.append(loss_weight_fn(microbatch))
        assert all(0 < weight < 1 for weight in weights)
        total_weight = sum(weights)
        torch.testing.assert_close(
            total_weight, torch.tensor(1.0), rtol=1e-6, atol=1e-6
        )
        loss = (
            sum(value * weight for value, weight in zip(losses, weights)) / total_weight
        )
        loss.backward()
        torch.testing.assert_close(loss, expected[group_ids[0]], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            parameter.grad, expected[group_ids[0]], rtol=1e-6, atol=1e-6
        )
        return {}

    actor = object.__new__(PPOActor)
    actor.config = PPOActorConfig(loss_aggregation="prompt_mean", ppo_n_minibatches=5)
    actor._mopd_loss_config = None
    actor.m2_threshold = None
    actor.engine = SimpleNamespace(
        train=lambda: None,
        train_batch=train_batch,
        get_version=lambda: 0,
        data_parallel_group=None,
    )
    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()):
        actor.ppo_update(groups)

    assert sorted(visited) == [0, 1]

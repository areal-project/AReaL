# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.actor import PPOActor


class DummyEngine:
    def __init__(self):
        pass


class TestStepLevelRewardsPPO:
    def _create_actor(self, gae_timestep_unit: str = "token", mask_no_eos_with_zero: bool = True):
        config = PPOActorConfig(
            gae_timestep_unit=gae_timestep_unit,
            mask_no_eos_with_zero=mask_no_eos_with_zero,
            discount=0.9,
            gae_lambda=1.0,
            kl_ctl=0.0,
            reward_bias=0.0,
            reward_scaling=1.0,
            reward_clip=10.0,
            adv_norm=None,
            reward_norm=None,
            use_decoupled_loss=False,
            recompute_logprob=False,
        )
        return PPOActor(config, engine=DummyEngine())

    def test_step_rewards_token_gae(self):
        actor = self._create_actor(gae_timestep_unit="token")

        bs = 1
        seqlen = 6
        input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
        attention_mask = torch.ones((bs, seqlen), dtype=torch.bool)
        loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.float32)
        logprobs = torch.zeros((bs, seqlen), dtype=torch.float32)
        rewards = torch.tensor([1.0])
        is_truncated = torch.tensor([False])

        data_baseline = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "logprobs": logprobs,
            "rewards": rewards,
            "is_truncated": is_truncated,
        }
        res_baseline = actor._compute_advantages(data_baseline)
        adv_baseline = res_baseline["advantages"]

        # Now add step_rewards at token index 2
        step_rewards = torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0]], dtype=torch.float32)
        data_step = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "logprobs": logprobs,
            "rewards": rewards,
            "step_rewards": step_rewards,
            "is_truncated": is_truncated,
        }
        res_step = actor._compute_advantages(data_step)
        adv_step = res_step["advantages"]

        # Step rewards should strictly increase the advantage at the active loss tokens
        assert torch.all(adv_step[:, 1:3] >= adv_baseline[:, 1:3])
        assert adv_step[0, 1].item() > adv_baseline[0, 1].item()

    def test_turn_level_rewards_turn_gae(self):
        actor = self._create_actor(gae_timestep_unit="turn")

        bs = 1
        seqlen = 6
        input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
        attention_mask = torch.ones((bs, seqlen), dtype=torch.bool)
        # Turn 0: tokens 1, 2; Turn 1: tokens 3, 4
        loss_mask = torch.tensor([[0, 1, 1, 1, 1, 0]], dtype=torch.float32)
        turn_ids = torch.tensor([[-1, 0, 0, 1, 1, -1]], dtype=torch.int32)
        logprobs = torch.zeros((bs, seqlen), dtype=torch.float32)
        rewards = torch.tensor([0.0])  # No outcome reward, only turn rewards
        is_truncated = torch.tensor([False])

        turn_rewards = torch.tensor([[0.5, 1.0]], dtype=torch.float32)
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "turn_ids": turn_ids,
            "logprobs": logprobs,
            "rewards": rewards,
            "turn_rewards": turn_rewards,
            "is_truncated": is_truncated,
        }
        res = actor._compute_advantages(data)
        adv = res["advantages"]

        # Turn 1 should receive advantage = 1.0 (since gamma=0.9, V=0)
        # Turn 0 should receive advantage = 0.5 + 0.9 * 1.0 = 1.4
        # Active turn 0 tokens (indices 0 and 1) have 1.4, active turn 1 tokens (indices 2 and 3) have 1.0
        assert pytest.approx(adv[0, 0].item(), abs=1e-4) == 1.4
        assert pytest.approx(adv[0, 1].item(), abs=1e-4) == 1.4
        assert pytest.approx(adv[0, 2].item(), abs=1e-4) == 1.0
        assert pytest.approx(adv[0, 3].item(), abs=1e-4) == 1.0

    def test_backward_compatibility_identical_to_baseline(self):
        actor = self._create_actor(gae_timestep_unit="token")

        bs = 2
        seqlen = 8
        input_ids = torch.randint(10, 100, (bs, seqlen))
        attention_mask = torch.ones((bs, seqlen), dtype=torch.bool)
        loss_mask = torch.tensor([
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
        ], dtype=torch.float32)
        logprobs = torch.randn((bs, seqlen)) * 0.1
        rewards = torch.tensor([1.5, -0.5])
        is_truncated = torch.tensor([False, False])

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "logprobs": logprobs,
            "rewards": rewards,
            "is_truncated": is_truncated,
        }
        res = actor._compute_advantages(data)
        assert "advantages" in res
        assert "returns" in res
        assert res["advantages"].shape == (bs, seqlen)
        assert res["returns"].shape == (bs, seqlen)

    def test_truncation_masking_zeros_step_rewards(self):
        actor = self._create_actor(gae_timestep_unit="token", mask_no_eos_with_zero=True)

        bs = 1
        seqlen = 6
        input_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
        attention_mask = torch.ones((bs, seqlen), dtype=torch.bool)
        loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0]], dtype=torch.float32)
        logprobs = torch.zeros((bs, seqlen), dtype=torch.float32)
        rewards = torch.tensor([1.0])
        step_rewards = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0, 0.0]], dtype=torch.float32)
        is_truncated = torch.tensor([True])

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "logprobs": logprobs,
            "rewards": rewards,
            "step_rewards": step_rewards,
            "is_truncated": is_truncated,
        }
        res = actor._compute_advantages(data)
        adv = res["advantages"]
        # Truncated trajectory with mask_no_eos_with_zero should zero out advantages
        assert torch.all(adv == 0.0)

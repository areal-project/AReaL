# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from examples.vlm.pacman.objectives import PacmanObjective
from examples.vlm.pacman.policy import TokenSetDistribution

from areal.api.cli_args import NormConfig, PPOActorConfig
from areal.utils.data import TrajBatchMeta


class TestPacmanObjectives:
    @staticmethod
    def batch(rewards: torch.Tensor) -> dict:
        rows = rewards.numel()
        return {
            "rewards": rewards,
            "loss_mask": torch.tensor([[0, 1]]).repeat(rows, 1),
            "logprobs": torch.zeros((rows, 2)),
            "prox_logp": torch.zeros((rows, 2)),
            "ref_logp": torch.zeros((rows, 2)),
        }

    @staticmethod
    def config(**kwargs) -> PPOActorConfig:
        return PPOActorConfig(
            ppo_n_minibatches=1,
            reward_norm=None,
            adv_norm=None,
            reward_clip=float("inf"),
            use_decoupled_loss=True,
            recompute_logprob=True,
            prox_logp_method="recompute",
            kl_ctl=0.01,
            **kwargs,
        )

    def test_c1_preserves_raw_reward_and_uses_proximal_kl(self):
        batch = self.batch(torch.tensor([50.0, -100.0]))
        batch["prox_logp"][:, 0] = -0.2
        batch["ref_logp"][:, 0] = -0.5
        config = self.config(kl_estimator="k1")
        result = PacmanObjective(1).compute_advantages(batch, None, config)
        torch.testing.assert_close(
            result["advantages"][:, 0],
            torch.tensor([49.997, -100.003]),
            rtol=0,
            atol=1e-5,
        )
        assert result["advantages"][:, 1].count_nonzero() == 0
        assert "loss_reduction_weights" not in result

    def test_c2_normalizes_unique_episodes_before_clip_and_weights_each_once(self):
        lengths = torch.arange(1, 13)
        returns = torch.arange(12, dtype=torch.float32) * 100
        ids = torch.arange(12).repeat_interleave(lengths)
        batch = self.batch(returns[ids])
        batch.update(
            episode_ids=ids,
            episode_returns=returns[ids],
            episode_group_sizes=torch.full_like(ids, 12),
        )
        config = self.config()
        config.reward_norm = NormConfig(
            mean_level="group", std_level="group", group_size=12
        )
        config.reward_clip = 0.5
        result = PacmanObjective(2).compute_advantages(
            batch, TrajBatchMeta(1, [ids.numel()], [2]), config
        )
        expected = (
            (returns - returns.mean()) / (returns.std() + config.reward_norm.eps)
        ).clamp(-0.5, 0.5)
        torch.testing.assert_close(
            result["rewards"], expected[ids], rtol=1e-5, atol=1e-6
        )
        mass = torch.zeros(12).scatter_add_(
            0, ids, result["loss_reduction_weights"].sum(-1)
        )
        torch.testing.assert_close(mass, torch.ones(12), rtol=1e-6, atol=1e-6)
        batch["rewards"] = batch["episode_returns"].clone()
        batch["episode_ids"][1] = 999
        with pytest.raises(ValueError, match="12 distinct"):
            PacmanObjective(2)._normalize_episodes(
                batch,
                result["loss_mask"],
                returns[ids],
                TrajBatchMeta(1, [ids.numel()], [2]),
                config,
            )


def test_token_set_distribution_matches_dense_softmax_and_gradient():
    logits = torch.tensor(
        [[0.0, 1.0, 3.0, 2.0], [2.0, 1.0, 0.0, 3.0]], requires_grad=True
    )
    support = torch.tensor([[1, 3, 0], [0, 0, 0]])
    logp, entropy = TokenSetDistribution().compute(
        logits, torch.tensor([2, 0]), {"policy_support": support}, 0.7, None
    )
    expected_logits = logits.detach().clone().requires_grad_()
    dense = torch.log_softmax(expected_logits[0, [0, 2]] / 0.7, dim=-1)
    expected_entropy = -(dense.exp() * dense).sum()
    torch.testing.assert_close(
        logp, torch.stack([dense[1], dense.new_zeros(())]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(entropy[0], expected_entropy, rtol=1e-6, atol=1e-6)
    (logp.sum() + entropy.sum()).backward()
    (dense[1] + expected_entropy).backward()
    torch.testing.assert_close(logits.grad, expected_logits.grad, rtol=1e-6, atol=1e-6)

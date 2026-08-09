from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from areal.api.cli_args import PPOActorConfig
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.trainer.ppo.actor import (
    PPOActor,
    _compute_token_level_gae,
    _compute_turn_level_gae,
)
from areal.utils.data import KLEstimator


def _make_actor(*, gae_timestep_unit: str = "token", kl_ctl: float = 0.0) -> PPOActor:
    config = PPOActorConfig(
        gae_timestep_unit=gae_timestep_unit,
        kl_ctl=kl_ctl,
    )
    actor = PPOActor.__new__(PPOActor)
    actor.config = config
    actor.reward_bias = 0.0
    actor.reward_scaling = 1.0
    actor.reward_clip = 20.0
    actor.reward_norm = None
    actor.adv_norm = None
    actor.kl_ctl = kl_ctl
    actor.kl_estimator = KLEstimator("k1")
    actor.discount = 1.0
    actor.gae_timestep_unit = gae_timestep_unit
    actor.gae_lambda = 1.0
    actor.mask_no_eos_with_zero = False
    actor.m2_threshold = None
    return actor


def _make_interaction(
    interaction_id: str,
    input_tokens: list[int],
    output_tokens: list[int],
    *,
    parent: InteractionWithTokenLogpReward | None = None,
) -> InteractionWithTokenLogpReward:
    response = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_len=len(input_tokens),
        output_len=len(output_tokens),
        output_logprobs=[-0.1] * len(output_tokens),
        output_versions=[1] * len(output_tokens),
    )
    return InteractionWithTokenLogpReward(
        model_response=response,
        reward=1.0,
        parent=parent,
        chat_template_type="concat",
        completion=SimpleNamespace(id=interaction_id, created=0),
        output_message_list=[{"role": "assistant", "content": interaction_id}],
    )


def test_turn_level_gae_broadcasts_advantage_and_steps_once_per_turn():
    """Lambda decay is applied between turns, not between action tokens."""
    rewards = torch.tensor([[0.0, 1.0, 0.0, 2.0]], dtype=torch.float32)

    advantages, returns = _compute_turn_level_gae(
        rewards=rewards,
        values=torch.zeros_like(rewards),
        loss_mask=torch.ones_like(rewards),
        turn_ids=torch.tensor([[0, 0, 1, 1]], dtype=torch.int32),
        seq_no_eos_mask=torch.tensor([False]),
        discount=1.0,
        gae_lambda=0.5,
    )

    expected = torch.tensor([[2.0, 2.0, 2.0, 2.0]], dtype=torch.float32)
    torch.testing.assert_close(advantages, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(returns, expected, rtol=0.0, atol=0.0)


def test_turn_level_gae_skips_gaps_and_uses_first_value_with_bootstrap():
    """Prompt gaps and unused IDs do not consume discount or lambda steps."""
    rewards = torch.tensor([[0.0, 3.0, 0.0, 0.0, 4.0]], dtype=torch.float32)
    values = torch.tensor([[1.0, 5.0, 9.0, 2.0, 8.0]], dtype=torch.float32)

    advantages, returns = _compute_turn_level_gae(
        rewards=rewards,
        values=values,
        loss_mask=torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]]),
        turn_ids=torch.tensor([[0, 0, -1, 2, 2]], dtype=torch.int64),
        seq_no_eos_mask=torch.tensor([True]),
        discount=0.5,
        gae_lambda=0.25,
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([[3.75, 3.75, 0.0, 6.0, 6.0]]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        returns,
        torch.tensor([[4.75, 4.75, 9.0, 8.0, 8.0]]),
        rtol=0.0,
        atol=0.0,
    )


def test_token_level_gae_matches_legacy_recurrence():
    """The default token mode remains numerically backward compatible."""
    rewards = torch.tensor(
        [[0.0, 0.5, 0.0, 2.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float32
    )
    values = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.2, 0.1, 0.0, 0.5]], dtype=torch.float32
    )
    loss_mask = torch.tensor(
        [[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]], dtype=torch.float32
    )
    seq_no_eos_mask = torch.tensor([False, True])
    discount = 0.9
    gae_lambda = 0.8

    expected_reversed = [torch.zeros(2, dtype=torch.float32)]
    lastgaelam = torch.zeros(2, dtype=torch.float32)
    nextvalues = values[:, -1] * seq_no_eos_mask
    for timestep in reversed(range(rewards.shape[1] - 1)):
        delta = rewards[:, timestep] + discount * nextvalues - values[:, timestep]
        newgaelam = delta + discount * gae_lambda * lastgaelam
        mask = loss_mask[:, timestep]
        nextvalues = nextvalues * (1 - mask) + values[:, timestep] * mask
        lastgaelam = lastgaelam * (1 - mask) + newgaelam * mask
        expected_reversed.append(lastgaelam)
    expected_advantages = torch.stack(expected_reversed[::-1], dim=1)

    advantages, returns = _compute_token_level_gae(
        rewards=rewards,
        values=values,
        loss_mask=loss_mask,
        seq_no_eos_mask=seq_no_eos_mask,
        discount=discount,
        gae_lambda=gae_lambda,
    )

    torch.testing.assert_close(advantages, expected_advantages, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        returns, expected_advantages + values, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize(
    ("turn_ids", "message"),
    [
        ([-1, 0, 0, 1], "non-negative"),
        ([0, 0, 4, 4], "smaller than the sequence length"),
        ([0, 1, 0, 2], "temporally nondecreasing"),
    ],
)
def test_turn_level_gae_rejects_malformed_active_turn_ids(turn_ids, message):
    """Malformed action-token IDs fail before scatter/gather can misroute data."""
    rewards = torch.zeros(1, 4)

    with pytest.raises(RuntimeError, match=message):
        _compute_turn_level_gae(
            rewards=rewards,
            values=torch.zeros_like(rewards),
            loss_mask=torch.ones_like(rewards),
            turn_ids=torch.tensor([turn_ids]),
            seq_no_eos_mask=torch.tensor([False]),
            discount=1.0,
            gae_lambda=1.0,
        )


def test_turn_level_gae_rejects_non_integral_ids():
    """Turn IDs must retain their structural integer representation."""
    rewards = torch.zeros(1, 2)

    with pytest.raises(ValueError, match="integer dtype"):
        _compute_turn_level_gae(
            rewards=rewards,
            values=torch.zeros_like(rewards),
            loss_mask=torch.ones_like(rewards),
            turn_ids=torch.tensor([[0.0, 0.0]]),
            seq_no_eos_mask=torch.tensor([False]),
            discount=1.0,
            gae_lambda=1.0,
        )


def test_turn_level_main_path_keeps_kl_token_local_and_returns_task_only():
    """Token KL affects actor advantages but does not enter critic targets."""
    actor = _make_actor(gae_timestep_unit="turn", kl_ctl=1.0)
    batch = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "loss_mask": torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.float32),
        "turn_ids": torch.tensor([[-1, 0, 0, 1, 1]], dtype=torch.int32),
        "logprobs": torch.tensor([[0.0, 0.0, 0.2, 0.4, 0.0]], dtype=torch.float32),
        "ref_logp": torch.zeros(1, 5, dtype=torch.float32),
        "attention_mask": torch.ones(1, 5, dtype=torch.bool),
        "rewards": torch.tensor([2.0]),
    }

    result = actor._compute_advantages(batch)

    torch.testing.assert_close(
        result["returns"],
        torch.tensor([[2.0, 2.0, 2.0, 2.0, 0.0]]),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        result["advantages"],
        torch.tensor([[2.0, 1.8, 1.6, 2.0, 0.0]]),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        result["tot_rewards"],
        torch.tensor([[0.0, -0.2, -0.4, 2.0, 0.0]]),
        rtol=0.0,
        atol=1e-6,
    )


def test_turn_level_main_path_requires_turn_ids():
    """Enabling turn recurrence fails clearly for legacy rollout payloads."""
    actor = _make_actor(gae_timestep_unit="turn")
    batch = {
        "input_ids": torch.zeros(1, 3, dtype=torch.long),
        "loss_mask": torch.tensor([[0, 1, 1]], dtype=torch.float32),
        "logprobs": torch.zeros(1, 3, dtype=torch.float32),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "rewards": torch.tensor([1.0]),
    }

    with pytest.raises(ValueError, match="requires rollout data.*turn_ids"):
        actor._compute_advantages(batch)


def test_ppo_actor_config_rejects_unknown_gae_timestep_unit():
    """The string-backed config enforces the two supported timestep units."""
    with pytest.raises(ValueError, match="gae_timestep_unit"):
        PPOActorConfig(gae_timestep_unit="episode")


def test_ppo_actor_config_is_omegaconf_structured_config_compatible():
    """The turn selector works with the project's pinned structured config path."""
    config = OmegaConf.structured(PPOActorConfig())

    assert config.gae_timestep_unit == "token"


def test_concat_interaction_emits_structured_turn_ids():
    """A concat child preserves parent turns and labels its own output."""
    parent = _make_interaction("parent", [1, 2], [3, 4])
    child = _make_interaction("child", [1, 2, 3, 4, 5], [6, 7], parent=parent)

    turn_ids = child.to_tensor_dict()["turn_ids"].squeeze(0)

    torch.testing.assert_close(
        turn_ids,
        torch.tensor([-1, -1, 0, 0, -1, 1, 1], dtype=torch.int32),
        rtol=0.0,
        atol=0.0,
    )

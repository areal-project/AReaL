"""Small CPU-only tests for the two PRM-to-advantage paths."""

from __future__ import annotations

import pytest
import torch

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.actor import (
    PPOActor,
    _shape_advantages_with_gvpo,
    _shape_advantages_with_process_weighting,
    resolve_gae_lambda_fn,
)
from areal.utils.data import KLEstimator


def _actor(
    *,
    direct: bool,
    gae_timestep_unit: str = "token",
    mask_no_eos_with_zero: bool = False,
) -> PPOActor:
    actor = PPOActor.__new__(PPOActor)
    actor.config = PPOActorConfig(
        token_rewards_as_adv=direct,
        gae_timestep_unit=gae_timestep_unit,
        kl_ctl=0.0,
    )
    actor.reward_bias = 0.0
    actor.reward_scaling = 1.0
    actor.reward_clip = 20.0
    actor.reward_norm = None
    actor.adv_norm = None
    actor.kl_ctl = 0.0
    actor.kl_estimator = KLEstimator("k1")
    actor.discount = 1.0
    actor.gae_lambda = 1.0
    actor.gae_lambda_fn, actor._gae_lambda_is_custom = resolve_gae_lambda_fn(1.0)
    actor.gae_lambda_kwargs = {}
    actor.gae_timestep_unit = gae_timestep_unit
    actor.mask_no_eos_with_zero = mask_no_eos_with_zero
    actor.token_rewards_as_adv = direct
    return actor


def _batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.zeros(1, 8, dtype=torch.long),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 0, 1, 1, 1]]),
        "turn_ids": torch.tensor([[-1, -1, 0, 0, -1, 1, 1, 1]]),
        "logprobs": torch.zeros(1, 8),
        "attention_mask": torch.ones(1, 8, dtype=torch.bool),
        "rewards": torch.tensor([1.0]),
        "token_rewards": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, -0.2, -0.2, -0.2]]),
    }


@pytest.mark.parametrize("gae_timestep_unit", ["token", "turn"])
def test_direct_mode_adds_local_reward_but_keeps_returns_outcome_only(
    gae_timestep_unit,
):
    output = _actor(
        direct=True,
        gae_timestep_unit=gae_timestep_unit,
    )._compute_advantages(_batch())

    torch.testing.assert_close(
        output["advantages"][0, [1, 2, 4, 5, 6]],
        torch.tensor([1.0, 1.0, 0.8, 0.8, 0.8]),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output["returns"][0, [1, 2, 4, 5, 6]],
        torch.ones(5),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output["tot_rewards"][0, [1, 2, 4, 5, 6]],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("gae_timestep_unit", ["token", "turn"])
def test_fold_mode_places_one_turn_reward_before_gae(gae_timestep_unit):
    output = _actor(
        direct=False,
        gae_timestep_unit=gae_timestep_unit,
    )._compute_advantages(_batch())

    torch.testing.assert_close(
        output["advantages"][0, [1, 2, 4, 5, 6]],
        torch.full((5,), 0.8),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output["returns"][0, [1, 2, 4, 5, 6]],
        torch.full((5,), 0.8),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output["tot_rewards"][0, [1, 2, 4, 5, 6]],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.8]),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("gae_timestep_unit", ["token", "turn"])
@pytest.mark.parametrize("second_reward", [0.2, 0.4])
def test_fold_mode_respects_adjacent_turn_ids(gae_timestep_unit, second_reward):
    batch = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        "loss_mask": torch.tensor([[0, 1, 1, 1, 1]]),
        "turn_ids": torch.tensor([[-1, 0, 0, 1, 1]]),
        "logprobs": torch.zeros(1, 5),
        "attention_mask": torch.ones(1, 5, dtype=torch.bool),
        "rewards": torch.zeros(1),
        "token_rewards": torch.tensor([[0, 0.2, 0.2, second_reward, second_reward]]),
    }
    output = _actor(
        direct=False, gae_timestep_unit=gae_timestep_unit
    )._compute_advantages(batch)
    torch.testing.assert_close(
        output["tot_rewards"][0],
        torch.tensor([0, 0.2, 0, second_reward, 0]),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        output["returns"][0, :4],
        torch.tensor(
            [0.2 + second_reward, 0.2 + second_reward, second_reward, second_reward]
        ),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("gae_timestep_unit", ["token", "turn"])
def test_fold_mode_rejects_nonuniform_rewards_inside_one_turn(gae_timestep_unit):
    batch = _batch()
    batch["token_rewards"][0, 6] = -0.3

    with pytest.raises(RuntimeError, match="requires uniform token rewards"):
        _actor(
            direct=False,
            gae_timestep_unit=gae_timestep_unit,
        )._compute_advantages(batch)


@pytest.mark.parametrize("gae_timestep_unit", ["token", "turn"])
@pytest.mark.parametrize("direct", [True, False])
def test_mask_no_eos_clears_process_rewards_in_both_modes(
    direct,
    gae_timestep_unit,
):
    """A truncated trajectory must not regain a signal through process rewards."""
    batch = _batch()
    batch["rewards"] = torch.tensor([1.0])
    batch["token_rewards"][0, 6] = -0.3
    actor = _actor(
        direct=direct,
        gae_timestep_unit=gae_timestep_unit,
        mask_no_eos_with_zero=True,
    )

    output = actor._compute_advantages(batch)

    valid_positions = output["loss_mask"].bool()
    torch.testing.assert_close(
        output["advantages"][valid_positions],
        torch.zeros_like(output["advantages"][valid_positions]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("gae_timestep_unit", ["token", "turn"])
def test_missing_token_rewards_is_a_noop_for_both_modes(gae_timestep_unit):
    left = _batch()
    right = _batch()
    del left["token_rewards"]
    del right["token_rewards"]

    direct = _actor(
        direct=True,
        gae_timestep_unit=gae_timestep_unit,
    )._compute_advantages(left)
    folded = _actor(
        direct=False,
        gae_timestep_unit=gae_timestep_unit,
    )._compute_advantages(right)

    torch.testing.assert_close(
        direct["advantages"], folded["advantages"], rtol=0.0, atol=0.0
    )


def test_gvpo_helper_piecewise_truth_table_and_masking():
    eps = 0.1
    advantages = torch.tensor([[-2.0, -eps, 0.0, eps, 2.0, 3.0, 4.0]])
    process_signal = torch.tensor([[-1.0, -1.0, -1.0, -1.0, -1.0, 0.0, -1.0]])
    loss_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])

    shaped = _shape_advantages_with_gvpo(
        advantages,
        process_signal,
        loss_mask,
        negative_scale=0.2,
        zero_penalty=0.4,
        zero_eps=eps,
    )

    torch.testing.assert_close(
        shaped,
        torch.tensor([[-2.4, -0.4, -0.4, -0.4, 0.0, 3.0, 4.0]]),
    )


def test_process_weighting_helper_piecewise_truth_table_and_masking():
    """Process weighting scales, replaces, or preserves each advantage branch."""
    advantages = torch.tensor([[-2.0, -1.0, 0.0, 2.0, 3.0, 4.0]])
    process_rewards = torch.tensor([[0.0, 0.25, 0.5, 0.25, 1.0, 2.0]])
    loss_mask = torch.tensor([[1, 1, 1, 1, 1, 0]])

    shaped = _shape_advantages_with_process_weighting(
        advantages,
        process_rewards,
        loss_mask,
    )

    torch.testing.assert_close(
        shaped,
        torch.tensor([[-2.0, 0.25, 0.0, 0.5, 3.0, 4.0]]),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("process_reward", [-0.01, 1.01])
def test_process_weighting_helper_rejects_active_reward_outside_unit_interval(
    process_reward,
):
    """Active process rewards must honor the mode's documented range contract."""
    with pytest.raises(RuntimeError, match=r"process rewards in \[0, 1\]"):
        _shape_advantages_with_process_weighting(
            torch.tensor([[1.0]]),
            torch.tensor([[process_reward]]),
            torch.tensor([[1]]),
        )


@pytest.mark.parametrize("gae_timestep_unit", ["token", "turn"])
def test_process_weighting_shapes_normalized_advantage_without_changing_returns(
    gae_timestep_unit,
):
    """Process weighting runs after normalization and leaves critic targets intact."""

    class _ShiftNegative:
        def __call__(self, advantages, loss_mask, group_sizes=None):
            del loss_mask, group_sizes
            return advantages - 2.0

    batch = _batch()
    batch["token_rewards"] = torch.tensor([[0.0, 0.0, 0.25, 0.25, 0.0, 0.5, 0.5, 0.5]])
    actor = _actor(direct=True, gae_timestep_unit=gae_timestep_unit)
    actor.adv_norm = _ShiftNegative()

    output = actor._compute_advantages(
        batch,
        advantage_shaping_mode="process_weighted",
    )

    torch.testing.assert_close(
        output["advantages"][0, [1, 2, 4, 5, 6]],
        torch.tensor([0.25, 0.25, 0.5, 0.5, 0.5]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output["returns"][0, [1, 2, 4, 5, 6]],
        torch.ones(5),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output["tot_rewards"][0, [1, 2, 4, 5, 6]],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_process_weighting_runtime_rejects_folded_process_rewards():
    """Piecewise process weighting requires direct per-token shaping."""
    with pytest.raises(ValueError, match="token_rewards_as_adv=True"):
        _actor(direct=False)._compute_advantages(
            _batch(),
            advantage_shaping_mode="process_weighted",
        )


def test_process_weighting_runtime_rejects_mask_no_eos_with_zero():
    """Process weighting cannot interpret an artificial no-EOS zero as a score."""
    actor = _actor(
        direct=True,
        mask_no_eos_with_zero=True,
    )

    with pytest.raises(ValueError, match="mask_no_eos_with_zero=True"):
        actor._compute_advantages(
            _batch(),
            advantage_shaping_mode="process_weighted",
        )


@pytest.mark.parametrize("gae_timestep_unit", ["token", "turn"])
def test_gvpo_shapes_after_gae_without_changing_critic_targets(gae_timestep_unit):
    baseline_batch = _batch()
    del baseline_batch["token_rewards"]
    baseline = _actor(
        direct=True, gae_timestep_unit=gae_timestep_unit
    )._compute_advantages(baseline_batch)

    gvpo = _actor(direct=True, gae_timestep_unit=gae_timestep_unit)._compute_advantages(
        _batch(), advantage_shaping_mode="gvpo"
    )

    torch.testing.assert_close(
        gvpo["advantages"][0, [1, 2, 4, 5, 6]],
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(gvpo["returns"], baseline["returns"])
    torch.testing.assert_close(gvpo["tot_rewards"], baseline["tot_rewards"])


@pytest.mark.parametrize("gae_timestep_unit", ["token", "turn"])
def test_gvpo_uses_normalized_advantage_for_piecewise_branch(gae_timestep_unit):
    class _CenterAtOne:
        def __call__(self, advantages, loss_mask, group_sizes=None):
            del loss_mask, group_sizes
            return advantages - 1.0

    actor = _actor(direct=True, gae_timestep_unit=gae_timestep_unit)
    actor.adv_norm = _CenterAtOne()

    output = actor._compute_advantages(
        _batch(), advantage_shaping_mode="gvpo", gvpo_zero_penalty=0.4
    )

    torch.testing.assert_close(
        output["advantages"][0, [4, 5, 6]], torch.full((3,), -0.4)
    )


def test_gvpo_runtime_rejects_folded_process_rewards_even_when_missing():
    batch = _batch()
    del batch["token_rewards"]

    with pytest.raises(ValueError, match="token_rewards_as_adv=True"):
        _actor(direct=False)._compute_advantages(batch, advantage_shaping_mode="gvpo")


@pytest.mark.parametrize("gae_timestep_unit", ["token", "turn"])
def test_gvpo_mask_no_eos_matches_no_process_reward_with_kl_and_values(
    gae_timestep_unit,
):
    def _configured_actor():
        actor = _actor(
            direct=True,
            gae_timestep_unit=gae_timestep_unit,
            mask_no_eos_with_zero=True,
        )
        actor.kl_ctl = 0.2
        return actor

    with_process = _batch()
    with_process["values"] = torch.linspace(0.0, 0.7, 8).unsqueeze(0)
    with_process["logprobs"] = torch.linspace(-0.8, -0.1, 8).unsqueeze(0)
    with_process["ref_logp"] = torch.linspace(-0.4, -0.05, 8).unsqueeze(0)
    without_process = {key: value.clone() for key, value in with_process.items()}
    del without_process["token_rewards"]

    gvpo = _configured_actor()._compute_advantages(
        with_process, advantage_shaping_mode="gvpo"
    )
    baseline = _configured_actor()._compute_advantages(without_process)

    torch.testing.assert_close(gvpo["advantages"], baseline["advantages"])
    torch.testing.assert_close(gvpo["returns"], baseline["returns"])
    torch.testing.assert_close(gvpo["tot_rewards"], baseline["tot_rewards"])


@pytest.mark.parametrize("gae_timestep_unit", ["token", "turn"])
def test_gvpo_does_not_change_returns_with_dynamic_gae_lambda(gae_timestep_unit):
    def _configured_actor():
        actor = _actor(direct=True, gae_timestep_unit=gae_timestep_unit)
        actor._gae_lambda_is_custom = True
        actor.gae_lambda_fn = lambda context, **kwargs: torch.full(
            (context["timestep_lengths"].shape[0],),
            0.5,
            device=context["timestep_lengths"].device,
        )
        return actor

    baseline_batch = _batch()
    del baseline_batch["token_rewards"]
    baseline = _configured_actor()._compute_advantages(baseline_batch)
    gvpo = _configured_actor()._compute_advantages(
        _batch(), advantage_shaping_mode="gvpo"
    )

    torch.testing.assert_close(gvpo["returns"], baseline["returns"])

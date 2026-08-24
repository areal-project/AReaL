from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from areal.api.cli_args import PPOActorConfig
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.trainer.ppo.actor import PPOActor
from areal.trainer.ppo.gae import (
    _compute_token_level_gae,
    _compute_turn_level_gae,
)
from areal.trainer.ppo.lambda_fn import (
    relative_position_gae_lambda,
    resolve_gae_lambda_fn,
    vapo_length_adaptive_gae,
)
from areal.utils.data import KLEstimator


def _make_actor(
    *,
    gae_timestep_unit: str = "token",
    gae_lambda: float | str = 1.0,
    gae_lambda_kwargs: dict | None = None,
    kl_ctl: float = 0.0,
) -> PPOActor:
    config = PPOActorConfig(
        gae_timestep_unit=gae_timestep_unit,
        gae_lambda=gae_lambda,
        gae_lambda_kwargs=gae_lambda_kwargs or {},
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
    actor.gae_lambda = gae_lambda
    actor.gae_lambda_fn, actor._gae_lambda_is_custom = resolve_gae_lambda_fn(gae_lambda)
    actor.gae_lambda_kwargs = (
        dict(config.gae_lambda_kwargs) if actor._gae_lambda_is_custom else {}
    )
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
        stop_reason="stop",
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
        bootstrap_values=torch.zeros(1),
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
        bootstrap_values=values[:, -1],
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


def test_turn_level_gae_applies_one_lambda_per_sample():
    """Turn recurrence broadcasts each trajectory's own lambda across its turns."""
    rewards = torch.tensor(
        [[0.0, 1.0, 0.0, 2.0], [0.0, 1.0, 0.0, 2.0]],
        dtype=torch.float32,
    )

    advantages, _ = _compute_turn_level_gae(
        rewards=rewards,
        values=torch.zeros_like(rewards),
        loss_mask=torch.ones_like(rewards),
        turn_ids=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.int32),
        seq_no_eos_mask=torch.tensor([False, False]),
        bootstrap_values=torch.zeros(2),
        discount=1.0,
        gae_lambda=torch.tensor([0.5, 0.0]),
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([[2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0]]),
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
        bootstrap_values=values[:, -1],
        discount=discount,
        gae_lambda=gae_lambda,
    )

    torch.testing.assert_close(advantages, expected_advantages, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        returns, expected_advantages + values, rtol=0.0, atol=0.0
    )


def test_token_level_gae_applies_one_lambda_per_sample():
    """A batch lambda tensor changes recurrence independently per trajectory."""
    rewards = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    loss_mask = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]],
        dtype=torch.float32,
    )

    advantages, _ = _compute_token_level_gae(
        rewards=rewards,
        values=torch.zeros_like(rewards),
        loss_mask=loss_mask,
        seq_no_eos_mask=torch.tensor([False, False]),
        bootstrap_values=torch.zeros(2),
        discount=1.0,
        gae_lambda=torch.tensor([0.5, 0.0]),
    )

    torch.testing.assert_close(
        advantages,
        torch.tensor([[0.25, 0.5, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        rtol=0.0,
        atol=0.0,
    )


def test_static_gae_lambda_ignores_kwargs_and_legacy_missing_turn_ids():
    """A float lambda remains compatible with token workflows lacking turn IDs."""
    actor = _make_actor(
        gae_lambda=0.25,
        gae_lambda_kwargs={"ignored": "value"},
    )
    loss_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    gae_lambda = actor._compute_gae_lambda(loss_mask, turn_ids=None)

    torch.testing.assert_close(
        gae_lambda,
        torch.tensor([0.25, 0.25]),
        rtol=0.0,
        atol=0.0,
    )


def test_static_gae_lambda_main_path_skips_dynamic_context(monkeypatch):
    """A static lambda avoids per-batch trajectory length construction."""
    actor = _make_actor(gae_lambda=0.25)

    def fail_dynamic_resolution(*_args, **_kwargs):
        raise AssertionError("static lambda should not use dynamic resolution")

    monkeypatch.setattr(actor, "_compute_gae_lambda", fail_dynamic_resolution)
    batch = {
        "input_ids": torch.zeros(1, 4, dtype=torch.long),
        "loss_mask": torch.tensor([[0, 1, 1, 1]], dtype=torch.float32),
        "logprobs": torch.zeros(1, 4, dtype=torch.float32),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "rewards": torch.tensor([1.0]),
    }

    result = actor._compute_advantages(batch)

    assert result["advantages"].shape == torch.Size([1, 4])


@pytest.mark.parametrize("gae_lambda", [float("nan"), float("inf")])
def test_static_gae_lambda_main_path_rejects_non_finite_values(gae_lambda):
    """The static fast path retains the dynamic path's finite-value check."""
    actor = _make_actor(gae_lambda=gae_lambda)
    batch = {
        "input_ids": torch.zeros(1, 3, dtype=torch.long),
        "loss_mask": torch.tensor([[0, 1, 1]], dtype=torch.float32),
        "logprobs": torch.zeros(1, 3, dtype=torch.float32),
        "attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "rewards": torch.tensor([1.0]),
    }

    with pytest.raises(ValueError, match="Static gae_lambda must be finite"):
        actor._compute_advantages(batch)


def test_vapo_gae_lambda_uses_selected_token_lengths_per_sample():
    """Token mode uses canonical generated-token counts for VAPO lambda."""
    actor = _make_actor(
        gae_lambda="areal.trainer.ppo.lambda_fn.vapo_length_adaptive_gae",
        gae_lambda_kwargs={"alpha": 1.0},
    )
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    turn_ids = torch.tensor(
        [
            [0, 0, 1, 1, -1],
            [0, 1, -1, -1, -1],
            [-1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )

    gae_lambda = actor._compute_gae_lambda(loss_mask, turn_ids)

    torch.testing.assert_close(
        gae_lambda,
        torch.tensor([0.75, 0.5, 0.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_vapo_gae_lambda_uses_selected_turn_counts_per_sample():
    """Turn mode uses unique active turn counts instead of generated tokens."""
    actor = _make_actor(
        gae_timestep_unit="turn",
        gae_lambda="areal.trainer.ppo.lambda_fn.vapo_length_adaptive_gae",
        gae_lambda_kwargs={"alpha": 1.0},
    )
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    turn_ids = torch.tensor(
        [
            [0, 0, -1, 2, 2],
            [0, 0, 0, 0, -1],
            [-1, -1, -1, -1, -1],
        ],
        dtype=torch.int32,
    )

    gae_lambda = actor._compute_gae_lambda(loss_mask, turn_ids)

    torch.testing.assert_close(
        gae_lambda,
        torch.tensor([0.5, 0.0, 0.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_relative_position_gae_lambda_preserves_endpoints_and_relative_positions():
    """Different lengths share decay at equal endpoint-normalized positions."""
    q = 0.3
    lengths = torch.tensor([10, 19])
    context = {
        "effective_token_lengths": lengths,
        "turn_counts": lengths,
        "timestep_lengths": lengths,
    }

    gae_lambda = relative_position_gae_lambda(context, q=q)
    first_step_retention = gae_lambda.pow((lengths - 1).float())
    # Step 5 of 10 and step 9 of 19 are both at relative position 4 / 9.
    equal_position_retention = gae_lambda.pow(torch.tensor([5.0, 10.0]))

    torch.testing.assert_close(
        first_step_retention,
        torch.full((2,), q),
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        equal_position_retention,
        torch.full((2,), q ** (5.0 / 9.0)),
        rtol=1e-6,
        atol=1e-7,
    )


def test_relative_position_gae_lambda_handles_empty_and_single_step_trajectories():
    """Empty rows use zero while a single timestep cannot undergo decay."""
    lengths = torch.tensor([0, 1, 2])
    context = {
        "effective_token_lengths": lengths,
        "turn_counts": lengths,
        "timestep_lengths": lengths,
    }

    gae_lambda = relative_position_gae_lambda(context, q=0.3)

    torch.testing.assert_close(
        gae_lambda,
        torch.tensor([0.0, 1.0, 0.3]),
        rtol=0.0,
        atol=1e-7,
    )


@pytest.mark.parametrize("q", [True, -0.1, 0.0, 1.1, float("nan"), float("inf")])
def test_relative_position_gae_lambda_rejects_invalid_q(q):
    """The retained fraction must be finite and belong to (0, 1]."""
    lengths = torch.tensor([2])
    context = {
        "effective_token_lengths": lengths,
        "turn_counts": lengths,
        "timestep_lengths": lengths,
    }

    with pytest.raises(ValueError, match=r"q must be .* in \(0, 1\]"):
        relative_position_gae_lambda(context, q=q)


def test_relative_position_gae_lambda_main_path_without_turn_ids():
    """Token workflows need no turn metadata to decay a terminal reward."""
    q = 0.3
    trajectory_length = 10
    actor = _make_actor(
        gae_lambda=("areal.trainer.ppo.lambda_fn.relative_position_gae_lambda"),
        gae_lambda_kwargs={"q": q},
    )
    batch = {
        "input_ids": torch.zeros(1, trajectory_length + 1, dtype=torch.long),
        "loss_mask": torch.tensor([[0] + [1] * trajectory_length], dtype=torch.float32),
        "logprobs": torch.zeros(1, trajectory_length + 1, dtype=torch.float32),
        "attention_mask": torch.ones(1, trajectory_length + 1, dtype=torch.bool),
        "rewards": torch.tensor([1.0]),
    }
    expected_active_advantages = torch.tensor(
        [
            q ** ((trajectory_length - step) / (trajectory_length - 1))
            for step in range(1, trajectory_length + 1)
        ],
        dtype=torch.float32,
    )
    expected = torch.cat([expected_active_advantages, torch.zeros(1)]).unsqueeze(0)

    result = actor._compute_advantages(batch)

    torch.testing.assert_close(result["advantages"], expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(result["returns"], expected, rtol=1e-6, atol=1e-7)


def test_relative_position_gae_lambda_turn_mode_uses_effective_turn_count():
    """Turn-level lambda normalizes decay by turns rather than action tokens."""
    q = 0.3
    actor = _make_actor(
        gae_timestep_unit="turn",
        gae_lambda="areal.trainer.ppo.lambda_fn.relative_position_gae_lambda",
        gae_lambda_kwargs={"q": q},
    )
    loss_mask = torch.ones(2, 6)
    turn_ids = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )

    gae_lambda = actor._compute_gae_lambda(loss_mask, turn_ids)

    torch.testing.assert_close(
        gae_lambda,
        torch.tensor([q ** (1.0 / 2.0), 1.0]),
        rtol=1e-6,
        atol=1e-7,
    )


def test_custom_token_gae_lambda_uses_lengths_without_turn_ids():
    """A token-level custom lambda remains compatible with legacy workflows."""
    actor = _make_actor(
        gae_lambda="areal.trainer.ppo.lambda_fn.vapo_length_adaptive_gae",
        gae_lambda_kwargs={"alpha": 1.0},
    )
    loss_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    gae_lambda = actor._compute_gae_lambda(loss_mask, turn_ids=None)

    torch.testing.assert_close(
        gae_lambda,
        torch.tensor([0.5, 0.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_custom_gae_lambda_rejects_scalar_output():
    """A custom function must return one explicit value per local trajectory."""
    actor = _make_actor()

    def scalar_lambda(context):
        return torch.tensor(0.5)

    actor._gae_lambda_is_custom = True
    actor.gae_lambda_fn = scalar_lambda
    actor.gae_lambda_kwargs = {}

    with pytest.raises(ValueError, match="one value per local trajectory"):
        actor._compute_gae_lambda(
            torch.ones(2, 2),
            torch.zeros(2, 2, dtype=torch.int32),
        )


def test_vapo_gae_lambda_rejects_non_positive_alpha():
    """VAPO requires a positive scale even when the batch is empty-length."""
    context = {
        "effective_token_lengths": torch.tensor([0]),
        "turn_counts": torch.tensor([0]),
        "timestep_lengths": torch.tensor([0]),
    }

    with pytest.raises(ValueError, match="alpha must be a positive number"):
        vapo_length_adaptive_gae(context, alpha=0.0)


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
            bootstrap_values=torch.zeros(1),
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
            bootstrap_values=torch.zeros(1),
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


def test_main_path_lambda_context_counts_shifted_training_tokens_including_eos():
    """Dynamic lengths use the same canonical mask as GAE and critic training."""
    actor = _make_actor()
    captured_context = {}

    def capture_context(context):
        captured_context.update({key: value.clone() for key, value in context.items()})
        return torch.ones_like(context["timestep_lengths"], dtype=torch.float32)

    actor._gae_lambda_is_custom = True
    actor.gae_lambda_fn = capture_context
    actor.gae_lambda_kwargs = {}
    batch = {
        "input_ids": torch.zeros(1, 5, dtype=torch.long),
        # The three generated tokens include the terminal EOS at position 3.
        "loss_mask": torch.tensor([[0, 1, 1, 1, 0]], dtype=torch.float32),
        "turn_ids": torch.tensor([[-1, 0, 0, 0, -1]], dtype=torch.int32),
        "logprobs": torch.zeros(1, 5, dtype=torch.float32),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.bool),
        "rewards": torch.tensor([1.0]),
    }

    actor._compute_advantages(batch)

    torch.testing.assert_close(
        captured_context["effective_token_lengths"],
        torch.tensor([3]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        captured_context["turn_counts"],
        torch.tensor([1]),
        rtol=0.0,
        atol=0.0,
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


def test_ppo_actor_config_accepts_lambda_function_path_and_kwargs():
    """OmegaConf preserves the dynamic lambda path and custom parameters."""
    config = OmegaConf.structured(
        PPOActorConfig(
            gae_lambda="areal.trainer.ppo.lambda_fn.vapo_length_adaptive_gae",
            gae_lambda_kwargs={"alpha": 0.5},
        )
    )

    assert config.gae_lambda == ("areal.trainer.ppo.lambda_fn.vapo_length_adaptive_gae")
    assert config.gae_lambda_kwargs == {"alpha": 0.5}


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

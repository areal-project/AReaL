from unittest.mock import MagicMock

import pytest
import torch

from areal.api.cli_args import PPOActorConfig
from areal.infra.platforms import current_platform
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data import concat_batch


def _make_actor(**overrides) -> PPOActor:
    options = dict(
        backend="fsdp:d1",
        kl_ctl=0.0,
        discount=1.0,
        gae_lambda=1.0,
        reward_norm=None,
        adv_norm=None,
        use_decoupled_loss=True,
    )
    options.update(overrides)
    return PPOActor(PPOActorConfig(**options), MagicMock())


@pytest.fixture
def rollout_batch(device) -> dict[str, torch.Tensor]:
    batch = {
        "input_ids": torch.zeros(4, 7, dtype=torch.long),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
            ],
            dtype=torch.bool,
        ),
        "loss_mask": torch.tensor(
            [
                [0, 0, 1, 1, 0, 1, 1],
                [0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.int32,
        ),
        "logprobs": torch.linspace(-1.0, -3.0, 7).repeat(4, 1),
        "ref_logp": torch.linspace(-0.8, -2.4, 7).repeat(4, 1),
        "rewards": torch.tensor([1.0, -1.0, 2.0, 3.0]),
    }
    return {key: value.to(device) for key, value in batch.items()}


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("accelerator", marks=[pytest.mark.gpu, pytest.mark.npu]),
    ]
)
def device(request):
    if request.param == "cpu":
        return "cpu"
    if current_platform.device_type not in ("cuda", "npu"):
        pytest.skip("Requires CUDA or Ascend NPU")
    if current_platform.device_count() < 1:
        pytest.skip("Requires one accelerator")
    return current_platform.device_type


def _reference_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    seq_no_eos_mask: torch.Tensor,
    discount: float,
    gae_lambda: float,
) -> torch.Tensor:
    """Keep the original recurrence as an oracle for the vectorized path."""
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros(
        batch_size, seq_len, dtype=torch.float32, device=values.device
    )
    last_advantage = 0
    next_value = values[:, -1] * seq_no_eos_mask
    for t in reversed(range(seq_len - 1)):
        delta = rewards[:, t] + discount * next_value - values[:, t]
        updated_advantage = delta + discount * gae_lambda * last_advantage
        mask = loss_mask[:, t]
        next_value = next_value * (1 - mask) + values[:, t] * mask
        last_advantage = last_advantage * (1 - mask) + updated_advantage * mask
        advantages[:, t] = last_advantage
    return advantages


@pytest.mark.parametrize("mask_no_eos_with_zero", [False, True])
def test_terminal_reward_fastpath_respects_masks_and_truncation(
    rollout_batch, mask_no_eos_with_zero
):
    """Carry rewards across tool gaps, but ignore masked terminal positions."""
    actor = _make_actor(mask_no_eos_with_zero=mask_no_eos_with_zero)

    result = actor._compute_advantages(rollout_batch, concat_batch([rollout_batch])[1])

    expected = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0],
            [-1, -1, -1, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
        device=result["advantages"].device,
    )
    if mask_no_eos_with_zero:
        expected[0] = 0
    torch.testing.assert_close(result["advantages"], expected, rtol=0, atol=0)
    torch.testing.assert_close(result["returns"], expected, rtol=0, atol=0)
    assert result["advantages"].dtype == torch.float32


def test_dense_kl_fastpath_matches_original_gae(rollout_batch):
    """Nonzero token KL rewards must accumulate rather than broadcast."""
    actor = _make_actor(kl_ctl=0.2)

    result = actor._compute_advantages(rollout_batch, concat_batch([rollout_batch])[1])

    expected = _reference_gae(
        result["tot_rewards"],
        torch.zeros_like(result["tot_rewards"]),
        result["loss_mask"],
        result["attention_mask"].all(dim=-1),
        discount=1.0,
        gae_lambda=1.0,
    )
    torch.testing.assert_close(result["advantages"], expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(result["returns"], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("has_critic", "discount", "gae_lambda"),
    [(True, 1.0, 1.0), (False, 0.9, 1.0), (False, 1.0, 0.8)],
)
def test_critic_or_discounted_gae_keeps_original_recurrence(
    rollout_batch, has_critic, discount, gae_lambda
):
    """A critic or non-unit gamma/lambda must retain the general estimator."""
    actor = _make_actor(kl_ctl=0.2, discount=discount, gae_lambda=gae_lambda)
    values = torch.zeros_like(rollout_batch["logprobs"])
    if has_critic:
        values = (
            torch.arange(28, dtype=torch.float32, device=values.device).reshape(4, 7)
            / 10
        )
        rollout_batch["values"] = values

    result = actor._compute_advantages(rollout_batch, concat_batch([rollout_batch])[1])

    expected = _reference_gae(
        result["tot_rewards"],
        values,
        result["loss_mask"],
        result["attention_mask"].all(dim=-1),
        discount=discount,
        gae_lambda=gae_lambda,
    )
    torch.testing.assert_close(result["advantages"], expected, rtol=0, atol=0)
    torch.testing.assert_close(result["returns"], expected + values, rtol=0, atol=0)


def test_fastpath_single_token_sequence_has_zero_advantage(device):
    """An empty cumulative-sum body still restores the final zero column."""
    actor = _make_actor()
    batch = {
        "input_ids": torch.zeros(1, 1, dtype=torch.long, device=device),
        "attention_mask": torch.ones(1, 1, dtype=torch.bool, device=device),
        "loss_mask": torch.zeros(1, 1, dtype=torch.int32, device=device),
        "logprobs": torch.zeros(1, 1, device=device),
        "rewards": torch.ones(1, device=device),
    }

    result = actor._compute_advantages(batch, concat_batch([batch])[1])

    expected = torch.zeros(1, 1, device=device)
    torch.testing.assert_close(result["advantages"], expected, rtol=0, atol=0)
    torch.testing.assert_close(result["returns"], expected, rtol=0, atol=0)

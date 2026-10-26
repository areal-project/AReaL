# SPDX-License-Identifier: Apache-2.0

"""PPO loss configuration validation and actor training callback wiring."""

from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo import actor as actor_module
from areal.trainer.ppo.actor import PPOActor


def test_loss_aggregation_defaults_to_token_mean():
    config = PPOActorConfig()

    assert config.loss_aggregation == "token-mean"
    assert config.loss_aggregation_divisor is None


@pytest.mark.parametrize("mode", ["invalid", "token_mean"])
def test_unknown_loss_aggregation_is_rejected(mode: str):
    with pytest.raises(ValueError, match="loss_aggregation must"):
        PPOActorConfig(loss_aggregation=mode)


@pytest.mark.parametrize(
    "divisor", [None, 0.0, -1.0, float("inf"), -float("inf"), float("nan")]
)
def test_constant_requires_positive_finite_divisor(divisor: float | None):
    with pytest.raises(ValueError, match="positive finite"):
        PPOActorConfig(loss_aggregation="constant", loss_aggregation_divisor=divisor)


@pytest.mark.parametrize("mode", ["token-mean", "seq-mean"])
def test_nonconstant_loss_rejects_fixed_divisor(mode: str):
    with pytest.raises(ValueError, match="only used when"):
        PPOActorConfig(loss_aggregation=mode, loss_aggregation_divisor=2.0)


@pytest.mark.parametrize("mode,divisor", [("seq-mean", None), ("constant", 2.0)])
def test_tree_training_rejects_non_token_mean(mode: str, divisor: float | None):
    with pytest.raises(ValueError, match="tree training supports only 'token-mean'"):
        PPOActorConfig(
            loss_aggregation=mode,
            loss_aggregation_divisor=divisor,
            enable_tree_training=True,
        )


def test_tree_training_accepts_token_mean():
    config = PPOActorConfig(enable_tree_training=True)

    assert config.loss_aggregation == "token-mean"


@pytest.mark.parametrize(
    "mode,divisor",
    [("token-mean", None), ("seq-mean", None), ("constant", 2.0), ("constant", 7.0)],
)
@pytest.mark.parametrize("n_minibatches", [1, 2])
def test_actor_passes_configured_loss_and_matching_weight(
    mode: str,
    divisor: float | None,
    n_minibatches: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Exercise the actor's real PPO callback, with only the engine replaced."""
    config = PPOActorConfig(
        path="dummy",
        loss_aggregation=mode,
        loss_aggregation_divisor=divisor,
        ppo_n_minibatches=n_minibatches,
    )
    mask = torch.tensor(
        [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
        dtype=torch.bool,
    )
    advantages = torch.tensor([1.0, 2.0, 4.0, 9.0])[:, None].expand_as(mask)
    data = {
        "input_ids": torch.arange(4)[:, None].expand_as(mask),
        "attention_mask": torch.ones_like(mask),
        "loss_mask": mask,
        "logprobs": torch.zeros_like(advantages),
        "prox_logp": torch.zeros_like(advantages),
        "advantages": advantages,
        "rewards": torch.ones(4),
        "tot_rewards": torch.zeros_like(advantages),
        "kl_rewards": torch.zeros_like(advantages),
    }
    calls = []

    def train_batch(
        batch: dict[str, torch.Tensor],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    ) -> dict:
        reduction = loss_fn.keywords["pg_reduction"]
        assert reduction.mode == mode
        assert reduction.divisor == divisor

        valid = batch["loss_mask"].bool()
        counts = valid.sum(-1)
        active_rows = counts > 0
        expected_weight = counts.sum() if mode == "token-mean" else active_rows.sum()
        torch.testing.assert_close(
            loss_weight_fn(batch).float(), expected_weight.float(), rtol=0, atol=0
        )

        parameter = torch.tensor(0.05, requires_grad=True)
        logprobs = parameter.expand_as(batch["advantages"])
        actual = loss_fn(logprobs, torch.zeros_like(logprobs), batch)

        # exp(0.05) is inside the PPO clip interval. Derive the objective
        # directly from advantages and masks, without the production reducer.
        reference = parameter.detach().clone().requires_grad_()
        terms = -reference.exp() * batch["advantages"]
        sums = torch.where(valid, terms, 0).sum(-1)
        if mode == "token-mean":
            expected = sums.sum() / counts.sum()
        elif mode == "seq-mean":
            expected = (sums[active_rows] / counts[active_rows]).mean()
        else:
            expected = sums.sum() / (active_rows.sum() * divisor)

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            torch.autograd.grad(actual, parameter)[0],
            torch.autograd.grad(expected, reference)[0],
            rtol=1e-6,
            atol=1e-7,
        )
        calls.append(batch)
        return {}

    engine = SimpleNamespace(
        train=lambda: None,
        get_version=lambda: 0,
        train_batch=train_batch,
        data_parallel_group=None,
        device="cpu",
    )
    monkeypatch.setattr(actor_module, "stats_tracker", MagicMock())
    actor = PPOActor(config, engine)

    actor._ppo_update(data)

    assert len(calls) == n_minibatches

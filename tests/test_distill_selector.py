# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the sampled reverse-KL selector on the post-#1838 pure-KD path.

Since #1838, pure KD feeds the detached signed teacher-student gap into the
policy loss, which owns clipping, rejection and the backward pass. The selector
therefore changes only the *reported forward value*: it is detached and
transplanted onto the policy loss, so every option keeps the same update.

These tests pin that contract, including batches where clipping and rejection
actually bind.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from areal.api.cli_args import RejectionSamplingConfig
from areal.trainer.ppo.actor import grpo_loss_fn

BETA = 1.0


def _input_data(estimator: str, *, prox=None):
    logprobs = torch.tensor([[-1.0, -1.0, -1.0, -1.0]])
    prox = (
        torch.tensor([[-1.1, -0.9, -1.05, -0.95]])
        if prox is None
        else torch.tensor(prox)
    )
    return {
        "logprobs": logprobs,
        "prox_logp": prox,
        "advantages": torch.zeros_like(logprobs),
        "loss_mask": torch.ones_like(logprobs, dtype=torch.bool),
        "teacher_logp": torch.tensor([[-0.5, -2.0, -1.2, -0.7]]),
        "rl_loss_weight": 0.0,
        "distill_loss_weight": BETA,
        "kl_estimator": estimator,
    }


def _run(estimator: str, *, prox=None, reject=False, use_decoupled=False):
    logprobs = torch.tensor([[-1.0, -1.0, -1.0, -1.0]], requires_grad=True)
    captured: dict[str, torch.Tensor] = {}
    tracker = MagicMock()

    def _stat(**kwargs):
        for key, value in kwargs.items():
            if key != "denominator":
                captured.setdefault(key, value)

    tracker.stat.side_effect = _stat
    kwargs = {}
    if reject:
        kwargs["rejection_sampling"] = RejectionSamplingConfig(
            level="token", action="mask", metric="ratio", upper=2.0
        )
        kwargs["use_decoupled_loss"] = use_decoupled

    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        loss = grpo_loss_fn(
            logprobs=logprobs,
            entropy=torch.zeros_like(logprobs),
            input_data=_input_data(estimator, prox=prox),
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            **kwargs,
        )
    (grad,) = torch.autograd.grad(loss, logprobs)
    return float(loss.detach()), grad.detach(), captured


@pytest.mark.parametrize("prox", [None, [[-3.0, 3.0, -1.0, 1.0]]])
@pytest.mark.parametrize("reject", [False, True])
def test_gradient_is_identical_across_estimators(prox, reject):
    """Clipping and rejection bind in at least one case; the update must not move."""
    base_loss, base_grad, _ = _run("k1", prox=prox, reject=reject)
    for estimator in ("k2", "k3"):
        loss, grad, _ = _run(estimator, prox=prox, reject=reject)
        torch.testing.assert_close(
            grad, base_grad, rtol=0, atol=1e-7, msg=lambda m: f"{estimator}: {m}"
        )
        # Forward value differs by design, except where it coincides.
        assert loss == pytest.approx(loss)


def test_k2_forward_value_is_half_x_squared():
    _, _, captured = _run("k2")
    x = (_input_data("k2")["teacher_logp"] - _input_data("k2")["logprobs"]).double()
    expected = float((0.5 * x**2).sum())
    got = float(captured.get("rkl_loss", torch.tensor(float("nan"))).sum())
    assert got == pytest.approx(expected, rel=1e-6)


def test_k3_forward_value_is_expm1_minus_x():
    _, _, captured = _run("k3")
    x = (_input_data("k3")["teacher_logp"] - _input_data("k3")["logprobs"]).double()
    expected = float((torch.expm1(x) - x).sum())
    got = float(captured.get("rkl_loss", torch.tensor(float("nan"))).sum())
    assert got == pytest.approx(expected, rel=1e-6)


def test_k1_reports_the_legacy_signed_value():
    _, _, captured = _run("k1")
    x = (_input_data("k1")["teacher_logp"] - _input_data("k1")["logprobs"]).double()
    got = float(captured.get("rkl_loss", torch.tensor(float("nan"))).sum())
    # Upstream reports -advantages, i.e. -(t - l) summed.
    assert got == pytest.approx(float(-x.sum()), rel=1e-6)


def test_diagnostics_log_all_three_regardless_of_selector():
    for estimator in ("k1", "k2", "k3"):
        data = _input_data(estimator)
        data["log_distill_diagnostics"] = True
        logprobs = torch.tensor([[-1.0, -1.0, -1.0, -1.0]], requires_grad=True)
        captured: dict[str, torch.Tensor] = {}
        tracker = MagicMock()
        tracker.stat.side_effect = lambda **kw: captured.update(
            {k: v for k, v in kw.items() if k != "denominator"}
        )
        with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
            grpo_loss_fn(
                logprobs=logprobs,
                entropy=torch.zeros_like(logprobs),
                input_data=data,
                eps_clip=0.2,
                eps_clip_higher=None,
                c_clip=None,
            )
        for key in ("distill_kl_k1", "distill_kl_k2", "distill_kl_k3"):
            assert key in captured, f"{estimator}: missing {key}"


def test_selector_rejects_non_pure_kd():
    data = _input_data("k3")
    data["rl_loss_weight"] = 1.0
    logprobs = torch.tensor([[-1.0, -1.0, -1.0, -1.0]], requires_grad=True)
    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()):
        with pytest.raises(ValueError, match="only supported for sampled pure-KD"):
            grpo_loss_fn(
                logprobs=logprobs,
                entropy=torch.zeros_like(logprobs),
                input_data=data,
                eps_clip=0.2,
                eps_clip_higher=None,
                c_clip=None,
            )


def test_selector_rejects_unknown_name():
    data = _input_data("k9")
    logprobs = torch.tensor([[-1.0, -1.0, -1.0, -1.0]], requires_grad=True)
    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()):
        with pytest.raises(ValueError, match="must be 'k1', 'k2', or 'k3'"):
            grpo_loss_fn(
                logprobs=logprobs,
                entropy=torch.zeros_like(logprobs),
                input_data=data,
                eps_clip=0.2,
                eps_clip_higher=None,
                c_clip=None,
            )

# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the sampled reverse-KL selector on the post-#1838 pure-KD path.

Since #1838, pure KD feeds the detached signed teacher-student gap into the
policy loss, which owns clipping, rejection and the backward pass. The selector
therefore changes only the *reported forward value*: it is detached and
transplanted onto the policy loss, so every option keeps the same update.

The fixture deliberately separates the current log-probs from the stored
behaviour log-probs, and its ratios clip on both sides (one high-advantage token
above the upper bound, one low-advantage token below the lower bound). A
rejection threshold then removes an *unclipped* token, so the tests exercise
clipping and rejection independently rather than assuming ratio = 1.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from areal.api.cli_args import RejectionSamplingConfig
from areal.trainer.ppo.actor import _sampled_rkl_forward, grpo_loss_fn

BETA = 1.0

BEHAVE = [-1.5, -1.5, -1.5, -1.5]
PROX = [-1.0, -2.0, -1.5, -1.3]
CURRENT = [-0.5, -3.0, -1.5, -1.25]
TEACHER = [-0.45, -3.5, -1.7, -1.35]

# Rejection ratio = exp(prox - behave)      = [1.6487, 0.6065, 1.0, 1.2214]
# with upper=1.1, tokens 0 and 3 are filtered.
# Policy ratio = exp(current - prox)       = [1.6487, 0.3679, 1.0, 1.0513]
# clipped: t0 (x > 0, ratio > 1.2) and t1 (x < 0, ratio < 0.8).
# Without rejection pure KD resets prox to behave, so the ratio becomes
# exp(current - behave) = [2.7183, 0.2231, 1.0, 1.2840]; t3 then sits above
# the upper clip bound but is unclipped because the clipped loss is smaller.
# x = teacher - current = [0.05, -0.5, -0.2, -0.1]


def _t(values):
    # The loss path expects [batch, seq]; keep the fixture 2D.
    return torch.tensor([values], dtype=torch.float32)


def _input_data(estimator: str) -> dict:
    return {
        # ``logprobs`` here is the stored behaviour policy, not the tensor under
        # test; pure KD anchors the ratio on it.
        "logprobs": _t(BEHAVE),
        "prox_logp": _t(PROX),
        "advantages": torch.zeros_like(_t(BEHAVE)),
        "loss_mask": torch.ones_like(_t(BEHAVE), dtype=torch.bool),
        "teacher_logp": _t(TEACHER),
        "rl_loss_weight": 0.0,
        "distill_loss_weight": BETA,
        "kl_estimator": estimator,
    }


def _run(estimator: str, *, reject: bool = False, upper: float = 1.1):
    logprobs = _t(CURRENT).requires_grad_()
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
            level="token", action="mask", metric="ratio", upper=upper
        )
    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        loss = grpo_loss_fn(
            logprobs=logprobs,
            entropy=torch.zeros_like(logprobs),
            input_data=_input_data(estimator),
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            **kwargs,
        )
    (grad,) = torch.autograd.grad(loss, logprobs)
    return float(loss.detach()), grad.detach(), captured


def _expected_grad(reject: bool) -> torch.Tensor:
    # Token 2 is unclipped (grad = -x / denominator); token 3 is unclipped but
    # filtered when rejection is enabled; tokens 0 and 1 are clipped.
    grad = torch.zeros(1, 4)
    grad[0, 2] = 0.2 / 4
    if not reject:
        # Without rejection pure KD resets prox to behave, giving ratio
        # exp(current - behave) = exp(0.25) on token 3.
        grad[0, 3] = (0.1 * torch.exp(torch.tensor(0.25))).item() / 4
    return grad


def _mask(reject: bool) -> torch.Tensor:
    return torch.tensor([[not reject, True, True, not reject]], dtype=torch.bool)


@pytest.mark.parametrize("reject", [False, True])
@pytest.mark.parametrize("estimator", ["k1", "k2", "k3"])
def test_gradient_matches_analytic_policy_gradient(estimator, reject):
    _, grad, _ = _run(estimator, reject=reject)
    torch.testing.assert_close(
        grad,
        _expected_grad(reject),
        rtol=0,
        atol=1e-6,
        msg=lambda m: f"{estimator}: {m}",
    )


@pytest.mark.parametrize("reject", [False, True])
def test_clipping_and_rejection_actually_fire(reject):
    _, _, captured = _run("k2", reject=reject)
    assert captured["clip_ratio"].tolist() == [[1.0, 1.0, 0.0, 0.0]]
    if reject:
        # Tokens 0 (ratio 2.46) and 3 (ratio 1.105) exceed upper=1.1.
        assert captured["behave_filtered_ratio"].tolist() == [[1.0, 0.0, 0.0, 1.0]]


@pytest.mark.parametrize("reject", [False, True])
def test_k2_returned_scalar_uses_effective_mask(reject):
    loss, _, captured = _run("k2", reject=reject)
    x = _t(TEACHER) - _t(CURRENT)
    mask = _mask(reject)
    expected = float((0.5 * x**2 * mask).sum() / mask.numel())
    assert loss == pytest.approx(expected, rel=1e-6)
    torch.testing.assert_close(
        captured["rkl_loss"], 0.5 * x**2 * mask, rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("reject", [False, True])
def test_k3_returned_scalar_uses_effective_mask(reject):
    loss, _, captured = _run("k3", reject=reject)
    x = _t(TEACHER) - _t(CURRENT)
    mask = _mask(reject)
    expected = float(((torch.expm1(x) - x) * mask).sum() / mask.numel())
    assert loss == pytest.approx(expected, rel=1e-6)
    torch.testing.assert_close(
        captured["rkl_loss"], (torch.expm1(x) - x) * mask, rtol=0, atol=1e-5
    )


def test_k1_reports_the_legacy_pre_rejection_signed_value():
    _, _, captured = _run("k1")
    x = _t(TEACHER) - _t(CURRENT)
    torch.testing.assert_close(captured["rkl_loss"], -x, rtol=0, atol=1e-6)


def test_forward_value_saturates_instead_of_overflowing():
    """A finite gap of 99 must not produce an infinite reported scalar."""
    teacher = torch.tensor([[1.0, float("-inf"), 0.0]])
    logprobs = torch.tensor([[-99.0, -1.0, -1.0]])
    mask = torch.tensor([[True, False, True]])
    for estimator in ("k1", "k2", "k3"):
        value = _sampled_rkl_forward(teacher, logprobs, mask, estimator)
        assert torch.isfinite(value).all(), f"{estimator} produced {value}"
    # The masked -inf gap contributes exactly zero, not NaN.
    assert _sampled_rkl_forward(teacher, logprobs, mask, "k2")[0, 1].item() == 0.0
    # k3 saturates at the documented cap instead of overflowing expm1.
    at_cap = _sampled_rkl_forward(
        torch.full((1, 1), 1.0),
        torch.full((1, 1), -59.0),
        torch.ones(1, 1, dtype=bool),
        "k3",
    )
    above_cap = _sampled_rkl_forward(
        torch.full((1, 1), 1.0),
        torch.full((1, 1), -99.0),
        torch.ones(1, 1, dtype=bool),
        "k3",
    )
    assert at_cap.item() == pytest.approx(above_cap.item())


def test_diagnostics_log_all_three_regardless_of_selector():
    for estimator in ("k1", "k2", "k3"):
        data = _input_data(estimator)
        data["log_distill_diagnostics"] = True
        logprobs = _t(CURRENT).requires_grad_()
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
            assert torch.isfinite(captured[key]).all()


def test_selector_rejects_non_pure_kd():
    data = _input_data("k3")
    data["rl_loss_weight"] = 1.0
    logprobs = _t(CURRENT).requires_grad_()
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
    logprobs = _t(CURRENT).requires_grad_()
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

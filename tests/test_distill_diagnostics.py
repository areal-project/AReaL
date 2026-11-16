# SPDX-License-Identifier: Apache-2.0
"""CPU tests for ``teacher.log_distill_diagnostics``.

The diagnostics are additive: they must not change the loss, its gradient, or
its normalization, and they must report all three estimator values on the same
active tokens. These tests exercise the real ``grpo_loss_fn`` path with
``rl_loss_weight=0`` so masking and normalization are the production ones.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from areal.api.cli_args import TeacherConfig
from areal.trainer.ppo.actor import grpo_loss_fn

DTYPE = torch.float64
BETA = 2.0

LOGPROBS = [[-1.3, -0.7, -2.1, -0.4], [-0.9, -1.8, -0.2, -2.5]]
OLD_LOGPROBS = [[-1.1, -0.9, -2.0, -0.5], [-1.0, -1.7, -0.3, -2.4]]
TEACHER_LOGPROBS = [[-1.5, -0.6, -2.4, -0.2], [-0.7, -2.0, -0.25, -2.2]]
LOSS_MASK = [[1, 1, 1, 0], [1, 1, 1, 1]]
N_VALID = sum(sum(row) for row in LOSS_MASK)


def _t(values):
    return torch.tensor(values, dtype=DTYPE)


def _input_data(*, diagnostics: bool):
    return {
        "logprobs": _t(LOGPROBS),
        "prox_logp": _t(LOGPROBS),
        "advantages": torch.zeros_like(_t(LOGPROBS)),
        "loss_mask": torch.tensor(LOSS_MASK, dtype=torch.bool),
        "teacher_logp": _t(TEACHER_LOGPROBS),
        "rl_loss_weight": 0.0,
        "distill_loss_weight": BETA,
        "log_distill_diagnostics": diagnostics,
    }


def _run(diagnostics: bool):
    """Return (loss, grad, captured stats)."""
    logprobs = _t(LOGPROBS).requires_grad_()
    tracker = MagicMock()
    captured: dict[str, torch.Tensor] = {}

    def _stat(**kwargs):
        for key, value in kwargs.items():
            if key != "denominator":
                captured[key] = value

    tracker.stat.side_effect = _stat
    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        loss = grpo_loss_fn(
            logprobs=logprobs,
            entropy=torch.zeros_like(logprobs),
            input_data=_input_data(diagnostics=diagnostics),
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
        )
    (grad,) = torch.autograd.grad(loss, logprobs)
    return loss.detach(), grad.detach(), captured


def test_diagnostics_do_not_change_loss_or_gradient():
    """Enabling the diagnostics must be a no-op for the update."""
    off_loss, off_grad, _ = _run(False)
    on_loss, on_grad, _ = _run(True)
    torch.testing.assert_close(on_loss, off_loss, rtol=0, atol=0)
    torch.testing.assert_close(on_grad, off_grad, rtol=0, atol=0)


def test_diagnostics_log_all_three_values_on_active_tokens():
    _, _, captured = _run(True)
    for key in ("distill_kl_k1", "distill_kl_k2", "distill_kl_k3"):
        assert key in captured, f"missing diagnostic {key}"

    x = _t(TEACHER_LOGPROBS) - _t(LOGPROBS)
    mask = torch.tensor(LOSS_MASK, dtype=torch.bool)
    expected = {
        "distill_kl_k1": -x,
        "distill_kl_k2": 0.5 * x**2,
        "distill_kl_k3": torch.expm1(x) - x,
    }
    for key, want in expected.items():
        got = captured[key]
        torch.testing.assert_close(
            got[mask], want[mask], rtol=0, atol=1e-12, msg=lambda m: f"{key}: {m}"
        )


def test_diagnostics_absent_by_default():
    _, _, captured = _run(False)
    assert not any(key.startswith("distill_kl_") for key in captured)


def test_diagnostics_off_does_not_require_the_field():
    """An input dict without the flag must behave like diagnostics off."""
    logprobs = _t(LOGPROBS).requires_grad_()
    data = _input_data(diagnostics=False)
    del data["log_distill_diagnostics"]
    tracker = MagicMock()
    with patch("areal.trainer.ppo.actor.stats_tracker", tracker):
        loss = grpo_loss_fn(
            logprobs=logprobs,
            entropy=torch.zeros_like(logprobs),
            input_data=data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
        )
    assert torch.isfinite(loss)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_diagnostics_finite_under_extreme_teacher_gap(dtype):
    """A large positive x must not produce inf in the reported values."""
    logprobs = torch.full((1, 3), -40.0, dtype=dtype, requires_grad=True)
    data = {
        "logprobs": logprobs.detach().clone(),
        "prox_logp": logprobs.detach().clone(),
        "advantages": torch.zeros_like(logprobs),
        "loss_mask": torch.ones_like(logprobs, dtype=torch.bool),
        "teacher_logp": torch.full((1, 3), -1.0, dtype=dtype),
        "rl_loss_weight": 0.0,
        "distill_loss_weight": BETA,
        "log_distill_diagnostics": True,
    }
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
    # x = 39 -> expm1(39) is ~8.7e16, finite in fp32 but not representable in
    # bf16's narrower mantissa; the diagnostic must still be finite there
    # because it is promoted to fp32 before the exponential.
    for key in ("distill_kl_k1", "distill_kl_k2", "distill_kl_k3"):
        assert torch.isfinite(captured[key]).all(), f"{key} not finite"


def test_teacher_config_defaults_are_upstream():
    """Both fields are opt-in; k1 preserves upstream behaviour."""
    cfg = TeacherConfig(rollout=MagicMock())
    assert cfg.log_distill_diagnostics is False
    assert cfg.kl_estimator == "k1"


def test_teacher_config_rejects_non_k1_with_joint_loss():
    """k2/k3 are only defined for sampled pure KD."""
    with pytest.raises(ValueError, match="only supported for sampled pure-KD"):
        TeacherConfig(rollout=MagicMock(), rl_loss_weight=1.0, kl_estimator="k3")


def test_teacher_config_rejects_unknown_estimator():
    with pytest.raises(ValueError, match="must be 'k1', 'k2', or 'k3'"):
        TeacherConfig(rollout=MagicMock(), kl_estimator="k9")

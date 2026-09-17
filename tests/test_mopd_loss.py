# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch

from areal.api.cli_args import MOPDLossConfig, RejectionSamplingConfig
from areal.trainer.mopd.loss import compose_mopd_loss, mopd_loss_fn
from areal.trainer.ppo.actor import PPOActor, grpo_loss_fn


def test_actor_binds_one_mopd_loss_config():
    actor = object.__new__(PPOActor)
    actor._mopd_loss_config = None
    config = MOPDLossConfig(importance_ratio_cap=1.5)

    actor.configure_mopd_loss(config)
    actor.configure_mopd_loss(config)

    with pytest.raises(RuntimeError, match="already bound"):
        actor.configure_mopd_loss(MOPDLossConfig(importance_ratio_cap=2.0))


def _loss_inputs():
    logprobs = torch.tensor(
        [[-0.7, -1.1, -0.4], [-0.2, -0.9, -1.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    old_logprobs = torch.tensor(
        [[-0.8, -1.0, -0.5], [-0.3, -0.8, -1.4]],
        dtype=torch.float64,
    )
    teacher_logp_sum = torch.tensor(
        [[-0.6, -0.9, -0.8], [-0.4, -1.0, -1.1]],
        dtype=torch.float64,
    )
    teacher_weight_sum = torch.tensor(
        [[1.0, 1.0, 2.0], [1.0, 0.5, 1.5]],
        dtype=torch.float64,
    )
    loss_mask = torch.tensor(
        [[True, True, False], [True, False, True]],
    )
    return (
        logprobs,
        old_logprobs,
        teacher_logp_sum,
        teacher_weight_sum,
        loss_mask,
    )


def test_mopd_loss_matches_exact_weighted_reverse_kl_oracle():
    """The score-function surrogate equals an enumerated categorical RKL."""
    student_logits = torch.tensor(
        [0.3, -0.7, 1.1], dtype=torch.float64, requires_grad=True
    )
    teacher_logits = torch.tensor(
        [[-0.2, 0.6, 0.1], [0.9, -0.4, 0.2]], dtype=torch.float64
    )
    teacher_weights = torch.tensor([0.25, 1.75], dtype=torch.float64)
    student_logp = student_logits.log_softmax(dim=0)
    teacher_logp = teacher_logits.log_softmax(dim=-1)
    old_logp = torch.full_like(
        student_logp, -torch.log(torch.tensor(3.0, dtype=torch.float64))
    )
    teacher_logp_sum = (teacher_weights[:, None] * teacher_logp).sum(dim=0)
    teacher_weight_sum = torch.ones_like(student_logp) * teacher_weights.sum()
    loss_mask = torch.ones_like(student_logp, dtype=torch.bool)

    surrogate, _ = mopd_loss_fn(
        student_logp,
        old_logp,
        teacher_logp_sum,
        teacher_weight_sum,
        loss_mask,
    )
    surrogate_grad = torch.autograd.grad(surrogate, student_logits, retain_graph=True)[
        0
    ]
    exact_reverse_kl = (
        teacher_weights[:, None]
        * student_logp.exp()[None, :]
        * (student_logp[None, :] - teacher_logp)
    ).sum()
    exact_grad = torch.autograd.grad(exact_reverse_kl, student_logits)[0]

    torch.testing.assert_close(
        surrogate.detach(), exact_reverse_kl.detach(), rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(surrogate_grad, exact_grad, rtol=1e-12, atol=1e-12)


def test_mopd_loss_masks_extreme_log_ratio_before_exponential():
    """A masked overflowing ratio cannot poison loss, stats, or gradients."""
    logprobs = torch.tensor([-1.0, 1000.0], dtype=torch.float64, requires_grad=True)
    old_logprobs = torch.tensor([-1.0, -1000.0], dtype=torch.float64)
    teacher_logp_sum = torch.tensor([-2.0, -2.0], dtype=torch.float64)
    teacher_weight_sum = torch.ones(2, dtype=torch.float64)
    loss_mask = torch.tensor([True, False])

    loss, stats = mopd_loss_fn(
        logprobs,
        old_logprobs,
        teacher_logp_sum,
        teacher_weight_sum,
        loss_mask,
    )
    loss.backward()

    torch.testing.assert_close(
        loss.detach(), torch.tensor(1.0, dtype=torch.float64), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        logprobs.grad,
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    assert all(torch.isfinite(value).all() for value in stats.values())
    for value in stats.values():
        if value.shape == logprobs.shape:
            torch.testing.assert_close(
                value[~loss_mask],
                torch.zeros_like(value[~loss_mask]),
                rtol=0.0,
                atol=0.0,
            )


def test_mopd_loss_caps_active_extreme_ratio_with_finite_score_gradient():
    """Truncated IS bounds stale tokens without zeroing their policy gradient."""
    logprobs = torch.tensor([1000.0], dtype=torch.float64, requires_grad=True)
    old_logprobs = torch.tensor([-1000.0], dtype=torch.float64)
    teacher_logp_sum = torch.tensor([-2.0], dtype=torch.float64)
    teacher_weight_sum = torch.ones(1, dtype=torch.float64)
    loss_mask = torch.tensor([True])

    loss, stats = mopd_loss_fn(
        logprobs,
        old_logprobs,
        teacher_logp_sum,
        teacher_weight_sum,
        loss_mask,
        importance_ratio_cap=5.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logprobs.grad).all()
    torch.testing.assert_close(
        stats["importance_weight"],
        torch.tensor([5.0], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        logprobs.grad,
        torch.tensor([5010.0], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_mopd_loss_rejects_nonfinite_active_logprobs(invalid):
    """Invalid active policy values fail explicitly instead of poisoning updates."""
    with pytest.raises(RuntimeError, match="must be finite"):
        mopd_loss_fn(
            torch.tensor([invalid], requires_grad=True),
            torch.tensor([-1.0]),
            torch.tensor([-2.0]),
            torch.tensor([1.0]),
            torch.tensor([True]),
        )


def test_mopd_loss_masks_nonfinite_inactive_inputs():
    """Non-finite padding remains harmless after active-token finite checks."""
    logprobs = torch.tensor([-1.0, float("nan")], requires_grad=True)
    loss, stats = mopd_loss_fn(
        logprobs,
        torch.tensor([-1.0, float("inf")]),
        torch.tensor([-2.0, float("nan")]),
        torch.tensor([1.0, float("inf")]),
        torch.tensor([True, False]),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logprobs.grad).all()
    assert all(torch.isfinite(value).all() for value in stats.values())


def test_mopd_loss_detaches_old_policy_and_teacher_targets():
    """Gradients only flow through current-policy importance weights."""
    inputs = list(_loss_inputs())
    for index in (1, 2, 3):
        inputs[index] = inputs[index].requires_grad_()

    loss, _ = mopd_loss_fn(*inputs)
    loss.backward()

    assert inputs[0].grad is not None
    assert inputs[1].grad is None
    assert inputs[2].grad is None
    assert inputs[3].grad is None


def test_mopd_loss_empty_mask_returns_differentiable_zero():
    """An empty response mask is finite and keeps a zero current-policy graph."""
    inputs = list(_loss_inputs())
    inputs[4] = torch.zeros_like(inputs[4])

    loss, stats = mopd_loss_fn(*inputs)
    loss.backward()

    torch.testing.assert_close(
        loss.detach(), torch.zeros_like(loss), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        inputs[0].grad, torch.zeros_like(inputs[0]), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        stats["reverse_kl"],
        torch.zeros_like(inputs[0]),
        rtol=0.0,
        atol=0.0,
    )


def test_compose_mopd_loss_disabled_returns_original_rl_loss():
    """Disabled MOPD returns the identical RL scalar and no MOPD statistics."""
    rl_loss = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)

    total_loss, stats = compose_mopd_loss(rl_loss, config=None)

    assert total_loss is rl_loss
    assert stats == {}


def test_compose_mopd_loss_joint_combines_objectives_and_gradients():
    """Joint mode applies independent RL and MOPD coefficients."""
    inputs = _loss_inputs()
    logprobs = inputs[0]
    rl_loss = logprobs.square().mean()
    config = MOPDLossConfig(
        rl_coefficient=0.4,
        distillation_coefficient=0.6,
    )
    expected_mopd_loss, _ = mopd_loss_fn(*inputs)
    expected_total = 0.4 * rl_loss + 0.6 * expected_mopd_loss
    expected_grad = torch.autograd.grad(expected_total, logprobs, retain_graph=True)[0]

    actual_total, _ = compose_mopd_loss(
        rl_loss,
        config=config,
        logprobs=logprobs,
        old_logprobs=inputs[1],
        teacher_logp_sum=inputs[2],
        teacher_weight_sum=inputs[3],
        loss_mask=inputs[4],
    )
    actual_grad = torch.autograd.grad(actual_total, logprobs)[0]

    torch.testing.assert_close(
        actual_total.detach(), expected_total.detach(), rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)


def test_compose_mopd_loss_pure_mode_ignores_nan_rl_loss():
    """Pure MOPD does not multiply a potentially invalid RL loss by zero."""
    inputs = _loss_inputs()
    rl_loss = torch.tensor(float("nan"), dtype=torch.float64)

    total_loss, _ = compose_mopd_loss(
        rl_loss,
        config=MOPDLossConfig(),
        logprobs=inputs[0],
        old_logprobs=inputs[1],
        teacher_logp_sum=inputs[2],
        teacher_weight_sum=inputs[3],
        loss_mask=inputs[4],
    )

    assert torch.isfinite(total_loss)


def test_mopd_loss_rejects_broadcastable_non_token_shape():
    """Teacher targets must be materialized token tensors, not broadcast views."""
    inputs = list(_loss_inputs())
    inputs[3] = inputs[3][:, :1]

    with pytest.raises(ValueError, match="token shape"):
        mopd_loss_fn(*inputs)


@pytest.mark.parametrize("cap", [0.0, -1.0, float("nan"), float("inf"), True])
def test_mopd_loss_rejects_invalid_importance_ratio_cap(cap):
    """The truncation bound must be finite, positive, and numeric."""
    with pytest.raises(ValueError, match="finite positive"):
        mopd_loss_fn(*_loss_inputs(), importance_ratio_cap=cap)


def test_grpo_loss_fn_composes_materialized_mopd_targets():
    """Pure distillation consumes MOPD targets without an RL advantage tensor."""
    inputs = _loss_inputs()
    logprobs, old_logprobs, teacher_sum, weight_sum, loss_mask = inputs
    proximal_logprobs = torch.zeros_like(old_logprobs)
    input_data = {
        "logprobs": proximal_logprobs,
        "prox_logp": proximal_logprobs,
        "loss_mask": loss_mask,
        "mopd_teacher_logp_sum": teacher_sum,
        "mopd_teacher_weight_sum": weight_sum,
        "mopd_behavior_logprobs": old_logprobs,
    }
    expected_loss, _ = mopd_loss_fn(*inputs, importance_ratio_cap=1.05)

    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()) as tracker:
        actual_loss = grpo_loss_fn(
            logprobs=logprobs,
            entropy=torch.zeros_like(logprobs),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            mopd_loss_config=MOPDLossConfig(importance_ratio_cap=1.05),
        )

    torch.testing.assert_close(
        actual_loss.detach(), expected_loss.detach(), rtol=1e-12, atol=1e-12
    )
    mopd_stat_call = next(
        call
        for call in tracker.stat.call_args_list
        if "mopd_teacher_weight_sum" in call.kwargs
    )
    assert "mopd_loss" in mopd_stat_call.kwargs


def test_grpo_loss_fn_scales_pure_rl_without_teacher_targets():
    """A disabled distillation objective neither requires targets nor drops RL scale."""
    logprobs = torch.tensor([[-0.3, -0.4]], dtype=torch.float64, requires_grad=True)
    input_data = {
        "logprobs": torch.tensor([[-0.5, -0.5]], dtype=torch.float64),
        "prox_logp": torch.tensor([[-0.5, -0.5]], dtype=torch.float64),
        "advantages": torch.ones_like(logprobs),
        "loss_mask": torch.ones_like(logprobs, dtype=torch.bool),
    }
    kwargs = dict(
        logprobs=logprobs,
        entropy=torch.zeros_like(logprobs),
        input_data=input_data,
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
    )

    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()):
        base_loss = grpo_loss_fn(**kwargs)
        scaled_loss = grpo_loss_fn(
            **kwargs,
            mopd_loss_config=MOPDLossConfig(
                rl_coefficient=0.25,
                distillation_coefficient=0.0,
            ),
        )

    torch.testing.assert_close(scaled_loss, 0.25 * base_loss, rtol=1e-12, atol=1e-12)


def test_mopd_loss_respects_m2po_filtered_mask():
    """M2PO removes high-variance tokens from both RL and MOPD objectives."""
    logprobs = torch.tensor([[-0.2, -0.4]], dtype=torch.float64, requires_grad=True)
    old_logprobs = torch.zeros_like(logprobs)
    response_mask = torch.ones_like(logprobs, dtype=torch.bool)
    input_data = {
        "logprobs": old_logprobs,
        "prox_logp": torch.tensor([[2.0, 0.0]], dtype=torch.float64),
        "advantages": torch.zeros_like(logprobs),
        "loss_mask": response_mask,
        "mopd_teacher_logp_sum": torch.tensor([[-1.0, -1.0]], dtype=torch.float64),
        "mopd_teacher_weight_sum": torch.ones_like(logprobs),
        "mopd_behavior_logprobs": old_logprobs,
    }

    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()) as tracker:
        loss = grpo_loss_fn(
            logprobs=logprobs,
            entropy=torch.zeros_like(logprobs),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            m2_threshold=1.0,
            mopd_loss_config=MOPDLossConfig(),
        )
    loss.backward()

    assert logprobs.grad[0, 0] == 0
    assert logprobs.grad[0, 1] != 0
    denominator_call = next(
        call
        for call in tracker.denominator.call_args_list
        if "n_mopd_tokens" in call.kwargs
    )
    assert torch.equal(
        denominator_call.kwargs["n_mopd_tokens"],
        torch.tensor([[False, True]]),
    )
    mopd_stat_call = next(
        call
        for call in tracker.stat.call_args_list
        if "mopd_teacher_weight_sum" in call.kwargs
    )
    assert mopd_stat_call.kwargs["denominator"] == "n_mopd_tokens"


@pytest.mark.parametrize(
    ("level", "shape", "prox_logp", "expected_mask"),
    [
        (
            "token",
            (1, 2),
            [[2.0, 0.0]],
            [[False, True]],
        ),
        (
            "sequence",
            (2, 2),
            [[2.0, 2.0], [0.0, 0.0]],
            [[False, False], [True, True]],
        ),
    ],
)
def test_mopd_loss_respects_behavioral_rejection_without_renormalizing(
    level, shape, prox_logp, expected_mask
):
    """Rejected stale tokens have zero KD gradient and do not amplify survivors."""
    logprobs = torch.full(shape, -0.2, dtype=torch.float64, requires_grad=True)
    old_logprobs = torch.zeros_like(logprobs)
    response_mask = torch.ones_like(logprobs, dtype=torch.bool)
    expected_mask = torch.tensor(expected_mask, dtype=torch.bool)
    input_data = {
        "logprobs": old_logprobs,
        "prox_logp": torch.tensor(prox_logp, dtype=torch.float64),
        "advantages": torch.zeros_like(logprobs),
        "loss_mask": response_mask,
        "mopd_teacher_logp_sum": torch.full_like(logprobs, -1.0),
        "mopd_teacher_weight_sum": torch.ones_like(logprobs),
        "mopd_behavior_logprobs": old_logprobs,
    }

    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()) as tracker:
        loss = grpo_loss_fn(
            logprobs=logprobs,
            entropy=torch.zeros_like(logprobs),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            rejection_sampling=RejectionSamplingConfig(
                level=level,
                action="mask",
                metric="ratio",
                upper=5.0,
            ),
            mopd_loss_config=MOPDLossConfig(),
        )
    loss.backward()

    assert torch.all(logprobs.grad[~expected_mask] == 0)
    assert torch.all(logprobs.grad[expected_mask] != 0)
    expected_loss = (
        torch.exp(logprobs.detach()) * (logprobs.detach() + 1.0) * expected_mask
    ).sum() / response_mask.count_nonzero()
    torch.testing.assert_close(loss.detach(), expected_loss, rtol=1e-12, atol=1e-12)
    denominator_call = next(
        call
        for call in tracker.denominator.call_args_list
        if "n_mopd_tokens" in call.kwargs
    )
    assert torch.equal(denominator_call.kwargs["n_mopd_tokens"], response_mask)

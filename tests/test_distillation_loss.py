# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import torch

from areal.api.cli_args import RejectionSamplingConfig
from areal.trainer.ppo.actor import grpo_loss_fn
from areal.trainer.ppo.loss_reduction import prepare_policy_gradient_batch

DISTILL = 0.5
STUDENT = [[-1.0, -2.0, -0.5, -3.0]]
TEACHER_A = [[-0.1, -4.0, -0.5, -1.0]]
TEACHER_B = [[-5.0, -0.2, -2.0, -0.3]]


def _tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def _input_data(student, teacher, rl_loss_weight, **overrides):
    detached = _tensor(student)
    data = {
        "logprobs": detached.clone(),
        "prox_logp": detached.clone(),
        "advantages": torch.zeros_like(detached),
        "loss_mask": torch.ones_like(detached, dtype=torch.bool),
    }
    if teacher is not None:
        data.update(
            teacher_logp=_tensor(teacher),
            rl_loss_weight=rl_loss_weight,
            distill_loss_weight=DISTILL,
        )
    data.update(overrides)
    return data


def _loss_and_grad(student, input_data, **kwargs):
    logprobs = _tensor(student).requires_grad_()
    with patch("areal.trainer.ppo.actor.stats_tracker", MagicMock()):
        loss = grpo_loss_fn(
            logprobs=logprobs,
            entropy=torch.zeros_like(logprobs),
            input_data=input_data,
            eps_clip=0.2,
            eps_clip_higher=None,
            c_clip=None,
            **kwargs,
        )
    (grad,) = torch.autograd.grad(loss, logprobs)
    return loss.detach(), grad


def test_joint_distillation_gradient_follows_teacher():
    """The joint KD term must move the student toward the teacher, not just down."""
    _, grad_a = _loss_and_grad(STUDENT, _input_data(STUDENT, TEACHER_A, 1.0))
    _, grad_b = _loss_and_grad(STUDENT, _input_data(STUDENT, TEACHER_B, 1.0))

    student, teacher = _tensor(STUDENT), _tensor(TEACHER_A)
    # Zero task advantages isolate d/dlogp mean(0.5 * (logp - logT)^2).
    torch.testing.assert_close(grad_a, DISTILL * (student - teacher) / student.numel())
    assert not torch.allclose(grad_a, grad_b)


def test_joint_distillation_ignores_masked_tokens():
    loss_mask = torch.tensor([[True, False, True, False]])
    input_data = _input_data(STUDENT, TEACHER_A, 1.0, loss_mask=loss_mask)
    _, grad = _loss_and_grad(STUDENT, input_data)

    student, teacher = _tensor(STUDENT), _tensor(TEACHER_A)
    expected = torch.where(loss_mask, DISTILL * (student - teacher) / 2, 0.0)
    torch.testing.assert_close(grad, expected)


def test_pure_distillation_on_policy_gradient_is_reverse_kl_reward():
    """With ratio 1 the update is the REINFORCE gradient of log pi_T - log pi_theta."""
    input_data = _input_data(
        STUDENT, TEACHER_A, 0.0, advantages=_tensor([[9.0, -9.0, 9.0, -9.0]])
    )
    _, grad = _loss_and_grad(STUDENT, input_data)

    student, teacher = _tensor(STUDENT), _tensor(TEACHER_A)
    torch.testing.assert_close(grad, -DISTILL * (teacher - student) / student.numel())


@pytest.mark.parametrize("mode", ["token_mean", "seq_mean", "prompt_mean"])
def test_pure_distillation_matches_policy_loss_with_reward_advantages(mode):
    """Pure KD reuses the configured policy loss and its loss aggregation."""
    student = [[-1.0, -2.0, -0.5, -3.0], [-0.7, -1.5, -2.5, -0.2]]
    teacher = [[-0.1, -4.0, -0.5, -1.0], [-2.0, -0.3, -1.0, -0.9]]
    old = [[-1.1, -1.8, -0.6, -2.9], [-0.6, -1.7, -2.4, -0.4]]
    loss_mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

    def step_kwargs(input_data):
        prepared = prepare_policy_gradient_batch(
            input_data, mode=mode, group_sizes=[2], divisor=None
        )
        return {"loss_step": prepared.for_steps([input_data])[0]}

    distill_data = _input_data(
        student, teacher, 0.0, logprobs=_tensor(old), prox_logp=_tensor(old)
    )
    distill_data["loss_mask"] = loss_mask
    distill_loss, distill_grad = _loss_and_grad(
        student, distill_data, **step_kwargs(distill_data)
    )

    reward = (_tensor(teacher) - _tensor(student)) * loss_mask
    rl_data = _input_data(
        student,
        None,
        None,
        logprobs=_tensor(old),
        prox_logp=_tensor(old),
        advantages=reward,
        loss_mask=loss_mask,
    )
    rl_loss, rl_grad = _loss_and_grad(student, rl_data, **step_kwargs(rl_data))

    torch.testing.assert_close(distill_loss, DISTILL * rl_loss)
    torch.testing.assert_close(distill_grad, DISTILL * rl_grad)


def test_pure_distillation_clips_large_importance_ratio():
    """A token far above its sampling probability must not get an exp(6) push."""
    student = [[-2.0, -1.0]]
    old = [[-8.0, -1.0]]
    input_data = _input_data(
        student,
        [[-0.5, -2.0]],
        0.0,
        logprobs=_tensor(old),
        prox_logp=_tensor(old),
    )
    _, grad = _loss_and_grad(student, input_data)

    assert torch.isfinite(grad).all()
    # Positive reward with ratio e^6 > 1 + eps_clip: the clipped surrogate is flat.
    assert grad[0, 0] == 0
    assert grad[0, 1] == pytest.approx(-DISTILL * (-2.0 - -1.0) / 2)


def test_pure_distillation_applies_rejection_sampling():
    """Tokens whose behavior ratio exceeds the rejection bound are masked."""
    student = [[-2.0, -1.0]]
    input_data = _input_data(
        student,
        [[-0.5, -0.5]],
        0.0,
        logprobs=_tensor([[-8.0, -1.0]]),
        prox_logp=_tensor(student),
    )
    _, grad = _loss_and_grad(
        student,
        input_data,
        use_decoupled_loss=True,
        rejection_sampling=RejectionSamplingConfig(
            level="token", action="mask", metric="ratio", upper=5.0
        ),
    )

    assert grad[0, 0] == 0
    assert grad[0, 1] != 0

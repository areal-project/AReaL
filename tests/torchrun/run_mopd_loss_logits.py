# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math

import torch

from areal.trainer.mopd.loss import mopd_loss_fn
from areal.utils.functional import gather_logprobs


def _test_logits_reverse_kl(device: torch.device) -> None:
    token_logits = torch.tensor(
        [[[0.2, -0.3, 1.1, 0.4], [0.7, -0.5, 0.1, 0.9]]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    labels = torch.tensor([[2, 0]], dtype=torch.long, device=device)
    token_logprobs = gather_logprobs(token_logits, labels)
    expected_token_logprobs = (
        token_logits.log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    )
    torch.testing.assert_close(
        token_logprobs,
        expected_token_logprobs,
        rtol=1e-6,
        atol=1e-6,
    )

    student_logits = torch.tensor(
        [0.3, -0.7, 1.1], dtype=torch.float64, device=device, requires_grad=True
    )
    teacher_logits = torch.tensor(
        [[-0.2, 0.6, 0.1], [0.9, -0.4, 0.2]],
        dtype=torch.float64,
        device=device,
    )
    teacher_weights = torch.tensor([0.25, 1.75], dtype=torch.float64, device=device)
    student_logp = student_logits.log_softmax(dim=0)
    teacher_logp = teacher_logits.log_softmax(dim=-1)
    old_logp = torch.full_like(student_logp, -math.log(3.0))
    teacher_logp_sum = (teacher_weights[:, None] * teacher_logp).sum(dim=0)
    teacher_weight_sum = torch.full_like(student_logp, teacher_weights.sum().item())
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


def _test_masked_overflow(device: torch.device) -> None:
    logprobs = torch.tensor(
        [-1.0, 1000.0], dtype=torch.float32, device=device, requires_grad=True
    )
    old_logprobs = torch.tensor([-1.0, -1000.0], dtype=torch.float32, device=device)
    teacher_logp_sum = torch.tensor([-2.0, -2.0], dtype=torch.float32, device=device)
    teacher_weight_sum = torch.ones(2, dtype=torch.float32, device=device)
    loss_mask = torch.tensor([True, False], device=device)
    loss, stats = mopd_loss_fn(
        logprobs,
        old_logprobs,
        teacher_logp_sum,
        teacher_weight_sum,
        loss_mask,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value).all() for value in stats.values())
    torch.testing.assert_close(
        logprobs.grad,
        torch.tensor([1.0, 0.0], dtype=torch.float32, device=device),
        rtol=0.0,
        atol=0.0,
    )

    active_logprobs = torch.tensor(
        [1000.0], dtype=torch.float32, device=device, requires_grad=True
    )
    active_loss, active_stats = mopd_loss_fn(
        active_logprobs,
        torch.tensor([-1000.0], dtype=torch.float32, device=device),
        torch.tensor([-2.0], dtype=torch.float32, device=device),
        torch.ones(1, dtype=torch.float32, device=device),
        torch.ones(1, dtype=torch.bool, device=device),
        importance_ratio_cap=5.0,
    )
    active_loss.backward()
    assert torch.isfinite(active_loss)
    assert torch.isfinite(active_logprobs.grad).all()
    torch.testing.assert_close(
        active_stats["importance_weight"],
        torch.tensor([5.0], dtype=torch.float32, device=device),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        active_logprobs.grad,
        torch.tensor([5010.0], dtype=torch.float32, device=device),
        rtol=1e-6,
        atol=1e-6,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", choices=("logits_reverse_kl", "masked_overflow"), required=True
    )
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA is unavailable in the GPU worker"
    device = torch.device("cuda:0")
    if args.case == "logits_reverse_kl":
        _test_logits_reverse_kl(device)
    else:
        _test_masked_overflow(device)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    print(f"Passed case={args.case}")


if __name__ == "__main__":
    main()

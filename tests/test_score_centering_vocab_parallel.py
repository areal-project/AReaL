# SPDX-License-Identifier: Apache-2.0
"""Real process-group checks against an independent dense SC oracle."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from areal.utils.score_centering import (
    score_centering_logprobs,
    score_centering_tis_weights,
)


def _dense(logits, labels, ids, q, tis_cap=None):
    logp = logits.log_softmax(-1)
    heads = logp.gather(-1, ids)
    with torch.no_grad():
        p = heads.exp()
        rho = (1 - q.exp().sum(-1)).clamp_min(0) / (1 - p.sum(-1)).clamp_min(1e-6)
        coefficients = q.exp() - rho[:, None] * p
        if tis_cap is not None:
            coefficients = (
                torch.minimum(p, tis_cap * q.exp())
                - (rho * tis_cap).clamp_max(1)[:, None] * p
            )
    return logp.gather(-1, labels[:, None]).squeeze(-1), (heads * coefficients).sum(-1)


def _worker(rank, world, rendezvous, device_type):
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    device = (
        torch.device(device_type, rank)
        if device_type == "cuda"
        else torch.device("cpu")
    )
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        # Use explicit subgroups, including groups whose local rank != global rank.
        for tp in [x for x in (1, 2, 4) if x <= world]:
            groups = [
                dist.new_group(list(range(i, i + tp))) for i in range(0, world, tp)
            ]
            group = groups[rank // tp]
            shard_rank = dist.get_rank(group)
            for dtype in (torch.float64, torch.float32, torch.bfloat16):
                for chunk in (1, 4, 32):
                    torch.manual_seed(901)
                    n, hidden, padded, vocab = 7, 9, 24, 19
                    h = torch.randn(n, hidden, device=device, dtype=dtype)
                    w = torch.randn(hidden, padded, device=device, dtype=dtype) * 0.2
                    labels = torch.tensor([0, 6, 12, 18, 1, 7, 13], device=device)
                    ids = torch.tensor([[0, 6, 12, 18]] * n, device=device)
                    # Duplicate label/head selection must accumulate gradients.
                    q = torch.full((n, 4), 0.08, device=device).log()
                    advantages = torch.linspace(-1, 1, n, device=device)
                    width = padded // tp
                    sl = slice(shard_rank * width, (shard_rank + 1) * width)
                    local_h = h.clone().requires_grad_()
                    local_w = w[:, sl].clone().requires_grad_()
                    local_logits = local_h @ local_w
                    local_logits.retain_grad()
                    # BF16 matmul reduction choices vary with shard shape. Compare
                    # logits-gradient math using the exact distributed forward logits.
                    gathered = [torch.empty_like(local_logits) for _ in range(tp)]
                    dist.all_gather(gathered, local_logits.detach(), group=group)
                    oracle = (
                        torch.cat(gathered, -1)
                        .detach()
                        .to(torch.float64 if dtype == torch.float64 else torch.float32)
                        .requires_grad_()
                    )
                    tis_cap = {1: None, 4: 2.0, 32: 0.5}[chunk]
                    lp, entropy, correction = score_centering_logprobs(
                        local_logits,
                        labels,
                        ids,
                        q,
                        chunk,
                        tp_group=group,
                        vocab_size=vocab,
                        tis_cap=tis_cap,
                    )
                    expected_lp, expected_c = _dense(
                        oracle[:, :vocab], labels, ids, q, tis_cap=tis_cap
                    )
                    sampled_q = torch.full_like(lp, -3.7)
                    weight = (
                        1.0
                        if tis_cap is None
                        else score_centering_tis_weights(lp, sampled_q, tis_cap)
                    )
                    expected_weight = (
                        1.0
                        if tis_cap is None
                        else (expected_lp.detach() - sampled_q).exp().clamp_max(tis_cap)
                    )
                    ((weight * lp - correction) * advantages).sum().backward()
                    (
                        (expected_weight * expected_lp - expected_c) * advantages
                    ).sum().backward()
                    tol = 1e-10 if dtype == torch.float64 else 2e-6
                    torch.testing.assert_close(lp, expected_lp, atol=tol, rtol=tol)
                    torch.testing.assert_close(
                        correction, expected_c, atol=tol, rtol=tol
                    )
                    expected_entropy = -(
                        oracle[:, :vocab].softmax(-1)
                        * oracle[:, :vocab].log_softmax(-1)
                    ).sum(-1)
                    torch.testing.assert_close(
                        entropy, expected_entropy, atol=tol, rtol=tol
                    )
                    assert not entropy.requires_grad
                    grad_tol = 0.004 if dtype == torch.bfloat16 else tol
                    torch.testing.assert_close(
                        local_logits.grad,
                        oracle.grad[:, sl].to(dtype),
                        atol=grad_tol,
                        rtol=grad_tol,
                    )
                    # Projection adjoints, including the TP hidden-state SUM.
                    dist.all_reduce(local_h.grad, group=group)
                    oracle_grad = oracle.grad.to(dtype)
                    expected_h_grad = oracle_grad @ w.T
                    expected_w_grad = h.T @ oracle_grad
                    linear_tol = (
                        0.02
                        if dtype == torch.bfloat16
                        else 5e-6
                        if dtype == torch.float32
                        else 1e-10
                    )
                    torch.testing.assert_close(
                        local_h.grad, expected_h_grad, atol=linear_tol, rtol=linear_tol
                    )
                    torch.testing.assert_close(
                        local_w.grad,
                        expected_w_grad[:, sl],
                        atol=linear_tol,
                        rtol=linear_tol,
                    )
                    # Check an optimizer step (FP32 parameters for BF16 updates).
                    updated = local_w.detach().float() - 0.01 * local_w.grad.float()
                    expected = w[:, sl].float() - 0.01 * expected_w_grad[:, sl].float()
                    torch.testing.assert_close(
                        updated, expected, atol=linear_tol * 0.01, rtol=linear_tol
                    )
                    assert torch.count_nonzero(oracle.grad[:, vocab:]) == 0
                    # A no-grad recomputation exercises the evaluation path too.
                    with torch.no_grad():
                        eval_lp, _, _ = score_centering_logprobs(
                            local_logits,
                            labels,
                            ids,
                            q,
                            chunk,
                            tp_group=group,
                            vocab_size=vocab,
                            tis_cap=tis_cap,
                        )
                    torch.testing.assert_close(eval_lp, lp, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.parametrize("world", [1, 2, 4])
def test_vocabulary_parallel_score_centering_dense_parity(tmp_path, world):
    mp.spawn(
        _worker,
        args=(world, str(tmp_path / "rendezvous"), "cpu"),
        nprocs=world,
        join=True,
    )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_vocabulary_parallel_score_centering_nccl(tmp_path):
    world = min(4, torch.cuda.device_count())
    mp.spawn(
        _worker,
        args=(world, str(tmp_path / "rendezvous"), "cuda"),
        nprocs=world,
        join=True,
    )

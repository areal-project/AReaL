# SPDX-License-Identifier: Apache-2.0
"""Paper Eq. 11-14: center the TIS-weighted score, not the raw score."""

from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import PPOActorConfig, RejectionSamplingConfig
from areal.utils.score_centering import (
    score_centering_correction,
    score_centering_logprobs,
    score_centering_tis_weights,
)


@pytest.mark.parametrize("cap", [0.5, 2.0, 10.0])
@pytest.mark.parametrize("full_head", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tis_sc_matches_full_vocabulary_gradient_and_adam(cap, full_head, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(93)
    h = torch.randn(5, 4, device=device, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, 11, device=device, dtype=torch.float64, requires_grad=True)
    h2, w2 = h.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    ids = torch.arange(11 if full_head else 4, device=device).expand(5, -1)
    # Mix sampled tokens inside and outside the collected head.
    labels = torch.tensor([0, 1, 7, 8, 10], device=device)
    q = torch.randn(5, 11, device=device, dtype=torch.float64).softmax(-1)
    sampled_q = q.gather(-1, labels[:, None]).squeeze(-1).log()
    advantages = torch.tensor([-2.0, -1.0, 0.0, 1.0, 3.0], device=device)
    lp, _, c = score_centering_logprobs(
        h @ w, labels, ids, q.gather(-1, ids).log(), 2, tis_cap=cap
    )
    weight = score_centering_tis_weights(lp, sampled_q, cap)
    ((c - weight * lp) * advantages).mean().backward()

    dense_lp = (h2 @ w2).log_softmax(-1)
    with torch.no_grad():
        p = dense_lp.exp()
        q_hat = q.clone()
        if not full_head:
            rho = (1 - q[:, :4].sum(-1)) / (1 - p[:, :4].sum(-1))
            q_hat[:, 4:] = rho[:, None] * p[:, 4:]
        dense_weights = (p / q_hat).clamp_max(cap)
        sampled_weights = (
            p.gather(-1, labels[:, None]).squeeze(-1) / sampled_q.exp()
        ).clamp_max(cap)
    dense_c = (q_hat * dense_weights * dense_lp).sum(-1)
    dense_sample = dense_lp.gather(-1, labels[:, None]).squeeze(-1)
    ((dense_c - sampled_weights * dense_sample) * advantages).mean().backward()
    for actual, expected in [(h.grad, h2.grad), (w.grad, w2.grad)]:
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    torch.optim.Adam([h, w], lr=1e-3).step()
    torch.optim.Adam([h2, w2], lr=1e-3).step()
    torch.testing.assert_close(h, h2, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(w, w2, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("cap", [0.5, 2.0, 100.0])
@pytest.mark.parametrize("full_head", [False, True])
def test_tis_sc_constant_reward_has_zero_expected_gradient(cap, full_head):
    x = torch.tensor(
        [2.0, -1.0, 0.5, 1.0, -2.0], dtype=torch.float64, requires_grad=True
    )
    p = x.detach().softmax(-1)
    q = torch.tensor([0.1, 0.3, 0.2, 0.15, 0.25], dtype=torch.float64)
    width = 5 if full_head else 2
    if not full_head:
        q[width:] = (1 - q[:width].sum()) * p[width:] / p[width:].sum()
    ids = torch.arange(width).expand(5, -1)
    lp, _, c = score_centering_logprobs(
        x.expand(5, -1),
        torch.arange(5),
        ids,
        q[:width].log().expand(5, -1),
        tis_cap=cap,
    )
    weight = score_centering_tis_weights(lp, q.log(), cap)
    loss = (q * (c - weight * lp)).sum()
    grad = torch.autograd.grad(loss, x, retain_graph=True)[0]
    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-12, rtol=0)
    if cap == 2.0:
        # Multiplying the old SC surrogate by the sampled weight is not equivalent.
        plain_c = score_centering_correction(
            x.log_softmax(-1)[:width].expand(5, -1), q[:width].log().expand(5, -1)
        )
        wrong = (q * weight * (plain_c - lp)).sum()
        assert torch.autograd.grad(wrong, x)[0].norm() > 1e-3


@pytest.mark.parametrize("cap", [0.5, 2.0])
@pytest.mark.parametrize(
    "case", ["zero_tail", "zero_head", "tiny_train_tail", "matched"]
)
def test_tis_sc_probability_edge_cases(case, cap):
    p = torch.tensor([0.4, 0.35, 0.2, 0.05], dtype=torch.float64)
    q = torch.tensor([0.6, 0.1], dtype=torch.float64)
    if case == "zero_tail":
        q = torch.tensor([0.75, 0.25], dtype=torch.float64)
    elif case == "zero_head":
        q[0] = 0
    elif case == "tiny_train_tail":
        p = torch.tensor([0.6, 0.4 - 1e-8, 5e-9, 5e-9], dtype=torch.float64)
    elif case == "matched":
        q = p[:2].clone()
    x = p.log().requires_grad_()
    lp = x.log_softmax(-1)
    c = score_centering_correction(lp[:2], q.log(), tis_cap=cap)
    c.backward()
    assert torch.isfinite(c) and torch.isfinite(x.grad).all()
    with torch.no_grad():
        rho = (1 - q.sum()).clamp_min(0) / (1 - p[:2].sum()).clamp_min(1e-6)
        weighted_mass = torch.minimum(p[:2], cap * q)
        coefficients = weighted_mass - (cap * rho).clamp_max(1) * p[:2]
        expected = -coefficients.sum() * p
        expected[:2] += coefficients
    torch.testing.assert_close(x.grad, expected, atol=1e-10, rtol=1e-8)
    if case == "matched":
        torch.testing.assert_close(x.grad, torch.zeros_like(x.grad), atol=1e-12, rtol=0)


@pytest.mark.parametrize("cap", [float("inf"), float("nan"), 1e100])
def test_tis_sc_rejects_invalid_upper(cap):
    with pytest.raises(ValueError, match="TIS upper"):
        PPOActorConfig(
            backend="fsdp:d1",
            use_score_centering=True,
            recompute_logprob=False,
            rejection_sampling=RejectionSamplingConfig(action="clamp", upper=cap),
        )


@pytest.mark.parametrize(
    "options",
    [
        dict(action="mask"),
        dict(action="clamp", level="sequence"),
        dict(action="clamp", lower=0.5),
        dict(action="mask", metric="kl_k2"),
    ],
)
def test_tis_sc_rejects_unsupported_rejection_modes(options):
    with pytest.raises(ValueError, match="supports rejection_sampling only"):
        PPOActorConfig(
            backend="fsdp:d1",
            use_score_centering=True,
            recompute_logprob=False,
            rejection_sampling=RejectionSamplingConfig(**options),
        )


@pytest.mark.parametrize("backend", ["fsdp:d1", "fsdp:d1t2c2"])
def test_tis_sc_accepts_existing_config_without_new_yaml_fields(backend):
    from dataclasses import fields

    from areal.utils.score_centering import resolve_score_centering_tis_cap

    config = PPOActorConfig(
        backend=backend,
        use_score_centering=True,
        recompute_logprob=False,
        rejection_sampling=RejectionSamplingConfig(action="clamp", upper=2.0),
    )
    assert resolve_score_centering_tis_cap(config.rejection_sampling) == 2.0
    assert resolve_score_centering_tis_cap(None) is None
    assert (
        resolve_score_centering_tis_cap(RejectionSamplingConfig(action="clamp")) == 5.0
    )
    assert "score_centering_tis_cap" not in {f.name for f in fields(PPOActorConfig)}


def test_tis_weights_detached_and_finite_for_extreme_ratios():
    lp = torch.tensor([0.0, -1000.0], requires_grad=True)
    q = torch.tensor([-1000.0, 0.0], requires_grad=True)
    weights = score_centering_tis_weights(lp, q, 2.0)
    torch.testing.assert_close(weights, torch.tensor([2.0, 0.0]), atol=0, rtol=0)
    assert not weights.requires_grad


@pytest.mark.parametrize("mode", ["token", "sequence"])
def test_actor_tis_uses_sampler_logprobs_and_weights_only_sampled_term(
    mode, monkeypatch
):
    from areal.trainer.ppo.actor import grpo_loss_fn
    from areal.utils.functional.loss_aggregation import SequenceMean, TokenMean
    from areal.utils.stats_tracker import DistributedStatsTracker

    monkeypatch.setattr(
        "areal.trainer.ppo.actor.stats_tracker", DistributedStatsTracker()
    )
    lp = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], requires_grad=True)
    old = torch.tensor([[-4.0, -1.0, -3.0], [-6.0, -5.0, -6.0]])
    # Intentionally different proximal policy; TIS must ignore it.
    prox = lp.detach() + 1
    mask = torch.tensor([[True, True, True], [True, False, False]])
    adv = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 0.0, 0.0]])
    correction = lp * 0.3
    reduction = TokenMean(mask) if mode == "token" else SequenceMean(mask)
    actual = grpo_loss_fn(
        lp,
        torch.zeros_like(lp),
        dict(logprobs=old, prox_logp=prox, advantages=adv, loss_mask=mask),
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        use_score_centering=True,
        sc_correction=correction,
        rejection_sampling=RejectionSamplingConfig(action="clamp", upper=2.0),
        loss_step=SimpleNamespace(bind=lambda _: reduction),
    )
    expected_weight = (lp.detach() - old).exp().clamp_max(2.0)
    expected = reduction.aggregate(adv * (correction - expected_weight * lp), mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        torch.autograd.grad(actual, lp, retain_graph=True)[0],
        torch.autograd.grad(expected, lp)[0],
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("lower", [None, 0.0])
def test_tis_existing_config_roundtrip_preserves_zero_lower(lower):
    from dataclasses import asdict

    from areal.utils.score_centering import resolve_score_centering_tis_cap

    rs = RejectionSamplingConfig(action="clamp", upper=2.0, lower=lower)
    restored = RejectionSamplingConfig(**asdict(rs))
    assert resolve_score_centering_tis_cap(restored) == 2.0
    assert restored.lower == 0.0


def test_non_sc_tis_config_remains_decoupled_ppo():
    config = PPOActorConfig(
        backend="fsdp:d1",
        use_decoupled_loss=True,
        rejection_sampling=RejectionSamplingConfig(action="clamp", upper=2.0),
    )
    assert not config.use_score_centering
    assert config.rejection_sampling.upper == 2.0


def test_sc_tis_does_not_warn_that_correction_is_ignored(monkeypatch):
    from unittest.mock import Mock

    import areal.api.cli_args as cli_args

    warning = Mock()
    monkeypatch.setattr(cli_args.logger, "warning", warning)
    PPOActorConfig(
        backend="fsdp:d1",
        use_score_centering=True,
        recompute_logprob=False,
        rejection_sampling=RejectionSamplingConfig(action="clamp", upper=2.0),
    )
    assert not any(
        "Filtering will be ignored" in str(call) for call in warning.call_args_list
    )
    PPOActorConfig(
        backend="fsdp:d1",
        use_score_centering=False,
        use_decoupled_loss=False,
        rejection_sampling=RejectionSamplingConfig(action="clamp", upper=2.0),
    )
    assert any(
        "Filtering will be ignored" in str(call) for call in warning.call_args_list
    )

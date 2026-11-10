# SPDX-License-Identifier: Apache-2.0
"""Head weighting for score centering: the none/tis/mis bands and the pad mask."""

import math

import pytest
import torch

from areal.api.cli_args import PPOActorConfig, RejectionSamplingConfig
from areal.utils.score_centering import (
    build_weight_fn,
    sampled_weight,
    score_centering_correction,
    score_centering_logprobs,
    score_centering_masses,
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
    weight_fn = build_weight_fn("tis", 0.0, cap)
    lp, _, c = score_centering_logprobs(
        h @ w, labels, ids, q.gather(-1, ids).log(), 2, weight_fn=weight_fn
    )
    weight = sampled_weight(lp, sampled_q, weight_fn)
    ((c - weight * lp) * advantages).mean().backward()

    dense_lp = (h2 @ w2).log_softmax(-1)
    with torch.no_grad():
        p = dense_lp.exp()
        q_hat = q.clone()
        if not full_head:
            numerator = (1 - q[:, :4].sum(-1)).clamp_min(1e-6)
            denominator = (1 - p[:, :4].sum(-1)).clamp_min(1e-6)
            q_hat[:, 4:] = (numerator / denominator)[:, None] * p[:, 4:]
        dense_weights = (p / q_hat).clamp(0.0, cap)
        sampled_weights = (
            p.gather(-1, labels[:, None]).squeeze(-1) / sampled_q.exp()
        ).clamp(0.0, cap)
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
    weight_fn = build_weight_fn("tis", 0.0, cap)
    lp, _, c = score_centering_logprobs(
        x.expand(5, -1),
        torch.arange(5),
        ids,
        q[:width].log().expand(5, -1),
        weight_fn=weight_fn,
    )
    weight = sampled_weight(lp, q.log(), weight_fn)
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
    sampler_head_logp = q.log()
    c = score_centering_correction(
        lp[:2], sampler_head_logp, weight_fn=build_weight_fn("tis", 0.0, cap)
    )
    c.backward()
    assert torch.isfinite(c) and torch.isfinite(x.grad).all()
    with torch.no_grad():
        # A -inf sampler slot is masked out of both sides.
        valid = torch.isfinite(sampler_head_logp)
        p_head = p[:2] * valid
        q_head = q * valid
        rho = (1 - q_head.sum()).clamp_min(1e-6) / (1 - p_head.sum()).clamp_min(1e-6)
        ratio = p_head / q_head.clamp_min(torch.finfo(p.dtype).tiny)
        ratio = torch.where(valid, ratio.clamp(0.0, cap), torch.zeros_like(ratio))
        alpha = rho * (1.0 / rho).clamp(0.0, cap)
        coefficients = q_head * ratio - alpha * p_head
        expected = -coefficients.sum() * p
        expected[:2] += coefficients
    torch.testing.assert_close(x.grad, expected, atol=1e-10, rtol=1e-8)
    if case == "matched":
        torch.testing.assert_close(x.grad, torch.zeros_like(x.grad), atol=1e-12, rtol=0)


def test_mis_zeroes_out_of_band_ratios():
    """MIS keeps the ratio inside [low, high] and drops it outside."""
    fn = build_weight_fn("mis", 0.5, 2.0)
    r = torch.tensor([0.1, 0.3, 0.5, 1.0, 2.0, 2.5, 10.0], dtype=torch.float64)
    torch.testing.assert_close(
        fn(r), torch.tensor([0.0, 0.0, 0.5, 1.0, 2.0, 0.0, 0.0], dtype=torch.float64)
    )
    # The same band drops sampled tokens whose ratio leaves it.
    train = torch.tensor([math.log(0.3), math.log(1.5)], dtype=torch.float64)
    sampler = torch.zeros(2, dtype=torch.float64)
    torch.testing.assert_close(
        sampled_weight(train, sampler, fn),
        torch.tensor([0.0, 1.5], dtype=torch.float64),
    )
    # 'none' is pure centering: every token keeps weight one.
    torch.testing.assert_close(build_weight_fn("none", 0.5, 2.0)(r), torch.ones_like(r))
    with pytest.raises(ValueError, match="Weight mode"):
        build_weight_fn("bogus", 0.5, 2.0)


def test_two_sided_tis_floors_the_low_side():
    """tis clamps both sides of the ratio band, not only the top."""
    r = torch.tensor([0.3, 0.5, 1.0, 5.0, 9.0], dtype=torch.float64)
    fn = build_weight_fn("tis", 0.5, 5.0)
    torch.testing.assert_close(
        fn(r), torch.tensor([0.5, 0.5, 1.0, 5.0, 5.0], dtype=torch.float64)
    )
    # A zero lower bound leaves small ratios untouched.
    torch.testing.assert_close(
        build_weight_fn("tis", 0.0, 5.0)(r),
        torch.tensor([0.3, 0.5, 1.0, 5.0, 5.0], dtype=torch.float64),
    )
    # The sampled-token weight obeys the same band.
    train = torch.full((5,), math.log(0.3), dtype=torch.float64)
    sampler = torch.zeros(5, dtype=torch.float64)
    torch.testing.assert_close(
        sampled_weight(train, sampler, fn), torch.full((5,), 0.5, dtype=torch.float64)
    )
    # Both sides of the band show up in the head residual.
    head = torch.tensor([[-0.1, -9.0]], dtype=torch.float64)
    sampler_head = torch.tensor([[-5.0, -2.0]], dtype=torch.float64)
    got = score_centering_correction(head, sampler_head, weight_fn=fn)
    with torch.no_grad():
        p = head.exp()
        q = sampler_head.exp()
        rho = (1 - q.sum()).clamp_min(1e-6) / (1 - p.sum()).clamp_min(1e-6)
        ratio = (p / q).clamp(0.5, 5.0)
        alpha = rho * (1.0 / rho).clamp(0.5, 5.0)
        residual = q * ratio - alpha * p
        expected = (residual * head).sum(-1)
    torch.testing.assert_close(got, expected)


def test_eps_floors_both_tail_masses():
    """Both tails are floored, so a degenerate sampler and trainer stay finite."""
    train = torch.tensor([[math.log1p(-2e-9)]], dtype=torch.float64, requires_grad=True)
    sampler = torch.tensor([[math.log1p(-1e-9)]], dtype=torch.float64)
    rho, q_mass, p_mass = score_centering_masses(train, sampler)
    assert torch.isfinite(rho).all()
    # Both masses sit within eps of one, so the ratio is exactly eps/eps = 1.
    torch.testing.assert_close(rho, torch.ones_like(rho))
    torch.testing.assert_close(q_mass, torch.tensor([1.0 - 1e-9], dtype=torch.float64))
    torch.testing.assert_close(p_mass, torch.tensor([1.0 - 2e-9], dtype=torch.float64))
    # Flooring only the denominator (the old form) reports rho ~ 1e-3 instead.
    unfloored = (1 - q_mass).clamp_min(0) / (1 - p_mass).clamp_min(1e-6)
    torch.testing.assert_close(
        unfloored, torch.full_like(unfloored, 1e-3), rtol=1e-6, atol=0
    )
    assert (rho > 1e2 * unfloored).all()
    correction = score_centering_correction(train, sampler)
    correction.sum().backward()
    assert torch.isfinite(correction).all() and torch.isfinite(train.grad).all()


def test_padded_sampler_slot_carries_no_residual():
    """A -inf sampler slot is masked out of q and p, so it cannot leak score."""
    train = torch.tensor([[-0.5, -0.3]], dtype=torch.float64, requires_grad=True)
    sampler = torch.tensor([[-1.0, -torch.inf]], dtype=torch.float64)
    correction = score_centering_correction(train, sampler)
    correction.sum().backward()
    assert train.grad[0, 1] == 0
    # The row equals the same head with the padded slot removed entirely.
    one_slot = score_centering_correction(train.detach()[:, :1], sampler[:, :1])
    torch.testing.assert_close(correction, one_slot)
    # Before masking, the trainer probability of the padded slot leaked in.
    with torch.no_grad():
        p_all = train.detach().exp()
        old_rho = (1 - sampler.exp().sum(-1)).clamp_min(0) / (1 - p_all.sum(-1))
        old = ((sampler.exp() - old_rho[:, None] * p_all) * train.detach()).sum(-1)
    assert not torch.allclose(correction, old)
    # The logged masses ignore the padded slot as well.
    rho, q_mass, p_mass = score_centering_masses(train.detach(), sampler)
    torch.testing.assert_close(
        q_mass, torch.tensor([math.exp(-1.0)], dtype=torch.float64)
    )
    torch.testing.assert_close(
        p_mass, torch.tensor([math.exp(-0.5)], dtype=torch.float64)
    )
    assert torch.isfinite(rho).all()


def test_sampled_weight_and_correction_share_one_weight_fn():
    """Both entry points apply the caller's f, so one config drives both."""
    train = torch.tensor([[-0.2, -1.5, -2.5]], dtype=torch.float64)
    sampler = torch.tensor([[-0.4, -1.2, -3.0]], dtype=torch.float64)

    def scale(r):
        return 2.0 * r

    weight = sampled_weight(train[:, :1], sampler[:, :1], scale)
    assert not weight.requires_grad
    torch.testing.assert_close(weight, 2.0 * (train[:, :1] - sampler[:, :1]).exp())
    # The correction applies the same f to the head ratio and to the tail ratio.
    got = score_centering_correction(train, sampler, weight_fn=scale)
    with torch.no_grad():
        p = train.exp()
        q = sampler.exp()
        rho = (1 - q.sum(-1)).clamp_min(1e-6) / (1 - p.sum(-1)).clamp_min(1e-6)
        residual = q * scale(p / q) - (rho * scale(1.0 / rho))[:, None] * p
        expected = (residual * train).sum(-1)
    torch.testing.assert_close(got, expected)
    # No f at all and the identity f agree bit for bit.
    torch.testing.assert_close(
        score_centering_correction(train, sampler),
        score_centering_correction(
            train, sampler, weight_fn=build_weight_fn("none", 0.5, 2.0)
        ),
        atol=0,
        rtol=0,
    )


def test_sampled_weight_is_detached_and_bounded_for_extreme_ratios():
    train = torch.tensor([0.0, -1000.0], requires_grad=True)
    sampler = torch.tensor([-1000.0, 0.0], requires_grad=True)
    weights = sampled_weight(train, sampler, build_weight_fn("tis", 0.0, 2.0))
    torch.testing.assert_close(weights, torch.tensor([2.0, 0.0]), atol=0, rtol=0)
    assert not weights.requires_grad
    # A finite band never sees an infinite ratio: out-of-band values are dropped.
    torch.testing.assert_close(
        sampled_weight(train, sampler, build_weight_fn("mis", 0.5, 2.0)),
        torch.zeros(2),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_sc_weight_config_accepts_every_mode(mode):
    config = PPOActorConfig(
        backend="fsdp:d1",
        use_score_centering=True,
        recompute_logprob=False,
        sc_mode=mode,
    )
    assert config.sc_mode == mode
    assert config.sc_clip_low == 0.5
    assert config.sc_clip_high == 5.0


@pytest.mark.parametrize(
    "overrides",
    [
        dict(sc_mode="bogus"),
        dict(sc_mode="tis", sc_clip_low=-0.1),
        dict(sc_mode="tis", sc_clip_high=float("inf")),
        dict(sc_mode="tis", sc_clip_high=float("nan")),
        dict(sc_mode="tis", sc_clip_low=2.0, sc_clip_high=2.0),
        dict(sc_mode="mis", sc_clip_low=-1.0),
        dict(sc_mode="mis", sc_clip_high=0.0),
        dict(sc_mode="mis", sc_clip_low=2.0, sc_clip_high=1.0),
    ],
)
def test_sc_weight_config_rejects_invalid_bands(overrides):
    with pytest.raises(ValueError, match="sc_mode"):
        PPOActorConfig(
            backend="fsdp:d1",
            use_score_centering=True,
            recompute_logprob=False,
            **overrides,
        )


def test_sc_owns_its_weighting_stages():
    """SC no longer reads rejection_sampling or importance_sampling."""
    rs = RejectionSamplingConfig(action="clamp", upper=2.0)
    with pytest.raises(ValueError, match="sc_mode/sc_clip_low/sc_clip_high"):
        PPOActorConfig(
            backend="fsdp:d1", use_score_centering=True, rejection_sampling=rs
        )
    with pytest.raises(ValueError, match="sc_mode/sc_clip_low/sc_clip_high"):
        PPOActorConfig(
            backend="fsdp:d1",
            use_score_centering=True,
            rejection_sampling=rs,
            importance_sampling=rs,
        )


def test_non_sc_config_keeps_rollout_correction():
    """Without SC, rejection_sampling still drives decoupled PPO."""
    config = PPOActorConfig(
        backend="fsdp:d1",
        use_decoupled_loss=True,
        rejection_sampling=RejectionSamplingConfig(action="clamp", upper=2.0),
    )
    assert not config.use_score_centering
    assert config.sc_mode == "none"
    assert config.rejection_sampling.upper == 2.0


def test_band_mask_marks_ratios_outside_the_clip_band():
    """The clamped fraction is the share of ratios outside [low, high]."""
    from areal.utils.score_centering import band_mask

    ratio = torch.tensor([0.2, 0.5, 1.0, 2.0, 3.0])
    inside = band_mask(ratio, 0.5, 2.0)
    assert inside.tolist() == [False, True, True, True, False]
    assert (~inside).float().mean().item() == pytest.approx(0.4)

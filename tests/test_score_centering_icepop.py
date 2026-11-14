# SPDX-License-Identifier: Apache-2.0
"""Independent weighted-score oracles for score centering composed with IcePop."""

from types import SimpleNamespace

import pytest
import torch

from areal.api.cli_args import PPOActorConfig, RejectionSamplingConfig
from areal.utils.score_centering import (
    resolve_score_centering_rejection,
    resolve_score_centering_tis_cap,
    score_centering_correction,
    score_centering_icepop_weights,
    score_centering_logprobs,
)


def dense_weight(p, q, bounds):
    """Probability-space reference, independent of the production log-space mask."""
    ratio = p / q
    return torch.where((ratio >= bounds[0]) & (ratio <= bounds[1]), ratio, 0.0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("full_head", [False, True])
@pytest.mark.parametrize("bounds", [(0.5, 5.0), (0.8, 1.25), (0.0, 2.0)])
def test_icepop_sc_matches_dense_hidden_weight_gradients_and_adam(
    device, full_head, bounds
):
    """Compare hidden/output-weight gradients and Adam with a dense expectation."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(19)
    h = torch.randn(7, 5, dtype=torch.float64, device=device, requires_grad=True)
    w = torch.randn(5, 13, dtype=torch.float64, device=device, requires_grad=True)
    h2, w2 = h.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    q = torch.randn(7, 13, dtype=torch.float64, device=device).softmax(-1)
    ids = q.topk(13 if full_head else 5, dim=-1).indices
    labels = torch.arange(7, device=device)
    advantage = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, -3.0, 3.0], device=device)
    q_sample = q.gather(-1, labels[:, None]).squeeze(-1)
    lp, _, correction = score_centering_logprobs(
        h @ w, labels, ids, q.gather(-1, ids).log(), 3, icepop_bounds=bounds
    )
    weight = score_centering_icepop_weights(lp, q_sample.log(), bounds)
    ((correction - weight * lp) * advantage).mean().backward()
    dense_lp = (h2 @ w2).log_softmax(-1)
    with torch.no_grad():
        p = dense_lp.exp()
        q_hat = q.clone()
        if not full_head:
            rho = (1 - q.gather(-1, ids).sum(-1)) / (1 - p.gather(-1, ids).sum(-1))
            q_hat = rho[:, None] * p
            q_hat.scatter_(-1, ids, q.gather(-1, ids))
        mass = q_hat * dense_weight(p, q_hat, bounds)
        sample_weight = dense_weight(
            p.gather(-1, labels[:, None]).squeeze(-1), q_sample, bounds
        )
    c2 = (mass * dense_lp).sum(-1)
    sampled_lp = dense_lp.gather(-1, labels[:, None]).squeeze(-1)
    ((c2 - sample_weight * sampled_lp) * advantage).mean().backward()
    for actual, expected in [(h.grad, h2.grad), (w.grad, w2.grad)]:
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    torch.optim.Adam([h, w], lr=1e-3).step()
    torch.optim.Adam([h2, w2], lr=1e-3).step()
    torch.testing.assert_close(h, h2, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(w, w2, atol=1e-11, rtol=1e-10)


@pytest.mark.parametrize("width", [2, 5])
@pytest.mark.parametrize("bounds", [(0.5, 5.0), (0.8, 1.25), (0.0, 2.0)])
def test_icepop_sc_constant_reward_has_zero_expected_gradient(width, bounds):
    """Enumerating all actions cancels drift for exact or modeled-tail q."""
    x = torch.tensor(
        [2.0, -1.0, 0.5, 1.0, -2.0], dtype=torch.float64, requires_grad=True
    )
    p = x.detach().softmax(-1)
    q = torch.tensor([0.1, 0.3, 0.2, 0.15, 0.25], dtype=torch.float64)
    if width < 5:
        q[width:] = (1 - q[:width].sum()) * p[width:] / p[width:].sum()
    lp, _, correction = score_centering_logprobs(
        x.expand(5, -1),
        torch.arange(5),
        torch.arange(width).expand(5, -1),
        q[:width].log().expand(5, -1),
        icepop_bounds=bounds,
    )
    weights = score_centering_icepop_weights(lp, q.log(), bounds)
    gradient = torch.autograd.grad((q * (correction - weights * lp)).sum(), x)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_icepop_weights_include_boundaries_and_reject_adjacent_outside_values(dtype):
    """Use exact log thresholds and their neighbors without tolerance widening."""
    bounds = (0.5, 5.0)
    thresholds = torch.tensor(bounds, dtype=dtype).log()
    lower, upper = thresholds.unbind()
    lp = torch.stack(
        [
            torch.nextafter(lower, lower.new_tensor(-torch.inf)),
            lower,
            upper,
            torch.nextafter(upper, upper.new_tensor(torch.inf)),
            lower.new_tensor(-1000.0),
            lower.new_tensor(1000.0),
        ]
    ).requires_grad_()
    q = torch.zeros_like(lp, requires_grad=True)
    actual = score_centering_icepop_weights(lp, q, bounds)
    torch.testing.assert_close(
        actual, lp.new_tensor([0.0, 0.5, 5.0, 0.0, 0.0, 0.0]), atol=1e-6, rtol=1e-6
    )
    assert not actual.requires_grad and torch.isfinite(actual).all()


@pytest.mark.parametrize("rho", [0.0, 0.1, 0.5, 1.0, 3.0])
def test_icepop_tail_acceptance_matches_dense_distribution(rho):
    """Cover zero tail and tails below, inside, and above the acceptance interval."""
    p = torch.tensor([0.4, 0.4, 0.1, 0.1], dtype=torch.float64)
    q = torch.tensor(
        [(1 - 0.2 * rho) * 0.7, (1 - 0.2 * rho) * 0.3, 0.1 * rho, 0.1 * rho],
        dtype=torch.float64,
    )
    x = p.log().requires_grad_()
    lp = x.log_softmax(-1)
    c = score_centering_correction(lp[:2], q[:2].log(), icepop_bounds=(0.5, 5.0))
    ratio = p / q
    mass = torch.where((ratio >= 0.5) & (ratio <= 5.0), p, 0.0)
    expected = (mass * lp).sum()
    torch.testing.assert_close(
        torch.autograd.grad(c, x, retain_graph=True)[0],
        torch.autograd.grad(expected, x)[0],
        atol=1e-12,
        rtol=1e-10,
    )


def test_icepop_zero_probability_heads_and_tiny_tail_have_finite_gradients():
    """Zero q and a floored trainer tail do not produce inf*0 or NaN."""
    x = torch.tensor([0.0, -1.0, -30.0, -31.0], dtype=torch.float64, requires_grad=True)
    lp = x.log_softmax(-1)
    q = torch.tensor([0.8, 0.0], dtype=torch.float64)
    c = score_centering_correction(lp[:2], q.log(), icepop_bounds=(0.5, 5.0))
    c.backward()
    assert torch.isfinite(c) and torch.isfinite(x.grad).all()
    zero = score_centering_correction(
        torch.tensor([-torch.inf]), torch.tensor([-torch.inf]), icepop_bounds=(0.5, 5.0)
    )
    torch.testing.assert_close(zero, torch.tensor(0.0), atol=0, rtol=0)


@pytest.mark.parametrize("backend", ["fsdp:d1", "fsdp:d1t2c2", "megatron:d1t2c2p2"])
@pytest.mark.parametrize("lower", [None, 0.5])
def test_icepop_sc_accepts_existing_config_and_actor_logging(backend, lower):
    """No new YAML field is needed; actor initialization accepts the same rule."""
    from areal.trainer.ppo.actor import PPOActor

    config = PPOActorConfig(
        backend=backend,
        use_score_centering=True,
        recompute_logprob=False,
        rejection_sampling=RejectionSamplingConfig(
            action="mask", lower=lower, upper=5.0
        ),
    )
    assert resolve_score_centering_rejection(config.rejection_sampling) == {
        "icepop_bounds": (lower or 0.0, 5.0)
    }
    PPOActor(config, SimpleNamespace())
    with pytest.raises(ValueError, match="supports rejection_sampling only"):
        resolve_score_centering_tis_cap(config.rejection_sampling)


@pytest.mark.parametrize(
    "bounds",
    [
        (-0.1, 5.0),
        (0.5, float("inf")),
        (0.5, float("nan")),
        (True, 5.0),
        (0.5, 1e100),
        (6.0, 5.0),
    ],
)
def test_icepop_sc_rejects_invalid_bounds(bounds):
    """Reject invalid bounds rather than silently changing the weighting rule."""
    rs = SimpleNamespace(
        level="token", metric="ratio", action="mask", lower=bounds[0], upper=bounds[1]
    )
    with pytest.raises(ValueError, match="IcePop"):
        resolve_score_centering_rejection(rs)


def test_icepop_sc_rejects_simultaneous_tis_and_icepop():
    """The sampled and centering terms cannot select two weighting rules."""
    with pytest.raises(ValueError, match="either TIS or IcePop"):
        score_centering_correction(
            torch.zeros(2), torch.zeros(2), tis_cap=2.0, icepop_bounds=(0.5, 5.0)
        )


@pytest.mark.parametrize("all_rejected", [False, True])
@pytest.mark.parametrize("mode", ["token", "sequence"])
def test_actor_icepop_keeps_correction_and_original_denominator(
    all_rejected, mode, monkeypatch
):
    """Rejection only zeros the sampled term, even when every sample is rejected."""
    from areal.trainer.ppo.actor import grpo_loss_fn
    from areal.utils.functional.loss_aggregation import SequenceMean, TokenMean
    from areal.utils.stats_tracker import DistributedStatsTracker

    monkeypatch.setattr(
        "areal.trainer.ppo.actor.stats_tracker", DistributedStatsTracker()
    )
    lp = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]], requires_grad=True)
    ratio = lp.new_tensor([[10.0, 0.1, 1.0], [2.0, 1.0, 1.0]])
    if all_rejected:
        ratio.fill_(10.0)
    old = lp.detach() - ratio.log()
    mask = torch.tensor([[True, True, True], [True, False, False]])
    adv = lp.new_tensor([[1.0, -2.0, 3.0], [-1.0, 0.0, 0.0]])
    correction = 0.3 * lp
    reduction = TokenMean(mask) if mode == "token" else SequenceMean(mask)
    actual = grpo_loss_fn(
        lp,
        torch.zeros_like(lp),
        dict(logprobs=old, prox_logp=lp.detach() + 1, advantages=adv, loss_mask=mask),
        eps_clip=0.2,
        eps_clip_higher=None,
        c_clip=None,
        use_score_centering=True,
        sc_correction=correction,
        rejection_sampling=RejectionSamplingConfig(action="mask", lower=0.5, upper=5.0),
        loss_step=SimpleNamespace(bind=lambda _: reduction),
    )
    weights = torch.where((ratio >= 0.5) & (ratio <= 5.0), ratio, 0.0)
    expected = reduction.aggregate(adv * (correction - weights * lp), mask)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    g = torch.autograd.grad(actual, lp, retain_graph=True)[0]
    torch.testing.assert_close(
        g, torch.autograd.grad(expected, lp)[0], atol=1e-6, rtol=1e-6
    )
    assert g[0, 0] != 0
    assert not g[~mask].any()

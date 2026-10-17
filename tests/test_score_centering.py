# SPDX-License-Identifier: Apache-2.0
"""Self-verification gate for Score Centering (arXiv:2609.20807).

Tests 1-4 exercise ``areal.utils.score_centering`` in isolation (torch only).
Test 5 exercises the ``grpo_loss_fn`` integration and is skipped when the full
AReaL stack (ray/transformers/...) is not importable.
"""

from __future__ import annotations

import math
import pathlib
import sys
import types

import pytest
import torch

# ``areal/__init__.py`` imports the full infra stack (ray, engines, ...), which
# is not available in a lightweight unit-test environment. Fall back to a
# namespace stub pointing at the real package path so the submodule still
# imports; the full stack is preferred when present.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
try:  # pragma: no cover - depends on the environment
    import areal  # noqa: F401
except Exception:  # pragma: no cover
    sys.modules.pop("areal", None)
    _areal = types.ModuleType("areal")
    _areal.__path__ = [str(_ROOT / "areal")]
    sys.modules["areal"] = _areal
    _areal_utils = types.ModuleType("areal.utils")
    _areal_utils.__path__ = [str(_ROOT / "areal" / "utils")]
    sys.modules["areal.utils"] = _areal_utils

from areal.utils.score_centering import (  # noqa: E402
    gather_train_head_logp,
    score_centering_correction,
)


def _full_vocab_ids(V: int, T: int) -> torch.Tensor:
    return torch.arange(V).view(1, 1, V).expand(1, T, V)


def test_gather_matches_log_softmax() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    ids = torch.arange(7).expand(2, 3, 7)
    got = gather_train_head_logp(logits, ids)
    expected = torch.log_softmax(logits, dim=-1)
    torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-12)


def test_gate1_onpolicy_gradient_is_zero() -> None:
    """Gate 1: p == q => the SC correction is an exact no-op (value and gradient).

    With the VeRL/Slime tail model, on-policy p=q gives residual=0, so the
    correction *value* is exactly 0 (not just its gradient). This is stronger
    than the old simplified implementation where the value was H(p) > 0.
    """
    torch.manual_seed(0)
    V, T = 8, 4
    logits = torch.randn(1, T, V, dtype=torch.float64, requires_grad=True)
    ids = _full_vocab_ids(V, T)
    sampler_logp = torch.log_softmax(logits, dim=-1).detach()
    train_head_logp = gather_train_head_logp(logits, ids)

    correction, s_bar, q_mass, p_mass, rho = score_centering_correction(
        train_head_logp, sampler_logp
    )

    torch.testing.assert_close(q_mass, torch.ones_like(q_mass), rtol=0.0, atol=1e-6)
    # On-policy: residual = q*1 - 1*p = 0, so correction = 0 (exact no-op).
    torch.testing.assert_close(
        correction, torch.zeros_like(correction), rtol=0.0, atol=1e-6
    )

    grad = torch.autograd.grad(correction.sum(), logits)[0]
    torch.testing.assert_close(grad, torch.zeros_like(grad), rtol=0.0, atol=1e-6)


def test_gate2_sampler_receives_no_gradient() -> None:
    """Gate 2: sampler quantities are detached; only trainer logits get grads."""
    torch.manual_seed(0)
    V, T = 6, 3
    logits = torch.randn(1, T, V, dtype=torch.float64, requires_grad=True)
    sampler_logp = (
        torch.log_softmax(logits, dim=-1).detach().clone().requires_grad_(True)
    )
    ids = _full_vocab_ids(V, T)
    train_head_logp = gather_train_head_logp(logits, ids)

    correction, _, _, _, _ = score_centering_correction(train_head_logp, sampler_logp)
    correction.sum().backward()

    assert sampler_logp.grad is None
    assert logits.grad is not None


def test_gate3_q_mass_is_one_when_k_equals_vocab() -> None:
    """Gate 3: k == vocab => captured mass q_mass == 1.0."""
    torch.manual_seed(0)
    V, T = 10, 2
    logits = torch.randn(1, T, V, dtype=torch.float64, requires_grad=True)
    ids = _full_vocab_ids(V, T)
    sampler_logp = torch.log_softmax(logits, dim=-1).detach()
    train_head_logp = gather_train_head_logp(logits, ids)

    _, _, q_mass, _, _ = score_centering_correction(train_head_logp, sampler_logp)
    torch.testing.assert_close(q_mass, torch.ones_like(q_mass), rtol=0.0, atol=1e-9)


def test_gate3_q_mass_is_captured_mass_for_topk() -> None:
    """Gate 3 (top-k): q_mass equals the sampler mass captured by the top-k."""
    torch.manual_seed(0)
    V, T, K = 12, 3, 4
    logits = torch.randn(1, T, V, dtype=torch.float64)
    logp = torch.log_softmax(logits, dim=-1)
    top = torch.topk(logp, K, dim=-1)
    ids = top.indices
    sampler_logp = top.values.detach()
    train_head_logp = gather_train_head_logp(logits, ids)

    _, _, q_mass, _, _ = score_centering_correction(train_head_logp, sampler_logp)
    expected = sampler_logp.exp().sum(-1)
    torch.testing.assert_close(q_mass, expected, rtol=0.0, atol=1e-9)
    assert torch.all(q_mass <= 1.0 + 1e-9)


def test_gate4_onpolicy_sc_loss_gradient_equals_baseline() -> None:
    """Gate 4: on-policy, adding the SC term leaves the gradient unchanged."""
    torch.manual_seed(0)
    V, T = 8, 5
    logits = torch.randn(1, T, V, dtype=torch.float64, requires_grad=True)
    logp = torch.log_softmax(logits, dim=-1)
    ids = _full_vocab_ids(V, T)
    sampler_logp = logp.detach()
    loss_mask = torch.ones(1, T, dtype=torch.bool)
    advantages = torch.randn(1, T, dtype=torch.float64)

    base = -(logp.gather(-1, ids) * advantages.unsqueeze(-1)).sum(-1)
    base = (base * loss_mask).sum() / loss_mask.sum()

    train_head_logp = gather_train_head_logp(logits, ids)
    correction, _, _, _, _ = score_centering_correction(train_head_logp, sampler_logp)
    sc_loss = torch.where(
        loss_mask, correction, torch.zeros_like(correction)
    ).sum() / loss_mask.sum().clamp(min=1)
    total = base + sc_loss

    g_base = torch.autograd.grad(base, logits, retain_graph=True)[0]
    g_total = torch.autograd.grad(total, logits)[0]
    torch.testing.assert_close(g_base, g_total, rtol=0.0, atol=1e-9)


def test_gate5_grpo_loss_uses_sc_only_when_enabled() -> None:
    """Gate 5: use_score_centering=False leaves the loss unchanged; True adds
    exactly the masked SC correction."""
    try:
        from areal.trainer.ppo.actor import grpo_loss_fn
    except Exception as exc:  # pragma: no cover - full stack not installed here
        pytest.skip(f"full AReaL stack unavailable: {exc}")

    torch.manual_seed(0)
    B, T, V, K = 1, 4, 16, 4
    logits = torch.randn(B, T, V, dtype=torch.float32, requires_grad=True)
    sampled = torch.randint(0, V, (B, T))
    logprobs = torch.log_softmax(logits, dim=-1).gather(-1, sampled.unsqueeze(-1))
    logprobs = logprobs.squeeze(-1).detach().requires_grad_(True)
    entropy = torch.zeros(B, T)
    advantages = torch.randn(B, T)
    loss_mask = torch.ones(B, T, dtype=torch.bool)

    top = torch.topk(torch.log_softmax(logits.detach(), dim=-1), K, dim=-1)
    input_data = {
        "logprobs": logprobs.detach(),
        "advantages": advantages,
        "loss_mask": loss_mask,
        "logits": logits,
        "sampler_topk_token_ids": top.indices,
        "sampler_topk_logps": top.values,
        "prox_logp": logprobs.detach(),  # FIX: required by grpo_loss_fn default recompute
    }
    base_inputs = {k: v for k, v in input_data.items() if not k.startswith("sampler")}
    base_inputs.pop("logits", None)

    common = dict(eps_clip=0.2, eps_clip_higher=None, c_clip=None)
    loss_off = grpo_loss_fn(logprobs, entropy, dict(base_inputs), **common)
    loss_off_with_fields = grpo_loss_fn(
        logprobs, entropy, dict(input_data), use_score_centering=False, **common
    )
    torch.testing.assert_close(loss_off, loss_off_with_fields, rtol=1e-5, atol=1e-8)

    loss_on = grpo_loss_fn(
        logprobs, entropy, dict(input_data), use_score_centering=True, **common
    )
    train_head_logp = gather_train_head_logp(logits, top.indices)
    correction, _, _, _, _ = score_centering_correction(train_head_logp, top.values)
    expected_sc = torch.where(
        loss_mask,
        advantages * correction,
        torch.zeros_like(correction),
    ).sum() / loss_mask.sum().clamp(min=1)
    torch.testing.assert_close(loss_on - loss_off, expected_sc, rtol=0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Fix-pass regression tests (review findings I1, I2, I4, I5, M2)
# ---------------------------------------------------------------------------


def test_gather_train_head_logp_applies_temperature() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 7, dtype=torch.float64)
    ids = _full_vocab_ids(7, 3)
    got = gather_train_head_logp(logits, ids, temperature=2.0)
    expected = torch.log_softmax(logits / 2.0, dim=-1)
    torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-12)


def test_mask_excludes_slots_from_mass() -> None:
    """I2: a masked slot must contribute zero mass, not exp(0) = 1."""
    torch.manual_seed(0)
    V, T, K = 8, 3, 5
    logits = torch.randn(1, T, V, dtype=torch.float64)
    logp = torch.log_softmax(logits, dim=-1)
    top = torch.topk(logp, K, dim=-1)
    ids, sampler = top.indices, top.values.detach()
    train_head_logp = gather_train_head_logp(logits, ids)

    mask = torch.ones(1, T, K, dtype=torch.bool)
    mask[:, :, -1] = False
    corr, s_bar, q_mass, p_mass, rho = score_centering_correction(
        train_head_logp, sampler, sampler_topk_mask=mask
    )

    # Masked slot must contribute zero to q_mass (exp(0)=1 would inflate it).
    # Reference: q_mass = sum of unmasked sampler probs.
    expected_q_mass = (sampler[:, :, :-1].exp()).sum(-1)
    torch.testing.assert_close(q_mass, expected_q_mass, rtol=0.0, atol=1e-10)

    # Unmasked correction should differ from full-k correction (fewer slots).
    corr_full, _, q_mass_full, _, _ = score_centering_correction(
        train_head_logp, sampler
    )
    # q_mass with mask < q_mass without (one slot dropped).
    assert torch.all(q_mass < q_mass_full)


def test_pad_topk_helpers_use_neg_inf() -> None:
    """I2: producer padding must be -inf (q = 0), never 0.0 (q = 1)."""
    from areal.utils.score_centering import pad_topk_ids, pad_topk_logps

    assert pad_topk_ids([3, 4], 4) == [3, 4, 0, 0]
    assert pad_topk_logps([-0.1, -0.2], 4) == [
        -0.1,
        -0.2,
        float("-inf"),
        float("-inf"),
    ]
    padded = torch.tensor([pad_topk_logps([-0.1], 3)], dtype=torch.float64)
    torch.testing.assert_close(
        padded.exp(),
        torch.tensor([[math.exp(-0.1), 0.0, 0.0]], dtype=torch.float64),
        rtol=0.0,
        atol=1e-12,
    )


def test_sp_guard_rejects_ulysses() -> None:
    """I1: SC is rejected under Ulysses SP (sp_size > 1)."""
    from areal.utils.score_centering import check_score_centering_sp_supported

    check_score_centering_sp_supported(True, 1)
    check_score_centering_sp_supported(False, 8)
    with pytest.raises(NotImplementedError):
        check_score_centering_sp_supported(True, 2)


def test_invalid_sc_mode_raises() -> None:
    """M2 / I5: invalid modes must fail loudly; 'mis' (IcePop) is now implemented."""
    torch.manual_seed(0)
    logits = torch.randn(1, 2, 4, dtype=torch.float64)
    sampler = torch.log_softmax(logits, dim=-1).detach()
    train_head_logp = gather_train_head_logp(logits, _full_vocab_ids(4, 2))
    with pytest.raises(ValueError):
        score_centering_correction(train_head_logp, sampler, sc_mode="bogus")
    # 'mis' is now implemented as IcePop band-mask (VeRL-style), should not raise.
    score_centering_correction(
        train_head_logp, sampler, sc_mode="mis", sc_clip_low=0.5, sc_clip_high=2.0
    )


def test_tis_uses_clamped_is_weights() -> None:
    """F-04: sc_mode=tis uses VeRL-style clamped IS weights (ratio.clamp(max=high)).

    The correction must match an independent reference computed from scratch
    using rho+alpha tail model and clamped head weights.
    """
    torch.manual_seed(0)
    V, T, K = 12, 3, 4
    logits = torch.randn(1, T, V, dtype=torch.float64)
    logp = torch.log_softmax(logits, dim=-1)
    top = torch.topk(logp, K, dim=-1)
    ids, sampler_logp = top.indices, top.values.detach()

    # Deliberately create off-policy: trainer logits shifted
    train_logits = logits + 0.5
    train_head_logp = gather_train_head_logp(train_logits, ids)

    clip_high = 2.0
    correction, s_bar, q_mass, p_mass, rho = score_centering_correction(
        train_head_logp,
        sampler_logp,
        sc_mode="tis",
        sc_clip_low=0.0,
        sc_clip_high=clip_high,
    )

    # Independent reference using VeRL's rho+alpha+clamp formula.
    p = train_head_logp.detach().exp()
    q = sampler_logp.exp()
    p_mass, q_mass_ref = p.sum(-1), q.sum(-1)
    eps = 1e-6
    rho = (1 - q_mass_ref).clamp_min(eps) / (1 - p_mass).clamp_min(eps)
    alpha = rho * (1.0 / rho).clamp(max=clip_high)
    head_w = ((train_head_logp - sampler_logp).exp()).clamp(max=clip_high)
    residual = q * head_w - alpha.unsqueeze(-1) * p
    expected_correction = (residual.detach() * train_head_logp).sum(-1)

    torch.testing.assert_close(correction, expected_correction, rtol=0.0, atol=1e-12)
    # q_mass uses raw q (sampler head mass), not IS weights
    torch.testing.assert_close(q_mass, q_mass_ref, rtol=0.0, atol=1e-12)


# === US-SC-VERIFY2: PPOActorConfig SC field validation (spec F-01) ===


def test_ppo_actor_config_has_sc_fields_with_defaults():
    """F-01: PPOActorConfig has 5 SC fields with correct defaults."""
    import dataclasses

    from areal.api.cli_args import PPOActorConfig

    field_names = {f.name for f in dataclasses.fields(PPOActorConfig)}
    sc_fields = {
        "use_score_centering",
        "sc_top_k",
        "sc_mode",
        "sc_clip_low",
        "sc_clip_high",
    }
    assert sc_fields.issubset(field_names), (
        f"Missing SC fields: {sc_fields - field_names}"
    )

    # Check defaults via __dataclass_fields__ (avoids full instantiation)
    fields_meta = PPOActorConfig.__dataclass_fields__
    assert fields_meta["use_score_centering"].default is False
    assert fields_meta["sc_top_k"].default == 128
    assert fields_meta["sc_mode"].default == "none"
    assert fields_meta["sc_clip_low"].default == 0.0
    assert fields_meta["sc_clip_high"].default == float("inf")


def test_ppo_actor_config_rejects_invalid_sc_mode():
    """F-01: sc_mode not in (none, tis, mis) -> ValueError."""
    import pytest

    from areal.api.cli_args import PPOActorConfig

    with pytest.raises(ValueError, match="sc_mode must be"):
        PPOActorConfig(
            experiment_name="test",
            trial_name="test",
            path="",
            sc_mode="bogus",
        )


def test_ppo_actor_config_rejects_invalid_clip_range():
    """F-01: sc_clip_low >= sc_clip_high -> ValueError."""
    import pytest

    from areal.api.cli_args import PPOActorConfig

    with pytest.raises(ValueError, match="sc_clip_low must be strictly less"):
        PPOActorConfig(
            experiment_name="test",
            trial_name="test",
            path="",
            sc_clip_low=1.0,
            sc_clip_high=1.0,
        )


# === END US-SC-VERIFY2 ===


# ---------------------------------------------------------------------------
# T2-T8: PR-review fix regression coverage (P0-A/B/C/D, P1-a/b/c)
# ---------------------------------------------------------------------------


def _sc_actor_config(**overrides):
    """Build a PPOActorConfig with Score Centering enabled by default."""
    from areal.api.cli_args import PPOActorConfig

    kwargs = dict(
        experiment_name="test",
        trial_name="test",
        path="",
        use_score_centering=True,
    )
    kwargs.update(overrides)
    return PPOActorConfig(**kwargs)


def test_t2_pad_slots_contribute_no_mass_and_stay_finite() -> None:
    """T2 (P0-C): ``-inf`` sampler pad slots are detected via finiteness, the
    trainer head ``p`` is masked, and no quantity becomes NaN/Inf."""
    torch.manual_seed(0)
    V, T, K = 8, 3, 5
    n_pad = 2
    trainer_logits = torch.randn(1, T, V, dtype=torch.float64, requires_grad=True)
    sampler_logits = torch.randn(1, T, V, dtype=torch.float64)
    top = torch.topk(torch.log_softmax(sampler_logits, dim=-1), K - n_pad, dim=-1)

    ids = torch.zeros(1, T, K, dtype=torch.long)
    ids[:, :, : K - n_pad] = top.indices
    sampler_logp = torch.full((1, T, K), float("-inf"), dtype=torch.float64)
    sampler_logp[:, :, : K - n_pad] = top.values
    train_head_logp = gather_train_head_logp(trainer_logits, ids)

    correction, s_bar, q_mass, p_mass, rho = score_centering_correction(
        train_head_logp, sampler_logp
    )

    for name, value in (
        ("correction", correction),
        ("s_bar", s_bar),
        ("q_mass", q_mass),
        ("p_mass", p_mass),
        ("rho", rho),
    ):
        assert torch.isfinite(value).all(), f"{name} contains non-finite entries"

    # Pad slots contribute zero mass and cannot inflate the trainer head mass.
    assert torch.all(p_mass < 1.0)
    expected_q_mass = (torch.isfinite(sampler_logp).float() * sampler_logp.exp()).sum(
        -1
    )
    torch.testing.assert_close(q_mass, expected_q_mass, rtol=0.0, atol=1e-12)

    grad = torch.autograd.grad(correction.sum(), trainer_logits)[0]
    assert torch.isfinite(grad).all()


def test_t2_safe_ratio_handles_neg_inf_pad_in_both_heads() -> None:
    """T2 (P0-C): a pad slot with ``-inf`` sampler logprob must be masked out of
    ``q``/``p``/head-weights so no ``inf`` leaks into the correction."""
    train_head_logp = torch.tensor(
        [[[-0.1, -0.2, -0.3]]], dtype=torch.float64, requires_grad=True
    )
    sampler_logp = torch.tensor([[[-0.1, -0.2, float("-inf")]]], dtype=torch.float64)
    mask = torch.tensor([[[True, True, False]]])

    correction, s_bar, q_mass, p_mass, rho = score_centering_correction(
        train_head_logp, sampler_logp, sampler_topk_mask=mask
    )
    for value in (correction, s_bar, q_mass, p_mass, rho):
        assert torch.isfinite(value).all()
    torch.testing.assert_close(
        q_mass, torch.exp(sampler_logp[..., :2]).sum(-1), rtol=0.0, atol=1e-12
    )
    # The masked pad slot must be excluded from the trainer head mass too.
    torch.testing.assert_close(
        p_mass,
        torch.exp(train_head_logp.detach()[..., :2]).sum(-1),
        rtol=0.0,
        atol=1e-12,
    )


def test_t3_sampler_topk_is_rolled_with_loss_mask() -> None:
    """T3 (P0-B): ``_compute_advantages`` must roll the sampler top-k fields by
    ``-1`` on the sequence dim so they align with the shifted logprobs."""
    try:
        from areal.trainer.ppo.actor import PPOActor
    except Exception as exc:  # pragma: no cover - full stack not installed here
        pytest.skip(f"full AReaL stack unavailable: {exc}")

    class _Cfg:
        overlong_reward_penalty = False
        overlong_tokens = None
        overlong_penalty_factor = None
        max_new_tokens = 16
        use_decoupled_loss = False
        recompute_logprob = False

    fake_self = types.SimpleNamespace(
        config=_Cfg(),
        reward_bias=0.0,
        reward_scaling=1.0,
        reward_clip=float("inf"),
        reward_norm=None,
        kl_ctl=0.0,
        kl_estimator=lambda old, ref: torch.zeros_like(old),
        mask_no_eos_with_zero=False,
        discount=1.0,
        gae_lambda=1.0,
        adv_norm=None,
    )

    B, T, K = 1, 4, 2
    sampler_topk_logps = torch.arange(B * T * K, dtype=torch.float64).reshape(B, T, K)
    sampler_topk_token_ids = torch.arange(B * T * K, dtype=torch.long).reshape(B, T, K)
    sampler_topk_mask = torch.ones(B, T, K, dtype=torch.bool)
    data = {
        "input_ids": torch.zeros(B, T, dtype=torch.long),
        "attention_mask": torch.ones(B, T, dtype=torch.bool),
        "loss_mask": torch.ones(B, T, dtype=torch.int32),
        "logprobs": torch.zeros(B, T, dtype=torch.float64),
        "rewards": torch.zeros(B, dtype=torch.float64),
        "sampler_topk_logps": sampler_topk_logps.clone(),
        "sampler_topk_token_ids": sampler_topk_token_ids.clone(),
        "sampler_topk_mask": sampler_topk_mask.clone(),
    }

    out = PPOActor._compute_advantages(fake_self, data)

    torch.testing.assert_close(
        out["sampler_topk_logps"], torch.roll(sampler_topk_logps, -1, dims=1)
    )
    torch.testing.assert_close(
        out["sampler_topk_token_ids"], torch.roll(sampler_topk_token_ids, -1, dims=1)
    )
    torch.testing.assert_close(
        out["sampler_topk_mask"], torch.roll(sampler_topk_mask, -1, dims=1)
    )


def test_t4_sc_correction_is_advantage_modulated() -> None:
    """T4 (P0-A): the SC loss must be ``mean(mask * advantage * correction)``;
    flipping the advantage sign flips the SC contribution."""
    try:
        from areal.trainer.ppo.actor import grpo_loss_fn
    except Exception as exc:  # pragma: no cover - full stack not installed here
        pytest.skip(f"full AReaL stack unavailable: {exc}")

    torch.manual_seed(0)
    B, T, V, K = 1, 4, 16, 4
    logits = torch.randn(B, T, V, dtype=torch.float64)
    top = torch.topk(torch.log_softmax(logits, dim=-1), K, dim=-1)
    logprobs = torch.zeros(B, T, dtype=torch.float64)
    entropy = torch.zeros(B, T, dtype=torch.float64)
    loss_mask = torch.ones(B, T, dtype=torch.bool)
    advantages = torch.tensor([[1.0, -2.0, 0.5, 3.0]], dtype=torch.float64)

    input_data = {
        "logprobs": logprobs,
        "advantages": advantages,
        "loss_mask": loss_mask,
        "logits": logits,
        "sampler_topk_token_ids": top.indices,
        "sampler_topk_logps": top.values,
        "prox_logp": logprobs,
    }
    base_inputs = {
        k: v
        for k, v in input_data.items()
        if k != "logits" and not k.startswith("sampler")
    }
    common = dict(eps_clip=0.2, eps_clip_higher=None, c_clip=None)

    loss_off = grpo_loss_fn(logprobs, entropy, dict(base_inputs), **common)
    loss_on = grpo_loss_fn(
        logprobs, entropy, dict(input_data), use_score_centering=True, **common
    )

    train_head_logp = gather_train_head_logp(logits, top.indices)
    correction, _, _, _, _ = score_centering_correction(train_head_logp, top.values)
    expected_sc = (advantages * correction)[loss_mask].sum() / loss_mask.sum()
    torch.testing.assert_close(loss_on - loss_off, expected_sc, rtol=0.0, atol=1e-6)

    # Sign flip: negating the advantage negates the SC contribution.
    neg_input_data = dict(input_data)
    neg_input_data["advantages"] = -advantages
    neg_base = dict(base_inputs)
    neg_base["advantages"] = -advantages
    loss_off_neg = grpo_loss_fn(logprobs, entropy, neg_base, **common)
    loss_on_neg = grpo_loss_fn(
        logprobs, entropy, neg_input_data, use_score_centering=True, **common
    )
    torch.testing.assert_close(
        loss_on_neg - loss_off_neg, -expected_sc, rtol=0.0, atol=1e-6
    )


def test_t5_correction_returns_five_values() -> None:
    """T5: ``score_centering_correction`` returns the 5-tuple
    ``(correction, s_bar, q_mass, p_mass, rho)``."""
    torch.manual_seed(0)
    V, T = 6, 2
    logits = torch.randn(1, T, V, dtype=torch.float64)
    sampler = torch.log_softmax(logits, dim=-1).detach()
    train_head_logp = gather_train_head_logp(logits, _full_vocab_ids(V, T))

    out = score_centering_correction(train_head_logp, sampler)
    assert isinstance(out, tuple) and len(out) == 5
    correction, s_bar, q_mass, p_mass, rho = out
    for value in out:
        assert value.shape == (1, T)
    # On-policy with k == vocab: head masses match and residual vanishes.
    torch.testing.assert_close(p_mass, q_mass, rtol=0.0, atol=1e-9)
    torch.testing.assert_close(
        correction, torch.zeros_like(correction), rtol=0.0, atol=1e-9
    )


def test_t6_sc_mode_requires_valid_clip_bounds() -> None:
    """T6 (P0-D): tis/mis reject clamp bounds that would degenerate the
    correction or produce 0*inf = NaN on pad slots."""
    with pytest.raises(ValueError, match="tis"):
        _sc_actor_config(sc_mode="tis")  # default sc_clip_high = inf
    with pytest.raises(ValueError, match="finite"):
        _sc_actor_config(sc_mode="mis")  # default sc_clip_high = inf
    with pytest.raises(ValueError, match="sc_clip_low >= 0"):
        _sc_actor_config(sc_mode="mis", sc_clip_low=-1.0, sc_clip_high=2.0)
    # Valid bounds must be accepted.
    assert _sc_actor_config(sc_mode="tis", sc_clip_high=2.0).sc_mode == "tis"
    assert (
        _sc_actor_config(sc_mode="mis", sc_clip_low=0.0, sc_clip_high=2.0).sc_mode
        == "mis"
    )


def test_t7_ppo_config_sc_sampling_prerequisites() -> None:
    """T7 (P1): SC requires stochastic, temperature>0, 0<top_p<=1, top_k>=sc_top_k,
    and zero frequency penalty."""
    from areal.api.cli_args import GenerationHyperparameters, PPOConfig

    def _config(**gconfig_overrides):
        gconfig = GenerationHyperparameters(sc_top_k=8, **gconfig_overrides)
        return PPOConfig(
            experiment_name="test",
            trial_name="test",
            actor=_sc_actor_config(),
            gconfig=gconfig,
        )

    with pytest.raises(ValueError, match="stochastic"):
        _config(greedy=True)
    with pytest.raises(ValueError, match="temperature > 0"):
        _config(temperature=0.0)
    with pytest.raises(ValueError, match="top_p"):
        _config(top_p=0.0)
    with pytest.raises(ValueError, match="top_k >= sc_top_k"):
        _config(top_k=4)
    with pytest.raises(ValueError, match="frequency_penalty"):
        _config(frequency_penalty=0.5)
    # A valid stochastic configuration must construct cleanly.
    assert _config().gconfig.sc_top_k == 8


def test_t8_ppo_actor_config_sc_incompatible_combinations() -> None:
    """T8 (P1): tree training is rejected; recompute_logprob only warns."""
    with pytest.raises(NotImplementedError, match="tree training"):
        _sc_actor_config(enable_tree_training=True)

    cfg = _sc_actor_config(recompute_logprob=True)
    assert cfg.use_score_centering
    assert cfg.recompute_logprob

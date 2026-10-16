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

    correction, s_bar, q_mass = score_centering_correction(
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

    correction, _, _ = score_centering_correction(train_head_logp, sampler_logp)
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

    _, _, q_mass = score_centering_correction(train_head_logp, sampler_logp)
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

    _, _, q_mass = score_centering_correction(train_head_logp, sampler_logp)
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
    correction, _, _ = score_centering_correction(train_head_logp, sampler_logp)
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
    correction, _, _ = score_centering_correction(train_head_logp, top.values)
    expected_sc = torch.where(
        loss_mask, correction, torch.zeros_like(correction)
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
    corr, s_bar, q_mass = score_centering_correction(
        train_head_logp, sampler, sampler_topk_mask=mask
    )

    # Masked slot must contribute zero to q_mass (exp(0)=1 would inflate it).
    # Reference: q_mass = sum of unmasked sampler probs.
    expected_q_mass = (sampler[:, :, :-1].exp()).sum(-1)
    torch.testing.assert_close(q_mass, expected_q_mass, rtol=0.0, atol=1e-10)

    # Unmasked correction should differ from full-k correction (fewer slots).
    corr_full, _, q_mass_full = score_centering_correction(train_head_logp, sampler)
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
    correction, s_bar, q_mass = score_centering_correction(
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

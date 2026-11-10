# SPDX-License-Identifier: Apache-2.0
"""Guards that keep Score Centering from silently running headless or misaligned.

Each test pins one review fix: a failure here means the fix regressed, not that the
test is flaky (all cases are pure CPU, no model, no rollout).
"""

from types import SimpleNamespace

import pytest

from areal.api.cli_args import (
    GenerationHyperparameters,
    InferenceEngineConfig,
    PPOActorConfig,
    PPOConfig,
    RejectionSamplingConfig,
)
from areal.engine.vllm_remote import _parse_top_logprobs, _token_to_id
from areal.experimental.openai.client import drop_vision_pad_tokens
from areal.infra.remote_inf_engine import require_topk_heads, validate_head_alignment
from areal.utils.data import _TOKEN_PAYLOAD_KEYS
from areal.utils.topk_padding import NEG_INF_PAD, zero_mass_row_count


def test_requested_heads_but_none_returned_raises():
    """SC on + every chunk headless must fail loudly, not continue headless."""
    with pytest.raises(RuntimeError, match="no chunk returned sampler top-k heads"):
        require_topk_heads([], 128, requested=True)
    # SC off: no heads is the normal case
    require_topk_heads([], 0, requested=True)
    # SC on with heads present: fine
    require_topk_heads([[[1, 2]]], 128, requested=True)


def test_v2_rollout_controller_is_rejected_at_load_time():
    """The inference-service bridge never carries top-k heads; reject the config."""
    gconfig = GenerationHyperparameters(score_centering_top_k=2)
    with pytest.raises(ValueError, match="v2 rollout controller"):
        PPOConfig(
            actor=PPOActorConfig(backend="fsdp:d1", use_score_centering=True),
            rollout=InferenceEngineConfig(backend="vllm:d1", _version="v2"),
            gconfig=gconfig,
        )
    # the default (v1) still loads
    PPOConfig(
        actor=PPOActorConfig(backend="fsdp:d1", use_score_centering=True),
        rollout=InferenceEngineConfig(backend="vllm:d1"),
        gconfig=GenerationHyperparameters(score_centering_top_k=2),
    )


def test_baseline_help_does_not_promise_tis_weighting():
    """The baseline branch is bare REINFORCE; the help text must not claim TIS."""
    help_text = PPOActorConfig.__dataclass_fields__["sc_reinforce_baseline"].metadata[
        "help"
    ]
    assert "TIS" not in help_text
    assert "rejection_sampling" not in help_text


def test_clamp_lower_must_be_non_negative():
    """A clamp floors the weight: 0 is meaningful, a negative floor is not."""
    assert (
        RejectionSamplingConfig(
            metric="ratio", action="clamp", upper=2.0, lower=0.0
        ).lower
        == 0.0
    )
    with pytest.raises(ValueError, match="non-negative for 'clamp'"):
        RejectionSamplingConfig(metric="ratio", action="clamp", upper=2.0, lower=-1.0)
    with pytest.raises(ValueError, match="lower must be positive for 'ratio'"):
        RejectionSamplingConfig(metric="ratio", action="mask", upper=2.0, lower=0.0)


def test_vision_pad_drop_keeps_sampler_heads_aligned():
    """Dropping tokens must drop the row-parallel heads, or every later head shifts."""
    response = SimpleNamespace(
        output_tokens=[11, 99, 12],
        output_logprobs=[-0.1, -0.2, -0.3],
        output_versions=[1, 1, 1],
        sampler_head_ids=[[11, 1], [99, 1], [12, 1]],
        sampler_head_logprobs=[[-0.1, -9.0], [-0.2, -9.0], [-0.3, -9.0]],
    )
    drop_vision_pad_tokens(response, frozenset({99}))

    assert response.output_tokens == [11, 12]
    assert response.output_logprobs == [-0.1, -0.3]
    assert response.sampler_head_ids == [[11, 1], [12, 1]]
    assert response.sampler_head_logprobs == [[-0.1, -9.0], [-0.3, -9.0]]


def test_guard_keys_off_the_score_centering_switch():
    """SC off with a leftover top-k width must not abort the rollout."""
    require_topk_heads([], 8, requested=False)  # must not raise
    with pytest.raises(RuntimeError, match="no chunk returned sampler top-k heads"):
        require_topk_heads([], 8, requested=True)


def test_actor_switch_is_mirrored_to_the_engine_config():
    """The transport cannot see actor.use_score_centering, so it is mirrored."""
    on = PPOConfig(
        actor=PPOActorConfig(backend="fsdp:d1", use_score_centering=True),
        rollout=InferenceEngineConfig(backend="vllm:d1"),
        gconfig=GenerationHyperparameters(score_centering_top_k=2),
    )
    assert on.rollout.use_score_centering is True
    off = PPOConfig(
        rollout=InferenceEngineConfig(backend="vllm:d1"),
        gconfig=GenerationHyperparameters(score_centering_top_k=2),
    )
    assert off.rollout.use_score_centering is False


def _rs(upper=5.0):
    return RejectionSamplingConfig(action="clamp", upper=upper)


def test_score_centering_rejects_both_rollout_correction_stages():
    """SC owns its weighting through sc_mode/sc_clip_low/sc_clip_high."""
    with pytest.raises(ValueError, match="sc_mode/sc_clip_low/sc_clip_high"):
        PPOActorConfig(
            backend="fsdp:d1",
            use_score_centering=True,
            rejection_sampling=_rs(),
        )
    with pytest.raises(ValueError, match="sc_mode/sc_clip_low/sc_clip_high"):
        PPOActorConfig(
            backend="fsdp:d1",
            use_score_centering=True,
            rejection_sampling=_rs(),
            importance_sampling=_rs(),
        )


def test_rejection_sampling_warning_still_fires_without_score_centering(monkeypatch):
    """SC no longer reads rejection_sampling, so the plain warning stays."""
    import areal.api.cli_args as cli_args_module

    warnings: list[str] = []
    monkeypatch.setattr(
        cli_args_module.logger,
        "warning",
        lambda message, *a, **k: warnings.append(message),
    )

    PPOActorConfig(backend="fsdp:d1", rejection_sampling=_rs())
    assert any("Filtering will be ignored" in w for w in warnings)

    warnings.clear()
    PPOActorConfig(backend="fsdp:d1", use_score_centering=True)
    assert not any("Filtering will be ignored" in w for w in warnings)


def test_plan_b_help_matches_the_enforced_contract():
    """The help must not promise a config the loader rejects."""
    help_text = PPOActorConfig.__dataclass_fields__["sc_loss_mode"].metadata["help"]
    assert "Both plans require" in help_text
    assert "recompute_logprob=False" in help_text
    assert "recompute_logprob=true" not in help_text


def test_plan_b_loads_only_without_recompute_logprob():
    """Plan B is loadable with the enforced contract, and only with it."""
    PPOActorConfig(
        backend="fsdp:d1",
        use_score_centering=True,
        sc_loss_mode="ppo_additive",
        recompute_logprob=False,
    )
    with pytest.raises(ValueError, match="recompute_logprob=False"):
        PPOActorConfig(
            backend="fsdp:d1",
            use_score_centering=True,
            sc_loss_mode="ppo_additive",
            recompute_logprob=True,
        )


def test_token_to_id_accepts_only_known_forms():
    """Both sampler key shapes resolve; ambiguous text and negative ids do not."""
    assert _token_to_id(42) == 42
    assert _token_to_id("token:7") == 7
    assert _token_to_id("token_id:7") == 7
    assert _token_to_id("step:12") is None
    assert _token_to_id(-3) is None
    assert _token_to_id(None) is None
    assert _token_to_id(True) is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.5])
def test_parser_rejects_invalid_log_probs(bad):
    """A sampler log-prob must be finite and non-positive."""
    with pytest.raises(ValueError, match="finite and <= 0"):
        _parse_top_logprobs({"token:1": bad})


def test_parser_rejects_duplicate_token_ids():
    """The same token twice in one row would double its probability mass."""
    row = [
        {"token_id": 1, "logprob": -0.5},
        {"token_id": 1, "logprob": -0.6},
    ]
    with pytest.raises(ValueError, match="duplicate token id"):
        _parse_top_logprobs(row)


def test_head_width_must_agree_across_config_blocks():
    """rollout.score_centering_top_k is auto-filled from gconfig."""
    cfg = PPOConfig(
        actor=PPOActorConfig(backend="fsdp:d1", use_score_centering=True),
        gconfig=GenerationHyperparameters(score_centering_top_k=8),
        rollout=InferenceEngineConfig(backend="vllm:d1p1t1", score_centering_top_k=0),
    )
    assert cfg.rollout.score_centering_top_k == 8


def test_head_alignment_rejects_a_dropped_chunk():
    """A chunk that loses its top-k rows must fail instead of shifting heads."""
    validate_head_alignment(5, 5, 5)
    with pytest.raises(ValueError, match="misaligned with the generated tokens"):
        validate_head_alignment(4, 5, 5)


def test_zero_mass_row_count_flags_all_neg_inf_rows():
    """Rows carrying no sampler mass are counted so the engine can warn."""
    assert zero_mass_row_count([[-0.5, -1.0], [NEG_INF_PAD, NEG_INF_PAD]]) == 1
    assert zero_mass_row_count([[]]) == 1
    assert zero_mass_row_count([[-0.5]]) == 0


def test_routed_experts_shares_the_token_payload_axis():
    """routed_experts is trimmed on the token axis exactly like the sampler heads."""
    assert "routed_experts" in _TOKEN_PAYLOAD_KEYS
    assert "sampler_head_ids" in _TOKEN_PAYLOAD_KEYS
    assert "sampler_head_logprobs" in _TOKEN_PAYLOAD_KEYS

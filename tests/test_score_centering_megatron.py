# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for Score Centering on the Megatron backend.

The numeric end-to-end checks need accelerator hardware; these tests cover the
parts that are purely local:

* the config-level backend matrix (``PPOActorConfig.__post_init__``),
* the packed-row alignment used when context parallelism slices token rows,
* the Megatron engine helper's validation, shape checks and diagnostics,
* the score-centering math against a plain ``log_softmax`` reference.

All tests run on CPU. The Megatron import is skipped gracefully when the optional
``megatron`` package is unavailable, so the module is safe in CPU-only CI.
"""

import logging
import types

import pytest
import torch

pytest.importorskip("megatron.core")

from areal.api.cli_args import MegatronEngineConfig, PPOActorConfig  # noqa: E402
from areal.engine.megatron_utils.packed_context_parallel import (  # noqa: E402
    split_packed_seqs_for_context_parallel,
)
from areal.utils.score_centering import (  # noqa: E402
    build_weight_fn,
    score_centering_logprobs,
)

BASE_CONFIG = dict(experiment_name="x", trial_name="y", path="p")


def _actor_config(**overrides) -> PPOActorConfig:
    """Build a PPOActorConfig from the minimal required fields plus overrides."""
    return PPOActorConfig(**BASE_CONFIG, **overrides)


@pytest.mark.parametrize(
    "backend",
    [
        "megatron:d1p1t1",
        "megatron:d2p1t1",  # data parallel
        "megatron:d1p1t2",  # tensor parallel
        "megatron:d2p1t2",  # dp x tp
        "megatron:d2p2t2",  # + pipeline parallel
        "megatron:d2p1t2c2",  # + context parallel
        "megatron:(attn:d2p1t1c1|ffn:d1p1t1e2)",  # + expert parallel (MoE)
    ],
)
def test_sc_config_megatron_parallel_axes_accepted(backend):
    """Test that every Megatron parallel axis is accepted when SC is enabled."""
    config = _actor_config(backend=backend, use_score_centering=True)

    assert config.backend == backend


def test_sc_config_fsdp_backend_still_accepted():
    """Test that the FSDP path keeps working unchanged."""
    config = _actor_config(backend="fsdp:d2p1t1", use_score_centering=True)

    assert config.backend == "fsdp:d2p1t1"


def test_sc_config_mtp_training_enabled_rejected():
    """Test that Megatron MTP training is rejected because it scores extra rows."""
    with pytest.raises(ValueError, match="MTP"):
        _actor_config(
            backend="megatron:d2p1t1",
            use_score_centering=True,
            megatron=MegatronEngineConfig(
                bridge_type="megatron-bridge", enable_mtp_training=True
            ),
        )


def test_sc_config_chunked_logits_enabled_rejected():
    """Test that the chunked LM head is rejected because it hides full logits."""
    with pytest.raises(ValueError, match="chunked_logits"):
        _actor_config(
            backend="megatron:d2p1t1",
            use_score_centering=True,
            megatron=MegatronEngineConfig(enable_chunked_logits=True),
        )


def test_sc_config_critic_rejected():
    """Test that a critic actor is rejected: SC scores sampler heads only."""
    with pytest.raises(ValueError, match="actor"):
        _actor_config(
            backend="megatron:d2p1t1", use_score_centering=True, is_critic=True
        )


def test_sc_config_pipeline_parallel_allowed_when_sc_disabled():
    """Test that PP configs without SC are untouched by the new validation."""
    config = _actor_config(backend="megatron:d2p2t2", use_score_centering=False)

    assert config.use_score_centering is False


def test_sc_config_guard_skipped_when_sc_disabled():
    """Test that MTP is only rejected while SC is enabled."""
    config = _actor_config(
        backend="megatron:d2p1t1",
        use_score_centering=False,
        megatron=MegatronEngineConfig(
            bridge_type="megatron-bridge", enable_mtp_training=True
        ),
    )

    assert config.use_score_centering is False


@pytest.fixture
def packed_rows():
    """Two packed sequences of 8 tokens each; row t carries its own index."""
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)
    rows = torch.arange(16, dtype=torch.float32)
    heads = torch.stack([rows * 10, rows * 10 + 1], dim=-1)
    return cu_seqlens, rows, heads


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_cp_split_keeps_head_rows_aligned(monkeypatch, packed_rows, cp_rank):
    """Test that the CP split keeps every head row attached to its own token."""
    from megatron.core import parallel_state as mpu

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: cp_rank)
    cu_seqlens, rows, heads = packed_rows

    local_rows = split_packed_seqs_for_context_parallel(rows, cu_seqlens)
    local_heads = split_packed_seqs_for_context_parallel(heads, cu_seqlens)

    assert local_rows.numel() == rows.numel() // 2
    assert local_heads.shape == (local_rows.numel(), 2)
    # Every local row must carry the heads of the token it belongs to.
    torch.testing.assert_close(local_heads[:, 0], local_rows * 10, rtol=0, atol=0)
    torch.testing.assert_close(local_heads[:, 1], local_rows * 10 + 1, rtol=0, atol=0)


def test_cp_split_leaves_rows_untouched_without_context_parallel(
    monkeypatch, packed_rows
):
    """Test that cp_size == 1 returns the input tensor unchanged."""
    from megatron.core import parallel_state as mpu

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    cu_seqlens, rows, _ = packed_rows

    assert split_packed_seqs_for_context_parallel(rows, cu_seqlens) is rows


class _StubEngine:
    """Binds the real engine helpers onto a stub so they run without hardware."""

    _ensure_score_centering_supported = None  # filled in by the fixture
    _warn_sc_head_padding = None
    _warn_sc_head_mass = None
    _score_centering_logprobs = None

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        mtp: bool = False,
        chunked: bool = False,
        tree: bool = False,
        vocab_size: int = 256,
    ):
        self.config = types.SimpleNamespace(
            temperature=temperature,
            logprobs_chunk_size=8,
            sc_mode="none",
            sc_clip_low=0.5,
            sc_clip_high=2.0,
            use_score_centering=True,
        )
        self.mcore_config = types.SimpleNamespace(
            enable_mtp_training=mtp, enable_chunked_logits=chunked
        )
        self.hf_config = types.SimpleNamespace(vocab_size=vocab_size)
        self.enable_tree_training = tree
        self.logger = logging.getLogger("TestScoreCentering")


@pytest.fixture
def engine_cls(monkeypatch):
    """Expose the real MegatronEngine helpers with a CPU-only TP stub."""
    from megatron.core import parallel_state as mpu

    from areal.engine.megatron_engine import MegatronEngine

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None)
    _StubEngine._ensure_score_centering_supported = (
        MegatronEngine._ensure_score_centering_supported
    )
    _StubEngine._warn_sc_head_padding = MegatronEngine._warn_sc_head_padding
    _StubEngine._warn_sc_head_mass = MegatronEngine._warn_sc_head_mass
    _StubEngine._score_centering_logprobs = MegatronEngine._score_centering_logprobs
    return _StubEngine


@pytest.fixture
def micro_batch():
    """A packed micro-batch whose sampler heads come from its own logits."""
    generator = torch.Generator().manual_seed(0)
    rows, vocab, heads = 12, 256, 16
    logits = torch.randn(rows, vocab, generator=generator)
    labels = torch.randint(0, vocab, (rows,), generator=generator)
    top = logits.softmax(-1).topk(heads, dim=-1)
    return {
        "logits": logits,
        "labels": labels,
        "ids": top.indices,
        "q": top.values.log(),
        "mask": torch.ones(rows),
    }


def test_engine_sc_scores_packed_micro_batch(engine_cls, micro_batch):
    """Test that the engine helper returns log-probs equal to log_softmax."""
    engine = engine_cls()

    logprobs, entropy, kwargs = engine._score_centering_logprobs(
        micro_batch["logits"],
        {
            "sampler_head_ids": micro_batch["ids"],
            "sampler_head_logprobs": micro_batch["q"],
            "loss_mask": micro_batch["mask"],
        },
        micro_batch["labels"],
    )

    reference = (
        micro_batch["logits"]
        .log_softmax(-1)
        .gather(-1, micro_batch["labels"][:, None])
        .squeeze(-1)
    )
    torch.testing.assert_close(logprobs, reference, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(entropy).all()
    assert set(kwargs) == {"sc_correction", "sc_metrics"}
    assert torch.isfinite(kwargs["sc_correction"]).all()


def test_engine_sc_prefers_context_parallel_local_heads(engine_cls, micro_batch):
    """Test that CP-local head rows win over global rows when both are present."""
    engine = engine_cls()
    local = {
        "_cp_local_sampler_head_ids": micro_batch["ids"],
        "_cp_local_sampler_head_logprobs": micro_batch["q"],
        "_cp_local_loss_mask": micro_batch["mask"],
        # Global heads belong to another alignment: they must not be used.
        "sampler_head_ids": micro_batch["ids"][:3],
        "sampler_head_logprobs": micro_batch["q"][:3],
    }

    logprobs, _, _ = engine._score_centering_logprobs(
        micro_batch["logits"], local, micro_batch["labels"]
    )

    assert logprobs.shape == micro_batch["labels"].shape


def test_engine_sc_missing_sampler_heads_rejected(engine_cls, micro_batch):
    """Test that a micro-batch without sampler heads raises a clear error."""
    engine = engine_cls()

    with pytest.raises(ValueError, match="carries no sampler top-k heads"):
        engine._score_centering_logprobs(
            micro_batch["logits"], {}, micro_batch["labels"]
        )


def test_engine_sc_misaligned_head_rows_rejected(engine_cls, micro_batch):
    """Test that head rows misaligned with logit rows are rejected."""
    engine = engine_cls()
    batch = {
        "sampler_head_ids": micro_batch["ids"][:5],
        "sampler_head_logprobs": micro_batch["q"][:5],
    }

    with pytest.raises(ValueError, match="one sampler-head row per logit row"):
        engine._score_centering_logprobs(
            micro_batch["logits"], batch, micro_batch["labels"]
        )


def test_engine_sc_temperature_not_one_rejected(engine_cls, micro_batch):
    """Test that temperature != 1 is rejected: the sampler probs must be intact."""
    engine = engine_cls(temperature=0.7)

    with pytest.raises(ValueError, match="temperature=1"):
        engine._score_centering_logprobs(
            micro_batch["logits"],
            {
                "sampler_head_ids": micro_batch["ids"],
                "sampler_head_logprobs": micro_batch["q"],
            },
            micro_batch["labels"],
        )


@pytest.mark.parametrize("kw", [dict(mtp=True), dict(chunked=True), dict(tree=True)])
def test_engine_sc_unsupported_modes_rejected(engine_cls, kw):
    """Test that MTP, chunked logits and tree training are refused by the engine."""
    engine = engine_cls(**kw)

    with pytest.raises(ValueError):
        engine._ensure_score_centering_supported()


def test_engine_sc_accepts_every_parallel_axis(engine_cls, micro_batch, monkeypatch):
    """Test that pp/tp/cp sizes are accepted by the engine guard."""
    from megatron.core import parallel_state as mpu

    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: object())
    engine = engine_cls(vocab_size=micro_batch["logits"].shape[-1] * 8)

    # tp>1 needs a real process group for the shard reduction, so only the mode
    # guard and the shard-coverage check are exercised here.
    engine._ensure_score_centering_supported()

    assert engine._sc_support_checked is True


def test_engine_sc_narrow_vocab_shards_rejected(engine_cls, micro_batch, monkeypatch):
    """Test that vocab shards narrower than the vocabulary are rejected."""
    from megatron.core import parallel_state as mpu

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: object())
    engine = engine_cls(vocab_size=micro_batch["logits"].shape[-1] * 16)

    with pytest.raises(ValueError, match="tp shards cover"):
        engine._score_centering_logprobs(
            micro_batch["logits"],
            {
                "sampler_head_ids": micro_batch["ids"],
                "sampler_head_logprobs": micro_batch["q"],
            },
            micro_batch["labels"],
        )


def test_engine_sc_low_head_mass_warns_once(engine_cls):
    """Test that incomplete sampler heads warn once and only once."""
    engine = engine_cls()
    weak_metrics = {"q_mass": torch.full((4,), 0.5)}

    engine._warn_sc_head_mass(weak_metrics)
    warned = engine._sc_mass_warned
    engine._warn_sc_head_mass(weak_metrics)

    assert warned is True
    assert engine._sc_mass_warned is True


def test_engine_sc_full_head_mass_does_not_warn(engine_cls):
    """Test that complete sampler heads stay quiet."""
    engine = engine_cls()

    engine._warn_sc_head_mass({"q_mass": torch.full((4,), 0.999)})

    assert getattr(engine, "_sc_mass_warned", False) is False


def test_engine_sc_guard_caches_success_only(engine_cls):
    """Test that the mode guard caches a pass but never caches a failure."""
    ok = engine_cls()
    ok._ensure_score_centering_supported()
    first = getattr(ok, "_sc_support_checked", False) is True
    ok._ensure_score_centering_supported()  # second call must be a no-op
    second = ok._sc_support_checked is True

    bad = engine_cls(chunked=True)
    raised = False
    try:
        bad._ensure_score_centering_supported()
    except ValueError:
        raised = True
    cached_after_failure = getattr(bad, "_sc_support_checked", False)

    assert first and second
    assert raised and cached_after_failure is False


def test_sc_logprobs_reference_matches_vocab_softmax():
    """Test that score_centering_logprobs matches a plain log_softmax gather."""
    generator = torch.Generator().manual_seed(1)
    rows, vocab, k = 8, 128, 16
    logits = torch.randn(rows, vocab, generator=generator)
    labels = torch.randint(0, vocab, (rows,), generator=generator)
    top = logits.softmax(-1).topk(k, dim=-1)

    logprobs, entropy, correction = score_centering_logprobs(
        logits,
        labels,
        top.indices,
        top.values.log(),
        chunk_size=4,
        weight_fn=build_weight_fn("none", 0.5, 2.0),
    )

    reference = logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    torch.testing.assert_close(logprobs, reference, rtol=1e-6, atol=1e-7)
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(correction).all()

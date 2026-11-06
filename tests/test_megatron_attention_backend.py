# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from areal.api.cli_args import MegatronEngineConfig


@pytest.mark.parametrize("backend", [None, "auto", "flash", "fused", "unfused"])
def test_attention_backend_structured_config_roundtrip(backend):
    config = OmegaConf.structured(MegatronEngineConfig())
    config.attention_backend = backend
    loaded = OmegaConf.to_object(config)
    assert loaded.attention_backend == backend
    assert not loaded.use_deterministic_algorithms


@pytest.mark.parametrize("backend", ["fa3", "FLASH", "invalid"])
def test_attention_backend_rejects_unknown_values(backend):
    with pytest.raises(ValueError, match="megatron.attention_backend"):
        MegatronEngineConfig(attention_backend=backend)


@pytest.mark.parametrize("backend", ["auto", "fused", "unfused"])
def test_attention_backend_rejects_deterministic_conflicts(backend):
    with pytest.raises(ValueError, match="requires attention_backend"):
        MegatronEngineConfig(
            attention_backend=backend, use_deterministic_algorithms=True
        )


@pytest.mark.parametrize("backend", [None, "flash"])
def test_attention_backend_accepts_deterministic_flash(backend):
    MegatronEngineConfig(attention_backend=backend, use_deterministic_algorithms=True)


@pytest.mark.parametrize("bridge_type", ["mbridge", "mcore-bridge", "native"])
def test_attention_backend_applied_before_model_build(monkeypatch, bridge_type):
    pytest.importorskip("mbridge")
    pytest.importorskip("megatron.core")
    from megatron.core.transformer.enums import AttnBackend

    from areal.models.mcore import registry

    config = SimpleNamespace(attention_backend=AttnBackend.auto, fp16=False, bf16=True)

    class ModelConstructionReached(Exception):
        pass

    def build(**kwargs):
        assert config.attention_backend == AttnBackend.flash
        raise ModelConstructionReached

    bridge = SimpleNamespace(config=config, get_model=build)
    hf_config = SimpleNamespace(
        rope_scaling=None, vocab_size=32, max_position_embeddings=128, rope_theta=10000
    )
    monkeypatch.setattr(registry, "make_mcore_layer_specs", lambda *args: None)
    monkeypatch.setattr(registry, "GPTModel", build)
    with pytest.raises(ModelConstructionReached):
        registry.make_mcore_model(
            hf_config,
            config,
            MegatronEngineConfig(attention_backend="flash"),
            bridge=None if bridge_type == "native" else bridge,
            bridge_type=bridge_type,
        )


@pytest.mark.parametrize(
    "backend,cp_size,expected_calls",
    [(None, 2, 0), ("fused", 2, 0), ("flash", 1, 0), ("flash", 2, 1)],
)
def test_attention_backend_cp_compat_only_for_explicit_flash(
    monkeypatch, backend, cp_size, expected_calls
):
    pytest.importorskip("mbridge")
    pytest.importorskip("megatron.core")
    from areal.models.mcore import flash_attention_compat, registry

    calls = []
    monkeypatch.setattr(
        flash_attention_compat, "patch_te_flash_attention_3", lambda: calls.append(True)
    )
    sentinel = object()
    config = SimpleNamespace(attention_backend=sentinel, context_parallel_size=cp_size)
    registry._configure_attention_backend(
        config, MegatronEngineConfig(attention_backend=backend)
    )
    assert len(calls) == expected_calls
    if backend is None:
        assert config.attention_backend is sentinel


def test_fa3_cp_legacy_interface_and_repeated_patch(monkeypatch):
    pytest.importorskip("transformer_engine.pytorch")
    from transformer_engine.pytorch.attention.dot_product_attention import backends

    from areal.models.mcore.flash_attention_compat import patch_te_flash_attention_3

    calls = []

    def forward(value, *, window_size):
        calls.append(("forward", value, window_size))
        return value

    def backward(value, *, causal, window_size):
        calls.append(("backward", value, causal, window_size))
        return None, None, None, value

    monkeypatch.setattr(backends, "_flash_attn_fwd_v3", forward)
    monkeypatch.setattr(backends, "_flash_attn_bwd_v3", backward)
    assert patch_te_flash_attention_3()
    assert not patch_te_flash_attention_3()
    assert backends._flash_attn_fwd_v3(7, window_size_left=3, window_size_right=1) == 7
    assert backends._flash_attn_bwd_v3(9, is_causal=True) == 9
    assert calls == [("forward", 7, (3, 1)), ("backward", 9, True, (-1, -1))]


@pytest.mark.parametrize("available", [False, True])
def test_fa3_cp_modern_or_absent_interface_is_unchanged(monkeypatch, available):
    pytest.importorskip("transformer_engine.pytorch")
    from transformer_engine.pytorch.attention.dot_product_attention import backends

    from areal.models.mcore.flash_attention_compat import patch_te_flash_attention_3

    def forward(*, window_size_left=-1, window_size_right=-1):
        return window_size_left, window_size_right

    original = forward if available else None
    monkeypatch.setattr(backends, "_flash_attn_fwd_v3", original)
    assert not patch_te_flash_attention_3()
    assert backends._flash_attn_fwd_v3 is original

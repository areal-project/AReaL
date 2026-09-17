# SPDX-License-Identifier: Apache-2.0

"""Integration contracts for native AWEX Qwen3-VL colocate support."""

from types import SimpleNamespace

import pytest

qwen3_vl = pytest.importorskip(
    "awex.models.qwen3_vl",
    reason="requires an AWEX release with native Qwen3-VL support",
)

from awex.models.registry import (  # noqa: E402
    ModelRegistry,
    get_infer_weights_converter,
    get_sharding_strategy,
)
from awex.sharding.param_sharding import ShardingType  # noqa: E402

from areal.engine.awex.colocate_reader import (  # noqa: E402
    _get_awex_infer_hf_config,
    _get_router_dtype,
)


class _CompositeVLConfig:
    def __init__(self, architectures=None, router_dtype="fp32"):
        self.architectures = architectures
        self.text_config = SimpleNamespace(
            num_hidden_layers=28,
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            router_dtype=router_dtype,
        )
        self.vision_config = SimpleNamespace(num_heads=2, hidden_size=4)

    def to_dict(self):
        return {
            "architectures": self.architectures,
            "model_type": "qwen3_vl",
            "text_config": vars(self.text_config),
            "vision_config": vars(self.vision_config),
        }


def _rank_info():
    return SimpleNamespace(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        ep_rank=0,
        ep_size=1,
        ep_tp_rank=0,
        ep_tp_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
    )


def _infer_engine_config():
    return SimpleNamespace(tp_size=1, ep_size=1, device_backend="cpu")


@pytest.mark.parametrize(
    "architecture",
    ["Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration"],
)
def test_awex_registry_provides_native_qwen3_vl_entries(architecture):
    entry = ModelRegistry.models[architecture]

    assert entry["sglang_converter"] is qwen3_vl.Qwen3VLSGlangToHFWeightConverter
    assert entry["sharding_strategy"] is qwen3_vl.Qwen3VLShardingStrategy
    assert get_sharding_strategy(architecture) is qwen3_vl.Qwen3VLShardingStrategy


@pytest.mark.parametrize(
    "architecture",
    ["Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration"],
)
def test_colocate_infer_converter_receives_composite_vl_config(architecture):
    config = _CompositeVLConfig(architectures=[architecture])

    converter = get_infer_weights_converter(
        "sglang",
        architecture,
        config,
        _rank_info(),
        _infer_engine_config(),
    )

    assert isinstance(converter, qwen3_vl.Qwen3VLSGlangToHFWeightConverter)
    assert converter.vl_model_config is config
    assert converter.model_config is config.text_config


@pytest.mark.parametrize(
    ("name", "expected_dim"),
    [
        ("model.visual.blocks.0.attn.qkv.weight", 0),
        ("model.visual.blocks.0.attn.proj.weight", 1),
    ],
)
def test_colocate_vision_tower_uses_awex_tp_sharding(name, expected_dim):
    rank_info = _rank_info()
    rank_info.tp_size = 2
    strategy = qwen3_vl.Qwen3VLShardingStrategy(
        engine_name="sglang",
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        moe_dense_tp_size=2,
        tp_size=2,
        ep_size=1,
        ep_tp_size=1,
        rank_info=rank_info,
        device_backend="cpu",
    )

    assert strategy.get_sharding_strategy(name) == (
        ShardingType.TP_SHARDING,
        expected_dim,
        2,
    )


def test_colocate_reader_serializes_complete_vl_config():
    config = _CompositeVLConfig(architectures=["Qwen3VLForConditionalGeneration"])

    class Qwen3VLForConditionalGeneration:
        pass

    model = Qwen3VLForConditionalGeneration()
    model.config = config

    awex_config = _get_awex_infer_hf_config(model)

    assert awex_config.architectures == ["Qwen3VLForConditionalGeneration"]
    assert awex_config.model_type == "qwen3_vl"
    assert awex_config.text_config["num_hidden_layers"] == 28
    assert awex_config.vision_config == {"num_heads": 2, "hidden_size": 4}


def test_colocate_reader_fills_missing_runtime_architecture():
    config = _CompositeVLConfig(architectures=None)

    class Qwen3VLForConditionalGeneration:
        pass

    model = Qwen3VLForConditionalGeneration()
    model.config = config

    awex_config = _get_awex_infer_hf_config(model)

    assert awex_config.architectures == ["Qwen3VLForConditionalGeneration"]
    assert awex_config.text_config["num_hidden_layers"] == 28
    assert awex_config.vision_config["num_heads"] == 2


def test_colocate_reader_uses_composite_config_for_vl_moe():
    """Use ModelRunner's composite config when the runtime model keeps text only."""
    config = _CompositeVLConfig(architectures=["Qwen3VLMoeForConditionalGeneration"])

    class Qwen3VLMoeForConditionalGeneration:
        pass

    model = Qwen3VLMoeForConditionalGeneration()
    model.config = config.text_config
    model_runner = SimpleNamespace(model_config=SimpleNamespace(hf_config=config))

    awex_config = _get_awex_infer_hf_config(model, model_runner)

    assert awex_config.architectures == ["Qwen3VLMoeForConditionalGeneration"]
    assert awex_config.text_config["num_hidden_layers"] == 28
    assert awex_config.vision_config == {"num_heads": 2, "hidden_size": 4}


def test_colocate_reader_reads_router_dtype_from_text_config():
    config = _CompositeVLConfig(router_dtype="fp32")

    assert _get_router_dtype(config) == "fp32"
    assert _get_router_dtype(SimpleNamespace(router_dtype="bf16")) == "bf16"

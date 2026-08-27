# SPDX-License-Identifier: Apache-2.0
"""Tests for v2 AWEX SGLang parameter conversion."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from awex.sharding.param_sharding import ShardingType

from areal.v2.weight_update.awex.sglang_adapter import AwexSGLangAdapter


def test_sglang_adapter_reuses_awex_converter_for_vlm_names(monkeypatch):
    """Use v1's AWEX converter contract for text and fused vision parameters."""

    class Qwen3VLForConditionalGeneration:
        def __init__(self):
            self.config = SimpleNamespace()
            self.text_weight = torch.nn.Parameter(torch.ones(4, 4))
            self.vision_qkv = torch.nn.Parameter(torch.ones(12, 4))

        def named_parameters(self):
            return iter(
                (
                    ("model.embed_tokens.weight", self.text_weight),
                    ("visual.blocks.0.attn.qkv_proj.weight", self.vision_qkv),
                )
            )

    model = Qwen3VLForConditionalGeneration()
    server_args = SimpleNamespace()
    scheduler = SimpleNamespace(
        server_args=server_args,
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(model=model)),
    )
    adapter = AwexSGLangAdapter(scheduler)
    rank_info = SimpleNamespace(
        tp_rank=0,
        attn_tp_rank=0,
        pp_rank=0,
        ep_rank=0,
        ep_tp_rank=0,
        global_rank=0,
        world_size=1,
        engine_rank=0,
        cp_rank=0,
        cp_size=1,
        cp_mode="none",
    )
    monkeypatch.setattr(adapter, "_build_rank_info", lambda: rank_info)

    strategy = MagicMock()
    strategy.get_sharding_strategy.return_value = (
        ShardingType.NO_SHARDING,
        0,
        1,
    )
    monkeypatch.setattr(adapter, "_build_sharding_strategy", lambda _: strategy)

    converter = MagicMock()

    def convert_param(name, tensor):
        if name.startswith("visual."):
            return [("model.visual.blocks.0.attn.qkv.weight", tensor)]
        return [("model.language_model.embed_tokens.weight", tensor)]

    converter.convert_param.side_effect = convert_param
    from awex.models import registry

    get_converter = MagicMock(return_value=converter)
    monkeypatch.setattr(registry, "get_infer_weights_converter", get_converter)

    metadata = adapter.get_weight_metadata()
    local_params = adapter.get_local_shard_parameters()

    expected_names = {
        "model.language_model.embed_tokens.weight",
        "model.visual.blocks.0.attn.qkv.weight",
    }
    assert {meta.name for meta in metadata} == expected_names
    assert set(local_params) == expected_names
    assert len(metadata) == len(local_params) == 2
    get_converter.assert_called_once_with(
        "sglang",
        "Qwen3VLForConditionalGeneration",
        hf_config=model.config,
        rank_info=rank_info,
        infer_engine_config=server_args,
    )


def test_sglang_adapter_tied_embedding_adds_missing_lm_head_alias(monkeypatch):
    """Mirror v1's tied-head fallback when the converter only exposes embedding."""

    class Qwen3ForCausalLM:
        def __init__(self):
            self.config = SimpleNamespace(tie_word_embeddings=True)
            self.embedding = torch.nn.Parameter(torch.ones(4, 4))

        def named_parameters(self):
            return iter((("model.embed_tokens.weight", self.embedding),))

    model = Qwen3ForCausalLM()
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(),
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace(model=model)),
    )
    adapter = AwexSGLangAdapter(scheduler)
    rank_info = SimpleNamespace(
        tp_rank=0,
        attn_tp_rank=0,
        pp_rank=0,
        pp_size=1,
        ep_rank=0,
        ep_tp_rank=0,
        global_rank=0,
        world_size=1,
        engine_rank=0,
        cp_rank=0,
        cp_size=1,
        cp_mode="none",
    )
    monkeypatch.setattr(adapter, "_build_rank_info", lambda: rank_info)

    strategy = MagicMock()
    strategy.get_sharding_strategy.return_value = (
        ShardingType.NO_SHARDING,
        0,
        1,
    )
    monkeypatch.setattr(adapter, "_build_sharding_strategy", lambda _: strategy)

    converter = MagicMock()
    converter.convert_param.return_value = [
        ("model.embed_tokens.weight", model.embedding)
    ]
    from awex.models import registry

    monkeypatch.setattr(
        registry,
        "get_infer_weights_converter",
        MagicMock(return_value=converter),
    )

    metadata = adapter.get_weight_metadata()
    local_params = adapter.get_local_shard_parameters()

    expected_names = {"model.embed_tokens.weight", "lm_head.weight"}
    assert {meta.name for meta in metadata} == expected_names
    assert set(local_params) == expected_names
    assert local_params["lm_head.weight"] is local_params["model.embed_tokens.weight"]

# SPDX-License-Identifier: Apache-2.0
"""FSDP canonical views of composite Qwen inference weights."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("awex")

from areal.engine.awex.adapters.sglang_adapter import AwexSGLangAdapter  # noqa: E402


def _reader(model_type, parameters, tp_size=1, architecture=None):
    text = SimpleNamespace(
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        hidden_size=4,
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=4,
        linear_value_head_dim=2,
        attn_output_gate=True,
    )
    defaults = {
        "qwen3_vl": "Qwen3VLForConditionalGeneration",
        "qwen3_5": "Qwen3_5ForConditionalGeneration",
    }
    config = SimpleNamespace(
        model_type=model_type,
        text_config=text,
        architectures=[architecture or defaults.get(model_type, "OtherModel")],
    )
    model = SimpleNamespace(
        config=text,
        named_parameters=lambda: iter(parameters.items()),
    )
    reader = object.__new__(AwexSGLangAdapter)
    reader._parameter_layout = "hf"
    reader._get_model = lambda: model
    reader._get_awex_hf_config = lambda: config
    return reader, SimpleNamespace(tp_size=tp_size, attn_tp_size=tp_size)


def test_fsdp_qwen3_vl_views_match_hf_keys_and_write_through():
    """Keep vision QKV fused and expose text QKV as live HF views."""
    text = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    vision = torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4)
    parameters = {
        "model.embed_tokens.weight": torch.ones(3, 4),
        "model.layers.0.self_attn.qkv_proj.weight": text,
        "visual.blocks.0.attn.qkv_proj.weight": vision,
    }
    reader, rank = _reader("qwen3_vl", parameters)
    result = dict(reader._iter_hf_params(rank))

    assert set(result) == {
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
        "visual.blocks.0.attn.qkv.weight",
    }
    torch.testing.assert_close(
        result["model.language_model.layers.0.self_attn.q_proj.weight"], text[:4]
    )
    torch.testing.assert_close(result["visual.blocks.0.attn.qkv.weight"], vision)
    result["model.language_model.layers.0.self_attn.k_proj.weight"].fill_(23)
    result["visual.blocks.0.attn.qkv.weight"].fill_(29)
    torch.testing.assert_close(text[4:6], torch.full_like(text[4:6], 23))
    torch.testing.assert_close(vision, torch.full_like(vision, 29))


def test_fsdp_qwen35_views_match_hf_keys_and_write_through():
    """Split gated full attention and linear attention with HF-compatible sizes."""
    full = torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4)
    qkvz = torch.arange(24 * 4, dtype=torch.float32).reshape(24, 4)
    ba = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    vision = torch.arange(12 * 4, dtype=torch.float32).reshape(12, 4)
    parameters = {
        "model.layers.0.qkv_proj.weight": full,
        "model.layers.0.o_proj.weight": torch.ones(4, 4),
        "model.layers.1.linear_attn.in_proj_qkvz.weight": qkvz,
        "model.layers.1.linear_attn.in_proj_ba.weight": ba,
        "visual.blocks.0.attn.qkv_proj.weight": vision,
    }
    reader, rank = _reader("qwen3_5", parameters)
    result = dict(reader._iter_hf_params(rank))
    prefix = "model.language_model.layers."
    expected = {
        prefix + "0.self_attn.q_proj.weight": full[:8],
        prefix + "0.self_attn.k_proj.weight": full[8:10],
        prefix + "0.self_attn.v_proj.weight": full[10:],
        prefix + "0.self_attn.o_proj.weight": parameters[
            "model.layers.0.o_proj.weight"
        ],
        prefix + "1.linear_attn.in_proj_qkv.weight": qkvz[:16],
        prefix + "1.linear_attn.in_proj_z.weight": qkvz[16:],
        prefix + "1.linear_attn.in_proj_b.weight": ba[:4],
        prefix + "1.linear_attn.in_proj_a.weight": ba[4:],
        "model.visual.blocks.0.attn.qkv.weight": vision,
    }
    assert set(result) == set(expected)
    for name, tensor in expected.items():
        torch.testing.assert_close(result[name], tensor)
    result[prefix + "1.linear_attn.in_proj_z.weight"].fill_(37)
    result[prefix + "1.linear_attn.in_proj_a.weight"].fill_(41)
    result[prefix + "0.self_attn.q_proj.weight"].fill_(43)
    torch.testing.assert_close(qkvz[16:], torch.full_like(qkvz[16:], 37))
    torch.testing.assert_close(ba[4:], torch.full_like(ba[4:], 41))
    torch.testing.assert_close(full[:8], torch.full_like(full[:8], 43))
    torch.testing.assert_close(
        qkvz[:16], torch.arange(64, dtype=torch.float32).reshape(16, 4)
    )
    torch.testing.assert_close(
        ba[:4], torch.arange(16, dtype=torch.float32).reshape(4, 4)
    )


@pytest.mark.parametrize("model_type", ["qwen3_vl", "qwen3_5"])
def test_fsdp_qwen_composite_rejects_unsupported_inference_tp(model_type):
    """Do not silently expose rank-packed weights as contiguous HF shards."""
    reader, rank = _reader(model_type, {}, tp_size=2)
    with pytest.raises(NotImplementedError, match="TP=1"):
        list(reader._iter_hf_params(rank))


def test_fsdp_other_model_keeps_generic_qkv_conversion():
    """Leave the existing text-only FSDP conversion unchanged."""
    text = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    reader, rank = _reader("qwen2", {"model.layers.0.qkv_proj.weight": text})
    result = dict(reader._iter_hf_params(rank))
    assert set(result) == {
        "model.layers.0.q_proj.weight",
        "model.layers.0.k_proj.weight",
        "model.layers.0.v_proj.weight",
    }
    torch.testing.assert_close(result["model.layers.0.q_proj.weight"], text[:4])


def test_fsdp_text_only_qwen35_does_not_use_composite_namespace():
    """Do not apply the conditional-generation contract to a text-only model."""
    weight = torch.ones(3, 4)
    reader, rank = _reader(
        "qwen3_5",
        {"model.embed_tokens.weight": weight},
        architecture="Qwen3_5ForCausalLM",
    )
    assert list(reader._iter_hf_params(rank))[0][0] == "model.embed_tokens.weight"


def test_megatron_awex_layout_still_uses_model_converter():
    """The FSDP-specific mapping must not replace Megatron's AWEX converter."""
    weight = torch.ones(3, 4)
    reader, rank = _reader("qwen3_vl", {"model.embed_tokens.weight": weight})
    reader._parameter_layout = "awex"
    reader._get_model().config.tie_word_embeddings = False
    reader._get_model_context = lambda: {"pp_rank": 0, "pp_size": 1}
    converter = SimpleNamespace(
        convert_param=lambda name, tensor: [("native." + name, tensor)]
    )
    reader._get_weight_converter = lambda _rank: converter

    converted = dict(reader._iter_hf_params(rank))
    assert set(converted) == {"native.model.embed_tokens.weight"}
    torch.testing.assert_close(converted["native.model.embed_tokens.weight"], weight)

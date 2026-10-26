# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Qwen4Exp Megatron-Bridge configuration and GDN packing."""

import pickle
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from areal.models.mcore.qwen4_exp_megatron_bridge import (
    _GDN_SHAPE_FIELDS,
    _gdn_shape_payload,
    _group_gdn_ba,
    _group_gdn_qkvz,
    _linear_attention_pattern,
    _provider_class,
    _simple_mappings,
    _strict_conversion_tasks,
    _ungroup_gdn_ba,
    _ungroup_gdn_qkvz,
)

# Parameter conversion is a permutation/reshape only. Declare exact tolerances
# before running the conversion; no floating-point arithmetic is involved.
PARAM_ROUNDTRIP_RTOL = 0.0
PARAM_ROUNDTRIP_ATOL = 0.0


def _gdn_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=7,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=3,
        linear_value_head_dim=5,
    )


def test_gdn_shape_broadcast_payload_is_minimal_and_pickleable():
    """GDN export never serializes the dynamic Megatron config class."""

    config = _gdn_config()
    config.unpickleable_model_class = lambda: None

    payload = _gdn_shape_payload(config)

    assert tuple(payload) == _GDN_SHAPE_FIELDS
    assert all(type(value) is int for value in payload.values())
    assert pickle.loads(pickle.dumps(payload)) == payload


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gdn_qkvz_pack_roundtrip_is_exact(dtype):
    """Decoupled QKV/Z packing preserves every value and source shape."""

    config = _gdn_config()
    qkv_rows = (
        2 * config.linear_num_key_heads * config.linear_key_head_dim
        + config.linear_num_value_heads * config.linear_value_head_dim
    )
    z_rows = config.linear_num_value_heads * config.linear_value_head_dim
    qkv = torch.arange(qkv_rows * config.hidden_size, dtype=dtype).reshape(
        qkv_rows, config.hidden_size
    )
    z = torch.arange(z_rows * config.hidden_size, dtype=dtype).reshape(
        z_rows, config.hidden_size
    )

    packed = _group_gdn_qkvz(config, qkv, z)
    actual_qkv, actual_z = _ungroup_gdn_qkvz(config, packed)

    torch.testing.assert_close(
        actual_qkv, qkv, rtol=PARAM_ROUNDTRIP_RTOL, atol=PARAM_ROUNDTRIP_ATOL
    )
    torch.testing.assert_close(
        actual_z, z, rtol=PARAM_ROUNDTRIP_RTOL, atol=PARAM_ROUNDTRIP_ATOL
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gdn_ba_pack_roundtrip_is_exact(dtype):
    """Decoupled B/A packing preserves every value and source shape."""

    config = _gdn_config()
    rows = config.linear_num_value_heads
    b = torch.arange(rows * config.hidden_size, dtype=dtype).reshape(
        rows, config.hidden_size
    )
    a = -torch.arange(rows * config.hidden_size, dtype=dtype).reshape(
        rows, config.hidden_size
    )

    packed = _group_gdn_ba(config, b, a)
    actual_b, actual_a = _ungroup_gdn_ba(config, packed)

    torch.testing.assert_close(
        actual_b, b, rtol=PARAM_ROUNDTRIP_RTOL, atol=PARAM_ROUNDTRIP_ATOL
    )
    torch.testing.assert_close(
        actual_a, a, rtol=PARAM_ROUNDTRIP_RTOL, atol=PARAM_ROUNDTRIP_ATOL
    )


def test_linear_attention_pattern_uses_explicit_hybrid_layout():
    """The provider preserves the checkpoint's explicit GDN/QSA schedule."""

    config = SimpleNamespace(
        layer_types=["linear_attention", "full_attention", "linear_attention"]
    )

    assert _linear_attention_pattern(config) == [True, False, True]


def test_provider_declares_qwen4_exp_extension_fields():
    """The lazy provider exposes every architecture-specific config family."""

    pytest.importorskip("megatron.bridge.models.gpt_provider")
    names = {item.name for item in fields(_provider_class())}
    expected = {
        # Gated residual, PLE, QSA, GDN, and native MTP compatibility.
        "hc_count",
        "ple_layer_ids",
        "indexer_n_heads",
        "linear_attention_freq",
        "linear_decoupled_in_proj",
        "moe_n_hash_layers",
        "mtp_shared_weights",
        "mtp_decoder_input_detach",
        "original_max_position_embeddings",
        "partial_rotary_factor",
    }

    assert expected <= names


def test_parameter_level_tp_exceptions_use_replicated_mapping():
    """Scalar gates and PLE depthwise convolution remain whole on TP ranks."""

    class Auto:
        def __init__(self, megatron_param, hf_param):
            self.megatron_param = megatron_param
            self.hf_param = hf_param

    class Replicated(Auto):
        pass

    mappings = {
        mapping.megatron_param: mapping
        for mapping in _simple_mappings(Auto, Replicated)
    }

    assert isinstance(
        mappings["decoder.layers.*.mlp.shared_experts.gate_weight"], Replicated
    )
    assert isinstance(
        mappings["mtp.layers.0.mtp_model_layer.mlp.shared_experts.gate_weight"],
        Replicated,
    )
    assert isinstance(mappings["decoder.layers.*.ple.conv1d.weight"], Replicated)


def test_conversion_tasks_skip_only_streamed_ple_table():
    """The large PLE table is loaded separately and absent from normal tasks."""

    mapped_task = object()
    names = [
        "decoder.layers.1.ple.ple_embedding.ngram_embedding.weight",
        "decoder.layers.1.ple.conv1d.weight",
    ]

    assert _strict_conversion_tasks(names, [None, mapped_task]) == [mapped_task]


def test_conversion_tasks_reject_any_other_missing_mapping():
    """A generic missing mapping must not be hidden by the PLE exception."""

    with pytest.raises(
        ValueError,
        match=(
            "No Qwen4Exp mapping for Megatron parameter: "
            "decoder.layers.1.ple.conv1d.weight"
        ),
    ):
        _strict_conversion_tasks(
            ["decoder.layers.1.ple.conv1d.weight"],
            [None],
        )


@pytest.mark.parametrize("freeze_ple_table", [False, True])
def test_trainable_ple_export_rejected_before_creating_output(
    tmp_path, freeze_ple_table
):
    """Reject an export that would silently replace a trained PLE table."""
    from areal.engine.megatron_engine import MegatronEngine

    engine = MegatronEngine.__new__(MegatronEngine)
    engine.model = [object()]
    engine.bridge_cls = "megatron-bridge"
    engine.hf_config = SimpleNamespace(model_type="qwen4_exp")
    engine.mcore_config = SimpleNamespace(
        freeze_ple_table=freeze_ple_table, mtp_only=False
    )
    destination = tmp_path / "export"

    with pytest.raises(ValueError, match="would discard trained PLE updates"):
        engine._save_model_to_hf(str(destination))
    assert not destination.exists()

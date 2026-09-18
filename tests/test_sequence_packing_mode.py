# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.engine.core.model import (
    SequencePackingMode,
    resolve_sequence_packing_mode,
    supports_model_packed_seq,
)


@pytest.mark.parametrize("model_type", ["qwen3_vl", "qwen3_vl_moe"])
def test_qwen3_vl_family_uses_model_thd_with_megatron_bridge(model_type):
    assert supports_model_packed_seq(model_type, "megatron-bridge")
    assert (
        resolve_sequence_packing_mode(model_type, "megatron-bridge")
        == SequencePackingMode.MODEL_THD
    )


def test_qwen4_exp_uses_wrapper_thd_with_mcore_bridge():
    assert not supports_model_packed_seq("qwen4_exp", "mcore-bridge")
    assert (
        resolve_sequence_packing_mode("qwen4_exp", "mcore-bridge")
        == SequencePackingMode.WRAPPER_THD
    )


def test_qwen4_exp_is_registered_as_vision_and_moe_model():
    from areal.engine.core.model import is_moe_model, is_valid_vision_model

    assert is_valid_vision_model("qwen4_exp")
    assert is_moe_model("qwen4_exp")


@pytest.mark.parametrize(
    ("model_type", "bridge_type"),
    [
        ("qwen3_vl", "mbridge"),
        ("qwen3_vl_moe", "mbridge"),
        ("qwen2_5_vl", "megatron-bridge"),
        ("qwen3_5", "megatron-bridge"),
        ("qwen3_5_moe", "megatron-bridge"),
        ("qwen4_exp", "megatron-bridge"),
        ("qwen3_vl", "mcore-bridge"),
    ],
)
def test_models_without_gpu_model_thd_contract_stay_padded(model_type, bridge_type):
    assert not supports_model_packed_seq(model_type, bridge_type)
    assert (
        resolve_sequence_packing_mode(model_type, bridge_type)
        == SequencePackingMode.PADDED
    )


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_moe", "llama"])
def test_text_models_keep_wrapper_thd(model_type):
    assert (
        resolve_sequence_packing_mode(model_type, "megatron-bridge")
        == SequencePackingMode.WRAPPER_THD
    )

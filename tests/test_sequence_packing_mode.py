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


@pytest.mark.parametrize(
    ("model_type", "bridge_type"),
    [
        ("qwen3_vl", "mbridge"),
        ("qwen3_vl_moe", "mbridge"),
        ("qwen2_5_vl", "megatron-bridge"),
        ("qwen3_5", "megatron-bridge"),
        ("qwen3_5_moe", "megatron-bridge"),
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

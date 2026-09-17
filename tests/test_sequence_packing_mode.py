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
def test_models_without_gpu_model_thd_contract_stay_padded(
    monkeypatch, model_type, bridge_type
):
    monkeypatch.setattr(
        "areal.engine.core.model.supports_gdn_packed_seq", lambda: False
    )
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


@pytest.mark.parametrize(
    "core_version,bridge_version,packed",
    [("0.17.0", "0.5.1", False), ("0.18.2", "0.4.0", False), ("0.18.2", "0.5.1", True)],
)
@pytest.mark.parametrize("text_only", [False, True])
def test_qwen35_packing_requires_compatible_releases(
    monkeypatch, core_version, bridge_version, packed, text_only
):
    from areal.engine.core import model

    versions = {"megatron-core": core_version, "megatron-bridge": bridge_version}
    monkeypatch.setattr(model, "version", versions.__getitem__)
    model_type = "qwen3_5_text" if text_only else "qwen3_5"
    expected = SequencePackingMode.PADDED
    if packed:
        expected = (
            SequencePackingMode.WRAPPER_THD
            if text_only
            else SequencePackingMode.MODEL_THD
        )
    assert resolve_sequence_packing_mode(model_type, "megatron-bridge") == expected
    assert model.requires_padded_seq(model_type) is not packed

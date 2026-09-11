# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.engine.core import model
from areal.engine.core.model import (
    SequencePackingMode,
    resolve_sequence_packing_mode,
    supports_model_packed_seq,
)


@pytest.mark.parametrize(
    "model_type", ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]
)
def test_qwen3_vl_family_uses_model_thd_with_megatron_bridge(model_type, monkeypatch):
    monkeypatch.setattr(
        model,
        "version",
        {"megatron-core": "0.18.2", "megatron-bridge": "0.5.1"}.__getitem__,
    )
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
        ("qwen3_5", "mbridge"),
        ("qwen3_5_moe", "mbridge"),
        ("qwen3_5_text", "megatron-bridge"),
        ("qwen3_5_moe_text", "megatron-bridge"),
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


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
@pytest.mark.parametrize("feature", ["use_chunked_lm_head", "enable_mtp_training"])
def test_qwen35_padded_loss_features_preserve_existing_layout(
    model_type, feature, monkeypatch
):
    monkeypatch.setattr(model, "supports_qwen35_packed_runtime", lambda: True)
    assert (
        resolve_sequence_packing_mode(model_type, "megatron-bridge", **{feature: True})
        == SequencePackingMode.PADDED
    )


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
@pytest.mark.parametrize(
    "core_version,bridge_version,expected",
    [
        ("0.17.0", "0.4.0", SequencePackingMode.PADDED),
        ("0.17.0", "0.5.1", SequencePackingMode.PADDED),
        ("0.18.2", "0.4.0", SequencePackingMode.PADDED),
        ("0.18.2", "0.5.1", SequencePackingMode.MODEL_THD),
        ("0.18.2rc1", "0.5.1", SequencePackingMode.PADDED),
        ("unknown", "0.5.1", SequencePackingMode.PADDED),
    ],
)
def test_qwen35_runtime_selects_compatible_layout(
    monkeypatch, model_type, core_version, bridge_version, expected
):
    versions = {"megatron-core": core_version, "megatron-bridge": bridge_version}
    monkeypatch.setattr(model, "version", versions.__getitem__)
    assert resolve_sequence_packing_mode(model_type, "megatron-bridge") == expected


def test_missing_megatron_runtime_preserves_padded_layout(monkeypatch):
    def missing_version(name):
        raise model.PackageNotFoundError(name)

    monkeypatch.setattr(model, "version", missing_version)
    assert (
        resolve_sequence_packing_mode("qwen3_5", "megatron-bridge")
        == SequencePackingMode.PADDED
    )
    assert (
        resolve_sequence_packing_mode("qwen3_vl", "megatron-bridge")
        == SequencePackingMode.MODEL_THD
    )

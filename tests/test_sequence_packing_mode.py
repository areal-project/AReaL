# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.engine.core.model import (
    SequencePackingMode,
    resolve_sequence_packing_mode,
    supports_model_packed_seq,
    validate_model_packed_seq_dependencies,
)


@pytest.mark.parametrize(
    "model_type", ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]
)
def test_qwen_vl_family_uses_model_thd_with_megatron_bridge(model_type):
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
        ("qwen3_5", "mbridge"),
        ("qwen3_5_moe", "mbridge"),
        ("qwen2_5_vl", "megatron-bridge"),
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


@pytest.mark.parametrize("model_type,cp_size", [("qwen3_5", 1), ("qwen3_vl", 2)])
@pytest.mark.parametrize(
    "core_version,bridge_version",
    [("0.17.0", "0.5.1"), ("0.18.2", "0.4.0")],
)
def test_model_thd_dependency_guard_rejects_old_releases(
    monkeypatch, model_type, cp_size, core_version, bridge_version
):
    from areal.engine.core import model

    versions = {"megatron-core": core_version, "megatron-bridge": bridge_version}
    monkeypatch.setattr(model, "version", versions.__getitem__)

    with pytest.raises(RuntimeError, match="megatron-core>=0.18.2"):
        validate_model_packed_seq_dependencies(model_type, "megatron-bridge", cp_size)


@pytest.mark.parametrize(
    "model_type,cp_size",
    [
        ("qwen3_5", 1),
        ("qwen3_5_moe", 2),
        ("qwen3_vl", 2),
        ("qwen3_vl_moe", 2),
    ],
)
def test_model_thd_dependency_guard_accepts_minimum_releases(
    monkeypatch, model_type, cp_size
):
    from areal.engine.core import model

    versions = {"megatron-core": "0.18.2", "megatron-bridge": "0.5.1"}
    monkeypatch.setattr(model, "version", versions.__getitem__)
    validate_model_packed_seq_dependencies(model_type, "megatron-bridge", cp_size)


@pytest.mark.parametrize(
    "model_type,bridge_type,cp_size",
    [
        ("qwen3_vl", "megatron-bridge", 1),
        ("qwen2_5_vl", "megatron-bridge", 2),
        ("qwen3_5", "mbridge", 2),
        ("qwen3", "megatron-bridge", 2),
    ],
)
def test_model_thd_dependency_guard_skips_unrelated_paths(
    monkeypatch, model_type, bridge_type, cp_size
):
    from areal.engine.core import model

    def unexpected(package):
        pytest.fail(f"Unrelated path must not query {package}")

    monkeypatch.setattr(model, "version", unexpected)
    validate_model_packed_seq_dependencies(model_type, bridge_type, cp_size)


@pytest.mark.parametrize("missing_package", ["megatron-core", "megatron-bridge"])
def test_model_thd_dependency_guard_reports_missing_package(
    monkeypatch, missing_package
):
    from areal.engine.core import model

    def installed_version(package):
        if package == missing_package:
            raise model.PackageNotFoundError(package)
        return {"megatron-core": "0.18.2", "megatron-bridge": "0.5.1"}[package]

    monkeypatch.setattr(model, "version", installed_version)
    with pytest.raises(RuntimeError, match=f"{missing_package}=not installed"):
        validate_model_packed_seq_dependencies("qwen3_5", "megatron-bridge", 1)


@pytest.mark.parametrize(
    "model_type", ["qwen3_5", "qwen3_5_moe", "qwen3_5_text", "qwen3_5_moe_text"]
)
@pytest.mark.parametrize("bridge_version", [None, "0.4.0", "0.5.1", "0.6.0"])
def test_qwen35_mbridge_stays_padded_when_unused_bridge_changes(
    monkeypatch, model_type, bridge_version
):
    from areal.engine.core import model

    def version(package):
        if package == "megatron-core":
            return "0.18.2"
        if bridge_version is None:
            raise model.PackageNotFoundError(package)
        return bridge_version

    monkeypatch.setattr(model, "version", version)
    assert not supports_model_packed_seq(model_type, "mbridge")
    assert model.requires_padded_seq(model_type, "mbridge")
    assert (
        resolve_sequence_packing_mode(model_type, "mbridge")
        == SequencePackingMode.PADDED
    )


@pytest.mark.parametrize("bridge_type", [None, "mbridge"])
def test_qwen35_unvalidated_bridge_does_not_query_package_versions(
    monkeypatch, bridge_type
):
    from areal.engine.core import model

    def unexpected(package):
        pytest.fail(f"Inactive bridge must not query {package}")

    monkeypatch.setattr(model, "version", unexpected)
    assert model.requires_padded_seq("qwen3_5_text", bridge_type)

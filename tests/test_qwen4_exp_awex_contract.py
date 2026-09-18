# SPDX-License-Identifier: Apache-2.0
"""Frozen-state declarations, checkpoint identity, and live parameter binding."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from areal.models.mcore.qwen4_exp_awex import build_mcore_converter
from areal.models.mcore.qwen4_exp_awex_binding import McoreFrozenBinder
from areal.models.mcore.qwen4_exp_awex_contract import Qwen4ExpFrozenContract

TABLE = "model.layers.1.ple.ple_embedding.ngram_embedding.weight"
VISUAL = "model.visual.patch_embed.proj.weight"
QSA = "model.layers.3.self_attn.indexer.index_qk_proj.weight"


@pytest.fixture
def contract():
    return Qwen4ExpFrozenContract(
        "a" * 64, frozenset({TABLE}), frozenset({VISUAL}), True, True
    )


def table():
    return nn.Parameter(torch.ones(4, 3, dtype=torch.bfloat16), requires_grad=False)


def test_wire_roundtrip_and_exact_exclusion_keep_frozen_qsa(contract):
    assert Qwen4ExpFrozenContract.from_dict(contract.to_dict()) == contract
    assert contract.excludes(TABLE, "actor")
    assert contract.excludes(VISUAL, "inference")
    assert not contract.excludes(VISUAL, "actor")
    assert not contract.excludes(QSA, "actor")
    assert not contract.excludes("model.visual.unexpected.weight", "inference")
    with pytest.raises(ValueError):
        contract.excludes(TABLE, "typo")


@pytest.mark.parametrize(
    "change",
    [
        {"freeze_ple_table": False},
        {"language_model_only": False},
        {"schema_version": 2},
        {"schema_version": True},
        {"checkpoint_manifest_sha256": "not-a-hash"},
        {"ple_table_names": frozenset({QSA})},
        {"visual_parameter_names": frozenset({"model.visual.*"})},
    ],
)
def test_invalid_declarations_rejected(contract, change):
    with pytest.raises(ValueError):
        replace(contract, **change)


@pytest.mark.parametrize("fault", ["extra", "missing", "duplicate", "string"])
def test_invalid_wire_contract_rejected(contract, fault):
    data = contract.to_dict()
    if fault == "extra":
        data["skip_unknown_parameters"] = True
    elif fault == "missing":
        del data["schema_version"]
    elif fault == "duplicate":
        data["ple_table_names"] *= 2
    else:
        data["ple_table_names"] = TABLE
    with pytest.raises(ValueError):
        Qwen4ExpFrozenContract.from_dict(data)


def test_actor_validates_original_parameters_and_local_pipeline_ownership(contract):
    parameter = table()
    contract.validate_actor_parameters({TABLE: parameter}, frozenset({TABLE}))
    contract.validate_actor_parameters({QSA: table()}, frozenset())
    with pytest.raises(TypeError, match="detached"):
        contract.validate_actor_parameters(
            {TABLE: parameter.detach()}, frozenset({TABLE})
        )
    parameter.requires_grad_(True)
    with pytest.raises(ValueError, match="trainable"):
        contract.validate_actor_parameters({TABLE: parameter}, frozenset({TABLE}))
    with pytest.raises(ValueError, match="ownership"):
        contract.validate_actor_parameters({}, frozenset({TABLE}))
    with pytest.raises(ValueError, match="visual"):
        contract.validate_actor_parameters({VISUAL: table()}, frozenset())


def test_inference_requires_exact_backup_and_exclusion_keys(contract):
    parameters = {TABLE: table(), VISUAL: table(), QSA: table()}
    contract.validate_inference_parameters(parameters, frozenset({VISUAL}))
    with pytest.raises(ValueError, match="preservation"):
        contract.validate_inference_parameters(parameters, frozenset())
    parameters["model.visual.extra.weight"] = table()
    with pytest.raises(ValueError, match="visual parameters"):
        contract.validate_inference_parameters(parameters, frozenset({VISUAL}))


@pytest.fixture
def manifest_files(tmp_path, contract):
    import hashlib
    import json

    model = tmp_path / "model"
    model.mkdir()
    config = b'{"architectures":["Qwen4ExpForConditionalGeneration"]}'
    index = b"{}"
    (model / "config.json").write_bytes(config)
    (model / "model.safetensors.index.json").write_bytes(index)
    basis = {
        "config_sha256": hashlib.sha256(config).hexdigest(),
        "weight_index_sha256": hashlib.sha256(index).hexdigest(),
        "ple_source_shards": [
            {
                "name": "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                "sha256": "b" * 64,
            }
        ],
        "visual_reference": [{"tp_rank": 0, "parameters": [{"name": VISUAL}]}],
    }
    digest = hashlib.sha256(
        json.dumps(basis, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    declared = replace(contract, checkpoint_manifest_sha256=digest)
    manifest = {"contract": declared.to_dict(), "identity_basis": basis}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, model, declared


def test_manifest_load_checks_checkpoint_identity(manifest_files):
    from areal.models.mcore.qwen4_exp_awex_contract import load_frozen_contract

    path, model, declared = manifest_files
    assert load_frozen_contract(path, model) == declared
    (model / "model.safetensors.index.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="weight index"):
        load_frozen_contract(path, model)


@pytest.mark.parametrize("fault", ["evidence", "ple", "visual", "config"])
def test_manifest_rejects_identity_or_exclusion_drift(manifest_files, fault):
    import json

    from areal.models.mcore.qwen4_exp_awex_contract import load_frozen_contract

    path, model, _ = manifest_files
    data = json.loads(path.read_text())
    if fault == "evidence":
        data["identity_basis"]["ple_source_shards"][0]["sha256"] = "c" * 64
    elif fault == "ple":
        data["contract"]["ple_table_names"] = [TABLE.replace("layers.1", "layers.2")]
    elif fault == "visual":
        data["contract"]["visual_parameter_names"] = ["model.visual.other.weight"]
    else:
        (model / "config.json").write_text("{}")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_frozen_contract(path, model)


def setup_binding(global_id=0, mapped=False, table=True):
    layer = nn.Module()
    layer.layer_number = global_id + 1
    layer.ple = nn.Module()
    layer.ple.ple_embedding = nn.Module()
    if table:
        layer.ple.ple_embedding.ngram_embedding = nn.Embedding(
            3, 2, dtype=torch.bfloat16
        )
        layer.ple.ple_embedding.ngram_embedding.weight.requires_grad_(False)
    chunk = nn.Module()
    chunk.decoder = nn.Module()
    chunk.decoder.layers = nn.ModuleList([layer])
    engine = SimpleNamespace(
        model=[chunk],
        mcore_config=SimpleNamespace(language_model_only=True, freeze_ple_table=True),
        hf_config=SimpleNamespace(architectures=["Qwen4ExpForConditionalGeneration"]),
    )
    contract = Qwen4ExpFrozenContract(
        "a" * 64,
        frozenset(
            {f"model.layers.{global_id}.ple.ple_embedding.ngram_embedding.weight"}
        ),
        frozenset({"model.visual.weight"}),
        True,
        True,
    )
    converter_cls = build_mcore_converter()
    converter = converter_cls.__new__(converter_cls)
    converter.rank_info = SimpleNamespace(pp_rank=1)
    converter._pp_stage_layer_id_map = {(1, 0): {0: global_id}} if mapped else {}
    return engine, converter, McoreFrozenBinder(engine, contract)


@pytest.mark.parametrize("global_id,mapped", [(0, False), (24, True)])
def test_binding_uses_original_parameters_and_preserves_native_numbering(
    global_id, mapped
):
    engine, converter, binder = setup_binding(global_id, mapped)
    original_map = dict(converter._pp_stage_layer_id_map)
    binder(converter)
    actual = engine.model[0].decoder.layers[0].ple.ple_embedding.ngram_embedding.weight
    assert next(iter(converter._qwen4_original_parameters.values())) is actual
    assert converter._pp_stage_layer_id_map == original_map
    replacement = nn.Parameter(torch.zeros_like(actual), requires_grad=False)
    engine.model[0].decoder.layers[
        0
    ].ple.ple_embedding.ngram_embedding.weight = replacement
    binder(converter)
    assert next(iter(converter._qwen4_original_parameters.values())) is replacement


@pytest.mark.parametrize(
    "fault", ["missing", "trainable", "map", "metadata_offset", "flags"]
)
def test_invalid_frozen_ownership_rejected(fault):
    engine, converter, binder = setup_binding(24, mapped=True, table=fault != "missing")
    if fault == "trainable":
        engine.model[0].decoder.layers[
            0
        ].ple.ple_embedding.ngram_embedding.weight.requires_grad_(True)
    elif fault == "map":
        converter._pp_stage_layer_id_map[(1, 0)][0] = 12
    elif fault == "metadata_offset":
        converter._pp_stage_layer_id_map = {}
    elif fault == "flags":
        engine.mcore_config.freeze_ple_table = False
    with pytest.raises(ValueError):
        binder(converter)


def test_production_conversion_refreshes_binding_before_detach():
    from areal.engine.awex.colocate_writer import AwexWeightPublisher

    engine, unused, binder = setup_binding()
    cls = build_mcore_converter(binder)
    converter = cls.__new__(cls)
    converter.rank_info = SimpleNamespace(pp_rank=0, pp_size=1)
    converter._pp_stage_layer_id_map = {}
    converter.hf_config = engine.hf_config
    converter.tf_config = SimpleNamespace()
    adapter = AwexWeightPublisher(engine)
    adapter._qwen4_frozen_binder = binder
    adapter._weight_converter = converter
    adapter._rank_info = converter.rank_info
    assert adapter._convert_parameters() == {}
    embedding = engine.model[0].decoder.layers[0].ple.ple_embedding.ngram_embedding
    old = embedding.weight
    embedding.weight = nn.Parameter(torch.zeros_like(old), requires_grad=False)
    assert adapter._convert_parameters() == {}
    assert next(iter(converter._qwen4_original_parameters.values())) is embedding.weight
    embedding.weight = nn.Parameter(torch.ones_like(old))
    with pytest.raises(ValueError, match="trainable"):
        adapter._convert_parameters()
    assert getattr(converter, "_qwen4_frozen_contract", None) is None


def test_qwen_publication_requires_explicit_evidence_and_keeps_other_models(
    monkeypatch,
):
    from areal.models.mcore.qwen4_exp_awex_binding import build_awex_train_info

    engine, _, _ = setup_binding()
    engine.bridge_cls = "mcore-bridge"
    monkeypatch.delenv("QWEN_AWEX_FROZEN_CONTRACT", raising=False)
    with pytest.raises(ValueError, match="QWEN_AWEX_FROZEN_CONTRACT"):
        build_awex_train_info(engine, 32)
    engine.hf_config.architectures = ["OtherModel"]
    assert build_awex_train_info(engine, 32) == {"train_world_size": 32}


@pytest.mark.parametrize("installed", [False, True])
def test_sglang_binding_requires_preservation_and_checks_live_parameters(installed):
    from areal.models.mcore.qwen4_exp_awex import build_sglang_converter
    from areal.models.mcore.qwen4_exp_awex_binding import SglangFrozenBinder
    from areal.models.mcore.qwen4_exp_frozen_state import install_static_state_hooks

    model_cls = type("Qwen4ExpForConditionalGeneration", (nn.Module,), {})
    model = model_cls()
    model.visual = nn.Linear(2, 2, bias=False)
    engine, _, _ = setup_binding()
    model.model = nn.Module()
    model.model.layers = engine.model[0].decoder.layers
    contract = Qwen4ExpFrozenContract(
        "a" * 64,
        frozenset({"model.layers.0.ple.ple_embedding.ngram_embedding.weight"}),
        frozenset({"model.visual.weight"}),
        True,
        True,
    )
    updater = SimpleNamespace(
        _export_static_state=lambda model: {},
        _import_static_state=lambda model, state: None,
    )
    if installed:
        install_static_state_hooks(updater)
    binder = SglangFrozenBinder(lambda: model, contract, updater)
    cls = build_sglang_converter(binder)
    converter = cls.__new__(cls)
    if not installed:
        with pytest.raises(ValueError, match="preservation"):
            converter.refresh_frozen_contract()
        return
    converter.refresh_frozen_contract()
    table = model.model.layers[0].ple.ple_embedding.ngram_embedding
    assert (
        converter.convert_param(
            "model.layers.0.ple.ple_embedding.ngram_embedding.weight", table.weight
        )
        == []
    )
    table.weight = nn.Parameter(torch.zeros_like(table.weight), requires_grad=False)
    converter.refresh_frozen_contract()
    assert (
        converter._qwen4_original_parameters[next(iter(contract.ple_table_names))]
        is table.weight
    )
    model.visual.register_parameter("unexpected", nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError, match="visual parameters differ"):
        converter.refresh_frozen_contract()
    assert getattr(converter, "_qwen4_frozen_contract", None) is None

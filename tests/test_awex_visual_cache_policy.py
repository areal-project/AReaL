# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.engine.awex.metadata import (
    build_awex_train_info,
    has_frozen_native_qwen_vision,
)


def _engine():
    model = torch.nn.Module()
    model.language_model = torch.nn.Linear(2, 2)
    model.vision_model = torch.nn.Module()
    model.vision_model.encoder = torch.nn.Linear(2, 2)
    model.vision_model.merger = torch.nn.Linear(2, 2)
    model.vision_model.deepstack_merger_list = torch.nn.ModuleList(
        [torch.nn.Linear(2, 2)]
    )
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"]
        ),
        mcore_config=SimpleNamespace(bridge_type="megatron-bridge"),
        model=[model],
    )


@pytest.mark.parametrize("trainable_part", [None, "encoder", "merger", "deepstack"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_visual_cache_policy_requires_every_visual_parameter_frozen(
    trainable_part, wrapped
):
    engine = _engine()
    model = engine.model[0]
    model.vision_model.requires_grad_(False)
    if trainable_part is not None:
        part = (
            model.vision_model.deepstack_merger_list
            if trainable_part == "deepstack"
            else getattr(model.vision_model, trainable_part)
        )
        part.requires_grad_(True)
    if wrapped:
        engine.model = [SimpleNamespace(module=SimpleNamespace(module=model))]

    # Language weights remain trainable in every case.
    assert all(p.requires_grad for p in model.language_model.parameters())
    assert has_frozen_native_qwen_vision(engine) is (trainable_part is None)


@pytest.mark.parametrize("missing", ["owner", "parameters", "models", "config"])
def test_visual_cache_policy_requires_positive_actor_ownership(missing):
    engine = _engine()
    engine.model[0].vision_model.requires_grad_(False)
    if missing == "owner":
        del engine.model[0].vision_model
    elif missing == "parameters":
        engine.model[0].vision_model = torch.nn.Module()
    elif missing == "models":
        engine.model = []
    else:
        del engine.mcore_config
    assert has_frozen_native_qwen_vision(engine) is False


@pytest.mark.parametrize("changed", ["bridge", "model_type", "other_chunk"])
def test_visual_cache_policy_rejects_unsupported_or_partially_trainable_models(changed):
    engine = _engine()
    engine.model[0].vision_model.requires_grad_(False)
    if changed == "bridge":
        engine.mcore_config.bridge_type = "mbridge"
    elif changed == "model_type":
        engine.hf_config.model_type = "unknown_vlm"
    else:
        engine.model += _engine().model
    assert has_frozen_native_qwen_vision(engine) is False


def test_training_metadata_publishes_verified_freezing_without_flash_import(
    monkeypatch,
):
    import builtins

    original_import = builtins.__import__

    def reject_flash_import(name, *args, **kwargs):
        if "qwen4_exp" in name:
            raise AssertionError("Native Qwen metadata must not load Flash extensions")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_flash_import)
    engine = _engine()
    engine.model[0].vision_model.requires_grad_(False)
    info = build_awex_train_info(engine, 8)
    assert info["frozen_visual_embeddings"] is True
    assert info["train_world_size"] == 8
    engine.model[0].vision_model.merger.requires_grad_(True)
    assert build_awex_train_info(engine, 8)["frozen_visual_embeddings"] is False


def test_frozen_native_visual_parameters_remain_in_awex_transfer():
    from awex.converter.mcore_converter import get_mcore_model_parameters

    qwen35 = pytest.importorskip("awex.models.qwen3_5")
    # Direct vision mappings must retain frozen Parameters, independent of the
    # language converter's layer/parallel metadata initialized by its constructor.
    converter = object.__new__(qwen35._MCORE_CONVERTER_FACTORY())
    model = torch.nn.Module()
    model.vision_model = torch.nn.Module()
    model.vision_model.patch_embed = torch.nn.Module()
    model.vision_model.patch_embed.proj = torch.nn.Linear(2, 2)
    model.vision_model.merger = torch.nn.Module()
    model.vision_model.merger.linear_fc1 = torch.nn.Linear(2, 2)
    model.requires_grad_(False)

    params = get_mcore_model_parameters(model)
    for source, target in (
        (
            "vision_model.patch_embed.proj.weight",
            "model.visual.patch_embed.proj.weight",
        ),
        (
            "vision_model.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc1.weight",
        ),
    ):
        assert params[source].requires_grad is False
        converted = converter.convert_param(source, params[source])
        assert len(converted) == 1
        assert converted[0][0] == target
        assert converted[0][1] is params[source]

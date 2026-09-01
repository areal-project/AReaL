# SPDX-License-Identifier: Apache-2.0

import types

import areal.engine.megatron_utils.mindspeed_gdn_patch as gdn_patch
from areal.engine.megatron_utils.mindspeed_gdn_patch import (
    ensure_mindspeed_gdn_conv1d,
    has_mindspeed_gdn_conv1d,
    has_mindspeed_gdn_model_classes,
)


def test_ensure_mindspeed_gdn_conv1d_binds_official_implementation(monkeypatch):
    original_conv = object()
    npu_conv = object()
    modules = {
        "megatron.core.ssm.gated_delta_net": types.SimpleNamespace(
            causal_conv1d=original_conv
        ),
        "mindspeed.core.ssm.gated_delta_net": types.SimpleNamespace(),
        "mindspeed.core.ssm.npu_causal_conv1d": types.SimpleNamespace(
            causal_conv1d=npu_conv
        ),
    }
    monkeypatch.setattr(gdn_patch.importlib, "import_module", modules.__getitem__)

    assert ensure_mindspeed_gdn_conv1d()
    assert modules["megatron.core.ssm.gated_delta_net"].causal_conv1d is npu_conv
    assert modules["mindspeed.core.ssm.gated_delta_net"].causal_conv1d is npu_conv


def test_ensure_mindspeed_gdn_conv1d_requires_official_implementation(monkeypatch):
    modules = {
        "megatron.core.ssm.gated_delta_net": types.SimpleNamespace(),
        "mindspeed.core.ssm.gated_delta_net": types.SimpleNamespace(),
    }

    def import_module(name):
        if name not in modules:
            raise ImportError(name)
        return modules[name]

    monkeypatch.setattr(gdn_patch.importlib, "import_module", import_module)

    assert not ensure_mindspeed_gdn_conv1d()


def test_has_mindspeed_gdn_conv1d_handles_missing_module_attribute():
    assert not has_mindspeed_gdn_conv1d(types.SimpleNamespace())


def test_has_mindspeed_gdn_model_classes_rejects_captured_mcore_class():
    mindspeed_class = type("GatedDeltaNet", (), {})
    mindspeed_class.__module__ = "mindspeed.core.ssm.gated_delta_net"
    mcore_class = type("GatedDeltaNet", (), {})
    mcore_class.__module__ = "megatron.core.ssm.gated_delta_net"

    assert has_mindspeed_gdn_model_classes(mindspeed_class, mindspeed_class)
    assert not has_mindspeed_gdn_model_classes(mindspeed_class, mcore_class)

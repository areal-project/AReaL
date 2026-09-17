# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _stub_module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_hf_load_module(monkeypatch):
    class _Platform:
        device_type = "cpu"

    class _Logger:
        def info(self, *args, **kwargs):
            return None

    class _FP8BlockwiseTensorHelper:
        pass

    _stub_module(monkeypatch, "mbridge")
    _stub_module(monkeypatch, "mbridge.core")
    _stub_module(monkeypatch, "mbridge.core.bridge", Bridge=type("Bridge", (), {}))
    _stub_module(monkeypatch, "megatron")
    core = _stub_module(monkeypatch, "megatron.core")
    core.parallel_state = _stub_module(monkeypatch, "megatron.core.parallel_state")
    _stub_module(
        monkeypatch,
        "megatron.core.fp8_utils",
        is_float8tensor=lambda value: False,
    )
    _stub_module(monkeypatch, "areal")
    _stub_module(monkeypatch, "areal.engine")
    _stub_module(monkeypatch, "areal.engine.core")
    _stub_module(
        monkeypatch,
        "areal.engine.core.model",
        lang_config=lambda config: config,
    )
    _stub_module(monkeypatch, "areal.engine.megatron_utils")
    _stub_module(
        monkeypatch,
        "areal.engine.megatron_utils.fp8",
        FP8BlockwiseTensorHelper=_FP8BlockwiseTensorHelper,
        dequantize_params=lambda *args, **kwargs: None,
        get_block_size_from_config=lambda *args, **kwargs: None,
    )
    _stub_module(monkeypatch, "areal.infra")
    _stub_module(
        monkeypatch,
        "areal.infra.platforms",
        current_platform=_Platform(),
    )
    _stub_module(monkeypatch, "areal.models")
    _stub_module(monkeypatch, "areal.models.mcore")
    _stub_module(
        monkeypatch,
        "areal.models.mcore.registry",
        unwrap_to_gpt_model=lambda model: model,
    )
    _stub_module(monkeypatch, "areal.utils")
    _stub_module(
        monkeypatch,
        "areal.utils.logging",
        getLogger=lambda name: _Logger(),
    )

    path = Path(__file__).resolve().parents[1] / "areal/models/mcore/hf_load.py"
    spec = importlib.util.spec_from_file_location("_test_bailing_v3_hf_load", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "weight_name",
    [
        "decoder.layers.0.self_attention.in_proj.weight",
        "decoder.layers.0.self_attention.conv1d.weight",
    ],
)
def test_kda_fused_dispatch_is_scoped_to_bailing_v3(monkeypatch, weight_name):
    hf_load = _load_hf_load_module(monkeypatch)
    kda_result = torch.tensor([1.0])
    generic_result = torch.tensor([2.0])
    monkeypatch.setattr(hf_load, "_merge_kda_fused_weight", lambda *args: kda_result)
    monkeypatch.setattr(hf_load, "_slice_generic_weight", lambda *args: generic_result)

    gdn_config = SimpleNamespace(architectures=["Qwen3_5ForCausalLM"])
    result = hf_load._weight_to_mcore_tp(
        hf_config=gdn_config,
        mcore_weights_name=weight_name,
        mcore_param_shape=[1],
        hf_weights_safe_slice=[torch.ones(1)],
        tp_rank=0,
        tp_size=1,
    )

    assert result is generic_result


@pytest.mark.parametrize(
    "weight_name",
    [
        "decoder.layers.0.self_attention.in_proj.weight",
        "decoder.layers.0.self_attention.conv1d.weight",
    ],
)
def test_kda_fused_dispatch_handles_bailing_v3(monkeypatch, weight_name):
    hf_load = _load_hf_load_module(monkeypatch)
    kda_result = torch.tensor([1.0])
    generic_result = torch.tensor([2.0])
    monkeypatch.setattr(hf_load, "_merge_kda_fused_weight", lambda *args: kda_result)
    monkeypatch.setattr(hf_load, "_slice_generic_weight", lambda *args: generic_result)

    bailing_config = SimpleNamespace(architectures=["BailingMoeV3ForCausalLM"])
    result = hf_load._weight_to_mcore_tp(
        hf_config=bailing_config,
        mcore_weights_name=weight_name,
        mcore_param_shape=[1],
        hf_weights_safe_slice=[torch.ones(1)],
        tp_rank=0,
        tp_size=1,
    )

    assert result is kda_result

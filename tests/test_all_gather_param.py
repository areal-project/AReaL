# SPDX-License-Identifier: Apache-2.0

"""Routing tests for ``all_gather_param``.

These verify that a parameter is treated as *replicated* (returned as-is,
no TP all-gather) exactly when it should be:
- ``tensor_model_parallel`` is False (what ``_mark_duplicated_params`` sets on
  duplicated params), or
- its name is listed in ``duplicated_param_names``.

and is actually all-gathered only when it is a genuine TP-sharded param.

The module imports ``megatron.core`` at import time, so this test runs in the
GPU/megatron CI env (like tests/test_megatron_engine.py). The gather branch is
stubbed, so no process group is required.
"""

import pytest
import torch

megatron = pytest.importorskip("areal.engine.megatron_utils.megatron")


def _make_param(tensor_model_parallel):
    """A plain Linear weight with the attributes all_gather_param reads."""
    weight = torch.nn.Linear(4, 4, bias=False).weight
    if tensor_model_parallel is not None:
        weight.tensor_model_parallel = tensor_model_parallel
    # Only read on the gather path, but harmless to always set.
    weight.partition_dim = 0
    weight.partition_stride = 1
    return weight


def test_returns_param_data_when_tensor_model_parallel_false(monkeypatch):
    calls = []
    monkeypatch.setattr(
        megatron,
        "_all_gather_and_concat",
        lambda *a, **k: calls.append(a) or torch.empty(0),
    )
    param = _make_param(tensor_model_parallel=False)

    out = megatron.all_gather_param("decoder.layers.0.mlp.dense.weight", param)

    assert out is param.data
    assert calls == []  # never entered the all-gather path


def test_returns_param_data_when_name_in_duplicated_set(monkeypatch):
    calls = []
    monkeypatch.setattr(
        megatron,
        "_all_gather_and_concat",
        lambda *a, **k: calls.append(a) or torch.empty(0),
    )
    # Even mis-marked as TP (TE default), the name set forces replicated handling.
    param = _make_param(tensor_model_parallel=True)
    name = "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight"

    out = megatron.all_gather_param(name, param, duplicated_param_names={name})

    assert out is param.data
    assert calls == []


def test_all_gathers_genuine_tp_sharded_param(monkeypatch):
    sentinel = torch.arange(8)
    calls = []

    def fake_gather(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(megatron, "_all_gather_and_concat", fake_gather)
    monkeypatch.setattr(megatron.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        megatron.mpu, "get_tensor_model_parallel_group", lambda: "tp_group"
    )
    param = _make_param(tensor_model_parallel=True)

    out = megatron.all_gather_param("decoder.layers.0.mlp.dense.weight", param)

    assert out is sentinel
    assert len(calls) == 1
    # (data, tp_size, tp_group, partition_dim, partition_stride, name)
    args = calls[0][0]
    assert args[1] == 2
    assert args[2] == "tp_group"

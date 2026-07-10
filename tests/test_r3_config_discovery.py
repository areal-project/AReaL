# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from areal.engine.r3.asserts import R3Error
from areal.engine.r3.config import resolve_r3_moe_config
from areal.engine.r3.discovery import discover_native_router_replay


def test_resolve_r3_moe_config_handles_dense_prefix_and_freq_list():
    cfg = SimpleNamespace(
        num_layers=6,
        moe_layer_freq=[1, 1, 0, 1, 0, 1],
        first_k_dense_replace=2,
        moe_router_topk=4,
    )

    resolved = resolve_r3_moe_config(cfg)

    assert resolved.num_moe_layers == 2
    assert resolved.moe_layer_indices == (3, 5)
    assert resolved.global_to_moe_index == {3: 0, 5: 1}
    assert resolved.topk == 4


def test_resolve_r3_moe_config_accepts_hf_num_hidden_layers():
    cfg = SimpleNamespace(
        num_hidden_layers=4,
        first_k_dense_replace=1,
        num_experts_per_tok=2,
    )

    resolved = resolve_r3_moe_config(cfg)

    assert resolved.num_layers == 4
    assert resolved.moe_layer_indices == (1, 2, 3)
    assert resolved.num_moe_layers == 3
    assert resolved.topk == 2


def test_resolve_r3_moe_config_rejects_flat_dim_guessing():
    cfg = SimpleNamespace(num_layers=2, moe_layer_freq=1)

    with pytest.raises(R3Error, match="Unable to resolve MoE router topk"):
        resolve_r3_moe_config(cfg)


def test_discover_native_router_replay_collects_instance_refs(monkeypatch):
    class FakeRouter(torch.nn.Module):
        def __init__(self, replay):
            super().__init__()
            self.router_replay = replay

    class FakeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = FakeRouter(replay=object())

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([FakeLayer(), FakeLayer()])
    monkeypatch.setattr(
        "areal.engine.r3.discovery._resolve_topk_router_type",
        lambda: FakeRouter,
    )

    grouped = discover_native_router_replay(model)

    refs = grouped[0]
    assert len(refs) == 2
    assert [ref.layer_number for ref in refs] == [0, 1]
    assert [ref.layer_number_is_global for ref in refs] == [False, False]
    assert refs[0].router_replay is model.layers[0].router.router_replay
    assert refs[1].router_replay is model.layers[1].router.router_replay


def test_discover_native_router_replay_normalizes_router_layer_number(monkeypatch):
    class FakeRouter(torch.nn.Module):
        def __init__(self, layer_number: int):
            super().__init__()
            self.router_replay = object()
            self.layer_number = layer_number

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([FakeRouter(3)])
    monkeypatch.setattr(
        "areal.engine.r3.discovery._resolve_topk_router_type",
        lambda: FakeRouter,
    )

    refs = discover_native_router_replay(model)[0]

    assert refs[0].layer_number == 2
    assert refs[0].layer_number_is_global


def test_discover_native_router_replay_missing_native_replay_raises(monkeypatch):
    class FakeRouter(torch.nn.Module):
        router_replay = None

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([FakeRouter()])
    monkeypatch.setattr(
        "areal.engine.r3.discovery._resolve_topk_router_type",
        lambda: FakeRouter,
    )

    with pytest.raises(R3Error, match="missing native router_replay"):
        discover_native_router_replay(model)

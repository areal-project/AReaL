# SPDX-License-Identifier: Apache-2.0
"""The bridge serves an artifact only for a registered version pair with a matching layout, in pool layout."""

import pathlib

import pytest
import torch

from areal.experimental.kvmap.artifact import identity_artifact
from areal.experimental.kvmap.sglang_bridge import create_mapper_bridge
from tests.experimental.kvmap.conftest import HEAD_DIM, KV_HEADS, LAYERS, SOURCE_SHA, TARGET_SHA, identity

LAYOUT = {"num_layers": LAYERS, "num_kv_heads": KV_HEADS, "head_dim": HEAD_DIM, "rope_theta": 1_000_000.0}


def _registry(tmp_path: pathlib.Path) -> pathlib.Path:
    registry = tmp_path / "registry"
    registry.mkdir()
    identity_artifact(source=identity(SOURCE_SHA), target=identity(TARGET_SHA), provenance={}).save(registry / "v3-v4")
    return registry


def test_resolve_finds_only_the_registered_pair_and_caches_it(tmp_path: pathlib.Path):
    bridge = create_mapper_bridge(registry=str(_registry(tmp_path)), model_layout=LAYOUT)
    handle = bridge.resolve(3, 4)
    assert handle is not None and handle.source_version == 3
    assert bridge.resolve(3, 4) is handle
    assert bridge.resolve(4, 5) is None


def test_translate_keeps_pool_layout_and_identity_values(tmp_path: pathlib.Path):
    bridge = create_mapper_bridge(registry=str(_registry(tmp_path)), model_layout=LAYOUT)
    handle = bridge.resolve(3, 4)
    tokens = 5
    keys = [torch.randn(tokens, KV_HEADS, HEAD_DIM) for _ in range(LAYERS)]
    values = [torch.randn(tokens, KV_HEADS, HEAD_DIM) for _ in range(LAYERS)]

    new_keys, new_values = bridge.translate(handle, keys=keys, values=values, positions=torch.arange(tokens))

    assert [k.shape for k in new_keys] == [(tokens, KV_HEADS, HEAD_DIM)] * LAYERS
    torch.testing.assert_close(new_keys[0], keys[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(new_values[1], values[1], rtol=0.0, atol=0.0)


def test_layout_mismatch_is_rejected(tmp_path: pathlib.Path):
    bridge = create_mapper_bridge(registry=str(_registry(tmp_path)), model_layout={**LAYOUT, "num_kv_heads": KV_HEADS + 1})
    with pytest.raises(ValueError, match="num_kv_heads is 2 but the server KV pool has 3"):
        bridge.resolve(3, 4)


def test_missing_registry_directory_is_an_error(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError, match="not a directory"):
        create_mapper_bridge(registry=str(tmp_path / "nope"), model_layout=LAYOUT)

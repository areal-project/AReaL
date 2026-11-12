# SPDX-License-Identifier: Apache-2.0

"""CPU round trips for frozen MTP exports, without importing Megatron/CUDA."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

_SPEC = importlib.util.spec_from_file_location(
    "mtp_checkpoint", Path(__file__).parents[1] / "areal/utils/mtp_checkpoint.py"
)
_HELPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPER)
restore = _HELPER.restore_frozen_mtp_weights


@pytest.fixture
def checkpoint(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
        },
    }
    (source / "config.json").write_text(json.dumps(config))
    (output / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"mtp_num_hidden_layers": 0},
                "trained": True,
            }
        )
    )
    original = {
        "mtp.fc.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "mtp.layers.0.mlp.up_proj.weight": torch.ones(4, 3, dtype=torch.bfloat16),
        "lm_head.weight": torch.zeros(3, 4),
        "model.language_model.embed_tokens.weight": torch.zeros(4, 3),
    }
    # MTP and shared tensors intentionally occupy the same source shard.
    save_file(original, source / "model-00001-of-00001.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    key: "model-00001-of-00001.safetensors" for key in original
                }
            }
        )
    )
    return source, output, original


def read_export(output):
    index = json.loads((output / "model.safetensors.index.json").read_text())
    tensors = {}
    for filename in set(index["weight_map"].values()):
        shard = load_file(output / filename)
        assert not tensors.keys() & shard.keys()
        tensors.update(shard)
        assert all(index["weight_map"][key] == filename for key in shard)
    assert index["metadata"]["total_size"] == sum(
        t.numel() * t.element_size() for t in tensors.values()
    )
    assert set(index["weight_map"]) == tensors.keys()
    return tensors


@pytest.mark.parametrize("single", [True, False])
def test_frozen_mtp_export_preserves_trained_shared_weights(
    checkpoint, single, monkeypatch
):
    source, output, original = checkpoint
    # Force multi-chunk copies even for these small tensors.
    monkeypatch.setattr(_HELPER, "_COPY_CHUNK_BYTES", 7)
    trained = {
        key: value + 10 for key, value in original.items() if not key.startswith("mtp.")
    }
    filename = "model.safetensors" if single else "model-00001-of-00001.safetensors"
    save_file(trained, output / filename)
    # Reproduce Bridge's stale index claiming absent MTP entries.
    (output / "model.safetensors.index.json").write_text(
        (source / "model.safetensors.index.json").read_text()
    )
    source_before = (source / "model-00001-of-00001.safetensors").read_bytes()

    assert restore(str(source), str(output))
    actual = read_export(output)
    for key, value in original.items():
        torch.testing.assert_close(
            actual[key],
            value if key.startswith("mtp.") else trained[key],
            rtol=0,
            atol=0,
        )
    config = json.loads((output / "config.json").read_text())
    assert config["text_config"]["mtp_num_hidden_layers"] == 1
    assert config["text_config"]["mtp_use_dedicated_embeddings"] is False
    assert config["trained"] is True
    assert not (output / "model.safetensors").exists()
    assert (source / "model-00001-of-00001.safetensors").read_bytes() == source_before

    # Idempotence and a real subsequent Bridge save with changed backbone values.
    assert restore(str(source), str(output))
    new_trained = {key: value + 5 for key, value in trained.items()}
    save_file(new_trained, output / filename)
    assert restore(str(source), str(output))
    for key, value in read_export(output).items():
        torch.testing.assert_close(
            value,
            original[key] if key.startswith("mtp.") else new_trained[key],
            rtol=0,
            atol=0,
        )


def test_resumed_hf_export_replaces_empty_mtp_shard(checkpoint, tmp_path):
    source, first_export, original = checkpoint
    trained = {
        key: value + 10 for key, value in original.items() if not key.startswith("mtp.")
    }
    save_file(trained, first_export / "model.safetensors")
    assert restore(str(source), str(first_export))

    resumed_export = tmp_path / "resumed"
    resumed_export.mkdir()
    for filename in ("config.json", "model.safetensors.index.json"):
        (resumed_export / filename).write_bytes((first_export / filename).read_bytes())
    # Bridge preserves source shard names with strict=False and writes an
    # empty dictionary for source shards whose tensors were all dropped.
    for shard in first_export.glob("*.safetensors"):
        backbone = {
            key: value + 5
            for key, value in load_file(shard).items()
            if not key.startswith("mtp.")
        }
        save_file(backbone, resumed_export / shard.name)
    assert load_file(resumed_export / "model-frozen-mtp.safetensors") == {}

    assert restore(str(first_export), str(resumed_export))
    actual = read_export(resumed_export)
    for key, value in original.items():
        torch.testing.assert_close(
            actual[key],
            value if key.startswith("mtp.") else trained[key] + 5,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    "failure",
    [
        "missing-source",
        "missing-indexed-tensor",
        "partial-output",
        "conflict",
        "empty-backbone",
        "truncated",
        "reserved-shard-conflict",
    ],
)
def test_invalid_mtp_export_fails_before_publishing(checkpoint, failure):
    source, output, original = checkpoint
    save_file(
        {"lm_head.weight": original["lm_head.weight"]}, output / "model.safetensors"
    )
    if failure == "missing-source":
        save_file(
            {"lm_head.weight": original["lm_head.weight"]},
            source / "model-00001-of-00001.safetensors",
        )
    elif failure == "missing-indexed-tensor":
        index_path = source / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["weight_map"]["mtp.missing.weight"] = "absent.safetensors"
        index_path.write_text(json.dumps(index))
    elif failure == "partial-output":
        save_file(
            {"mtp.fc.weight": original["mtp.fc.weight"]}, output / "partial.safetensors"
        )
    elif failure == "conflict":
        save_file(
            {"unrelated.weight": torch.zeros(1)}, output / "model-backbone.safetensors"
        )
    elif failure == "empty-backbone":
        (output / "model.safetensors").unlink()
    elif failure == "truncated":
        shard = source / "model-00001-of-00001.safetensors"
        shard.write_bytes(shard.read_bytes()[:-4])
    elif failure == "reserved-shard-conflict":
        save_file(
            {"unrelated.weight": torch.zeros(1)},
            output / "model-frozen-mtp.safetensors",
        )
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError):
        restore(str(source), str(output))
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_unsupported_model_retains_existing_policy(checkpoint):
    source, output, _ = checkpoint
    (source / "config.json").write_text('{"model_type": "other_model"}')
    assert restore(str(source), str(output)) is False
    assert not (output / "model-frozen-mtp.safetensors").exists()


def test_same_directory_rejected(checkpoint):
    source, _, _ = checkpoint
    with pytest.raises(ValueError, match="must differ"):
        restore(str(source), str(source))


def engine_save_method():
    # Execute the real method body with CPU bridge doubles; Megatron itself
    # requires CUDA dependencies unavailable in the unit-test environment.
    import ast
    import os
    from typing import Any
    from unittest.mock import Mock

    engine_file = Path(__file__).parents[1] / "areal/engine/megatron_engine.py"
    tree = ast.parse(engine_file.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_save_model_to_hf"
    )
    namespace = {"Any": Any, "os": os, "dist": Mock(), "current_platform": Mock()}
    namespace["dist"].get_rank.return_value = 0
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(engine_file), "exec"),
        namespace,
    )
    return namespace["_save_model_to_hf"]


@pytest.mark.parametrize("mode", ["dropped", "enabled", "lora", "same-source"])
def test_engine_export_routes_frozen_mtp_only_when_dropped(
    checkpoint, monkeypatch, mode
):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    source, output, original = checkpoint
    monkeypatch.setitem(sys.modules, "areal.utils.mtp_checkpoint", _HELPER)
    bridge = Mock()
    bridge.hf_pretrained.state.source.path = source

    def save_backbone(model, path, **kwargs):
        save_file(
            {"lm_head.weight": original["lm_head.weight"] + 42},
            Path(path) / "model.safetensors",
        )

    bridge.save_hf_pretrained.side_effect = save_backbone
    engine = SimpleNamespace(
        model=object(),
        bridge_cls="megatron-bridge",
        bridge=bridge,
        config=SimpleNamespace(
            is_critic=False, use_lora=mode == "lora", path=str(source)
        ),
        _mtp_head_dropped=mode != "enabled",
        bridge_lora=object(),
        cpu_group=object(),
        _scrub_mtp_from_saved_config=Mock(),
        _rebuild_index_from_saved_shards=Mock(),
    )
    save = engine_save_method()
    if mode == "same-source":
        with pytest.raises(ValueError, match="must not overwrite"):
            save(engine, str(source))
        bridge.save_hf_pretrained.assert_not_called()
        return
    save(engine, str(output))
    if mode == "dropped":
        actual = read_export(output)
        torch.testing.assert_close(
            actual["mtp.fc.weight"], original["mtp.fc.weight"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual["lm_head.weight"], original["lm_head.weight"] + 42, rtol=0, atol=0
        )
        assert bridge.save_hf_pretrained.call_args.kwargs["strict"] is False
    else:
        assert not (output / "model-frozen-mtp.safetensors").exists()
    if mode == "lora":
        bridge.save_hf_adapter.assert_called_once()
        bridge.save_hf_pretrained.assert_not_called()
    engine._scrub_mtp_from_saved_config.assert_not_called()

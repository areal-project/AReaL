# SPDX-License-Identifier: Apache-2.0
"""Recovery state fingerprints distinguish optimizer and batch changes."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from examples.mtp.qwen38_flash_next.recovery_checks import (
    assert_state_close,
    cpu_state,
    frozen_model_state,
    snapshot_training_state,
    state_digest,
)
from examples.mtp.qwen38_flash_next.verify_recovery_export import _verify_frozen_export


def test_optimizer_moment_change_changes_state_digest():
    before = {"state": {0: {"exp_avg": torch.tensor([0.5]), "step": 4}}}
    after = cpu_state(before)
    after["state"][0]["exp_avg"].add_(0.25)
    assert state_digest(before) != state_digest(after)


def test_batch_reordering_changes_state_digest():
    first = {"input_ids": torch.tensor([[1, 2], [3, 4]]), "loss_mask": torch.ones(2, 2)}
    other = {"input_ids": first["input_ids"].flip(0), "loss_mask": first["loss_mask"]}
    assert state_digest(first) != state_digest(other)


def test_scalar_bfloat16_and_dictionary_order_have_stable_digest():
    left = {"weight": torch.tensor(0.5, dtype=torch.bfloat16), "step": 4}
    right = {"step": 4, "weight": left["weight"].clone()}
    assert state_digest(left) == state_digest(right)


def _model():
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(2, 2, bias=False)
    model.mtp = torch.nn.Linear(2, 2, bias=False)
    return model


def test_recovery_boundary_rejects_corrupted_frozen_backbone(tmp_path, monkeypatch):
    model = _model()
    actor = SimpleNamespace(
        model=[model],
        optimizer=SimpleNamespace(optimizer=None),
        lr_scheduler=SimpleNamespace(state_dict=lambda: {"step": 4}),
        checkpointer=SimpleNamespace(
            get_rng_state=lambda **_: {"rng": torch.tensor([1])}
        ),
    )
    monkeypatch.setenv("RECOVERY_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    snapshot_training_state(actor, "saved")
    with torch.no_grad():
        model.backbone.weight[0, 0].add_(1)
    snapshot_training_state(actor, "restored")
    saved = torch.load(tmp_path / "saved.rank-00.pt", weights_only=True)
    restored = torch.load(tmp_path / "restored.rank-00.pt", weights_only=True)
    with pytest.raises(AssertionError, match="frozen_parameters.*sha256"):
        assert_state_close(saved, restored, atol=0, rtol=0)


def test_frozen_manifest_excludes_mtp_updates_and_tracks_shared_aliases():
    model = _model()
    model.mtp.shared_backbone = model.backbone.weight
    before = frozen_model_state([model])
    with torch.no_grad():
        model.mtp.weight.add_(1)
    assert frozen_model_state([model]) == before
    aliases = before["parameters"]["chunk_0.backbone.weight"]["aliases"]
    assert aliases == ["chunk_0.backbone.weight", "chunk_0.mtp.shared_backbone"]
    with torch.no_grad():
        model.mtp.shared_backbone.add_(1)
    with pytest.raises(AssertionError, match="sha256"):
        assert_state_close(before, frozen_model_state([model]), atol=0, rtol=0)


def test_frozen_manifest_rejects_changed_ple_hash_buffer():
    model = _model()
    model.ple = torch.nn.Module()
    model.ple.register_buffer("layer_multipliers", torch.tensor([3, 5]))
    before = frozen_model_state([model])
    model.ple.layer_multipliers[0] += 1
    with pytest.raises(AssertionError, match="ple_buffers.*sha256"):
        assert_state_close(before, frozen_model_state([model]), atol=0, rtol=0)


def test_full_frozen_export_rejects_ordinary_backbone_corruption(tmp_path):
    backbone_key = "model.language_model.layers.0.mlp.experts.0.up_proj.weight"
    mtp_key = "mtp.layers.0.weight"
    values = {backbone_key: torch.ones(2, 2), mtp_key: torch.zeros(2, 2)}
    source, exported = tmp_path / "source", tmp_path / "exported"
    for root in (source, exported):
        root.mkdir()
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors" for key in values}})
        )
        save_file(values, root / "model.safetensors")
    values[mtp_key].add_(1)
    save_file(values, exported / "model.safetensors")
    result = _verify_frozen_export(source, exported, {mtp_key})
    assert result["keys"] == 1
    values[backbone_key][0, 0].add_(1)
    save_file(values, exported / "model.safetensors")
    with pytest.raises(AssertionError, match="Frozen non-MTP HF export changed"):
        _verify_frozen_export(source, exported, {mtp_key})

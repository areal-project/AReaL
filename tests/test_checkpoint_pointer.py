import os

import pytest

from areal.utils import checkpoint_pointer as cp


def make_generation(root: str, step: int, engines=("default",)) -> str:
    generation = cp.generation_dir(root, step)
    os.makedirs(cp.manifest_dir(generation))
    for name in engines:
        os.makedirs(cp.payload_dir(generation, name))
    return generation


def test_publish_latest_atomically_selects_generation(tmp_path):
    root = str(tmp_path)
    make_generation(root, 4)
    record = cp.make_record(4, ["default"])

    cp.publish_latest(root, record.to_json())

    assert cp.read_latest(root) == record
    source = cp.resolve_checkpoint(root, ["default"])
    assert source.manifest == cp.manifest_dir(cp.generation_dir(root, 4))
    assert source.transactional is True


def test_publish_latest_removes_previous_generation_after_switch(tmp_path):
    root = str(tmp_path)
    old_generation = make_generation(root, 4)
    cp.publish_latest(root, cp.make_record(4, ["default"]).to_json())
    new_generation = make_generation(root, 7)

    cp.publish_latest(root, cp.make_record(7, ["default"]).to_json())

    assert not os.path.exists(old_generation)
    assert os.path.isdir(new_generation)
    assert cp.read_latest(root).global_step == 7


def test_partial_newer_generation_does_not_change_recovery_source(tmp_path):
    root = str(tmp_path)
    make_generation(root, 4)
    cp.publish_latest(root, cp.make_record(4, ["default"]).to_json())
    make_generation(root, 7)

    source = cp.resolve_checkpoint(root, ["default"])

    assert source.payloads["default"] == cp.payload_dir(
        cp.generation_dir(root, 4), "default"
    )


def test_resolve_checkpoint_rejects_engine_set_mismatch(tmp_path):
    root = str(tmp_path)
    make_generation(root, 4, engines=("actor", "critic"))
    cp.publish_latest(root, cp.make_record(4, ["actor", "critic"]).to_json())

    with pytest.raises(cp.CheckpointConsistencyError, match="requires"):
        cp.resolve_checkpoint(root, ["actor"])


def test_publish_latest_rejects_missing_payload_directory(tmp_path):
    root = str(tmp_path)
    generation = cp.generation_dir(root, 4)
    os.makedirs(cp.manifest_dir(generation))

    with pytest.raises(cp.CheckpointConsistencyError, match="missing paths"):
        cp.publish_latest(root, cp.make_record(4, ["default"]).to_json())

    assert not os.path.exists(cp.latest_path(root))


def test_invalid_latest_does_not_fall_back_to_legacy_checkpoint(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, cp.LEGACY_MANIFEST_DIRNAME))
    os.makedirs(os.path.join(root, "default", cp.LEGACY_PAYLOAD_DIRNAME))
    with open(cp.latest_path(root), "w") as f:
        f.write("not json")

    with pytest.raises(cp.CheckpointConsistencyError, match="Invalid recovery"):
        cp.resolve_checkpoint(root, ["default"])


def test_resolve_checkpoint_accepts_pre_pointer_layout(tmp_path):
    root = str(tmp_path)
    manifest = os.path.join(root, cp.LEGACY_MANIFEST_DIRNAME)
    payload = os.path.join(root, "default", cp.LEGACY_PAYLOAD_DIRNAME)
    os.makedirs(manifest)
    os.makedirs(payload)

    source = cp.resolve_checkpoint(root, ["default"])

    assert source.manifest == manifest
    assert source.payloads == {"default": payload}
    assert source.transactional is False


def test_prepare_generation_replaces_only_unpublished_same_step(tmp_path):
    root = str(tmp_path)
    stale = make_generation(root, 7)
    stale_file = os.path.join(stale, "partial")
    open(stale_file, "w").close()

    generation, record = cp.prepare_generation(root, 7, ["default"])

    assert generation == stale
    assert record.global_step == 7
    assert not os.path.exists(stale_file)


def test_prepare_generation_refuses_to_overwrite_published_step(tmp_path):
    root = str(tmp_path)
    make_generation(root, 7)
    cp.publish_latest(root, cp.make_record(7, ["default"]).to_json())

    with pytest.raises(cp.CheckpointConsistencyError, match="already points"):
        cp.prepare_generation(root, 7, ["default"])


def test_publish_latest_refuses_to_move_pointer_backwards(tmp_path):
    root = str(tmp_path)
    make_generation(root, 4)
    make_generation(root, 7)
    cp.publish_latest(root, cp.make_record(7, ["default"]).to_json())

    with pytest.raises(cp.CheckpointConsistencyError, match="backwards"):
        cp.publish_latest(root, cp.make_record(4, ["default"]).to_json())

    assert cp.read_latest(root).global_step == 7

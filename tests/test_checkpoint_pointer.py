import os

import pytest

from areal.utils import checkpoint_pointer


def make_generation(root: str, step: int, engines=("default",)) -> str:
    generation = checkpoint_pointer.generation_dir(root, step)
    os.makedirs(checkpoint_pointer.manifest_dir(generation))
    for name in engines:
        os.makedirs(checkpoint_pointer.payload_dir(generation, name))
    return generation


def test_publish_latest_atomically_selects_generation(tmp_path):
    root = str(tmp_path)
    make_generation(root, 4)
    record = checkpoint_pointer.make_record(4, ["default"])

    checkpoint_pointer.publish_latest(root, record.to_json())

    assert checkpoint_pointer.read_latest(root) == record
    source = checkpoint_pointer.resolve_checkpoint(root, ["default"])
    assert source.manifest == checkpoint_pointer.manifest_dir(
        checkpoint_pointer.generation_dir(root, 4)
    )
    assert source.transactional is True


def test_publish_latest_removes_previous_generation_after_switch(tmp_path):
    root = str(tmp_path)
    old_generation = make_generation(root, 4)
    checkpoint_pointer.publish_latest(
        root, checkpoint_pointer.make_record(4, ["default"]).to_json()
    )
    new_generation = make_generation(root, 7)

    checkpoint_pointer.publish_latest(
        root, checkpoint_pointer.make_record(7, ["default"]).to_json()
    )

    assert not os.path.exists(old_generation)
    assert os.path.isdir(new_generation)
    assert checkpoint_pointer.read_latest(root).global_step == 7


def test_partial_newer_generation_does_not_change_recovery_source(tmp_path):
    root = str(tmp_path)
    make_generation(root, 4)
    checkpoint_pointer.publish_latest(
        root, checkpoint_pointer.make_record(4, ["default"]).to_json()
    )
    make_generation(root, 7)

    source = checkpoint_pointer.resolve_checkpoint(root, ["default"])

    assert source.payloads["default"] == checkpoint_pointer.payload_dir(
        checkpoint_pointer.generation_dir(root, 4), "default"
    )


def test_resolve_checkpoint_rejects_engine_set_mismatch(tmp_path):
    root = str(tmp_path)
    make_generation(root, 4, engines=("actor", "critic"))
    checkpoint_pointer.publish_latest(
        root,
        checkpoint_pointer.make_record(4, ["actor", "critic"]).to_json(),
    )

    with pytest.raises(checkpoint_pointer.CheckpointConsistencyError, match="requires"):
        checkpoint_pointer.resolve_checkpoint(root, ["actor"])


def test_publish_latest_rejects_missing_payload_directory(tmp_path):
    root = str(tmp_path)
    generation = checkpoint_pointer.generation_dir(root, 4)
    os.makedirs(checkpoint_pointer.manifest_dir(generation))

    with pytest.raises(
        checkpoint_pointer.CheckpointConsistencyError, match="missing paths"
    ):
        checkpoint_pointer.publish_latest(
            root, checkpoint_pointer.make_record(4, ["default"]).to_json()
        )

    assert not os.path.exists(checkpoint_pointer.latest_path(root))


def test_invalid_latest_does_not_fall_back_to_legacy_checkpoint(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, checkpoint_pointer.LEGACY_MANIFEST_DIRNAME))
    os.makedirs(
        os.path.join(root, "default", checkpoint_pointer.LEGACY_PAYLOAD_DIRNAME)
    )
    with open(checkpoint_pointer.latest_path(root), "w") as f:
        f.write("not json")

    with pytest.raises(
        checkpoint_pointer.CheckpointConsistencyError, match="Invalid recovery"
    ):
        checkpoint_pointer.resolve_checkpoint(root, ["default"])


def test_resolve_checkpoint_accepts_pre_pointer_layout(tmp_path):
    root = str(tmp_path)
    manifest = os.path.join(root, checkpoint_pointer.LEGACY_MANIFEST_DIRNAME)
    payload = os.path.join(root, "default", checkpoint_pointer.LEGACY_PAYLOAD_DIRNAME)
    os.makedirs(manifest)
    os.makedirs(payload)

    source = checkpoint_pointer.resolve_checkpoint(root, ["default"])

    assert source.manifest == manifest
    assert source.payloads == {"default": payload}
    assert source.transactional is False


def test_prepare_generation_replaces_only_unpublished_same_step(tmp_path):
    root = str(tmp_path)
    stale = make_generation(root, 7)
    stale_file = os.path.join(stale, "partial")
    open(stale_file, "w").close()

    generation, record = checkpoint_pointer.prepare_generation(root, 7, ["default"])

    assert generation == stale
    assert record.global_step == 7
    assert not os.path.exists(stale_file)


def test_prepare_generation_refuses_to_overwrite_published_step(tmp_path):
    root = str(tmp_path)
    make_generation(root, 7)
    checkpoint_pointer.publish_latest(
        root, checkpoint_pointer.make_record(7, ["default"]).to_json()
    )

    with pytest.raises(
        checkpoint_pointer.CheckpointConsistencyError, match="already points"
    ):
        checkpoint_pointer.prepare_generation(root, 7, ["default"])


def test_publish_latest_refuses_to_move_pointer_backwards(tmp_path):
    root = str(tmp_path)
    make_generation(root, 4)
    make_generation(root, 7)
    checkpoint_pointer.publish_latest(
        root, checkpoint_pointer.make_record(7, ["default"]).to_json()
    )

    with pytest.raises(
        checkpoint_pointer.CheckpointConsistencyError, match="backwards"
    ):
        checkpoint_pointer.publish_latest(
            root, checkpoint_pointer.make_record(4, ["default"]).to_json()
        )

    assert checkpoint_pointer.read_latest(root).global_step == 7

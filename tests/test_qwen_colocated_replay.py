# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from examples.swe.qwen38_flash_next.batch_snapshot import (
    load_diagnostic_replay,
    replay_image_storage_stats,
    replay_training_batches,
    split_replay_image_aliases,
)
from examples.swe.qwen38_flash_next.train_rl import replay_only_workflow

from areal.api.workflow_api import RolloutWorkflow


def test_replay_workflow_disables_proxy_and_fails_if_generation_starts() -> None:
    workflow_type = replay_only_workflow()
    assert issubclass(workflow_type, RolloutWorkflow)
    with pytest.raises(RuntimeError, match="must not generate rollouts"):
        asyncio.run(workflow_type().arun_episode(None, {}))


def test_replay_supplies_prepared_batch_without_collecting(tmp_path: Path) -> None:
    snapshot = tmp_path / "prepare_batch-0000.output.pt"
    torch.save(
        {
            "schema_version": 2,
            "metadata": {
                "method": "prepare_batch",
                "call_index": 0,
                "model_path": "model",
                "allocation_mode": "shared",
                "n_samples": 12,
            },
            "batch": [{"rewards": torch.tensor([1.0])}],
        },
        snapshot,
    )
    actor = Mock()
    original = actor.prepare_batch
    expected = {"model_path": "model", "allocation_mode": "shared", "n_samples": 12}
    with replay_training_batches(actor, [snapshot], expected):
        batch = actor.prepare_batch(None)
        torch.testing.assert_close(batch[0]["rewards"], torch.tensor([1.0]))
        with pytest.raises(RuntimeError, match="already consumed"):
            actor.prepare_batch(None)
    original.assert_not_called()


def test_replay_rejects_different_allocation(tmp_path: Path) -> None:
    snapshot = tmp_path / "prepare_batch-0000.output.pt"
    torch.save(
        {
            "schema_version": 2,
            "metadata": {
                "method": "prepare_batch",
                "call_index": 0,
                "allocation_mode": "other",
            },
            "batch": [{"rewards": torch.tensor([1.0])}],
        },
        snapshot,
    )
    with pytest.raises(ValueError, match="allocation_mode"):
        with replay_training_batches(Mock(), [snapshot], {"allocation_mode": "shared"}):
            pass


def test_split_replay_images_preserves_values_and_intra_record_aliases() -> None:
    image = torch.arange(8, dtype=torch.float32)
    record = {"pixels": image, "nested": [image]}
    rewards = torch.tensor([0.0, 1.0])
    batch = [{"multi_modal_input": [record, record], "rewards": rewards}]

    result = split_replay_image_aliases(batch)
    records = result[0]["multi_modal_input"]

    assert result[0]["rewards"] is rewards
    assert records[0]["pixels"] is records[0]["nested"][0]
    assert records[0]["pixels"] is not records[1]["pixels"]
    assert records[0]["pixels"] is not image
    for item in records:
        torch.testing.assert_close(item["pixels"], image, rtol=0, atol=0)
    before = replay_image_storage_stats(batch)
    after = replay_image_storage_stats(result)
    assert before["references"] == after["references"] == 4
    assert before["unique_storages"] == 1
    assert after["unique_storages"] == 2
    assert after["unique_storage_bytes"] == 2 * before["unique_storage_bytes"]


@pytest.mark.parametrize("profile", ["fixed", "unfixed"])
def test_worker_replay_returns_remote_batches_once_and_restores_actor(
    tmp_path: Path, profile: str
) -> None:
    snapshot = tmp_path / "prepare_batch-0000.output.pt"
    metadata = {"method": "prepare_batch", "call_index": 0, "n_samples": 12}
    torch.save(
        {
            "schema_version": 2,
            "metadata": metadata,
            "batch": [{"rewards": torch.tensor([1.0])}],
        },
        snapshot,
    )
    actor = Mock(prepare_batch=Mock())
    original = actor.prepare_batch
    remote_batch = [{"rewards": object()}]
    actor._custom_function_call_all_dp_heads.return_value = [remote_batch]

    with replay_training_batches(
        actor, [snapshot], {"n_samples": 12}, worker_profile=profile
    ):
        assert actor.prepare_batch(None) is remote_batch
        with pytest.raises(RuntimeError, match="already consumed"):
            actor.prepare_batch(None)

    assert actor.prepare_batch is original
    original.assert_not_called()
    actor._custom_function_call_all_dp_heads.assert_called_once_with(
        "load_diagnostic_replay", str(snapshot), {"n_samples": 12}, profile
    )


@pytest.mark.parametrize("profile, same_image", [("fixed", True), ("unfixed", False)])
def test_worker_snapshot_load_preserves_or_splits_images(
    tmp_path: Path, profile: str, same_image: bool
) -> None:
    image = torch.arange(8, dtype=torch.float32)
    snapshot = tmp_path / "prepare_batch-0000.output.pt"
    torch.save(
        {
            "schema_version": 2,
            "metadata": {"method": "prepare_batch", "call_index": 0, "n_samples": 12},
            "batch": [{"multi_modal_input": [{"pixels": image}, {"pixels": image}]}],
        },
        snapshot,
    )
    engine = Mock()
    engine.is_data_parallel_head.return_value = True

    result = load_diagnostic_replay(engine, str(snapshot), {"n_samples": 12}, profile)
    records = result[0]["multi_modal_input"]

    assert (records[0]["pixels"] is records[1]["pixels"]) is same_image
    for record in records:
        torch.testing.assert_close(record["pixels"], image, rtol=0, atol=0)
    with pytest.raises(ValueError, match="n_samples"):
        load_diagnostic_replay(engine, str(snapshot), {"n_samples": 8}, profile)


def test_worker_snapshot_non_head_does_not_load_missing_file(tmp_path: Path) -> None:
    engine = Mock()
    engine.is_data_parallel_head.return_value = False

    assert (
        load_diagnostic_replay(engine, str(tmp_path / "missing.pt"), {}, "fixed")
        is None
    )

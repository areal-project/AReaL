# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from examples.swe.qwen38_flash_next.batch_snapshot import replay_training_batches
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

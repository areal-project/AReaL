# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch

from areal.infra.workflow_executor import (
    _RolloutResult,
    clear_rejected_trajectory,
    minimum_generated_version,
)


def test_rpc_preserves_version_summary_and_plain_tensor_container():
    from areal.infra.remote_inf_engine import RemoteInfEngine

    trajectory = {"input_ids": torch.tensor([[1, 2]])}
    dispatcher = Mock()
    dispatcher.wait_for_task.return_value = _RolloutResult(
        task_id=7, trajectory=trajectory, min_generated_version=3
    )
    engine = SimpleNamespace(workflow_executor=SimpleNamespace(dispatcher=dispatcher))
    result = RemoteInfEngine.get_rollout_result(engine, task_id=7)
    assert type(result) is dict
    assert result["min_generated_version"] == 3
    assert result["trajectory"] is trajectory
    dispatcher.wait_for_task.assert_called_once_with(
        7, None, True, consume=False, raise_terminal_error=False
    )


@pytest.mark.parametrize(
    "versions,mask,expected",
    [
        ([-1, 4, 5, -1], [0, 1, 1, 0], 4),
        ([4, -1, 5], [1, 1, 1], -1),
        ([0, 4, 5], [1, 1, 1], 0),
        ([-1, -1], [0, 0], None),
    ],
)
def test_generated_version_ignores_only_masked_tokens(versions, mask, expected):
    trajectory = {
        "input_ids": torch.zeros(1, len(versions), dtype=torch.long),
        "versions": torch.tensor([versions]),
        "loss_mask": torch.tensor([mask]),
    }
    assert minimum_generated_version(trajectory) == expected


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"versions": torch.tensor([[1, 2]])},
        {"versions": torch.tensor([[1.5, 2.5]]), "loss_mask": torch.ones(1, 2)},
        {"versions": torch.tensor([[1, 2]]), "loss_mask": torch.ones(2)},
    ],
)
def test_tensor_payload_with_invalid_versions_fails_closed(metadata):
    assert minimum_generated_version({"input_ids": torch.zeros(1, 2), **metadata}) == -1


def test_non_tensor_workflow_has_no_policy_version():
    assert minimum_generated_version({"text": "result"}) is None


@pytest.mark.asyncio
async def test_rejected_cleanup_attempts_every_node_despite_timeout():
    cleanup = AsyncMock(side_effect=[TimeoutError(), None])
    shards = {"node-a": ["shard-a"], "node-b": ["shard-b"]}
    with (
        patch(
            "areal.infra.workflow_executor.RTensor.collect_shards", return_value=shards
        ),
        patch("areal.infra.workflow_executor.RTensor.clear_node", cleanup),
    ):
        await clear_rejected_trajectory({"input_ids": object()})
    assert cleanup.await_count == 2
    assert {call.args[0] for call in cleanup.await_args_list} == set(shards)


@pytest.mark.asyncio
async def test_rejected_local_payload_does_not_call_remote_cleanup():
    cleanup = AsyncMock()
    with (
        patch("areal.infra.workflow_executor.RTensor.collect_shards", return_value={}),
        patch("areal.infra.workflow_executor.RTensor.clear_node", cleanup),
    ):
        await clear_rejected_trajectory(None)
    cleanup.assert_not_awaited()

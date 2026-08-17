from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from areal.experimental.openai import InteractionWithTokenLogpReward
from areal.experimental.openai.proxy import workflow as proxy_workflow
from areal.infra import workflow_context
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow


class RecordingWorkflow:
    def __init__(self, *, failing_indices: set[int] | None = None):
        self.calls: list[dict] = []
        self.failing_indices = failing_indices or set()
        self.processor_input = None

    async def arun_episode(self, _engine, data):
        index = len(self.calls)
        self.calls.append(data)
        if index in self.failing_indices:
            raise RuntimeError(f"child {index} failed")
        interaction = InteractionWithTokenLogpReward()
        interaction.interaction_id = f"interaction-{index}"
        return {interaction.interaction_id: interaction}

    def process_group_results(self, results):
        self.processor_input = results
        return results


@pytest.mark.asyncio
async def test_grouped_workflow_isolates_child_exception_and_keeps_siblings():
    """Test that one child exception becomes None without discarding its siblings."""
    child = RecordingWorkflow(failing_indices={3})
    logger = MagicMock()
    workflow = GroupedRolloutWorkflow(child, 8, logger=logger)

    merged = await workflow.arun_episode(None, {"task": "x"})

    assert len(child.calls) == 8
    assert child.processor_input is not None
    assert child.processor_input[3] is None
    assert list(merged) == [f"interaction-{index}" for index in range(8) if index != 3]
    logger.warning.assert_called()


@pytest.mark.asyncio
async def test_grouped_workflow_injects_stable_task_local_group_id():
    """Test that siblings receive one stable group ID without mutating input data."""
    child = RecordingWorkflow()
    workflow = GroupedRolloutWorkflow(child, 3, logger=MagicMock())
    data = {"task": "x"}
    workflow_context.set(workflow_context.WorkflowContext(task_id=41))

    await workflow.arun_episode(None, data)

    assert data == {"task": "x"}
    assert [call["group_id"] for call in child.calls] == ["41", "41", "41"]
    assert all(call is not data for call in child.calls)
    assert len({id(call) for call in child.calls}) == 3


class InvalidRAOAgent:
    async def run(self, _data):
        return None


class FakeProxyClient:
    def __init__(self, **_kwargs):
        self.session_api_key = "session-key"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


@pytest.mark.asyncio
async def test_proxy_rejects_invalid_rao_result_without_raising(monkeypatch):
    """Test that invalid RAO output returns None and logs its rejection reason."""
    workflow = proxy_workflow.OpenAIProxyWorkflow(
        mode="inline", agent=InvalidRAOAgent()
    )
    invalid = SimpleNamespace(
        interaction_rewards={},
        root_reward=0.0,
        valid=False,
        group_id="task-41",
        rejection_reason="judge_unavailable",
        error=None,
    )
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(workflow, "_run_agent", AsyncMock(return_value=invalid))
    monkeypatch.setattr(proxy_workflow, "OpenAIProxyClient", FakeProxyClient)
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=MagicMock())
    )
    logger = MagicMock()
    monkeypatch.setattr(proxy_workflow, "logger", logger)

    result = await workflow.arun_episode(None, {"task": "x"})

    assert result is None
    logger.warning.assert_called_once_with(
        "Rejecting invalid RAO rollout group_id=%s: %s",
        "task-41",
        "judge_unavailable",
    )

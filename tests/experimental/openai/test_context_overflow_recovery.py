# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from areal.experimental.openai.proxy import workflow as workflow_module
from areal.experimental.openai.proxy.server import (
    derive_session_gateway_api_key,
    derive_session_gateway_token,
)
from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.infra import workflow_context
from areal.infra.workflow_context import WorkflowContext
from areal.utils import stats_tracker


class _FailingAgent:
    async def run(self, data, **kwargs):
        raise RuntimeError("harness failed")


class _SuccessfulAgent:
    async def run(self, data, **kwargs):
        return 1.0


class _RecordingSuccessfulAgent(_SuccessfulAgent):
    def __init__(self):
        self.persisted_rewards = []

    async def persist_episode_result(self, data, reward):
        self.persisted_rewards.append((data, reward))


class _SystemFailingAgent(_FailingAgent):
    @staticmethod
    def classify_proxy_failure(error, *, context_overflow, interaction_count):
        return "system_failure_reject"


class _ModelFailingAgent(_FailingAgent):
    @staticmethod
    def classify_proxy_failure(error, *, context_overflow, interaction_count):
        return "model_failure_zero"


class _RecordingFailingAgent(_FailingAgent):
    def __init__(self):
        self.failure_dispositions = []
        self.persisted_rewards = []

    async def record_failure_disposition(self, data, error, disposition):
        self.failure_dispositions.append((data, error, disposition))

    async def persist_episode_result(self, data, reward):
        self.persisted_rewards.append((data, reward))


class _FakeProxyClient:
    context_overflow = True
    context_overflow_message = "prompt exceeds context window"
    system_error = False
    system_error_message = ""

    def __init__(self, *args, interaction_count: int = 1, **kwargs):
        self.session_api_key = "session-key"
        self.session_id = "session-1"
        self.interaction_count = interaction_count
        self.interaction = InteractionWithTokenLogpReward(reward=None)
        self.last_reward = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    async def set_last_reward(self, reward: float):
        self.last_reward = reward
        self.interaction.reward = reward

    async def set_reward(self, completion_id: str, reward: float):
        raise AssertionError("Per-completion rewards are not expected")

    async def export_interactions(self, **kwargs):
        if self.interaction_count == 0:
            return {}
        return {"completion-1": self.interaction}


@pytest.mark.asyncio
async def test_inline_agent_receives_proxy_session_id(monkeypatch):
    calls = []

    class RecordingAgent:
        async def run(self, data, **kwargs):
            calls.append((data, kwargs))
            return 1.0

    http_client = object()
    monkeypatch.setattr(
        workflow_context, "get_httpx_client", AsyncMock(return_value=http_client)
    )
    workflow = OpenAIProxyWorkflow(mode="inline", agent=RecordingAgent())

    reward = await workflow._run_agent(
        "session-key",
        {"data_id": "data-1"},
        session_id="session-1",
        worker_runtime="worker-1",
    )

    assert reward == 1.0
    assert calls == [
        (
            {"data_id": "data-1"},
            {
                "base_url": "",
                "http_client": http_client,
                "api_key": "session-key",
                "proxy_gateway_api_key": derive_session_gateway_api_key(
                    workflow._admin_api_key
                ),
                "session_id": "session-1",
                "proxy_session_token": derive_session_gateway_token(
                    workflow._admin_api_key, "session-1"
                ),
                "worker_runtime": "worker-1",
            },
        )
    ]


@pytest.mark.asyncio
async def test_context_overflow_recovers_existing_interactions_with_zero_reward(
    monkeypatch,
):
    """Harness failure after overflow should return the preceding trajectory."""
    fake_client = _FakeProxyClient()
    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda *args, **kwargs: fake_client
    )
    agent = _RecordingFailingAgent()
    workflow = OpenAIProxyWorkflow(mode="inline", agent=agent)
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    stats_tracker.export_all(reset=True)
    workflow_context.set(WorkflowContext(task_id=1))
    try:
        result = await workflow.arun_episode(engine=None, data={})
    finally:
        workflow_context.set(WorkflowContext())

    assert result == {"completion-1": fake_client.interaction}
    assert fake_client.last_reward == 0.0
    assert len(agent.failure_dispositions) == 1
    assert agent.failure_dispositions[0][2] == "model_failure_zero"
    assert agent.persisted_rewards == [({}, 0.0)]
    stats = stats_tracker.export_all(reset=True)
    assert stats["rollout/context_overflow"] == 1.0
    assert stats["rollout/reward"] == 0.0


@pytest.mark.asyncio
async def test_context_overflow_overrides_successful_agent_reward_with_zero(
    monkeypatch,
):
    """Any observed overflow should make the recovered trajectory a failure."""
    fake_client = _FakeProxyClient()
    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda *args, **kwargs: fake_client
    )
    workflow = OpenAIProxyWorkflow(mode="inline", agent=_SuccessfulAgent())
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    workflow_context.set(WorkflowContext(task_id=4))
    try:
        result = await workflow.arun_episode(engine=None, data={})
    finally:
        workflow_context.set(WorkflowContext())

    assert result == {"completion-1": fake_client.interaction}
    assert fake_client.last_reward == 0.0
    assert fake_client.interaction.reward == 0.0


@pytest.mark.asyncio
async def test_first_request_context_overflow_without_interactions_returns_none(
    monkeypatch,
):
    """No trajectory can be recovered when overflow precedes all generations."""
    fake_client = _FakeProxyClient(interaction_count=0)
    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda *args, **kwargs: fake_client
    )
    workflow = OpenAIProxyWorkflow(mode="inline", agent=_FailingAgent())
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    workflow_context.set(WorkflowContext(task_id=2))
    try:
        result = await workflow.arun_episode(engine=None, data={})
    finally:
        workflow_context.set(WorkflowContext())

    assert result is None
    assert fake_client.last_reward is None


@pytest.mark.asyncio
async def test_successful_agent_with_empty_export_is_rejected_and_audited(monkeypatch):
    """Dropped/orphan-only exports are not usable grouped-rollout slots."""
    fake_client = _FakeProxyClient(interaction_count=0)
    fake_client.context_overflow = False
    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda *args, **kwargs: fake_client
    )
    agent = _RecordingSuccessfulAgent()
    workflow = OpenAIProxyWorkflow(mode="inline", agent=agent)
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    stats_tracker.export_all(reset=True)
    workflow_context.set(WorkflowContext(task_id=6))
    try:
        result = await workflow.arun_episode(engine=None, data={})
    finally:
        workflow_context.set(WorkflowContext())

    assert result is None
    assert agent.persisted_rewards == [({}, None)]
    assert "rollout/reward" not in stats_tracker.export_all(reset=True)


@pytest.mark.asyncio
async def test_non_context_agent_failure_still_propagates(monkeypatch):
    """Infrastructure and unrelated Harness failures must not become reward zero."""
    fake_client = _FakeProxyClient()
    fake_client.context_overflow = False
    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda *args, **kwargs: fake_client
    )
    workflow = OpenAIProxyWorkflow(mode="inline", agent=_FailingAgent())
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    workflow_context.set(WorkflowContext(task_id=3))
    try:
        with pytest.raises(RuntimeError, match="harness failed"):
            await workflow.arun_episode(engine=None, data={})
    finally:
        workflow_context.set(WorkflowContext())


@pytest.mark.asyncio
async def test_context_overflow_does_not_mask_classified_system_failure(monkeypatch):
    """Authoritative system attribution must override an overflow signal."""
    fake_client = _FakeProxyClient()
    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda *args, **kwargs: fake_client
    )
    workflow = OpenAIProxyWorkflow(mode="inline", agent=_SystemFailingAgent())
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    workflow_context.set(WorkflowContext(task_id=5))
    try:
        with pytest.raises(RuntimeError, match="harness failed"):
            await workflow.arun_episode(engine=None, data={})
    finally:
        workflow_context.set(WorkflowContext())

    assert fake_client.last_reward is None


@pytest.mark.asyncio
async def test_proxy_system_error_overrides_model_failure_classifier(monkeypatch):
    """An observed proxy 500 must reject even if Harness reports agent failure."""
    fake_client = _FakeProxyClient()
    fake_client.context_overflow = False
    fake_client.system_error = True
    fake_client.system_error_message = "backend unavailable"
    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda *args, **kwargs: fake_client
    )
    workflow = OpenAIProxyWorkflow(mode="inline", agent=_ModelFailingAgent())
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    workflow_context.set(WorkflowContext(task_id=7))
    try:
        with pytest.raises(RuntimeError, match="harness failed"):
            await workflow.arun_episode(engine=None, data={})
    finally:
        workflow_context.set(WorkflowContext())

    assert fake_client.last_reward is None

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import torch

from examples.swe.arena_agent import ArenaStreamAgentWorkflow
from examples.swe.arena_client import ArenaTaskFailedError, ArenaTaskResult

from areal.api import ModelResponse
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.proxy import workflow as workflow_module
from areal.experimental.openai.proxy.proxy_gateway import CompletedSessionInfo
from areal.experimental.openai.proxy.server import (
    derive_session_gateway_api_key,
    derive_session_gateway_token,
)
from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.experimental.openai.types import (
    InteractionWithTokenLogpReward,
    concat_tensor_interactions,
    normalize_logical_rollout_rewards,
)
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
@pytest.mark.parametrize("mode", ["inline", "online"])
@pytest.mark.parametrize("explicit_reference", [None, 2.0])
async def test_discounted_individual_export_supplies_terminal_reference(
    monkeypatch, mode, explicit_reference
):
    interactions = {}
    for key, reward in [("first", 0.0), ("last", 1.0)]:
        interaction = InteractionWithTokenLogpReward(
            reward=reward,
            output_message_list=[],
            model_response=ModelResponse(
                input_tokens=[1],
                output_tokens=[2],
                output_logprobs=[0.0],
                output_versions=[0],
            ),
        )
        interaction.interaction_id = key
        interactions[key] = interaction
    interactions["first"].rollout_reward = explicit_reference
    cache = InteractionCache.from_dict(interactions)

    class DiscountedProxyClient(_FakeProxyClient):
        context_overflow = False

        async def export_interactions(self, *, discount, style, **kwargs):
            return cache.export_interactions(style=style, reward_discount=discount)

    fake_client = DiscountedProxyClient()
    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda *args, **kwargs: fake_client
    )
    workflow = OpenAIProxyWorkflow(
        mode=mode,
        agent=_SuccessfulAgent(),
        discount=0.5,
        proxy_gateway_addr="http://localhost",
    )
    if mode == "online":
        monkeypatch.setattr(
            workflow,
            "_run_agent",
            AsyncMock(
                return_value=CompletedSessionInfo(
                    session_api_key="session-key",
                    session_id="session-1",
                    worker_addr="",
                )
            ),
        )
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    workflow_context.set(WorkflowContext(task_id=1))
    try:
        result = await workflow.arun_episode(engine=None, data={})
    finally:
        workflow_context.set(WorkflowContext())
        stats_tracker.export_all(reset=True)

    assert [v.reward for v in result.values()] == [0.5, 1.0]
    reference = explicit_reference if explicit_reference is not None else 1.0
    assert concat_tensor_interactions(result)["rollout_group"].rewards == (reference,)
    peer = InteractionWithTokenLogpReward(reward=3.0)
    assert normalize_logical_rollout_rewards([result, {"peer": peer}])
    expected = (torch.tensor([0.5, 1.0]) - (reference + 3.0) / 2) / (
        (3.0 - reference) / 2
    )
    torch.testing.assert_close(
        concat_tensor_interactions(result)["rewards"], expected, rtol=1e-6, atol=1e-6
    )


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


@pytest.mark.asyncio
@pytest.mark.parametrize("model_failure", [False, True])
@pytest.mark.parametrize("explicit_reference", [None, 2.0])
async def test_concat_episode_reward_preserves_unscored_branch_and_cached_tensors(
    monkeypatch, model_failure, explicit_reference
):
    missing = InteractionWithTokenLogpReward(reward=None)
    missing._cache = {
        "rewards": torch.zeros(2, dtype=torch.float64),
        "original_rewards": torch.zeros(2, dtype=torch.float64),
    }
    explicit = InteractionWithTokenLogpReward(
        reward=0.25, rollout_reward=explicit_reference
    )

    class BranchedClient(_FakeProxyClient):
        context_overflow = False

        async def export_interactions(self, **kwargs):
            return {"child": missing, "explicit": explicit, "main": self.interaction}

    client = BranchedClient()
    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda *args, **kwargs: client
    )
    agent = _ModelFailingAgent() if model_failure else _SuccessfulAgent()
    workflow = OpenAIProxyWorkflow(mode="inline", agent=agent, export_style="concat")
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    workflow_context.set(WorkflowContext(task_id=1))
    try:
        result = await workflow.arun_episode(None, {})
    finally:
        workflow_context.set(WorkflowContext())
        stats_tracker.export_all(reset=True)
    expected = 0.0 if model_failure else 1.0
    assert result["child"].reward is None
    assert result["main"].reward == expected
    assert result["explicit"].reward == 0.25
    for key in ["rewards", "original_rewards"]:
        torch.testing.assert_close(
            missing._cache[key], torch.zeros(2, dtype=torch.float64)
        )
    assert result["main"].rollout_reward is None
    assert explicit.rollout_reward == explicit_reference
    assert not normalize_logical_rollout_rewards([result])


@pytest.mark.asyncio
async def test_concat_per_completion_rewards_do_not_fill_unscored_branch(monkeypatch):
    class PerCompletionAgent:
        async def run(self, data, **kwargs):
            return {"main": 1.0}

    missing = InteractionWithTokenLogpReward(reward=None)

    class BranchedClient(_FakeProxyClient):
        context_overflow = False

        async def set_reward(self, completion_id, reward):
            assert completion_id == "main"
            self.interaction.reward = reward

        async def export_interactions(self, **kwargs):
            return {"child": missing, "main": self.interaction}

    client = BranchedClient()
    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda *args, **kwargs: client
    )
    workflow = OpenAIProxyWorkflow(
        mode="inline", agent=PerCompletionAgent(), export_style="concat"
    )
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    workflow_context.set(WorkflowContext(task_id=1))
    try:
        result = await workflow.arun_episode(None, {})
    finally:
        workflow_context.set(WorkflowContext())
        stats_tracker.export_all(reset=True)
    assert result["child"].reward is None
    assert result["main"].reward == 1.0
    assert not normalize_logical_rollout_rewards([result])


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["HARNESS_FAILED", "TIMEOUT", "NO_OUTPUT"])
async def test_arena_unhealthy_receipt_with_overflow_never_exports(monkeypatch, status):
    """An actual Arena classifier rejection survives the proxy overflow path."""
    monkeypatch.setenv("ARENA_OPENAPI_BASE", "https://arena.example")
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    fake_client = _FakeProxyClient()
    fake_client.export_interactions = AsyncMock()
    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda *args, **kwargs: fake_client
    )
    agent = ArenaStreamAgentWorkflow(
        econfig={
            "arena_streams": [
                {
                    "name": "test",
                    "stream_id": "stream",
                }
            ]
        }
    )
    error = ArenaTaskFailedError(
        task_id="task",
        status=status,
        result=ArenaTaskResult(
            task_id="task",
            status=status,
            score=0.0,
            raw={
                "nativeRlReceiptVersion": 1,
                "nativeExportHealthy": False,
                "outcome_code": "AGENT_MAX_TURNS_EXCEEDED",
                "error": "harness: harness agent phase exited with code 1: "
                "harness: agent phase error: claude reported error:",
            },
        ),
    )
    workflow = OpenAIProxyWorkflow(mode="inline", agent=agent)
    monkeypatch.setattr(workflow, "_run_agent", AsyncMock(side_effect=error))
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    workflow_context.set(WorkflowContext(task_id=6))
    try:
        with pytest.raises(ArenaTaskFailedError) as caught:
            await workflow.arun_episode(engine=None, data={})
        assert caught.value is error
    finally:
        workflow_context.set(WorkflowContext())
        stats_tracker.export_all(reset=True)
    assert fake_client.last_reward is None
    fake_client.export_interactions.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("context_overflow", [False, True])
@pytest.mark.parametrize(
    "raw,requires_overflow",
    [
        ({"outcome_code": "AGENT_MAX_TURNS_EXCEEDED"}, False),
        (
            {
                "error": "harness: harness agent phase exited with code 1: "
                "harness: agent phase error: claude reported error: Prompt is too long"
            },
            False,
        ),
        (
            {
                "error": "harness: harness agent phase exited with code 1: "
                "2026/09/23 12:00:00 harness: agent phase error: claude reported error:\n"
                "2026/09/23 12:00:00 harness: running collect hook\n"
                "2026/09/23 12:00:00 harness: claude reported error:"
            },
            True,
        ),
    ],
)
@pytest.mark.parametrize(
    "trajectory",
    ["usable", "system_error", "no_interactions", "empty_export", "failed_export"],
)
async def test_arena_legacy_model_failure_requires_usable_proxy_trajectory(
    monkeypatch, raw, requires_overflow, context_overflow, trajectory
):
    """Real legacy classification only trains successfully exported model interactions."""
    monkeypatch.setenv("ARENA_OPENAPI_BASE", "https://arena.example")
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    fake_client = _FakeProxyClient(
        interaction_count=0 if trajectory == "no_interactions" else 1
    )
    fake_client.context_overflow = context_overflow
    fake_client.system_error = trajectory == "system_error"
    fake_client.export_interactions = AsyncMock(
        return_value={}
        if trajectory == "empty_export"
        else {"completion-1": fake_client.interaction},
        side_effect=RuntimeError("export failed")
        if trajectory == "failed_export"
        else None,
    )
    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda *args, **kwargs: fake_client
    )
    agent = ArenaStreamAgentWorkflow(
        econfig={"arena_streams": [{"name": "test", "stream_id": "stream"}]}
    )
    agent.persist_episode_result = AsyncMock()
    error = ArenaTaskFailedError(
        task_id="task",
        status="HARNESS_FAILED",
        result=ArenaTaskResult(
            task_id="task", status="HARNESS_FAILED", score=0.0, raw=raw
        ),
    )
    workflow = OpenAIProxyWorkflow(mode="inline", agent=agent)
    monkeypatch.setattr(workflow, "_run_agent", AsyncMock(side_effect=error))
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(
        workflow, "_record_interaction_stats", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        workflow, "record_episode_metrics", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    workflow_context.set(WorkflowContext(task_id=8))
    try:
        if trajectory in {"system_error", "no_interactions"} or (
            requires_overflow and not context_overflow
        ):
            with pytest.raises(ArenaTaskFailedError):
                await workflow.arun_episode(engine=None, data={})
            assert fake_client.last_reward is None
            fake_client.export_interactions.assert_not_awaited()
        elif trajectory == "failed_export":
            with pytest.raises(RuntimeError, match="export failed"):
                await workflow.arun_episode(engine=None, data={})
            agent.persist_episode_result.assert_not_awaited()
        else:
            result = await workflow.arun_episode(engine=None, data={})
            if trajectory == "empty_export":
                assert result is None
                agent.persist_episode_result.assert_awaited_once_with({}, None)
            else:
                assert result["completion-1"].reward == 0.0
                agent.persist_episode_result.assert_awaited_once_with({}, 0.0)
    finally:
        workflow_context.set(WorkflowContext())
        stats_tracker.export_all(reset=True)

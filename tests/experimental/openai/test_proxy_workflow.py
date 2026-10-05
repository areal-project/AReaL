# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from areal.experimental.openai.proxy import workflow as workflow_module
from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.infra import workflow_context
from areal.infra.workflow_context import WorkflowContext


class _Agent:
    async def run(self, data, **kwargs):
        return 0.0


class _EmptyProxyClient:
    session_api_key = "session-key"

    def __init__(self, **kwargs):
        self.behaviour_metrics = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def set_last_reward(self, reward):
        return None

    async def export_interactions(self, *, discount, style):
        return {}


@pytest.mark.asyncio
async def test_inline_proxy_workflow_empty_export_raises(monkeypatch):
    """An empty export must fail its rollout group instead of dropping one member."""
    proxy_workflow = OpenAIProxyWorkflow(mode="inline", agent=_Agent())
    proxy_workflow._grant_capacity = AsyncMock()
    proxy_workflow._run_agent = AsyncMock(return_value=0.0)

    async def get_session():
        return object()

    monkeypatch.setattr(workflow_context, "get_aiohttp_session", get_session)
    monkeypatch.setattr(workflow_module, "OpenAIProxyClient", _EmptyProxyClient)
    workflow_context.set(WorkflowContext(task_id=7, group_size=8))

    with pytest.raises(RuntimeError, match="produced no complete interactions"):
        await proxy_workflow.arun_episode(engine=None, data={})

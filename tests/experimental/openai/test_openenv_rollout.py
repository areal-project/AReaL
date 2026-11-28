# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from areal.api import ModelResponse
from areal.api.cli_args import AgentConfig, InferenceEngineConfig
from areal.experimental.openai.proxy import workflow as workflow_module
from areal.experimental.openai.proxy.server import SessionData
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.infra import workflow_context
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.infra.workflow_context import WorkflowContext
from areal.utils import stats_tracker


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "step_rewards, done, agent_kwargs, export_style, model_stop_reason, "
    "expected_assignments, expected_returns",
    [
        pytest.param(
            (0.25, 0.5),
            True,
            {"reward_mode": "dense", "truncation_policy": "drop"},
            "individual",
            "length",
            [("completion-1", 0.25), ("completion-2", 0.5)],
            [0.75, 0.5],
            id="dense-individual",
        ),
        pytest.param(
            (None, None),
            True,
            {"reward_mode": "dense", "missing_reward_policy": "ignore"},
            "individual",
            "stop",
            [],
            [0.0, 0.0],
            id="dense-all-missing",
        ),
        pytest.param(
            (0.25, 0.5),
            True,
            {"reward_mode": "dense"},
            "concat",
            "stop",
            [],
            None,
            id="dense-concat",
        ),
        pytest.param(
            (0.25,),
            False,
            {"truncation_policy": "retain_terminal"},
            "individual",
            "stop",
            [(None, 0.25)],
            [0.25],
            id="retain-terminal",
        ),
        pytest.param(
            (0.25,),
            False,
            {"truncation_policy": "retain_zero"},
            "individual",
            "stop",
            [(None, 0.0)],
            [0.0],
            id="retain-zero",
        ),
        pytest.param(
            (0.25,),
            False,
            {"truncation_policy": "drop"},
            "individual",
            "stop",
            [],
            None,
            id="drop",
        ),
    ],
)
async def test_openenv_agent_exports_environment_metadata_through_proxy(
    monkeypatch,
    step_rewards,
    done,
    agent_kwargs,
    export_style,
    model_stop_reason,
    expected_assignments,
    expected_returns,
):
    """Exercise the inline runtime, SDK requests, and individual proxy exporter."""

    class EchoClient:
        def __init__(self, *, base_url):
            assert base_url == "http://env.invalid"
            self.turn = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            closed.append(self)

        async def reset(self):
            return SimpleNamespace(observation="ready", done=False)

        async def step(self, action):
            self.turn += 1
            assert action.tool_name == "echo_message"
            assert action.arguments == {"message": f"reply-{self.turn}"}
            return SimpleNamespace(
                observation=f"echo-{self.turn}",
                reward=step_rewards[self.turn - 1],
                done=done and self.turn == len(step_rewards),
                metadata=None,
            )

    def completion_endpoint(request):
        assert request.url.path == "/chat/completions"
        assert request.headers["authorization"] == "Bearer session-key"
        messages = json.loads(request.content)["messages"]
        request_messages.append(messages)
        turn = len(request_messages)
        completion_id = f"completion-{turn}"
        assistant = {
            "role": "assistant",
            "content": f"reply-{turn}",
            "reasoning_content": f"reasoning-{turn}",
            "name": "echo-model",
        }
        session.completions[completion_id] = InteractionWithTokenLogpReward(
            _interaction_id=completion_id,
            messages=messages,
            output_message_list=[assistant],
            chat_template_type="concat" if export_style == "concat" else "hf",
            model_response=ModelResponse(
                input_tokens=[1] + [2] * (turn - 1),
                output_tokens=[2],
                output_logprobs=[-0.1],
                output_versions=[1],
                stop_reason=model_stop_reason,
            ),
        )
        return httpx.Response(
            200,
            json={
                "id": completion_id,
                "object": "chat.completion",
                "created": 0,
                "model": "default",
                "choices": [
                    {
                        "index": 0,
                        "message": {**assistant, "refusal": None},
                        "finish_reason": model_stop_reason,
                    }
                ],
            },
        )

    class ProxyClient:
        context_overflow = False
        system_error = False
        session_id = "session-1"
        session_api_key = "session-key"

        @property
        def interaction_count(self):
            return len(session.completions)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            session.finish()

        async def set_last_reward(self, reward):
            received.append((None, reward))
            session.completions.set_last_reward(reward)

        async def set_reward(self, completion_id, reward):
            received.append((completion_id, reward))
            session.completions.set_reward(completion_id, reward)

        async def export_interactions(
            self, *, discount, style, drop_retry_orphans, is_eval
        ):
            export_calls.append((discount, style, drop_retry_orphans, is_eval))
            assert session.is_completed
            return session.export_interactions(
                discount=discount, style=style, drop_retry_orphans=drop_retry_orphans
            )

    closed = []
    received = []
    request_messages = []
    export_calls = []
    session = SessionData(session_id="session-1")
    monkeypatch.setitem(
        sys.modules,
        "echo_env",
        SimpleNamespace(
            EchoEnv=EchoClient,
            CallToolAction=SimpleNamespace,
        ),
    )
    monkeypatch.setenv("OPENENV_BASE_URL", "http://env.invalid")
    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda *_a, **_kw: ProxyClient()
    )
    monkeypatch.setattr(workflow_context, "get_aiohttp_session", AsyncMock())
    agent_config = AgentConfig(
        agent_cls_path="examples.openenv.agent.EchoAgent",
        mode="inline",
        export_style=export_style,
    )
    engine = RemoteInfEngine(
        InferenceEngineConfig(agent=agent_config), backend=SimpleNamespace()
    )
    workflow = engine._resolve_workflow(
        agent_config.agent_cls_path,
        {"max_turns": len(step_rewards), **agent_kwargs},
        proxy_addr="http://proxy.invalid",
    )
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    original_context = workflow_context.get()
    workflow_context.set(WorkflowContext(task_id=1))
    try:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(completion_endpoint)
        ) as http:
            monkeypatch.setattr(
                workflow_context, "get_httpx_client", AsyncMock(return_value=http)
            )
            if export_style == "concat":
                with pytest.raises(ValueError, match="agent.export_style='individual'"):
                    await workflow.arun_episode(engine, {"messages": []})
            else:
                interactions = await workflow.arun_episode(engine, {"messages": []})
        metrics = stats_tracker.export_all(reset=True)
    finally:
        workflow_context.set(original_context)

    assert session.is_completed
    assert received == expected_assignments
    if export_style == "concat":
        assert not request_messages
        assert not closed
        assert not export_calls
        return

    assert len(request_messages) == len(step_rewards)
    if len(step_rewards) == 2:
        assert request_messages[1] == [
            {"role": "user", "content": "ready"},
            {
                "role": "assistant",
                "content": "reply-1",
                "reasoning_content": "reasoning-1",
                "name": "echo-model",
            },
            {"role": "user", "content": "echo-1"},
        ]
        assert (
            session.completions["completion-2"].parent
            is session.completions["completion-1"]
        )
    assert len(closed) == 1
    assert metrics["rollout/env_truncated"] == float(not done)
    if expected_returns is None:
        assert interactions is None
        assert not export_calls
        return

    assert export_calls == [(1.0, "individual", False, False)]
    assert list(interactions) == [
        f"completion-{turn}" for turn in range(1, len(step_rewards) + 1)
    ]
    assert [
        interaction.reward for interaction in interactions.values()
    ] == expected_returns
    for interaction in interactions.values():
        assert interaction.metadata == {
            "env_terminated": done,
            "env_truncated": not done,
            "env_stop_reason": None if done else "max_turns",
            "session_id": "session-1",
        }
        assert interaction.to_tensor_dict()["is_truncated"].item() == (
            model_stop_reason == "length"
        )

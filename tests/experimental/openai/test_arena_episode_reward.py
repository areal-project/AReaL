# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import torch

from examples.swe.arena_agent import ArenaStreamAgentWorkflow
from examples.swe.arena_client import ArenaTaskResult

from areal.api import ModelResponse
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.proxy import proxy_rollout_server as server_module
from areal.experimental.openai.proxy import workflow as workflow_module
from areal.experimental.openai.proxy.client_session import OpenAIProxyClient
from areal.experimental.openai.proxy.server import (
    ExportTrajectoriesRequest,
    SessionData,
    deserialize_interactions,
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


def _compacted_session(chain_lengths=(71, 1, 15)) -> InteractionCache:
    """Observed topology, including a rejected Summary followed by another."""
    cache = InteractionCache()
    start = 1
    for chain_index, length in enumerate(chain_lengths):
        stop = start + length
        system = "solve" if chain_index in (0, len(chain_lengths) - 1) else "summary"
        messages = [{"role": "system", "content": system}]
        input_tokens = [1]
        for index in range(start, stop):
            messages = messages + [{"role": "user", "content": f"input-{index}"}]
            input_tokens = input_tokens + [2]
            output = [{"role": "assistant", "content": f"output-{index}"}]
            turn = InteractionWithTokenLogpReward(
                messages=messages,
                output_message_list=output,
                chat_template_type="concat",
                model_response=ModelResponse(
                    input_tokens=input_tokens,
                    output_tokens=[index + 10],
                    output_logprobs=[-0.1],
                    output_versions=[0],
                ),
            )
            turn.interaction_id = str(index)
            cache[str(index)] = turn
            messages = messages + output
            input_tokens = input_tokens + [index + 10]
        start = stop
    return cache


@pytest.fixture
def arena_agent(monkeypatch):
    monkeypatch.setenv("ARENA_OPENAPI_BASE", "https://arena.example")
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    agent = ArenaStreamAgentWorkflow(
        econfig={"arena_streams": [{"name": "test", "stream_id": "stream"}]}
    )
    agent.persist_episode_result = AsyncMock()
    agent.record_episode_metrics = Mock()
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("reward", [0.0, 0.5275])
@pytest.mark.parametrize("chain_lengths", [(71, 1, 15), (18, 1, 1, 76)])
async def test_arena_compacted_episode_scores_all_actions_once(
    monkeypatch, arena_agent, reward, chain_lengths
):
    cache = _compacted_session(chain_lengths)
    leaves = cache.export_interactions(style="concat")
    expected_leaf_ids = [
        str(sum(chain_lengths[: index + 1])) for index in range(len(chain_lengths))
    ]
    assert list(leaves) == expected_leaf_ids
    # The proxy may already have serialized tensors before episode finalization.
    for leaf in leaves.values():
        leaf.to_tensor_dict()

    class ProxyClient:
        context_overflow = False
        system_error = False
        interaction_count = len(cache)
        session_id = "watermill-session"
        session_api_key = "session-key"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def set_last_reward(self, value):
            cache.set_last_reward(value)

        async def export_interactions(
            self, *, style, discount, drop_retry_orphans, **kwargs
        ):
            session = SessionData(session_id=self.session_id)
            session._completions = cache
            session.finish()
            monkeypatch.setattr(
                server_module, "_session_cache", {self.session_id: session}
            )
            monkeypatch.setattr(server_module, "_prm_runner", None)
            response = await server_module.export_trajectories(
                ExportTrajectoriesRequest(
                    session_id=self.session_id,
                    style=style,
                    discount=discount,
                    drop_retry_orphans=drop_retry_orphans,
                    **kwargs,
                )
            )
            return deserialize_interactions(response.interactions)

    monkeypatch.setattr(
        workflow_module, "OpenAIProxyClient", lambda **kwargs: ProxyClient()
    )
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    workflow = OpenAIProxyWorkflow(
        mode="inline", agent=arena_agent, export_style="concat", drop_retry_orphans=True
    )
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(workflow, "_run_agent", AsyncMock(return_value=reward))
    arena_agent._task_result.set(
        ArenaTaskResult(task_id="task", status="OK", score=reward)
    )
    workflow_context.set(WorkflowContext(task_id=15, sample_idx=5, group_size=12))
    try:
        result = await workflow.arun_episode(None, {})
    finally:
        workflow_context.set(WorkflowContext())
        stats_tracker.export_all(reset=True)

    assert list(result) == expected_leaf_ids
    num_rows = len(chain_lengths)
    assert [leaf.reward for leaf in result.values()] == [reward] * num_rows
    assert [leaf.rollout_reward for leaf in result.values()] == [reward] * num_rows
    tensors = concat_tensor_interactions(result)
    assert tensors["rollout_group"].rewards == (reward,)
    assert tensors["loss_mask"].sum().item() == sum(chain_lengths)
    for key in ["rewards", "original_rewards"]:
        torch.testing.assert_close(tensors[key], torch.full((num_rows,), reward))
    # Multiple exported rows still represent only one sample in the GRPO mean.
    peer = {"peer": InteractionWithTokenLogpReward(reward=1.0)}
    assert normalize_logical_rollout_rewards([result, peer], use_std=False)
    torch.testing.assert_close(
        concat_tensor_interactions(result)["rewards"],
        torch.full((num_rows,), reward - (reward + 1.0) / 2),
    )


@pytest.mark.parametrize(
    "ambiguity",
    [
        "shared_ancestor",
        "scalar_conflict",
        "reference_conflict",
        "token_rewards",
        "ancestor_reward",
        "incomplete",
        "cycle",
    ],
)
def test_arena_episode_reward_leaves_ambiguous_exports_unchanged(ambiguity):
    cache = _compacted_session()
    cache.set_last_reward(0.5)
    leaves = cache.export_interactions(style="concat")
    if ambiguity == "shared_ancestor":
        leaves["72"].parent = cache["1"]
    elif ambiguity == "scalar_conflict":
        leaves["72"].reward = 0.25
    elif ambiguity == "reference_conflict":
        leaves["72"].rollout_reward = 0.25
    elif ambiguity == "token_rewards":
        cache["1"].token_rewards = torch.tensor([0.25])
    elif ambiguity == "ancestor_reward":
        cache["1"].reward = 0.25
    elif ambiguity == "incomplete":
        cache["1"].output_message_list = None
    elif ambiguity == "cycle":
        cache["1"].parent = cache["71"]
    before = [(x.reward, x.rollout_reward, x.original_reward) for x in cache.values()]
    InteractionCache.from_dict(leaves).assign_episode_reward(0.5)

    assert [
        (x.reward, x.rollout_reward, x.original_reward) for x in cache.values()
    ] == before
    assert leaves["71"].reward is None


@pytest.mark.parametrize("status", ["HARNESS_FAILED", "EVAL_FAILED", "RUNNING"])
def test_arena_non_success_does_not_fill_missing_scores(arena_agent, status):
    leaves = _compacted_session().export_interactions(style="concat")
    arena_agent._task_result.set(
        ArenaTaskResult(task_id="task", status=status, score=0.0)
    )
    assert (
        arena_agent.get_episode_reward_for_export({}, 0.0, export_style="concat")
        is None
    )
    assert all(x.reward is None and x.rollout_reward is None for x in leaves.values())


def test_arena_individual_export_retains_turn_rewards(arena_agent):
    cache = _compacted_session()
    cache.set_last_reward(0.5)
    turns = cache.export_interactions(style="individual", reward_discount=0.5)
    before = [(x.reward, x.rollout_reward) for x in turns.values()]
    arena_agent._task_result.set(
        ArenaTaskResult(task_id="task", status="OK", score=0.5)
    )
    assert (
        arena_agent.get_episode_reward_for_export({}, 0.5, export_style="individual")
        is None
    )
    assert [(x.reward, x.rollout_reward) for x in turns.values()] == before


@pytest.mark.asyncio
@pytest.mark.parametrize("episode_reward", [None, 0.5])
@pytest.mark.parametrize("prm_enabled", [False, True])
async def test_proxy_episode_return_requires_opt_in_and_preserves_prm(
    monkeypatch, episode_reward, prm_enabled
):
    cache = _compacted_session()
    cache.set_last_reward(0.5)
    session = SessionData(session_id="episode")
    session._completions = cache
    session.finish()
    monkeypatch.setattr(server_module, "_session_cache", {"episode": session})
    monkeypatch.setattr(server_module, "_prm_runner", Mock() if prm_enabled else None)
    scorer = AsyncMock(side_effect=lambda interactions, *args, **kwargs: interactions)
    monkeypatch.setattr(server_module, "_score_prm_branches", scorer)

    response = await server_module.export_trajectories(
        ExportTrajectoriesRequest(
            session_id="episode", style="concat", episode_reward=episode_reward
        )
    )
    restored = deserialize_interactions(response.interactions)

    expected = (
        [0.5] * 3
        if episode_reward is not None and not prm_enabled
        else [None, None, 0.5]
    )
    assert [leaf.reward for leaf in restored.values()] == expected
    assert scorer.await_count == int(prm_enabled)
    stats_tracker.export_all(reset=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("episode_reward", [None, 0.0, 0.5])
async def test_proxy_client_sends_episode_reward_only_when_requested(episode_reward):
    response = MagicMock()
    response.__aenter__ = AsyncMock(return_value=response)
    response.json = AsyncMock(return_value={"interactions": {}})
    session = Mock()
    session.post.return_value = response
    client = OpenAIProxyClient(
        session=session,
        base_url="http://proxy.example",
        task_id="task",
        admin_api_key="key",
    )
    client.session_id = "episode"

    await client.export_interactions(style="concat", episode_reward=episode_reward)

    payload = session.post.call_args.kwargs["json"]
    if episode_reward is None:
        assert "episode_reward" not in payload
    else:
        assert payload["episode_reward"] == episode_reward

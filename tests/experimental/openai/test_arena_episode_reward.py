# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import torch

from examples.swe.arena_agent import ArenaStreamAgentWorkflow
from examples.swe.arena_client import ArenaTaskFailedError, ArenaTaskResult

from areal.api import ModelResponse
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.finalization import finalize_interactions
from areal.experimental.openai.proxy import proxy_rollout_server as server_module
from areal.experimental.openai.proxy import workflow as workflow_module
from areal.experimental.openai.proxy.client_session import OpenAIProxyClient
from areal.experimental.openai.proxy.server import (
    ExportTrajectoriesRequest,
    SessionData,
    deserialize_interactions,
    serialize_interactions,
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
@pytest.mark.parametrize("context_overflow", [False, True])
async def test_arena_compacted_episode_scores_all_actions_once(
    monkeypatch, arena_agent, reward, chain_lengths, context_overflow
):
    overflow = context_overflow
    if overflow:
        reward = 0.0
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
        context_overflow = overflow
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
    task_result = ArenaTaskResult(
        task_id="task",
        status="HARNESS_FAILED" if overflow else "OK",
        score=reward,
        raw={
            "error": "harness: running setup hook (timeout=10m0s)",
            "harness": {"phase": "agent", "exit_code": 1, "result_status": "ERROR"},
        },
    )
    arena_agent._task_result.set(task_result)
    error = (
        ArenaTaskFailedError(
            task_id="task", status="HARNESS_FAILED", result=task_result
        )
        if overflow
        else None
    )
    monkeypatch.setattr(
        workflow, "_run_agent", AsyncMock(return_value=reward, side_effect=error)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_response", [None, "earlier", "terminal"])
@pytest.mark.parametrize("materialize_first", [False, True])
@pytest.mark.parametrize("use_std", [False, True])
async def test_scored_concat_segments_export_policy_adjusted_episode_reference(
    monkeypatch, invalid_response, materialize_first, use_std
):
    """RPC export preserves row returns and one validated episode reference."""
    cache = _compacted_session((2, 1, 2))
    cache.set_reward("2", 0.2)
    cache.set_reward("3", 0.6)
    cache.set_last_reward(0.8)
    if invalid_response is not None:
        invalid_id = "2" if invalid_response == "earlier" else "5"
        cache[invalid_id].metadata["response_parse"] = {
            "tool_call_parse_failed": False,
            "has_content": False,
            "has_tool_calls": False,
        }
    leaves = cache.export_interactions(style="concat")
    if materialize_first:
        for leaf in leaves.values():
            leaf.to_tensor_dict()
    session = SessionData(session_id="scored-episode")
    session._completions = cache
    session.finish()
    monkeypatch.setattr(server_module, "_session_cache", {session.session_id: session})
    monkeypatch.setattr(server_module, "_prm_runner", None)

    response = await server_module.export_trajectories(
        ExportTrajectoriesRequest(
            session_id=session.session_id, style="concat", episode_reward=0.8
        )
    )
    restored = deserialize_interactions(response.interactions)

    row_returns = [
        0.0 if invalid_response == "earlier" else 0.2,
        0.6,
        0.0 if invalid_response == "terminal" else 0.8,
    ]
    reference = 0.0 if invalid_response == "terminal" else 0.8
    assert [leaf.reward for leaf in restored.values()] == row_returns
    assert [leaf.rollout_reward for leaf in restored.values()] == [reference] * 3
    peer = {"peer": InteractionWithTokenLogpReward(reward=0.4)}
    assert normalize_logical_rollout_rewards([restored, peer], use_std=use_std)
    mean = (reference + 0.4) / 2
    scale = abs(reference - 0.4) / 2 + 1e-8 if use_std else 1.0
    tensors = concat_tensor_interactions(restored)
    torch.testing.assert_close(
        tensors["rewards"],
        (torch.tensor(row_returns) - mean) / scale,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        tensors["original_rewards"], torch.tensor(row_returns), rtol=0, atol=0
    )
    assert tensors["rollout_group"].rewards == pytest.approx(
        ((reference - mean) / scale,)
    )
    assert peer["peer"].reward == pytest.approx((0.4 - mean) / scale)
    assert tensors["loss_mask"].sum().item() == 5
    stats_tracker.export_all(reset=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("materialize_first", [False, True])
@pytest.mark.parametrize("use_std", [False, True])
async def test_scored_retry_branches_share_episode_reference_across_rpc(
    monkeypatch, materialize_first, use_std
):
    """A failed retry and its successful sibling keep distinct training returns."""
    cache = _compacted_session((2,))
    retry = deepcopy(cache["2"])
    retry.interaction_id = "3"
    retry.output_message_list = [{"role": "assistant", "content": "success"}]
    retry.model_response.output_tokens = [13]
    cache["3"] = retry
    cache.set_reward("2", 0.0)
    cache.set_last_reward(0.8)
    cache["2"].metadata["response_parse"] = {
        "tool_call_parse_failed": True,
        "has_content": False,
        "has_tool_calls": False,
    }
    leaves = cache.export_interactions(style="concat")
    assert list(leaves) == ["2", "3"]
    assert leaves["2"].parent is leaves["3"].parent is cache["1"]
    if materialize_first:
        for leaf in leaves.values():
            leaf.to_tensor_dict()
    session = SessionData(session_id="retry-episode")
    session._completions = cache
    session.finish()
    monkeypatch.setattr(server_module, "_session_cache", {session.session_id: session})
    monkeypatch.setattr(server_module, "_prm_runner", None)

    response = await server_module.export_trajectories(
        ExportTrajectoriesRequest(
            session_id=session.session_id, style="concat", episode_reward=0.8
        )
    )
    restored = deserialize_interactions(response.interactions)

    assert [leaf.reward for leaf in restored.values()] == [0.0, 0.8]
    references = [leaf.rollout_reward for leaf in restored.values()]
    peer = {"peer": InteractionWithTokenLogpReward(reward=0.4)}
    assert normalize_logical_rollout_rewards([restored, peer], use_std=use_std)
    assert references == [0.8, 0.8]
    tensors = concat_tensor_interactions(restored)
    scale = 0.2 + 1e-8 if use_std else 1.0
    torch.testing.assert_close(
        tensors["rewards"],
        torch.tensor([-0.6, 0.2]) / scale,
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        tensors["original_rewards"], torch.tensor([0.0, 0.8]), rtol=0, atol=0
    )
    assert tensors["rollout_group"].rewards == pytest.approx((0.2 / scale,))
    stats_tracker.export_all(reset=True)


def test_episode_reference_preserves_differing_cached_rows_across_rpc():
    """A scalar leaf may already contain several physical rows with own returns."""
    cache = _compacted_session((1,))
    cache.set_last_reward(0.8)
    leaf = cache["1"]
    tensors = leaf.to_tensor_dict()
    leaf._cache = {
        key: value.repeat((2,) + (1,) * (value.ndim - 1))
        for key, value in tensors.items()
    }
    leaf._cache["rewards"] = torch.tensor([0.2, 0.8])
    leaf._cache["original_rewards"] = torch.tensor([0.2, 0.8])

    exported = finalize_interactions(cache, style="concat", episode_reward=0.8)
    restored = deserialize_interactions(serialize_interactions(exported))

    assert restored["1"].rollout_reward == 0.8
    peer = {"peer": InteractionWithTokenLogpReward(reward=0.4)}
    assert normalize_logical_rollout_rewards([restored, peer], use_std=False)
    tensors = concat_tensor_interactions(restored)
    assert tensors["rollout_group"].row_counts == (2,)
    torch.testing.assert_close(
        tensors["rewards"], torch.tensor([-0.4, 0.2]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        tensors["original_rewards"], torch.tensor([0.2, 0.8]), rtol=0, atol=0
    )


@pytest.mark.parametrize("references", [(2.0, None, None), (2.0, 3.0, None)])
def test_episode_export_preserves_existing_references_and_conflicts(references):
    """An episode hook cannot silently replace a producer's explicit reference."""
    cache = _compacted_session((2, 1, 2))
    cache.set_reward("2", 0.2)
    cache.set_reward("3", 0.6)
    cache.set_last_reward(0.8)
    for leaf, reference in zip(
        cache.export_interactions(style="concat").values(), references
    ):
        leaf.rollout_reward = reference

    exported = finalize_interactions(cache, style="concat", episode_reward=0.8)

    assert [leaf.reward for leaf in exported.values()] == [0.2, 0.6, 0.8]
    assert tuple(leaf.rollout_reward for leaf in exported.values()) == references
    if references[1] is not None:
        with pytest.raises(ValueError, match="must share rollout_reward"):
            normalize_logical_rollout_rewards([exported])
    else:
        assert concat_tensor_interactions(exported)["rollout_group"].rewards == (2.0,)


@pytest.mark.parametrize("invalid_terminal", [False, True])
def test_equally_scored_segments_keep_episode_response_validation(invalid_terminal):
    """Already scored segments still share the true terminal response gate."""
    cache = _compacted_session((1, 1))
    cache.set_reward("1", 0.8)
    cache.set_last_reward(0.8)
    # A summary segment may end without user-facing content; the episode's
    # actual terminal response alone determines the shared validation result.
    for interaction_id in ["1", "2"] if invalid_terminal else ["1"]:
        cache[interaction_id].metadata["response_parse"] = {
            "tool_call_parse_failed": False,
            "has_content": False,
            "has_tool_calls": False,
        }
    cache.move_to_end("1")  # Export order must not determine the terminal ID.

    exported = finalize_interactions(cache, style="concat", episode_reward=0.8)

    expected = 0.0 if invalid_terminal else 0.8
    assert [leaf.reward for leaf in exported.values()] == [expected, expected]
    assert [leaf.rollout_reward for leaf in exported.values()] == [expected, expected]


@pytest.mark.parametrize("token_rewards", [False, True])
def test_episode_reference_does_not_score_unscored_shared_branch(token_rewards):
    """A normalization reference does not admit an ambiguously unscored row."""
    cache = _compacted_session((2, 1))
    cache["3"].parent = cache["1"]
    if token_rewards:
        cache["1"].token_rewards = torch.tensor([0.25])
    cache.set_last_reward(0.8)

    exported = finalize_interactions(cache, style="concat", episode_reward=0.8)

    assert [leaf.reward for leaf in exported.values()] == [None, 0.8]
    assert [leaf.rollout_reward for leaf in exported.values()] == [0.8, 0.8]
    assert exported["2"].parent is exported["3"].parent is cache["1"]
    if token_rewards:
        torch.testing.assert_close(
            cache["1"].token_rewards, torch.tensor([0.25]), rtol=0, atol=0
        )
    assert not normalize_logical_rollout_rewards([exported])


def test_episode_reference_requires_identifiable_terminal_response():
    """Several scored leaves do not identify which response ended the episode."""
    cache = _compacted_session((1, 1))
    cache.set_reward("1", 0.2)
    cache.set_reward("2", 0.8)

    with pytest.raises(ValueError, match="requires a final response ID"):
        finalize_interactions(cache, style="concat", episode_reward=0.8)

    assert [leaf.reward for leaf in cache.values()] == [0.2, 0.8]
    assert all(leaf.rollout_reward is None for leaf in cache.values())


@pytest.mark.parametrize("reference", [None, 0.8])
def test_discounted_episode_export_does_not_reapply_policy_or_infer_reference(
    reference,
):
    """Late export preserves discounted returns or rejects a missing reference."""
    cache = _compacted_session((1, 1))
    cache.set_last_reward(0.8)
    cache["2"].rollout_reward = reference
    cache["2"].metadata["response_parse"] = {
        "tool_call_parse_failed": True,
        "has_content": False,
        "has_tool_calls": False,
    }
    cache.apply_reward_discount(turn_discount=0.5)
    before = [(leaf.reward, leaf.rollout_reward) for leaf in cache.values()]

    if reference is None:
        with pytest.raises(ValueError, match="before reward discounting"):
            finalize_interactions(cache, style="concat", episode_reward=0.8)
    else:
        exported = finalize_interactions(cache, style="concat", episode_reward=0.8)
        assert exported == dict(cache)

    assert [(leaf.reward, leaf.rollout_reward) for leaf in cache.values()] == before


@pytest.mark.parametrize(
    ("ambiguity", "reason"),
    [
        ("shared_ancestor", "shared_ancestor"),
        ("scalar_conflict", "reward_conflict"),
        ("reference_conflict", "rollout_reward_conflict"),
        ("original_reward_conflict", "original_reward_conflict"),
        ("token_rewards", "token_rewards"),
        ("ancestor_reward", "ancestor_reward"),
        ("matching_ancestor_reward", "ancestor_reward"),
        ("incomplete", "incomplete"),
        ("missing_id", "missing_id"),
        ("duplicate_id", "duplicate_id"),
        ("non_concat", "non_concat"),
        ("cycle", "cycle"),
    ],
)
def test_arena_episode_reward_leaves_ambiguous_exports_unchanged(
    monkeypatch, ambiguity, reason
):
    warning = Mock()
    monkeypatch.setattr("areal.experimental.openai.cache.logger.warning", warning)
    cache = _compacted_session()
    cache.set_last_reward(0.5)
    leaves = cache.export_interactions(style="concat")
    if ambiguity == "shared_ancestor":
        leaves["72"].parent = cache["1"]
    elif ambiguity == "scalar_conflict":
        leaves["72"].reward = 0.25
    elif ambiguity == "reference_conflict":
        leaves["72"].rollout_reward = 0.25
    elif ambiguity == "original_reward_conflict":
        leaves["72"].original_reward = 0.25
    elif ambiguity == "token_rewards":
        cache["1"].token_rewards = torch.tensor([0.25])
    elif ambiguity == "ancestor_reward":
        cache["1"].reward = 0.25
    elif ambiguity == "matching_ancestor_reward":
        cache["1"].reward = 0.5
    elif ambiguity == "incomplete":
        cache["1"].output_message_list = None
    elif ambiguity == "missing_id":
        cache["1"].interaction_id = None
    elif ambiguity == "duplicate_id":
        cache["1"].interaction_id = "2"
    elif ambiguity == "non_concat":
        cache["1"].chat_template_type = "individual"
    elif ambiguity == "cycle":
        cache["1"].parent = cache["71"]
    before = [(x.reward, x.rollout_reward, x.original_reward) for x in cache.values()]
    InteractionCache.from_dict(leaves).assign_episode_reward(0.5)

    assert [
        (x.reward, x.rollout_reward, x.original_reward) for x in cache.values()
    ] == before
    assert leaves["71"].reward is None
    warning.assert_called_once()
    assert warning.call_args.args[2] == reason
    assert warning.call_args.args[-1] == len(leaves)


def test_arena_episode_reward_rejection_log_bounds_ids_without_message_content(
    monkeypatch,
):
    warning = Mock()
    monkeypatch.setattr("areal.experimental.openai.cache.logger.warning", warning)
    cache = _compacted_session(chain_lengths=(1, 1))
    leaf = cache["1"]
    leaf.interaction_id = "x" * 1000
    leaf.messages = [{"role": "user", "content": "private-image-payload"}]
    leaf.token_rewards = torch.tensor([0.25])

    InteractionCache.from_dict({"leaf": leaf}).assign_episode_reward(0.5)

    warning.assert_called_once()
    fmt, *args = warning.call_args.args
    message = fmt % tuple(args)
    assert "reason=token_rewards" in message
    assert "x" * 80 in message
    assert "x" * 81 not in message
    assert "private-image-payload" not in message
    assert leaf.reward is None


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

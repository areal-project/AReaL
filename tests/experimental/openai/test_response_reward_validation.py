# SPDX-License-Identifier: Apache-2.0

"""Invalid retained responses must not inherit a successful task score."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from areal.api import ModelResponse
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.client import ArealOpenAI
from areal.experimental.openai.finalization import (
    final_response_error,
    finalize_interactions,
)
from areal.experimental.openai.proxy.server import (
    SessionData,
    deserialize_interactions,
    serialize_interactions,
)
from areal.experimental.openai.tool_call_parser import InvalidToolCallError
from areal.experimental.openai.types import InteractionWithTokenLogpReward

_TOOLS = [
    {
        "type": "function",
        "function": {"name": "Ping", "parameters": {"type": "object"}},
    }
]
_CALL = "<tool_call><function=Ping></function></tool_call>"


def _interaction(
    name: str,
    *,
    messages: list[dict] | None = None,
    error: str | None = None,
) -> InteractionWithTokenLogpReward:
    messages = messages or [{"role": "user", "content": "task"}]
    return InteractionWithTokenLogpReward(
        _interaction_id=name,
        messages=messages,
        output_message_list=[{"role": "assistant", "content": name}],
        model_response=ModelResponse(
            input_tokens=[1] * (10 * len(messages)),
            output_tokens=[2, 3],
            output_logprobs=[-0.1, -0.2],
            output_versions=[0, 0],
        ),
        chat_template_type="concat",
        metadata={
            "response_parse": {
                "tool_call_parse_failed": error == "invalid_tool_call",
                "has_content": error != "empty_assistant_response",
                "has_tool_calls": False,
            }
        },
    )


def _child(
    name: str,
    parent: InteractionWithTokenLogpReward,
    *,
    error: str | None = None,
) -> InteractionWithTokenLogpReward:
    child = _interaction(
        name,
        messages=parent.messages
        + parent.output_message_list
        + [{"role": "user", "content": "continue"}],
        error=error,
    )
    # Concat inherits action metadata only for an unchanged sampled prefix.
    prefix = parent.model_response.input_tokens + parent.model_response.output_tokens
    child.model_response.input_tokens[: len(prefix)] = prefix
    return child


def _client_with_cache(cache: InteractionCache) -> ArealOpenAI:
    # Reward/export methods do not require a live HTTP or generation engine.
    client = object.__new__(ArealOpenAI)
    client._cache = cache
    client.response_validator = final_response_error
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "stream", "responses"])
@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ("Done.", None),
        (" \n\t", "empty_assistant_response"),
        ("<think>unfinished", "empty_assistant_response"),
        ("<think>reasoning</think>", "empty_assistant_response"),
        ("<think>" * 10, "empty_assistant_response"),
        (_CALL, None),
        (_CALL + "<tool_call", "invalid_tool_call"),
        ("Delivered.<tool_call><function=Ping>", "invalid_tool_call"),
        pytest.param("bad JSON", "invalid_tool_call", id="json-decode-error"),
    ],
)
async def test_client_export_validates_parsed_response(monkeypatch, api, raw, error):
    """Both API paths keep malformed tokens but zero their task outcome."""
    tokenizer = SimpleNamespace(
        eos_token_id=0,
        pad_token_id=0,
        decode=lambda ids: raw,
        encode=lambda text, **kwargs: list(range(len(text))),
    )
    response = ModelResponse(
        input_tokens=[1],
        output_tokens=[2, 0],
        output_logprobs=[-0.1, -0.2],
        output_versions=[0, 0],
        tokenizer=tokenizer,
    )
    monkeypatch.setattr(
        "areal.experimental.openai.client.apply_chat_template", lambda *a, **kw: [1]
    )
    if raw == "bad JSON":

        def fail_parse(*args, **kwargs):
            raise json.JSONDecodeError("bad tool args", "{", 1)

        monkeypatch.setattr(
            "areal.experimental.openai.client.process_tool_calls", fail_parse
        )
    async with ArealOpenAI(
        engine=SimpleNamespace(agenerate=AsyncMock(return_value=response)),
        tokenizer=tokenizer,
        api_key="test-key",
        chat_template_type="concat",
        tool_call_parser="qwen3_coder",
        tool_call_parser_allow_recovery=False,
        reasoning_parser="qwen3" if raw.startswith("<think>") else "",
    ) as client:

        async def request():
            if api == "responses":
                return await client.responses.create(
                    input="task",
                    tools=[{"type": "function", **tool["function"]} for tool in _TOOLS],
                    max_output_tokens=32,
                )
            return await client.chat.completions.create(
                messages=[{"role": "user", "content": "task"}],
                tools=_TOOLS,
                max_completion_tokens=32,
                stream=api == "stream",
            )

        if error == "invalid_tool_call" and raw != "bad JSON":
            with pytest.raises(InvalidToolCallError):
                await request()
        else:
            result = await request()
            if api == "stream":
                # Export must also work if the consumer never reads the stream.
                await result.aclose()

        cache = client._cache
        interaction = cache[cache.last_interaction_id]
        assert final_response_error(interaction) == error
        assert "reward_validation_error" not in interaction.metadata
        assert interaction.metadata["response_parse"]["tool_call_parse_failed"] == (
            error == "invalid_tool_call"
        )
        client.set_last_reward(0.8)
        assert interaction.reward == 0.8  # Gate is deferred until retry cleanup.
        exported = client.export_interactions(style="concat", drop_retry_orphans=True)
        expected = 0.0 if error else 0.8
        assert exported[interaction.interaction_id].reward == expected
        tensors = interaction.to_tensor_dict()
        for key in ("rewards", "original_rewards"):
            torch.testing.assert_close(
                tensors[key], torch.tensor([expected]), rtol=0, atol=0
            )
        torch.testing.assert_close(
            tensors["loss_mask"], torch.tensor([[0, 1, 1]]), rtol=0, atol=0
        )


@pytest.mark.parametrize("reward", [1.0, 0.0, -0.5])
@pytest.mark.parametrize("materialize_first", [False, True])
def test_invalid_final_response_zeroes_full_chain_and_task_metrics(
    reward, materialize_first
):
    """Ship success followed by a broken final response is a zero-score sample."""
    cache = InteractionCache()
    root = _interaction("ship")
    leaf = _child("final", root, error="empty_assistant_response")
    cache["ship"] = root
    cache["final"] = leaf
    cache.set_last_reward(reward)
    leaf.original_reward = reward
    leaf.turn_reward = 0.25
    leaf.token_rewards = torch.tensor([0.1, 0.2])
    before = leaf.to_tensor_dict() if materialize_first else None

    exported = finalize_interactions(cache, style="concat", drop_retry_orphans=True)
    assert list(exported) == ["final"]
    assert leaf.reward == leaf.original_reward == cache.total_reward == 0.0
    assert leaf.metadata["reward_before_response_validation"] == reward
    tensors = leaf.to_tensor_dict()
    for key in ("rewards", "original_rewards"):
        torch.testing.assert_close(tensors[key], torch.tensor([0.0]), rtol=0, atol=0)
    assert tensors["loss_mask"].sum() == 4  # Both turns still train.
    assert leaf.turn_reward == 0.25
    torch.testing.assert_close(
        leaf.token_rewards, torch.tensor([0.1, 0.2]), rtol=0, atol=0
    )
    if before is not None:
        for key in (
            "input_ids",
            "loss_mask",
            "logprobs",
            "token_rewards",
        ):
            torch.testing.assert_close(tensors[key], before[key], rtol=0, atol=0)

    restored = deserialize_interactions(serialize_interactions(exported))["final"]
    assert restored.reward == 0.0
    assert restored.metadata["reward_before_response_validation"] == reward
    torch.testing.assert_close(
        restored.to_tensor_dict()["original_rewards"],
        torch.tensor([0.0]),
        rtol=0,
        atol=0,
    )
    finalize_interactions(cache, style="concat")
    assert leaf.metadata["reward_before_response_validation"] == reward


@pytest.mark.parametrize("continued", [False, True])
def test_unused_invalid_retry_does_not_zero_successful_trajectory(continued):
    """Remove failed siblings before checking the retained final response."""
    cache = InteractionCache()
    orphan = _interaction("orphan", error="invalid_tool_call")
    retry = _interaction("retry")
    cache["orphan"] = orphan
    cache["retry"] = retry
    if continued:
        leaf = _child("final", retry)
        cache["final"] = leaf
    else:
        leaf = retry
    cache.set_last_reward(1.0)

    exported = finalize_interactions(cache, style="concat", drop_retry_orphans=True)

    assert list(exported) == [leaf.interaction_id]
    assert "orphan" not in cache
    assert leaf.reward == cache.total_reward == 1.0
    assert "reward_validation_error" not in orphan.metadata


def test_invalid_leaf_does_not_zero_valid_sibling_or_shared_parent():
    """Branch outcomes remain independent when only one final answer is invalid."""
    root = _interaction("root")
    valid = _child("valid", root)
    invalid = _child("invalid", root, error="invalid_tool_call")
    cache = InteractionCache()
    for interaction in (root, valid, invalid):
        cache[interaction.interaction_id] = interaction
        cache.set_reward(interaction.interaction_id, 1.0)

    exported = finalize_interactions(cache, style="concat")

    assert set(exported) == {"valid", "invalid"}
    assert root.reward == valid.reward == 1.0
    assert invalid.reward == 0.0
    assert cache.total_reward == 2.0


def test_concat_validation_checks_final_response_only():
    """An earlier retained error does not change the agreed final-answer policy."""
    root = _interaction("root", error="empty_assistant_response")
    leaf = _child("final", root)
    cache = InteractionCache()
    cache["root"] = root
    cache["final"] = leaf
    cache.set_last_reward(1.0)

    exported = finalize_interactions(cache, style="concat", drop_retry_orphans=True)

    assert exported["final"].reward == 1.0
    assert "reward_validation_error" not in root.metadata


@pytest.mark.parametrize("explicit_discount", [False, True])
def test_invalid_final_response_is_zeroed_before_individual_discount(explicit_discount):
    """A broken last response cannot propagate a successful task score backward."""
    root = _interaction("root")
    leaf = _child("final", root, error="invalid_tool_call")
    cache = InteractionCache()
    cache["root"] = root
    cache["final"] = leaf
    cache.set_last_reward(1.0)
    client = _client_with_cache(cache)
    if explicit_discount:
        client.apply_reward_discount(turn_discount=0.9, drop_retry_orphans=True)

    exported = client.export_interactions(
        style="individual",
        reward_discount=0.9,
        drop_retry_orphans=not explicit_discount,
    )

    assert all(interaction.reward == 0.0 for interaction in exported.values())


@pytest.mark.parametrize("materialize_first", [False, True])
def test_export_after_explicit_discount_preserves_return_and_zero_task_score(
    materialize_first,
):
    """Later returns must not be erased by revalidating an earlier invalid turn."""
    cache = InteractionCache()
    root = _interaction("root", error="empty_assistant_response")
    leaf = _child("final", root)
    cache["root"] = root
    cache["final"] = leaf
    cache.set_reward("root", 0.7)
    cache.set_reward("final", 1.0)
    if materialize_first:
        root.to_tensor_dict()
    client = _client_with_cache(cache)

    client.apply_reward_discount(turn_discount=0.9)
    exported = client.export_interactions(style="individual")
    exported = client.export_interactions(style="individual")

    assert exported["root"].reward == pytest.approx(0.9)
    assert exported["root"].original_reward == 0.0
    assert exported["root"].metadata["reward_before_response_validation"] == 0.7
    tensors = exported["root"].to_tensor_dict()
    torch.testing.assert_close(tensors["rewards"], torch.tensor([0.9]), rtol=0, atol=0)
    torch.testing.assert_close(
        tensors["original_rewards"], torch.tensor([0.0]), rtol=0, atol=0
    )


def test_generic_cache_does_not_apply_response_reward_policy():
    """Cache operations only select and propagate rewards, independent of policy."""
    cache = InteractionCache()
    cache["final"] = _interaction("final", error="invalid_tool_call")
    cache.set_last_reward(1.0)

    assert cache.export_interactions(style="concat")["final"].reward == 1.0
    assert finalize_interactions(cache, style="concat")["final"].reward == 0.0


@pytest.mark.parametrize("validator", [None, lambda interaction: None])
def test_response_reward_policy_can_be_disabled_or_replaced(validator):
    """A workflow may accept reasoning-only termination without changing parsers."""
    cache = InteractionCache()
    cache["final"] = _interaction("final", error="empty_assistant_response")
    cache.set_last_reward(1.0)
    client = _client_with_cache(cache)
    client.response_validator = validator

    exported = client.export_interactions(style="concat")

    assert exported["final"].reward == 1.0
    assert "reward_validation_error" not in exported["final"].metadata


def test_missing_parse_facts_does_not_imply_invalid_response():
    """External traces and legacy interactions are unclassified, not invalid."""
    cache = InteractionCache()
    interaction = _interaction("external")
    interaction.metadata = None
    interaction.output_message_list = []
    cache["external"] = interaction
    cache.set_last_reward(1.0)

    exported = finalize_interactions(cache, style="concat")

    assert exported["external"].reward == 1.0
    assert exported["external"].metadata is None


def test_proxy_export_runs_policy_after_retry_cleanup():
    """The server entry point gates its retained final response before serialization."""
    session = SessionData(session_id="test-session")
    cache = session.completions
    cache["orphan"] = _interaction("orphan", error="invalid_tool_call")
    cache["retry"] = _interaction("retry")
    cache["final"] = _child("final", cache["retry"], error="empty_assistant_response")
    cache.set_last_reward(1.0)

    exported = session.export_interactions(
        discount=1.0, style="concat", drop_retry_orphans=True
    )

    assert list(exported) == ["final"]
    assert "orphan" not in cache
    assert exported["final"].reward == 0.0


def test_v2_export_finalizes_each_reward_bounded_trajectory():
    """One online session can export an invalid trajectory then a valid one."""
    from areal.v2.inference_service.data_proxy.session import (
        SessionData as V2SessionData,
    )

    session = V2SessionData(session_id="test-session")
    for index, error in enumerate(("empty_assistant_response", None)):
        name = f"final-{index}"
        session.active_completions[name] = _interaction(name, error=error)
        reward_result = session.set_reward(interaction_id=name, reward=1.0)
        trajectory_id, exported = session.export_trajectory(
            discount=1.0, style="concat"
        )

        assert trajectory_id == reward_result.trajectory_id
        assert list(exported) == [name]
        assert exported[name].reward == (0.0 if error else 1.0)


@pytest.mark.parametrize(
    "error", [None, "invalid_tool_call", "empty_assistant_response"]
)
@pytest.mark.parametrize("materialized", [False, True])
@pytest.mark.parametrize("early_error", [None, "empty_assistant_response"])
def test_episode_return_is_validated_after_assignment(error, materialized, early_error):
    """Compaction cannot revive an invalid final or reject a valid continuation."""
    cache = InteractionCache()
    early = _interaction(
        "early",
        messages=[{"role": "user", "content": "before compaction"}],
        error=early_error,
    )
    final = _interaction(
        "final",
        messages=[{"role": "user", "content": "compacted context"}],
        error=error,
    )
    cache["early"] = early
    cache["final"] = final
    cache.set_last_reward(1.0)
    if materialized:
        before = final.to_tensor_dict()
        tokens = before["input_ids"].clone()
        mask = before["loss_mask"].clone()
    result = finalize_interactions(cache, style="concat", episode_reward=1.0)
    expected = 0.0 if error else 1.0
    for item in result.values():
        assert item.reward == expected
        assert item.rollout_reward == expected
        tensors = item.to_tensor_dict()
        assert tensors["rewards"].item() == expected
        if error:
            assert item.original_reward == 0.0
            assert item.metadata["reward_before_response_validation"] == 1.0
            assert item.metadata["reward_validation_error"] == error
    tensors = final.to_tensor_dict()
    if materialized:
        assert torch.equal(tensors["input_ids"], tokens)
        assert torch.equal(tensors["loss_mask"], mask)


def test_episode_return_rejects_individual_export():
    cache = InteractionCache()
    cache["final"] = _interaction("final")
    with pytest.raises(ValueError, match="concat"):
        finalize_interactions(cache, style="individual", episode_reward=1.0)


@pytest.mark.parametrize("error", [None, "invalid_tool_call"])
def test_episode_only_return_accepts_unambiguous_single_leaf(error):
    cache = InteractionCache()
    cache["final"] = _interaction("final", error=error)
    result = finalize_interactions(cache, style="concat", episode_reward=1.0)
    assert result["final"].reward == (0.0 if error else 1.0)


@pytest.mark.parametrize("error", [None, "invalid_tool_call"])
def test_proxy_last_reward_identifies_final_of_prescored_segments(monkeypatch, error):
    from areal.experimental.openai.proxy import proxy_rollout_server as srv

    session = SessionData(session_id="episode-terminal")
    early = _interaction("early", messages=[{"role": "user", "content": "old"}])
    final = _interaction(
        "final", messages=[{"role": "user", "content": "new"}], error=error
    )
    session.completions["early"] = early
    session.completions["final"] = final
    session.completions.set_reward("early", 1.0)
    session.completions.set_reward("final", 1.0)
    monkeypatch.setitem(srv._session_cache, session.session_id, session)
    srv.set_reward(srv.SetRewardRequest(reward=1.0), session_id=session.session_id)
    result = session.export_interactions(1.0, "concat", episode_reward=1.0)
    assert all(item.reward == (0.0 if error else 1.0) for item in result.values())
    if error:
        assert all(item.rollout_reward == 0.0 for item in result.values())

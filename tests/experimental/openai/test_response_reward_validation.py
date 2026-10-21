# SPDX-License-Identifier: Apache-2.0

"""Invalid retained responses must not inherit a successful task score."""

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
    deserialize_interactions,
    serialize_interactions,
)
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
    return _interaction(
        name,
        messages=parent.messages
        + parent.output_message_list
        + [{"role": "user", "content": "continue"}],
        error=error,
    )


def _client_with_cache(cache: InteractionCache) -> ArealOpenAI:
    # Reward/export methods do not require a live HTTP or generation engine.
    client = object.__new__(ArealOpenAI)
    client._cache = cache
    client.response_validator = final_response_error
    return client


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

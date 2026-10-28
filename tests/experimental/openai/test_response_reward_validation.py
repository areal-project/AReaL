# SPDX-License-Identifier: Apache-2.0

"""Invalid retained responses must not inherit a successful task score."""

import pytest
import torch

from areal.api import ModelResponse
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.finalization import (
    finalize_interactions,
)
from areal.experimental.openai.proxy.server import (
    deserialize_interactions,
    serialize_interactions,
)
from areal.experimental.openai.types import InteractionWithTokenLogpReward


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

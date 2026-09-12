"""Small CPU-only tests for process rewards on interaction token spans."""

from __future__ import annotations

import pytest
import torch

from areal.api import ModelResponse
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward


def _interaction(
    interaction_id: str,
    input_tokens: list[int],
    output_tokens: list[int],
    *,
    parent: InteractionWithTokenLogpReward | None = None,
):
    response = ModelResponse(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_logprobs=[0.0] * len(output_tokens),
        output_versions=[0] * len(output_tokens),
    )
    interaction = InteractionWithTokenLogpReward(
        model_response=response,
        chat_template_type="concat",
        parent=parent,
        messages=[{"role": "user", "content": "x"}],
        output_message_list=[{"role": "assistant", "content": "y"}],
    )
    interaction._interaction_id = interaction_id
    return interaction


def test_concat_stitches_each_turn_reward_into_its_output_span():
    parent = _interaction("parent", [1, 2], [3, 4])
    parent.token_rewards = torch.tensor([0.5, 0.0])
    leaf = _interaction("leaf", [1, 2, 3, 4, 5], [6, 7], parent=parent)
    leaf.token_rewards = torch.tensor([0.0, -1.0])

    actual = leaf.to_tensor_dict()["token_rewards"].squeeze(0)

    torch.testing.assert_close(
        actual,
        torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, -1.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_no_process_signal_preserves_original_tensor_schema():
    parent = _interaction("parent", [1, 2], [3])
    leaf = _interaction("leaf", [1, 2, 3, 4], [5], parent=parent)

    assert "token_rewards" not in leaf.to_tensor_dict()


def test_cache_accumulates_float32_detached_rewards_and_invalidates_tensor_cache():
    interaction = _interaction("turn", [1, 2], [3, 4])
    cache = InteractionCache.from_dict({"turn": interaction})
    first = interaction.to_tensor_dict()

    cache.add_token_rewards(
        "turn", torch.tensor([0.5, -0.5], dtype=torch.float64, requires_grad=True)
    )
    cache.add_token_rewards("turn", torch.tensor([0.25, 0.25]))

    assert interaction.token_rewards is not None
    assert interaction.token_rewards.dtype == torch.float32
    assert interaction.token_rewards.requires_grad is False
    assert interaction.to_tensor_dict() is not first
    torch.testing.assert_close(
        interaction.token_rewards,
        torch.tensor([0.75, -0.25]),
        rtol=0.0,
        atol=0.0,
    )


def test_parent_reward_update_invalidates_cached_descendant_tensorization():
    """A cached leaf must rebuild after an ancestor receives process rewards."""
    parent = _interaction("parent", [1, 2], [3, 4])
    leaf = _interaction("leaf", [1, 2, 3, 4, 5], [6], parent=parent)
    cache = InteractionCache.from_dict({"parent": parent, "leaf": leaf})
    first = leaf.to_tensor_dict()

    cache.add_token_rewards("parent", torch.tensor([0.25, -0.5]))
    second = leaf.to_tensor_dict()

    assert second is not first
    torch.testing.assert_close(
        second["token_rewards"].squeeze(0),
        torch.tensor([0.0, 0.0, 0.25, -0.5, 0.0, 0.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_clone_chain_returns_root_first_isolated_cache():
    """Branch cloning must preserve chronology without sharing mutable rewards."""
    root = _interaction("root", [1], [2])
    middle = _interaction("middle", [1, 2, 3], [4], parent=root)
    leaf = _interaction("leaf", [1, 2, 3, 4, 5], [6], parent=middle)
    root.token_rewards = torch.tensor([0.5])
    root.to_tensor_dict()
    leaf.to_tensor_dict()

    cache, cloned_leaf = InteractionCache.clone_chain(leaf, session_id="session")

    assert list(cache) == ["root", "middle", "leaf"]
    assert cloned_leaf is cache["leaf"]
    assert cloned_leaf is not leaf
    assert cloned_leaf.parent is cache["middle"]
    assert cache["middle"].parent is cache["root"]
    assert cache["root"] is not root
    assert all(interaction._cache is None for interaction in cache.values())
    assert cache["root"].token_rewards is not root.token_rewards
    cache["root"].messages[0]["content"] = "changed"
    assert root.messages[0]["content"] == "x"

    cache.add_token_rewards("root", torch.tensor([0.5]))
    torch.testing.assert_close(
        cache["root"].token_rewards,
        torch.tensor([1.0]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        root.token_rewards,
        torch.tensor([0.5]),
        rtol=0.0,
        atol=0.0,
    )


def test_tensorization_rejects_wrong_output_shape():
    interaction = _interaction("turn", [1, 2], [3, 4])
    interaction.token_rewards = torch.tensor([1.0])

    with pytest.raises(ValueError, match="token_rewards must be shape"):
        interaction.to_tensor_dict()


def test_v1_export_filters_incomplete_interaction():
    complete = _interaction("complete", [1], [2])
    incomplete = InteractionWithTokenLogpReward(
        model_response=ModelResponse(input_tokens=[1], output_tokens=[3]),
        chat_template_type="concat",
        messages=[{"role": "user", "content": "still running"}],
        output_message_list=None,
    )
    cache = InteractionCache.from_dict(
        {"complete": complete, "in-flight-key": incomplete}
    )

    exported = cache.export_interactions(style="concat")

    assert list(exported) == ["complete"]

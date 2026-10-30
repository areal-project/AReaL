# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from areal.api import ModelResponse
from areal.experimental.openai.proxy.server import (
    deserialize_interactions,
    serialize_interactions,
)
from areal.experimental.openai.types import (
    InteractionWithTokenLogpReward,
    concat_tensor_interactions,
)


def _routing_rows(data, index=0):
    offsets = data["r3_cu_seqlens"]
    return data["routed_experts"][int(offsets[index]) : int(offsets[index + 1])].int()


def _response(
    *,
    input_tokens: list[int],
    output_tokens: list[int],
    routed: np.ndarray | None,
) -> ModelResponse:
    return ModelResponse(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_logprobs=[-0.1] * len(output_tokens),
        output_versions=[1] * len(output_tokens),
        routed_experts=routed,
    )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("serialize", [False, True])
def test_legacy_cached_leaf_mixes_with_packed_leaf_preserving_history(
    reverse, serialize
):
    def make_interaction(name, valid):
        interaction = InteractionWithTokenLogpReward(
            model_response=_response(
                input_tokens=[1, 2],
                output_tokens=[3, 4],
                routed=np.arange(12, dtype=np.int32).reshape(3, 4),
            ),
            reward=2.5,
            r3_num_moe_layers=2,
            r3_topk=2,
        )
        interaction.interaction_id = name
        interaction.to_tensor_dict()["r3_routing_valid"][0] = valid
        return interaction

    old = make_interaction("old", False)
    modern = make_interaction("new", True)
    old_cache = old._cache.copy()
    old_cache["routed_experts"] = old_cache["routed_experts"].unsqueeze(0).int()
    del old_cache["r3_cu_seqlens"]
    old._cache = old_cache
    old.model_response = None  # Old serialized caches need no original response.
    expected_routes = old_cache["routed_experts"].clone()
    interactions = {"old": old, "new": modern}
    if serialize:
        interactions = deserialize_interactions(serialize_interactions(interactions))
    if reverse:
        interactions = dict(reversed(list(interactions.items())))
    group = concat_tensor_interactions(interactions)
    assert group["r3_routing_valid"].tolist() == [
        name != "old" for name in interactions
    ]
    assert group["rewards"].tolist() == [2.5, 2.5]
    for row in range(2):
        torch.testing.assert_close(
            _routing_rows(group, row), expected_routes[0], rtol=0, atol=0
        )
    assert "r3_cu_seqlens" not in old_cache
    torch.testing.assert_close(
        old_cache["routed_experts"], expected_routes, rtol=0, atol=0
    )


def test_interaction_to_tensor_dict_includes_r3_routing():
    routed = np.arange(1, 1 + 3 * 4, dtype=np.int32).reshape(3, 4)
    interaction = InteractionWithTokenLogpReward(
        model_response=_response(
            input_tokens=[1, 2],
            output_tokens=[3, 4],
            routed=routed,
        ),
        reward=1.0,
        r3_num_moe_layers=2,
        r3_topk=2,
    )

    tensor_dict = interaction.to_tensor_dict()

    assert tensor_dict["routed_experts"].shape == (4, 2, 2)
    assert tensor_dict["routed_experts"].dtype == torch.int16
    assert tensor_dict["r3_routing_valid"].tolist() == [True]
    torch.testing.assert_close(
        _routing_rows(tensor_dict)[3],
        _routing_rows(tensor_dict)[2],
        rtol=0,
        atol=0,
    )


def test_concat_interaction_to_tensor_dict_concatenates_parent_routing():
    parent_routed = np.arange(1, 1 + 2 * 4, dtype=np.int32).reshape(2, 4)
    child_routed = np.arange(101, 101 + 5 * 4, dtype=np.int32).reshape(5, 4)
    parent = InteractionWithTokenLogpReward(
        model_response=_response(
            input_tokens=[1],
            output_tokens=[2, 3],
            routed=parent_routed,
        ),
        chat_template_type="concat",
        r3_num_moe_layers=2,
        r3_topk=2,
    )
    child = InteractionWithTokenLogpReward(
        model_response=_response(
            input_tokens=[1, 2, 3, 4],
            output_tokens=[5, 6],
            routed=child_routed,
        ),
        parent=parent,
        chat_template_type="concat",
        r3_num_moe_layers=2,
        r3_topk=2,
    )

    tensor_dict = child.to_tensor_dict()

    routed = _routing_rows(tensor_dict)
    assert routed.shape == (6, 2, 2)
    torch.testing.assert_close(
        routed[:2],
        _routing_rows(parent.to_tensor_dict())[:2],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        routed[2:],
        torch.as_tensor(
            np.stack(
                [
                    child_routed.reshape(5, 2, 2)[2],
                    child_routed.reshape(5, 2, 2)[3],
                    child_routed.reshape(5, 2, 2)[4],
                    child_routed.reshape(5, 2, 2)[4],
                ]
            )
        ).int(),
        rtol=0,
        atol=0,
    )
    assert tensor_dict["r3_routing_valid"].tolist() == [True]


@pytest.mark.parametrize("invalid_turn", [None, 0, 1, 2])
def test_concat_three_turns_preserves_historical_routes_and_validity(invalid_turn):
    """Each continuation replaces only its parent's unforwarded final row."""
    parent = None
    expected = None
    cached = []
    for turn in range(3):
        input_tokens = list(range(1, 2 * turn + 2))
        output_tokens = [2 * turn + 2]
        seq_len = len(input_tokens) + 1
        raw = np.arange(1, 1 + (seq_len - 1) * 4, dtype=np.int32).reshape(-1, 4)
        raw += 100 * turn
        if turn == invalid_turn:
            raw[:] = 0
        interaction = InteractionWithTokenLogpReward(
            model_response=_response(
                input_tokens=input_tokens, output_tokens=output_tokens, routed=raw
            ),
            parent=parent,
            chat_template_type="concat",
            r3_num_moe_layers=2,
            r3_topk=2,
        )
        result = interaction.to_tensor_dict()
        if invalid_turn is None:
            current = torch.as_tensor(raw).reshape(-1, 2, 2)
            current = torch.cat([current, current[-1:]])
            expected = (
                current
                if expected is None
                else torch.cat([expected[:-1], current[expected.shape[0] - 1 :]])
            )
            torch.testing.assert_close(_routing_rows(result), expected, rtol=0, atol=0)
        assert result["r3_routing_valid"].item() == (
            invalid_turn is None or turn < invalid_turn
        )
        cached.append(result["routed_experts"].clone())
        parent = interaction
    # Exporting a child must not mutate cached parent routing.
    ancestor = parent
    for snapshot in reversed(cached):
        torch.testing.assert_close(
            ancestor.to_tensor_dict()["routed_experts"], snapshot, rtol=0, atol=0
        )
        ancestor = ancestor.parent


def test_proxy_interaction_serialization_preserves_r3_tensor_cache():
    routed = np.arange(1, 1 + 3 * 4, dtype=np.int32).reshape(3, 4)
    interaction = InteractionWithTokenLogpReward(
        model_response=_response(
            input_tokens=[1, 2],
            output_tokens=[3, 4],
            routed=routed,
        ),
        reward=1.0,
    )
    interaction.interaction_id = "completion-1"

    serialized = serialize_interactions(
        {"completion-1": interaction},
        r3_num_moe_layers=2,
        r3_topk=2,
    )
    restored = deserialize_interactions(serialized)

    tensor_dict = restored["completion-1"].to_tensor_dict()
    assert tensor_dict["routed_experts"].shape == (4, 2, 2)
    assert tensor_dict["r3_routing_valid"].tolist() == [True]


@pytest.mark.parametrize("empty_output", [False, True])
@pytest.mark.parametrize("missing_first", [False, True])
@pytest.mark.parametrize("serialize", [False, True])
def test_group_export_mixed_missing_routes_retains_invalid_row(
    empty_output, missing_first, serialize
):
    """One missing-route leaf must not prevent its siblings from exporting."""
    valid = InteractionWithTokenLogpReward(
        model_response=_response(
            input_tokens=[1, 2],
            output_tokens=[3, 4],
            routed=np.arange(1, 13, dtype=np.int32).reshape(3, 4),
        ),
        reward=1.0,
        r3_num_moe_layers=2,
        r3_topk=2,
    )
    missing = InteractionWithTokenLogpReward(
        model_response=_response(
            input_tokens=[5, 6],
            output_tokens=[] if empty_output else [7],
            routed=None,
        ),
        reward=0.0,
        r3_num_moe_layers=2,
        r3_topk=2,
    )
    valid_routes = _routing_rows(valid.to_tensor_dict()).clone()
    interactions = {"valid": valid, "missing": missing}
    if missing_first:
        interactions = dict(reversed(list(interactions.items())))
    if serialize:
        interactions = deserialize_interactions(
            serialize_interactions(interactions, r3_num_moe_layers=2, r3_topk=2)
        )

    result = concat_tensor_interactions(interactions)

    missing_idx = 0 if missing_first else 1
    valid_idx = 1 - missing_idx
    assert result["r3_routing_valid"].tolist() == (
        [False, True] if missing_first else [True, False]
    )
    assert result["routed_experts"].shape == (6 if empty_output else 7, 2, 2)
    torch.testing.assert_close(
        _routing_rows(result, valid_idx), valid_routes, rtol=0, atol=0
    )
    torch.testing.assert_close(
        _routing_rows(result, missing_idx),
        torch.zeros_like(_routing_rows(result, missing_idx)),
        rtol=0,
        atol=0,
    )
    assert result["attention_mask"][missing_idx].sum().item() == (
        2 if empty_output else 3
    )


@pytest.mark.parametrize("missing_parent", [False, True])
def test_concat_missing_parent_or_child_routes_marks_trajectory_invalid(missing_parent):
    """Missing historical or current routes cannot become replay-safe."""
    parent = InteractionWithTokenLogpReward(
        model_response=_response(
            input_tokens=[1],
            output_tokens=[2, 3],
            routed=None if missing_parent else np.ones((2, 4), dtype=np.int32),
        ),
        chat_template_type="concat",
        r3_num_moe_layers=2,
        r3_topk=2,
    )
    child = InteractionWithTokenLogpReward(
        model_response=_response(
            input_tokens=[1, 2, 3, 4],
            output_tokens=[5, 6],
            routed=np.full((5, 4), 2, dtype=np.int32) if missing_parent else None,
        ),
        parent=parent,
        chat_template_type="concat",
        r3_num_moe_layers=2,
        r3_topk=2,
    )

    result = child.to_tensor_dict()

    assert result["routed_experts"].shape == (6, 2, 2)
    assert result["r3_routing_valid"].tolist() == [False]
    expected = torch.zeros((6, 2, 2), dtype=torch.int32)
    if missing_parent:
        expected[2:] = 2
    else:
        expected[:2] = 1
    torch.testing.assert_close(_routing_rows(result), expected, rtol=0, atol=0)


def test_interaction_missing_routes_without_r3_config_preserves_plain_export():
    """Ordinary non-R3 workflows must not receive replay side-channel keys."""
    interaction = InteractionWithTokenLogpReward(
        model_response=_response(input_tokens=[1], output_tokens=[2], routed=None)
    )

    result = interaction.to_tensor_dict()

    assert "routed_experts" not in result
    assert "r3_routing_valid" not in result


def test_serialization_configures_r3_for_cached_missing_route_leaf():
    """Late R3 configuration must rebuild a previously cached plain leaf."""
    valid = InteractionWithTokenLogpReward(
        model_response=_response(
            input_tokens=[1], output_tokens=[2], routed=np.ones((1, 4), dtype=np.int32)
        )
    )
    missing = InteractionWithTokenLogpReward(
        model_response=_response(input_tokens=[3], output_tokens=[], routed=None)
    )
    assert "routed_experts" not in missing.to_tensor_dict()

    restored = deserialize_interactions(
        serialize_interactions(
            {"valid": valid, "missing": missing},
            r3_num_moe_layers=2,
            r3_topk=2,
        )
    )
    result = concat_tensor_interactions(restored)

    assert result["r3_routing_valid"].tolist() == [True, False]
    torch.testing.assert_close(
        _routing_rows(result, 1),
        torch.zeros_like(_routing_rows(result, 1)),
        rtol=0,
        atol=0,
    )

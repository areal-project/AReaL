# SPDX-License-Identifier: Apache-2.0
"""Sampler heads across concat multi-turn interactions."""

import pytest
import torch

from areal.api import ModelResponse
from areal.experimental.openai.types import InteractionWithTokenLogpReward

NEG_INF = float("-inf")
PAD_IDS = [0, 0]
PAD_LOGPS = [NEG_INF, NEG_INF]

# One distinct head per generated token, so a misplaced row cannot go unnoticed.
HEADS = {
    21: ([21, 5], [-0.1, -2.5]),
    22: ([6, 22], [-0.7, -0.9]),
    41: ([41, 7], [-0.3, -1.6]),
    51: ([8, 51], [-0.5, -1.1]),
    52: ([52, 9], [-0.2, -1.9]),
}


def _turn(input_tokens, output_tokens, parent=None, top_k=2, heads=True):
    response = ModelResponse(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_logprobs=[0.0] * len(output_tokens),
        output_versions=[3] * len(output_tokens),
        sampler_head_ids=[list(HEADS[t][0]) for t in output_tokens] if heads else None,
        sampler_head_logprobs=(
            [list(HEADS[t][1]) for t in output_tokens] if heads else None
        ),
    )
    return InteractionWithTokenLogpReward(
        model_response=response,
        parent=parent,
        chat_template_type="concat",
        score_centering_top_k=top_k,
        reward=0.0,
    )


def _three_turns():
    # prompt [10, 11] -> [21, 22]; tool [30, 31] -> [41]; tool [32] -> [51, 52]
    first = _turn([10, 11], [21, 22])
    second = _turn([10, 11, 21, 22, 30, 31], [41], parent=first)
    third = _turn([10, 11, 21, 22, 30, 31, 41, 32], [51, 52], parent=second)
    return first, second, third


def test_concat_child_keeps_each_turns_head_on_its_own_tokens():
    """Each generated token keeps the head of the turn that sampled it."""
    *_, third = _three_turns()

    result = third.to_tensor_dict()

    seq = result["input_ids"].squeeze(0).tolist()
    assert seq == [10, 11, 21, 22, 30, 31, 41, 32, 51, 52]
    expected_ids = [HEADS[t][0] if t in HEADS else PAD_IDS for t in seq]
    expected_logps = [HEADS[t][1] if t in HEADS else PAD_LOGPS for t in seq]
    torch.testing.assert_close(
        result["sampler_head_ids"].squeeze(0),
        torch.tensor(expected_ids, dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        result["sampler_head_logprobs"].squeeze(0),
        torch.tensor(expected_logps, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    # Scored rows and non-empty heads are the same rows.
    scored = result["loss_mask"].squeeze(0).bool()
    has_head = torch.isfinite(result["sampler_head_logprobs"].squeeze(0)).any(-1)
    torch.testing.assert_close(scored, has_head, rtol=0, atol=0)


def test_concat_child_rejects_a_turn_without_heads():
    """A turn sampled without heads cannot be stitched next to one with heads."""
    first = _turn([10, 11], [21, 22])
    second = _turn([10, 11, 21, 22, 30, 31], [41], parent=first, heads=False)

    with pytest.raises(ValueError, match="must carry sampler heads"):
        second.to_tensor_dict()


def test_concat_child_rejects_heads_of_different_widths():
    """Without score_centering_top_k, turns of different widths must fail."""
    first = _turn([10, 11], [21, 22], top_k=0)
    second = _turn([10, 11, 21, 22, 30], [41], parent=first, top_k=0)
    second.model_response.sampler_head_ids[0].append(12)
    second.model_response.sampler_head_logprobs[0].append(-4.0)

    with pytest.raises(ValueError, match="differ in width"):
        second.to_tensor_dict()


def test_concat_child_that_drops_its_parent_keeps_only_its_own_heads():
    """When the parent is masked out, its rows must not leak into the child."""
    first = _turn([10, 11], [21, 22])
    # A child prompt no longer than the parent sequence drops the parent.
    second = _turn([10, 11, 30], [41], parent=first)

    result = second.to_tensor_dict()

    torch.testing.assert_close(
        result["sampler_head_ids"].squeeze(0),
        torch.tensor([PAD_IDS] * 3 + [HEADS[41][0]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.api import ModelResponse
from areal.experimental.openai.client import concat_prompt_token_ids_with_parent
from areal.experimental.openai.types import InteractionWithTokenLogpReward


def _make_interaction(
    input_tokens: list[int],
    output_tokens: list[int],
    output_logprobs: list[float],
    output_versions: list[int],
    parent: InteractionWithTokenLogpReward | None = None,
) -> InteractionWithTokenLogpReward:
    return InteractionWithTokenLogpReward(
        model_response=ModelResponse(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            output_logprobs=output_logprobs,
            output_versions=output_versions,
        ),
        parent=parent,
        chat_template_type="concat",
    )


def test_to_tensor_dict_mismatched_parent_prefix_raises():
    parent = _make_interaction([10], [20], [-0.2], [7])
    child = _make_interaction([10, 999, 30], [40], [-0.4], [8], parent)

    with pytest.raises(ValueError, match="does not match the parent token sequence"):
        child.to_tensor_dict()


def test_to_tensor_dict_matching_parent_prefix_preserves_parent_data():
    parent = _make_interaction([10], [20], [-0.2], [7])
    child = _make_interaction([10, 20, 30], [40], [-0.4], [8], parent)

    result = child.to_tensor_dict()

    expected = {
        "input_ids": torch.tensor([[10, 20, 30, 40]]),
        "loss_mask": torch.tensor([[0, 1, 0, 1]]),
        "logprobs": torch.tensor([[0.0, -0.2, 0.0, -0.4]]),
        "versions": torch.tensor([[-1, 7, -1, 8]]),
    }
    for key, expected_tensor in expected.items():
        torch.testing.assert_close(result[key], expected_tensor, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("child_input_tokens", "expected"),
    [
        (
            [10, 20],
            {
                "input_ids": torch.tensor([[10, 20, 40]]),
                "loss_mask": torch.tensor([[0, 0, 1]]),
                "logprobs": torch.tensor([[0.0, 0.0, -0.4]]),
                "versions": torch.tensor([[-1, -1, 8]]),
            },
        ),
        (
            [10],
            {
                "input_ids": torch.tensor([[10, 40]]),
                "loss_mask": torch.tensor([[0, 1]]),
                "logprobs": torch.tensor([[0.0, -0.4]]),
                "versions": torch.tensor([[-1, 8]]),
            },
        ),
    ],
    ids=["equal", "shorter"],
)
def test_to_tensor_dict_equal_or_shorter_child_ignores_parent(
    child_input_tokens: list[int], expected: dict[str, torch.Tensor]
):
    parent = _make_interaction([10], [20], [-0.2], [7])
    child = _make_interaction(child_input_tokens, [40], [-0.4], [8], parent)

    result = child.to_tensor_dict()

    for key, expected_tensor in expected.items():
        torch.testing.assert_close(result[key], expected_tensor, rtol=0.0, atol=0.0)


def test_concat_prompt_rejects_replaced_parent_stop_token(monkeypatch):
    class _Tokenizer:
        eos_token_id = 99
        pad_token_id = 0

    parent = _make_interaction([10], [20, 0], [-0.2, -0.3], [7, 7])
    parent.model_response.tokenizer = _Tokenizer()

    monkeypatch.setattr(
        "areal.experimental.openai.client.apply_chat_template",
        lambda *args, **kwargs: [10, 20, 99, 30],
    )

    with pytest.raises(ValueError, match="does not match the parent token sequence"):
        concat_prompt_token_ids_with_parent(
            message_list=[],
            parent=parent,
            tokenizer=_Tokenizer(),
        )

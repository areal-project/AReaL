# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.utils.data import concat_batch, concat_padded_tensors, split_batch


@pytest.mark.parametrize("repetitions", [1, 8])
def test_concat_batch_uneven_rollouts_preserves_values_and_round_trips(
    repetitions: int,
):
    """Batch grouped rollouts without changing masks, metadata, or row ordering."""
    rollouts = [
        {
            "input_ids": torch.tensor([[11, 12, 13], [21, 22, 23]]),
            "attention_mask": torch.tensor([[True, True, True], [True, True, False]]),
            "logprobs": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            "ids": ["a", "b"],
            "model": "shared",
        },
        {
            "input_ids": torch.tensor([[31, 32, 33, 34, 35]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.bool),
            "logprobs": torch.tensor([[0.7, 0.8, 0.9, 1.0, 1.1]]),
            "ids": ["c"],
            "model": "shared",
        },
    ]

    rollouts *= repetitions
    padded = concat_padded_tensors(rollouts, pad_value=-7)
    torch.testing.assert_close(
        padded["input_ids"],
        torch.tensor(
            [[11, 12, 13, -7, -7], [21, 22, 23, -7, -7], [31, 32, 33, 34, 35]]
        ).repeat(repetitions, 1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        padded["attention_mask"],
        torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, False, False, False],
                [True] * 5,
            ]
        ).repeat(repetitions, 1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        padded["logprobs"],
        torch.tensor(
            [
                [0.1, 0.2, 0.3, -7, -7],
                [0.4, 0.5, 0.6, -7, -7],
                [0.7, 0.8, 0.9, 1.0, 1.1],
            ]
        ).repeat(repetitions, 1),
        rtol=0,
        atol=0,
    )
    assert padded["ids"] == ["a", "b", "c"] * repetitions
    assert padded["model"] == "shared"

    batched, metadata = concat_batch(rollouts)
    restored = split_batch(batched, metadata)
    for actual, expected in zip(restored, rollouts, strict=True):
        for key in ("input_ids", "attention_mask", "logprobs"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert actual["model"] == expected["model"]


def test_concat_padded_tensors_multidimensional_inputs_pads_all_nonbatch_axes():
    """Pad every non-batch axis, including a rollout with no rows."""
    inputs = [
        {"x": torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])},
        {"x": torch.tensor([[[5.0, 6.0, 7.0]]])},
        {"x": torch.empty(0, 1, 2)},
    ]
    result = concat_padded_tensors(inputs * 8, pad_value=0.5)
    expected = torch.tensor(
        [
            [[1.0, 0.5, 0.5], [2.0, 0.5, 0.5]],
            [[3.0, 0.5, 0.5], [4.0, 0.5, 0.5]],
            [[5.0, 6.0, 7.0], [0.5, 0.5, 0.5]],
        ]
    )
    torch.testing.assert_close(result["x"], expected.repeat(8, 1, 1), rtol=0, atol=0)


def test_concat_padded_tensors_mixed_dtypes_preserves_padding_and_promotion():
    """Preserve per-input padding conversion before concatenation promotes dtype."""
    inputs = [
        {"x": torch.tensor([[1]], dtype=torch.int64)},
        {"x": torch.tensor([[2.5, 3.5]], dtype=torch.float64)},
    ]
    result = concat_padded_tensors(inputs * 8, pad_value=0.5)
    torch.testing.assert_close(
        result["x"],
        torch.tensor([[1.0, 0.0], [2.5, 3.5]], dtype=torch.float64).repeat(8, 1),
        rtol=0,
        atol=0,
    )


def test_concat_padded_tensors_transposed_inputs_preserves_values():
    """Accept strided views without changing their logical values."""
    inputs = [
        {"x": torch.arange(12).reshape(4, 3).t()},
        {"x": torch.arange(6).reshape(3, 2).t()},
    ]
    result = concat_padded_tensors(inputs * 8, pad_value=-1)
    torch.testing.assert_close(
        result["x"],
        torch.tensor(
            [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11], [0, 2, 4, -1], [1, 3, 5, -1]]
        ).repeat(8, 1),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("second_requires_grad", [False, True])
def test_concat_padded_tensors_differentiable_inputs_preserves_gradients(
    second_requires_grad: bool,
):
    """Backpropagate only through original elements, excluding padding."""
    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    second = torch.tensor([[5.0, 6.0, 7.0]], requires_grad=second_requires_grad)
    result = concat_padded_tensors([{"x": first}, {"x": second}] * 8, pad_value=0.5)
    (result["x"] * torch.arange(1, 10).reshape(3, 3).repeat(8, 1)).sum().backward()
    torch.testing.assert_close(
        first.grad, torch.tensor([[1.0, 2.0], [4.0, 5.0]]) * 8, rtol=0, atol=0
    )
    if second_requires_grad:
        torch.testing.assert_close(
            second.grad, torch.tensor([[7.0, 8.0, 9.0]]) * 8, rtol=0, atol=0
        )
    else:
        assert second.grad is None


def test_concat_padded_tensors_rollout_assembly_allocates_only_output_storage():
    """Avoid a batch of temporary padded tensors when assembling CPU rollouts."""
    rollouts = [
        {
            "input_ids": torch.arange(length).unsqueeze(0),
            "attention_mask": torch.ones(1, length, dtype=torch.bool),
        }
        for length in range(1, 33)
    ]
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
    ) as profile:
        result = concat_padded_tensors(rollouts)
    allocated_bytes = sum(
        max(0, event.self_cpu_memory_usage) for event in profile.events()
    )
    output_bytes = sum(tensor.nbytes for tensor in result.values())
    assert allocated_bytes <= output_bytes

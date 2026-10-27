# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch

from areal.api.cli_args import PPOActorConfig
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data import (
    batched_call,
    concat_batch,
    split_and_unpad_tensor,
    split_batch,
    split_training_batch_into_microbatches,
)


def _trajectory(group_size: int, seq_len: int, image_id: int, key: str) -> dict:
    image = torch.full((4, 3), float(image_id))
    return {
        "input_ids": torch.full((group_size, seq_len), image_id, dtype=torch.long),
        "attention_mask": torch.ones(group_size, seq_len, dtype=torch.bool),
        "loss_mask": torch.tensor([[0] + [1] * (seq_len - 1)] * group_size),
        "logprobs": torch.zeros(group_size, seq_len),
        "rewards": torch.ones(group_size),
        "is_truncated": torch.zeros(group_size, dtype=torch.bool),
        key: [{"pixel_values": image} for _ in range(group_size)],
    }


@pytest.mark.parametrize("key", ["multi_modal_input", "multi_modal_input_aux"])
def test_split_batch_multimodal_preserves_group_rows_and_unpadded_pixels(key):
    """Each trajectory owns only its image rows, independently of token lengths."""
    trajectories = [
        _trajectory(2, 2, 11, key),
        _trajectory(1, 5, 22, key),
        _trajectory(3, 3, 33, key),
    ]
    trajectories[1][key] = [{}]  # A row with no visual input still occupies a slot.

    batch, meta = concat_batch(trajectories)
    outputs = split_batch(batch, meta)

    assert [len(output[key]) for output in outputs] == [2, 1, 3]
    assert outputs[1][key] == [{}]
    for original, output in zip(trajectories, outputs, strict=True):
        torch.testing.assert_close(
            output["input_ids"], original["input_ids"], rtol=0, atol=0
        )
        for expected, actual in zip(original[key], output[key], strict=True):
            if expected:
                torch.testing.assert_close(
                    actual["pixel_values"], expected["pixel_values"], rtol=0, atol=0
                )

    # Preserve isolation without cloning any other trajectory's pixels.
    outputs[0][key][0]["pixel_values"].zero_()
    assert torch.all(trajectories[0][key][0]["pixel_values"] == 11)
    assert torch.all(outputs[2][key][0]["pixel_values"] == 33)


@pytest.mark.parametrize("n_trajs", [1, 2, 8])
def test_batched_call_multimodal_storage_scales_linearly(n_trajs):
    """Shared prompt images are copied once per group, not once per whole batch."""
    trajectories = [
        _trajectory(4, 3, i + 1, "multi_modal_input") for i in range(n_trajs)
    ]
    outputs = batched_call(lambda batch: batch, trajectories)

    tensors = [
        entry["pixel_values"]
        for output in outputs
        for entry in output["multi_modal_input"]
    ]
    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for tensor in tensors
    }
    assert len(storages) == n_trajs
    assert (
        sum(storages.values())
        == n_trajs * 4 * 3 * torch.tensor([], dtype=torch.float32).element_size()
    )
    assert len(tensors) == 4 * n_trajs
    for original, output in zip(trajectories, outputs, strict=True):
        pixels = output["multi_modal_input"][0]["pixel_values"]
        assert (
            pixels.data_ptr()
            != original["multi_modal_input"][0]["pixel_values"].data_ptr()
        )
        assert all(
            entry["pixel_values"] is pixels for entry in output["multi_modal_input"]
        )


def test_compute_advantages_keeps_multimodal_rows_for_rebatched_training():
    """Exercise the real PPO advantage path before regrouping into microbatches."""
    actor = PPOActor(PPOActorConfig(backend="fsdp:d1", kl_ctl=0.0), MagicMock())
    trajectories = [
        _trajectory(2, 3, 11, "multi_modal_input"),
        _trajectory(1, 5, 22, "multi_modal_input"),
        _trajectory(1, 4, 33, "multi_modal_input"),
    ]

    outputs = actor.compute_advantages(trajectories)
    assert [len(output["multi_modal_input"]) for output in outputs] == [2, 1, 1]
    assert all("advantages" in output for output in outputs)

    # A per-trajectory result must remain aligned when selected or reordered.
    rebatch, _ = concat_batch([outputs[2], outputs[0]])
    microbatches = split_training_batch_into_microbatches(rebatch, n_mbs=1)
    for microbatch in microbatches:
        for row, payload in zip(
            microbatch["input_ids"], microbatch["multi_modal_input"], strict=True
        ):
            assert torch.all(payload["pixel_values"] == row[0])


def test_split_batch_keeps_ordinary_list_metadata_independent():
    """A shared metadata list must not become row-aligned just by matching length."""
    metadata = [{"name": "first"}, {"name": "second"}]
    outputs = split_and_unpad_tensor(
        {"input_ids": torch.ones(2, 3), "metadata": metadata}, n_trajs=2
    )
    assert outputs[0]["metadata"] == metadata
    assert outputs[1]["metadata"] == metadata
    outputs[0]["metadata"][0]["name"] = "changed"
    assert outputs[1]["metadata"][0]["name"] == "first"
    assert metadata[0]["name"] == "first"


@pytest.mark.parametrize("payload_count", [0, 1, 3])
def test_split_multimodal_mismatched_row_count_raises(payload_count):
    """Malformed multimodal batches must not be silently sliced or duplicated."""
    with pytest.raises(ValueError, match="multi_modal_input.*2.*rows"):
        split_and_unpad_tensor(
            {"input_ids": torch.ones(2, 3), "multi_modal_input": [{}] * payload_count},
            n_trajs=2,
        )


def test_split_multimodal_preserves_empty_trajectory_groups():
    """An empty group consumes no image rows and leaves later offsets unchanged."""
    outputs = split_and_unpad_tensor(
        {
            "input_ids": torch.ones(3, 3),
            "multi_modal_input": [{"id": i} for i in range(3)],
        },
        n_trajs=3,
        traj_group_sizes=[0, 2, 1],
    )
    assert [output["multi_modal_input"] for output in outputs] == [
        [],
        [{"id": 0}, {"id": 1}],
        [{"id": 2}],
    ]

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.infra.workflow_executor import check_trajectory_format


def _trajectory(multi_modal_input: dict) -> dict:
    return {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "multi_modal_input": [multi_modal_input],
    }


@pytest.mark.parametrize(
    "multi_modal_input",
    [
        {
            "pixel_values": torch.randn(4, 8),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        {
            "pixel_values_videos": torch.randn(8, 8),
            "video_grid_thw": torch.tensor([[2, 2, 2]]),
        },
    ],
)
def test_check_trajectory_format_image_or_video_pixels_accepted(multi_modal_input):
    assert check_trajectory_format(_trajectory(multi_modal_input))


def test_check_trajectory_format_payload_without_pixels_rejected():
    with pytest.raises(ValueError, match="pixel_values_videos"):
        check_trajectory_format(
            _trajectory({"image_grid_thw": torch.tensor([[1, 2, 2]])})
        )

# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "areal_pacman", reason="Pinned external recipe is an optional example dependency"
)

from areal_pacman.level1.workflow import PacmanImageOnlyWorkflow

from examples.vlm.pacman.workflow import _Episode


@pytest.mark.asyncio
async def test_episode_adapter_preserves_images_support_and_complete_return(
    monkeypatch,
):
    async def run(episode, data):
        episode.last_episode = {
            "trajectory_sample_id": "episode-a",
            "total_shaped_reward": 9.0,
        }
        return {"first": 4.0, "second": 5.0}

    monkeypatch.setattr(PacmanImageOnlyWorkflow, "run", run)
    episode = object.__new__(_Episode)
    episode.owner = SimpleNamespace(
        options={"reward_objective_contract": "episode_return_group_v1"},
        gconfig=SimpleNamespace(n_samples=12),
    )
    episode.turns = {}
    for name, prompt, sampled, allowed in (
        ("first", [9, 8], 4, [1, 4]),
        ("second", [9, 8, 7], 1, [1, 2, 3]),
    ):
        processed = {
            "mm_token_type_ids": torch.tensor([[0] * len(prompt)]),
            "pixel_values": torch.ones((4, 2)),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }
        response = SimpleNamespace(
            input_tokens=prompt,
            output_tokens=[sampled],
            output_logprobs=[-0.5],
            output_versions=[2],
        )
        episode.turns[name] = processed, response, allowed
    result = await episode.collect({})
    torch.testing.assert_close(
        result["rewards"], torch.tensor([9.0, 9.0]), rtol=0, atol=0
    )
    assert result["episode_ids"].unique().numel() == 1
    torch.testing.assert_close(
        result["episode_group_sizes"],
        torch.tensor([12, 12], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    expected = torch.zeros((2, 4, 3), dtype=torch.long)
    expected[0, 1] = torch.tensor([2, 5, 0])
    expected[1, 2] = torch.tensor([2, 3, 4])
    torch.testing.assert_close(result["policy_support"], expected, rtol=0, atol=0)
    assert len(result["multi_modal_input"]) == 2
    for image in result["multi_modal_input"]:
        torch.testing.assert_close(
            image["pixel_values"], torch.ones((4, 2)), rtol=0, atol=0
        )

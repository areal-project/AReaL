# SPDX-License-Identifier: Apache-2.0

import asyncio
from copy import deepcopy
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "areal_pacman", reason="Pinned external recipe is an optional example dependency"
)

from areal_pacman.level1.workflow import (
    PacmanImageOnlyWorkflow,
    PacmanNativeVisionWorkflow,
)
from PIL import Image

from examples.vlm.pacman.workflow import PacmanWorkflow, _Episode

from areal.infra import workflow_context
from areal.infra.processor_cache import ProcessorCallCache
from areal.infra.rpc.rtensor import RTensor
from areal.utils.data import concat_padded_tensors
from areal.utils.image import image2base64


@pytest.fixture
def processor_episode():
    """Exercise the pinned message adapter with a small deterministic processor."""

    class Processor:
        def __init__(self):
            self.calls = 0
            self.lock = Lock()

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return messages[0]["content"][0]["text"]

        def __call__(self, *, text, images, **kwargs):
            assert kwargs == {
                "padding": False,
                "truncation": False,
                "return_tensors": "pt",
            }
            with self.lock:
                self.calls += 1
            return {
                "input_ids": torch.tensor([[9, len(text[0])]], dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[0, 1]], dtype=torch.long),
                "pixel_values": torch.tensor(
                    list(images[0].getdata()), dtype=torch.float32
                ),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
            }

    episode = object.__new__(_Episode)
    episode.processor = Processor()
    png = image2base64(Image.new("RGB", (2, 2), "red"))[0]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Choose an action"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{png}"},
                },
            ],
        }
    ]
    parent = workflow_context.get()
    cache = ProcessorCallCache()
    workflow_context.set(workflow_context.WorkflowContext(processor_cache=cache))
    try:
        yield episode, messages
    finally:
        current_cache = workflow_context.get().processor_cache
        if current_cache is not None:
            current_cache.close()
        cache.close()
        workflow_context.set(parent)


@pytest.mark.asyncio
async def test_processor_cache_group_shares_tensors_through_trajectory_and_rpc(
    processor_episode,
    monkeypatch,
):
    """Twelve worker-thread callers produce one processor call and one image shard."""
    episode, messages = processor_episode
    prepared = await asyncio.gather(
        *(
            asyncio.to_thread(episode._prepare_model_input, deepcopy(messages))
            for _ in range(12)
        )
    )
    assert episode.processor.calls == 1
    image, _, expected, expected_ids = PacmanNativeVisionWorkflow._process_messages(
        episode, messages
    )
    for image_data, processed, input_ids in prepared:
        assert image_data == image2base64(image)
        assert input_ids == expected_ids
        for key in expected:
            torch.testing.assert_close(processed[key], expected[key], rtol=0, atol=0)
        assert processed["pixel_values"] is prepared[0][1]["pixel_values"]
    assert prepared[0][0] is not prepared[1][0]
    assert prepared[0][1] is not prepared[1][1]
    assert prepared[0][2] is not prepared[1][2]
    prepared[0][1]["private_metadata"] = "first decision only"
    assert "private_metadata" not in episode._prepare_model_input(messages)[1]

    async def run(candidate, data):
        candidate.last_episode = {
            "trajectory_sample_id": "sample",
            "total_shaped_reward": 1.0,
        }
        return {"turn": 1.0}

    monkeypatch.setattr(PacmanImageOnlyWorkflow, "run", run)
    results = []
    for _, processed, input_ids in prepared:
        candidate = object.__new__(_Episode)
        candidate.owner = SimpleNamespace(
            options={"reward_objective_contract": "step_local_raw_v1"}
        )
        response = SimpleNamespace(
            input_tokens=input_ids,
            output_tokens=[1],
            output_logprobs=[-0.5],
            output_versions=[0],
        )
        candidate.turns = {"turn": (processed, response, [1, 2])}
        results.append(await candidate.collect({}))
    grouped = concat_padded_tensors(results)
    assert grouped["rewards"].shape == (12,)
    for multimodal in grouped["multi_modal_input"]:
        assert multimodal["pixel_values"] is prepared[0][1]["pixel_values"]

    stored = []

    def store(tensor):
        stored.append(tensor)
        return len(stored) - 1

    monkeypatch.setattr(
        "areal.infra.rpc.rtensor.get_backend", lambda: SimpleNamespace(store=store)
    )
    remote = RTensor.remotize(
        grouped, node_addr="test-node", preserve_tensor_aliases=True
    )
    multimodal = remote["multi_modal_input"]
    assert all(
        item["pixel_values"] is multimodal[0]["pixel_values"] for item in multimodal
    )
    assert sum(tensor.shape == expected["pixel_values"].shape for tensor in stored) == 1


@pytest.mark.parametrize("changed", ["text", "image", "processor", "group"])
def test_processor_cache_changed_input_is_not_reused(processor_episode, changed):
    """Changed observations, instructions, processors and groups cannot share results."""
    episode, messages = processor_episode
    first = episode._prepare_model_input(messages)
    if changed == "text":
        messages[0]["content"][0]["text"] += " again"
    elif changed == "image":
        png = image2base64(Image.new("RGB", (2, 2), "blue"))[0]
        messages[0]["content"][1]["image_url"]["url"] = f"data:image/png;base64,{png}"
    elif changed == "processor":
        original_processor = episode.processor
        episode.processor = type(original_processor)()
    else:
        workflow_context.set(
            workflow_context.WorkflowContext(processor_cache=ProcessorCallCache())
        )
    second = episode._prepare_model_input(messages)
    assert second[1]["pixel_values"] is not first[1]["pixel_values"]


@pytest.mark.asyncio
async def test_processor_cache_finalization_and_absent_context_use_native_path(
    processor_episode,
):
    """Closing a group drops cached entries without invalidating trajectory tensors."""
    episode, messages = processor_episode
    first = episode._prepare_model_input(messages)
    workflow = object.__new__(PacmanWorkflow)
    await workflow._afinalize_processor_cache_group(workflow_context.get())
    second = episode._prepare_model_input(messages)
    workflow_context.set(workflow_context.WorkflowContext())
    third = episode._prepare_model_input(messages)
    assert episode.processor.calls == 3
    assert first[1]["pixel_values"] is not second[1]["pixel_values"]
    assert second[1]["pixel_values"] is not third[1]["pixel_values"]
    for key in first[1]:
        torch.testing.assert_close(first[1][key], third[1][key], rtol=0, atol=0)


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

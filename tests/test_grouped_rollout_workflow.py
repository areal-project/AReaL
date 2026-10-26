# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import Mock

import pytest
import torch

from tests.experimental.openai.vision_stubs import (
    IMAGE_PLACEHOLDER,
    StubProcessor,
    StubTokenizer,
    make_image,
)

from areal.api import ModelResponse, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters, NormConfig
from areal.infra import workflow_context
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.trainer.ppo.actor import _trajectory_row_counts
from areal.trainer.ppo.loss_reduction import prepare_policy_gradient_batch
from areal.utils.data import Normalization
from areal.workflow.vision_rlvr import VisionRLVRWorkflow


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["child_error", "child_cancel", "parent_cancel"])
async def test_group_failure_cancels_siblings_before_finalizing(failure):
    """Failed or cancelled groups must not wait forever on unfinished siblings."""
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    never_set = asyncio.Event()
    finalized = []
    original_error = ValueError("candidate failed")

    class FailingWorkflow(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            if workflow_context.get().sample_idx == 0:
                await sibling_started.wait()
                if failure == "child_error":
                    raise original_error
                if failure == "child_cancel":
                    raise asyncio.CancelledError()
                await never_set.wait()
            else:
                sibling_started.set()
                try:
                    await never_set.wait()
                except asyncio.CancelledError:
                    # Include async teardown to verify it is drained as well.
                    await asyncio.sleep(0)
                    sibling_cancelled.set()
                    raise

        async def _afinalize_processor_cache_group(self, context):
            finalized.append(sibling_cancelled.is_set())

    workflow = GroupedRolloutWorkflow(FailingWorkflow(), group_size=2, logger=Mock())
    task = asyncio.create_task(workflow.arun_episode(engine=None, data={}))
    try:
        await asyncio.wait_for(sibling_started.wait(), timeout=1)
        if failure == "parent_cancel":
            task.cancel()
        error_type = ValueError if failure == "child_error" else asyncio.CancelledError
        with pytest.raises(error_type) as exc_info:
            # Shield keeps the timeout from making the broken implementation
            # pass by cancelling its otherwise indefinitely waiting siblings.
            await asyncio.wait_for(asyncio.shield(task), timeout=1)
        if failure == "child_error":
            assert exc_info.value is original_error
        assert sibling_cancelled.is_set()
        assert finalized == [True]
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_grouped_vision_rollouts_share_images_only_within_group(monkeypatch):
    """Identical candidates share processing; the next group computes afresh."""
    calls = []

    class Processor(StubProcessor):
        def __call__(self, *args, **kwargs):
            calls.append(1)
            kwargs["text"] = [kwargs["text"]]
            kwargs["images"] = [kwargs["images"]]
            return super().__call__(*args, **kwargs)

    tokenizer = StubTokenizer()
    workflow = VisionRLVRWorkflow(
        reward_fn=lambda **kwargs: 1,
        gconfig=GenerationHyperparameters(),
        tokenizer=tokenizer,
        processor=Processor(),
        enable_thinking=False,
    )

    async def collect(engine, request, prompt, data):
        assert request.collapsed_input_ids is not None
        return ModelResponse(
            input_tokens=list(request.input_ids),
            output_tokens=[11],
            output_logprobs=[-0.1],
            output_versions=[0],
            stop_reason="length",
            tokenizer=tokenizer,
        ), 1.0

    monkeypatch.setattr(workflow, "_collect_samples", collect)
    grouped = GroupedRolloutWorkflow(workflow, 4, Mock())
    data = {"images": make_image(7), "messages": f"{IMAGE_PLACEHOLDER} describe"}
    first = await grouped.arun_episode(None, data)
    second = await grouped.arun_episode(None, data)
    assert calls == [1, 1]
    for result in (first, second):
        images = [item["pixel_values"] for item in result["multi_modal_input"]]
        assert len(images) == 4
        assert all(image is images[0] for image in images)
    assert (
        first["multi_modal_input"][0]["pixel_values"]
        is not second["multi_modal_input"][0]["pixel_values"]
    )
    torch.testing.assert_close(first["input_ids"], second["input_ids"], rtol=0, atol=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_first_marker", [False, True])
@pytest.mark.parametrize("empty_sample", [None, "none", "empty_dict", "empty_rows"])
async def test_grouped_tensor_episodes_preserve_multirow_trajectory_means(
    explicit_first_marker, empty_sample
):
    """Two rows from one episode count once for both loss and reward baseline."""
    episodes = []
    for n_rows, reward in [(2, 1.0), (1, 10.0)]:
        shape = (n_rows, 2)
        episodes.append(
            {
                "input_ids": torch.ones(shape, dtype=torch.long),
                "attention_mask": torch.ones(shape, dtype=torch.bool),
                "loss_mask": torch.ones(shape, dtype=torch.bool),
                "rewards": torch.full((n_rows,), reward),
            }
        )
    if explicit_first_marker:
        episodes[0]["begin_of_trajectory"] = torch.tensor([1, 0], dtype=torch.int32)
    candidates = list(episodes)
    if empty_sample is not None:
        if empty_sample == "none":
            empty = None
        elif empty_sample == "empty_dict":
            empty = {}
        else:
            empty = {key: value[:0] for key, value in episodes[0].items()}
        candidates.insert(0, empty)

    class TensorWorkflow(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            return candidates[workflow_context.get().sample_idx]

    workflow = GroupedRolloutWorkflow(TensorWorkflow(), len(candidates), Mock())

    result = await workflow.arun_episode(None, {})

    torch.testing.assert_close(
        result["begin_of_trajectory"],
        torch.tensor([1, 0, 1], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    assert ("begin_of_trajectory" in episodes[0]) == explicit_first_marker
    assert "begin_of_trajectory" not in episodes[1]
    sizes = _trajectory_row_counts(result["begin_of_trajectory"], 3)
    assert sizes == [2, 1]
    for mode, group_sizes, expected in [
        ("traj-mean", sizes, 5.5),
        ("seq-mean", None, 4.0),
        ("prompt-mean", [3], 4.0),
    ]:
        prepared = prepare_policy_gradient_batch(
            result, mode=mode, group_sizes=group_sizes
        )
        reduction = prepared.for_steps([result])[0]
        loss = reduction.aggregate(
            result["rewards"].unsqueeze(-1).expand_as(result["loss_mask"]),
            result["loss_mask"],
            unit_weights=result.get("unit_weights"),
        )
        torch.testing.assert_close(loss, torch.tensor(expected), rtol=0, atol=0)

    centered = Normalization(
        NormConfig(mean_level="group", std_level=None, group_size=2)
    )(
        result["rewards"],
        traj_starts=result["begin_of_trajectory"],
        group_sizes=[3],
    )
    torch.testing.assert_close(
        centered, torch.tensor([-4.5, -4.5, 4.5]), rtol=0, atol=0
    )


@pytest.mark.asyncio
async def test_grouped_tensor_episodes_keep_explicit_internal_boundaries():
    """Existing logical trajectory boundaries take precedence over defaults."""
    episodes = [
        {
            "input_ids": torch.ones(3, 2, dtype=torch.long),
            "attention_mask": torch.ones(3, 2, dtype=torch.bool),
            "begin_of_trajectory": torch.tensor([1, 0, 1], dtype=torch.int32),
        },
        {
            "input_ids": torch.ones(1, 2, dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.bool),
        },
    ]

    class TensorWorkflow(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            return episodes[workflow_context.get().sample_idx]

    result = await GroupedRolloutWorkflow(TensorWorkflow(), 2, Mock()).arun_episode(
        None, {}
    )

    torch.testing.assert_close(
        result["begin_of_trajectory"],
        torch.tensor([1, 0, 1, 1], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.asyncio
async def test_grouped_tensor_episodes_all_empty_return_none():
    """Empty tensor episodes cannot supply a trajectory start row."""

    class EmptyWorkflow(RolloutWorkflow):
        async def arun_episode(self, engine, data):
            return {"input_ids": torch.empty(0, 2, dtype=torch.long)}

    result = await GroupedRolloutWorkflow(EmptyWorkflow(), 2, Mock()).arun_episode(
        None, {}
    )

    assert result is None

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from areal.experimental.openai.client import _PreparedPrompt, _share_multimodal_tensors
from areal.infra.processor_cache import ProcessorCallCache


def _prompt(value=1.0, dtype=torch.float32):
    return _PreparedPrompt(
        input_ids=[1, 2],
        mm_token_type_ids=[0, 1],
        multi_modal_input={
            "pixel_values": torch.full((4, 8), value, dtype=dtype),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
    )


def test_identical_images_across_turns_share_storage_preserve_text():
    cache = ProcessorCallCache()
    processor = object()
    first = _share_multimodal_tensors(_prompt(), cache, processor, ["image"])
    second = _prompt()
    second.input_ids = [1, 2, 3, 4]
    second.mm_token_type_ids = [0, 1, 0, 0]
    second = _share_multimodal_tensors(second, cache, processor, ["image"])
    assert second.input_ids == [1, 2, 3, 4]
    assert second.mm_token_type_ids == [0, 1, 0, 0]
    assert second.multi_modal_input is not first.multi_modal_input
    for name, tensor in second.multi_modal_input.items():
        assert tensor is first.multi_modal_input[name]
    cache.close()
    third = _share_multimodal_tensors(_prompt(), cache, processor, ["image"])
    assert (
        third.multi_modal_input["pixel_values"]
        is not first.multi_modal_input["pixel_values"]
    )


@pytest.mark.parametrize(
    "change", ["value", "dtype", "shape", "keys", "signed_zero", "image", "processor"]
)
def test_different_vision_inputs_keep_original_storage(change):
    cache = ProcessorCallCache()
    processor = object()
    first = _prompt(0.0)
    _share_multimodal_tensors(first, cache, processor, ["image"])
    second = _prompt(0.0)
    images = ["image"]
    if change == "value":
        second.multi_modal_input["pixel_values"].fill_(2)
    elif change == "dtype":
        second.multi_modal_input["pixel_values"] = second.multi_modal_input[
            "pixel_values"
        ].double()
    elif change == "shape":
        second.multi_modal_input["pixel_values"] = second.multi_modal_input[
            "pixel_values"
        ].reshape(8, 4)
    elif change == "keys":
        second.multi_modal_input.pop("image_grid_thw")
    elif change == "signed_zero":
        second.multi_modal_input["pixel_values"].fill_(-0.0)
    elif change == "image":
        images = ["different_image"]
    elif change == "processor":
        processor = object()
    original = second.multi_modal_input["pixel_values"]
    result = _share_multimodal_tensors(second, cache, processor, images)
    assert result.multi_modal_input["pixel_values"] is original
    assert original is not first.multi_modal_input["pixel_values"]

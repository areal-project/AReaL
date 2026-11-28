# SPDX-License-Identifier: Apache-2.0
"""Packed sdpa must not split image patches by the text sample boundaries.

Since transformers 5.9, Qwen-VL forwards ``**kwargs`` into the vision tower. The
packed sdpa wrapper reads ``packed_seq_bounds`` from those kwargs, so without the
vision-tower filter it would split image patches at *text* boundaries.
"""

import pytest
import torch
from torch.testing import assert_close
from transformers import (
    AttentionInterface,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from areal.engine.fsdp_engine import FSDPEngine
from areal.engine.fsdp_utils.attn_impl import (
    PACKED_SEQ_BOUNDS_KWARG,
    register_packed_sdpa,
)

IMAGE_TOKEN, VISION_START, VISION_END, VIDEO_TOKEN = 250, 248, 249, 251
PATCH, MERGE, TEMPORAL = 2, 2, 2
GRID = (1, 8, 8)  # t, h, w in patches -> 64 patches, 16 merged image tokens
NO_MASK = dict(full_attention=None, sliding_attention=None)


@pytest.fixture(autouse=True)
def preserve_sdpa_registration():
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    try:
        yield
    finally:
        AttentionInterface.register("sdpa", original)


def _tiny_qwen2_5_vl():
    torch.manual_seed(0)
    config = Qwen2_5_VLConfig(
        text_config=dict(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            rope_parameters=dict(
                rope_type="default", rope_theta=10000.0, mrope_section=[2, 3, 3]
            ),
        ),
        vision_config=dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            out_hidden_size=64,
            patch_size=PATCH,
            spatial_merge_size=MERGE,
            temporal_patch_size=TEMPORAL,
            window_size=PATCH * MERGE * 2,
            fullatt_block_indexes=[1],
            in_channels=3,
        ),
        image_token_id=IMAGE_TOKEN,
        video_token_id=VIDEO_TOKEN,
        vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
    )
    model = Qwen2_5_VLForConditionalGeneration._from_config(
        config, attn_implementation="sdpa"
    )
    return model.eval()


def _packed_row():
    """Sample 0: text + one image. Sample 1: text only."""
    g = torch.Generator().manual_seed(1)
    n_image_tokens = GRID[0] * GRID[1] * GRID[2] // (MERGE * MERGE)
    sample0 = torch.cat(
        [
            torch.randint(1, 240, (5,), generator=g),
            torch.tensor([VISION_START]),
            torch.full((n_image_tokens,), IMAGE_TOKEN),
            torch.tensor([VISION_END]),
            torch.randint(1, 240, (4,), generator=g),
        ]
    )
    sample1 = torch.randint(1, 240, (9,), generator=g)
    input_ids = torch.cat([sample0, sample1])[None]
    bounds = (0, sample0.numel(), sample0.numel() + sample1.numel())
    n_patches = GRID[0] * GRID[1] * GRID[2]
    pixel_values = torch.randn(n_patches, 3 * TEMPORAL * PATCH * PATCH, generator=g)
    image_grid_thw = torch.tensor([GRID])
    return input_ids, bounds, pixel_values, image_grid_thw


def _merged_image_embeds(output):
    """Vision tower output -> merged image embeddings (ModelOutput or tensor)."""
    pooled = getattr(output, "pooler_output", None)
    if isinstance(pooled, (tuple, list)):
        # get_image_features later replaces it in place with a per-image split
        return torch.cat(pooled)
    if pooled is not None:
        return pooled
    if torch.is_tensor(output):
        return output
    return output[1]


def _image_embeds_in_packed_forward(model, isolate_vision_tower):
    if isolate_vision_tower:
        FSDPEngine._isolate_vision_tower_from_text_packing_kwargs(model)
    input_ids, bounds, pixel_values, image_grid_thw = _packed_row()
    captured = []
    handle = model.model.visual.register_forward_hook(
        lambda module, args, output: captured.append(output)
    )
    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                attention_mask=NO_MASK,
                use_cache=False,
                **{PACKED_SEQ_BOUNDS_KWARG: bounds},
            )
    finally:
        handle.remove()
    return _merged_image_embeds(captured[0])


def _reference_image_embeds(model):
    _, _, pixel_values, image_grid_thw = _packed_row()
    with torch.no_grad():
        out = model.model.visual(pixel_values, grid_thw=image_grid_thw)
    return _merged_image_embeds(out)


def test_packed_bounds_do_not_split_image_patches():
    register_packed_sdpa()
    model = _tiny_qwen2_5_vl()
    reference = _reference_image_embeds(model)
    packed = _image_embeds_in_packed_forward(model, isolate_vision_tower=True)
    assert_close(packed, reference, rtol=1e-5, atol=1e-5)


def test_without_vision_filter_text_bounds_corrupt_image_embeds():
    """Guards the test above: it must be able to see the bug."""
    register_packed_sdpa()
    model = _tiny_qwen2_5_vl()
    reference = _reference_image_embeds(model)
    packed = _image_embeds_in_packed_forward(model, isolate_vision_tower=False)
    if torch.allclose(packed, reference, rtol=1e-5, atol=1e-5):
        pytest.skip(
            "this transformers version does not forward kwargs to the vision tower"
        )
    assert not torch.allclose(packed, reference, rtol=1e-5, atol=1e-5)

# SPDX-License-Identifier: Apache-2.0

"""Batching tests for training batches that mix image and text-only rows.

A dataset can carry screenshots on some tasks and none on others, and a
multi-turn agent can attach an image partway through an episode. Both produce
trajectories whose key sets disagree, which the batching helpers must absorb
rather than reject.
"""

import pytest
import torch

from areal.engine.fsdp_engine import _prepare_multimodal_forward_inputs
from areal.utils.data import (
    concat_batch,
    concat_padded_tensors,
    normalize_vision_keys,
)


def _vision_row(seq_len: int = 5, n_patches: int = 4, batch_size: int = 1) -> dict:
    return {
        "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        "mm_token_type_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "multi_modal_input": [
            {
                "pixel_values": torch.ones(n_patches, 8),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
            for _ in range(batch_size)
        ],
    }


def _text_row(seq_len: int = 5, batch_size: int = 1) -> dict:
    return {
        "input_ids": torch.ones(batch_size, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# normalize_vision_keys
# ---------------------------------------------------------------------------


def test_normalize_returns_input_unchanged_when_homogeneous():
    """Test that an all-vision or all-text batch is not copied."""
    all_vision = [_vision_row(), _vision_row()]
    all_text = [_text_row(), _text_row()]

    assert normalize_vision_keys(all_vision) is all_vision
    assert normalize_vision_keys(all_text) is all_text


def test_normalize_fills_only_the_rows_that_need_it():
    """Test that vision rows are passed through by identity, not rebuilt."""
    vision, text = _vision_row(), _text_row()

    normalized = normalize_vision_keys([vision, text])

    assert normalized[0] is vision
    assert normalized[1] is not text
    assert normalized[1]["multi_modal_input"] == [{}]
    assert torch.equal(
        normalized[1]["mm_token_type_ids"], torch.zeros(1, 5, dtype=torch.long)
    )
    # The caller's dict must not be mutated.
    assert "multi_modal_input" not in text


def test_normalize_matches_placeholder_count_to_batch_size():
    """Test that a grouped trajectory gets one placeholder per sequence."""
    grouped_text = _text_row(batch_size=4)

    normalized = normalize_vision_keys([_vision_row(), grouped_text])

    assert normalized[1]["multi_modal_input"] == [{}, {}, {}, {}]
    assert normalized[1]["mm_token_type_ids"].shape == (4, 5)


def test_normalize_leaves_non_vision_key_mismatches_alone():
    """Test that the strict key check still catches genuine batching bugs."""
    good = _text_row()
    bad = {**_text_row(), "unexpected_key": torch.ones(1, 5)}

    with pytest.raises(ValueError, match="different keys"):
        concat_padded_tensors([good, bad])


# ---------------------------------------------------------------------------
# concat_padded_tensors / concat_batch
# ---------------------------------------------------------------------------


def test_concat_padded_tensors_batches_mixed_rows():
    """Test that a batch mixing image and text-only trajectories concatenates."""
    batch = concat_padded_tensors([_vision_row(), _text_row(), _vision_row()])

    assert batch["input_ids"].shape[0] == 3
    assert len(batch["multi_modal_input"]) == 3
    assert "pixel_values" in batch["multi_modal_input"][0]
    assert batch["multi_modal_input"][1] == {}
    assert "pixel_values" in batch["multi_modal_input"][2]
    assert batch["mm_token_type_ids"].shape == batch["input_ids"].shape


def test_concat_padded_tensors_pads_mixed_rows_of_different_lengths():
    """Test that padding still applies across mixed rows."""
    batch = concat_padded_tensors([_vision_row(seq_len=7), _text_row(seq_len=3)])

    assert batch["input_ids"].shape == (2, 7)
    assert batch["mm_token_type_ids"].shape == (2, 7)
    # The synthesized row is entirely non-multimodal.
    assert batch["mm_token_type_ids"][1].sum() == 0


def test_concat_batch_handles_mixed_rows():
    """Test the entry point the engines use to build a training batch."""
    batch, meta = concat_batch([_vision_row(), _text_row(), _text_row()])

    assert meta.n_trajs == 3
    assert batch["input_ids"].shape[0] == 3
    assert len(batch["multi_modal_input"]) == 3


def test_text_only_batch_gains_no_vision_keys():
    """Test that batches without any images stay exactly as before."""
    batch = concat_padded_tensors([_text_row(), _text_row()])

    assert "multi_modal_input" not in batch
    assert "mm_token_type_ids" not in batch


# ---------------------------------------------------------------------------
# Engine consumption
# ---------------------------------------------------------------------------


def test_fsdp_gathers_pixels_from_image_rows_only():
    """Test that the FSDP forward prep skips placeholder rows."""
    batch = concat_padded_tensors([_vision_row(), _text_row(), _vision_row()])
    mb = dict(batch)
    padded_mb = dict(batch)

    _prepare_multimodal_forward_inputs(mb, padded_mb)

    # Two image rows of 4 patches each; the text row contributes nothing.
    assert padded_mb["pixel_values"].shape == (8, 8)
    assert padded_mb["image_grid_thw"].shape == (2, 3)
    assert "multi_modal_input" not in padded_mb


def test_fsdp_prep_tolerates_an_all_placeholder_batch():
    """Test that a batch of placeholders yields no vision tensors at all."""
    rows = normalize_vision_keys([_vision_row(), _text_row()])
    all_placeholder = {**rows[1], "multi_modal_input": [{}]}
    mb = dict(all_placeholder)
    padded_mb = dict(all_placeholder)

    _prepare_multimodal_forward_inputs(mb, padded_mb)

    assert "pixel_values" not in padded_mb
    assert "image_grid_thw" not in padded_mb


# The Megatron counterpart lives in test_mixed_multimodal_batch_megatron.py:
# on NPU, MindSpeed must patch transformer_engine before anything under
# areal.engine is imported, which cannot be arranged from inside this module.

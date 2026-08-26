# SPDX-License-Identifier: Apache-2.0

"""Megatron-side batching test for mixed image / text-only rows.

Kept separate from ``test_mixed_multimodal_batch.py`` because of import order:
on NPU, MindSpeed patches ``transformer_engine`` when it is imported, and that
must happen before anything under ``areal.engine`` pulls the unpatched module
in. Run this file in its own pytest process::

    python -m pytest tests/test_mixed_multimodal_batch_megatron.py
"""

try:
    # Must precede every areal.engine import; absent on non-NPU platforms.
    import mindspeed.megatron_adaptor  # noqa: F401  # isort: skip
except ImportError:
    pass

import pytest
import torch

from areal.utils.data import concat_padded_tensors


def _vision_row(seq_len: int = 5, n_patches: int = 4) -> dict:
    return {
        "input_ids": torch.ones(1, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
        "mm_token_type_ids": torch.zeros(1, seq_len, dtype=torch.long),
        "multi_modal_input": [
            {
                "pixel_values": torch.ones(n_patches, 8),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }
        ],
    }


def _text_row(seq_len: int = 5) -> dict:
    return {
        "input_ids": torch.ones(1, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
    }


def test_megatron_gathers_pixels_from_image_rows_only():
    """Test that the Megatron vision extraction skips placeholder rows."""
    megatron_utils = pytest.importorskip(
        "areal.engine.megatron_utils.packed_context_parallel",
        reason="megatron-core / MindSpeed not available",
    )
    batch = concat_padded_tensors([_vision_row(), _text_row(), _vision_row()])
    mb = dict(batch)
    padded_mb = dict(batch)

    megatron_utils.extract_vision_from_multi_modal(mb, padded_mb)

    # Two image rows of 4 patches each; the text row contributes nothing.
    assert padded_mb["pixel_values"].shape == (8, 8)
    assert padded_mb["image_grid_thw"].shape == (2, 3)
    assert "multi_modal_input" not in padded_mb
    assert "multi_modal_input" not in mb


def test_megatron_extraction_tolerates_an_all_placeholder_batch():
    """Test that a batch with no images yields no vision tensors."""
    megatron_utils = pytest.importorskip(
        "areal.engine.megatron_utils.packed_context_parallel",
        reason="megatron-core / MindSpeed not available",
    )
    batch = concat_padded_tensors([_vision_row(), _text_row()])
    batch["multi_modal_input"] = [{}, {}]
    mb = dict(batch)
    padded_mb = dict(batch)

    megatron_utils.extract_vision_from_multi_modal(mb, padded_mb)

    assert "pixel_values" not in padded_mb
    assert "image_grid_thw" not in padded_mb

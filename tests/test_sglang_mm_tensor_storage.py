# SPDX-License-Identifier: Apache-2.0

import pickle
from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from areal.engine.sglang_processors._tensor_storage import compact_mm_item_storage


@dataclass
class ImageItem:
    feature: torch.Tensor | None = None
    precomputed_embeddings: torch.Tensor | None = None
    model_specific_data: dict[str, Any] = field(default_factory=dict)
    hash: int = 12345
    pad_value: int = 54321
    offsets: list[tuple[int, int]] = field(default_factory=lambda: [(2, 17)])


@pytest.mark.parametrize("image_count", [1, 2, 4, 8])
def test_split_images_pickle_payload_stays_linear(image_count):
    batch = torch.arange(image_count * 4096, dtype=torch.float32).reshape(-1, 16)
    items = [ImageItem(feature=part) for part in batch.chunk(image_count)]
    logical_bytes = batch.numel() * batch.element_size()

    compacted = compact_mm_item_storage(items)
    decoded = pickle.loads(pickle.dumps(compacted))

    assert len(pickle.dumps(compacted)) < logical_bytes + 4096 * image_count
    if image_count > 1:
        assert len(pickle.dumps(items)) > logical_bytes * image_count
    for original, compact, transported in zip(items, compacted, decoded, strict=True):
        torch.testing.assert_close(
            transported.feature, original.feature, rtol=0, atol=0
        )
        assert compact.feature.untyped_storage().nbytes() == (
            compact.feature.numel() * compact.feature.element_size()
        )
        assert original.feature.untyped_storage().nbytes() == logical_bytes
        assert compact.hash == original.hash
        assert compact.pad_value == original.pad_value
        assert compact.offsets == original.offsets


def test_nested_tensor_aliases_remain_shared_without_mutating_cached_items():
    batch = torch.arange(256, dtype=torch.float32)
    view = batch[16:32]
    cached = ImageItem(
        feature=view,
        model_specific_data={"nested": [view, (view,)]},
    )

    compacted = compact_mm_item_storage([cached, cached])

    assert compacted[0] is not cached
    assert compacted[0].feature is compacted[1].feature
    assert compacted[0].feature is compacted[0].model_specific_data["nested"][0]
    assert compacted[0].feature is compacted[0].model_specific_data["nested"][1][0]
    assert cached.feature is view
    assert cached.model_specific_data["nested"][0] is view
    assert cached.feature.untyped_storage().nbytes() == batch.numel() * 4


def test_precomputed_embedding_views_are_compacted_and_full_tensors_reused():
    batch = torch.arange(256, dtype=torch.float32)
    full = torch.ones(16)
    items = [
        ImageItem(precomputed_embeddings=batch[16:32]),
        ImageItem(feature=full, model_specific_data={"external_transport": object()}),
    ]

    compacted = compact_mm_item_storage(items)

    torch.testing.assert_close(
        compacted[0].precomputed_embeddings, batch[16:32], rtol=0, atol=0
    )
    assert compacted[0].precomputed_embeddings.untyped_storage().nbytes() == 16 * 4
    assert compacted[1].feature is full
    assert (
        compacted[1].model_specific_data["external_transport"]
        is items[1].model_specific_data["external_transport"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_views_keep_native_storage_and_transport():
    view = torch.ones(256, device="cuda")[16:32]

    compacted = compact_mm_item_storage([ImageItem(feature=view)])

    assert compacted[0].feature is view

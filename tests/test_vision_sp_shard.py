"""
Unit tests for Vision SP Shard utilities (CPU-only, no distributed).

Test naming convention: test_<what>_<condition>_<expected>()
"""

from unittest.mock import patch

import pytest
import torch

from areal.models.transformers.vision_sp_shard import (
    _assign_images_to_dp_ranks,
    _gather_vision_embeddings,
    _get_image_embedding_counts,
    _get_image_patch_counts,
    _patch_vision_class,
    _prepare_local_vision_inputs,
    _unpack_deepstack,
    apply_vision_sp_shard_patch,
    create_dp_vision_forward,
)


def _assert_all_images_assigned(assignments, num_images):
    """Assert every image index in [0, num_images) appears exactly once."""
    all_idx = [i for rank in assignments for i in rank]
    assert sorted(all_idx) == list(range(num_images))


class TestGetImagePatchCounts:
    @pytest.mark.parametrize(
        "grid_thw,expected",
        [
            ([[2, 4, 4], [1, 2, 2], [1, 8, 8]], [32, 4, 64]),
            ([[1, 4, 4]], [16]),
            ([[4, 4, 4]], [64]),
        ],
        ids=["multi-image", "single-image", "video-frames"],
    )
    def test_patch_counts_various_grids_correct_products(self, grid_thw, expected):
        counts = _get_image_patch_counts(torch.tensor(grid_thw))
        assert counts == expected

    def test_patch_counts_empty_input_returns_empty_list(self):
        counts = _get_image_patch_counts(torch.empty((0, 3), dtype=torch.long))
        assert counts == []


class TestGetImageEmbeddingCounts:
    @pytest.mark.parametrize(
        "grid_thw,merge_size,expected",
        [
            ([[1, 8, 8]], 1, [64]),
            ([[1, 8, 8]], 2, [16]),
            ([[1, 6, 6], [1, 4, 4]], 2, [9, 4]),
        ],
        ids=["no-merge", "merge-2", "multi-image-merge"],
    )
    def test_embedding_counts_with_merge_size_correct(
        self, grid_thw, merge_size, expected
    ):
        counts = _get_image_embedding_counts(torch.tensor(grid_thw), merge_size)
        assert counts == expected

    def test_embedding_counts_empty_input_returns_empty_list(self):
        counts = _get_image_embedding_counts(torch.empty((0, 3), dtype=torch.long))
        assert counts == []

    def test_embedding_counts_merge_1_equals_patch_counts(self):
        """merge_size=1 path should produce identical results to _get_image_patch_counts."""
        grid = torch.tensor([[2, 4, 4], [1, 8, 8], [1, 6, 6]])
        assert _get_image_embedding_counts(grid, 1) == _get_image_patch_counts(grid)


class TestAssignImagesToDpRanks:
    @pytest.mark.parametrize(
        "patch_counts,dp_size,expected_all_assigned",
        [
            ([100, 100, 100, 100], 2, True),
            ([100, 200, 300], 1, True),
            ([100, 100, 100, 100, 100, 100], 3, True),
        ],
        ids=["balanced-2ranks", "single-rank", "balanced-3ranks"],
    )
    def test_assign_all_images_distributed_correctly(
        self, patch_counts, dp_size, expected_all_assigned
    ):
        assignments, loads = _assign_images_to_dp_ranks(patch_counts, dp_size)
        _assert_all_images_assigned(assignments, len(patch_counts))
        assert sum(loads) == sum(patch_counts)

    def test_assign_fewer_images_than_ranks_all_assigned(self):
        assignments, loads = _assign_images_to_dp_ranks([100, 200], dp_size=4)
        non_empty = sum(1 for a in assignments if len(a) > 0)
        assert non_empty == 2
        _assert_all_images_assigned(assignments, 2)

    def test_assign_empty_input_returns_empty_lists(self):
        assignments, loads = _assign_images_to_dp_ranks([], dp_size=4)
        assert all(len(a) == 0 for a in assignments)
        assert all(load == 0 for load in loads)

    def test_assign_image_order_preserved_contiguous(self):
        assignments, _ = _assign_images_to_dp_ranks([10, 20, 30, 40, 50], dp_size=2)
        for rank_assignment in assignments:
            assert rank_assignment == sorted(rank_assignment)

    def test_assign_load_balanced_unequal_patches_reduces_imbalance(self):
        """With unequal patch counts, greedy balancing should reduce imbalance."""
        # 4096 + 256 + 256 + 256 = 4864, target per rank = 2432
        patch_counts = [4096, 256, 256, 256]
        assignments, loads = _assign_images_to_dp_ranks(patch_counts, dp_size=2)
        _assert_all_images_assigned(assignments, len(patch_counts))
        # Load imbalance should be less than the naive count-based split (8.5x)
        max_load = max(loads)
        min_load = min(load for load in loads if load > 0)
        assert max_load / min_load < 8.0

    def test_assign_contiguous_coverage_all_dp_sizes(self):
        """All images are covered exactly once across ranks for various dp_size."""
        patch_counts = [10, 20, 30, 40, 50, 60, 70]
        for dp_size in [1, 2, 3, 4, 7]:
            assignments, _ = _assign_images_to_dp_ranks(patch_counts, dp_size)
            _assert_all_images_assigned(assignments, len(patch_counts))


class TestPrepareLocalVisionInputs:
    def test_prepare_two_images_splits_correctly(self):
        pixel_values = torch.randn(100, 768)
        grid_thw = torch.tensor([[1, 6, 6], [1, 8, 8]])  # 36 + 64 = 100
        image_assignments = [[0], [1]]

        # Rank 0
        pix, grid, indices = _prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=0
        )
        assert pix.shape[0] == 36
        assert grid.shape[0] == 1
        assert indices == [0]
        assert torch.allclose(pix, pixel_values[:36])

        # Rank 1
        pix, grid, indices = _prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=1
        )
        assert pix.shape[0] == 64
        assert indices == [1]
        assert torch.allclose(pix, pixel_values[36:100])

    def test_prepare_multiple_contiguous_images_per_rank(self):
        pixel_values = torch.randn(200, 768)
        grid_thw = torch.tensor([[1, 5, 10]] * 4)  # 4 x 50 patches
        image_assignments = [[0, 1], [2, 3]]

        pix, grid, indices = _prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=0
        )
        assert pix.shape[0] == 100
        assert grid.shape[0] == 2
        assert indices == [0, 1]
        assert torch.allclose(pix, pixel_values[:100])

    def test_prepare_empty_rank_returns_empty_tensors(self):
        pixel_values = torch.randn(100, 768)
        grid_thw = torch.tensor([[1, 10, 10]])
        image_assignments = [[0], []]

        pix, grid, indices = _prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=1
        )
        assert pix.shape[0] == 0
        assert grid.shape[0] == 0
        assert indices == []

    def test_prepare_local_inputs_grid_thw_values_preserved(self):
        pixel_values = torch.randn(150, 768)
        grid_thw = torch.tensor([[1, 5, 5], [2, 5, 5], [3, 5, 5]])  # 25 + 50 + 75
        image_assignments = [[0, 1], [2]]

        _, local_grid, _ = _prepare_local_vision_inputs(
            pixel_values, grid_thw, image_assignments, dp_rank=0
        )
        assert local_grid.shape == (2, 3)
        assert torch.equal(local_grid[0], grid_thw[0])
        assert torch.equal(local_grid[1], grid_thw[1])


class TestGatherVisionEmbeddings:
    def test_gather_embeddings_none_group_returns_input_unchanged(self):
        """dp_group=None should short-circuit and return the same tensor (zero-copy)."""
        embeddings = torch.randn(10, 64)
        result = _gather_vision_embeddings(embeddings, dp_group=None, all_counts=[10])
        assert torch.equal(result, embeddings)
        # Verify zero-copy — same storage, not a clone
        assert result.data_ptr() == embeddings.data_ptr()


class TestCreateDpVisionForward:
    def test_dp_vision_forward_sp_size_1_calls_original_directly(self):
        """When sp_size <= 1, the wrapper should call original_forward directly."""
        call_log = []

        def original_forward(self, hidden_states, grid_thw, **kwargs):
            call_log.append("original")
            return hidden_states

        wrapped = create_dp_vision_forward(original_forward)

        hidden_states = torch.randn(100, 768)
        grid_thw = torch.tensor([[1, 10, 10]])

        # Mock Ulysses SP to return sp_size=1
        with (
            patch(
                "areal.models.transformers.vision_sp_shard.get_ulysses_sequence_parallel_group",
                return_value=None,
            ),
            patch(
                "areal.models.transformers.vision_sp_shard.get_ulysses_sequence_parallel_world_size",
                return_value=1,
            ),
            patch(
                "areal.models.transformers.vision_sp_shard.get_ulysses_sequence_parallel_rank",
                return_value=0,
            ),
        ):
            result = wrapped(None, hidden_states, grid_thw)

        assert call_log == ["original"]
        assert torch.equal(result, hidden_states)


class TestPatchVisionClass:
    def test_patch_vision_class_replaces_forward(self):
        """_patch_vision_class should replace the class forward method."""

        class FakeVisionModel:
            @staticmethod
            def forward(self, hidden_states, grid_thw, **kwargs):
                return hidden_states

        original = FakeVisionModel.forward
        _patch_vision_class(FakeVisionModel, "FakeVisionModel")
        assert FakeVisionModel.forward is not original
        assert getattr(FakeVisionModel, "_vision_sp_shard_patched", False) is True

    def test_patch_vision_class_idempotent_only_wraps_once(self):
        """Calling _patch_vision_class twice should only wrap forward once."""

        class FakeVisionModel:
            @staticmethod
            def forward(self, hidden_states, grid_thw, **kwargs):
                return hidden_states

        _patch_vision_class(FakeVisionModel, "FakeVisionModel")
        first_patched = FakeVisionModel.forward

        # Patch again — should be a no-op
        _patch_vision_class(FakeVisionModel, "FakeVisionModel")
        assert (
            FakeVisionModel.forward is first_patched
        )  # same wrapper, not double-wrapped


class TestApplyVisionSpShardPatch:
    def test_apply_patch_import_error_does_not_raise(self):
        """ImportError for unavailable models should not crash."""
        apply_vision_sp_shard_patch()  # should not raise


class TestIntegration:
    def test_full_workflow_all_patches_covered(self):
        grid_thw = torch.tensor([[1, 4, 4], [1, 8, 8], [1, 4, 4], [1, 6, 6], [1, 4, 4]])
        total_patches = 16 + 64 + 16 + 36 + 16  # 148
        pixel_values = torch.randn(total_patches, 768)

        patch_counts = _get_image_patch_counts(grid_thw)
        assert patch_counts == [16, 64, 16, 36, 16]

        assignments, loads = _assign_images_to_dp_ranks(patch_counts, dp_size=2)
        _assert_all_images_assigned(assignments, len(patch_counts))

        total_local_patches = 0
        for rank in range(2):
            pix, grid, indices = _prepare_local_vision_inputs(
                pixel_values, grid_thw, assignments, dp_rank=rank
            )
            expected = sum(patch_counts[i] for i in indices)
            assert pix.shape[0] == expected
            assert grid.shape[0] == len(indices)
            total_local_patches += pix.shape[0]

        assert total_local_patches == total_patches

    def test_same_size_images_4_ranks_balanced(self):
        num_images = 50
        grid_thw = torch.tensor([[1, 8, 8]] * num_images)
        patch_counts = _get_image_patch_counts(grid_thw)
        assignments, loads = _assign_images_to_dp_ranks(patch_counts, dp_size=4)

        for rank in range(4):
            assert 12 <= len(assignments[rank]) <= 13
        for load in loads:
            assert load in [768, 832]


class TestUnpackDeepstack:
    def test_unpack_no_deepstack_attr_returns_none(self):
        """Model without deepstack_merger_list should return None."""

        class FakeModel:
            pass

        emb = torch.randn(10, 64)
        hidden = torch.randn(100, 64)
        assert _unpack_deepstack(FakeModel(), emb, hidden) is None

    def test_unpack_tuple_input_returns_unpacked(self):
        """Qwen3-VL normal rank: forward returns (embeddings, list[Tensor])."""

        class FakeModel:
            deepstack_merger_list = [None, None, None]  # 3 deepstack layers

        emb = torch.randn(10, 64)
        ds = [torch.randn(10, 64) for _ in range(3)]
        hidden = torch.randn(100, 64)

        result = _unpack_deepstack(FakeModel(), (emb, ds), hidden)
        assert result is not None
        unpacked_emb, unpacked_ds = result
        assert torch.equal(unpacked_emb, emb)
        assert len(unpacked_ds) == 3
        for orig, unpacked in zip(ds, unpacked_ds):
            assert torch.equal(orig, unpacked)

    def test_unpack_empty_rank_creates_grad_tensors(self):
        """Empty rank must create empty deepstack tensors with requires_grad for NCCL."""

        class FakeModel:
            deepstack_merger_list = [None, None]  # 2 deepstack layers

        emb = torch.empty(0, 64)  # empty rank, no images
        hidden = torch.randn(100, 64)

        result = _unpack_deepstack(FakeModel(), emb, hidden)
        assert result is not None
        unpacked_emb, unpacked_ds = result

        # Embeddings returned as-is
        assert torch.equal(unpacked_emb, emb)

        # Deepstack: correct count, empty, correct dtype/device, grad-enabled
        assert len(unpacked_ds) == 2
        for ds_tensor in unpacked_ds:
            assert ds_tensor.shape == (0, 64)
            assert ds_tensor.dtype == hidden.dtype
            assert ds_tensor.device == hidden.device
            assert ds_tensor.requires_grad

# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for vision-tower deduplication (CPU-only, no distributed).

Two questions every test here has to survive:
  1. would this still pass if the wrapper were dropped?  -> TestLoadBearing
  2. would this still pass if the wrapper returned garbage? -> the parity tests
A test that only checks shape/dtype answers neither.

Test naming convention: test_<what>_<condition>_<expected>()
"""

import importlib

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from areal.models.transformers.vision_dedup import (
    _group_duplicates,
    _patch_vision_class,
    apply_vision_dedup,
    dedup_vision_forward,
)
from areal.models.transformers.vision_sp_shard import create_dp_vision_forward


class _VisionTower(torch.nn.Module):
    """Stand-in with the real signature: forward(hidden_states, grid_thw).

    Merges 2x2 patches like Qwen2.5-VL, so output rows = input rows / 4, which
    is what makes the expand-back arithmetic non-trivial.
    """

    MERGE = 4

    def __init__(self, dim=8, out=6):
        super().__init__()
        self.proj = torch.nn.Linear(dim * self.MERGE, out)
        self.calls = 0
        self.rows_seen = 0

    def forward(self, hidden_states, grid_thw=None):
        self.calls += 1
        self.rows_seen += hidden_states.shape[0]
        merged = hidden_states.reshape(hidden_states.shape[0] // self.MERGE, -1)
        return self.proj(merged)


class _RealShapeTower(torch.nn.Module):
    """Returns the real dataclass with fields at two different granularities.

    A test double that is more convenient than the real thing tests the double:
    the first version of this suite returned a bare tensor, so nothing exercised
    the per-field expansion and a wrong shared divisor stayed green.
    """

    MERGE = 4

    def __init__(self, dim=8, out=6):
        super().__init__()
        self.proj = torch.nn.Linear(dim * self.MERGE, out)
        self.rows_seen = 0

    def forward(self, hidden_states, grid_thw=None):
        self.rows_seen += hidden_states.shape[0]
        merged = self.proj(
            hidden_states.reshape(hidden_states.shape[0] // self.MERGE, -1)
        )
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states * 2.0,  # pre-merge granularity
            pooler_output=merged,  # post-merge granularity
        )


class _DeepstackTower(torch.nn.Module):
    """Returns ``(embeddings, [deepstack, ...])`` the way Qwen3-VL's tower does."""

    MERGE = 4

    def __init__(self, dim=8, out=6):
        super().__init__()
        self.proj = torch.nn.Linear(dim * self.MERGE, out)
        self.rows_seen = 0

    def forward(self, hidden_states, grid_thw=None):
        self.rows_seen += hidden_states.shape[0]
        merged = self.proj(
            hidden_states.reshape(hidden_states.shape[0] // self.MERGE, -1)
        )
        return merged, [hidden_states * 3.0]


_ORIGINALS = {
    cls: cls.forward for cls in (_VisionTower, _RealShapeTower, _DeepstackTower)
}


@pytest.fixture(autouse=True)
def _restore_forward():
    """Undo class-level patching between tests, for real.

    An earlier version restored with ``type(wrapped).forward = type(plain).forward``
    -- but both are the SAME class, so that assigned the patched function back
    onto itself and the patch leaked into every later test. Capture the original
    function objects once, at import, and put those back.
    """
    yield
    for cls, fn in _ORIGINALS.items():
        cls.forward = fn
        if hasattr(cls, "_vision_dedup_patched"):
            del cls._vision_dedup_patched
        if hasattr(cls, "_vision_sp_shard_patched"):
            del cls._vision_sp_shard_patched


def _make(mults, rows=8, dim=8, seed=0):
    """Build (hidden_states, grid_thw) where each entry of ``mults`` is a
    multiplicity: [4, 4] = two distinct images, each repeated four times."""
    g = torch.Generator().manual_seed(seed)
    imgs, order = [], []
    for gi, mult in enumerate(mults):
        imgs.append(torch.randn(rows, dim, generator=g))
        order.extend([gi] * mult)
    flat = torch.cat([imgs[i] for i in order], dim=0)
    grid = torch.tensor([[rows // 4, 2, 2]] * len(order))
    assert int(grid[0].prod()) == rows
    return flat, grid, len(imgs), len(order)


class TestGroupDuplicates:
    def test_group_duplicates_repeated_images_finds_representatives(self):
        flat, grid, n_unique, n_total = _make([4, 4])
        dims = [tuple(r) for r in grid.tolist()]
        counts = [t * h * w for t, h, w in dims]
        reps, uniques = _group_duplicates(flat, dims, counts)
        assert len(uniques) == n_unique
        assert len(reps) == n_total
        assert reps == [0, 0, 0, 0, 4, 4, 4, 4]

    def test_group_duplicates_all_distinct_returns_every_index(self):
        flat, grid, n_unique, n_total = _make([1, 1, 1])
        dims = [tuple(r) for r in grid.tolist()]
        counts = [t * h * w for t, h, w in dims]
        reps, uniques = _group_duplicates(flat, dims, counts)
        assert uniques == [0, 1, 2]
        assert reps == [0, 1, 2]


class TestDedupParity:
    @pytest.mark.parametrize("mults", [[8], [4, 4], [8, 1, 3], [1, 1, 1]])
    def test_dedup_forward_any_multiplicity_matches_undeduped(self, mults):
        flat, grid, _, _ = _make(mults)
        plain = _VisionTower()
        expected = plain(flat, grid)

        wrapped = _VisionTower()
        wrapped.load_state_dict(plain.state_dict())
        type(wrapped).forward = dedup_vision_forward(_ORIGINALS[_VisionTower])
        got = wrapped(flat, grid)

        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("mults", [[8], [4, 4], [8, 1, 3]])
    def test_model_output_return_each_field_expanded_at_its_own_granularity(
        self, mults
    ):
        flat, grid, _, _ = _make(mults)
        plain = _RealShapeTower()
        expected = plain(flat, grid)

        wrapped = _RealShapeTower()
        wrapped.load_state_dict(plain.state_dict())
        type(wrapped).forward = dedup_vision_forward(_ORIGINALS[_RealShapeTower])
        got = wrapped(flat, grid)

        assert type(got) is type(expected)
        torch.testing.assert_close(
            got.last_hidden_state, expected.last_hidden_state, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            got.pooler_output, expected.pooler_output, rtol=1e-5, atol=1e-6
        )

    def test_tuple_return_with_deepstack_expanded_elementwise(self):
        flat, grid, _, _ = _make([4, 4])
        plain = _DeepstackTower()
        exp_emb, exp_ds = plain(flat, grid)

        wrapped = _DeepstackTower()
        wrapped.load_state_dict(plain.state_dict())
        type(wrapped).forward = dedup_vision_forward(_ORIGINALS[_DeepstackTower])
        got_emb, got_ds = wrapped(flat, grid)

        torch.testing.assert_close(got_emb, exp_emb, rtol=1e-5, atol=1e-6)
        assert len(got_ds) == len(exp_ds)
        torch.testing.assert_close(got_ds[0], exp_ds[0], rtol=1e-5, atol=1e-6)


class TestLoadBearing:
    def test_dedup_forward_repeated_images_tower_processes_fewer_rows(self):
        """Fails if the wrapper is a no-op -- the point of the whole change."""
        flat, grid, n_unique, n_total = _make([4, 4])
        rows_per_image = flat.shape[0] // n_total

        wrapped = _VisionTower()
        type(wrapped).forward = dedup_vision_forward(_ORIGINALS[_VisionTower])
        wrapped(flat, grid)

        assert wrapped.rows_seen == n_unique * rows_per_image
        assert wrapped.rows_seen < flat.shape[0]

    def test_dedup_forward_all_distinct_is_a_passthrough(self):
        flat, grid, _, n_total = _make([1, 1, 1])
        wrapped = _VisionTower()
        type(wrapped).forward = dedup_vision_forward(_ORIGINALS[_VisionTower])
        wrapped(flat, grid)
        assert wrapped.rows_seen == flat.shape[0]


class TestFallbacks:
    def test_input_requires_grad_falls_back_and_every_duplicate_gets_gradient(self):
        flat, grid, _, _ = _make([4, 4])
        flat = flat.clone().requires_grad_(True)

        wrapped = _VisionTower()
        type(wrapped).forward = dedup_vision_forward(_ORIGINALS[_VisionTower])
        wrapped(flat, grid).sum().backward()

        assert wrapped.rows_seen == flat.shape[0]  # no dedup happened
        per_image = flat.grad.reshape(8, -1).abs().sum(dim=1)
        assert (per_image > 0).all()  # no duplicate was silently starved

    def test_near_duplicates_differing_by_1e_3_are_not_merged(self):
        flat, grid, _, n_total = _make([2])
        rows = flat.shape[0] // n_total
        flat[rows:] += 1e-3

        wrapped = _VisionTower()
        type(wrapped).forward = dedup_vision_forward(_ORIGINALS[_VisionTower])
        wrapped(flat, grid)
        assert wrapped.rows_seen == flat.shape[0]

    def test_single_image_batch_skips_the_wrapper(self):
        flat, grid, _, _ = _make([1])
        wrapped = _VisionTower()
        type(wrapped).forward = dedup_vision_forward(_ORIGINALS[_VisionTower])
        wrapped(flat, grid)
        assert wrapped.rows_seen == flat.shape[0]


class TestParameterGradients:
    def test_parameter_gradients_match_undeduped(self):
        flat, grid, _, _ = _make([4, 4])

        plain = _VisionTower()
        plain(flat, grid).sum().backward()
        expected = {n: p.grad.clone() for n, p in plain.named_parameters()}

        wrapped = _VisionTower()
        wrapped.load_state_dict(plain.state_dict())
        type(wrapped).forward = dedup_vision_forward(_ORIGINALS[_VisionTower])
        wrapped(flat, grid).sum().backward()

        for n, p in wrapped.named_parameters():
            torch.testing.assert_close(p.grad, expected[n], rtol=1e-4, atol=1e-6)


class TestPatchVisionClass:
    def test_patch_vision_class_replaces_forward(self):
        original = _VisionTower.forward
        assert _patch_vision_class(_VisionTower, "_VisionTower") is True
        assert _VisionTower.forward is not original
        assert getattr(_VisionTower, "_vision_dedup_patched", False) is True

    def test_patch_vision_class_idempotent_only_wraps_once(self):
        _patch_vision_class(_VisionTower, "_VisionTower")
        first = _VisionTower.forward
        assert _patch_vision_class(_VisionTower, "_VisionTower") is False
        assert _VisionTower.forward is first


class TestApplyVisionDedup:
    def test_apply_patch_through_the_real_entry_point_does_not_raise(self):
        """Goes through the registration path, not the wrapper directly.

        A test that only constructs ``dedup_vision_forward`` stays green after
        the entry point is deleted.
        """
        apply_vision_dedup()

    def test_apply_patch_disabled_by_env_var_patches_nothing(self, monkeypatch):
        monkeypatch.setenv("AREAL_VISION_DEDUP", "0")
        assert apply_vision_dedup() == 0


class TestComposesWithVisionSpShard:
    """Both patches assign to ``cls.forward``. This is the stacking order the
    engine produces: ``apply_monkey_patch`` (and therefore Vision SP Shard) runs
    first, then ``apply_vision_dedup``, so dedup ends up on the outside.

    At ``sp_size <= 1`` ``dp_vision_forward`` passes straight through, so this
    runs on CPU with no process group and still exercises the real stacking.
    """

    def test_dedup_outside_sp_shard_matches_unpatched_output(self):
        flat, grid, _, _ = _make([4, 4])
        plain = _VisionTower()
        expected = plain(flat, grid)

        stacked = _VisionTower()
        stacked.load_state_dict(plain.state_dict())
        # order as in FSDPEngine.initialize
        _VisionTower.forward = create_dp_vision_forward(_ORIGINALS[_VisionTower])
        _VisionTower.forward = dedup_vision_forward(_VisionTower.forward)
        got = stacked(flat, grid)

        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)

    def test_dedup_outside_sp_shard_still_deduplicates(self):
        flat, grid, n_unique, n_total = _make([4, 4])
        rows_per_image = flat.shape[0] // n_total

        stacked = _VisionTower()
        _VisionTower.forward = create_dp_vision_forward(_ORIGINALS[_VisionTower])
        _VisionTower.forward = dedup_vision_forward(_VisionTower.forward)
        stacked(flat, grid)

        assert stacked.rows_seen == n_unique * rows_per_image


class TestEngineWiring:
    """The patch has to be applied by FSDPEngine.initialize, not just be applicable.

    Reverting the call site leaves every CPU test above green, because they all
    reach the wrapper directly. This is the repo's own way of covering an engine
    -applied model patch -- `tests/test_tree_training.py` builds a real
    FSDPEngine for exactly the same reason.
    """

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_fsdp_engine_initialize_patches_the_vision_tower(self):
        import os

        import areal.models.transformers.vision_dedup as vd
        from areal.api.alloc_mode import ModelAllocation
        from areal.api.cli_args import (
            MicroBatchSpec,
            OptimizerConfig,
            TrainEngineConfig,
        )
        from areal.api.io_struct import FinetuneSpec
        from areal.engine import FSDPEngine

        # Resolved inline rather than through tests.utils.get_model_path:
        # importing areal.utils.testing_utils evaluates DENSE_MODEL_PATHS and
        # MOE_MODEL_PATHS at module scope, which resolves eight checkpoints --
        # including three 30B-class MoEs -- so on any machine without the CI
        # model store the import alone starts fetching them. Only this model is
        # needed here. The local path is the spelling the repo already uses
        # (tests/test_packed_vs_padded_consistency.py).
        local = "/storage/openpsi/models/Qwen2.5-VL-3B-Instruct"
        model_path = local if os.path.exists(local) else "Qwen/Qwen2.5-VL-3B-Instruct"

        # Clear the guard so this test cannot pass on a patch some earlier test
        # left behind.
        for module_path, class_name in vd._VISION_CLASSES:
            try:
                cls = getattr(importlib.import_module(module_path), class_name)
            except (ImportError, AttributeError):
                continue
            if hasattr(cls, "_vision_dedup_patched"):
                del cls._vision_dedup_patched

        os.environ.update(
            {
                "WORLD_SIZE": "1",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "7931",
            }
        )
        config = TrainEngineConfig(
            backend="fsdp:d1",
            experiment_name="test-vision-dedup",
            trial_name="test",
            path=model_path,
            mb_spec=MicroBatchSpec(max_tokens_per_mb=256),
            optimizer=OptimizerConfig(),
            attn_impl="sdpa",
        )
        engine = FSDPEngine(config)
        alloc = ModelAllocation.from_str("fsdp:d1p1t1")
        ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=8)
        engine.create_process_group(alloc.parallel)
        engine.initialize(addr=None, ft_spec=ft_spec, parallel_strategy=alloc.parallel)
        try:
            tower = importlib.import_module(
                "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"
            ).Qwen2_5_VisionTransformerPretrainedModel
            assert getattr(tower, "_vision_dedup_patched", False) is True
        finally:
            engine.destroy()

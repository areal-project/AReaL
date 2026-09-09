"""Tests for variable-size group support in reward/advantage normalization.

These cover the ``group_sizes`` argument added to ``Normalization`` so that
group-level normalization slices each trajectory group by its actual sample
count instead of a fixed ``group_size``. Variable-size groups arise when some
rollout samples fail or are filtered, leaving groups smaller than ``n_samples``.
A fixed-size slice would otherwise straddle two groups or leave a tail.
"""

import pytest
import torch

from areal.api.cli_args import NormConfig
from areal.utils.data import Normalization, batched_call


def _group_norm_config() -> NormConfig:
    return NormConfig(
        mean_level="group",
        std_level="group",
        group_size=2,
        mean_leave1out=False,
        std_unbiased=False,
        eps=0.0,
    )


def _ref_group_norm(x: torch.Tensor, boundaries: list[int]) -> torch.Tensor:
    """Reference: independently mean/std normalize each variable-size group."""
    out = torch.empty_like(x, dtype=torch.float64)
    offset = 0
    for size in boundaries:
        chunk = x[offset : offset + size].to(torch.float64)
        mean = chunk.mean()
        std = chunk.std(unbiased=False)
        out[offset : offset + size] = (chunk - mean) / std if std > 0 else 0.0
        offset += size
    return out


def test_variable_group_sizes_normalize_each_group_independently():
    """Unequal group sizes are sliced exactly, not by a fixed group_size."""
    norm = Normalization(_group_norm_config())
    # Three groups of sizes 3, 2, 3 (total 8); fixed group_size=2 would be wrong.
    x = torch.tensor(
        [0.0, 3.0, 6.0, 10.0, 20.0, 1.0, 2.0, 3.0],
        dtype=torch.float32,
    )
    boundaries = [3, 2, 3]

    out = norm(x, group_sizes=boundaries)
    expected = _ref_group_norm(x, boundaries)

    torch.testing.assert_close(out.double(), expected, rtol=1e-5, atol=1e-5)
    # Each group is zero-mean after normalization.
    offset = 0
    for size in boundaries:
        torch.testing.assert_close(
            out[offset : offset + size].double().mean(),
            torch.tensor(0.0, dtype=torch.float64),
            rtol=1e-5,
            atol=1e-5,
        )
        offset += size


def test_group_sizes_none_matches_fixed_group_size():
    """Without group_sizes, behavior is unchanged (backward compatible)."""
    norm = Normalization(_group_norm_config())  # group_size=2
    x = torch.tensor([0.0, 2.0, 10.0, 14.0], dtype=torch.float32)

    out_default = norm(x)
    out_explicit = norm(x, group_sizes=[2, 2])

    torch.testing.assert_close(out_default, out_explicit, rtol=1e-6, atol=1e-6)


def test_variable_group_with_zero_variance_group_is_finite():
    """A zero-variance group stays finite thanks to eps in the denominator.

    When a group's rewards are all equal its std is 0; dividing by ``std + eps``
    (eps > 0) keeps the result finite (and zero, since the numerator is also 0)
    instead of producing NaN/inf. This is the guard that prevents a degenerate
    group from blowing up advantages.
    """
    cfg = NormConfig(
        mean_level="group",
        std_level="group",
        group_size=2,
        mean_leave1out=False,
        std_unbiased=False,
        eps=1e-5,
    )
    norm = Normalization(cfg)
    # Group 0 (size 3) has variance; group 1 (size 2) is all-equal -> std 0.
    x = torch.tensor([0.0, 3.0, 6.0, 5.0, 5.0], dtype=torch.float32)
    boundaries = [3, 2]

    out = norm(x, group_sizes=boundaries)

    assert torch.isfinite(out).all(), "normalized output must be finite"
    # The degenerate group collapses to zero (x_centered=0 / (0+eps)).
    torch.testing.assert_close(
        out[3:].double(),
        torch.zeros(2, dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_leave_one_out_singleton_group_outputs_zero():
    """A size-1 leave-one-out group has no peer baseline, so it is zeroed."""
    cfg = NormConfig(
        mean_level="group",
        std_level="group",
        group_size=1,
        mean_leave1out=True,
        std_unbiased=True,
        eps=1e-5,
    )
    norm = Normalization(cfg)
    x = torch.tensor([7.0, 1.0, 5.0], dtype=torch.float32)
    # One singleton group then a size-2 group.
    out = norm(x, group_sizes=[1, 2])

    assert torch.isfinite(out).all()
    assert out[0].item() == 0.0


def test_group_sizes_sum_mismatch_raises():
    """Boundaries whose sizes do not sum to the batch size are rejected."""
    norm = Normalization(_group_norm_config())
    x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)  # bs=4
    with pytest.raises(ValueError, match="must equal"):
        norm(x, group_sizes=[3, 2])  # sums to 5 != 4


def test_group_sizes_non_positive_raises():
    """Boundaries containing a non-positive size are rejected."""
    norm = Normalization(_group_norm_config())
    x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)  # bs=4
    with pytest.raises(ValueError, match="positive"):
        norm(x, group_sizes=[4, 0])


def test_group_sizes_none_rejects_non_divisible_group_batch():
    """Implicit fixed-size grouping must not leave a partial tail unnormalized."""
    norm = Normalization(_group_norm_config())  # group_size=2
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)  # bs=3

    with pytest.raises(ValueError, match="divisible by group_size"):
        norm(x)


def test_adv_norm_style_2d_variable_groups_normalize_per_group():
    """Token-level (2D) advantages normalize per variable-size group on dim 0.

    Mirrors how PPOActor.adv_norm is applied to the GAE advantage tensor.
    """
    norm = _group_norm_config()  # group mean+std, eps=0
    adv_norm = Normalization(norm)
    # bs=5 (groups [3, 2]), seqlen=2.
    x = torch.tensor(
        [[0.0, 6.0], [3.0, 6.0], [6.0, 6.0], [10.0, 0.0], [20.0, 0.0]],
        dtype=torch.float32,
    )
    loss_mask = torch.ones_like(x)
    out = adv_norm(x, loss_mask, group_sizes=[3, 2])

    assert torch.isfinite(out).all()
    assert out.shape == x.shape


def test_batched_call_pass_meta_provides_traj_group_sizes():
    data = [
        {
            "attention_mask": torch.ones(2, 3, dtype=torch.bool),
            "values": torch.tensor([[1.0], [2.0]]),
        },
        {
            "attention_mask": torch.ones(1, 3, dtype=torch.bool),
            "values": torch.tensor([[3.0]]),
        },
    ]

    def fn(batch, meta):
        assert meta.traj_group_sizes == [2, 1]
        return {"values": batch["values"] + 1}

    out = batched_call(fn, data, pass_meta=True)

    assert len(out) == 2
    torch.testing.assert_close(out[0]["values"], torch.tensor([[2.0], [3.0]]))
    torch.testing.assert_close(out[1]["values"], torch.tensor([[4.0]]))

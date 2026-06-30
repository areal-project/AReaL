import pytest
import torch

from areal.api.cli_args import NormConfig
from areal.utils.data import Normalization, concat_batch


def _group_norm(mean_level="group", std_level="group", group_size=4, **kwargs):
    return Normalization(
        NormConfig(
            mean_level=mean_level,
            std_level=std_level,
            group_size=group_size,
            **kwargs,
        )
    )


def test_group_sizes_match_fixed_stride_for_full_groups():
    reward = torch.tensor([0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0, 3.0])
    advantages = torch.arange(40, dtype=torch.float32).reshape(8, 5)
    loss_mask = torch.ones(8, 5)
    loss_mask[:, -1] = 0.0

    norm = _group_norm()

    torch.testing.assert_close(norm(reward), norm(reward, group_sizes=[4, 4]))
    torch.testing.assert_close(
        norm(advantages, loss_mask),
        norm(advantages, loss_mask, group_sizes=[4, 4]),
    )


def test_group_sizes_include_trailing_partial_group():
    x = torch.tensor([0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0, 3.0, 0.0, 1.0, 2.0])

    out = _group_norm()(x, group_sizes=[4, 4, 3])

    assert torch.isfinite(out).all()
    for group_slice in (slice(0, 4), slice(4, 8), slice(8, 11)):
        torch.testing.assert_close(
            out[group_slice].mean(), torch.tensor(0.0), atol=1e-6, rtol=0
        )


def test_group_sizes_keep_later_groups_aligned():
    x = torch.tensor(
        [
            0.5,
            -0.5,
            1.0,
            -1.0,
            2.0,
            -2.0,
            0.0,
            3.0,
            0.0,
            1.0,
            2.0,
            100.0,
        ]
    )
    norm = _group_norm()

    fixed_stride = norm(x)
    actual_groups = norm(x, group_sizes=[4, 4, 3, 1])

    torch.testing.assert_close(fixed_stride[:8], actual_groups[:8])
    assert fixed_stride[10] < 0
    assert actual_groups[10] > 0
    torch.testing.assert_close(actual_groups[11], torch.tensor(0.0), atol=1e-6, rtol=0)


@pytest.mark.parametrize("std_unbiased", [True, False])
def test_single_row_group_has_finite_zero_reward_advantage(std_unbiased):
    x = torch.tensor([0.5, -0.5, 1.0, -1.0, 8.0])

    out = _group_norm(std_unbiased=std_unbiased)(x, group_sizes=[4, 1])

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[-1], torch.tensor(0.0), atol=1e-6, rtol=0)


@pytest.mark.parametrize(
    ("x_size", "group_sizes", "match"),
    [
        (11, None, "not divisible"),
        (8, [4, 3], "group_sizes sum"),
        (8, [8, 0], "group_sizes must be positive"),
    ],
)
def test_invalid_group_sizes_raise(x_size, group_sizes, match):
    with pytest.raises(ValueError, match=match):
        _group_norm()(torch.randn(x_size), group_sizes=group_sizes)


def test_batch_level_norm_does_not_require_group_sizes():
    out = _group_norm(mean_level="batch", std_level="batch")(torch.randn(11))

    assert torch.isfinite(out).all()


def test_concat_batch_records_group_sizes():
    def traj(group_size):
        return {
            "rewards": torch.randn(group_size),
            "attention_mask": torch.ones(group_size, 3),
        }

    _, meta = concat_batch([traj(4), traj(4), traj(3)])

    assert meta.traj_group_sizes == [4, 4, 3]

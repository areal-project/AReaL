import pytest
import torch

from areal.api.cli_args import NormConfig
from areal.utils.data import Normalization, concat_batch


def _group_norm(mean_level="group", std_level="group", group_size=4, **kw):
    return Normalization(
        NormConfig(
            mean_level=mean_level, std_level=std_level, group_size=group_size, **kw
        )
    )


def test_group_sizes_matches_positional_for_full_groups_reward_path():
    torch.manual_seed(0)
    x = torch.randn(8)
    norm = _group_norm()
    torch.testing.assert_close(norm(x), norm(x, group_sizes=[4, 4]))


def test_group_sizes_matches_positional_for_full_groups_adv_path():
    torch.manual_seed(1)
    x = torch.randn(8, 5)
    mask = torch.ones(8, 5)
    mask[:, 4] = 0.0
    norm = _group_norm()
    torch.testing.assert_close(norm(x, mask), norm(x, mask, group_sizes=[4, 4]))


def test_trailing_partial_group_does_not_blow_up():
    torch.manual_seed(2)
    x = torch.randn(11) * 3.0
    out = _group_norm()(x, group_sizes=[4, 4, 3])
    assert torch.isfinite(out).all()
    for s in (slice(0, 4), slice(4, 8), slice(8, 11)):
        torch.testing.assert_close(out[s].mean(), torch.tensor(0.0), atol=1e-5, rtol=0)


def test_mid_batch_partial_group_avoids_cross_prompt_sign_flip():
    x = torch.tensor([0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.0, 3.0, 0.0, 1.0, 2.0, 100.0])
    norm = _group_norm()

    out_positional = norm(x)
    out_grouped = norm(x, group_sizes=[4, 4, 3, 1])

    torch.testing.assert_close(out_positional[:8], out_grouped[:8])
    assert out_grouped[10] > 0 and out_positional[10] < 0
    torch.testing.assert_close(out_grouped[11], torch.tensor(0.0), atol=1e-6, rtol=0)


@pytest.mark.parametrize("std_unbiased", [True, False])
def test_singleton_group_is_finite_and_zero(std_unbiased):
    torch.manual_seed(3)
    x = torch.randn(5) * 10.0
    out = _group_norm(std_unbiased=std_unbiased)(x, group_sizes=[4, 1])
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[4], torch.tensor(0.0), atol=1e-6, rtol=0)


@pytest.mark.parametrize("std_unbiased", [True, False])
def test_singleton_group_masked_path_is_finite(std_unbiased):
    torch.manual_seed(4)
    x = torch.randn(5, 6) * 10.0
    mask = torch.ones(5, 6)
    mask[:, 5] = 0.0
    out = _group_norm(std_unbiased=std_unbiased)(x, mask, group_sizes=[4, 1])
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[:, 5], torch.zeros(5), atol=1e-6, rtol=0)


def test_non_divisible_batch_without_group_sizes_raises():
    with pytest.raises(ValueError, match="not divisible"):
        _group_norm()(torch.randn(11))


def test_group_sizes_must_sum_to_batch_size():
    with pytest.raises(ValueError, match="group_sizes sum"):
        _group_norm()(torch.randn(8), group_sizes=[4, 3])


def test_group_sizes_must_be_positive():
    with pytest.raises(ValueError, match="group_sizes must be positive"):
        _group_norm()(torch.randn(8), group_sizes=[8, 0])


def test_batch_level_ignores_group_sizes():
    out = _group_norm(mean_level="batch", std_level="batch")(torch.randn(11))
    assert torch.isfinite(out).all()


def test_concat_batch_records_partial_group_sizes():
    def traj(k):
        return {"rewards": torch.randn(k), "attention_mask": torch.ones(k, 3)}

    _, meta = concat_batch([traj(4), traj(4), traj(3)])
    assert meta.traj_group_sizes == [4, 4, 3]

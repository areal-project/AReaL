import pytest
import torch

from areal.api.cli_args import RejectionSamplingConfig
from areal.utils.functional import apply_rejection_sampling
from areal.utils.functional.functional import compute_binary_kl_divergence

DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


@pytest.mark.parametrize("upper", [0.0, -0.1])
def test_binary_kl_requires_positive_upper_bound(upper):
    with pytest.raises(ValueError, match="upper must be positive"):
        RejectionSamplingConfig(metric="binary_kl", upper=upper)


@pytest.mark.parametrize("dtype", DTYPES)
def test_identical_distributions_have_zero_binary_kl_at_endpoints(dtype):
    log_probs = torch.tensor(
        [0.0, -1e-8, -1e-4, -0.01, -0.1, -1.0, -10.0, -100.0, -1000.0],
        dtype=dtype,
    )

    divergence = compute_binary_kl_divergence(log_probs, log_probs)

    assert divergence.dtype == dtype
    assert torch.isfinite(divergence).all()
    torch.testing.assert_close(divergence, torch.zeros_like(log_probs))


@pytest.mark.parametrize("dtype", DTYPES)
def test_binary_kl_matches_bernoulli_reference(dtype):
    p = torch.tensor([0.1, 0.4, 0.7, 0.9], dtype=dtype)
    q = torch.tensor([0.2, 0.8, 0.3, 0.6], dtype=dtype)
    log_p, log_q = p.log(), q.log()
    # Compare the same represented probabilities to a float64 reference.
    p64, q64 = log_p.exp().double(), log_q.exp().double()
    expected = p64 * (p64 / q64).log() + (1 - p64) * ((1 - p64) / (1 - q64)).log()

    actual = compute_binary_kl_divergence(log_p, log_q)

    torch.testing.assert_close(actual, expected.to(dtype), rtol=0.03, atol=0.002)


@pytest.mark.parametrize("dtype", DTYPES)
def test_binary_kl_unit_probability_against_interior_probability(dtype):
    log_p = torch.tensor([0.0], dtype=dtype)
    log_q = torch.tensor([0.5], dtype=dtype).log()

    actual = compute_binary_kl_divergence(log_p, log_q)

    torch.testing.assert_close(actual, -log_q, rtol=0.03, atol=0.002)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("level", ["token", "sequence"])
@pytest.mark.parametrize("packed", [False, True])
def test_binary_kl_rejection_keeps_identical_policies(dtype, level, packed):
    log_probs = torch.tensor([[0.0, -1e-4, -1.0], [-0.01, -10.0, -100.0]], dtype=dtype)
    loss_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    cu_seqlens = None
    if packed:
        log_probs = log_probs.flatten()
        loss_mask = loss_mask.flatten()
        cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    config = RejectionSamplingConfig(level=level, metric="binary_kl", upper=0.1)

    result = apply_rejection_sampling(
        log_probs, log_probs, loss_mask, cu_seqlens=cu_seqlens, config=config
    )

    torch.testing.assert_close(result.loss_mask, loss_mask)
    torch.testing.assert_close(
        result.behave_imp_weight, loss_mask.to(result.behave_imp_weight.dtype)
    )
    assert result.filtered_fraction == 0.0


def test_binary_kl_rejection_still_filters_different_policies():
    config = RejectionSamplingConfig(metric="binary_kl", upper=0.1)
    old_logprobs = torch.tensor([[0.5, 0.5]]).log()
    proximal_logprobs = torch.tensor([[1.0, 0.5]]).log()

    result = apply_rejection_sampling(
        proximal_logprobs,
        old_logprobs,
        torch.ones(1, 2, dtype=torch.bool),
        cu_seqlens=None,
        config=config,
    )

    torch.testing.assert_close(result.loss_mask, torch.tensor([[False, True]]))
    assert result.filtered_fraction == 0.5

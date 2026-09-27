# SPDX-License-Identifier: Apache-2.0
"""The streamed ridge solve recovers a planted affine map and rejects ill-posed inputs."""

import pytest
import torch

from areal.experimental.kvmap.ridge import RidgeAccumulator

FEATURE_DIM = 6
TARGET_DIM = 3
ROWS = 500


def _planted(seed: int):
    generator = torch.Generator().manual_seed(seed)
    features = (
        torch.randn(ROWS, FEATURE_DIM, dtype=torch.float64, generator=generator) + 2.0
    )
    weight = torch.randn(
        FEATURE_DIM, TARGET_DIM, dtype=torch.float64, generator=generator
    )
    bias = torch.randn(TARGET_DIM, dtype=torch.float64, generator=generator)
    return features, weight, bias, features @ weight + bias


def test_noise_free_planted_map_is_recovered_with_unit_r_squared():
    features, weight, bias, targets = _planted(0)
    accumulator = RidgeAccumulator(feature_dim=FEATURE_DIM, target_dim=TARGET_DIM)
    accumulator.update(features, targets)

    fit = accumulator.solve(lambda_=1e-9)

    torch.testing.assert_close(fit.weight, weight, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(fit.bias, bias, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(
        fit.r_squared, torch.ones(TARGET_DIM, dtype=torch.float64), rtol=0.0, atol=1e-9
    )
    assert fit.rows == ROWS


def test_streaming_in_chunks_equals_one_batch():
    features, _, _, targets = _planted(1)
    whole = RidgeAccumulator(feature_dim=FEATURE_DIM, target_dim=TARGET_DIM)
    whole.update(features, targets)
    chunked = RidgeAccumulator(feature_dim=FEATURE_DIM, target_dim=TARGET_DIM)
    for start in (0, 1, 128, 129, 499):
        end = {0: 1, 1: 128, 128: 129, 129: 499, 499: 500}[start]
        chunked.update(features[start:end], targets[start:end])

    torch.testing.assert_close(
        chunked.solve(lambda_=0.01).weight,
        whole.solve(lambda_=0.01).weight,
        rtol=1e-12,
        atol=1e-12,
    )


def test_larger_lambda_shrinks_the_weight_so_regularization_is_live():
    features, _, _, targets = _planted(2)
    accumulator = RidgeAccumulator(feature_dim=FEATURE_DIM, target_dim=TARGET_DIM)
    accumulator.update(features, targets)

    small = accumulator.solve(lambda_=1e-6).weight.norm()
    large = accumulator.solve(lambda_=1e4).weight.norm()

    assert large < 0.5 * small


@pytest.mark.parametrize("lambda_", [0.0, -1.0])
def test_non_positive_lambda_is_rejected(lambda_):
    features, _, _, targets = _planted(3)
    accumulator = RidgeAccumulator(feature_dim=FEATURE_DIM, target_dim=TARGET_DIM)
    accumulator.update(features, targets)
    with pytest.raises(ValueError, match="lambda_ must be > 0"):
        accumulator.solve(lambda_=lambda_)


def test_solve_with_one_row_is_rejected():
    accumulator = RidgeAccumulator(feature_dim=FEATURE_DIM, target_dim=TARGET_DIM)
    accumulator.update(torch.ones(1, FEATURE_DIM), torch.ones(1, TARGET_DIM))
    with pytest.raises(ValueError, match="at least 2 rows, got 1"):
        accumulator.solve(lambda_=0.01)


def test_update_rejects_dimension_and_row_mismatch():
    accumulator = RidgeAccumulator(feature_dim=FEATURE_DIM, target_dim=TARGET_DIM)
    with pytest.raises(ValueError, match="same n"):
        accumulator.update(torch.ones(3, FEATURE_DIM), torch.ones(2, TARGET_DIM))
    with pytest.raises(ValueError, match="expected feature_dim 6"):
        accumulator.update(torch.ones(3, FEATURE_DIM + 1), torch.ones(3, TARGET_DIM))
    with pytest.raises(ValueError, match="at least one row"):
        accumulator.update(torch.ones(0, FEATURE_DIM), torch.ones(0, TARGET_DIM))
    assert accumulator.rows == 0

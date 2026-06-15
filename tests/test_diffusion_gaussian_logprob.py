"""Unit tests for ``DiffusionInferenceEngine._gaussian_logprob``.

These tests pin down a regression that previously lived in the DDIM log-prob
kernel: the standard deviation used to be turned into a Python float via
``math.log(float(std))``. Calling ``float()`` on a tensor (a) forces a
device-to-host sync on every denoising step in the hot path, and (b) raises
outright once ``std`` carries more than one element. The current
implementation keeps everything on-device with ``torch.log(std)``.

The key regression guard here is ``test_per_element_std_does_not_crash`` /
``test_matches_torch_normal_per_element_std``: if anyone reintroduces
``float(std)`` on a non-scalar tensor, these tests fail immediately instead of
silently only working for scalar std.

All tests are pure CPU tensor math -- no GPU, no diffusers pipeline required.
"""

import math

import pytest
import torch

from areal.experimental.diffusion.diffusion_engine import DiffusionInferenceEngine

# Bind the staticmethod once for readability.
_gaussian_logprob = DiffusionInferenceEngine._gaussian_logprob


def _reference_logprob(x: torch.Tensor, mean: torch.Tensor, std) -> torch.Tensor:
    """Independent reference: per-sample Gaussian log-density summed over all
    dims except the leading batch dim.

    Intentionally written with ``torch.distributions.Normal`` so it does not
    share code with the implementation under test.
    """
    std_t = torch.as_tensor(std, dtype=x.dtype) + 1e-8
    logp = torch.distributions.Normal(mean, std_t).log_prob(x)
    return logp.sum(dim=tuple(range(1, logp.ndim)))


class TestGaussianLogprob:
    """Test suite for the on-device Gaussian log-probability kernel."""

    def test_scalar_std_matches_reference(self):
        """Scalar std (the easy, always-worked case) matches the reference."""
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4)
        mean = torch.randn(2, 3, 4)
        std = torch.tensor(0.7)

        out = _gaussian_logprob(x, mean, std)
        expected = _reference_logprob(x, mean, std)

        assert out.shape == (2,)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_per_element_std_does_not_crash(self):
        """REGRESSION: a per-element std tensor must not raise.

        The old ``math.log(float(std))`` implementation raised here because
        ``float()`` only accepts a one-element tensor. This is the core guard
        that prevents that code from being reintroduced.
        """
        torch.manual_seed(1)
        x = torch.randn(2, 3, 4)
        mean = torch.randn(2, 3, 4)
        std = torch.rand(2, 3, 4) + 0.1  # strictly positive, multi-element

        # Must complete without raising RuntimeError.
        out = _gaussian_logprob(x, mean, std)
        assert out.shape == (2,)
        assert torch.isfinite(out).all()

    def test_matches_torch_normal_per_element_std(self):
        """Per-element std produces the correct log-density values."""
        torch.manual_seed(2)
        x = torch.randn(3, 5)
        mean = torch.randn(3, 5)
        std = torch.rand(3, 5) + 0.2

        out = _gaussian_logprob(x, mean, std)
        expected = _reference_logprob(x, mean, std)

        assert out.shape == (3,)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_scalar_and_broadcast_std_agree(self):
        """A scalar std and a full tensor filled with that same value agree."""
        torch.manual_seed(3)
        x = torch.randn(4, 6)
        mean = torch.randn(4, 6)

        scalar_std = torch.tensor(0.5)
        tensor_std = torch.full_like(x, 0.5)

        out_scalar = _gaussian_logprob(x, mean, scalar_std)
        out_tensor = _gaussian_logprob(x, mean, tensor_std)

        torch.testing.assert_close(out_scalar, out_tensor, rtol=1e-5, atol=1e-5)

    def test_known_closed_form_value(self):
        """Check one hand-computed value end to end (no reliance on Normal)."""
        # x == mean, so (x-mean)^2 term vanishes; only the normalization remains.
        x = torch.zeros(1, 1)
        mean = torch.zeros(1, 1)
        std = torch.tensor(1.0)

        # log N(0; 0, 1) = -0.5 * log(2*pi); std gets +1e-8 internally.
        std_eff = 1.0 + 1e-8
        expected_val = -0.5 * (2 * math.log(std_eff) + math.log(2 * math.pi))

        out = _gaussian_logprob(x, mean, std)
        torch.testing.assert_close(
            out, torch.tensor([expected_val], dtype=out.dtype), rtol=1e-6, atol=1e-6
        )

    def test_differentiable_wrt_mean(self):
        """Gradient flows back to ``mean`` (needed for the policy gradient)."""
        torch.manual_seed(4)
        x = torch.randn(2, 3)
        mean = torch.randn(2, 3, requires_grad=True)
        std = torch.rand(2, 3) + 0.3

        out = _gaussian_logprob(x, mean, std)
        out.sum().backward()

        assert mean.grad is not None
        assert torch.isfinite(mean.grad).all()

    def test_no_python_float_on_tensor_std(self):
        """Guard against ``float()`` being called on a >1-element std tensor.

        We pass a custom tensor subclass that makes ``float()`` blow up so the
        test fails loudly if anyone reintroduces ``math.log(float(std))``. The
        current ``torch.log(std)`` implementation never calls ``float()``.
        """

        class NoFloatTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data)

            def __float__(self):  # pragma: no cover - only hit on regression
                raise AssertionError(
                    "float() was called on the std tensor -- "
                    "this reintroduces the device-sync / multi-element bug"
                )

        x = torch.randn(2, 4)
        mean = torch.randn(2, 4)
        std = NoFloatTensor(torch.rand(2, 4) + 0.5)

        # Should run purely with torch ops; __float__ must never be invoked.
        out = _gaussian_logprob(x, mean, std)
        assert out.shape == (2,)
        assert torch.isfinite(out).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

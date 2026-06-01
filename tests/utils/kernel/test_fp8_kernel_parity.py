# SPDX-License-Identifier: Apache-2.0

"""Parity tests between Triton and PyTorch fallback FP8 quantization paths."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Mock areal.utils.math before loading the kernel module
# ---------------------------------------------------------------------------
math_mod = types.ModuleType("areal.utils.math")
math_mod.ceil_div = lambda x, y: (x + y - 1) // y
sys.modules["areal"] = types.ModuleType("areal")
sys.modules["areal.utils"] = types.ModuleType("areal.utils")
sys.modules["areal.utils"].__path__ = []
sys.modules["areal.utils.math"] = math_mod

_KERNEL_PATH = Path(__file__).parents[3] / "areal" / "utils" / "kernel" / "fp8_kernel.py"

spec = importlib.util.spec_from_file_location("fp8_kernel_parity", str(_KERNEL_PATH))
fp8_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp8_mod)

CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = fp8_mod._TRITON_AVAILABLE


# ---------------------------------------------------------------------------
# TestTritonFallbackParity
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestTritonFallbackParity:
    """Verify Triton kernel and PyTorch fallback produce identical outputs."""

    @pytest.mark.parametrize(
        "shape,block_size",
        [
            ((256, 512), [128, 128]),
            ((4096, 4096), [128, 128]),
            ((4096, 14336), [128, 128]),
            ((100, 300), [128, 128]),
            ((128, 128), [128, 128]),
            ((8192, 8192), [128, 128]),
        ],
    )
    def test_triton_vs_fallback(self, shape, block_size, monkeypatch):
        """Same input through Triton and fallback paths must match."""
        torch.manual_seed(42)
        data = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

        # Triton path
        monkeypatch.delenv("DISABLE_TRITON_FP8", raising=False)
        importlib.reload(fp8_mod)
        fp8_triton, scale_triton = fp8_mod.scaled_fp8_blockwise(
            data, weight_block_size=block_size
        )

        # PyTorch fallback path
        monkeypatch.setenv("DISABLE_TRITON_FP8", "1")
        importlib.reload(fp8_mod)
        fp8_fallback, scale_fallback = fp8_mod.scaled_fp8_blockwise(
            data, weight_block_size=block_size
        )

        torch.testing.assert_close(fp8_triton, fp8_fallback)
        torch.testing.assert_close(scale_triton, scale_fallback)

    def test_all_zeros_parity(self, monkeypatch):
        """All-zeros input: both paths must return scale == 1.0."""
        data = torch.zeros(128, 128, dtype=torch.bfloat16, device="cuda")

        monkeypatch.delenv("DISABLE_TRITON_FP8", raising=False)
        importlib.reload(fp8_mod)
        fp8_triton, scale_triton = fp8_mod.scaled_fp8_blockwise(data)

        monkeypatch.setenv("DISABLE_TRITON_FP8", "1")
        importlib.reload(fp8_mod)
        fp8_fallback, scale_fallback = fp8_mod.scaled_fp8_blockwise(data)

        torch.testing.assert_close(fp8_triton, fp8_fallback)
        torch.testing.assert_close(scale_triton, scale_fallback)
        assert scale_triton.item() == pytest.approx(1.0, abs=1e-5)
        assert scale_fallback.item() == pytest.approx(1.0, abs=1e-5)

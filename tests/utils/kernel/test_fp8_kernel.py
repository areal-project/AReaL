# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the FP8 block-wise quantization kernel."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

# Force PyTorch fallback for Triton-incompatible GPUs (e.g. SM86).
os.environ["DISABLE_TRITON_FP8"] = "1"

# ---------------------------------------------------------------------------
# Mock areal.utils.math before loading the kernel module
# ---------------------------------------------------------------------------
math_mod = types.ModuleType("areal.utils.math")
math_mod.ceil_div = lambda x, y: (x + y - 1) // y
sys.modules["areal"] = types.ModuleType("areal")
sys.modules["areal.utils"] = types.ModuleType("areal.utils")
sys.modules["areal.utils"].__path__ = []
sys.modules["areal.utils.math"] = math_mod

spec = importlib.util.spec_from_file_location(
    "fp8_kernel",
    str(Path(__file__).parents[3] / "areal" / "utils" / "kernel" / "fp8_kernel.py"),
)
fp8_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp8_mod)

scaled_fp8_blockwise = fp8_mod.scaled_fp8_blockwise
should_quantize_param = fp8_mod.should_quantize_param
FP8_MAX = fp8_mod.FP8_MAX


# ---------------------------------------------------------------------------
# TestShouldQuantizeParam
# ---------------------------------------------------------------------------
class TestShouldQuantizeParam:
    """Tests for should_quantize_param()."""

    @pytest.mark.parametrize(
        "param_name",
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ],
    )
    def test_quantize_linear_layers(self, param_name: str) -> None:
        """Linear projection weights should be quantized."""
        assert should_quantize_param(param_name) is True

    def test_skip_embedding(self) -> None:
        """Embedding token weights should NOT be quantized."""
        assert should_quantize_param("model.embed_tokens.weight") is False

    def test_skip_lm_head(self) -> None:
        """LM head weights should NOT be quantized."""
        assert should_quantize_param("lm_head.weight") is False

    @pytest.mark.parametrize(
        "param_name",
        [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.norm.weight",
        ],
    )
    def test_skip_norm(self, param_name: str) -> None:
        """Normalization layer weights should NOT be quantized."""
        assert should_quantize_param(param_name) is False

    def test_skip_bias(self) -> None:
        """Bias parameters (not .weight) should NOT be quantized."""
        assert should_quantize_param("model.layers.0.self_attn.q_proj.bias") is False

    def test_skip_moe_router(self) -> None:
        """MoE router gate weights should NOT be quantized."""
        assert should_quantize_param("model.layers.0.mlp.gate.weight") is False


# ---------------------------------------------------------------------------
# TestScaledFp8Blockwise
# ---------------------------------------------------------------------------
class TestScaledFp8Blockwise:
    """Tests for scaled_fp8_blockwise()."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_basic_quantization(self) -> None:
        """256x512 BF16 -> fp8 shape (256,512), scale shape (2,4)."""
        data = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
        fp8_data, scale = scaled_fp8_blockwise(data, weight_block_size=[128, 128])

        assert fp8_data.shape == (256, 512)
        assert fp8_data.dtype == torch.float8_e4m3fn
        assert scale.shape == (2, 4)
        assert scale.dtype == torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_roundtrip_dequant_approximate(self) -> None:
        """Quant then dequant; relative error should be < 5%."""
        torch.manual_seed(42)
        data = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
        fp8_data, scale = scaled_fp8_blockwise(data, weight_block_size=[128, 128])

        # Dequantize: fp8_data * scale (per-block)
        blk_m, blk_n = scale.shape
        block_m = data.shape[0] // blk_m
        block_n = data.shape[1] // blk_n

        dequant = torch.zeros_like(data, dtype=torch.float32)
        fp8_f32 = fp8_data.to(torch.float32)
        for i in range(blk_m):
            for j in range(blk_n):
                row_start = i * block_m
                row_end = row_start + block_m
                col_start = j * block_n
                col_end = col_start + block_n
                dequant[row_start:row_end, col_start:col_end] = (
                    fp8_f32[row_start:row_end, col_start:col_end] * scale[i, j]
                )

        data_f32 = data.to(torch.float32)
        rel_err = (dequant - data_f32).abs().mean() / data_f32.abs().mean()
        assert rel_err < 0.05, f"Mean relative error {rel_err} >= 0.05"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_non_multiple_dimensions(self) -> None:
        """100x300 (not multiple of 128) -> fp8 shape (100,300), scale shape (1,3)."""
        data = torch.randn(100, 300, dtype=torch.bfloat16, device="cuda")
        fp8_data, scale = scaled_fp8_blockwise(data, weight_block_size=[128, 128])

        assert fp8_data.shape == (100, 300)
        assert fp8_data.dtype == torch.float8_e4m3fn
        assert scale.shape == (1, 3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_all_zeros(self) -> None:
        """128x128 zeros -> scale should be 1.0."""
        data = torch.zeros(128, 128, dtype=torch.bfloat16, device="cuda")
        fp8_data, scale = scaled_fp8_blockwise(data, weight_block_size=[128, 128])

        assert fp8_data.shape == (128, 128)
        assert scale.shape == (1, 1)
        assert scale.item() == pytest.approx(1.0, abs=1e-5)

    def test_cpu_fallback(self) -> None:
        """CPU BF16 tensor should work via PyTorch fallback."""
        data = torch.randn(128, 128, dtype=torch.bfloat16, device="cpu")
        fp8_data, scale = scaled_fp8_blockwise(data, weight_block_size=[128, 128])

        assert fp8_data.shape == (128, 128)
        assert fp8_data.dtype == torch.float8_e4m3fn
        assert scale.shape == (1, 1)
        assert scale.dtype == torch.float32

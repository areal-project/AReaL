import pytest
import torch

from areal.experimental.models.archon.fp8_checkpoint import (
    dequant_fp8_state_dict,
    weight_dequant_cpu,
)

CUDA_AVAILABLE = torch.cuda.is_available()


class TestWeightDequantCPU:
    def test_basic_correctness(self):
        x_bf16 = torch.randn(128, 128, dtype=torch.bfloat16)
        x_fp8 = x_bf16.to(torch.float8_e4m3fn)
        scale_inv = torch.tensor([[2.0]], dtype=torch.float32)

        result = weight_dequant_cpu(x_fp8, scale_inv, block_size=128)
        expected = x_fp8.to(torch.bfloat16) * 2.0
        torch.testing.assert_close(result, expected.to(torch.bfloat16), rtol=0, atol=0)

    def test_output_dtype(self):
        x_fp8 = torch.zeros(128, 128, dtype=torch.float8_e4m3fn)
        scale_inv = torch.ones(1, 1, dtype=torch.float32)
        result = weight_dequant_cpu(x_fp8, scale_inv, dst_dtype=torch.bfloat16)
        assert result.dtype == torch.bfloat16

    def test_multi_block(self):
        x_fp8 = torch.ones(256, 256, dtype=torch.float8_e4m3fn)
        scale_inv = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

        result = weight_dequant_cpu(x_fp8, scale_inv, block_size=128)
        assert result.dtype == torch.bfloat16
        torch.testing.assert_close(
            result[:128, :128].float().mean(),
            torch.tensor(1.0),
            rtol=1e-2,
            atol=1e-2,
        )
        torch.testing.assert_close(
            result[:128, 128:256].float().mean(),
            torch.tensor(2.0),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_non_aligned_dimensions(self):
        x_fp8 = torch.ones(200, 300, dtype=torch.float8_e4m3fn)
        scale_inv = torch.ones(2, 3, dtype=torch.float32)

        result = weight_dequant_cpu(x_fp8, scale_inv, block_size=128)
        assert result.shape == (200, 300)
        assert result.dtype == torch.bfloat16


class TestDequantFP8StateDict:
    def _make_fp8_state_dict(self):
        M, N = 256, 512
        return {
            "model.layers.0.q_proj.weight": torch.randn(M, N, dtype=torch.bfloat16).to(
                torch.float8_e4m3fn
            ),
            "model.layers.0.q_proj.weight_scale_inv": torch.ones(
                M // 128, N // 128, dtype=torch.float32
            ),
        }

    def test_replaces_fp8_with_bf16(self):
        result = dequant_fp8_state_dict(self._make_fp8_state_dict())
        assert result["model.layers.0.q_proj.weight"].dtype == torch.bfloat16

    def test_removes_scale_keys(self):
        result = dequant_fp8_state_dict(self._make_fp8_state_dict())
        assert "model.layers.0.q_proj.weight_scale_inv" not in result

    def test_bf16_passthrough(self):
        sd = {
            "model.layers.0.q_proj.weight": torch.randn(256, 512, dtype=torch.bfloat16)
        }
        assert dequant_fp8_state_dict(sd) is sd

    def test_missing_scale_key_raises(self):
        sd = {
            "model.layers.0.q_proj.weight": torch.zeros(
                256, 512, dtype=torch.float8_e4m3fn
            ),
        }
        with pytest.raises(KeyError, match="no matching scale"):
            dequant_fp8_state_dict(sd)

    def test_preserves_non_fp8_keys(self):
        sd = self._make_fp8_state_dict()
        sd["model.norm.weight"] = torch.randn(256, dtype=torch.bfloat16)
        result = dequant_fp8_state_dict(sd)
        assert "model.norm.weight" in result
        assert result["model.norm.weight"].dtype == torch.bfloat16


class TestDequantGPU:
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_plain_tensor_gpu_dequant(self):
        M, N = 256, 256
        torch.manual_seed(42)
        weight_fp8 = torch.randn(M, N, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_inv = torch.rand(2, 2, dtype=torch.float32) + 0.5

        expected = weight_dequant_cpu(weight_fp8, scale_inv)

        sd = {
            "w": weight_fp8,
            "w_scale_inv": scale_inv,
        }

        result_sd = dequant_fp8_state_dict(sd, target_dtype=torch.bfloat16)

        assert result_sd["w"].dtype == torch.bfloat16
        torch.testing.assert_close(result_sd["w"].cpu(), expected, rtol=1e-3, atol=1e-3)

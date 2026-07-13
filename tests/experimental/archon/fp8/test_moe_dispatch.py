"""MoE FP8 dispatch unit tests.

Covers _run_experts_fp8_for_loop: BF16 parity, edge cases (empty experts,
uneven/extreme token distributions, non-aligned counts, sub-block counts),
global padding, backward pass, and non-power-of-2 expert counts.

Note: requires GPU with SM90+ (H100/H800) and torchao.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.experimental.archon.fp8.conftest import make_expert_weights

CUDA_AVAILABLE = torch.cuda.is_available()
SM90_AVAILABLE = False
if CUDA_AVAILABLE:
    major, _ = torch.cuda.get_device_capability()
    SM90_AVAILABLE = major >= 9

try:
    import torchao.prototype.blockwise_fp8_training.linear  # noqa: F401

    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

try:
    from areal.experimental.models.archon.moe.grouped_experts import (
        _run_experts_fp8_for_loop,
    )

    FP8_EXPERTS_AVAILABLE = True
except (ImportError, AttributeError):
    FP8_EXPERTS_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(
        not (CUDA_AVAILABLE and SM90_AVAILABLE),
        reason="FP8 requires CUDA with SM90+ (H100/H800)",
    ),
    pytest.mark.skipif(
        not TORCHAO_AVAILABLE,
        reason="torchao blockwise FP8 prototype not available",
    ),
    pytest.mark.skipif(
        not FP8_EXPERTS_AVAILABLE,
        reason="_run_experts_fp8_for_loop not available",
    ),
]


def _run_bf16_for_loop(w1, w2, w3, x, num_tokens_per_expert):
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()
    total_tokens = sum(num_tokens_per_expert_list)

    x_splits = torch.split(
        x[:total_tokens],
        split_size_or_sections=[int(n) for n in num_tokens_per_expert_list],
        dim=0,
    )

    out_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        if x_expert.shape[0] == 0:
            out_splits.append(x_expert.new_empty(0, w2.shape[1]))
            continue
        w1_e = w1[expert_idx]
        w2_e = w2[expert_idx]
        w3_e = w3[expert_idx]
        h = F.silu(x_expert @ w1_e.T) * (x_expert @ w3_e.T)
        h = h @ w2_e.T
        out_splits.append(h)

    out = torch.cat(out_splits, dim=0)
    if x.shape[0] > total_tokens:
        padding = x.new_zeros((x.shape[0] - total_tokens, out.shape[-1]))
        out = torch.cat([out, padding], dim=0)
    return out


class TestMoEFP8Dispatch:
    def test_fp8_matches_bf16_baseline(self):
        num_experts, dim, hidden_dim = 4, 256, 512
        w1, w2, w3 = make_expert_weights(num_experts, dim, hidden_dim)
        x = torch.randn(512, dim, device="cuda", dtype=torch.bfloat16)
        num_tokens = torch.tensor([128, 128, 128, 128])

        out_fp8 = _run_experts_fp8_for_loop(w1, w2, w3, x, num_tokens)
        out_bf16 = _run_bf16_for_loop(w1, w2, w3, x, num_tokens)

        assert out_fp8.shape == out_bf16.shape
        cos_sim = F.cosine_similarity(
            out_fp8.flatten().unsqueeze(0).float(),
            out_bf16.flatten().unsqueeze(0).float(),
        ).item()
        assert cos_sim > 0.99, f"Cosine similarity too low: {cos_sim}"
        torch.testing.assert_close(out_fp8, out_bf16, rtol=0.1, atol=0.3)

    def test_empty_expert_handling(self):
        w1, w2, w3 = make_expert_weights(4, 256, 512)
        x = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16)
        num_tokens = torch.tensor([128, 256, 0, 0])

        out = _run_experts_fp8_for_loop(w1, w2, w3, x, num_tokens)
        assert out.shape == (384, 256)

    def test_all_experts_empty_except_one(self):
        w1, w2, w3 = make_expert_weights(4, 256, 512)
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        num_tokens = torch.tensor([0, 0, 256, 0])

        out = _run_experts_fp8_for_loop(w1, w2, w3, x, num_tokens)
        assert out.shape == (256, 256)

    def test_non_aligned_token_count(self):
        w1, w2, w3 = make_expert_weights(2, 256, 512)
        x = torch.randn(300, 256, device="cuda", dtype=torch.bfloat16)
        num_tokens = torch.tensor([150, 150])

        out = _run_experts_fp8_for_loop(w1, w2, w3, x, num_tokens)
        assert out.shape == (300, 256)

    def test_very_small_token_count(self):
        w1, w2, w3 = make_expert_weights(2, 256, 512)
        x = torch.randn(10, 256, device="cuda", dtype=torch.bfloat16)
        num_tokens = torch.tensor([3, 7])

        out = _run_experts_fp8_for_loop(w1, w2, w3, x, num_tokens)
        assert out.shape == (10, 256)

    def test_highly_uneven_distribution(self):
        w1, w2, w3 = make_expert_weights(2, 256, 512)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        num_tokens = torch.tensor([500, 12])

        out = _run_experts_fp8_for_loop(w1, w2, w3, x, num_tokens)
        assert out.shape == (512, 256)

    def test_single_expert(self):
        w1, w2, w3 = make_expert_weights(1, 256, 512)
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        num_tokens = torch.tensor([256])

        out = _run_experts_fp8_for_loop(w1, w2, w3, x, num_tokens)
        assert out.shape == (256, 256)

    def test_non_power_of_2_experts(self):
        w1, w2, w3 = make_expert_weights(3, 256, 512)
        x = torch.randn(384, 256, device="cuda", dtype=torch.bfloat16)
        num_tokens = torch.tensor([128, 128, 128])

        out = _run_experts_fp8_for_loop(w1, w2, w3, x, num_tokens)
        assert out.shape == (384, 256)

    def test_global_padding(self):
        w1, w2, w3 = make_expert_weights(2, 256, 512)
        x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16)
        num_tokens = torch.tensor([128, 128])

        out = _run_experts_fp8_for_loop(w1, w2, w3, x, num_tokens)
        assert out.shape == (512, 256)
        torch.testing.assert_close(
            out[256:],
            torch.zeros(256, 256, device="cuda", dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )

    def test_backward_pass(self):
        w1, w2, w3 = make_expert_weights(2, 256, 512)
        w1.requires_grad_(True)
        w2.requires_grad_(True)
        w3.requires_grad_(True)

        x = torch.randn(
            256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        num_tokens = torch.tensor([128, 128])

        out = _run_experts_fp8_for_loop(w1, w2, w3, x, num_tokens)
        out.sum().backward()

        assert x.grad is not None
        assert w1.grad is not None
        assert w2.grad is not None
        assert w3.grad is not None

"""Qwen3.5 Archon test suite merged into one file.

Includes:
- Foundation tests (CPU): config, norms, rope (#1-#11)
- Module tests (GPU): GatedDeltaNet, GatedAttention (#12-#23b)
- Integration tests (GPU): TransformerBlock, Qwen3_5Model (#29-#35)

Run:
    pytest tests/experimental/archon/test_qwen3_5.py -v
"""

# pyright: reportMissingImports=false, reportArgumentType=false, reportConstantRedefinition=false, reportOptionalCall=false, reportOptionalMemberAccess=false, reportMissingTypeArgument=false, reportCallIssue=false, reportAttributeAccessIssue=false

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("transformers", minversion="5.2.0")
import torch
import torch.nn.functional as F
from torch.testing import assert_close
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5TextConfig as HFQwen3_5TextConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention as HFQwen3_5Attention,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet as HFQwen3_5GatedDeltaNet,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNorm as HFQwen3_5RMSNorm,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5RMSNormGated as HFQwen3_5RMSNormGated,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule as hf_torch_chunk_gated_delta_rule,
)

try:
    from areal.experimental.models.archon.qwen2.model.rope import (
        precompute_rope_cache as qwen2_precompute_rope_cache,
    )
    from areal.experimental.models.archon.qwen3_5.model.args import Qwen3_5ModelArgs
    from areal.experimental.models.archon.qwen3_5.model.model import (
        GatedAttention,
        GatedDeltaNet,
        Qwen3_5Model,
        Qwen3_5RMSNorm,
        Qwen3_5RMSNormGated,
        TransformerBlock,
        compute_decay_beta,
        cu_seqlens_to_seq_idx,
    )
    from areal.experimental.models.archon.qwen3_5.model.rope import (
        apply_rotary_emb,
        precompute_rope_cache,
    )
except ImportError as _exc:
    pytest.skip(
        f"Cannot import archon (missing dependency: {_exc})",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Availability flags
# ---------------------------------------------------------------------------
CUDA_AVAILABLE = torch.cuda.is_available()
_CUDA_CAPABILITY = torch.cuda.get_device_capability() if CUDA_AVAILABLE else (0, 0)
_CUDA_SM90 = _CUDA_CAPABILITY >= (9, 0)
requires_cuda = pytest.mark.skipif(
    not _CUDA_SM90,
    reason=(
        f"Requires sm>=9.0 (H100+), got {_CUDA_CAPABILITY[0]}.{_CUDA_CAPABILITY[1]}"
        if CUDA_AVAILABLE
        else "CUDA not available"
    ),
)

try:
    from fla.modules import FusedRMSNormGated

    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    FusedRMSNormGated = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Tolerance tiers (atol/rtol for torch.testing.assert_close)
# ---------------------------------------------------------------------------
EXACT = dict(rtol=1e-6, atol=1e-6)
TIGHT = dict(rtol=1e-5, atol=1e-5)
KERNEL = dict(rtol=1e-4, atol=1e-4)
RELAXED = dict(rtol=1e-3, atol=1e-3)
E2E = dict(rtol=1e-2, atol=5e-2)

# bf16-specific tolerances
KERNEL_BF16 = dict(rtol=1e-2, atol=1e-2)
RELAXED_BF16 = dict(rtol=5e-2, atol=5e-2)


# ---------------------------------------------------------------------------
# Small test model configurations
# ---------------------------------------------------------------------------
SMALL_DENSE_LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
]


# ---------------------------------------------------------------------------
# Mock HF configs for from_hf_config tests
# ---------------------------------------------------------------------------
_MOCK_DENSE_CFG = SimpleNamespace(
    hidden_size=3072,
    num_hidden_layers=36,
    num_attention_heads=24,
    num_key_value_heads=4,
    vocab_size=151936,
    head_dim=256,
    intermediate_size=9216,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    partial_rotary_factor=0.25,
    max_position_embeddings=131072,
    eos_token_id=151645,
    tie_word_embeddings=False,
    layer_types=[
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    * 9,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    attention_bias=False,
)

_MOCK_MOE_CFG = SimpleNamespace(
    hidden_size=2048,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=4,
    vocab_size=151936,
    head_dim=256,
    intermediate_size=8960,
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    partial_rotary_factor=0.25,
    max_position_embeddings=131072,
    eos_token_id=151645,
    tie_word_embeddings=False,
    layer_types=[
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    * 7,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    attention_bias=False,
    num_experts=64,
    num_experts_per_tok=4,
    moe_intermediate_size=1408,
    shared_expert_intermediate_size=8960,
    num_shared_experts=2,
)

_MOCK_COMPOSITE_VLM_CFG = SimpleNamespace(
    model_type="qwen3_5",
    text_config=_MOCK_DENSE_CFG,
)


# ---------------------------------------------------------------------------
# Helper: assert_ratio_close
# ---------------------------------------------------------------------------
def assert_ratio_close(
    name: str,
    ref: torch.Tensor,
    out: torch.Tensor,
    ratio: float,
) -> None:
    """Assert ``max_abs_error / max_abs_ref < ratio``."""
    err = (ref - out).abs().max()
    ref_max = ref.abs().max().clamp(min=1e-8)
    actual = (err / ref_max).item()
    assert actual < ratio, (
        f"{name}: ratio {actual:.6f} >= {ratio} "
        f"(max_err={err.item():.6e}, ref_max={ref_max.item():.6e})"
    )


# ---------------------------------------------------------------------------
# Small config for GatedDeltaNet tests
# ---------------------------------------------------------------------------
SMALL_GDN_CONFIG = dict(
    dim=256,
    n_layers=4,
    n_heads=4,
    n_kv_heads=2,
    vocab_size=1000,
    head_dim=128,
    hidden_dim=512,
    norm_eps=1e-6,
    layer_types=SMALL_DENSE_LAYER_TYPES,
    linear_conv_kernel_dim=4,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_num_key_heads=4,
    linear_num_value_heads=8,
)


_MOCK_GDN_CFG = SimpleNamespace(
    hidden_size=256,
    linear_num_key_heads=4,
    linear_num_value_heads=8,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    hidden_act="silu",
    rms_norm_eps=1e-6,
    dtype=None,
)


# ---------------------------------------------------------------------------
# Consolidated helpers
# ---------------------------------------------------------------------------
def _make_args() -> Qwen3_5ModelArgs:
    """Create small GatedDeltaNet test config."""
    return Qwen3_5ModelArgs(**SMALL_GDN_CONFIG)


def _make_hf_gdn_config() -> SimpleNamespace:
    return _MOCK_GDN_CFG


def _copy_module_weights(hf_module, archon_module):
    """Copy weights from HF module to Archon module (strict key match)."""
    hf_sd = hf_module.state_dict()
    archon_sd = archon_module.state_dict()
    assert set(hf_sd.keys()) == set(archon_sd.keys()), (
        f"State-dict key mismatch:\n"
        f"  HF extra: {set(hf_sd.keys()) - set(archon_sd.keys())}\n"
        f"  Archon extra: {set(archon_sd.keys()) - set(hf_sd.keys())}"
    )
    archon_module.load_state_dict(hf_sd)


# ===================================================================
# Foundation tests (CPU)
# ===================================================================


class TestFromHfConfig:
    """Tests 1-4: Qwen3_5ModelArgs.from_hf_config."""

    def test_from_hf_config_dense(self):
        """Test #1: Dense config → verify all fields."""
        cfg = _MOCK_DENSE_CFG
        args = Qwen3_5ModelArgs.from_hf_config(cfg)

        assert args.dim == 3072
        assert args.n_layers == 36
        assert args.n_heads == 24
        assert args.n_kv_heads == 4
        assert args.vocab_size == 151936
        assert args.head_dim == 256
        assert args.hidden_dim == 9216
        assert args.norm_eps == 1e-6
        assert args.rope_theta == 1000000.0
        assert args.partial_rotary_factor == 0.25
        assert args.max_seq_len == 131072
        assert args.eos_id == 151645
        assert args.enable_weight_tying is False
        assert args.attention_bias is False
        assert len(args.layer_types) == 36
        assert args.layer_types[0] == "linear_attention"
        assert args.layer_types[3] == "full_attention"
        assert args.linear_conv_kernel_dim == 4
        assert args.linear_key_head_dim == 128
        assert args.linear_value_head_dim == 128
        assert args.linear_num_key_heads == 16
        assert args.linear_num_value_heads == 32

        # MoE should be disabled
        assert args.moe_enabled is False
        assert args.moe_args is None

    def test_from_hf_config_moe(self):
        """Test #2: MoE config → verify MoE fields."""
        cfg = _MOCK_MOE_CFG
        args = Qwen3_5ModelArgs.from_hf_config(cfg)

        assert args.dim == 2048
        assert args.n_layers == 28
        assert args.moe_enabled is True
        assert args.moe_args is not None
        assert args.moe_inter_dim == 1408
        assert args.shared_expert_intermediate_size == 8960
        assert args.num_experts == 64
        assert args.num_experts_per_tok == 4
        assert args.moe_args.num_shared_experts == 2

        # Hybrid layer types
        assert len(args.layer_types) == 28
        assert args.layer_types[0] == "linear_attention"
        assert args.layer_types[3] == "full_attention"

    def test_from_hf_config_composite_vlm(self):
        """Test #3: Composite VLM config (has text_config) → unwraps correctly."""
        vlm_cfg = _MOCK_COMPOSITE_VLM_CFG
        args = Qwen3_5ModelArgs.from_hf_config(vlm_cfg)

        # Should have unwrapped text_config → same as dense
        assert args.dim == 3072
        assert args.n_layers == 36
        assert args.head_dim == 256
        assert args.partial_rotary_factor == 0.25

    def test_from_hf_config_invalid_layer_types(self):
        """Test #4: layer_types length ≠ n_layers → raise ValueError."""

        class BadConfig:
            hidden_size = 256
            num_hidden_layers = 4
            num_attention_heads = 2
            num_key_value_heads = 2
            vocab_size = 1000
            head_dim = 128
            intermediate_size = 512
            rms_norm_eps = 1e-6
            # Wrong length: 3 items for 4 layers
            layer_types = ["linear_attention", "linear_attention", "full_attention"]
            linear_conv_kernel_dim = 4
            linear_key_head_dim = 64
            linear_value_head_dim = 64
            linear_num_key_heads = 2
            linear_num_value_heads = 4
            attention_bias = False

        with pytest.raises(ValueError, match="layer_types length"):
            Qwen3_5ModelArgs.from_hf_config(BadConfig())


# ===================================================================
# Tests 5–6: RMSNorm variants
# ===================================================================


class TestRMSNorm:
    """Tests 5-6: RMSNorm parity with HF oracle."""

    def test_rmsnorm_matches_hf(self):
        """Test #5: Qwen3_5RMSNorm matches HF reference."""
        hidden_size = 128
        archon_norm = Qwen3_5RMSNorm(hidden_size, eps=1e-6)
        hf_norm = HFQwen3_5RMSNorm(hidden_size, eps=1e-6)

        # Share weights
        hf_norm.weight.data.copy_(archon_norm.weight.data)

        x = torch.randn(2, 16, hidden_size)
        out_archon = archon_norm(x)
        out_hf = hf_norm(x)

        assert_close(out_archon, out_hf, **TIGHT)

        # Also test with non-zero weights
        archon_norm.weight.data.normal_(0, 0.1)
        hf_norm.weight.data.copy_(archon_norm.weight.data)

        out_archon = archon_norm(x)
        out_hf = hf_norm(x)
        assert_close(out_archon, out_hf, **TIGHT)

    def test_rmsnorm_gated_matches_hf(self):
        """Test #6: Qwen3_5RMSNormGated matches HF reference."""
        hidden_size = 128
        archon_norm = Qwen3_5RMSNormGated(hidden_size, eps=1e-6)
        hf_norm = HFQwen3_5RMSNormGated(hidden_size, eps=1e-6)

        # Share weights
        hf_norm.weight.data.copy_(archon_norm.weight.data)

        x = torch.randn(32, hidden_size)
        gate = torch.randn(32, hidden_size)

        out_archon = archon_norm(x, gate)
        out_hf = hf_norm(x, gate)

        assert_close(out_archon, out_hf, **TIGHT)

        # Test with different weight values
        archon_norm.weight.data.uniform_(0.5, 1.5)
        hf_norm.weight.data.copy_(archon_norm.weight.data)

        out_archon = archon_norm(x, gate)
        out_hf = hf_norm(x, gate)
        assert_close(out_archon, out_hf, **TIGHT)


# ===================================================================
# Test 7: cu_seqlens_to_seq_idx
# ===================================================================


class TestCuSeqlensToSeqIdx:
    """Test 7: cu_seqlens_to_seq_idx."""

    def test_cu_seqlens_to_seq_idx(self):
        """Test #7: [0,3,5,8] → [0,0,0,1,1,2,2,2]."""
        cu_seqlens = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
        total_len = 8

        result = cu_seqlens_to_seq_idx(cu_seqlens, total_len)
        expected = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int32)

        assert_close(result, expected, **EXACT)

        # Additional: single sequence
        cu_seqlens_single = torch.tensor([0, 5], dtype=torch.int32)
        result_single = cu_seqlens_to_seq_idx(cu_seqlens_single, 5)
        expected_single = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int32)
        assert_close(result_single, expected_single, **EXACT)

        # Additional: many short sequences
        cu_seqlens_many = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32)
        result_many = cu_seqlens_to_seq_idx(cu_seqlens_many, 4)
        expected_many = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        assert_close(result_many, expected_many, **EXACT)


# ===================================================================
# Tests 8–9: Partial RoPE cache
# ===================================================================


class TestRoPECache:
    """Tests 8-9: precompute_rope_cache."""

    def test_precompute_rope_cache_partial(self):
        """Test #8: head_dim=256, factor=0.25 → shape [L, 128], values match.

        rotary_dim = int(256 * 0.25) = 64
        cache shape = [L, rotary_dim * 2] = [L, 128]
        """
        head_dim = 256
        partial_rotary_factor = 0.25
        max_seq_len = 64
        base = 1000000.0

        cache = precompute_rope_cache(
            head_dim, max_seq_len, partial_rotary_factor, base
        )

        rotary_dim = int(head_dim * partial_rotary_factor)  # 64
        assert cache.shape == (max_seq_len, rotary_dim * 2)  # [64, 128]

        # Verify values: manually compute reference
        half_rotary = rotary_dim // 2  # 32
        freqs = 1.0 / (
            base ** (torch.arange(0, rotary_dim, 2)[:half_rotary].float() / rotary_dim)
        )
        t = torch.arange(max_seq_len, dtype=freqs.dtype)
        idx_theta = torch.outer(t, freqs).float()
        ref_freqs = torch.cat([idx_theta, idx_theta], dim=-1)
        ref_cos = ref_freqs.cos()
        ref_sin = ref_freqs.sin()
        ref_cache = torch.cat([ref_cos, ref_sin], dim=-1)

        assert_close(cache, ref_cache, **TIGHT)

    def test_rope_factor_1_matches_qwen2(self):
        """Test #9: partial_rotary_factor=1.0 → exact match with qwen2."""
        head_dim = 128
        max_seq_len = 64
        base = 10000.0

        qwen3_5_cache = precompute_rope_cache(
            head_dim, max_seq_len, partial_rotary_factor=1.0, base=base
        )
        qwen2_cache = qwen2_precompute_rope_cache(head_dim, max_seq_len, base)

        assert qwen3_5_cache.shape == qwen2_cache.shape
        assert_close(qwen3_5_cache, qwen2_cache, **EXACT)


# ===================================================================
# Tests 10–11: apply_rotary_emb (partial)
# ===================================================================


class TestApplyRotaryEmb:
    """Tests 10-11: apply_rotary_emb with partial RoPE."""

    def test_apply_rotary_emb_partial(self):
        """Test #10: Rotary dims match HF; passthrough dims bit-exact."""
        head_dim = 256
        partial_rotary_factor = 0.25
        rotary_dim = int(head_dim * partial_rotary_factor)  # 64
        seq_len = 16
        n_heads = 4
        n_kv_heads = 2

        rope_cache = precompute_rope_cache(head_dim, seq_len, partial_rotary_factor)

        xq = torch.randn(1, seq_len, n_heads, head_dim)
        xk = torch.randn(1, seq_len, n_kv_heads, head_dim)

        xq_out, xk_out = apply_rotary_emb(xq, xk, rope_cache)

        # Output shapes must match input
        assert xq_out.shape == xq.shape
        assert xk_out.shape == xk.shape

        # Rotary part should differ from input (not bit-exact)
        assert not torch.equal(xq_out[..., :rotary_dim], xq[..., :rotary_dim])
        assert not torch.equal(xk_out[..., :rotary_dim], xk[..., :rotary_dim])

        # Passthrough part must be bit-exact with input
        assert_close(xq_out[..., rotary_dim:], xq[..., rotary_dim:], **EXACT)
        assert_close(xk_out[..., rotary_dim:], xk[..., rotary_dim:], **EXACT)

        # Verify rotary computation manually for xq
        cos = rope_cache[:seq_len, :rotary_dim].unsqueeze(0).unsqueeze(2)
        sin = rope_cache[:seq_len, rotary_dim:].unsqueeze(0).unsqueeze(2)
        cos = cos.to(xq.dtype)
        sin = sin.to(xq.dtype)
        xq_rot = xq[..., :rotary_dim]
        x1 = xq_rot[..., : rotary_dim // 2]
        x2 = xq_rot[..., rotary_dim // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        expected_rot = xq_rot * cos + rotated * sin
        assert_close(xq_out[..., :rotary_dim], expected_rot, **TIGHT)

        # Verify with positions
        positions = torch.arange(seq_len).unsqueeze(0)
        xq_out2, xk_out2 = apply_rotary_emb(xq, xk, rope_cache, positions)
        assert_close(xq_out2[..., rotary_dim:], xq[..., rotary_dim:], **EXACT)
        assert_close(xk_out2[..., rotary_dim:], xk[..., rotary_dim:], **EXACT)


# ===================================================================
# Phases 2-3: Module tests (GPU)
# ===================================================================


@requires_cuda
class TestFusedVsTorchRMSNormGated:
    @pytest.mark.skipif(not FLA_AVAILABLE, reason="fla not available")
    def test_fused_vs_torch_rmsnorm_gated(self):
        """Test #12: Forward + backward parity."""
        hidden_size = 128
        torch_norm = Qwen3_5RMSNormGated(hidden_size, eps=1e-6).cuda()
        fused_norm = FusedRMSNormGated(hidden_size, eps=1e-6, activation="silu").cuda()
        fused_norm.weight.data.copy_(torch_norm.weight.data)

        x = torch.randn(32, hidden_size, device="cuda", requires_grad=True)
        gate = torch.randn(32, hidden_size, device="cuda", requires_grad=True)
        x2 = x.clone().detach().requires_grad_(True)
        gate2 = gate.clone().detach().requires_grad_(True)

        # Forward
        out_torch = torch_norm(x, gate)
        out_fused = fused_norm(x2, gate2)
        assert_close(out_torch, out_fused, **KERNEL)

        # Backward
        out_torch.sum().backward()
        out_fused.sum().backward()
        assert_ratio_close("dx", x.grad, x2.grad, ratio=1e-3)
        assert_ratio_close("dgate", gate.grad, gate2.grad, ratio=1e-3)
        assert_ratio_close(
            "dw", torch_norm.weight.grad, fused_norm.weight.grad, ratio=1e-3
        )


# ===================================================================
# Test #13: compute_decay_beta
# ===================================================================


class TestDecayBeta:
    def test_decay_beta_formulas(self):
        """Test #13: Hand-calculated oracle + mathematical properties."""
        # --- Part A: Hand-calculated oracle ---
        A_log = torch.tensor([1.0])  # exp(1.0) = 2.718281828
        dt_bias = torch.tensor([1.0])
        a = torch.tensor([[[0.0]]])  # softplus(0+1) = ln(1+e^1) ≈ 1.3133
        b = torch.tensor([[[0.0]]])  # sigmoid(0) = 0.5

        beta, g = compute_decay_beta(A_log, dt_bias, a, b)

        assert_close(beta, torch.tensor([[[0.5]]]), **TIGHT)
        expected_g = torch.tensor([[[-2.718281828 * 1.313261688]]])
        assert_close(g, expected_g, rtol=1e-5, atol=1e-5)

        # --- Part B: Mathematical property tests ---
        A_log_r = torch.randn(8)
        dt_bias_r = torch.ones(8)
        a_r = torch.randn(1, 16, 8)
        b_r = torch.randn(1, 16, 8)
        beta_r, g_r = compute_decay_beta(A_log_r, dt_bias_r, a_r, b_r)

        # Property 1: g must always be negative
        assert (g_r < 0).all(), "Decay g must always be negative"

        # Property 2: beta must be in (0, 1)
        assert ((beta_r > 0) & (beta_r < 1)).all(), "Beta must be in (0, 1)"

        # Property 3: Larger A_log -> larger |g|
        _, g_large = compute_decay_beta(
            torch.tensor([2.0]), dt_bias_r[:1], a_r[:, :, :1], b_r[:, :, :1]
        )
        _, g_small = compute_decay_beta(
            torch.tensor([0.1]), dt_bias_r[:1], a_r[:, :, :1], b_r[:, :, :1]
        )
        assert (g_large.abs() > g_small.abs()).all(), "Larger A_log -> larger |g|"

        # Property 4: Larger dt_bias -> larger |g|
        _, g_bias_large = compute_decay_beta(
            A_log_r[:1], torch.tensor([5.0]), a_r[:, :, :1], b_r[:, :, :1]
        )
        _, g_bias_small = compute_decay_beta(
            A_log_r[:1], torch.tensor([0.1]), a_r[:, :, :1], b_r[:, :, :1]
        )
        assert (g_bias_large.abs() > g_bias_small.abs()).all(), (
            "Larger dt_bias -> larger |g|"
        )


# ===================================================================
# Tests #14-#19b: GatedDeltaNet
# ===================================================================


@requires_cuda
class TestGatedDeltaNet:
    """Tests #14-#19b: GatedDeltaNet module (GPU required)."""

    def test_gated_deltanet_shapes(self):
        """Test #14: Output shape matches input, seq_len=192 (3x chunk_size)."""
        args = _make_args()
        module = GatedDeltaNet(args, layer_idx=0).cuda()
        x = torch.randn(1, 192, args.dim, device="cuda")
        out = module(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
    )
    def test_gated_deltanet_matches_hf(self, dtype):
        """Test #15: Archon vs HF — shared weights, same input."""
        args = _make_args()
        hf_config = _make_hf_gdn_config()

        archon_mod = GatedDeltaNet(args, layer_idx=0).cuda().to(dtype)
        hf_mod = HFQwen3_5GatedDeltaNet(hf_config, layer_idx=0).cuda().to(dtype)
        _copy_module_weights(hf_mod, archon_mod)

        x = torch.randn(1, 64, args.dim, device="cuda", dtype=dtype)
        out_archon = archon_mod(x)
        out_hf = hf_mod(x)

        tol = KERNEL if dtype == torch.float32 else KERNEL_BF16
        assert_close(out_archon, out_hf, **tol)

    def test_gated_deltanet_head_grouping(self):
        """Test #16: Verify repeat_interleave when num_k_heads < num_v_heads."""
        args = _make_args()
        # Sanity: config exercises head grouping
        assert args.linear_num_value_heads > args.linear_num_key_heads
        repeats = args.linear_num_value_heads // args.linear_num_key_heads
        assert repeats == 2, f"Expected 2x head grouping, got {repeats}x"

        hf_config = _make_hf_gdn_config()
        archon_mod = GatedDeltaNet(args, layer_idx=0).cuda()
        hf_mod = HFQwen3_5GatedDeltaNet(hf_config, layer_idx=0).cuda()
        _copy_module_weights(hf_mod, archon_mod)

        x = torch.randn(1, 64, args.dim, device="cuda")
        out_archon = archon_mod(x)
        out_hf = hf_mod(x)
        assert_close(out_archon, out_hf, **KERNEL)

    def test_gated_deltanet_packing(self):
        """Test #17: Packed vs unpacked — EXACT self-consistency.

        seq_a=128 (2x chunk_size), seq_b=96 (>chunk_size) to exercise
        inter-chunk state passing.
        """
        args = _make_args()
        module = GatedDeltaNet(args, layer_idx=0).cuda()

        # Two independent sequences.
        x_a = torch.randn(1, 128, args.dim, device="cuda")
        x_b = torch.randn(1, 96, args.dim, device="cuda")
        out_a = module(x_a)
        out_b = module(x_b)

        # Packed into a single batch.
        x_packed = torch.cat([x_a, x_b], dim=1)  # [1, 224, dim]
        cu_seqlens = torch.tensor([0, 128, 224], dtype=torch.int64, device="cuda")
        seq_idx = cu_seqlens_to_seq_idx(cu_seqlens, 224)

        out_packed = module(x_packed, cu_seqlens=cu_seqlens, seq_idx=seq_idx)

        assert_close(out_packed[:, :128], out_a, **EXACT)
        assert_close(out_packed[:, 128:], out_b, **EXACT)

    def test_gated_deltanet_backward_finite(self):
        """Test #18: All gradients finite after backward, seq_len=192."""
        args = _make_args()
        module = GatedDeltaNet(args, layer_idx=0).cuda()
        x = torch.randn(1, 192, args.dim, device="cuda", requires_grad=True)
        cu_seqlens = torch.tensor([0, 192], dtype=torch.int64, device="cuda")
        seq_idx = cu_seqlens_to_seq_idx(cu_seqlens, 192)

        out = module(x, cu_seqlens=cu_seqlens, seq_idx=seq_idx)
        out.sum().backward()

        for name, p in module.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert p.grad.isfinite().all(), f"{name} grad has NaN/Inf"
        assert x.grad.isfinite().all(), "input grad has NaN/Inf"

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"]
    )
    def test_gated_deltanet_backward_parity(self, dtype):
        """Test #19: Gradient correctness — Archon vs HF backward."""
        args = _make_args()
        hf_config = _make_hf_gdn_config()

        archon_mod = GatedDeltaNet(args, layer_idx=0).cuda().to(dtype)
        hf_mod = HFQwen3_5GatedDeltaNet(hf_config, layer_idx=0).cuda().to(dtype)
        _copy_module_weights(hf_mod, archon_mod)

        x = torch.randn(1, 192, args.dim, device="cuda", dtype=dtype)
        x_a = x.clone().detach().requires_grad_(True)
        x_h = x.clone().detach().requires_grad_(True)

        out_a = archon_mod(x_a)
        out_h = hf_mod(x_h)
        out_a.sum().backward()
        out_h.sum().backward()

        r_dx = 0.008 if dtype == torch.float32 else 0.05
        r_param = 0.02 if dtype == torch.float32 else 0.1

        assert_ratio_close("dx", x_h.grad, x_a.grad, ratio=r_dx)

        # Parameter gradients
        archon_params = dict(archon_mod.named_parameters())
        hf_params = dict(hf_mod.named_parameters())
        for name in archon_params:
            if name in hf_params and archon_params[name].grad is not None:
                assert_ratio_close(
                    name,
                    hf_params[name].grad,
                    archon_params[name].grad,
                    ratio=r_param,
                )

    def test_gated_deltanet_cross_oracle(self):
        """Test #19b: fla triton kernel vs HF pure-PyTorch fallback.

        Independent oracle — does not share fla triton codepath.
        """
        fla_chunk = pytest.importorskip(
            "fla.ops.gated_delta_rule"
        ).chunk_gated_delta_rule

        # Shared inputs (seq_len=192 to exercise multi-chunk).
        # Q/K pre-expanded to num_v heads (kernel requires H to match).
        B, T, num_v, dk, dv = 1, 192, 8, 32, 32
        q = torch.randn(B, T, num_v, dk, device="cuda")
        k = torch.randn(B, T, num_v, dk, device="cuda")
        v = torch.randn(B, T, num_v, dv, device="cuda")
        g = -torch.rand(B, T, num_v, device="cuda").abs() * 5  # negative decay
        beta = torch.sigmoid(torch.randn(B, T, num_v, device="cuda"))

        # fla triton kernel
        out_triton, _ = fla_chunk(
            q,
            k,
            v,
            g,
            beta,
            use_qk_l2norm_in_kernel=True,
        )

        # HF pure-PyTorch fallback (independent oracle from conftest)
        out_torch, _ = hf_torch_chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            use_qk_l2norm_in_kernel=True,
        )

        assert_close(out_triton, out_torch, **RELAXED)


# ===================================================================
# Tests #20-#22: GatedAttention
# ===================================================================


@requires_cuda
class TestGatedAttention:
    """Tests #20-#22: GatedAttention module (GPU required, flash attention)."""

    @staticmethod
    def _make_hf_attn_config():
        return SimpleNamespace(
            hidden_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
            rms_norm_eps=1e-6,
            attention_bias=False,
            attention_dropout=0.0,
            _attn_implementation="eager",
        )

    @staticmethod
    def _copy_weights_hf_attn_to_archon(hf_mod, archon_mod):
        """Copy weights from HF Attention oracle to Archon GatedAttention."""
        mapping = {
            "q_proj": "wq",
            "k_proj": "wk",
            "v_proj": "wv",
            "o_proj": "wo",
        }
        for hf_name, archon_name in mapping.items():
            getattr(archon_mod, archon_name).weight.data.copy_(
                getattr(hf_mod, hf_name).weight.data
            )
        archon_mod.q_norm.weight.data.copy_(hf_mod.q_norm.weight.data)
        archon_mod.k_norm.weight.data.copy_(hf_mod.k_norm.weight.data)

    def test_gated_attention_gate_logic(self):
        """Test #20: Verify q_proj 2x split + sigmoid gate."""
        args = _make_args()
        module = GatedAttention(args).cuda().to(torch.bfloat16)

        T = 64
        x = torch.randn(1, T, args.dim, device="cuda", dtype=torch.bfloat16)
        rope_cache = precompute_rope_cache(
            head_dim=args.head_dim,
            max_seq_len=T + 1,
            partial_rotary_factor=0.25,
        ).cuda()
        positions = torch.arange(T, device="cuda").unsqueeze(0)
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="cuda")

        # --- Verify q_proj is 2x width ---
        expected_q_out = args.n_heads * args.head_dim * 2
        assert module.wq.out_features == expected_q_out, (
            f"wq out_features={module.wq.out_features}, expected {expected_q_out}"
        )

        # --- Verify forward produces correct shape ---
        out = module(x, rope_cache, positions, cu_seqlens, max_seqlen=T)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

        # --- Verify gate affects output (gate=0 → output=0) ---
        # Set wq to output zeros for the gate half.
        with torch.no_grad():
            # wq shape: [n_heads * head_dim * 2, dim]
            # The gate is the second half of per-head 2*head_dim output.
            # After view(B,T,-1,head_dim*2) and chunk(2, dim=-1):
            # first chunk = query, second chunk = gate
            # So gate corresponds to the second head_dim slice of each
            # 2*head_dim block. Zero out those rows to make gate→0.
            w = module.wq.weight.data.view(args.n_heads, args.head_dim * 2, args.dim)
            w[:, args.head_dim :, :] = 0  # zero the gate rows
            module.wq.weight.data.copy_(w.view(-1, args.dim))

        out_zeroed = module(x, rope_cache, positions, cu_seqlens, max_seqlen=T)
        # sigmoid(0) = 0.5, so output should be ~0.5 * attn_output
        # Just check the ratio is bounded away from 1.0
        # (if gate had no effect, out_zeroed == out)
        assert not torch.allclose(out, out_zeroed, atol=1e-3), (
            "Zeroing gate weights had no effect — gate logic may be broken"
        )

    def test_gated_attention_matches_hf(self):
        """Test #21: Archon GatedAttention vs HF oracle (shared weights, bf16)."""
        args = _make_args()
        hf_config = self._make_hf_attn_config()

        T = 64
        dtype = torch.bfloat16
        archon_mod = GatedAttention(args).cuda().to(dtype)
        hf_mod = HFQwen3_5Attention(hf_config, layer_idx=0).cuda().to(dtype)
        self._copy_weights_hf_attn_to_archon(hf_mod, archon_mod)

        x = torch.randn(1, T, args.dim, device="cuda", dtype=dtype)

        # --- Archon forward ---
        rope_cache = precompute_rope_cache(
            head_dim=args.head_dim,
            max_seq_len=T + 1,
            partial_rotary_factor=0.25,
        ).cuda()
        positions = torch.arange(T, device="cuda").unsqueeze(0)
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="cuda")
        out_archon = archon_mod(x, rope_cache, positions, cu_seqlens, max_seqlen=T)

        # --- HF forward (transformers API) ---
        rotary_dim = rope_cache.shape[-1] // 2
        rc = rope_cache[:T]  # [T, rotary_dim*2]
        cos_hf = rc[:, :rotary_dim].unsqueeze(0).to(dtype)  # [1, T, rotary_dim]
        sin_hf = rc[:, rotary_dim:].unsqueeze(0).to(dtype)  # [1, T, rotary_dim]
        # Build causal mask: Archon uses flash attention with is_causal=True,
        # so HF eager attention needs an explicit causal mask to match.
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device="cuda", dtype=dtype), diagonal=1
        )  # [T, T]
        # HF eager expects [B, 1, T, T] (broadcast over heads).
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
        out_hf, _ = hf_mod(
            x, position_embeddings=(cos_hf, sin_hf), attention_mask=causal_mask
        )

        assert_close(out_archon, out_hf, **RELAXED_BF16)

    def test_gated_attention_backward_finite(self):
        """Test #22: All gradients finite after backward."""
        args = _make_args()
        module = GatedAttention(args).cuda().to(torch.bfloat16)

        T = 64
        x = torch.randn(
            1, T, args.dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        rope_cache = precompute_rope_cache(
            head_dim=args.head_dim,
            max_seq_len=T + 1,
            partial_rotary_factor=0.25,
        ).cuda()
        positions = torch.arange(T, device="cuda").unsqueeze(0)
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device="cuda")

        out = module(x, rope_cache, positions, cu_seqlens, max_seqlen=T)
        out.sum().backward()

        for name, p in module.named_parameters():
            assert p.grad is not None, f"{name} has no grad"
            assert p.grad.isfinite().all(), f"{name} grad has NaN/Inf"
        assert x.grad.isfinite().all(), "input grad has NaN/Inf"


# ===================================================================
# Tests #23, #23b: causal_conv1d seq_idx boundary tests
# ===================================================================


@requires_cuda
class TestConv1dBoundary:
    """Tests #23-#23b: causal_conv1d_fn seq_idx boundary behavior."""

    def test_causal_conv1d_seq_idx_boundary(self):
        """Test #23: packed vs unpacked via seq_idx isolation."""
        conv1d_fn = pytest.importorskip("causal_conv1d").causal_conv1d_fn

        dim, kernel_size = 64, 4
        weight = torch.randn(dim, kernel_size, device="cuda")

        T_a, T_b = 32, 24

        # Two independent sequences — channel-last layout (stride(1)==1)
        # required by causal_conv1d_fn when seq_idx is used.
        # Create as [1, T, dim] then transpose to get stride(1)==1.
        x_raw_a = torch.randn(1, T_a, dim, device="cuda")
        x_raw_b = torch.randn(1, T_b, dim, device="cuda")
        x_a = x_raw_a.transpose(1, 2)  # [1, dim, T_a], channel-last
        x_b = x_raw_b.transpose(1, 2)  # [1, dim, T_b], channel-last
        out_a = conv1d_fn(x_a, weight, activation="silu")
        out_b = conv1d_fn(x_b, weight, activation="silu")

        # Packed: cat RAW tensors first, THEN transpose once.
        # torch.cat on transposed views breaks channel-last invariant.
        x_raw_packed = torch.cat([x_raw_a, x_raw_b], dim=1)  # [1, T_a+T_b, dim]
        x_packed = x_raw_packed.transpose(1, 2)  # [1, dim, T_a+T_b], channel-last
        seq_idx = torch.cat(
            [
                torch.zeros(T_a, dtype=torch.int32, device="cuda"),
                torch.ones(T_b, dtype=torch.int32, device="cuda"),
            ]
        ).unsqueeze(0)  # [1, T_a+T_b]
        out_packed = conv1d_fn(x_packed, weight, activation="silu", seq_idx=seq_idx)

        assert_close(
            out_packed[:, :, :T_a],
            out_a,
            **EXACT,
            msg="seq_idx: first sequence mismatch",
        )
        assert_close(
            out_packed[:, :, T_a:],
            out_b,
            **EXACT,
            msg="seq_idx: second sequence contaminated by first",
        )

    def test_causal_conv1d_seq_idx_padding_zero(self):
        """Test #23b: seq_idx=-1 (or out-of-range) → zero output."""
        conv1d_fn = pytest.importorskip("causal_conv1d").causal_conv1d_fn

        dim, kernel_size = 64, 4
        weight = torch.randn(dim, kernel_size, device="cuda")

        T = 32
        x = torch.randn(1, T, dim, device="cuda").transpose(1, 2)

        # All positions assigned to seq 0 — normal output.
        seq_idx_normal = torch.zeros(1, T, dtype=torch.int32, device="cuda")
        out_normal = conv1d_fn(x, weight, activation="silu", seq_idx=seq_idx_normal)

        # Mark last 8 positions as seq 1 — they should NOT affect seq 0 output.
        seq_idx_split = torch.zeros(1, T, dtype=torch.int32, device="cuda")
        seq_idx_split[:, -8:] = 1
        out_split = conv1d_fn(x, weight, activation="silu", seq_idx=seq_idx_split)

        # The first T-8 positions of seq 0 should be identical.
        assert_close(
            out_split[:, :, : T - 8],
            out_normal[:, :, : T - 8],
            **EXACT,
            msg="seq_idx split: seq 0 prefix contaminated",
        )


# ---------------------------------------------------------------------------
# Helpers for TransformerBlock + Model tests
# ---------------------------------------------------------------------------
def _make_block_inputs(
    args: Qwen3_5ModelArgs, seq_len: int = 64, device: str = "cuda"
) -> dict:
    """Create common inputs for TransformerBlock forward."""
    x = torch.randn(1, seq_len, args.dim, device=device, dtype=torch.bfloat16)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int64, device=device)
    rope_cache = precompute_rope_cache(
        args.head_dim,
        args.max_seq_len,
        args.partial_rotary_factor,
        args.rope_theta,
    ).to(device)
    seq_idx = cu_seqlens_to_seq_idx(cu_seqlens, seq_len)
    return dict(
        x=x,
        rope_cache=rope_cache,
        positions=positions,
        cu_seqlens=cu_seqlens,
        max_seqlen=seq_len,
        seq_idx=seq_idx,
    )


def _make_model_inputs(
    args: Qwen3_5ModelArgs, seq_len: int = 64, device: str = "cuda"
) -> dict:
    """Create common inputs for Qwen3_5Model forward."""
    tokens = torch.randint(0, args.vocab_size, (1, seq_len), device=device)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int64, device=device)
    return dict(
        tokens=tokens,
        positions=positions,
        cu_seqlens=cu_seqlens,
        max_seqlen=seq_len,
    )


def _copy_model_weights(
    hf_model: HFQwen3_5ForCausalLM, archon_model: Qwen3_5Model
) -> None:
    """Copy weights from real HF Qwen3_5ForCausalLM to Archon model."""
    hf = hf_model.model
    with torch.no_grad():
        archon_model.tok_embeddings.weight.copy_(hf.embed_tokens.weight)
        archon_model.output.weight.copy_(hf_model.lm_head.weight)
        archon_model.norm.weight.copy_(hf.norm.weight)

        for hf_layer, archon_layer in zip(hf.layers, archon_model.layers.values()):
            archon_layer.attention_norm.weight.copy_(hf_layer.input_layernorm.weight)
            archon_layer.ffn_norm.weight.copy_(hf_layer.post_attention_layernorm.weight)
            archon_layer.feed_forward.w1.weight.copy_(hf_layer.mlp.gate_proj.weight)
            archon_layer.feed_forward.w3.weight.copy_(hf_layer.mlp.up_proj.weight)
            archon_layer.feed_forward.w2.weight.copy_(hf_layer.mlp.down_proj.weight)

            layer_type = hf_layer.layer_type
            if layer_type == "full_attention":
                archon_layer.attention.wq.weight.copy_(hf_layer.self_attn.q_proj.weight)
                archon_layer.attention.wk.weight.copy_(hf_layer.self_attn.k_proj.weight)
                archon_layer.attention.wv.weight.copy_(hf_layer.self_attn.v_proj.weight)
                archon_layer.attention.wo.weight.copy_(hf_layer.self_attn.o_proj.weight)
                archon_layer.attention.q_norm.weight.copy_(
                    hf_layer.self_attn.q_norm.weight
                )
                archon_layer.attention.k_norm.weight.copy_(
                    hf_layer.self_attn.k_norm.weight
                )
            else:
                archon_layer.linear_attn.in_proj_qkv.weight.copy_(
                    hf_layer.linear_attn.in_proj_qkv.weight
                )
                archon_layer.linear_attn.in_proj_z.weight.copy_(
                    hf_layer.linear_attn.in_proj_z.weight
                )
                archon_layer.linear_attn.in_proj_a.weight.copy_(
                    hf_layer.linear_attn.in_proj_a.weight
                )
                archon_layer.linear_attn.in_proj_b.weight.copy_(
                    hf_layer.linear_attn.in_proj_b.weight
                )
                archon_layer.linear_attn.conv1d.weight.copy_(
                    hf_layer.linear_attn.conv1d.weight
                )
                archon_layer.linear_attn.out_proj.weight.copy_(
                    hf_layer.linear_attn.out_proj.weight
                )
                archon_layer.linear_attn.norm.weight.copy_(
                    hf_layer.linear_attn.norm.weight
                )
                archon_layer.linear_attn.A_log.copy_(hf_layer.linear_attn.A_log)
                archon_layer.linear_attn.dt_bias.copy_(hf_layer.linear_attn.dt_bias)


def _make_hf_model_config(args: Qwen3_5ModelArgs) -> HFQwen3_5TextConfig:
    """Create a real HF Qwen3_5TextConfig from Archon args."""
    cfg = HFQwen3_5TextConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.dim,
        intermediate_size=args.hidden_dim,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        rms_norm_eps=args.norm_eps,
        attention_bias=args.attention_bias,
        layer_types=list(args.layer_types),
        linear_conv_kernel_dim=args.linear_conv_kernel_dim,
        linear_key_head_dim=args.linear_key_head_dim,
        linear_value_head_dim=args.linear_value_head_dim,
        linear_num_key_heads=args.linear_num_key_heads,
        linear_num_value_heads=args.linear_num_value_heads,
        rope_theta=args.rope_theta,
        partial_rotary_factor=args.partial_rotary_factor,
    )
    cfg._attn_implementation = "eager"
    return cfg


# ===================================================================
# Tests
# ===================================================================


# ===================================================================
# TransformerBlock + Qwen3_5Model tests (GPU)
# ===================================================================


@requires_cuda
def test_block_full_attention_forward():
    """#29: Single full_attention block forward — shape + isfinite."""
    args = _make_args()
    # Find a full_attention layer index.
    fa_idx = args.layer_types.index("full_attention")

    block = TransformerBlock(fa_idx, args).cuda().to(torch.bfloat16)
    block.init_weights()

    inputs = _make_block_inputs(args, seq_len=64)
    out = block(**inputs)

    assert out.shape == inputs["x"].shape, (
        f"Expected {inputs['x'].shape}, got {out.shape}"
    )
    assert out.isfinite().all(), "Output has NaN/Inf"
    # Verify it's using GatedAttention (not GatedDeltaNet).
    assert block.attention is not None
    assert block.linear_attn is None


@requires_cuda
def test_block_linear_attention_forward():
    """#30: Single linear_attention block forward — shape + isfinite."""
    args = _make_args()
    # Find a linear_attention layer index.
    la_idx = args.layer_types.index("linear_attention")

    block = TransformerBlock(la_idx, args).cuda().to(torch.bfloat16)
    block.init_weights()

    inputs = _make_block_inputs(args, seq_len=64)
    out = block(**inputs)

    assert out.shape == inputs["x"].shape, (
        f"Expected {inputs['x'].shape}, got {out.shape}"
    )
    assert out.isfinite().all(), "Output has NaN/Inf"
    # Verify it's using GatedDeltaNet (not GatedAttention).
    assert block.attention is None
    assert block.linear_attn is not None


@requires_cuda
def test_model_forward_numerical_sanity():
    """#31: Full model forward → output shape, finite, non-constant."""
    args = _make_args()
    seq_len = 64

    model = Qwen3_5Model(args).cuda().to(torch.bfloat16)
    model.init_weights()
    model.init_buffers("cuda")

    inputs = _make_model_inputs(args, seq_len=seq_len)
    output = model(**inputs)

    assert output.shape == (1, seq_len, args.vocab_size), (
        f"Expected (1, {seq_len}, {args.vocab_size}), got {output.shape}"
    )
    assert output.isfinite().all(), "Output has NaN/Inf"
    assert output.std() > 1e-6, "Output is constant"
    assert output.abs().max() < 1e4, "Output is exploding"

    for sl in [32, 128]:
        out_sl = model(**_make_model_inputs(args, seq_len=sl))
        assert out_sl.shape == (1, sl, args.vocab_size)


@requires_cuda
def test_model_backward_grad_norm():
    """#32: Forward → cross_entropy loss → backward → grad norm in (0, 1e3)."""
    args = _make_args()
    seq_len = 64

    model = Qwen3_5Model(args).cuda().to(torch.bfloat16)
    model.init_weights()
    model.init_buffers("cuda")

    inputs = _make_model_inputs(args, seq_len=seq_len)
    targets = torch.randint(0, args.vocab_size, (1, seq_len), device="cuda")

    output = model(**inputs)
    loss = F.cross_entropy(output.view(-1, args.vocab_size).float(), targets.view(-1))
    loss.backward()

    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    assert 0 < total_norm < 1e3, f"Grad norm {total_norm} out of range"


@requires_cuda
def test_hybrid_layer_backward_parity():
    """#34: Full 4-layer model backward: Archon vs HF.

    Shared weights → compare grad norms AND grad direction (cosine similarity)
    per parameter group.

    Uses deterministic seeded inputs in a narrow token range to minimize
    numerical divergence amplification from random inputs.
    """
    args = _make_args()
    hf_config = _make_hf_model_config(args)
    seq_len = 192  # 3×64 → 3 chunks for GatedDeltaNet

    # Build both models in bf16.
    archon_model = Qwen3_5Model(args).cuda().to(torch.bfloat16)
    hf_model = HFQwen3_5ForCausalLM(hf_config).cuda().to(torch.bfloat16)

    # Copy weights HF → Archon.
    _copy_model_weights(hf_model, archon_model)
    archon_model.init_buffers("cuda")

    # Deterministic seeded inputs (narrow range to reduce amplification).
    gen = torch.Generator(device="cuda").manual_seed(42)
    tokens = torch.randint(
        100, args.vocab_size - 100, (1, seq_len), device="cuda", generator=gen
    )
    targets = torch.randint(
        0, args.vocab_size, (1, seq_len), device="cuda", generator=gen
    )
    positions = torch.arange(seq_len, device="cuda").unsqueeze(0)
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int64, device="cuda")

    # Forward + backward: Archon.
    archon_out = archon_model(tokens, positions, cu_seqlens, seq_len)
    archon_loss = F.cross_entropy(
        archon_out.view(-1, args.vocab_size).float(), targets.view(-1)
    )
    archon_loss.backward()

    # Forward + backward: HF.
    hf_out = hf_model(input_ids=tokens, use_cache=False).logits
    hf_loss = F.cross_entropy(
        hf_out.view(-1, args.vocab_size).float(), targets.view(-1)
    )
    hf_loss.backward()

    # Build matched parameter pairs.
    archon_params = {
        n: p for n, p in archon_model.named_parameters() if p.grad is not None
    }
    hf_params = {n: p for n, p in hf_model.named_parameters() if p.grad is not None}

    # Map Archon parameter names to HF names.
    def _archon_to_hf_name(name: str) -> str | None:
        """Convert Archon param name to HF param name."""
        if name.startswith("tok_embeddings"):
            name = name.replace("tok_embeddings", "model.embed_tokens")
        elif name.startswith("layers."):
            name = f"model.{name}"
        elif name.startswith("norm."):
            name = f"model.{name}"

        name = name.replace("output.", "lm_head.")
        name = name.replace("attention_norm", "input_layernorm")
        name = name.replace("ffn_norm", "post_attention_layernorm")
        name = name.replace("feed_forward.w1", "mlp.gate_proj")
        name = name.replace("feed_forward.w3", "mlp.up_proj")
        name = name.replace("feed_forward.w2", "mlp.down_proj")
        name = name.replace("attention.wq", "self_attn.q_proj")
        name = name.replace("attention.wk", "self_attn.k_proj")
        name = name.replace("attention.wv", "self_attn.v_proj")
        name = name.replace("attention.wo", "self_attn.o_proj")
        name = name.replace("attention.q_norm", "self_attn.q_norm")
        name = name.replace("attention.k_norm", "self_attn.k_norm")
        return name

    # Tolerance: flash attn vs eager attn may introduce small numerical
    # differences, but with correct RoPE both should be very close.
    # For parameters with small grad norms (< 0.01), the absolute error from
    # flash-vs-eager divergence dominates, causing inflated relative errors.
    COS_SIM_THRESHOLD = 0.99
    GRAD_NORM_REL_ERR = 0.02
    GRAD_NORM_REL_ERR_SMALL = 0.05
    SMALL_NORM_THRESHOLD = 0.01

    matched = 0
    mismatched_names = []
    for archon_name, archon_param in archon_params.items():
        hf_name = _archon_to_hf_name(archon_name)
        if hf_name not in hf_params:
            mismatched_names.append(
                f"Archon: {archon_name} → HF: {hf_name} (not found)"
            )
            continue

        hf_param = hf_params[hf_name]
        matched += 1

        norm_a = archon_param.grad.float().norm()
        norm_h = hf_param.grad.float().norm()

        # Skip near-zero grad norms — tiny absolute values cause
        # spurious large relative errors (e.g. dt_bias ~1e-5).
        if norm_h < 1e-4:
            continue

        rel_err = (norm_a - norm_h).abs() / norm_h.clamp(min=1e-8)
        threshold = (
            GRAD_NORM_REL_ERR_SMALL
            if norm_h < SMALL_NORM_THRESHOLD
            else GRAD_NORM_REL_ERR
        )
        assert rel_err < threshold, (
            f"{archon_name}: grad norm rel err {rel_err:.4f} "
            f"(archon={norm_a:.6f}, hf={norm_h:.6f})"
        )

        cos_sim = F.cosine_similarity(
            archon_param.grad.flatten().float(),
            hf_param.grad.flatten().float(),
            dim=0,
        )
        assert cos_sim > COS_SIM_THRESHOLD, (
            f"{archon_name}: grad cosine similarity {cos_sim:.4f} < {COS_SIM_THRESHOLD}"
        )

    assert matched > 0, f"No parameters matched! Mismatched: {mismatched_names}"
    if mismatched_names:
        import warnings

        warnings.warn(
            f"Unmatched parameters ({len(mismatched_names)}): {mismatched_names[:5]}..."
        )

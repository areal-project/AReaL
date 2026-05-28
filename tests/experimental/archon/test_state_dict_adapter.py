"""Tests for Archon State Dict Adapter roundtrip conversion.

These tests verify key conversion between HuggingFace and Archon formats
by testing full roundtrip conversions (HF -> Archon -> HF).

Run tests:
    pytest tests/experimental/archon/test_state_dict_adapter.py -v

Note: For E2E weight sync tests with real models, see test_weight_sync.py
and test_checkpoint_e2e.py.
"""

import pytest
import torch
from torch.testing import assert_close

from areal.experimental.models.archon.qwen3 import Qwen3StateDictAdapter

try:
    from areal.experimental.models.archon.qwen3_5.model.state_dict_adapter import (
        Qwen3_5StateDictAdapter,
    )

    _HAS_QWEN3_5_ADAPTER = True
except ImportError:
    _HAS_QWEN3_5_ADAPTER = False

# =============================================================================
# Mock Configs
# =============================================================================


class MockQwen3Config:
    """Mock Qwen3 model config for testing."""

    model_type = "qwen2"
    num_local_experts = 0  # Dense model
    tie_word_embeddings = False


class MockQwen3MoEConfig:
    """Mock Qwen3 MoE model config for testing."""

    model_type = "qwen3_moe"
    num_local_experts = 8
    tie_word_embeddings = False


# =============================================================================
# Dense Model Roundtrip Tests
# =============================================================================


class TestQwen3StateDictAdapterDense:
    """Roundtrip tests for Qwen3StateDictAdapter with dense models."""

    @pytest.fixture
    def adapter(self):
        return Qwen3StateDictAdapter(MockQwen3Config())

    def test_roundtrip_dense_state_dict(self, adapter):
        """Test full roundtrip conversion for dense model."""
        # Create a minimal HF state dict
        hf_state = {
            "model.embed_tokens.weight": torch.randn(32000, 4096),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(1024, 4096),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(1024, 4096),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(4096, 4096),
            "model.layers.0.self_attn.q_norm.weight": torch.randn(128),
            "model.layers.0.self_attn.k_norm.weight": torch.randn(128),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(11008, 4096),
            "model.layers.0.mlp.up_proj.weight": torch.randn(11008, 4096),
            "model.layers.0.mlp.down_proj.weight": torch.randn(4096, 11008),
            "model.layers.0.input_layernorm.weight": torch.randn(4096),
            "model.layers.0.post_attention_layernorm.weight": torch.randn(4096),
            "model.norm.weight": torch.randn(4096),
            "lm_head.weight": torch.randn(32000, 4096),
        }

        # HF -> Archon
        archon_state = adapter.from_hf(hf_state)

        # Verify Archon keys
        expected_archon_keys = {
            "tok_embeddings.weight",
            "layers.0.attention.wq.weight",
            "layers.0.attention.wk.weight",
            "layers.0.attention.wv.weight",
            "layers.0.attention.wo.weight",
            "layers.0.attention.q_norm.weight",
            "layers.0.attention.k_norm.weight",
            "layers.0.feed_forward.w1.weight",
            "layers.0.feed_forward.w3.weight",
            "layers.0.feed_forward.w2.weight",
            "layers.0.attention_norm.weight",
            "layers.0.ffn_norm.weight",
            "norm.weight",
            "output.weight",
        }
        assert set(archon_state.keys()) == expected_archon_keys

        # Archon -> HF
        roundtrip_state = adapter.to_hf(archon_state)

        # Verify roundtrip
        for key in hf_state:
            if "rotary_emb" in key:
                continue
            assert key in roundtrip_state, f"Missing key: {key}"
            assert torch.allclose(hf_state[key], roundtrip_state[key]), (
                f"Mismatch at {key}"
            )


# =============================================================================
# MoE Model Roundtrip Tests
# =============================================================================


class TestQwen3StateDictAdapterMoE:
    """Roundtrip tests for Qwen3StateDictAdapter with MoE models."""

    @pytest.fixture
    def adapter(self):
        return Qwen3StateDictAdapter(MockQwen3MoEConfig())

    def test_moe_roundtrip(self, adapter):
        """Test roundtrip conversion for MoE experts."""
        # Create 3D Archon weight
        archon_state = {
            "layers.0.moe.experts.w1": torch.randn(8, 11008, 4096),
            "layers.0.moe.experts.w2": torch.randn(8, 4096, 11008),
            "layers.0.moe.experts.w3": torch.randn(8, 11008, 4096),
        }

        # Archon -> HF
        hf_state = adapter.to_hf(archon_state)

        # Should have 24 keys (8 experts x 3 projections)
        assert len(hf_state) == 24

        # HF -> Archon
        roundtrip_state = adapter.from_hf(hf_state)

        # Verify roundtrip
        for key in archon_state:
            assert key in roundtrip_state
            assert torch.allclose(archon_state[key], roundtrip_state[key])

    def test_full_moe_state_dict_roundtrip(self, adapter):
        """Test roundtrip for complete MoE layer state dict including router."""
        # Create full MoE layer state dict
        archon_state = {
            # Expert weights
            "layers.0.moe.experts.w1": torch.randn(8, 1024, 512),
            "layers.0.moe.experts.w2": torch.randn(8, 512, 1024),
            "layers.0.moe.experts.w3": torch.randn(8, 1024, 512),
            # Router
            "layers.0.moe.router.gate.weight": torch.randn(8, 512),
        }

        # Archon -> HF
        hf_state = adapter.to_hf(archon_state)

        # Should have 24 expert keys + 1 router key
        assert len(hf_state) == 25
        assert "model.layers.0.mlp.gate.weight" in hf_state

        # HF -> Archon
        roundtrip_state = adapter.from_hf(hf_state)

        # Verify roundtrip for experts
        for key in [
            "layers.0.moe.experts.w1",
            "layers.0.moe.experts.w2",
            "layers.0.moe.experts.w3",
        ]:
            assert key in roundtrip_state
            assert torch.allclose(archon_state[key], roundtrip_state[key])

        # Verify roundtrip for router
        router_key = "layers.0.moe.router.gate.weight"
        assert router_key in roundtrip_state
        assert torch.allclose(archon_state[router_key], roundtrip_state[router_key])


# =============================================================================
# MoE Model Forward Match Test with Adapter Roundtrip
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMoEWeightLoadingRoundtrip:
    """Test MoE weight loading roundtrip with forward pass verification."""

    def test_moe_weight_roundtrip_forward_match(self):
        """Test MoE forward output matches after Archon->HF->Archon roundtrip.

        This test verifies that the state dict adapter correctly converts
        MoE weights between Archon and HuggingFace formats by:
        1. Creating a model and running forward pass
        2. Converting state dict: Archon -> HF -> Archon via adapter
        3. Loading converted weights into a new model
        4. Verifying forward outputs match
        """
        from areal.experimental.models.archon.moe import MoEArgs
        from areal.experimental.models.archon.qwen3 import Qwen3Model, Qwen3ModelArgs

        # Create MoE model args
        model_args = Qwen3ModelArgs(
            dim=64,
            hidden_dim=128,
            moe_inter_dim=96,
            n_heads=4,
            n_kv_heads=2,
            n_layers=2,
            head_dim=16,
            vocab_size=1000,
            max_seq_len=32,
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=4, top_k=2, use_grouped_mm=False),
            decoder_sparse_step=1,  # All layers are MoE
            attn_type="sdpa",
        )

        # Create and initialize first model
        model1 = Qwen3Model(model_args).cuda()
        model1.init_weights()
        model1.init_buffers(buffer_device=torch.device("cuda"))

        # Get original Archon state dict
        archon_state = {k: v.clone() for k, v in model1.state_dict().items()}

        # Create adapter config that matches the model's num_experts
        class AdapterConfig:
            model_type = "qwen3_moe"
            num_local_experts = 4  # Must match MoEArgs(num_experts=4)
            tie_word_embeddings = False

        # Create adapter and perform roundtrip conversion
        adapter = Qwen3StateDictAdapter(AdapterConfig())

        # Archon -> HF -> Archon
        hf_state = adapter.to_hf(archon_state)
        roundtrip_state = adapter.from_hf(hf_state)

        # Create second model and load roundtrip weights
        # Note: strict=False because expert_bias is an Archon-only buffer
        model2 = Qwen3Model(model_args).cuda()
        model2.init_buffers(buffer_device=torch.device("cuda"))
        model2.load_state_dict(roundtrip_state, strict=False)

        # Create test input (packed sequence format)
        torch.manual_seed(42)
        tokens = torch.randint(0, 1000, (1, 16), device="cuda")
        positions = torch.arange(16, device="cuda").unsqueeze(0)
        cu_seqlens = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
        max_seqlen = 16

        # Forward pass on both models
        with torch.no_grad():
            output1 = model1(
                tokens, positions, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
            output2 = model2(
                tokens, positions, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )

        # Verify outputs match
        assert torch.allclose(output1, output2, atol=1e-5, rtol=1e-5), (
            f"Forward output mismatch after adapter roundtrip. "
            f"Max diff: {(output1 - output2).abs().max()}"
        )


# =============================================================================
# Qwen3.5 State Dict Adapter Tests
# =============================================================================


_requires_qwen3_5 = pytest.mark.skipif(
    not _HAS_QWEN3_5_ADAPTER,
    reason="Qwen3.5 adapter not available (requires transformers >= 5.2)",
)

_QWEN3_5_EXACT = dict(rtol=1e-6, atol=1e-6)

_QWEN3_5_LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
]


class _Qwen3_5DenseConfig:
    """Minimal dense config for Qwen3.5 adapter tests."""

    hidden_size = 256
    num_hidden_layers = 4
    num_attention_heads = 4
    num_key_value_heads = 2
    vocab_size = 1000
    head_dim = 128
    intermediate_size = 512
    rms_norm_eps = 1e-6
    rope_theta = 1000000.0
    partial_rotary_factor = 0.25
    tie_word_embeddings = False
    layer_types = list(_QWEN3_5_LAYER_TYPES)
    linear_conv_kernel_dim = 4
    linear_key_head_dim = 32
    linear_value_head_dim = 32
    linear_num_key_heads = 4
    linear_num_value_heads = 8
    attention_bias = False


class _Qwen3_5MoEConfig:
    """Minimal MoE config for Qwen3.5 adapter tests."""

    hidden_size = 256
    num_hidden_layers = 4
    num_attention_heads = 4
    num_key_value_heads = 2
    vocab_size = 1000
    head_dim = 128
    intermediate_size = 512
    rms_norm_eps = 1e-6
    rope_theta = 1000000.0
    partial_rotary_factor = 0.25
    tie_word_embeddings = False
    layer_types = list(_QWEN3_5_LAYER_TYPES)
    linear_conv_kernel_dim = 4
    linear_key_head_dim = 32
    linear_value_head_dim = 32
    linear_num_key_heads = 4
    linear_num_value_heads = 8
    attention_bias = False
    # MoE
    num_experts = 4
    num_experts_per_tok = 2
    moe_intermediate_size = 128
    shared_expert_intermediate_size = 512
    moe_enabled = True


# ---------------------------------------------------------------------------
# Qwen3.5 Helpers
# ---------------------------------------------------------------------------
def _all_full_attn_hf_keys(layer_idx: int) -> list[str]:
    """All HF keys for a Qwen3.5 full_attention layer."""
    suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ]
    return [f"model.layers.{layer_idx}.{s}" for s in suffixes]


def _all_linear_attn_hf_keys(layer_idx: int) -> list[str]:
    """All HF keys for a Qwen3.5 linear_attention (GatedDeltaNet) layer."""
    weight_suffixes = [
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.conv1d.weight",
        "linear_attn.out_proj.weight",
        "linear_attn.norm.weight",
    ]
    bare_suffixes = [
        "linear_attn.A_log",
        "linear_attn.dt_bias",
    ]
    keys = [f"model.layers.{layer_idx}.{s}" for s in weight_suffixes]
    keys += [f"model.layers.{layer_idx}.{s}" for s in bare_suffixes]
    return keys


def _common_per_layer_hf_keys(layer_idx: int, moe: bool = False) -> list[str]:
    """Common per-layer HF keys (norms + FFN)."""
    keys = [
        f"model.layers.{layer_idx}.input_layernorm.weight",
        f"model.layers.{layer_idx}.post_attention_layernorm.weight",
    ]
    if moe:
        keys.append(f"model.layers.{layer_idx}.mlp.gate.weight")
    else:
        keys += [
            f"model.layers.{layer_idx}.mlp.gate_proj.weight",
            f"model.layers.{layer_idx}.mlp.up_proj.weight",
            f"model.layers.{layer_idx}.mlp.down_proj.weight",
        ]
    return keys


def _global_hf_keys() -> list[str]:
    """Global (non-layer) HF keys for Qwen3.5."""
    return [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]


def _generate_qwen3_5_hf_state_dict(
    config: _Qwen3_5DenseConfig | _Qwen3_5MoEConfig,
) -> dict[str, torch.Tensor]:
    """Generate a complete dummy HF state dict with random tensors."""
    sd: dict[str, torch.Tensor] = {}
    dim = config.hidden_size
    hidden_dim = config.intermediate_size
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    is_moe = hasattr(config, "moe_enabled") and config.moe_enabled

    # Global
    sd["model.embed_tokens.weight"] = torch.randn(config.vocab_size, dim)
    sd["model.norm.weight"] = torch.randn(dim)
    sd["lm_head.weight"] = torch.randn(config.vocab_size, dim)

    for i in range(config.num_hidden_layers):
        lt = config.layer_types[i]

        # Norms (common to all layer types)
        sd[f"model.layers.{i}.input_layernorm.weight"] = torch.randn(dim)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.randn(dim)

        if lt == "full_attention":
            # q_proj is 2x for gated attention
            sd[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.randn(
                n_heads * head_dim * 2, dim
            )
            sd[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.randn(
                n_kv_heads * head_dim, dim
            )
            sd[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.randn(
                n_kv_heads * head_dim, dim
            )
            sd[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.randn(
                dim, n_heads * head_dim
            )
            sd[f"model.layers.{i}.self_attn.q_norm.weight"] = torch.randn(head_dim)
            sd[f"model.layers.{i}.self_attn.k_norm.weight"] = torch.randn(head_dim)
        else:
            # linear_attention (GatedDeltaNet)
            nk = config.linear_num_key_heads
            nv = config.linear_num_value_heads
            dk = config.linear_key_head_dim
            dv = config.linear_value_head_dim
            qkv_dim = 2 * nk * dk + nv * dv
            sd[f"model.layers.{i}.linear_attn.in_proj_qkv.weight"] = torch.randn(
                qkv_dim, dim
            )
            sd[f"model.layers.{i}.linear_attn.in_proj_z.weight"] = torch.randn(
                nv * dv, dim
            )
            sd[f"model.layers.{i}.linear_attn.in_proj_a.weight"] = torch.randn(nv, dim)
            sd[f"model.layers.{i}.linear_attn.in_proj_b.weight"] = torch.randn(nv, dim)
            sd[f"model.layers.{i}.linear_attn.conv1d.weight"] = torch.randn(
                qkv_dim, 1, config.linear_conv_kernel_dim
            )
            sd[f"model.layers.{i}.linear_attn.out_proj.weight"] = torch.randn(
                dim, nv * dv
            )
            sd[f"model.layers.{i}.linear_attn.norm.weight"] = torch.randn(dv)
            # Bare parameters
            sd[f"model.layers.{i}.linear_attn.A_log"] = torch.randn(nv)
            sd[f"model.layers.{i}.linear_attn.dt_bias"] = torch.randn(nv)

        # FFN / MoE
        if is_moe:
            num_exp = config.num_experts
            moe_dim = config.moe_intermediate_size
            sd[f"model.layers.{i}.mlp.gate.weight"] = torch.randn(num_exp, dim)
            for e in range(num_exp):
                sd[f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                    moe_dim, dim
                )
                sd[f"model.layers.{i}.mlp.experts.{e}.up_proj.weight"] = torch.randn(
                    moe_dim, dim
                )
                sd[f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"] = torch.randn(
                    dim, moe_dim
                )
        else:
            sd[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.randn(hidden_dim, dim)
            sd[f"model.layers.{i}.mlp.up_proj.weight"] = torch.randn(hidden_dim, dim)
            sd[f"model.layers.{i}.mlp.down_proj.weight"] = torch.randn(dim, hidden_dim)

    return sd


# =============================================================================
# Qwen3.5 Key Mapping Tests
# =============================================================================
@_requires_qwen3_5
class TestQwen3_5KeyMapping:
    """Key mapping roundtrip for Qwen3.5 (full_attention + linear_attention)."""

    def test_key_mapping_full_attention_roundtrip(self):
        """HF full_attention key → Archon → HF roundtrip."""
        adapter = Qwen3_5StateDictAdapter(_Qwen3_5DenseConfig())

        # Layer 3 is full_attention in our test config
        hf_keys = _all_full_attn_hf_keys(3)
        for hf_key in hf_keys:
            archon_key = adapter._convert_key_from_hf(hf_key)
            assert archon_key is not None, f"HF key unmapped: {hf_key}"

            roundtrip_key = adapter._convert_key_to_hf(archon_key)
            assert roundtrip_key == hf_key, (
                f"Roundtrip mismatch: {hf_key} → {archon_key} → {roundtrip_key}"
            )

        # Verify expected Archon key format
        wq = adapter._convert_key_from_hf("model.layers.3.self_attn.q_proj.weight")
        assert wq == "layers.3.attention.wq.weight"
        q_norm = adapter._convert_key_from_hf("model.layers.3.self_attn.q_norm.weight")
        assert q_norm == "layers.3.attention.q_norm.weight"

    def test_key_mapping_linear_attention_roundtrip(self):
        """HF linear_attention key → Archon → HF roundtrip."""
        adapter = Qwen3_5StateDictAdapter(_Qwen3_5DenseConfig())

        # Layer 0 is linear_attention in our test config
        hf_keys = _all_linear_attn_hf_keys(0)
        for hf_key in hf_keys:
            archon_key = adapter._convert_key_from_hf(hf_key)
            assert archon_key is not None, f"HF key unmapped: {hf_key}"

            roundtrip_key = adapter._convert_key_to_hf(archon_key)
            assert roundtrip_key == hf_key, (
                f"Roundtrip mismatch: {hf_key} → {archon_key} → {roundtrip_key}"
            )

        # Verify bare params (no .weight suffix)
        a_log = adapter._convert_key_from_hf("model.layers.0.linear_attn.A_log")
        assert a_log == "layers.0.linear_attn.A_log"
        dt_bias = adapter._convert_key_from_hf("model.layers.0.linear_attn.dt_bias")
        assert dt_bias == "layers.0.linear_attn.dt_bias"

        # Verify conv1d
        conv = adapter._convert_key_from_hf("model.layers.0.linear_attn.conv1d.weight")
        assert conv == "layers.0.linear_attn.conv1d.weight"

    def test_key_mapping_skip_rotary(self):
        """Rotary emb inv_freq should map to None (skipped)."""
        adapter = Qwen3_5StateDictAdapter(_Qwen3_5DenseConfig())
        result = adapter._convert_key_from_hf(
            "model.layers.3.self_attn.rotary_emb.inv_freq"
        )
        assert result is None


# =============================================================================
# Qwen3.5 MoE Key Mapping
# =============================================================================
@_requires_qwen3_5
class TestQwen3_5MoEKeyMapping:
    """MoE expert 2D↔3D conversion for Qwen3.5."""

    def test_key_mapping_moe_roundtrip(self):
        """MoE expert weights 2D ↔ 3D roundtrip."""
        config = _Qwen3_5MoEConfig()
        adapter = Qwen3_5StateDictAdapter(config)

        dim = config.hidden_size
        moe_dim = config.moe_intermediate_size
        num_exp = config.num_experts

        # Create HF-format 2D expert weights for layer 1
        hf_sd = {}
        for e in range(num_exp):
            hf_sd[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = torch.randn(
                moe_dim, dim
            )
            hf_sd[f"model.layers.1.mlp.experts.{e}.up_proj.weight"] = torch.randn(
                moe_dim, dim
            )
            hf_sd[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = torch.randn(
                dim, moe_dim
            )

        # HF → Archon: should produce 3D tensors
        archon_sd = adapter.from_hf(hf_sd)
        assert "layers.1.moe.experts.w1" in archon_sd
        assert "layers.1.moe.experts.w2" in archon_sd
        assert "layers.1.moe.experts.w3" in archon_sd
        assert archon_sd["layers.1.moe.experts.w1"].shape == (num_exp, moe_dim, dim)
        assert archon_sd["layers.1.moe.experts.w2"].shape == (num_exp, dim, moe_dim)
        assert archon_sd["layers.1.moe.experts.w3"].shape == (num_exp, moe_dim, dim)

        # Verify per-expert data integrity (w1 = gate_proj)
        for e in range(num_exp):
            assert_close(
                archon_sd["layers.1.moe.experts.w1"][e],
                hf_sd[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"],
                **_QWEN3_5_EXACT,
            )

        # Archon → HF: should produce 2D tensors
        roundtrip_sd = adapter.to_hf(archon_sd)
        for key in hf_sd:
            assert key in roundtrip_sd, f"Missing roundtrip key: {key}"
            assert_close(roundtrip_sd[key], hf_sd[key], **_QWEN3_5_EXACT)


# =============================================================================
# Qwen3.5 Key Coverage
# =============================================================================
@_requires_qwen3_5
class TestQwen3_5KeyCoverage:
    """Verify complete HF key coverage for Qwen3.5."""

    def test_all_hf_keys_covered(self):
        """All HF keys map to Archon keys (zero unmapped)."""
        config = _Qwen3_5DenseConfig()
        adapter = Qwen3_5StateDictAdapter(config)

        expected_keys: list[str] = []
        for i in range(config.num_hidden_layers):
            lt = config.layer_types[i]
            if lt == "full_attention":
                expected_keys += _all_full_attn_hf_keys(i)
            else:
                expected_keys += _all_linear_attn_hf_keys(i)
            expected_keys += _common_per_layer_hf_keys(i, moe=False)
        expected_keys += _global_hf_keys()

        unmapped = [k for k in expected_keys if adapter._convert_key_from_hf(k) is None]
        assert unmapped == [], f"Unmapped HF keys: {unmapped}"

    def test_all_hf_keys_covered_moe(self):
        """All MoE HF keys map correctly (router + experts via parse)."""
        config = _Qwen3_5MoEConfig()
        adapter = Qwen3_5StateDictAdapter(config)

        # Test router key
        router_key = "model.layers.0.mlp.gate.weight"
        assert adapter._convert_key_from_hf(router_key) is not None

        # Test expert keys parse correctly
        for e in range(config.num_experts):
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                hf_key = f"model.layers.0.mlp.experts.{e}.{proj}.weight"
                parsed = adapter._parse_expert_key(hf_key)
                assert parsed is not None, f"Expert key not parseable: {hf_key}"


# =============================================================================
# Qwen3.5 Full State Dict Roundtrip
# =============================================================================
@_requires_qwen3_5
class TestQwen3_5StateDictRoundtrip:
    """Full HF → Archon → HF roundtrip for Qwen3.5."""

    def test_state_dict_adapter_roundtrip(self):
        """Complete state dict roundtrip with per-key exact match."""
        config = _Qwen3_5DenseConfig()
        adapter = Qwen3_5StateDictAdapter(config)

        hf_sd = _generate_qwen3_5_hf_state_dict(config)
        archon_sd = adapter.from_hf(hf_sd)
        roundtrip_sd = adapter.to_hf(archon_sd)

        # All original keys must be present
        assert set(roundtrip_sd.keys()) == set(hf_sd.keys()), (
            f"Key mismatch.\n"
            f"Missing: {set(hf_sd.keys()) - set(roundtrip_sd.keys())}\n"
            f"Extra: {set(roundtrip_sd.keys()) - set(hf_sd.keys())}"
        )

        # Per-key tensor equality
        for key in hf_sd:
            assert_close(
                roundtrip_sd[key],
                hf_sd[key],
                **_QWEN3_5_EXACT,
                msg=f"Tensor mismatch for key: {key}",
            )

    def test_state_dict_adapter_roundtrip_moe(self):
        """MoE state dict roundtrip with expert 2D↔3D."""
        config = _Qwen3_5MoEConfig()
        adapter = Qwen3_5StateDictAdapter(config)

        hf_sd = _generate_qwen3_5_hf_state_dict(config)
        archon_sd = adapter.from_hf(hf_sd)
        roundtrip_sd = adapter.to_hf(archon_sd)

        assert set(roundtrip_sd.keys()) == set(hf_sd.keys()), (
            f"Key mismatch.\n"
            f"Missing: {set(hf_sd.keys()) - set(roundtrip_sd.keys())}\n"
            f"Extra: {set(roundtrip_sd.keys()) - set(hf_sd.keys())}"
        )

        for key in hf_sd:
            assert_close(
                roundtrip_sd[key],
                hf_sd[key],
                **_QWEN3_5_EXACT,
                msg=f"Tensor mismatch for key: {key}",
            )

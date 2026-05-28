"""Tests for Qwen3ModelArgs - Qwen3-specific configuration handling.

This file tests Qwen3ModelArgs-specific functionality that is NOT covered by
test_moe_args.py (which tests the underlying MoEArgs dataclass).

Focus areas:
- Qwen3ModelArgs default values
- decoder_sparse_step configuration (Qwen3-specific)
- Qwen3ModelArgs.from_hf_config() integration with MoE

For MoEArgs-specific tests (num_experts, top_k, route_norm, etc.),
see test_moe_args.py.

Run tests:
    pytest tests/experimental/archon/test_qwen3_args.py -v
"""

from areal.experimental.models.archon.moe import MoEArgs
from areal.experimental.models.archon.qwen3 import Qwen3ModelArgs


class TestQwen3ModelArgsDefaults:
    """Tests for Qwen3ModelArgs default values."""

    def test_default_values(self):
        """Test default values for non-MoE model."""
        args = Qwen3ModelArgs()

        assert args.dim == 1024
        assert args.n_layers == 28
        assert args.moe_enabled is False
        assert args.moe_args is None
        assert args.decoder_sparse_step == 1

    def test_moe_disabled_by_default(self):
        """Test that MoE is disabled by default."""
        args = Qwen3ModelArgs()

        assert args.moe_enabled is False
        assert args.moe_args is None


class TestQwen3ModelArgsDecoderSparseStep:
    """Tests for decoder_sparse_step configuration (Qwen3-specific)."""

    def test_decoder_sparse_step_default(self):
        """Test decoder_sparse_step defaults to 1 (all layers MoE when enabled)."""
        args = Qwen3ModelArgs(
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=8),
        )
        assert args.decoder_sparse_step == 1

    def test_decoder_sparse_step_2(self):
        """Test decoder_sparse_step=2 (every other layer is MoE)."""
        args = Qwen3ModelArgs(
            moe_enabled=True,
            moe_args=MoEArgs(num_experts=8),
            decoder_sparse_step=2,
        )
        assert args.decoder_sparse_step == 2

    def test_decoder_sparse_step_from_hf_config(self):
        """Test from_hf_config extracts decoder_sparse_step."""

        class MockMoEConfig:
            hidden_size = 2048
            num_hidden_layers = 36
            num_attention_heads = 16
            num_key_value_heads = 4
            vocab_size = 151936
            intermediate_size = 8960
            rms_norm_eps = 1e-5
            num_experts = 64
            num_experts_per_tok = 4
            decoder_sparse_step = 2  # Every other layer is MoE

        args = Qwen3ModelArgs.from_hf_config(MockMoEConfig())

        assert args.decoder_sparse_step == 2


class TestQwen3ModelArgsFromHfConfig:
    """Tests for Qwen3ModelArgs.from_hf_config() - integration tests.

    Note: Basic MoE field extraction (num_experts, top_k, etc.) is tested
    in test_moe_args.py. This class tests Qwen3-specific integration.
    """

    def test_from_hf_config_dense(self):
        """Test from_hf_config for dense (non-MoE) model."""

        class MockDenseConfig:
            hidden_size = 1024
            num_hidden_layers = 24
            num_attention_heads = 16
            num_key_value_heads = 8
            vocab_size = 100000
            intermediate_size = 2816
            rms_norm_eps = 1e-6

        args = Qwen3ModelArgs.from_hf_config(MockDenseConfig())

        assert args.moe_enabled is False
        assert args.moe_args is None
        assert args.dim == 1024
        assert args.n_layers == 24

    def test_from_hf_config_moe_enables_flag(self):
        """Test from_hf_config correctly enables moe_enabled flag."""

        class MockMoEConfig:
            hidden_size = 2048
            num_hidden_layers = 36
            num_attention_heads = 16
            num_key_value_heads = 4
            vocab_size = 151936
            intermediate_size = 8960
            rms_norm_eps = 1e-5
            num_experts = 128
            num_experts_per_tok = 8
            moe_intermediate_size = 1408
            norm_topk_prob = True

        args = Qwen3ModelArgs.from_hf_config(MockMoEConfig())

        assert args.moe_enabled is True
        assert args.moe_args is not None
        assert args.moe_inter_dim == 1408

    def test_single_expert_not_moe(self):
        """Test that num_experts=1 does not enable MoE."""

        class MockSingleExpert:
            hidden_size = 1024
            num_hidden_layers = 24
            num_attention_heads = 16
            num_key_value_heads = 8
            vocab_size = 100000
            intermediate_size = 2816
            rms_norm_eps = 1e-6
            num_experts = 1

        args = Qwen3ModelArgs.from_hf_config(MockSingleExpert())

        assert args.moe_enabled is False
        assert args.moe_args is None

    def test_from_hf_config_moe_with_shared_experts(self):
        """Test from_hf_config with shared experts (Qwen3 feature)."""

        class MockMoEConfig:
            hidden_size = 2048
            num_hidden_layers = 36
            num_attention_heads = 16
            num_key_value_heads = 4
            vocab_size = 151936
            intermediate_size = 8960
            rms_norm_eps = 1e-5
            num_experts = 64
            num_experts_per_tok = 4
            num_shared_experts = 2

        args = Qwen3ModelArgs.from_hf_config(MockMoEConfig())

        assert args.moe_args.num_shared_experts == 2

"""CPU-only unit tests for Megatron engine VLM (Vision-Language Model) support.

Distributed integration tests live in
``tests/test_megatron_engine_vlm_distributed.py`` — they launch torchrun
subprocesses and require GPUs.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch


class TestUnwrapToGptModel:
    """Test unwrap_to_gpt_model with VLM model structures."""

    def test_plain_gpt_model_unwraps(self):
        """Plain GPTModel (no wrapping) should return itself."""
        from megatron.core.models.gpt.gpt_model import GPTModel

        from areal.models.mcore.registry import unwrap_to_gpt_model

        mock_gpt = MagicMock(spec=GPTModel)
        mock_gpt.__class__ = GPTModel
        result = unwrap_to_gpt_model(mock_gpt)
        assert result is mock_gpt

    def test_ddp_wrapped_gpt_model_unwraps(self):
        """GPTModel wrapped in DDP (via .module) should unwrap correctly."""
        from megatron.core.models.gpt.gpt_model import GPTModel

        from areal.models.mcore.registry import unwrap_to_gpt_model

        mock_gpt = MagicMock(spec=GPTModel)
        mock_gpt.__class__ = GPTModel
        wrapper = MagicMock()
        wrapper.__class__ = type("DDPWrapper", (), {})
        wrapper.module = mock_gpt
        result = unwrap_to_gpt_model(wrapper)
        assert result is mock_gpt

    def test_vlm_model_with_language_model_unwraps(self):
        """VLM model (e.g. Qwen2_5VLModel) with .language_model attribute unwraps."""
        from megatron.core.models.gpt.gpt_model import GPTModel

        from areal.models.mcore.registry import unwrap_to_gpt_model

        mock_gpt = MagicMock(spec=GPTModel)
        mock_gpt.__class__ = GPTModel

        # VLM model: no .module, but has .language_model
        vlm_model = MagicMock()
        vlm_model.__class__ = type("Qwen2_5VLModel", (), {})
        del vlm_model.module  # ensure no .module attribute
        vlm_model.language_model = mock_gpt

        result = unwrap_to_gpt_model(vlm_model)
        assert result is mock_gpt

    def test_ddp_wrapped_vlm_model_unwraps(self):
        """VLM model wrapped in DDP should unwrap to language_model."""
        from megatron.core.models.gpt.gpt_model import GPTModel

        from areal.models.mcore.registry import unwrap_to_gpt_model

        mock_gpt = MagicMock(spec=GPTModel)
        mock_gpt.__class__ = GPTModel

        vlm_model = MagicMock()
        vlm_model.__class__ = type("Qwen2_5VLModel", (), {})
        del vlm_model.module
        vlm_model.language_model = mock_gpt

        wrapper = MagicMock()
        wrapper.__class__ = type("DDPWrapper", (), {})
        wrapper.module = vlm_model

        result = unwrap_to_gpt_model(wrapper)
        assert result is mock_gpt

    def test_unsupported_model_raises_type_error(self):
        """Model with neither GPTModel nor .language_model should raise TypeError."""
        from areal.models.mcore.registry import unwrap_to_gpt_model

        unsupported = MagicMock()
        unsupported.__class__ = type("RandomModel", (), {})
        del unsupported.module
        del unsupported.language_model

        with pytest.raises(TypeError, match="could not be unwrapped"):
            unwrap_to_gpt_model(unsupported)


class TestExtractVisionFromMultiModal:
    """Test _extract_vision_from_multi_modal helper."""

    def test_extracts_pixel_values_and_grid(self):
        """Vision tensors should land on padded_mb only, not duplicated on mb."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        mb: dict[str, Any] = {
            "multi_modal_input": [
                {
                    "pixel_values": torch.randn(10, 3, 14, 14),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                },
                {
                    "pixel_values": torch.randn(5, 3, 14, 14),
                    "image_grid_thw": torch.tensor([[1, 1, 1]]),
                },
            ]
        }
        padded_mb: dict[str, Any] = dict(mb)

        extract_vision_from_multi_modal(mb, padded_mb)

        # Forward side (padded_mb) gets the concatenated vision tensors.
        assert padded_mb["pixel_values"].shape[0] == 15  # 10 + 5
        assert padded_mb["image_grid_thw"].shape[0] == 2  # 2 grids
        # Loss side (mb) does not carry them.
        assert "pixel_values" not in mb
        assert "image_grid_thw" not in mb
        # multi_modal_input is consumed and removed from both sides.
        assert "multi_modal_input" not in mb
        assert "multi_modal_input" not in padded_mb

    def test_handles_video_grid_thw(self):
        """Should extract video_grid_thw onto padded_mb."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        mb: dict[str, Any] = {
            "multi_modal_input": [
                {
                    "pixel_values": torch.randn(4, 3, 14, 14),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                    "video_grid_thw": torch.tensor([[2, 2, 2]]),
                },
            ]
        }
        padded_mb: dict[str, Any] = dict(mb)

        extract_vision_from_multi_modal(mb, padded_mb)

        assert padded_mb["video_grid_thw"].shape[0] == 1
        assert "video_grid_thw" not in mb

    def test_no_multi_modal_input_is_noop(self):
        """Should do nothing if multi_modal_input not in dict."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        mb: dict[str, Any] = {"input_ids": torch.tensor([1, 2, 3])}
        padded_mb: dict[str, Any] = dict(mb)

        extract_vision_from_multi_modal(mb, padded_mb)

        assert "pixel_values" not in mb
        assert "image_grid_thw" not in mb

    def test_empty_multi_modal_input_is_noop(self):
        """Should not add keys if multi_modal_input items have no vision data,
        but should still pop multi_modal_input itself."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        mb: dict[str, Any] = {"multi_modal_input": [{}]}
        padded_mb: dict[str, Any] = dict(mb)

        extract_vision_from_multi_modal(mb, padded_mb)

        assert "pixel_values" not in mb
        assert "pixel_values" not in padded_mb
        assert "multi_modal_input" not in mb
        assert "multi_modal_input" not in padded_mb

    def test_falls_back_to_padded_mb(self):
        """When mb lacks multi_modal_input but padded_mb has it, the fallback
        branch should still concatenate vision tensors onto padded_mb."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        pixel_values = [torch.randn(2, 4)]
        mb: dict[str, Any] = {"input_ids": torch.ones(2, dtype=torch.long)}
        padded_mb: dict[str, Any] = {
            "input_ids": torch.ones(2, dtype=torch.long),
            "multi_modal_input": [{"pixel_values": pixel_values[0]}],
        }

        extract_vision_from_multi_modal(mb, padded_mb)

        assert "multi_modal_input" not in mb
        assert "multi_modal_input" not in padded_mb
        assert "pixel_values" not in mb
        assert torch.equal(padded_mb["pixel_values"], torch.cat(pixel_values, dim=0))


class TestPrepareMbListRebindCallerSafety:
    """Verify _prepare_mb_list rebinds mb_list.data to a filtered copy so the
    caller's input dict survives across repeated forward() calls.

    `split_padded_tensor_dict_into_mb_list` constructs `MicroBatchList(data=input_)`
    where `mb_list.data` is the *same object* as the caller's input dict (not a
    copy). An earlier draft did `_drop_multi_modal_payload(mb_list.data)` in-place,
    which mutated the caller's dict and broke `engine.forward(input_)` when the
    same `input_` was forwarded a second time (e.g. the save/load round-trip
    test). The current implementation rebinds `mb_list.data = {filtered copy}`
    instead — this test pins that contract.
    """

    def test_rebind_does_not_mutate_caller_input(self):
        from areal.engine.megatron_utils.packed_context_parallel import (
            _is_multi_modal_payload_key,
        )
        from areal.utils.data import MicroBatchList, MicroBatchSpec

        input_ = {
            "input_ids": torch.ones(4, dtype=torch.long),
            "multi_modal_input": [{"pixel_values": torch.randn(2, 4)}],
            "pixel_values": torch.randn(2, 4),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }
        snapshot_keys = set(input_.keys())

        mb_list = MicroBatchList(
            data=input_,
            mb_spec=MicroBatchSpec(),
            mbs=[],
            group_lens=[],
        )
        assert mb_list.data is input_, (
            "MicroBatchList.data must alias caller's input dict to mirror "
            "split_padded_tensor_dict_into_mb_list semantics."
        )

        mb_list.data = {
            k: v for k, v in mb_list.data.items() if not _is_multi_modal_payload_key(k)
        }

        assert set(input_.keys()) == snapshot_keys, (
            "Rebind must not mutate the caller's input dict — regression to "
            "in-place `_drop_multi_modal_payload(mb_list.data)` would fail here."
        )
        assert "multi_modal_input" not in mb_list.data
        assert "pixel_values" not in mb_list.data
        assert "image_grid_thw" not in mb_list.data
        assert "input_ids" in mb_list.data
        assert mb_list.data is not input_


class TestVisionModelDetection:
    """Test is_valid_vision_model detection."""

    def test_qwen2_5_vl_detected(self):
        from areal.engine.core.model import is_valid_vision_model

        assert is_valid_vision_model("qwen2_5_vl") is True

    def test_qwen2_vl_detected(self):
        from areal.engine.core.model import is_valid_vision_model

        assert is_valid_vision_model("qwen2_vl") is True

    def test_qwen3_vl_detected(self):
        from areal.engine.core.model import is_valid_vision_model

        assert is_valid_vision_model("qwen3_vl") is True

    def test_qwen3_vl_moe_detected(self):
        from areal.engine.core.model import (
            is_qwen3_vl_model,
            is_qwen3_vl_moe_model,
            is_valid_vision_model,
        )

        assert is_valid_vision_model("qwen3_vl_moe") is True
        assert is_qwen3_vl_moe_model("qwen3_vl_moe") is True
        # Family-level helper covers both dense and MoE.
        assert is_qwen3_vl_model("qwen3_vl_moe") is True
        assert is_qwen3_vl_model("qwen3_vl") is True
        assert is_qwen3_vl_moe_model("qwen3_vl") is False

    def test_qwen3_not_detected(self):
        from areal.engine.core.model import is_valid_vision_model

        assert is_valid_vision_model("qwen3") is False

    def test_text_model_not_detected(self):
        from areal.engine.core.model import is_valid_vision_model

        assert is_valid_vision_model("llama") is False


class TestConvertQwen25VLToHF:
    """Test mcore→HF weight name conversion for Qwen2.5-VL."""

    @pytest.fixture()
    def tf_config(self):
        """Mock TransformerConfig with Qwen2.5-VL-3B values."""
        cfg = MagicMock()
        cfg.hidden_size = 2048
        cfg.num_attention_heads = 16
        cfg.num_query_groups = 2
        cfg.kv_channels = 128
        return cfg

    @pytest.fixture()
    def hf_config(self):
        """Mock HF config with vision_config for Qwen2.5-VL-3B."""
        cfg = MagicMock()
        cfg.vision_config.num_heads = 16
        cfg.vision_config.hidden_size = 1280
        return cfg

    # --- Registry dispatch ---

    def test_registry_dispatches_qwen2_5_vl_before_qwen2(self, tf_config, hf_config):
        """convert_to_hf with model_name='qwen2_5_vl' must use the VLM converter."""
        from areal.engine.megatron_utils.megatron import convert_to_hf

        param = torch.randn(2048)
        result = convert_to_hf(
            tf_config,
            "qwen2_5_vl",
            "module.module.vision_model.decoder.final_layernorm.weight",
            param,
            hf_config=hf_config,
        )
        assert result[0][0] == "visual.merger.ln_q.weight"

    # --- Language model delegation ---

    def test_language_model_embedding_delegates(self, tf_config):
        """language_model.embedding should strip prefix and delegate to qwen2."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(151936, 2048)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.language_model.embedding.word_embeddings.weight",
            param,
        )
        assert len(result) == 1
        assert result[0][0] == "model.embed_tokens.weight"

    def test_language_model_output_layer_delegates(self, tf_config):
        """language_model.output_layer should map to lm_head."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(151936, 2048)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.language_model.output_layer.weight",
            param,
        )
        assert result[0][0] == "lm_head.weight"

    def test_language_model_decoder_layer_qkv_delegates(self, tf_config):
        """language_model decoder QKV should split into q/k/v projections."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        # Qwen2.5-VL-3B: num_query_groups=2, value_num_per_group=8, kv_channels=128
        # QKV weight shape: (num_query_groups * (value_num_per_group + 2) * kv_channels, hidden)
        qkv_dim = tf_config.num_query_groups * (8 + 1 + 1) * tf_config.kv_channels
        param = torch.randn(qkv_dim, tf_config.hidden_size)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.5.self_attention.linear_qkv.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "model.layers.5.self_attn.q_proj.weight" in names
        assert "model.layers.5.self_attn.k_proj.weight" in names
        assert "model.layers.5.self_attn.v_proj.weight" in names

    def test_language_model_mlp_fc1_splits_gate_up(self, tf_config):
        """language_model MLP fc1 should split into gate_proj and up_proj."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(22016, 2048)  # gate + up stacked
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc1.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "model.layers.0.mlp.gate_proj.weight" in names
        assert "model.layers.0.mlp.up_proj.weight" in names

    # --- Vision model direct mappings ---

    def test_vision_patch_embed(self, tf_config):
        """vision_model.patch_embed maps to visual.patch_embed."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280, 3, 2, 14, 14)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.patch_embed.proj.weight",
            param,
        )
        assert result == [("visual.patch_embed.proj.weight", param)]

    def test_vision_merger_ln(self, tf_config):
        """vision final layernorm maps to visual.merger.ln_q."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.final_layernorm.weight",
            param,
        )
        assert result[0][0] == "visual.merger.ln_q.weight"

    def test_vision_merger_mlp(self, tf_config):
        """vision projection MLP maps to visual.merger.mlp."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(5120, 5120)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.projection.encoder.linear_fc1.weight",
            param,
        )
        assert result[0][0] == "visual.merger.mlp.0.weight"

    # --- Vision model per-layer params ---

    def test_vision_layer_attn_qkv(self, tf_config, hf_config):
        """Vision attention QKV maps to visual.blocks.<idx>.attn.qkv with reordering."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(3840, 1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.7.self_attention.linear_qkv.weight",
            param,
            hf_config=hf_config,
        )
        assert result[0][0] == "visual.blocks.7.attn.qkv.weight"
        assert result[0][1].shape == param.shape  # same shape, different ordering

    def test_vision_layer_attn_proj(self, tf_config):
        """Vision attention proj maps to visual.blocks.<idx>.attn.proj."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280, 1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.3.self_attention.linear_proj.weight",
            param,
        )
        assert result[0][0] == "visual.blocks.3.attn.proj.weight"

    def test_vision_layer_norm1(self, tf_config):
        """Vision input layernorm maps to visual.blocks.<idx>.norm1."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
            param,
        )
        assert result[0][0] == "visual.blocks.0.norm1.weight"

    def test_vision_layer_mlp_fc1_splits_gate_up(self, tf_config):
        """Vision MLP fc1 splits into gate_proj and up_proj."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(6840, 1280)  # gate + up stacked
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.2.mlp.linear_fc1.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "visual.blocks.2.mlp.gate_proj.weight" in names
        assert "visual.blocks.2.mlp.up_proj.weight" in names

    def test_vision_layer_mlp_fc2(self, tf_config):
        """Vision MLP fc2 maps to visual.blocks.<idx>.mlp.down_proj."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280, 3420)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.10.mlp.linear_fc2.weight",
            param,
        )
        assert result[0][0] == "visual.blocks.10.mlp.down_proj.weight"

    def test_vision_layer_norm2(self, tf_config):
        """Vision post-attention layernorm maps to visual.blocks.<idx>.norm2."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.1.mlp.linear_fc1.layer_norm_weight",
            param,
        )
        assert result[0][0] == "visual.blocks.1.norm2.weight"

    # --- Error handling ---

    def test_unknown_param_raises_value_error(self, tf_config):
        """Unknown parameter name should raise ValueError."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        with pytest.raises(ValueError, match="Unknown parameter name"):
            convert_qwen2_5_vl_to_hf(
                tf_config,
                "module.module.some_unknown.weight",
                torch.randn(10),
            )


class TestConvertQwen3VLToHF:
    """Test mcore→HF weight name conversion for Qwen3-VL (dense)."""

    @pytest.fixture()
    def tf_config(self):
        """Mock TransformerConfig matching Qwen/Qwen3-VL-2B-Instruct text_config:
        hidden=2048, attn_heads=16, kv_heads=8, head_dim=128.
        """
        cfg = MagicMock()
        cfg.hidden_size = 2048
        cfg.num_attention_heads = 16
        cfg.num_query_groups = 8
        cfg.kv_channels = 128
        return cfg

    @pytest.fixture()
    def hf_config(self):
        """Mock HF config matching Qwen/Qwen3-VL-2B-Instruct vision_config:
        hidden=1024, num_heads=16 (head_dim=64).
        """
        cfg = MagicMock()
        cfg.vision_config.num_heads = 16
        cfg.vision_config.hidden_size = 1024
        return cfg

    # --- Registry dispatch ---

    def test_registry_dispatches_qwen3_vl_before_qwen3(self, tf_config, hf_config):
        """convert_to_hf with model_name='qwen3_vl' must use the VLM converter,
        not fall through to qwen3 (substring match order)."""
        from areal.engine.megatron_utils.megatron import convert_to_hf

        param = torch.randn(1024)
        result = convert_to_hf(
            tf_config,
            "qwen3_vl",
            "module.module.vision_model.merger.patch_norm.weight",
            param,
            hf_config=hf_config,
        )
        assert result[0][0] == "model.visual.merger.norm.weight"

    # --- Language model (Qwen3-VL has model.language_model.* prefix) ---

    def test_language_model_embedding_uses_language_model_prefix(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(151936, 2048)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.embedding.word_embeddings.weight",
            param,
        )
        assert result == [("model.language_model.embed_tokens.weight", param)]

    def test_language_model_final_norm(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(2048)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.final_layernorm.weight",
            param,
        )
        assert result == [("model.language_model.norm.weight", param)]

    def test_language_model_output_layer(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(151936, 2048)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.output_layer.weight",
            param,
        )
        assert result == [("lm_head.weight", param)]

    def test_language_model_qkv_split(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        # value_num_per_group = 16/8 = 2, head_dim = 128
        # qkv_dim = num_query_groups * (value_num_per_group + 2) * head_dim
        qkv_dim = tf_config.num_query_groups * (2 + 1 + 1) * tf_config.kv_channels
        param = torch.randn(qkv_dim, tf_config.hidden_size)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.5.self_attention.linear_qkv.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "model.language_model.layers.5.self_attn.q_proj.weight" in names
        assert "model.language_model.layers.5.self_attn.k_proj.weight" in names
        assert "model.language_model.layers.5.self_attn.v_proj.weight" in names

    def test_language_model_qkv_bias_split(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        qkv_dim = tf_config.num_query_groups * (2 + 1 + 1) * tf_config.kv_channels
        param = torch.randn(qkv_dim)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.self_attention.linear_qkv.bias",
            param,
        )
        names = [n for n, _ in result]
        assert "model.language_model.layers.0.self_attn.q_proj.bias" in names
        assert "model.language_model.layers.0.self_attn.k_proj.bias" in names
        assert "model.language_model.layers.0.self_attn.v_proj.bias" in names

    def test_language_model_q_norm(self, tf_config):
        """Qwen3-specific q_layernorm → self_attn.q_norm."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(128)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.3.self_attention.q_layernorm.weight",
            param,
        )
        assert result == [
            ("model.language_model.layers.3.self_attn.q_norm.weight", param)
        ]

    def test_language_model_k_norm(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(128)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.3.self_attention.k_layernorm.weight",
            param,
        )
        assert result == [
            ("model.language_model.layers.3.self_attn.k_norm.weight", param)
        ]

    def test_language_model_o_proj(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(2048, 2048)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.self_attention.linear_proj.weight",
            param,
        )
        assert result == [
            ("model.language_model.layers.0.self_attn.o_proj.weight", param)
        ]

    def test_language_model_input_layernorm(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(2048)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
            param,
        )
        assert result == [
            ("model.language_model.layers.0.input_layernorm.weight", param)
        ]

    def test_language_model_post_attention_layernorm(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(2048)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc1.layer_norm_weight",
            param,
        )
        assert result == [
            ("model.language_model.layers.0.post_attention_layernorm.weight", param)
        ]

    def test_language_model_mlp_fc1_splits_gate_up(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(12288, 2048)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc1.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "model.language_model.layers.0.mlp.gate_proj.weight" in names
        assert "model.language_model.layers.0.mlp.up_proj.weight" in names

    def test_language_model_mlp_fc2(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(2048, 6144)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc2.weight",
            param,
        )
        assert result == [("model.language_model.layers.0.mlp.down_proj.weight", param)]

    # --- Vision model: direct mappings (model.visual.* prefix) ---

    def test_vision_patch_embed_weight(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(1024, 3, 2, 16, 16)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.vision_model.patch_embed.proj.weight",
            param,
        )
        assert result == [("model.visual.patch_embed.proj.weight", param)]

    def test_vision_patch_embed_bias(self, tf_config):
        """Qwen3-VL adds patch_embed bias (Qwen2.5-VL was bias=False)."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(1024)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.vision_model.patch_embed.proj.bias",
            param,
        )
        assert result == [("model.visual.patch_embed.proj.bias", param)]

    def test_vision_pos_embed(self, tf_config):
        """Qwen3-VL has pos_embed (new vs Qwen2.5-VL)."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(2304, 1024)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.vision_model.pos_embed.weight",
            param,
        )
        assert result == [("model.visual.pos_embed.weight", param)]

    def test_vision_merger_patch_norm(self, tf_config):
        """merger.patch_norm.{weight,bias} → merger.norm.{weight,bias}."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for kind in ("weight", "bias"):
            param = torch.randn(1024)
            result = convert_qwen3_vl_to_hf(
                tf_config,
                f"module.module.vision_model.merger.patch_norm.{kind}",
                param,
            )
            assert result == [(f"model.visual.merger.norm.{kind}", param)]

    def test_vision_merger_linear_fc1_fc2(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for fc in ("linear_fc1", "linear_fc2"):
            for kind in ("weight", "bias"):
                param = (
                    torch.randn(4096, 4096) if kind == "weight" else torch.randn(4096)
                )
                result = convert_qwen3_vl_to_hf(
                    tf_config,
                    f"module.module.vision_model.merger.{fc}.{kind}",
                    param,
                )
                assert result == [(f"model.visual.merger.{fc}.{kind}", param)]

    # --- Vision model: per-layer params ---

    def test_vision_block_qkv_weight_reordered(self, tf_config, hf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        # 3 * hidden_vision = 3 * 1024 = 3072
        param = torch.randn(3072, 1024)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.7.self_attention.linear_qkv.weight",
            param,
            hf_config=hf_config,
        )
        assert result[0][0] == "model.visual.blocks.7.attn.qkv.weight"
        assert result[0][1].shape == param.shape

    def test_vision_block_qkv_bias_reordered(self, tf_config, hf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(3072)
        result = convert_qwen3_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.0.self_attention.linear_qkv.bias",
            param,
            hf_config=hf_config,
        )
        assert result[0][0] == "model.visual.blocks.0.attn.qkv.bias"
        assert result[0][1].shape == param.shape

    def test_vision_block_qkv_requires_hf_config(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        param = torch.randn(3072, 1024)
        with pytest.raises(ValueError, match="vision_config.num_heads"):
            convert_qwen3_vl_to_hf(
                tf_config,
                "module.module.vision_model.decoder.layers.0.self_attention.linear_qkv.weight",
                param,
            )

    def test_vision_block_attn_proj(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for kind, shape in (("weight", (1024, 1024)), ("bias", (1024,))):
            param = torch.randn(*shape)
            result = convert_qwen3_vl_to_hf(
                tf_config,
                f"module.module.vision_model.decoder.layers.3.self_attention.linear_proj.{kind}",
                param,
            )
            assert result == [(f"model.visual.blocks.3.attn.proj.{kind}", param)]

    def test_vision_block_norm1_weight_and_bias(self, tf_config):
        """Qwen3-VL adds norm1.bias (Qwen2.5-VL was weight-only)."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for kind in ("weight", "bias"):
            param = torch.randn(1024)
            result = convert_qwen3_vl_to_hf(
                tf_config,
                f"module.module.vision_model.decoder.layers.0.self_attention.linear_qkv.layer_norm_{kind}",
                param,
            )
            assert result == [(f"model.visual.blocks.0.norm1.{kind}", param)]

    def test_vision_block_norm2_weight_and_bias(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for kind in ("weight", "bias"):
            param = torch.randn(1024)
            result = convert_qwen3_vl_to_hf(
                tf_config,
                f"module.module.vision_model.decoder.layers.1.mlp.linear_fc1.layer_norm_{kind}",
                param,
            )
            assert result == [(f"model.visual.blocks.1.norm2.{kind}", param)]

    def test_vision_block_mlp_is_NOT_gated(self, tf_config):
        """Regression guard: Qwen3-VL vision MLP is NOT gated.

        linear_fc1 must map 1:1 (no chunk into gate_proj/up_proj).
        """
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for kind, shape in (("weight", (4096, 1024)), ("bias", (4096,))):
            param = torch.randn(*shape)
            result = convert_qwen3_vl_to_hf(
                tf_config,
                f"module.module.vision_model.decoder.layers.2.mlp.linear_fc1.{kind}",
                param,
            )
            assert result == [(f"model.visual.blocks.2.mlp.linear_fc1.{kind}", param)]

    def test_vision_block_mlp_fc2(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for kind, shape in (("weight", (1024, 4096)), ("bias", (1024,))):
            param = torch.randn(*shape)
            result = convert_qwen3_vl_to_hf(
                tf_config,
                f"module.module.vision_model.decoder.layers.10.mlp.linear_fc2.{kind}",
                param,
            )
            assert result == [(f"model.visual.blocks.10.mlp.linear_fc2.{kind}", param)]

    # --- Deepstack mergers (new in Qwen3-VL) ---

    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_deepstack_merger_norm(self, tf_config, idx):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for kind in ("weight", "bias"):
            param = torch.randn(1024)
            result = convert_qwen3_vl_to_hf(
                tf_config,
                f"module.module.vision_model.decoder.deepstack_merger_list.{idx}.patch_norm.{kind}",
                param,
            )
            assert result == [
                (f"model.visual.deepstack_merger_list.{idx}.norm.{kind}", param)
            ]

    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_deepstack_merger_linear_fc(self, tf_config, idx):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        for fc in ("linear_fc1", "linear_fc2"):
            for kind, shape in (("weight", (4096, 4096)), ("bias", (4096,))):
                param = torch.randn(*shape)
                result = convert_qwen3_vl_to_hf(
                    tf_config,
                    f"module.module.vision_model.decoder.deepstack_merger_list.{idx}.{fc}.{kind}",
                    param,
                )
                assert result == [
                    (f"model.visual.deepstack_merger_list.{idx}.{fc}.{kind}", param)
                ]

    # --- Error handling ---

    def test_unknown_lang_param_raises(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        with pytest.raises(ValueError, match="Unknown Qwen3-VL language-model"):
            convert_qwen3_vl_to_hf(
                tf_config,
                "module.module.language_model.decoder.layers.0.unknown.weight",
                torch.randn(10),
            )

    def test_unknown_param_raises(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_to_hf

        with pytest.raises(ValueError, match="Unknown Qwen3-VL parameter"):
            convert_qwen3_vl_to_hf(
                tf_config,
                "module.module.some_unknown.weight",
                torch.randn(10),
            )


class TestConvertQwen3VLMoEToHF:
    """Test mcore→HF weight name conversion for Qwen3-VL-MoE.

    Qwen3-VL-MoE shares vision tower + language attention with dense
    Qwen3-VL; this class focuses on MoE-specific tensors (experts, router,
    pre_mlp_layernorm) plus registry-ordering and a few cross-cutting
    regression guards.
    """

    @pytest.fixture()
    def tf_config(self):
        """Mock TransformerConfig matching Qwen/Qwen3-VL-30B-A3B-Instruct text_config:
        hidden=2048, attn_heads=32, kv_heads=4, head_dim=128.
        """
        cfg = MagicMock()
        cfg.hidden_size = 2048
        cfg.num_attention_heads = 32
        cfg.num_query_groups = 4
        cfg.kv_channels = 128
        return cfg

    @pytest.fixture()
    def hf_config(self):
        """Mock HF config matching Qwen/Qwen3-VL-30B-A3B-Instruct vision_config."""
        cfg = MagicMock()
        cfg.vision_config.num_heads = 16
        cfg.vision_config.hidden_size = 1152
        return cfg

    # --- Registry dispatch (must hit MoE converter before dense) ---

    def test_registry_dispatches_qwen3_vl_moe_before_qwen3_vl(
        self, tf_config, hf_config
    ):
        """``qwen3_vl_moe`` model_name must dispatch to the MoE converter, not
        fall through to ``qwen3_vl`` / ``qwen3_moe`` / ``qwen3`` (substring match
        order). MoE-specific names like ``mlp.experts.linear_fc1.weight0`` would
        otherwise raise ``Unknown Qwen3-VL language-model parameter``.
        """
        from areal.engine.megatron_utils.megatron import convert_to_hf

        param = torch.randn(1536, 2048)  # 2 * expert_dim x hidden
        result = convert_to_hf(
            tf_config,
            "qwen3_vl_moe",
            "module.module.language_model.decoder.layers.0.mlp.experts.linear_fc1.weight5",
            param,
            hf_config=hf_config,
        )
        names = [n for n, _ in result]
        assert "model.language_model.layers.0.mlp.experts.5.gate_proj.weight" in names
        assert "model.language_model.layers.0.mlp.experts.5.up_proj.weight" in names

    def test_registry_dense_still_routes_to_dense(self, tf_config, hf_config):
        """Sanity: ``qwen3_vl`` model_name still dispatches to dense converter
        even after MoE entry is inserted. Dense MLP fc1 must chunk gate/up.
        """
        from areal.engine.megatron_utils.megatron import convert_to_hf

        param = torch.randn(12288, 2048)
        result = convert_to_hf(
            tf_config,
            "qwen3_vl",
            "module.module.language_model.decoder.layers.0.mlp.linear_fc1.weight",
            param,
            hf_config=hf_config,
        )
        names = [n for n, _ in result]
        assert "model.language_model.layers.0.mlp.gate_proj.weight" in names
        assert "model.language_model.layers.0.mlp.up_proj.weight" in names

    # --- Shared global / attention paths (regression guards on factoring) ---

    def test_lm_global_embedding_shared_with_dense(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        param = torch.randn(151936, 2048)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.embedding.word_embeddings.weight",
            param,
        )
        assert result == [("model.language_model.embed_tokens.weight", param)]

    def test_lm_attention_qkv_split_shared_with_dense(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        # value_num_per_group = 32/4 = 8, head_dim = 128
        qkv_dim = tf_config.num_query_groups * (8 + 1 + 1) * tf_config.kv_channels
        param = torch.randn(qkv_dim, tf_config.hidden_size)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.4.self_attention.linear_qkv.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "model.language_model.layers.4.self_attn.q_proj.weight" in names
        assert "model.language_model.layers.4.self_attn.k_proj.weight" in names
        assert "model.language_model.layers.4.self_attn.v_proj.weight" in names

    def test_lm_attention_q_norm_shared_with_dense(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        param = torch.randn(128)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.7.self_attention.q_layernorm.weight",
            param,
        )
        assert result == [
            ("model.language_model.layers.7.self_attn.q_norm.weight", param)
        ]

    # --- MoE-specific: experts, router, pre_mlp_layernorm ---

    def test_expert_fc1_chunks_gate_up_per_expert(self, tf_config):
        """Per-expert ``linear_fc1.weight{idx}`` chunks into gate_proj and
        up_proj at the per-expert HF path.
        """
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        param = torch.randn(1536, 2048)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.3.mlp.experts.linear_fc1.weight12",
            param,
        )
        assert len(result) == 2
        names = [n for n, _ in result]
        assert "model.language_model.layers.3.mlp.experts.12.gate_proj.weight" in names
        assert "model.language_model.layers.3.mlp.experts.12.up_proj.weight" in names

    def test_expert_fc2_passthrough_per_expert(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        param = torch.randn(2048, 768)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.3.mlp.experts.linear_fc2.weight12",
            param,
        )
        assert result == [
            ("model.language_model.layers.3.mlp.experts.12.down_proj.weight", param)
        ]

    def test_router_renamed_to_mlp_gate(self, tf_config):
        """``mlp.router.weight`` → HF ``mlp.gate.weight`` (NOT ``mlp.router.weight``)."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        param = torch.randn(128, 2048)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.router.weight",
            param,
        )
        assert result == [("model.language_model.layers.0.mlp.gate.weight", param)]

    def test_pre_mlp_layernorm_renamed_to_post_attention(self, tf_config):
        """``pre_mlp_layernorm.weight`` → HF ``post_attention_layernorm.weight``."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        param = torch.randn(2048)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.5.pre_mlp_layernorm.weight",
            param,
        )
        assert result == [
            ("model.language_model.layers.5.post_attention_layernorm.weight", param)
        ]

    def test_dense_mlp_fallback_for_mixed_sparse_step(self, tf_config):
        """Layers left non-sparse by ``decoder_sparse_step > 1`` keep the dense
        MLP path. Qwen3-VL-30B-A3B-Instruct uses ``decoder_sparse_step=1`` so
        these branches are dead for that checkpoint, but they must exist for
        future variants with mixed dense/sparse layouts.
        """
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        # linear_fc1: chunk gate/up
        param = torch.randn(12288, 2048)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc1.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "model.language_model.layers.0.mlp.gate_proj.weight" in names
        assert "model.language_model.layers.0.mlp.up_proj.weight" in names

        # linear_fc1.layer_norm_weight: post_attention_layernorm
        ln_param = torch.randn(2048)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc1.layer_norm_weight",
            ln_param,
        )
        assert result == [
            ("model.language_model.layers.0.post_attention_layernorm.weight", ln_param)
        ]

        # linear_fc2: down_proj passthrough
        fc2_param = torch.randn(2048, 6144)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc2.weight",
            fc2_param,
        )
        assert result == [
            ("model.language_model.layers.0.mlp.down_proj.weight", fc2_param)
        ]

    # --- Vision tower (delegates to shared helper) ---

    def test_vision_block_qkv_reordered_via_shared_helper(self, tf_config, hf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        # 3 * hidden_vision = 3 * 1152 = 3456
        param = torch.randn(3456, 1152)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.7.self_attention.linear_qkv.weight",
            param,
            hf_config=hf_config,
        )
        assert result[0][0] == "model.visual.blocks.7.attn.qkv.weight"
        assert result[0][1].shape == param.shape

    def test_deepstack_merger_via_shared_helper(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        param = torch.randn(1152)
        result = convert_qwen3_vl_moe_to_hf(
            tf_config,
            "module.module.vision_model.decoder.deepstack_merger_list.1.patch_norm.weight",
            param,
        )
        assert result == [("model.visual.deepstack_merger_list.1.norm.weight", param)]

    # --- Error handling ---

    def test_unknown_lang_param_raises(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        with pytest.raises(ValueError, match="Unknown Qwen3-VL-MoE language-model"):
            convert_qwen3_vl_moe_to_hf(
                tf_config,
                "module.module.language_model.decoder.layers.0.unknown.weight",
                torch.randn(10),
            )

    def test_unknown_expert_kind_raises(self, tf_config):
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        with pytest.raises(ValueError, match="Unknown Qwen3-VL-MoE expert"):
            convert_qwen3_vl_moe_to_hf(
                tf_config,
                "module.module.language_model.decoder.layers.0.mlp.experts.linear_fc99.weight0",
                torch.randn(10),
            )

    def test_unknown_top_level_param_raises(self, tf_config):
        """Non-LM, non-vision names should raise via the shared vision helper."""
        from areal.engine.megatron_utils.megatron import convert_qwen3_vl_moe_to_hf

        with pytest.raises(ValueError, match="Unknown Qwen3-VL parameter"):
            convert_qwen3_vl_moe_to_hf(
                tf_config,
                "module.module.some_unknown.weight",
                torch.randn(10),
            )


class TestRemovePaddingVLM:
    """Test remove_padding handles VLM language_model prefixed params."""

    def test_vlm_embedding_padding_removed(self):
        """VLM language_model embedding should be trimmed to vocab_size."""
        from areal.engine.megatron_utils.megatron import remove_padding

        vocab_size = 151936
        padded = torch.randn(vocab_size + 64, 2048)  # padded for TP alignment
        result = remove_padding(
            "module.module.language_model.embedding.word_embeddings.weight",
            padded,
            vocab_size,
        )
        assert result.shape[0] == vocab_size

    def test_vlm_output_layer_padding_removed(self):
        """VLM language_model output_layer should be trimmed to vocab_size."""
        from areal.engine.megatron_utils.megatron import remove_padding

        vocab_size = 151936
        padded = torch.randn(vocab_size + 64, 2048)
        result = remove_padding(
            "module.module.language_model.output_layer.weight",
            padded,
            vocab_size,
        )
        assert result.shape[0] == vocab_size

    def test_non_embedding_param_unchanged(self):
        """Non-embedding params should pass through unchanged."""
        from areal.engine.megatron_utils.megatron import remove_padding

        param = torch.randn(1280)
        result = remove_padding(
            "module.module.vision_model.decoder.final_layernorm.weight",
            param,
            151936,
        )
        assert result is param

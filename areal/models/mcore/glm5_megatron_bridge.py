# SPDX-License-Identifier: Apache-2.0

"""Megatron-Bridge registration for GLM-5.1 (GlmMoeDsaForCausalLM).

Registers the GLM-5.1 architecture with NVIDIA's open-source megatron-bridge,
enabling ``bridge_type: megatron-bridge`` in AReaL for this model.

GLM-5.1 uses Multi-Latent Attention (MLA) like DeepSeek V3, plus DSA
(Dynamic Sparse Attention) indexer weights.
"""

import os
from collections.abc import Mapping
from functools import partial

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import (
    MegatronModelBridge,
    WeightConversionTask,
)
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel

from areal.utils import logging

logger = logging.getLogger("GLM5Bridge")

try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


# ---------------------------------------------------------------------------
# GLM-5.1 Bridge
# ---------------------------------------------------------------------------

# DSA indexer weight definitions: (megatron_name, hf_name)
_DSA_INDEXER_MAPPINGS: list[tuple[str, str]] = [
    (
        "decoder.layers.*.self_attention.wq_b.weight",
        "model.layers.*.self_attn.indexer.wq_b.weight",
    ),
    (
        "decoder.layers.*.self_attention.wk.weight",
        "model.layers.*.self_attn.indexer.wk.weight",
    ),
    (
        "decoder.layers.*.self_attention.weights_proj.weight",
        "model.layers.*.self_attn.indexer.weights_proj.weight",
    ),
    (
        "decoder.layers.*.self_attention.k_norm.weight",
        "model.layers.*.self_attn.indexer.k_norm.weight",
    ),
    (
        "decoder.layers.*.self_attention.k_norm.bias",
        "model.layers.*.self_attn.indexer.k_norm.bias",
    ),
]


def _get_rope_theta(hf_config) -> float:
    """Extract rope_theta from HF config, handling GLM-5.1's nested structure."""
    if hasattr(hf_config, "rope_parameters") and isinstance(
        hf_config.rope_parameters, dict
    ):
        return float(hf_config.rope_parameters.get("rope_theta", 10000.0))
    return float(getattr(hf_config, "rope_theta", 10000.0))


@MegatronModelBridge.register_bridge(
    source="GlmMoeDsaForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="glm_moe_dsa",
)
class GLM5Bridge(MegatronModelBridge):
    """Megatron Bridge for GLM-5.1 (GlmMoeDsa) with MLA + MoE + DSA."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Layer spec
        provider.transformer_layer_spec = partial(
            get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE
        )

        # Architecture basics
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
        provider.multi_latent_attention = True

        # MoE
        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "none"
        provider.moe_shared_expert_overlap = False
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_router_dtype = "fp32"
        provider.moe_permute_fusion = True
        provider.moe_z_loss_coeff = 3.5e-06

        # Fusions
        provider.apply_rope_fusion = False
        provider.bias_activation_fusion = True
        provider.bias_dropout_fusion = True
        provider.cross_entropy_loss_fusion = False
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True
        provider.gradient_accumulation_fusion = True

        # Misc
        provider.hidden_dropout = 0.0
        provider.attention_softmax_in_fp32 = True
        provider.disable_bf16_reduced_precision_matmul = True
        provider.make_vocab_size_divisible_by = 128
        provider.seq_length = getattr(hf_config, "max_position_embeddings", 4096)

        # Rope — GLM-5.1 stores rope_theta in rope_parameters dict
        provider.rotary_base = _get_rope_theta(hf_config)
        provider.rotary_scaling_factor = 1.0
        provider.rope_type = "rope"

        # MoE layer frequency
        provider.moe_layer_freq = [0] * hf_config.first_k_dense_replace + [1] * (
            hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        )
        provider.moe_shared_expert_intermediate_size = (
            hf_config.moe_intermediate_size * hf_config.n_shared_experts
        )

        # MTP
        mtp_num_layers = getattr(hf_config, "num_nextn_predict_layers", None)
        if os.environ.get("AREAL_DISABLE_MTP", "0") == "1" and mtp_num_layers:
            logger.warning(
                f"AREAL_DISABLE_MTP=1: overriding mtp_num_layers from {mtp_num_layers} to 0"
            )
            mtp_num_layers = 0
        provider.mtp_num_layers = mtp_num_layers

        # DSA (Dynamic Sparse Attention) — set on provider so the internal
        # TransformerConfig picks them up when creating DSAMLASelfAttention.
        if (
            getattr(hf_config, "index_topk", None) is not None
            and getattr(hf_config, "index_n_heads", None) is not None
        ):
            provider.dsa_indexer_n_heads = hf_config.index_n_heads
            provider.dsa_indexer_head_dim = hf_config.index_head_dim
            provider.dsa_indexer_topk = hf_config.index_topk
            provider.dsa_indexer_loss_coeff = getattr(
                hf_config, "dsa_indexer_loss_coeff", 0.0
            )
            provider.dsa_indexer_use_sparse_loss = getattr(
                hf_config, "dsa_indexer_use_sparse_loss", False
            )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = get_common_mapping_list()

        # Expert bias
        mapping_list.append(
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.router.expert_bias",
                hf_param="model.layers.*.mlp.gate.e_score_correction_bias",
            )
        )

        # DSA indexer weights
        for mcore_name, hf_name in _DSA_INDEXER_MAPPINGS:
            mapping_list.append(
                AutoMapping(
                    megatron_param=mcore_name,
                    hf_param=hf_name,
                )
            )

        # MTP layer mappings (if present)
        mapping_list.extend(self._get_mtp_mappings())

        return MegatronMappingRegistry(*mapping_list)

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Add rotary inv_freq to HF state dict if the original checkpoint had it."""
        global_name = task.global_param_name
        if not global_name.startswith("decoder.layers.") or not global_name.endswith(
            ".input_layernorm.weight"
        ):
            return converted_weights_dict

        parts = global_name.split(".")
        if len(parts) < 4 or not parts[2].isdigit():
            return converted_weights_dict

        inv_freq_prefix = "model.layers."
        inv_freq_suffix = ".self_attn.rotary_emb.inv_freq"
        layer_idx = int(parts[2])
        inv_freq_key = f"{inv_freq_prefix}{layer_idx}{inv_freq_suffix}"
        if inv_freq_key in converted_weights_dict:
            return converted_weights_dict

        has_inv_freq = getattr(self, "_glm5_has_inv_freq", None)
        if has_inv_freq is None:
            has_inv_freq = any(
                key.startswith(inv_freq_prefix) and key.endswith(inv_freq_suffix)
                for key in hf_state_dict.keys()
            )
            self._glm5_has_inv_freq = has_inv_freq
        if not has_inv_freq:
            return converted_weights_dict

        inv_freq = getattr(self, "_glm5_inv_freq", None)
        if inv_freq is None:
            rotary_dim = self.hf_config.qk_rope_head_dim
            rotary_base = _get_rope_theta(self.hf_config)
            inv_freq = 1.0 / (
                rotary_base
                ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
            )
            self._glm5_inv_freq = inv_freq

        if converted_weights_dict:
            ref = next(iter(converted_weights_dict.values()))
            if inv_freq.device != ref.device:
                inv_freq = inv_freq.to(device=ref.device)
                self._glm5_inv_freq = inv_freq

        converted_weights_dict[inv_freq_key] = inv_freq
        return converted_weights_dict

    # ---------------------------------------------------------------
    # MTP layer mappings
    # ---------------------------------------------------------------

    def _get_mtp_mappings(self) -> list:
        hf_config = getattr(self, "hf_config", None)
        if hf_config is None:
            return []
        num_mtp = getattr(hf_config, "num_nextn_predict_layers", 0)
        if not num_mtp or num_mtp <= 0:
            return []

        num_layers = hf_config.num_hidden_layers
        mappings: list = []

        _MTP_LAYER_MAPPINGS = {
            "mtp.layers.*.transformer_layer.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "mtp.layers.*.transformer_layer.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "mtp.layers.*.transformer_layer.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "mtp.layers.*.transformer_layer.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "mtp.layers.*.transformer_layer.self_attention.linear_kv_down_proj.weight": "model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
            "mtp.layers.*.transformer_layer.self_attention.linear_kv_up_proj.weight": "model.layers.*.self_attn.kv_b_proj.weight",
            "mtp.layers.*.transformer_layer.self_attention.linear_kv_up_proj.layer_norm_weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            "mtp.layers.*.transformer_layer.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            "mtp.layers.*.transformer_layer.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "mtp.layers.*.transformer_layer.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "mtp.layers.*.transformer_layer.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
            "mtp.layers.*.transformer_layer.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "mtp.layers.*.transformer_layer.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
            "mtp.layers.*.transformer_layer.self_attention.linear_q_down_proj.weight": "model.layers.*.self_attn.q_a_proj.weight",
            "mtp.layers.*.transformer_layer.self_attention.linear_q_up_proj.weight": "model.layers.*.self_attn.q_b_proj.weight",
            "mtp.layers.*.transformer_layer.self_attention.linear_q_up_proj.layer_norm_weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            "mtp.layers.*.transformer_layer.q_layernorm.weight": "model.layers.*.self_attn.q_a_layernorm.weight",
        }

        for mtp_idx in range(num_mtp):
            layer_idx = mtp_idx + num_layers

            # MTP-specific weights
            mappings.extend(
                [
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_idx}.enorm.weight",
                        hf_param=f"model.layers.{layer_idx}.enorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_idx}.hnorm.weight",
                        hf_param=f"model.layers.{layer_idx}.hnorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_idx}.eh_proj.weight",
                        hf_param=f"model.layers.{layer_idx}.eh_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_idx}.final_layernorm.weight",
                        hf_param=f"model.layers.{layer_idx}.shared_head.norm.weight",
                    ),
                ]
            )

            # Standard layer mappings adapted for MTP
            for mcore_pat, hf_pat in _MTP_LAYER_MAPPINGS.items():
                mappings.append(
                    AutoMapping(
                        megatron_param=mcore_pat.replace("*", str(mtp_idx), 1),
                        hf_param=hf_pat.replace("*", str(layer_idx), 1),
                    )
                )

            # GatedMLP for MTP
            mappings.extend(
                [
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_idx}.transformer_layer.mlp.linear_fc1.weight",
                        gate=f"model.layers.{layer_idx}.mlp.gate_proj.weight",
                        up=f"model.layers.{layer_idx}.mlp.up_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_idx}.transformer_layer.mlp.experts.linear_fc1.weight*",
                        gate=f"model.layers.{layer_idx}.mlp.experts.*.gate_proj.weight",
                        up=f"model.layers.{layer_idx}.mlp.experts.*.up_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_idx}.transformer_layer.mlp.shared_experts.linear_fc1.weight",
                        gate=f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight",
                        up=f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight",
                    ),
                ]
            )

            # DSA indexer weights for MTP layers
            for mcore_pat, hf_pat in _DSA_INDEXER_MAPPINGS:
                mappings.append(
                    AutoMapping(
                        megatron_param=mcore_pat.replace(
                            "decoder.layers.*",
                            f"mtp.layers.{mtp_idx}.transformer_layer",
                        ),
                        hf_param=hf_pat.replace("layers.*", f"layers.{layer_idx}"),
                    )
                )

        return mappings

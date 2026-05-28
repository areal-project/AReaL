# SPDX-License-Identifier: Apache-2.0

"""mbridge Bridge for DeepSeek V3 / GLM-5.1 / GLM-4.7-Flash.

Registers with mbridge so that MegatronEngine.initialize() can use AutoBridge
to load and manage DeepSeek V3 / GLM-5.1 / GLM-4.7-Flash models with
homogeneous MLA attention and MoE layers.

Key differences from BailingMoeBridge:
- All layers use MLA (no Lightning Attention heterogeneity)
- q_lora_rank is always non-None (Q uses low-rank decomposition)
- No fused QKV weight conversion needed
- HF uses 'self_attn' prefix (not 'attention') and 'o_proj' (not 'dense')
- HF embedding key is 'model.embed_tokens.weight' (not 'model.word_embeddings.weight')
- MoE expert count field is 'n_routed_experts' (not 'num_experts')
- scoring_func defaults to sigmoid
- GLM-5.1 adds DSA indexer weights (wq_b, wk, k_norm, weights_proj)

Note: GLM-4.7-Flash has num_nextn_predict_layers=1 with extra weights at
layer index num_hidden_layers. These weights are automatically ignored since
megatron-core only builds num_hidden_layers transformer layers.
"""

import os

import torch
from mbridge.core import LLMBridge, register_model
from megatron.core.transformer import MLATransformerConfig
from megatron.core.transformer.enums import AttnBackend

from areal.models.mcore.deepseek_v3 import make_mcore_layer_specs_deepseek_v3
from areal.utils import logging

logger = logging.getLogger("DeepSeekV3Bridge")

# MLA Q-LoRA mapping (mcore suffix -> HF name templates)
# DeepSeek V3 / GLM-5.1 always uses Q-LoRA (q_lora_rank=1536)
_MLA_Q_LORA_MAPPING = {
    "self_attention.linear_q_down_proj.weight": [
        "model.layers.{layer_number}.self_attn.q_a_proj.weight"
    ],
    "self_attention.linear_q_up_proj.layer_norm_weight": [
        "model.layers.{layer_number}.self_attn.q_a_layernorm.weight"
    ],
    "self_attention.linear_q_up_proj.weight": [
        "model.layers.{layer_number}.self_attn.q_b_proj.weight"
    ],
}

# MLA KV compression + output projection mapping
_MLA_COMMON_MAPPING = {
    "input_layernorm.weight": ["model.layers.{layer_number}.input_layernorm.weight"],
    "self_attention.linear_kv_down_proj.weight": [
        "model.layers.{layer_number}.self_attn.kv_a_proj_with_mqa.weight"
    ],
    "self_attention.linear_kv_up_proj.layer_norm_weight": [
        "model.layers.{layer_number}.self_attn.kv_a_layernorm.weight"
    ],
    "self_attention.linear_kv_up_proj.weight": [
        "model.layers.{layer_number}.self_attn.kv_b_proj.weight"
    ],
    "self_attention.linear_proj.weight": [
        "model.layers.{layer_number}.self_attn.o_proj.weight"
    ],
}

# Combined MLA attention mapping (always Q-LoRA for DeepSeek V3)
_MLA_ATTENTION_MAPPING = {**_MLA_COMMON_MAPPING, **_MLA_Q_LORA_MAPPING}

# DSA indexer weight mapping (GLM-5.1 only)
# slime-style DSAMLASelfAttention attaches indexer submodules directly on the
# self-attention module (NOT inside a core_attention.indexer container), using
# bare names wq_b / wk / k_norm / weights_proj. LayerNorm k_norm has weight+bias.
_DSA_INDEXER_MAPPING = {
    "self_attention.wq_b.weight": [
        "model.layers.{layer_number}.self_attn.indexer.wq_b.weight"
    ],
    "self_attention.wk.weight": [
        "model.layers.{layer_number}.self_attn.indexer.wk.weight"
    ],
    "self_attention.k_norm.weight": [
        "model.layers.{layer_number}.self_attn.indexer.k_norm.weight"
    ],
    "self_attention.k_norm.bias": [
        "model.layers.{layer_number}.self_attn.indexer.k_norm.bias"
    ],
    "self_attention.weights_proj.weight": [
        "model.layers.{layer_number}.self_attn.indexer.weights_proj.weight"
    ],
}


@register_model("deepseek_v3")
@register_model("glm_moe_dsa")
@register_model("glm4_moe_lite")
class DeepSeekV3Bridge(LLMBridge):
    """Bridge for DeepSeek V3 / GLM-5.1 with homogeneous MLA + MoE."""

    TransformerConfigClass = MLATransformerConfig

    @property
    def _has_dsa_indexer(self) -> bool:
        # slime-style native DSA module: wq_b / wk / k_norm / weights_proj are
        # attached directly on DSAMLASelfAttention (see dsa_mla_attention.py),
        # NOT inside a DSAttention container. Enabled when HF config exposes
        # index_topk + index_n_heads.
        return (
            getattr(self.hf_config, "index_topk", None) is not None
            and getattr(self.hf_config, "index_n_heads", None) is not None
        )

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _MLP_MAPPING = {
        # Dense MLP (layers < first_k_dense_replace)
        "mlp.linear_fc1.layer_norm_weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        # MoE shared experts
        "mlp.shared_experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.shared_experts.down_proj.weight"
        ],
        "mlp.shared_experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.shared_experts.gate_proj.weight",
            "model.layers.{layer_number}.mlp.shared_experts.up_proj.weight",
        ],
        # MoE pre-MLP layernorm
        "pre_mlp_layernorm.weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        # MoE router
        "mlp.router.weight": ["model.layers.{layer_number}.mlp.gate.weight"],
        "mlp.router.expert_bias": [
            "model.layers.{layer_number}.mlp.gate.e_score_correction_bias"
        ],
        # MoE experts
        "mlp.experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
            "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
        ],
        "mlp.experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"
        ],
    }

    def _build_config(self):
        hf_config = self.hf_config

        # Build moe_layer_freq: 0 for dense, 1 for MoE
        num_layers = hf_config.num_hidden_layers
        first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 3)
        moe_layer_freq = [
            0 if i < first_k_dense_replace else 1 for i in range(num_layers)
        ]

        # Shared expert intermediate size
        n_shared_experts = getattr(hf_config, "n_shared_experts", 0)
        moe_intermediate_size = getattr(
            hf_config, "moe_intermediate_size", hf_config.intermediate_size
        )
        shared_expert_intermediate_size = (
            n_shared_experts * moe_intermediate_size if n_shared_experts > 0 else None
        )

        # Number of routed experts
        n_routed_experts = getattr(hf_config, "n_routed_experts", None)
        if n_routed_experts is None:
            n_routed_experts = getattr(hf_config, "num_local_experts", None)

        # YaRN RoPE scaling parameters
        # G20 (2026-05-08): only pass mscale/mscale_all_dim when rope_scaling
        # actually requests YaRN (rope_type=='yarn'). GLM-5.1 has
        # rope_scaling={'rope_theta':..., 'rope_type':'default'} which is a
        # truthy dict but NOT YaRN — previously we set mscale=0.707 default,
        # which 0.5x'd softmax_scale and caused 7x SFT loss vs sglang.
        rope_scaling_dict = getattr(hf_config, "rope_scaling", None) or {}
        rotary_scaling_factor = rope_scaling_dict.get("factor", 1.0)

        # rope_theta: top-level field or inside rope_parameters (GLM-5.1)
        rope_theta = getattr(hf_config, "rope_theta", None)
        if rope_theta is None:
            rope_params = getattr(hf_config, "rope_parameters", None) or {}
            rope_theta = rope_params.get("rope_theta", 10000.0)

        mscale_kwargs = {}
        if (
            rope_scaling_dict.get("rope_type") == "yarn"
            or rope_scaling_dict.get("type") == "yarn"
        ):
            mscale_kwargs["mscale"] = rope_scaling_dict.get("mscale", 0.707)
            mscale_kwargs["mscale_all_dim"] = rope_scaling_dict.get(
                "mscale_all_dim", 0.707
            )

        return self._build_base_config(
            attention_backend=AttnBackend.fused,
            layernorm_epsilon=hf_config.rms_norm_eps,
            ffn_hidden_size=hf_config.intermediate_size,
            qk_layernorm=True,
            # MLA parameters
            multi_latent_attention=True,
            q_lora_rank=getattr(hf_config, "q_lora_rank", None),
            kv_lora_rank=getattr(hf_config, "kv_lora_rank", 512),
            qk_head_dim=getattr(hf_config, "qk_nope_head_dim", 128),
            qk_pos_emb_head_dim=getattr(hf_config, "qk_rope_head_dim", 64),
            v_head_dim=getattr(hf_config, "v_head_dim", 128),
            rotary_base=rope_theta,
            rope_type="rope",
            rotary_percent=getattr(hf_config, "partial_rotary_factor", 1.0),
            rotary_scaling_factor=rotary_scaling_factor,
            apply_rope_fusion=False,
            **mscale_kwargs,
            # MoE parameters
            moe_ffn_hidden_size=moe_intermediate_size,
            moe_token_dispatcher_type="alltoall",
            moe_router_enable_expert_bias=True,
            moe_router_topk=getattr(hf_config, "num_experts_per_tok", 8),
            num_moe_experts=n_routed_experts,
            moe_shared_expert_intermediate_size=shared_expert_intermediate_size,
            moe_router_score_function="sigmoid",
            moe_router_num_groups=getattr(hf_config, "n_group", 8),
            moe_router_group_topk=getattr(hf_config, "topk_group", 4),
            moe_router_topk_scaling_factor=getattr(
                hf_config, "routed_scaling_factor", None
            ),
            moe_router_load_balancing_type="none",
            moe_grouped_gemm=True,
            moe_layer_freq=moe_layer_freq,
            moe_router_dtype="fp32",
            moe_router_bias_update_rate=0.0,
            moe_z_loss_coeff=3.5e-6,
            moe_enable_routing_replay=bool(os.environ.get("AREAL_DUMP_ROUTING", "")),
            # Other
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # DSA parameters (GLM-5.1).
            # NOTE: do NOT set experimental_attention_variant="dsa" — we use a
            # slime-style custom self_attention spec (DSAMLASelfAttention) and
            # bypass mcore's own DSA paths.
            **(
                {
                    "dsa_indexer_n_heads": hf_config.index_n_heads,
                    "dsa_indexer_head_dim": hf_config.index_head_dim,
                    "dsa_indexer_topk": hf_config.index_topk,
                    "dsa_indexer_loss_coeff": getattr(
                        hf_config, "dsa_indexer_loss_coeff", 0.0
                    ),
                    "dsa_indexer_use_sparse_loss": getattr(
                        hf_config, "dsa_indexer_use_sparse_loss", False
                    ),
                }
                if self._has_dsa_indexer
                else {}
            ),
        )

    def _get_gptmodel_args(self) -> dict:
        rope_theta = getattr(self.hf_config, "rope_theta", None)
        if rope_theta is None:
            rope_params = getattr(self.hf_config, "rope_parameters", None) or {}
            rope_theta = rope_params.get("rope_theta", 10000.0)
        return dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=rope_theta,
        )

    def _get_transformer_layer_spec(self, vp_stage: int | None = None):
        """Return homogeneous MLA layer specs (all layers use MLA).

        PP slicing is handled inside make_mcore_layer_specs_deepseek_v3.
        """
        assert self.config.normalization == "RMSNorm"
        self.has_vp_stage = vp_stage is not None
        return make_mcore_layer_specs_deepseek_v3(
            self.config, self.hf_config, use_te=True, vp_stage=vp_stage
        )

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        assert "_extra_state" not in mcore_weights_name

        if mcore_weights_name in self._DIRECT_MAPPING:
            return [self._DIRECT_MAPPING[mcore_weights_name]]

        if (
            "self_attention" in mcore_weights_name
            or "input_layernorm.weight" in mcore_weights_name
        ):
            return self._weight_name_mapping_attention(mcore_weights_name)
        elif "mlp" in mcore_weights_name or "pre_mlp_layernorm" in mcore_weights_name:
            return self._weight_name_mapping_mlp(mcore_weights_name)
        else:
            raise NotImplementedError(
                f"Unsupported parameter name: {mcore_weights_name}"
            )

    def _weight_merge_across_tp(
        self,
        mcore_weights_name: str,
        tp_shards: list[torch.Tensor],
        param: torch.Tensor,
    ) -> torch.Tensor:
        """Handle MLA and DSA duplicated weights.

        linear_q_down_proj and linear_kv_down_proj use parallel_mode='duplicated'
        in megatron-core MLA — they are replicated (not sharded) across TP ranks.
        DSA indexer weights (wq_b, wk, k_norm, weights_proj) live directly under
        self_attention in our slime-style DSAMLASelfAttention and are also
        duplicated via parallel_mode='duplicated'.
        All shards are identical, so just return the first one.
        """
        if (
            "linear_q_down_proj." in mcore_weights_name
            or "linear_kv_down_proj." in mcore_weights_name
            or "self_attention.wq_b." in mcore_weights_name
            or "self_attention.wk." in mcore_weights_name
            or "self_attention.k_norm." in mcore_weights_name
            or "self_attention.weights_proj." in mcore_weights_name
        ):
            return tp_shards[0].clone()
        return super()._weight_merge_across_tp(mcore_weights_name, tp_shards, param)

    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        """Map MLA attention weights. All layers use MLA (no heterogeneous dispatch).

        For GLM-5.1, also handles DSA indexer weights.
        """
        layer_number_str = name.split(".")[2]

        # Check DSA indexer mappings first
        mapping = _MLA_ATTENTION_MAPPING
        if self._has_dsa_indexer:
            mapping = {**mapping, **_DSA_INDEXER_MAPPING}

        convert_names = []
        for keyword, mapping_names in mapping.items():
            if keyword in name:
                convert_names.extend(
                    [x.format(layer_number=layer_number_str) for x in mapping_names]
                )
                break

        if not convert_names:
            raise NotImplementedError(f"Unsupported attention parameter: {name}")
        return convert_names

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                if "{expert_id}" in mapping_names[0]:
                    expert_id = name.split("weight")[-1]
                    convert_names.extend(
                        [
                            x.format(layer_number=layer_number, expert_id=expert_id)
                            for x in mapping_names
                        ]
                    )
                else:
                    convert_names.extend(
                        [x.format(layer_number=layer_number) for x in mapping_names]
                    )
                break
        if not convert_names:
            raise NotImplementedError(f"Unsupported MLP parameter: {name}")
        return convert_names

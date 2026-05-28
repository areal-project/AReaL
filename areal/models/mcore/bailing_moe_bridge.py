# SPDX-License-Identifier: Apache-2.0

"""mbridge Bridge for BailingMoeV2.5 (bailing_moe_linear model_type).

Registers with mbridge so that MegatronEngine.initialize() can use AutoBridge
to load and manage BailingMoeV2.5 models with heterogeneous attention layers
(Lightning Attention + MLA).
"""

import torch
from mbridge.core import LLMBridge, register_model
from megatron.core.transformer import MLATransformerConfig
from megatron.core.transformer.enums import AttnBackend

from areal.models.mcore.bailing_moe import (
    is_lightning_layer,
    make_mcore_layer_specs_bailing_moe,
)
from areal.utils import logging

logger = logging.getLogger("BailingMoeBridge")

# Lightning Attention mcore suffix -> HF name templates
_LIGHTNING_ATTENTION_MAPPING = {
    "input_layernorm.weight": ["model.layers.{layer_number}.input_layernorm.weight"],
    "self_attention.linear_qkv.weight": [
        "model.layers.{layer_number}.attention.query_key_value.weight"
    ],
    "self_attention.linear_proj.weight": [
        "model.layers.{layer_number}.attention.dense.weight"
    ],
    "self_attention.linear_gate.weight": [
        "model.layers.{layer_number}.attention.g_proj.weight"
    ],
    "self_attention.gate_norm.weight": [
        "model.layers.{layer_number}.attention.g_norm.weight"
    ],
    "self_attention.q_layernorm.weight": [
        "model.layers.{layer_number}.attention.query_layernorm.weight"
    ],
    "self_attention.k_layernorm.weight": [
        "model.layers.{layer_number}.attention.key_layernorm.weight"
    ],
}

# MLA mcore suffix -> HF name templates
# q_lora_rank=None: uses linear_q_proj (direct Q projection)
# q_lora_rank!=None: uses linear_q_down_proj + linear_q_up_proj (low-rank Q decomposition)
_MLA_ATTENTION_MAPPING_Q_DIRECT = {
    "self_attention.linear_q_proj.weight": [
        "model.layers.{layer_number}.attention.q_proj.weight"
    ],
}

_MLA_ATTENTION_MAPPING_Q_LORA = {
    "self_attention.linear_q_down_proj.weight": [
        "model.layers.{layer_number}.attention.q_a_proj.weight"
    ],
    "self_attention.linear_q_up_proj.layer_norm_weight": [
        "model.layers.{layer_number}.attention.q_a_layernorm.weight"
    ],
    "self_attention.linear_q_up_proj.weight": [
        "model.layers.{layer_number}.attention.q_b_proj.weight"
    ],
}

_MLA_ATTENTION_MAPPING_COMMON = {
    "input_layernorm.weight": ["model.layers.{layer_number}.input_layernorm.weight"],
    "self_attention.linear_kv_down_proj.weight": [
        "model.layers.{layer_number}.attention.kv_a_proj_with_mqa.weight"
    ],
    "self_attention.linear_kv_up_proj.layer_norm_weight": [
        "model.layers.{layer_number}.attention.kv_a_layernorm.weight"
    ],
    "self_attention.linear_kv_up_proj.weight": [
        "model.layers.{layer_number}.attention.kv_b_proj.weight"
    ],
    "self_attention.linear_proj.weight": [
        "model.layers.{layer_number}.attention.dense.weight"
    ],
}


@register_model("bailing_moe_v2")
@register_model("bailing_moe_linear")
@register_model("bailing_hybrid")
class BailingMoeBridge(LLMBridge):
    """Bridge for BailingMoeV2.5 with heterogeneous Lightning + MLA attention."""

    TransformerConfigClass = MLATransformerConfig

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.word_embeddings.weight",
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
        "mlp.router.expert_bias": ["model.layers.{layer_number}.mlp.gate.expert_bias"],
        # MoE experts
        "mlp.experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
            "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
        ],
        "mlp.experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"
        ],
    }

    # _ATTENTION_MAPPING is not used directly; we override _weight_name_mapping_attention

    def _build_config(self):
        hf_config = self.hf_config

        # Build moe_layer_freq
        num_layers = hf_config.num_hidden_layers
        first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 0)
        moe_layer_freq = [
            0 if i < first_k_dense_replace else 1 for i in range(num_layers)
        ]

        # Shared expert intermediate size
        shared_expert_intermediate_size = getattr(
            hf_config, "moe_shared_expert_intermediate_size", None
        )
        if shared_expert_intermediate_size is None:
            num_shared_experts = getattr(hf_config, "num_shared_experts", 0)
            if num_shared_experts > 0:
                shared_expert_intermediate_size = num_shared_experts * getattr(
                    hf_config, "moe_intermediate_size", hf_config.intermediate_size
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
            rotary_base=(getattr(hf_config, "rope_parameters", None) or {}).get(
                "rope_theta", getattr(hf_config, "rope_theta", 10000.0)
            ),
            rope_type="rope",
            rotary_percent=1.0,
            rotary_scaling_factor=(getattr(hf_config, "rope_scaling", None) or {}).get(
                "factor", 1.0
            ),
            apply_rope_fusion=False,
            # MoE parameters
            moe_ffn_hidden_size=getattr(hf_config, "moe_intermediate_size", None),
            moe_token_dispatcher_type="alltoall",
            moe_router_enable_expert_bias=True,
            moe_router_topk=getattr(hf_config, "num_experts_per_tok", 8),
            num_moe_experts=getattr(hf_config, "num_experts", None),
            moe_shared_expert_intermediate_size=shared_expert_intermediate_size,
            moe_router_score_function=getattr(hf_config, "scoring_func", "sigmoid"),
            moe_router_num_groups=getattr(hf_config, "n_group", 8),
            moe_router_group_topk=getattr(hf_config, "topk_group", 4),
            moe_router_topk_scaling_factor=getattr(
                hf_config, "routed_scaling_factor", None
            ),
            moe_router_load_balancing_type="none",
            moe_grouped_gemm=True,
            moe_layer_freq=moe_layer_freq,
            # Other
            moe_router_dtype="fp32",
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
        )

    def _get_gptmodel_args(self) -> dict:
        return dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=getattr(self.hf_config, "rope_theta", 10000.0),
        )

    def _get_transformer_layer_spec(self, vp_stage: int | None = None):
        """Override to return heterogeneous layer specs (Lightning + MLA).

        PP slicing is handled inside make_mcore_layer_specs_bailing_moe via
        get_num_layers_to_build() and get_transformer_layer_offset(), matching
        the slicing logic in get_gpt_decoder_block_spec().

        VPP (virtual pipeline parallelism) is not supported for BailingMoe.
        """
        assert self.config.normalization == "RMSNorm"
        self.has_vp_stage = False  # VPP not supported
        return make_mcore_layer_specs_bailing_moe(
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
        """Override to handle MLA duplicated weights.

        linear_q_down_proj and linear_kv_down_proj use parallel_mode='duplicated'
        in megatron-core MLA — they are replicated (not sharded) across TP ranks.
        All shards are identical, so just return the first one.
        """
        if (
            "linear_q_down_proj." in mcore_weights_name
            or "linear_kv_down_proj." in mcore_weights_name
        ):
            return tp_shards[0].clone()
        return super()._weight_merge_across_tp(mcore_weights_name, tp_shards, param)

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> torch.Tensor:
        """Convert HF weights to mcore format.

        For Lightning Attention layers, the fused QKV weight needs to be
        converted from HF concatenated [Q|K|V] format to megatron per-head
        interleaved [q0,k0,v0|q1,k1,v1|...] format. This is required for
        correct TP splitting and QKV extraction in the forward pass.
        """
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
            and len(hf_weights) == 1
        ):
            # Single fused QKV weight — this is a Lightning Attention layer.
            # HF format: [3*H*D, hidden] with layout [Q_all | K_all | V_all]
            # mcore format: [H*3*D, hidden] with layout [q0,k0,v0 | q1,k1,v1 | ...]
            weight = hf_weights[0]
            logger.info(
                f"Converting Lightning QKV weight: {mcore_weights_name} "
                f"shape={weight.shape} from [3,H,D] to [H,3,D] format"
            )
            num_heads = self.hf_config.num_attention_heads
            head_dim = getattr(
                self.hf_config,
                "head_dim",
                self.hf_config.hidden_size // num_heads,
            )
            is_bias = ".bias" in mcore_weights_name
            if is_bias:
                # bias shape: [3*H*D]
                weight = weight.view(3, num_heads, head_dim)
                weight = weight.permute(1, 0, 2).contiguous()
                return weight.view(-1)
            else:
                # weight shape: [3*H*D, hidden_size]
                hidden_size = weight.shape[1]
                weight = weight.view(3, num_heads, head_dim, hidden_size)
                weight = weight.permute(1, 0, 2, 3).contiguous()
                return weight.view(num_heads * 3 * head_dim, hidden_size)

        return super()._weight_to_mcore_format(mcore_weights_name, hf_weights)

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """Convert mcore weights to HF format.

        For Lightning Attention layers, convert fused QKV weight back from
        mcore per-head interleaved [q0,k0,v0|...] to HF concatenated [Q|K|V].
        """
        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        if (
            "self_attention.linear_qkv." in mcore_weights_name
            and "layer_norm" not in mcore_weights_name
            and len(hf_names) == 1
        ):
            # Single HF name = Lightning Attention fused QKV.
            # mcore format: [H*3*D, hidden] → HF format: [3*H*D, hidden]
            weight = mcore_weights
            num_heads = self.hf_config.num_attention_heads
            head_dim = getattr(
                self.hf_config,
                "head_dim",
                self.hf_config.hidden_size // num_heads,
            )
            is_bias = ".bias" in mcore_weights_name
            if is_bias:
                weight = weight.view(num_heads, 3, head_dim)
                weight = weight.permute(1, 0, 2).contiguous()
                return hf_names, [weight.view(-1)]
            else:
                hidden_size = weight.shape[1]
                weight = weight.view(num_heads, 3, head_dim, hidden_size)
                weight = weight.permute(1, 0, 2, 3).contiguous()
                return hf_names, [weight.view(3 * num_heads * head_dim, hidden_size)]

        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        """Dispatch to Lightning or MLA mapping based on layer number."""
        layer_number_str = name.split(".")[2]
        layer_number = int(layer_number_str)
        layer_group_size = getattr(self.hf_config, "layer_group_size", 1)

        if is_lightning_layer(layer_number, layer_group_size):
            mapping = _LIGHTNING_ATTENTION_MAPPING
        else:
            # Build MLA mapping based on q_lora_rank
            q_lora_rank = getattr(self.hf_config, "q_lora_rank", None)
            q_mapping = (
                _MLA_ATTENTION_MAPPING_Q_LORA
                if q_lora_rank is not None
                else _MLA_ATTENTION_MAPPING_Q_DIRECT
            )
            mapping = {**_MLA_ATTENTION_MAPPING_COMMON, **q_mapping}

        convert_names = []
        for keyword, mapping_names in mapping.items():
            if keyword in name:
                convert_names.extend(
                    [x.format(layer_number=layer_number_str) for x in mapping_names]
                )
                break

        if not convert_names:
            raise NotImplementedError(
                f"Unsupported attention parameter: {name} "
                f"(lightning={is_lightning_layer(layer_number, layer_group_size)})"
            )
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

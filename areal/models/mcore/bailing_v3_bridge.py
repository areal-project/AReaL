# SPDX-License-Identifier: Apache-2.0

"""mbridge Bridge for BailingMoeV3 (Ling V3, model_type bailing_moe_v3).

Registers with mbridge so MegatronEngine.initialize() can use AutoBridge to load and
manage BailingMoeV3 models with heterogeneous attention (KDA + gated MLA).

HF weight names follow the antllm BailingMoeV3 checkpoint layout (verified against
HybridEngine's bailing_moe_huggingface_ckpt_conversion.py:
``transform_kda_attention_and_layernorm_weights``):

KDA layers (no_kda_lora=True):
    self_attention.in_proj.weight   -> attention.{q,k,v,f,g}_proj.weight  (split q|k|v|g|gate)
    self_attention.conv1d.weight    -> attention.{q,k,v}_conv1d.weight    (split q|k|v)
    self_attention.beta_proj.weight -> attention.b_proj.weight
    self_attention.out_norm.weight  -> attention.o_norm.weight
    self_attention.out_proj.weight  -> attention.o_proj.weight
    self_attention.dt_bias          -> attention.dt_bias
    self_attention.A_log            -> attention.A_log
    input_layernorm.weight          -> input_layernorm.weight
MLA layers (gated):  same as V2.5 MLA, plus
    self_attention.linear_gate.weight -> attention.g_proj.weight

The fused-in-proj / fused-conv1d split (multiple HF tensors -> one mcore tensor) is
handled on the load path by areal/models/mcore/hf_load.py (KDA branch in
_weight_to_mcore_tp). The reverse split for save is in _weight_to_hf_format below.
"""

import torch
from mbridge.core import LLMBridge, register_model
from megatron.core.transformer import MLATransformerConfig
from megatron.core.transformer.enums import AttnBackend

from areal.models.mcore.bailing_v3 import (
    is_kda_layer,
    make_mcore_layer_specs_bailing_v3,
)
from areal.utils import logging

logger = logging.getLogger("BailingV3Bridge")

# KDA (linear-attention) mcore suffix -> HF name templates.
# NOTE: mcore in_proj split order is [query, key, value, g, gate]; HF names are
# [q_proj, k_proj, v_proj, f_proj, g_proj] (mcore "g" -> HF f_proj, mcore "gate" -> HF g_proj).
_KDA_ATTENTION_MAPPING = {
    "input_layernorm.weight": ["model.layers.{layer_number}.input_layernorm.weight"],
    "self_attention.in_proj.weight": [
        "model.layers.{layer_number}.attention.q_proj.weight",
        "model.layers.{layer_number}.attention.k_proj.weight",
        "model.layers.{layer_number}.attention.v_proj.weight",
        "model.layers.{layer_number}.attention.f_proj.weight",
        "model.layers.{layer_number}.attention.g_proj.weight",
    ],
    "self_attention.conv1d.weight": [
        "model.layers.{layer_number}.attention.q_conv1d.weight",
        "model.layers.{layer_number}.attention.k_conv1d.weight",
        "model.layers.{layer_number}.attention.v_conv1d.weight",
    ],
    "self_attention.beta_proj.weight": [
        "model.layers.{layer_number}.attention.b_proj.weight"
    ],
    "self_attention.out_norm.weight": [
        "model.layers.{layer_number}.attention.o_norm.weight"
    ],
    "self_attention.out_proj.weight": [
        "model.layers.{layer_number}.attention.o_proj.weight"
    ],
    "self_attention.dt_bias": ["model.layers.{layer_number}.attention.dt_bias"],
    "self_attention.A_log": ["model.layers.{layer_number}.attention.A_log"],
}

# Gated-MLA mcore suffix -> HF name templates.
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
    # v3 head-wise gated attention
    "self_attention.linear_gate.weight": [
        "model.layers.{layer_number}.attention.g_proj.weight"
    ],
}


@register_model("bailing_moe_v3")
class BailingV3Bridge(LLMBridge):
    """Bridge for BailingMoeV3 with heterogeneous KDA + gated MLA attention."""

    TransformerConfigClass = MLATransformerConfig

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.word_embeddings.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.layer_norm_weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.shared_experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.shared_experts.down_proj.weight"
        ],
        "mlp.shared_experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.shared_experts.gate_proj.weight",
            "model.layers.{layer_number}.mlp.shared_experts.up_proj.weight",
        ],
        "pre_mlp_layernorm.weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight"
        ],
        "mlp.router.weight": ["model.layers.{layer_number}.mlp.gate.weight"],
        "mlp.router.expert_bias": ["model.layers.{layer_number}.mlp.gate.expert_bias"],
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
        num_layers = hf_config.num_hidden_layers
        first_k_dense_replace = getattr(hf_config, "first_k_dense_replace", 0)
        moe_layer_freq = [
            0 if i < first_k_dense_replace else 1 for i in range(num_layers)
        ]
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
            # MLA parameters (rotary_percent=1.0: qk_pos_emb_head_dim is the pure RoPE
            # slice in AReaL's mcore MLA; see bailing_v3.hf_to_mcore_config_bailing_v3).
            multi_latent_attention=True,
            q_lora_rank=getattr(hf_config, "q_lora_rank", None),
            kv_lora_rank=getattr(hf_config, "kv_lora_rank", 512),
            qk_head_dim=getattr(hf_config, "qk_nope_head_dim", 128),
            qk_pos_emb_head_dim=getattr(hf_config, "qk_rope_head_dim", 64),
            v_head_dim=getattr(hf_config, "v_head_dim", 128),
            rotary_base=getattr(hf_config, "rope_theta", 10000.0),
            rope_type="rope",
            rotary_percent=1.0,
            rotary_scaling_factor=(getattr(hf_config, "rope_scaling", None) or {}).get(
                "factor", 1.0
            ),
            apply_rope_fusion=False,
            # Keep in sync with bailing_v3.hf_to_mcore_config_bailing_v3: pin the
            # YaRN mscale knobs so the softmax scale does not depend on the mcore
            # (or HybridEngine-fork) MLATransformerConfig defaults. Inert while
            # rotary_scaling_factor == 1.0, decisive for rope-scaled checkpoints.
            mscale=0.707,
            mscale_all_dim=0.707,
            original_max_position_embeddings=(
                (getattr(hf_config, "rope_scaling", None) or {}).get(
                    "original_max_position_embeddings"
                )
                or getattr(hf_config, "original_max_position_embeddings", None)
                or getattr(hf_config, "max_position_embeddings", 4096)
            ),
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
            moe_router_dtype="fp32",
            moe_router_bias_update_rate=getattr(
                hf_config, "router_bias_update_speed", 0.0
            ),
            moe_z_loss_coeff=3.5e-6,
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
        """Return heterogeneous layer specs (KDA + gated MLA). VPP is not supported."""
        if vp_stage is not None:
            raise ValueError(
                "BailingMoeV3 does not support virtual pipeline parallel stages."
            )
        assert self.config.normalization == "RMSNorm"
        self.has_vp_stage = False
        return make_mcore_layer_specs_bailing_v3(
            self.config, self.hf_config, use_te=True, vp_stage=vp_stage
        )

    # ------------------------------------------------------------------
    # Weight name mapping
    # ------------------------------------------------------------------
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

    def _weight_name_mapping_attention(self, name: str) -> list[str]:
        """Dispatch to KDA or gated-MLA mapping based on the layer index."""
        layer_number_str = name.split(".")[2]
        layer_number = int(layer_number_str)
        layer_group_size = getattr(self.hf_config, "layer_group_size", 1)

        if is_kda_layer(
            layer_number,
            layer_group_size,
            getattr(self.hf_config, "num_hidden_layers", None),
        ):
            mapping = _KDA_ATTENTION_MAPPING
        else:
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
            is_kda = is_kda_layer(
                layer_number,
                layer_group_size,
                getattr(self.hf_config, "num_hidden_layers", None),
            )
            raise NotImplementedError(
                f"Unsupported attention parameter: {name} (kda={is_kda})"
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

    def _weight_merge_across_tp(
        self,
        mcore_weights_name: str,
        tp_shards: list[torch.Tensor],
        param: torch.Tensor,
    ) -> torch.Tensor:
        """MLA's linear_q_down_proj / linear_kv_down_proj are replicated across TP."""
        if (
            "linear_q_down_proj." in mcore_weights_name
            or "linear_kv_down_proj." in mcore_weights_name
        ):
            return tp_shards[0].clone()
        return super()._weight_merge_across_tp(mcore_weights_name, tp_shards, param)

    # ------------------------------------------------------------------
    # KDA fused-weight split for the save path (mcore -> HF)
    # ------------------------------------------------------------------
    def _kda_split_sections(self, kind: str) -> list[int]:
        """Global split sections for KDA fused tensors (no_kda_lora layout)."""
        from areal.models.mcore.bailing_v3 import _kda_head_dim

        head_dim = _kda_head_dim(self.hf_config)
        num_heads = self.hf_config.num_attention_heads
        qk_dim = head_dim * num_heads
        v_dim = head_dim * num_heads
        if kind == "in_proj":
            # [query, key, value, g(->f_proj), gate(->g_proj)]
            return [qk_dim, qk_dim, v_dim, qk_dim, v_dim]
        elif kind == "conv1d":
            return [qk_dim, qk_dim, v_dim]
        raise ValueError(kind)

    def _deinterleave_and_split(
        self, tensor: torch.Tensor, sections: list[int]
    ) -> list[torch.Tensor]:
        """Split a TP-merged (rank-interleaved) fused tensor into full components.

        After the default cross-TP merge, a fused ColumnParallel weight is laid out
        rank-major: [c0_r0, c1_r0, ..., c0_r1, c1_r1, ...]. Deinterleave back into
        contiguous full components [c0_full, c1_full, ...]. For tp_size==1 this is a
        plain split.
        """
        from megatron.core import parallel_state as mpu

        try:
            tp_size = mpu.get_tensor_model_parallel_world_size()
        except (RuntimeError, AssertionError):
            tp_size = 1
        if tp_size <= 1:
            return list(torch.split(tensor, sections, dim=0))

        per_rank = [s // tp_size for s in sections]
        components = [[] for _ in sections]
        for chunk in torch.split(tensor, sum(per_rank), dim=0):
            for i, part in enumerate(torch.split(chunk, per_rank, dim=0)):
                components[i].append(part)
        return [torch.cat(c, dim=0).contiguous() for c in components]

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        """Convert mcore weights to HF format, splitting KDA fused in_proj / conv1d."""
        hf_names = self._weight_name_mapping_mcore_to_hf(mcore_weights_name)
        if "self_attention.in_proj.weight" in mcore_weights_name:
            comps = self._deinterleave_and_split(
                mcore_weights, self._kda_split_sections("in_proj")
            )
            return hf_names, comps
        if "self_attention.conv1d.weight" in mcore_weights_name:
            # conv1d weight is [conv_dim, 1, kernel]; split on dim 0.
            comps = self._deinterleave_and_split(
                mcore_weights, self._kda_split_sections("conv1d")
            )
            return hf_names, comps
        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

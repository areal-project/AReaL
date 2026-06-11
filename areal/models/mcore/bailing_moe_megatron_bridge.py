# SPDX-License-Identifier: Apache-2.0

"""Megatron-Bridge registration for BailingMoeV2.5.

Registers BailingMoeV2.5 (and its Linear/Hybrid variants) with NVIDIA's
open-source megatron-bridge, enabling ``bridge_type: megatron-bridge`` in
AReaL for these models.

BailingMoeV2.5 uses heterogeneous attention layers:
- Lightning Attention (linear attention with learned decay) for most layers
- MLA (Multi-Latent Attention) for the last layer of every ``layer_group_size``
  group

The mbridge backend remains the default; this module is only imported (and
its decorators only run) when the user opts in via ``mcore.bridge_type:
megatron-bridge``.

Implementation outline:
- One ``BailingMoeV25Bridge`` class with three ``@register_bridge`` decorators
  for the three HF architectures (V2_5 / Linear / Hybrid)
- ``provider_bridge()`` sets MLA + MoE fields and injects the heterogeneous
  layer spec from ``bailing_moe.make_mcore_layer_specs_bailing_moe``
- ``mapping_registry()`` enumerates attention mappings per-layer
  (Lightning vs MLA) because ``AutoMapping`` wildcards cannot express
  per-layer mapping selection
- ``LightningQKVMapping`` is an ``AutoMapping`` subclass that permutes the
  fused QKV weight between HF ``[Q|K|V]`` and mcore ``[q0,k0,v0,...]``
  layouts before TP sharding (HF -> mcore) and after TP gathering
  (mcore -> HF).
"""

import os
from functools import partial

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt.gpt_model import GPTModel

from areal.models.mcore.bailing_moe import (
    is_lightning_layer,
    make_mcore_layer_specs_bailing_moe,
)
from areal.utils import logging

logger = logging.getLogger("BailingMoeMegatronBridge")


# ---------------------------------------------------------------------------
# Lightning Attention fused-QKV permutation
# ---------------------------------------------------------------------------


def _permute_qkv_hf_to_mcore(
    weight: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Permute HF fused QKV ([3,H,D,...]) to mcore layout ([H,3,D,...])."""
    if weight.ndim == 1:
        w = weight.view(3, num_heads, head_dim)
        w = w.permute(1, 0, 2).contiguous()
        return w.view(-1)
    hidden = weight.shape[1]
    w = weight.view(3, num_heads, head_dim, hidden)
    w = w.permute(1, 0, 2, 3).contiguous()
    return w.view(num_heads * 3 * head_dim, hidden)


def _permute_qkv_mcore_to_hf(
    weight: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Permute mcore fused QKV ([H,3,D,...]) back to HF layout ([3,H,D,...])."""
    if weight.ndim == 1:
        w = weight.view(num_heads, 3, head_dim)
        w = w.permute(1, 0, 2).contiguous()
        return w.view(-1)
    hidden = weight.shape[1]
    w = weight.view(num_heads, 3, head_dim, hidden)
    w = w.permute(1, 0, 2, 3).contiguous()
    return w.view(3 * num_heads * head_dim, hidden)


class LightningQKVMapping(AutoMapping):
    """Fused-QKV mapping for BailingMoe Lightning Attention layers.

    HF layout : ``[3*H*D, hidden]`` ordered as ``[Q_all | K_all | V_all]``
    mcore layout: ``[H*3*D, hidden]`` ordered as ``[q0,k0,v0 | q1,k1,v1 | ...]``

    Both ``weight`` and ``bias`` need the same permutation. ``layer_norm``
    sub-parameters of the fused QKV are NOT routed here (the registry
    excludes them by giving them their own ``AutoMapping`` entries).
    """

    def __init__(
        self,
        megatron_param: str,
        hf_param: str,
        *,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__(megatron_param=megatron_param, hf_param=hf_param)
        self._num_heads = num_heads
        self._head_dim = head_dim

    def hf_to_megatron(self, hf_weights, *args, **kwargs):
        """Apply [3,H,D] -> [H,3,D] permutation when converting HF -> mcore.

        The permutation runs on the full tensor before ``super()`` shards it
        to TP ranks (the permutation does not commute with TP splitting).
        ``hf_weights`` is None on non-source TP ranks, which only participate
        in the scatter.
        """
        weight = hf_weights[0] if isinstance(hf_weights, (list, tuple)) else hf_weights
        if weight is not None:
            weight = _permute_qkv_hf_to_mcore(weight, self._num_heads, self._head_dim)
        return super().hf_to_megatron(weight, *args, **kwargs)

    def megatron_to_hf(self, megatron_weight, *args, **kwargs):
        """Apply [H,3,D] -> [3,H,D] permutation when converting mcore -> HF."""
        result = super().megatron_to_hf(megatron_weight, *args, **kwargs)
        if not result:
            return result
        return {
            k: _permute_qkv_mcore_to_hf(v, self._num_heads, self._head_dim)
            for k, v in result.items()
        }


# ---------------------------------------------------------------------------
# Per-layer mapping helpers
# ---------------------------------------------------------------------------

_GLOBAL_MAPPINGS = [
    ("embedding.word_embeddings.weight", "model.word_embeddings.weight"),
    ("decoder.final_layernorm.weight", "model.norm.weight"),
    ("output_layer.weight", "lm_head.weight"),
]


def _dense_mlp_mappings(layer_idx: int) -> list:
    """Mappings for a dense (non-MoE) MLP layer."""
    return [
        AutoMapping(
            megatron_param=(
                f"decoder.layers.{layer_idx}.mlp.linear_fc1.layer_norm_weight"
            ),
            hf_param=f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        ),
        AutoMapping(
            megatron_param=f"decoder.layers.{layer_idx}.mlp.linear_fc2.weight",
            hf_param=f"model.layers.{layer_idx}.mlp.down_proj.weight",
        ),
        GatedMLPMapping(
            megatron_param=f"decoder.layers.{layer_idx}.mlp.linear_fc1.weight",
            gate=f"model.layers.{layer_idx}.mlp.gate_proj.weight",
            up=f"model.layers.{layer_idx}.mlp.up_proj.weight",
        ),
    ]


def _moe_mlp_mappings(layer_idx: int) -> list:
    """Mappings for a MoE MLP layer (router + shared experts + expert MLP)."""
    return [
        AutoMapping(
            megatron_param=f"decoder.layers.{layer_idx}.pre_mlp_layernorm.weight",
            hf_param=f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        ),
        AutoMapping(
            megatron_param=f"decoder.layers.{layer_idx}.mlp.router.weight",
            hf_param=f"model.layers.{layer_idx}.mlp.gate.weight",
        ),
        AutoMapping(
            megatron_param=f"decoder.layers.{layer_idx}.mlp.router.expert_bias",
            hf_param=f"model.layers.{layer_idx}.mlp.gate.expert_bias",
        ),
        AutoMapping(
            megatron_param=(
                f"decoder.layers.{layer_idx}.mlp.shared_experts.linear_fc2.weight"
            ),
            hf_param=(f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"),
        ),
        GatedMLPMapping(
            megatron_param=(
                f"decoder.layers.{layer_idx}.mlp.shared_experts.linear_fc1.weight"
            ),
            gate=(f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"),
            up=f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight",
        ),
        AutoMapping(
            megatron_param=(
                f"decoder.layers.{layer_idx}.mlp.experts.linear_fc2.weight*"
            ),
            hf_param=f"model.layers.{layer_idx}.mlp.experts.*.down_proj.weight",
        ),
        GatedMLPMapping(
            megatron_param=(
                f"decoder.layers.{layer_idx}.mlp.experts.linear_fc1.weight*"
            ),
            gate=f"model.layers.{layer_idx}.mlp.experts.*.gate_proj.weight",
            up=f"model.layers.{layer_idx}.mlp.experts.*.up_proj.weight",
        ),
    ]


def _lightning_attention_mappings(
    layer_idx: int, num_heads: int, head_dim: int
) -> list:
    """Mappings for a Lightning Attention layer.

    Uses ``LightningQKVMapping`` for the fused QKV weight/bias so the
    [Q|K|V] -> [q0,k0,v0,...] permutation is applied during conversion.
    """
    prefix_mcore = f"decoder.layers.{layer_idx}.self_attention"
    prefix_hf = f"model.layers.{layer_idx}.attention"
    return [
        AutoMapping(
            megatron_param=f"decoder.layers.{layer_idx}.input_layernorm.weight",
            hf_param=f"model.layers.{layer_idx}.input_layernorm.weight",
        ),
        LightningQKVMapping(
            megatron_param=f"{prefix_mcore}.linear_qkv.weight",
            hf_param=f"{prefix_hf}.query_key_value.weight",
            num_heads=num_heads,
            head_dim=head_dim,
        ),
        AutoMapping(
            megatron_param=f"{prefix_mcore}.linear_proj.weight",
            hf_param=f"{prefix_hf}.dense.weight",
        ),
        AutoMapping(
            megatron_param=f"{prefix_mcore}.linear_gate.weight",
            hf_param=f"{prefix_hf}.g_proj.weight",
        ),
        AutoMapping(
            megatron_param=f"{prefix_mcore}.gate_norm.weight",
            hf_param=f"{prefix_hf}.g_norm.weight",
        ),
        AutoMapping(
            megatron_param=f"{prefix_mcore}.q_layernorm.weight",
            hf_param=f"{prefix_hf}.query_layernorm.weight",
        ),
        AutoMapping(
            megatron_param=f"{prefix_mcore}.k_layernorm.weight",
            hf_param=f"{prefix_hf}.key_layernorm.weight",
        ),
    ]


def _mla_attention_mappings(layer_idx: int, q_lora_rank: int | None) -> list:
    """Mappings for an MLA Attention layer.

    Two sub-shapes depending on whether Q uses a LoRA decomposition:
    - ``q_lora_rank=None``  : direct ``q_proj``
    - ``q_lora_rank != None``: low-rank ``q_a_proj`` + ``q_a_layernorm`` +
      ``q_b_proj``
    """
    prefix_mcore = f"decoder.layers.{layer_idx}.self_attention"
    prefix_hf = f"model.layers.{layer_idx}.attention"
    common = [
        AutoMapping(
            megatron_param=f"decoder.layers.{layer_idx}.input_layernorm.weight",
            hf_param=f"model.layers.{layer_idx}.input_layernorm.weight",
        ),
        AutoMapping(
            megatron_param=f"{prefix_mcore}.linear_kv_down_proj.weight",
            hf_param=f"{prefix_hf}.kv_a_proj_with_mqa.weight",
        ),
        AutoMapping(
            megatron_param=(f"{prefix_mcore}.linear_kv_up_proj.layer_norm_weight"),
            hf_param=f"{prefix_hf}.kv_a_layernorm.weight",
        ),
        AutoMapping(
            megatron_param=f"{prefix_mcore}.linear_kv_up_proj.weight",
            hf_param=f"{prefix_hf}.kv_b_proj.weight",
        ),
        AutoMapping(
            megatron_param=f"{prefix_mcore}.linear_proj.weight",
            hf_param=f"{prefix_hf}.dense.weight",
        ),
    ]
    if q_lora_rank is None:
        q_specific = [
            AutoMapping(
                megatron_param=f"{prefix_mcore}.linear_q_proj.weight",
                hf_param=f"{prefix_hf}.q_proj.weight",
            ),
        ]
    else:
        q_specific = [
            AutoMapping(
                megatron_param=f"{prefix_mcore}.linear_q_down_proj.weight",
                hf_param=f"{prefix_hf}.q_a_proj.weight",
            ),
            AutoMapping(
                megatron_param=(f"{prefix_mcore}.linear_q_up_proj.layer_norm_weight"),
                hf_param=f"{prefix_hf}.q_a_layernorm.weight",
            ),
            AutoMapping(
                megatron_param=f"{prefix_mcore}.linear_q_up_proj.weight",
                hf_param=f"{prefix_hf}.q_b_proj.weight",
            ),
        ]
    return common + q_specific


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


@MegatronModelBridge.register_bridge(
    source="BailingMoeV2_5ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="bailing_moe_v2",
)
@MegatronModelBridge.register_bridge(
    source="BailingMoeLinearForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="bailing_moe_linear",
)
@MegatronModelBridge.register_bridge(
    source="BailingHybridForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="bailing_hybrid",
)
class BailingMoeV25Bridge(MegatronModelBridge):
    """Megatron Bridge for BailingMoeV2.5 (Lightning + MLA heterogeneous)."""

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        # Inject AReaL's heterogeneous layer spec. The provider's default
        # get_gpt_decoder_block_spec is uniform and does not know about
        # Lightning Attention. The signature
        # ``(tf_config, vp_stage=None) -> TransformerBlockSubmodules``
        # matches what megatron-bridge calls.
        # This assignment serves direct megatron-bridge usage (e.g.
        # ``AutoBridge.to_megatron_provider()`` outside AReaL); AReaL's own
        # flow overrides it again in ``registry.make_mcore_model()`` with a
        # spec built in the current TP/PP/CP/EP context.
        provider.transformer_layer_spec = partial(
            make_mcore_layer_specs_bailing_moe,
            hf_config=hf_config,
            use_te=True,
        )

        # ---------------- MTP ----------------
        mtp_num_layers = getattr(hf_config, "num_nextn_predict_layers", None)
        if os.environ.get("AREAL_DISABLE_MTP", "0") == "1" and mtp_num_layers:
            logger.warning(
                f"AREAL_DISABLE_MTP=1: overriding mtp_num_layers from {mtp_num_layers} to 0"
            )
            mtp_num_layers = 0
        if mtp_num_layers:
            # mapping_registry() has no mtp.* entries yet; megatron-bridge only
            # warns on unmapped params, so MTP layers would stay randomly
            # initialized instead of loading from the HF checkpoint.
            logger.warning(
                f"mtp_num_layers={mtp_num_layers} from HF config, but MTP weight "
                "mappings are not implemented — MTP layers will NOT be loaded "
                "from the checkpoint. Set AREAL_DISABLE_MTP=1 to disable MTP."
            )
        provider.mtp_num_layers = mtp_num_layers or 0

        # ---------------- Architecture basics ----------------
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.qk_layernorm = True
        provider.multi_latent_attention = True

        # ---------------- MLA dimensions ----------------
        provider.q_lora_rank = getattr(hf_config, "q_lora_rank", None)
        provider.kv_lora_rank = getattr(hf_config, "kv_lora_rank", 512)
        provider.qk_head_dim = getattr(hf_config, "qk_nope_head_dim", 128)
        provider.qk_pos_emb_head_dim = getattr(hf_config, "qk_rope_head_dim", 64)
        provider.v_head_dim = getattr(hf_config, "v_head_dim", 128)

        # ---------------- MoE ----------------
        num_layers = hf_config.num_hidden_layers
        first_k_dense = getattr(hf_config, "first_k_dense_replace", 0)
        provider.moe_layer_freq = [
            0 if i < first_k_dense else 1 for i in range(num_layers)
        ]

        shared_expert_intermediate = getattr(
            hf_config, "moe_shared_expert_intermediate_size", None
        )
        if shared_expert_intermediate is None:
            num_shared = getattr(hf_config, "num_shared_experts", 0)
            inter = getattr(
                hf_config,
                "moe_intermediate_size",
                getattr(hf_config, "intermediate_size", None),
            )
            shared_expert_intermediate = (
                num_shared * inter if num_shared > 0 and inter is not None else None
            )
        provider.moe_shared_expert_intermediate_size = shared_expert_intermediate

        provider.num_moe_experts = getattr(hf_config, "num_experts", None)
        provider.moe_ffn_hidden_size = getattr(hf_config, "moe_intermediate_size", None)
        provider.moe_router_topk = getattr(hf_config, "num_experts_per_tok", 8)
        provider.moe_router_score_function = getattr(
            hf_config, "scoring_func", "sigmoid"
        )
        provider.moe_router_num_groups = getattr(hf_config, "n_group", 8)
        provider.moe_router_group_topk = getattr(hf_config, "topk_group", 4)
        provider.moe_router_topk_scaling_factor = getattr(
            hf_config, "routed_scaling_factor", None
        )
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_load_balancing_type = "none"
        provider.moe_router_bias_update_rate = 0.0
        provider.moe_router_dtype = "fp32"
        provider.moe_grouped_gemm = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_z_loss_coeff = 3.5e-6
        provider.moe_shared_expert_overlap = False
        provider.moe_permute_fusion = True

        # ---------------- RoPE ----------------
        # Field values mirror hf_to_mcore_config_bailing_moe to avoid drift
        # against newer mcore defaults (mscale=1.0/mscale_all_dim=0.0 vs
        # original 0.707/0.707).
        rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
        provider.rope_type = "rope"
        provider.rotary_base = (getattr(hf_config, "rope_parameters", None) or {}).get(
            "rope_theta", getattr(hf_config, "rope_theta", 10000.0)
        )
        provider.rotary_percent = 1.0
        provider.rotary_scaling_factor = rope_scaling.get("factor", 1.0)
        provider.apply_rope_fusion = False
        provider.mscale = 0.707
        provider.mscale_all_dim = 0.707
        provider.original_max_position_embeddings = (
            rope_scaling.get("original_max_position_embeddings")
            or getattr(hf_config, "original_max_position_embeddings", None)
            or getattr(hf_config, "max_position_embeddings", 4096)
        )

        # ---------------- Fusions / misc ----------------
        provider.bias_activation_fusion = True
        provider.bias_dropout_fusion = True
        provider.cross_entropy_loss_fusion = False
        provider.masked_softmax_fusion = True
        provider.persist_layer_norm = True
        provider.gradient_accumulation_fusion = True
        provider.attention_softmax_in_fp32 = True
        provider.disable_bf16_reduced_precision_matmul = True
        provider.hidden_dropout = 0.0
        provider.make_vocab_size_divisible_by = 128
        provider.seq_length = getattr(hf_config, "max_position_embeddings", 4096)

        # ---------------- Layer norm epsilon ----------------
        provider.layernorm_epsilon = getattr(
            hf_config, "rms_norm_eps", provider.layernorm_epsilon
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        hf_config = self.hf_config
        num_layers = hf_config.num_hidden_layers
        layer_group_size = getattr(hf_config, "layer_group_size", 1)
        first_k_dense = getattr(hf_config, "first_k_dense_replace", 0)
        q_lora_rank = getattr(hf_config, "q_lora_rank", None)
        num_heads = hf_config.num_attention_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // num_heads)

        mappings: list = []

        # 1. Global tensors (note: BailingMoe uses ``word_embeddings`` not
        # ``embed_tokens``; share is False)
        for mcore_name, hf_name in _GLOBAL_MAPPINGS:
            mappings.append(AutoMapping(megatron_param=mcore_name, hf_param=hf_name))

        # 2. MLP — same structure within dense / MoE halves, but enumerated
        # per-layer to keep mappings flat and avoid wildcard ambiguity with
        # attention parameters that share the layer-prefix.
        for layer_idx in range(num_layers):
            if layer_idx < first_k_dense:
                mappings.extend(_dense_mlp_mappings(layer_idx))
            else:
                mappings.extend(_moe_mlp_mappings(layer_idx))

        # 3. Attention — per-layer Lightning vs MLA dispatch
        n_lightning = 0
        n_mla = 0
        for layer_idx in range(num_layers):
            if is_lightning_layer(layer_idx, layer_group_size):
                mappings.extend(
                    _lightning_attention_mappings(layer_idx, num_heads, head_dim)
                )
                n_lightning += 1
            else:
                mappings.extend(_mla_attention_mappings(layer_idx, q_lora_rank))
                n_mla += 1

        logger.info(
            f"Built BailingMoe megatron-bridge mapping registry: "
            f"{num_layers} layers ({n_lightning} Lightning + {n_mla} MLA), "
            f"layer_group_size={layer_group_size}, first_k_dense={first_k_dense}, "
            f"q_lora_rank={q_lora_rank}, total mappings={len(mappings)}"
        )

        return MegatronMappingRegistry(*mappings)

# SPDX-License-Identifier: Apache-2.0

"""DSA (Deep Sparse Attention) MLA self-attention for GLM-5.1.

Ported from slime_plugins/models/glm5/glm5.py (L33-L604) with these adaptations
for AReaL:

* Drop modelopt-based `Linear` path — only TE/standard linears supported here.
* Drop `backward_dw` / `set_for_recompute_input_layernorm` — AReaL does not
  split weight gradient updates across micro-batch boundaries.
* Use TE's `fused_apply_rotary_pos_emb_thd` (slime notes precision is slightly
  worse than apex's, but apex.transformer is unavailable in our container).
* `index_topk` falls back to 2048 if `config.dsa_indexer_topk` is absent.

This module inherits `megatron.core.transformer.attention.Attention` directly
(NOT `mcore` DSAttention container), so packed THD inputs flow naturally
without the `assert packed_seq_params is None` that mcore's DSAttention raises.
"""

import math
from dataclasses import dataclass
from typing import NoReturn

import torch
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELinear,
    fused_apply_rotary_pos_emb_thd,
)
from megatron.core.models.common.embeddings import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    _yarn_get_mscale,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
)
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.moe.moe_utils import (
    RouterGatingLinearFunction as WeightLinearFunction,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig

from areal.experimental.ops.dsa.indexer import (
    generate_varlen_mask_params,
    lighting_indexer,
)
from areal.experimental.ops.dsa.sparse_mla import SparseMLA


@dataclass
class DSASelfAttentionSubmodules:
    """Submodules for the DSA MLA self-attention layer."""

    linear_q_down_proj: ModuleSpec | type = None
    linear_q_up_proj: ModuleSpec | type = None
    linear_kv_down_proj: ModuleSpec | type = None
    linear_kv_up_proj: ModuleSpec | type = None
    linear_v_up_proj: ModuleSpec | type = None
    core_attention: ModuleSpec | type = None
    linear_proj: ModuleSpec | type = None
    q_layernorm: ModuleSpec | type = None
    kv_layernorm: ModuleSpec | type = None
    # added for indexer
    wq_b: ModuleSpec | type = None
    wk: ModuleSpec | type = None
    k_norm: ModuleSpec | type = None
    weights_proj: ModuleSpec | type = None


class DSAMultiLatentAttention(Attention):
    """DSA-enabled Multi-Latent Attention base class.

    Holds the shared init (output proj, rotary embedding, softmax scale) and
    the forward path that composes indexer + SparseMLA kernel. Self-attention
    specialization (q/kv down/up projections, indexer submodules) lives on
    `DSAMLASelfAttention`.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        is_mtp_layer: bool = False,
        cp_comm_type: str | None = None,
        model_comm_pgs=None,
        pg_collection=None,
    ) -> None:
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attention_type=attention_type,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
        self.query_projection_size = (
            self.config.v_head_dim * self.config.num_attention_heads
        )
        self.q_head_dim = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim

        # Overwrite base class kv shape for MLA inference compatibility.
        self.key_hidden_size = self.q_head_dim
        self.val_hidden_size = self.config.v_head_dim

        self.recompute_up_proj = (
            self.config.recompute_granularity == "selective"
            and "mla_up_proj" in self.config.recompute_modules
        )
        self.qkv_up_checkpoint = None

        mscale = _yarn_get_mscale(self.config.rotary_scaling_factor, self.config.mscale)
        self.softmax_scale = mscale * mscale / math.sqrt(self.q_head_dim)

        if self.config.rope_type == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_percent=self.config.rotary_percent,
                rotary_base=self.config.rotary_base,
                cp_group=self.pg_collection.cp,
            )
        elif self.config.rope_type == "yarn":
            self.rotary_pos_emb = YarnRotaryEmbedding(
                self.config.qk_pos_emb_head_dim,
                rotary_base=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                mscale=self.config.mscale,
                mscale_all_dim=self.config.mscale_all_dim,
                cp_group=self.pg_collection.cp,
            )
        else:
            raise ValueError(
                f"Unsupported RoPE type: {self.config.rope_type}; "
                "supported types are 'rope' and 'yarn'"
            )

        # Output projection.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
            tp_group=self.pg_collection.tp,
        )

        self.index_topk = getattr(self.config, "dsa_indexer_topk", None) or 2048

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        position_ids=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
        router_token_masks=None,
        loss_mask=None,
    ):
        """Forward pass for DSA multi-latent attention."""
        assert rotary_pos_emb is None, "Rotary pos emb should not be passed into MLA."
        assert attention_bias is None, "Attention bias should not be passed into MLA."
        assert rotary_pos_cos is None and rotary_pos_sin is None, (
            "MLA does not support Flash Decoding"
        )

        q, kv, wv, index_query, index_key, head_weights = (
            self.get_absorb_query_key_value_tensors(
                hidden_states,
                key_value_states,
                position_ids,
                packed_seq_params,
                inference_context=inference_context,
            )
        )

        def fused_select_topk(index_q, index_k, w, starts, ends, block_size=8192):
            seq_len = index_q.shape[0]
            # Clip topk to available key length. TileLang indexer_bwd kernel
            # requires topk to be a power of 2 (assert in
            # tilelang_indexer_bwd.py:27). key_len may not be power-of-2
            # (e.g. 9728 under CP=2), so clip to largest 2^n that still fits.
            raw_cap = min(self.index_topk, int(index_k.shape[0]))
            effective_topk = 1 << max(0, (raw_cap).bit_length() - 1)
            if effective_topk > raw_cap:
                effective_topk >>= 1
            indexer_topk_scores = []
            topk_indices = []
            for start in range(0, seq_len, block_size):
                end = min(start + block_size, seq_len)
                index_q_block = index_q[start:end]
                w_block = w[start:end]
                starts_block = starts[start:end]
                ends_block = ends[start:end]
                scores_block, indices_block = lighting_indexer(
                    index_q_block,
                    index_k,
                    w_block,
                    starts_block.to(torch.int32),
                    ends_block.to(torch.int32),
                    effective_topk,
                    topk_indices=None,
                )
                scores_block = torch.softmax(scores_block, dim=-1)
                indexer_topk_scores.append(scores_block)
                topk_indices.append(indices_block)
            return (
                torch.cat(indexer_topk_scores, dim=0),
                torch.cat(topk_indices, dim=0).unsqueeze(1),
            )

        index_key = index_key.squeeze(1)
        head_weights = head_weights.unsqueeze(-1)

        cp_size = parallel_state.get_context_parallel_world_size()

        # R18 diagnostic: AREAL_DSA_FORCE_CP1=1 bypasses CP zigzag logic,
        # using simple causal starts/ends even under CP>1. If NLL drops to
        # baseline, the bug is in the zigzag remap above.
        import os as _dsa_os

        _force_cp1 = _dsa_os.environ.get("AREAL_DSA_FORCE_CP1", "") == "1"

        if cp_size > 1 and not _force_cp1:
            # index_key is cp-gathered in zigzag-interleaved layout:
            #   [rank0_front; rank0_back; rank1_front; rank1_back; ...]
            # = [chunk0; chunk_{2C-1}; chunk1; chunk_{2C-2}; ...]
            # The indexer's clean_logits kernel enforces causal mask via a
            # contiguous [starts, ends) range. In zigzag layout the causally
            # valid KV set is non-contiguous, so we unzigzag index_key to
            # sequential layout for the indexer. Safe because index_key is
            # detached — no gradient flows through it.
            from areal.models.mcore.lightning_attention import (
                _build_zigzag_undo_indices,
            )

            total_len = index_key.shape[0]
            undo_idx = _build_zigzag_undo_indices(
                total_len, cp_size, packed_seq_params.cu_seqlens_q, index_key.device
            )
            index_key_seq = index_key[undo_idx]

            # Map each local query to its real sequential position.
            cp_rank = parallel_state.get_context_parallel_rank()
            local_len = total_len // cp_size
            gathered_pos = torch.arange(
                cp_rank * local_len,
                (cp_rank + 1) * local_len,
                device=index_key.device,
            )
            redo_idx = torch.empty(total_len, dtype=torch.long, device=index_key.device)
            redo_idx[undo_idx] = torch.arange(total_len, device=index_key.device)
            real_pos = redo_idx[gathered_pos]

            cu_sq = packed_seq_params.cu_seqlens_q
            seq_ids = torch.searchsorted(cu_sq, real_pos, right=True) - 1
            starts = cu_sq[seq_ids]
            ends = real_pos + 1

            indexer_topk_scores, topk_indices = fused_select_topk(
                index_query, index_key_seq, head_weights, starts, ends
            )

            # Remap topk_indices from sequential back to zigzag space for
            # SparseMLA (which operates on zigzag-layout kv).
            ti = topk_indices.squeeze(1)
            valid = ti != -1
            remapped = undo_idx[ti.long().clamp(min=0)]
            remapped[~valid] = -1
            topk_indices = remapped.to(topk_indices.dtype).unsqueeze(1)
        else:
            starts, ends = generate_varlen_mask_params(packed_seq_params.cu_seqlens_q)
            indexer_topk_scores, topk_indices = fused_select_topk(
                index_query, index_key, head_weights, starts, ends
            )

        core_attn_out, _ = SparseMLA.apply(q, kv, topk_indices, self.softmax_scale)
        core_attn_out = torch.einsum("thm,hdm->thd", core_attn_out, wv)
        core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        if self.recompute_up_proj:
            assert self.qkv_up_checkpoint is not None
            self.qkv_up_checkpoint.discard_output_and_register_recompute(core_attn_out)
            self.qkv_up_checkpoint = None

        output, bias = self.linear_proj(core_attn_out)
        return output, bias


class DSAMLASelfAttention(DSAMultiLatentAttention):
    """DSA Multi-Latent Self-Attention layer.

    Takes input of shape [s, b, h] and returns output of the same shape. Adds
    the indexer submodules (wq_b, wk, k_norm, weights_proj) on top of the
    standard MLA projections.
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: DSASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        is_mtp_layer: bool = False,
        cp_comm_type: str | None = None,
        model_comm_pgs=None,
        pg_collection=None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            is_mtp_layer=is_mtp_layer,
            cp_comm_type=cp_comm_type,
            model_comm_pgs=model_comm_pgs,
            pg_collection=pg_collection,
        )

        q_down_proj_kwargs: dict = {}
        if submodules.linear_q_down_proj is TELinear:
            q_down_proj_kwargs["parallel_mode"] = "duplicated"
        elif submodules.linear_q_down_proj in (
            TEColumnParallelLinear,
            ColumnParallelLinear,
        ):
            q_down_proj_kwargs["gather_output"] = False
        else:
            raise ValueError(
                f"Unsupported linear_q_down_proj: {submodules.linear_q_down_proj}"
            )

        self.linear_q_down_proj = build_module(
            submodules.linear_q_down_proj,
            self.config.hidden_size,
            self.config.q_lora_rank,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="q_down_proj",
            skip_weight_param_allocation=False,
            **q_down_proj_kwargs,
        )

        self.linear_q_up_proj = build_module(
            submodules.linear_q_up_proj,
            self.config.q_lora_rank,
            self.config.num_attention_heads * self.q_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="q_up_proj",
        )

        kv_down_proj_kwargs: dict = {}
        if submodules.linear_kv_down_proj is TELinear:
            kv_down_proj_kwargs["parallel_mode"] = "duplicated"
        elif submodules.linear_kv_down_proj in (
            TEColumnParallelLinear,
            ColumnParallelLinear,
        ):
            kv_down_proj_kwargs["gather_output"] = False
        else:
            raise ValueError(
                f"Unsupported linear_kv_down_proj: {submodules.linear_kv_down_proj}"
            )

        self.linear_kv_down_proj = build_module(
            submodules.linear_kv_down_proj,
            self.config.hidden_size,
            self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_down_proj",
            skip_weight_param_allocation=False,
            **kv_down_proj_kwargs,
        )

        self.linear_kv_up_proj = build_module(
            submodules.linear_kv_up_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads
            * (self.config.qk_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv_up_proj",
        )

        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=self.config.q_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        # Indexer submodules.
        indexer_linear_kwargs = dict(
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
            parallel_mode="duplicated",
            skip_weight_param_allocation=False,
        )

        self.wq_b = build_module(
            submodules.wq_b,
            input_size=self.config.q_lora_rank,
            output_size=self.config.dsa_indexer_n_heads
            * self.config.dsa_indexer_head_dim,
            tp_comm_buffer_name="wq_b",
            **indexer_linear_kwargs,
        )
        self.wq_b.weight._skip_gather = True

        self.wk = build_module(
            submodules.wk,
            input_size=self.config.hidden_size,
            output_size=self.config.dsa_indexer_head_dim,
            tp_comm_buffer_name="wk",
            **indexer_linear_kwargs,
        )

        # k_norm uses LayerNorm (not RMSNorm) per DSA design. Toggle config temporarily.
        old_norm = self.config.normalization
        assert config.normalization == "RMSNorm"
        self.config.normalization = "LayerNorm"
        self.k_norm = build_module(
            submodules.k_norm,
            hidden_size=self.config.dsa_indexer_head_dim,
            config=self.config,
            eps=1e-6,  # hardcoded per DSA reference implementation
        )
        self.config.normalization = old_norm

        self.weights_proj = build_module(
            submodules.weights_proj,
            input_size=self.config.hidden_size,
            output_size=self.config.dsa_indexer_n_heads,
            tp_comm_buffer_name="weights_proj",
            **indexer_linear_kwargs,
        )
        self.weights_proj.weight._skip_gather = True

        # 2026-04-30: freeze 4 个 indexer 模块,对齐 slime 默认行为
        # (slime/utils/arguments.py:197 `--freeze-params-name-list
        #  self_attention.wq_b self_attention.wk self_attention.k_norm
        #  self_attention.weights_proj`)。
        # 原因:slime 用 `q_compressed.detach()` 切断 indexer 与主反传链,
        # indexer params 自然不参与 grad,DDP 跳过 grad ready check;我们
        # 之前用 `core_attn_out + scores.sum() * 0` reattach,目的同是绕
        # DDP assert,但下游 grad NaN×0=NaN 反而污染 indexer params。
        # 走 slime 的 freeze 方案,既对齐 reference,又消除 NaN 通路。
        # 可以用 AREAL_DSA_TRAIN_INDEXER=1 关闭 freeze(以备未来 RL 阶段
        # 解锁 indexer 训练)。
        import os as _os_freeze

        _train_indexer = _os_freeze.environ.get("AREAL_DSA_TRAIN_INDEXER", "0") == "1"
        if not _train_indexer:
            for _mod in (self.wq_b, self.wk, self.k_norm, self.weights_proj):
                for _p in _mod.parameters():
                    _p.requires_grad = False

    def get_absorb_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
    ):
        """Derive `query`, `key` and `value` tensors from `hidden_states`."""
        assert hidden_states.ndim == 3, (
            f"hidden_states should be 3D [s, b, n*h], got {hidden_states.ndim}D"
        )
        assert packed_seq_params is not None

        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_context, None, hidden_states, self.config, packed_seq_params
        )
        rotary_pos_emb = self.rotary_pos_emb(
            rotary_seq_len, packed_seq=packed_seq_params is not None
        )
        # RotaryEmbedding returns a Tensor, YarnRotaryEmbedding returns
        # (emb, mscale). softmax_scale is precomputed in __init__ via
        # _yarn_get_mscale, so mscale at runtime is unused either way.
        if isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb[0]

        cu_seqlens_q = packed_seq_params.cu_seqlens_q
        cu_seqlens_kv = packed_seq_params.cu_seqlens_kv

        # QKV down projection + layer norm.
        q_compressed, _ = self.linear_q_down_proj(hidden_states)
        q_compressed = q_compressed.squeeze(1)

        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        if self.config.sequence_parallel:
            kv_combined = gather_from_sequence_parallel_region(kv_combined)
        kv_compressed, k_pos_emb = torch.split(
            kv_combined,
            [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim],
            dim=-1,
        )
        kv_compressed = self.kv_layernorm(kv_compressed)

        # Absorb.
        q_compressed = self.q_layernorm(q_compressed)
        q, _ = self.linear_q_up_proj(q_compressed)
        q = q.view(
            *q.size()[:-1],
            self.num_attention_heads_per_partition,
            self.q_head_dim,
        )
        q_no_pe, q_pos_emb = torch.split(
            q,
            [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim],
            dim=-1,
        )

        w_kc, w_vc = self.linear_kv_up_proj.weight.unflatten(
            0,
            (-1, self.config.qk_head_dim + self.config.v_head_dim),
        ).split([self.config.qk_head_dim, self.config.v_head_dim], dim=1)

        q_no_pe = torch.einsum("thd,hdm->thm", q_no_pe, w_kc)

        # Fuse rms_norm with layer_norm_weight so kv gradient all-reduces in TP.
        kv_compressed = torch.nn.functional.rms_norm(
            kv_compressed.float(),
            normalized_shape=(kv_compressed.shape[-1],),
            weight=self.linear_kv_up_proj.layer_norm_weight.float(),
            eps=self.config.layernorm_epsilon,
        ).to(kv_compressed.dtype)

        cp_group = parallel_state.get_context_parallel_group()
        _cp_size = parallel_state.get_context_parallel_world_size()

        def _cp_all_gather(t):
            if _cp_size <= 1:
                return t
            t = t.contiguous()
            t_list = [torch.empty_like(t) for _ in range(_cp_size)]
            torch.distributed.all_gather(t_list, t, group=cp_group)
            return torch.cat(t_list, dim=0)

        k_pos_emb = _cp_all_gather(k_pos_emb)
        kv_compressed = _cp_all_gather(kv_compressed)

        def fuse_rope(t_in, cu_seqlens, gathered=False):
            # MLA interleaved rope: split into [x0,x2,...] + [x1,x3,...].
            x1 = t_in[..., 0::2]
            x2 = t_in[..., 1::2]
            t = torch.cat((x1, x2), dim=-1)
            _cp_size = parallel_state.get_context_parallel_world_size()
            if _cp_size <= 1:
                return fused_apply_rotary_pos_emb_thd(
                    t, cu_seqlens, rotary_pos_emb.squeeze(0)
                )
            from areal.models.mcore.lightning_attention import (
                _build_zigzag_redo_indices,
                _build_zigzag_undo_indices,
            )

            if not gathered:
                # t is the local zigzag slice; all-gather to get full zigzag tensor.
                t_list = [torch.empty_like(t) for _ in range(_cp_size)]
                torch.distributed.all_gather(t_list, t, group=cp_group)
                t = torch.cat(t_list, dim=0)
            # t is now the full zigzag tensor; unzigzag → rope → rezigzag.
            _total = t.shape[0]
            _undo = _build_zigzag_undo_indices(_total, _cp_size, cu_seqlens, t.device)
            _redo = _build_zigzag_redo_indices(_undo)
            t_seq = t[_undo]
            # rotary_pos_emb is sequential (pos 0..total-1), not zigzag-sliced.
            _rope_seq = rotary_pos_emb.squeeze(0)
            t_seq = fused_apply_rotary_pos_emb_thd(t_seq, cu_seqlens, _rope_seq)
            t_zz = t_seq[_redo]
            if not gathered:
                # Return only this rank's local slice.
                _cp_rank = parallel_state.get_context_parallel_rank()
                _local = _total // _cp_size
                return t_zz[_cp_rank * _local : (_cp_rank + 1) * _local]
            return t_zz

        q_pos_emb = fuse_rope(q_pos_emb, cu_seqlens_q, gathered=False)
        k_pos_emb = fuse_rope(k_pos_emb, cu_seqlens_kv, gathered=True)

        query = torch.cat([q_no_pe, q_pos_emb], dim=-1).contiguous()
        key = torch.cat([kv_compressed, k_pos_emb], dim=-1).contiguous()

        # Indexer. Detach to cut gradient flow from indexer into base projections.
        q_compressed = q_compressed.detach()
        hidden_states = hidden_states.detach()
        rotary_pos_emb = rotary_pos_emb.detach()

        index_q, _ = self.wq_b(q_compressed)
        index_q = index_q.view(
            *index_q.size()[:-1],
            self.config.dsa_indexer_n_heads,
            self.config.dsa_indexer_head_dim,
        )
        if self.config.sequence_parallel:
            index_q = gather_from_sequence_parallel_region(index_q)

        index_k, _ = self.wk(hidden_states)
        index_k = self.k_norm(index_k.squeeze(1).float()).bfloat16()
        if self.config.sequence_parallel:
            index_k = gather_from_sequence_parallel_region(index_k)
        index_k = _cp_all_gather(index_k).unsqueeze(1)

        head_weights = WeightLinearFunction.apply(
            hidden_states, self.weights_proj.weight, None, torch.float32
        )
        head_weights = head_weights.squeeze(1) * (
            (self.config.dsa_indexer_n_heads**-0.5)
            * (self.config.dsa_indexer_head_dim**-0.5)
        )
        if self.config.sequence_parallel:
            head_weights = gather_from_sequence_parallel_region(head_weights)

        # GLM-5.1 indexer weight layout: first rope_dim dims are RoPE,
        # remaining dims are position-independent (matches SGLang/slime).
        index_q_pe, index_q_no_pe = torch.split(
            index_q,
            [
                self.config.qk_pos_emb_head_dim,
                self.config.dsa_indexer_head_dim - self.config.qk_pos_emb_head_dim,
            ],
            dim=-1,
        )
        index_q_pe = fuse_rope(index_q_pe, cu_seqlens_q, gathered=False)
        index_query = torch.cat([index_q_pe, index_q_no_pe], dim=-1)

        index_k_pe, index_k_no_pe = torch.split(
            index_k,
            [
                self.config.qk_pos_emb_head_dim,
                self.config.dsa_indexer_head_dim - self.config.qk_pos_emb_head_dim,
            ],
            dim=-1,
        )
        index_k_pe = fuse_rope(index_k_pe, cu_seqlens_kv, gathered=True)
        index_key = torch.cat([index_k_pe, index_k_no_pe], dim=-1)

        return query, key, w_vc, index_query, index_key, head_weights

    def get_query_key_value_tensors(self) -> NoReturn:
        raise NotImplementedError(
            "DSAMLASelfAttention uses get_absorb_query_key_value_tensors(); "
            "the standard path is not supported."
        )

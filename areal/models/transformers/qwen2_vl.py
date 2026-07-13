# SPDX-License-Identifier: Apache-2.0

# Adapted from verl


import torch
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

from areal.models.fsdp.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
    get_ulysses_sequence_parallel_world_size,
)


def ulysses_flash_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # bsz = 1, q_len = total_seqlen / sp_size
    bsz, q_len, _ = hidden_states.size()

    # (1, total_seqlen / sp_size, num_heads * head_dim)
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # (1, num_heads, total_seqlen / sp_size, head_dim)
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    if ulysses_sp_size > 1:
        if self.num_heads % ulysses_sp_size != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by Ulysses sequence parallel size({ulysses_sp_size})"
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # (1, num_heads / sp_size, total_seqlen, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    # NOTE: This is the unpatched vanilla implementation in transformers
    # (1, total_seqlen, num_heads / sp_size, head_dim)
    attn_output, _ = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,
        is_causal=self.is_causal,
    )

    if ulysses_sp_size > 1:
        # (1, total_seqlen / sp_size, num_heads, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None

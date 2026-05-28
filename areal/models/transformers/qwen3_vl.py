# SPDX-License-Identifier: Apache-2.0

# Qwen3VL-specific Ulysses sequence parallelism patch
# Based on transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextAttention

import types
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Replicate
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
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
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Ulysses sequence parallelism forward for Qwen3VLTextAttention.

    This follows the Qwen3VLTextAttention.forward signature and implementation,
    adding Ulysses sequence parallelism support.

    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(
        1, 2
    )
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    if ulysses_sp_size > 1:
        if self.config.num_attention_heads % ulysses_sp_size != 0:
            raise ValueError(
                f"num_attention_heads ({self.config.num_attention_heads}) must be divisible by Ulysses sequence parallel size({ulysses_sp_size})"
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # (1, num_heads / sp_size, total_seqlen, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=2, head_dim=1)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=2, head_dim=1)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=2, head_dim=1)

    cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    attn_output, attn_weights = flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        is_causal=self.is_causal,
        **kwargs,
    )

    if ulysses_sp_size > 1:
        # (1, total_seqlen / sp_size, num_heads, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=2, seq_dim=1)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def patch_qwen3_vl_deepstack_process_for_tp(language_model: nn.Module):
    """Patch _deepstack_process to convert visual_pos_masks and visual_embeds
    to DTensor when TP is enabled.
    """
    if not hasattr(language_model, "_deepstack_process"):
        return

    original_deepstack_process = language_model._deepstack_process

    def patched_deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        # Check if hidden_states is a DTensor (TP is enabled)
        if isinstance(hidden_states, DTensor):
            device_mesh = hidden_states.device_mesh
            placements = hidden_states.placements

            replicated_placements = [Replicate()] * device_mesh.ndim
            # Convert to local tensor for boolean indexing operations
            # DTensor's sharding propagation cannot handle dynamic output shapes from boolean indexing
            hidden_states = hidden_states.redistribute(placements=replicated_placements)
            hidden_states = hidden_states.to_local()

            hidden_states = hidden_states.clone()

            hidden_states = original_deepstack_process(
                hidden_states, visual_pos_masks, visual_embeds
            )

            hidden_states = DTensor.from_local(
                hidden_states,
                device_mesh,
                replicated_placements,
                run_check=False,
            )

            hidden_states = hidden_states.redistribute(placements=placements)

            return hidden_states
        else:
            return original_deepstack_process(
                hidden_states, visual_pos_masks, visual_embeds
            )

    language_model._deepstack_process = types.MethodType(
        patched_deepstack_process, language_model
    )

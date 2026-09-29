# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 tree GDN and attention adapters for Megatron-Core."""

from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_layer import TransformerLayer

from areal.models.tree_attn.causal_conv import tree_causal_conv1d
from areal.models.tree_attn.harts_core import run_linear_attention_plan
from areal.models.tree_attn.harts_plan import LinearAttentionPlan
from areal.models.tree_attn.module_megatron import PytorchFlexAttention
from areal.models.tree_attn.tree import TrieNode

_TREE_MASKS: dict[int, tuple[TrieNode, LinearAttentionPlan, torch.Tensor]] = {}
_TOKEN_MULTIPLICITY: ContextVar[torch.Tensor | None] = ContextVar(
    "tree_token_multiplicity", default=None
)


def register_tree_mask(
    mask: torch.Tensor, trie: TrieNode, plan: LinearAttentionPlan
) -> int:
    """Keep immutable tree metadata through Megatron activation recomputation."""
    pointer = mask.data_ptr()
    multiplicity = torch.zeros(mask.shape[0], dtype=torch.float32, device=mask.device)
    for node in trie.nodes:
        multiplicity[node.start_idx : node.end_idx + 1] = len(node.sequence_ids)
    _TREE_MASKS[pointer] = trie, plan, multiplicity
    return pointer


def unregister_tree_masks(pointers: list[int]) -> None:
    for pointer in pointers:
        _TREE_MASKS.pop(pointer, None)


class TreeGatedDeltaNet(GatedDeltaNet):
    """GDN with one projection and true-ancestor convolution per compact token."""

    def forward(self, hidden_states, attention_mask, **kwargs):
        metadata = (
            _TREE_MASKS.get(attention_mask.data_ptr())
            if isinstance(attention_mask, torch.Tensor)
            else None
        )
        if metadata is None:
            return GatedDeltaNet.forward(self, hidden_states, attention_mask, **kwargs)
        if kwargs.get("packed_seq_params") is not None:
            raise ValueError("tree GDN requires padded BSHD input")
        from fla.modules.l2norm import l2norm
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        trie, plan, _ = metadata
        _, batch, _ = hidden_states.shape
        if batch != 1:
            raise ValueError("tree GDN requires batch size one")
        count = trie.num_tokens
        projected, _ = self.in_proj(hidden_states)
        projected = projected.transpose(0, 1)
        sequence_length = projected.shape[1]
        qkv, gate, beta, alpha = torch.split(
            projected,
            [
                self.conv_dim_local_tp,
                self.v_dim // self.tp_size,
                self.num_value_heads // self.tp_size,
                self.num_value_heads // self.tp_size,
            ],
            dim=-1,
        )
        qkv = tree_causal_conv1d(
            qkv[0, :count],
            self.conv1d.weight,
            self.conv1d.bias,
            trie,
            self.act_fn,
        ).unsqueeze(0)
        query, key, value = torch.split(
            qkv,
            [
                self.qk_dim // self.tp_size,
                self.qk_dim // self.tp_size,
                self.v_dim // self.tp_size,
            ],
            dim=-1,
        )
        query = query.reshape(1, count, -1, self.key_head_dim)
        key = key.reshape(1, count, -1, self.key_head_dim)
        value = value.reshape(1, count, -1, self.value_head_dim)
        if self.use_qk_l2norm:
            query = l2norm(query.contiguous())
            key = l2norm(key.contiguous())
        if self.num_value_heads > self.num_key_heads:
            repeat = self.num_value_heads // self.num_key_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        gate = gate[:, :count].reshape(1, count, -1, self.value_head_dim)
        beta = beta[:, :count].reshape(1, count, -1).sigmoid()
        alpha = alpha[:, :count].reshape(1, count, -1)
        g = -self.A_log.exp() * F.softplus(alpha.float() + self.dt_bias)
        core = run_linear_attention_plan(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            plan,
            chunk_gated_delta_rule,
        )
        normalized = self._apply_gated_norm(core, gate)
        normalized = normalized.reshape(count, 1, -1)
        if count < sequence_length:
            normalized = F.pad(normalized, (0, 0, 0, 0, 0, sequence_length - count))
        return self.out_proj(normalized)


class TreeTransformerLayer(TransformerLayer):
    """Expose trie multiplicity to the router during forward and recomputation."""

    def forward(self, hidden_states, attention_mask, **kwargs):
        metadata = (
            _TREE_MASKS.get(attention_mask.data_ptr())
            if isinstance(attention_mask, torch.Tensor)
            else None
        )
        token = _TOKEN_MULTIPLICITY.set(metadata[2] if metadata is not None else None)
        try:
            return TransformerLayer.forward(
                self, hidden_states, attention_mask, **kwargs
            )
        finally:
            _TOKEN_MULTIPLICITY.reset(token)


def weighted_router_stats(
    scores: torch.Tensor,
    routing_map: torch.Tensor,
    multiplicity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match token counts and probability sums of the unpacked trajectories."""
    if scores.shape != routing_map.shape:
        raise ValueError("router scores and map must have matching shapes")
    if scores.shape[0] != multiplicity.numel():
        raise ValueError("router multiplicity must align with local tokens")
    weighted_scores = scores * multiplicity[:, None]
    counts = (routing_map.float() * multiplicity[:, None]).sum(dim=0)
    return weighted_scores, counts, multiplicity.sum()


class TreeTopKRouter(TopKRouter):
    """Weight Qwen3.5 global auxiliary routing loss by trajectory multiplicity."""

    def _apply_global_aux_loss(
        self, probs, scores_for_aux_loss, routing_map, with_padding_mask=False
    ):
        multiplicity = _TOKEN_MULTIPLICITY.get()
        if multiplicity is None:
            return TopKRouter._apply_global_aux_loss(
                self, probs, scores_for_aux_loss, routing_map, with_padding_mask
            )
        coefficient = self.get_aux_loss_coeff("global_aux_loss")
        if coefficient == 0:
            return probs
        if self.config.moe_router_fusion:
            raise ValueError("tree routing requires unfused auxiliary loss")
        from megatron.core.transformer.moe.moe_utils import (
            switch_load_balancing_loss_func,
        )

        local_rows = scores_for_aux_loss.shape[0]
        if multiplicity.numel() != local_rows:
            tp_size = self.tp_group.size()
            if multiplicity.numel() != local_rows * tp_size:
                raise ValueError("tree multiplicity does not match router token layout")
            rank = self.tp_group.rank()
            multiplicity = multiplicity[rank * local_rows : (rank + 1) * local_rows]
        weighted_scores, counts, local_tokens = weighted_router_stats(
            scores_for_aux_loss, routing_map, multiplicity
        )
        dist.all_reduce(counts, group=self.tp_dp_cp_group)
        total_tokens = local_tokens.clone()
        dist.all_reduce(total_tokens, group=self.tp_dp_cp_group)
        self.global_tokens_per_expert += counts
        self.ga_steps += 1
        average_counts = self.global_tokens_per_expert / self.ga_steps
        aux_loss = switch_load_balancing_loss_func(
            probs=weighted_scores,
            tokens_per_expert=average_counts,
            total_num_tokens=total_tokens.clamp_min(1),
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=coefficient,
        )
        return self.attach_and_log_load_balancing_loss(
            probs,
            coefficient,
            aux_loss,
            "global_load_balancing_loss",
            self.tp_dp_cp_group,
            reduce_group_has_dp=True,
            valid_token_count=local_tokens,
        )


def patch_qwen35_tree_model(model: torch.nn.Module) -> None:
    """Replace only GDN and full-attention cores; preserve checkpoint keys."""
    language_model = getattr(model, "language_model", model)
    for module in language_model.modules():
        if type(module) is GatedDeltaNet:
            module.forward = MethodType(TreeGatedDeltaNet.forward, module)
        elif type(module) is TransformerLayer:
            module.forward = MethodType(TreeTransformerLayer.forward, module)
        elif type(module) is TopKRouter:
            if module.get_aux_loss_coeff("aux_loss") or module.get_aux_loss_coeff(
                "seq_aux_loss"
            ):
                raise ValueError("tree routing supports global auxiliary loss only")
            module._apply_global_aux_loss = MethodType(
                TreeTopKRouter._apply_global_aux_loss, module
            )
        elif isinstance(module, SelfAttention):
            old = module.core_attention
            module.core_attention = PytorchFlexAttention(
                config=module.config,
                layer_number=module.layer_number,
                attn_mask_type=old.attn_mask_type,
                attention_type=module.attention_type,
                attention_dropout=getattr(old, "attention_dropout", None),
                softmax_scale=getattr(old, "softmax_scale", None),
            )


@contextmanager
def qwen35_tree_positions(position_ids: torch.Tensor):
    """Use trie depth for text-only mRoPE in Qwen3VLModel.forward."""
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import model as qwen_model

    original = qwen_model.get_rope_index

    def get_tree_rope_index(*args, **kwargs):
        del args, kwargs
        positions = position_ids.unsqueeze(0).expand(3, -1, -1)
        return positions, torch.zeros(
            (position_ids.shape[0], 1), device=position_ids.device, dtype=torch.long
        )

    qwen_model.get_rope_index = get_tree_rope_index
    try:
        yield
    finally:
        qwen_model.get_rope_index = original

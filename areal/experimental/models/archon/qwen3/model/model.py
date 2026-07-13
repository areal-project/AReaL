# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/models/qwen3/model/model.py

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor

from areal.experimental.models.archon.attention import (
    SDPAWrapper,
    TreeAttentionMeta,
    TreeAttentionWrapper,
    VarlenAttentionWrapper,
)
from areal.experimental.models.archon.base import BaseArchonModel
from areal.experimental.models.archon.moe import MoE
from areal.experimental.models.archon.qwen3.model.args import Qwen3ModelArgs
from areal.experimental.models.archon.qwen3.model.rope import (
    apply_rotary_emb,
    precompute_rope_cache,
    repeat_kv,
)
from areal.experimental.models.archon.ulysses import (
    gather_heads_scatter_seq,
    gather_seq_scatter_heads,
)


def maybe_to_local(x: torch.Tensor) -> torch.Tensor:
    """Convert DTensor to local tensor if needed."""
    if isinstance(x, DTensor):
        return x.to_local()
    return x


def _is_moe_layer(layer_id: int, model_args: Qwen3ModelArgs) -> bool:
    """Determine if a layer should use MoE instead of dense FFN.

    Args:
        layer_id: The layer index (0-based).
        model_args: Model configuration.

    Returns:
        True if this layer should use MoE, False otherwise.

    Examples:
        - decoder_sparse_step=1: All layers are MoE (layers 0,1,2,3,... are MoE)
        - decoder_sparse_step=2: Every other layer starting from layer 1
          (layers 1,3,5,... are MoE; layers 0,2,4,... are dense)
        - decoder_sparse_step=0 or negative: No MoE layers
    """
    if not model_args.moe_enabled:
        return False

    if model_args.moe_args is None:
        return False

    sparse_step = model_args.decoder_sparse_step
    if sparse_step <= 0:
        return False

    # (layer_id + 1) % sparse_step == 0 means:
    # - sparse_step=1: all layers are MoE
    # - sparse_step=2: layers 1,3,5,... are MoE
    # This follows HuggingFace Qwen3-MoE convention where layer numbering is 1-based
    # for the sparse step check (i.e., "every Nth layer" counts from 1, not 0).
    return (layer_id + 1) % sparse_step == 0


class RMSNorm(nn.Module):
    """RMSNorm with float32 intermediate computation."""

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True
    ):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            return self.weight * x.to(input_dtype)
        return x.to(input_dtype)

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)


class Attention(nn.Module):
    """Multi-head attention module with Q/K norm, GQA, and Ulysses SP support.

    Ulysses SP (Sequence Parallelism) uses All-to-All communication to split
    attention heads across GPUs while keeping the full sequence on each GPU.
    This is different from Ring Attention which splits the sequence.
    """

    q_norm: RMSNorm | None
    k_norm: RMSNorm | None

    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim
        self.scaling = self.head_dim**-0.5

        # Q/K normalization
        if model_args.qk_norm:
            self.q_norm = RMSNorm(
                self.head_dim, eps=model_args.norm_eps, elementwise_affine=True
            )
            self.k_norm = RMSNorm(
                self.head_dim, eps=model_args.norm_eps, elementwise_affine=True
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # Linear projections
        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )

        # Select attention backend
        if model_args.attn_type == "tree":
            self.packed_attn = TreeAttentionWrapper()
        elif model_args.attn_type == "varlen":
            self.packed_attn = VarlenAttentionWrapper()
        else:
            self.packed_attn = SDPAWrapper()

        # Ulysses SP state (set by parallelize_fn via set_cp_group)
        self._cp_group: ProcessGroup | None = None
        self._cp_size: int = 1
        self._cp_rank: int = 0

    def set_cp_group(self, cp_group: ProcessGroup | None):
        """Configure Ulysses sequence parallelism.

        Args:
            cp_group: Process group for Ulysses All-to-All communication.
                     If None or size 1, SP is disabled.
        """
        self._cp_group = cp_group
        if cp_group is not None:
            self._cp_size = dist.get_world_size(cp_group)
            self._cp_rank = dist.get_rank(cp_group)
        else:
            self._cp_size = 1
            self._cp_rank = 0

    @property
    def _sp_enabled(self) -> bool:
        """Check if Ulysses SP is enabled."""
        return self._cp_group is not None and self._cp_size > 1

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        tree_attn_meta: TreeAttentionMeta | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.q_norm:
            xq = self.q_norm(xq)
        if self.k_norm:
            xk = self.k_norm(xk)

        # Convert DTensor to local tensor (for TP compatibility)
        xq = maybe_to_local(xq)
        xk = maybe_to_local(xk)

        xq, xk = apply_rotary_emb(xq, xk, rope_cache, positions)

        # [Optional] Ulysses All-to-All
        if self._sp_enabled:
            # Repeat kv heads if needed for gather_seq_scatter_heads
            kv_heads = xk.size(2)
            if kv_heads < self._cp_size:
                repeats = self._cp_size // kv_heads
                xk = repeat_kv(xk, repeats)
                xv = repeat_kv(xv, repeats)

            xq = gather_seq_scatter_heads(
                xq,
                seq_dim=1,
                head_dim=2,
                unpadded_dim_size=seqlen * self._cp_size,
                group=self._cp_group,
            )
            xk = gather_seq_scatter_heads(
                xk,
                seq_dim=1,
                head_dim=2,
                unpadded_dim_size=seqlen * self._cp_size,
                group=self._cp_group,
            )
            xv = gather_seq_scatter_heads(
                xv,
                seq_dim=1,
                head_dim=2,
                unpadded_dim_size=seqlen * self._cp_size,
                group=self._cp_group,
            )

            seqlen = xq.shape[1]

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = self.packed_attn(
            xq,
            xk,
            xv,
            scale=self.scaling,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            tree_attn_meta=tree_attn_meta,
        )

        output = output.transpose(1, 2).contiguous()

        # [Optional] Ulysses All-to-All
        if self._sp_enabled:
            output = gather_heads_scatter_seq(
                output, head_dim=2, seq_dim=1, group=self._cp_group
            )
            seqlen = output.shape[1]

        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU feedforward module."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with attention and feedforward/MoE."""

    def __init__(self, layer_id: int, model_args: Qwen3ModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim

        self.attention = Attention(model_args)
        self.attention_norm = RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = RMSNorm(model_args.dim, eps=model_args.norm_eps)

        # Determine if this layer uses MoE or dense FFN
        self.moe_enabled = _is_moe_layer(layer_id, model_args)
        if self.moe_enabled:
            # MoE layer uses moe_inter_dim for expert hidden dimension
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
            self.feed_forward = None
        else:
            # Dense layer uses hidden_dim
            self.moe = None
            self.feed_forward = FeedForward(
                dim=model_args.dim, hidden_dim=model_args.hidden_dim
            )

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        tree_attn_meta: TreeAttentionMeta | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.attention_norm(x),
            rope_cache,
            positions,
            cu_seqlens,
            max_seqlen,
            tree_attn_meta=tree_attn_meta,
        )
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self):
        """Initialize layer parameters."""
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std)
        else:
            self.feed_forward.init_weights(self.weight_init_std)

    def init_buffers(self, buffer_device: torch.device | str):
        """Initialize layer buffers (MoE buffers if enabled).

        Args:
            buffer_device: Device for buffers.
        """
        if self.moe_enabled:
            self.moe.init_buffers(buffer_device)


class Qwen3Model(BaseArchonModel):
    """Qwen3 transformer model."""

    def __init__(self, model_args: Qwen3ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id
        self.head_dim = model_args.head_dim
        self.is_critic = model_args.is_critic

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer(
            "rope_cache", self._precompute_rope_cache(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.is_critic:
            self.output = None
            self.score = nn.Linear(model_args.dim, model_args.num_labels, bias=False)
        else:
            self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
            self.score = None

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def init_weights(self):
        """Initialize model parameters."""
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()

        if self.norm is not None:
            self.norm.reset_parameters()

        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3

        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

        if self.score is not None:
            nn.init.trunc_normal_(
                self.score.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def init_buffers(self, buffer_device: torch.device | str):
        """Initialize model buffers (rope_cache and MoE buffers).

        Args:
            buffer_device: Device for buffers.
        """
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        # Initialize MoE buffers in each layer
        for layer in self.layers.values():
            if layer is not None:
                layer.init_buffers(buffer_device)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int | torch.Tensor,
        tree_attn_meta: TreeAttentionMeta | None = None,
    ) -> torch.Tensor:
        # When pipeline parallelism enabled, cu_seqlens is [1, B+1]
        if cu_seqlens.ndim == 2:
            cu_seqlens = cu_seqlens.squeeze(0)

        # When pipeline parallelism enabled, max_seqlen is [1]
        if isinstance(max_seqlen, torch.Tensor):
            max_seqlen = int(max_seqlen.item())

        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        for layer in self.layers.values():
            h = layer(
                h,
                self.rope_cache,
                positions,
                cu_seqlens,
                max_seqlen,
                tree_attn_meta=tree_attn_meta,
            )

        h = self.norm(h) if self.norm else h

        if self.is_critic:
            output = self.score(h) if self.score else h
        else:
            output = self.output(h) if self.output else h
        return output


__all__ = [
    "Qwen3ModelArgs",
    "Attention",
    "FeedForward",
    "TransformerBlock",
    "Qwen3Model",
]

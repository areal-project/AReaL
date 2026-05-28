# SPDX-License-Identifier: Apache-2.0

import os

import torch

from .tilelang_indexer_bwd import indexer_bwd_interface
from .tilelang_indexer_fwd import indexer_fwd_interface

# DSA indexer topk_indices recording for routing comparison.
# When AREAL_DUMP_ROUTING is set, each layer's forward appends its
# topk_indices here. Cleared after dump in megatron_engine.py.
_recorded_dsa_indices: list[torch.Tensor] = []
_record_dsa = bool(os.environ.get("AREAL_DUMP_ROUTING", ""))


def get_recorded_dsa_indices() -> list[torch.Tensor]:
    return _recorded_dsa_indices


def clear_recorded_dsa_indices():
    _recorded_dsa_indices.clear()


def pytorch_extract_topk_scores(logits, topk_indices, dim=-1):
    valid_mask = topk_indices != -1
    safe_indices = topk_indices.clamp(min=0).to(torch.int64)
    scores = torch.gather(logits, dim=dim, index=safe_indices)
    scores = torch.where(valid_mask, scores, float("-inf"))
    return scores


class IndexerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        index_q: torch.Tensor,
        index_k: torch.Tensor,
        weights: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        topk: int,
        topk_indices: torch.Tensor | None = None,
    ):
        _, head_num, _ = index_q.shape
        logits = indexer_fwd_interface(
            index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True
        )
        if topk_indices is None:
            sorted_indices = torch.argsort(-logits, dim=-1, stable=True)
            topk_indices = sorted_indices[..., :topk].to(torch.int32)
            index_score = torch.gather(
                logits, dim=-1, index=topk_indices.to(torch.int64)
            )
            topk_indices = topk_indices.masked_fill(index_score == -torch.inf, -1)

        index_score = pytorch_extract_topk_scores(logits, topk_indices)

        if _record_dsa:
            _recorded_dsa_indices.append(topk_indices.detach().cpu())

        ctx.save_for_backward(
            index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices
        )
        ctx.topk = topk
        ctx.head_num = head_num
        return index_score, topk_indices

    @staticmethod
    def backward(ctx, grad_scores, grad_indices):
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices = (
            ctx.saved_tensors
        )
        grad_q, grad_w, grad_k = indexer_bwd_interface(
            index_q, weights, index_k, topk_indices, grad_scores
        )
        # 7 returns matching forward inputs (excluding ctx):
        # index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices
        return grad_q, grad_k, grad_w, None, None, None, None


def lighting_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk: int,
    topk_indices: torch.Tensor | None = None,
):
    weights = weights.squeeze(-1)
    return IndexerFunction.apply(
        index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk, topk_indices
    )


def generate_varlen_mask_params(cu_seqlens):
    seq_len = cu_seqlens[-1].item()
    q_indices = torch.arange(0, seq_len, device=cu_seqlens.device)
    seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
    starts = cu_seqlens[seq_indices]
    ends = q_indices + 1
    assert torch.all((ends - starts) > 0)
    return starts, ends

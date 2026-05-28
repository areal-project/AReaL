# SPDX-License-Identifier: Apache-2.0

import os

import torch

from .tilelang_sparse_mla_bwd import sparse_mla_bwd
from .tilelang_sparse_mla_fwd import sparse_mla_fwd_interface


def _pytorch_sparse_mla_bwd(q, kv, tl_out, grad_output, indices, tl_lse, scaling):
    """G11s: chunked pure-pytorch SparseMLA backward.

    Memory: materializing k_gathered at shape (S, G, TOPK, D_full) can be
    ~22 GB per rank. Chunk along S dimension (64 rows at a time ≈ 300 MB).

    Math correctness: kernel's tl_lse is log2(sum_i exp(score_i * scaling)).
    softmax prob = exp(score_i * scaling - tl_lse * ln(2)).
    """
    S, H, D_full = q.shape
    S_kv, G, _ = kv.shape
    D_v = tl_out.shape[-1]
    TOPK = indices.shape[-1]
    H_per_group = H // G
    ln2 = 0.6931471805599453

    q_f = q.float()
    kv_f = kv.float()
    do_f = grad_output.float()
    o_f = tl_out.float()

    # Precompute Delta = sum_d o * do (small, fp32)
    delta_full = (o_f * do_f).sum(dim=-1)  # (S, H)

    # Outputs
    dq = torch.zeros(S, H, D_full, device=q.device, dtype=torch.float32)
    dkv_fp32 = torch.zeros(S_kv, G, D_full, device=q.device, dtype=torch.float32)

    CHUNK = 32
    for s0 in range(0, S, CHUNK):
        s1 = min(s0 + CHUNK, S)
        cs = s1 - s0  # chunk size
        idx_c = indices[s0:s1]  # (cs, G, TOPK)
        safe_c = idx_c.clamp(min=0).long()  # (cs, G, TOPK)
        valid_c = idx_c != -1  # (cs, G, TOPK)

        # Gather K per group: (cs, G, TOPK, D_full)
        k_c = torch.stack(
            [
                kv_f[:, g, :]
                .index_select(0, safe_c[:, g, :].reshape(-1))
                .view(cs, TOPK, D_full)
                for g in range(G)
            ],
            dim=1,
        )
        # Compute per-head via per-group q split (avoid 8x replication)
        q_c = q_f[s0:s1].view(cs, G, H_per_group, D_full)  # (cs, G, Hg, D_full)
        do_c = do_f[s0:s1].view(cs, G, H_per_group, D_v)  # (cs, G, Hg, D_v)
        v_c = k_c[..., :D_v]  # (cs, G, TOPK, D_v)
        lse_c = tl_lse[s0:s1].view(cs, G, H_per_group)  # (cs, G, Hg)
        delta_c = delta_full[s0:s1].view(cs, G, H_per_group)  # (cs, G, Hg)

        # score[cs, g, hg, t] = q_c @ k_c
        score = torch.einsum("cghd,cgtd->cght", q_c, k_c) * scaling
        # valid mask broadcast
        valid_chg = valid_c.unsqueeze(2)  # (cs, G, 1, TOPK)
        score = torch.where(
            valid_chg,
            score,
            torch.tensor(-1e30, device=score.device, dtype=score.dtype),
        )
        # prob = exp(score - lse * ln(2))
        prob = torch.exp(score - lse_c.unsqueeze(-1) * ln2)
        prob = torch.where(valid_chg, prob, torch.zeros_like(prob))
        del score

        # dp_raw = do @ V^T
        dp_raw = torch.einsum("cghd,cgtd->cght", do_c, v_c)
        dp = prob * (dp_raw - delta_c.unsqueeze(-1)) * scaling
        dp = torch.where(valid_chg, dp, torch.zeros_like(dp))
        del dp_raw

        # dq[cs, g, hg, d] = sum_t dp * K
        dq_c = torch.einsum("cght,cgtd->cghd", dp, k_c).view(cs, H, D_full)
        dq[s0:s1] = dq_c
        del dq_c

        # dkv scatter per group
        for g in range(G):
            idx_g = safe_c[:, g, :].reshape(-1)  # (cs*TOPK,)
            # K contribution: (cs, Hg, TOPK) dp @ q
            dp_g = dp[:, g, :, :]  # (cs, Hg, TOPK)
            q_g = q_c[:, g, :, :]  # (cs, Hg, D_full)
            prob_g = prob[:, g, :, :]  # (cs, Hg, TOPK)
            do_g = do_c[:, g, :, :]  # (cs, Hg, D_v)
            valid_gm = valid_c[:, g, :].unsqueeze(-1)  # (cs, TOPK, 1)
            k_contrib = torch.einsum("cht,chd->ctd", dp_g, q_g)  # (cs, TOPK, D_full)
            v_contrib = torch.einsum("cht,chd->ctd", prob_g, do_g)  # (cs, TOPK, D_v)
            k_contrib = torch.where(valid_gm, k_contrib, torch.zeros_like(k_contrib))
            v_contrib = torch.where(valid_gm, v_contrib, torch.zeros_like(v_contrib))
            dkv_fp32[:, g, :].index_add_(0, idx_g, k_contrib.reshape(-1, D_full))
            dkv_fp32[:, g, :D_v].index_add_(0, idx_g, v_contrib.reshape(-1, D_v))
            del k_contrib, v_contrib
        del prob, dp, k_c, v_c

    return dq.to(q.dtype).contiguous(), dkv_fp32.contiguous()


class SparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, indices, scaling):
        """
        Args:
            q: Query tensor (seq_len, heads, dim_plus_tail_dim)
            kv: Key-Value tensor (seq_len_kv, kv_group, dim_plus_tail_dim)
            indices: Sparse indices tensor (seq_len, kv_group, topk)

        Returns:
            out: Output tensor (seq_len, heads, dim)
        """
        indices = indices.contiguous()
        q, kv = q.contiguous(), kv.contiguous()
        ctx.scaling = scaling
        tl_out, tl_lse = sparse_mla_fwd_interface(q, kv, indices, sm_scale=scaling)

        # Save tensors for backward pass
        ctx.save_for_backward(q, kv, indices, tl_out, tl_lse)

        return tl_out, tl_lse

    @staticmethod
    def backward(ctx, grad_output, grad_lse):
        """
        Args:
            grad_output: Gradient of the loss with respect to output

        Returns:
            Gradients for q, kv, and indices (None for indices)
        """
        q, kv, indices, tl_out, tl_lse = ctx.saved_tensors
        scaling = ctx.scaling

        # G11r: pure-pytorch fallback when AREAL_SPARSE_MLA_PYTORCH_BWD=1.
        # Used to bypass the L20X TileLang bwd kernel NaN bug while we figure
        # out the kernel cache invalidation problem.
        if os.environ.get("AREAL_SPARSE_MLA_PYTORCH_BWD", "0") == "1":
            tl_dq, tl_dkv = _pytorch_sparse_mla_bwd(
                q, kv, tl_out, grad_output, indices, tl_lse, scaling
            )
        else:
            tl_dq, tl_dkv = sparse_mla_bwd(
                q,
                kv,
                tl_out,
                grad_output.contiguous(),
                indices,
                tl_lse,
                sm_scale=scaling,
            )

        # Return gradients for each input (None for indices as it's not differentiable)
        return tl_dq, tl_dkv, None, None

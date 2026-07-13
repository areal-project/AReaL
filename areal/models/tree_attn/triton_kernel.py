# SPDX-License-Identifier: Apache-2.0

"""Triton-based tree attention kernel with sparse block iteration."""

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


@dataclass
class TreeAttentionData:
    packed_mask: torch.Tensor  # (B, N, ceil(N/64))
    kv_indices: torch.Tensor  # 1D sparse indices
    kv_offsets: torch.Tensor  # (B, num_q_blocks + 1)
    q_indices: torch.Tensor  # 1D sparse indices
    q_offsets: torch.Tensor  # (B, num_kv_blocks + 1)


def compute_packed_mask(fa: torch.Tensor) -> torch.Tensor:
    """
    Compute packed ancestor mask directly from parent array.

    Uses DP: for each token, inherit parent's ancestor bits and set self bit.
    Avoids materializing the full N×N boolean mask.

    Args:
        fa: Parent array of shape (B, N). fa[b, i] is parent of token i in batch b,
            or -1 for roots.

    Returns:
        packed: (B, N, ceil(N/64)) int64 tensor where bit j of packed[b, i, j//64]
                is set if token j is an ancestor of token i in batch b.
    """
    B, N = fa.shape
    num_words = (N + 63) >> 6
    packed = torch.zeros(B, N, num_words, dtype=torch.int64)

    for b in range(B):
        for i in range(N):
            parent = fa[b, i].item()
            if parent >= 0:
                parent_word = parent >> 6
                packed[b, i, : parent_word + 1] |= packed[b, parent, : parent_word + 1]
            bit_val = torch.tensor(1, dtype=torch.int64) << (i & 63)
            packed[b, i, i >> 6] |= bit_val

    return packed


def compute_kv_indices_packed(
    packed_mask: torch.Tensor,
    BLOCK_M: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute packed KV block indices for each Q block from bit-packed mask.

    Args:
        packed_mask: (B, N, num_words) int64 tensor
        BLOCK_M: Query block size

    Returns:
        kv_indices: 1D tensor of KV block indices (all batches concatenated)
        kv_offsets: (B, num_q_blocks + 1) tensor of absolute offsets into kv_indices
    """
    B, size_q, num_kv_blocks = packed_mask.shape
    num_q_blocks = (size_q + BLOCK_M - 1) // BLOCK_M

    all_indices = []
    offsets = torch.zeros(B, num_q_blocks + 1, dtype=torch.int32)

    for b in range(B):
        for qb in range(num_q_blocks):
            offsets[b, qb] = len(all_indices)
            q_start = qb * BLOCK_M
            q_end = min((qb + 1) * BLOCK_M, size_q)

            for kvb in range(num_kv_blocks):
                if packed_mask[b, q_start:q_end, kvb].any():
                    all_indices.append(kvb)

        offsets[b, num_q_blocks] = len(all_indices)

    kv_indices = torch.tensor(all_indices, dtype=torch.int32)

    return kv_indices, offsets


def compute_q_indices_packed(
    packed_mask: torch.Tensor,
    BLOCK_M: int,
    BLOCK_N: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Q block indices for each KV block (reverse mapping for backward dK/dV).

    For each KV block, returns the list of Q blocks that attend to it.

    Args:
        packed_mask: (B, N, num_words) int64 tensor
        BLOCK_M: Query block size
        BLOCK_N: KV block size (default 64)

    Returns:
        q_indices: 1D tensor of Q block indices (all batches concatenated)
        q_offsets: (B, num_kv_blocks + 1) tensor of absolute offsets into q_indices
    """
    B, size_q, num_kv_blocks = packed_mask.shape
    num_q_blocks = (size_q + BLOCK_M - 1) // BLOCK_M

    all_indices = []
    offsets = torch.zeros(B, num_kv_blocks + 1, dtype=torch.int32)

    for b in range(B):
        for kvb in range(num_kv_blocks):
            offsets[b, kvb] = len(all_indices)

            for qb in range(num_q_blocks):
                q_start = qb * BLOCK_M
                q_end = min((qb + 1) * BLOCK_M, size_q)

                if packed_mask[b, q_start:q_end, kvb].any():
                    all_indices.append(qb)

        offsets[b, num_kv_blocks] = len(all_indices)

    q_indices = torch.tensor(all_indices, dtype=torch.int32)

    return q_indices, offsets


def precompute_tree_attention_data(
    fa: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 64,
) -> TreeAttentionData:
    """
    Precompute packed mask, KV indices, and Q indices from parent array.

    Args:
        fa: Parent array of shape (B, N). fa[b, i] is parent of token i in batch b,
            or -1 for roots.
        BLOCK_M: Query block size (default 128)
        BLOCK_N: KV block size (default 64)

    Returns:
        TreeAttentionData object containing all precomputed tensors.
    """
    packed_mask = compute_packed_mask(fa)
    kv_indices, kv_offsets = compute_kv_indices_packed(packed_mask, BLOCK_M)
    q_indices, q_offsets = compute_q_indices_packed(packed_mask, BLOCK_M, BLOCK_N)
    return TreeAttentionData(
        packed_mask=packed_mask,
        kv_indices=kv_indices,
        kv_offsets=kv_offsets,
        q_indices=q_indices,
        q_offsets=q_offsets,
    )


if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        ],
        key=["N", "HEAD_DIM"],
    )
    @triton.jit
    def _tree_attn_fwd_triton(
        Q,
        K,
        V,
        o,
        LSE,
        packed_mask,
        kv_indices,
        kv_offsets,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_on,
        stride_od,
        stride_lse_b,
        stride_lse_h,
        stride_lse_n,
        stride_mask_b,
        stride_mask_q,
        stride_mask_k,
        stride_off_b,
        stride_off_q,
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        GQA_GROUP_SIZE: tl.constexpr,
    ):
        """
        Tree attention forward kernel with dense bit-packed mask.

        Grid: (cdiv(N, BLOCK_M), B * H)
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        off_b = pid_bh // H
        off_h = pid_bh % H
        # Handle GQA: map query head to key/value head
        off_h_kv = off_h // GQA_GROUP_SIZE

        qo_offset = off_b * stride_qb + off_h * stride_qh
        kv_offset = off_b * stride_kb + off_h_kv * stride_kh

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        # Load Q block
        q_ptrs = (
            Q + qo_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)

        # Get KV block range for this Q block (batch-indexed)
        kv_start_idx = tl.load(kv_offsets + off_b * stride_off_b + pid_m * stride_off_q)
        kv_end_idx = tl.load(
            kv_offsets + off_b * stride_off_b + (pid_m + 1) * stride_off_q
        )
        num_kv = kv_end_idx - kv_start_idx

        # Initialize online softmax accumulators
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        qk_scale = sm_scale * 1.44269504

        # Iterate over valid KV blocks only
        for i in range(num_kv):
            kv_block_idx = tl.load(kv_indices + kv_start_idx + i)
            n_start = kv_block_idx * BLOCK_N
            offs_n = n_start + tl.arange(0, BLOCK_N)

            # Load K, V blocks
            k_ptrs = (
                K
                + kv_offset
                + offs_n[:, None] * stride_kn
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=offs_n[:, None] < N, other=0.0)

            v_ptrs = (
                V
                + kv_offset
                + offs_n[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)

            # Compute QK^T
            qk = tl.dot(q, tl.trans(k))

            # Load and unpack mask block (BLOCK_N=64, one int64 word per block)
            mask_ptrs = (
                packed_mask
                + off_b * stride_mask_b
                + offs_m * stride_mask_q
                + kv_block_idx * stride_mask_k
            )
            mask_word = tl.load(mask_ptrs, mask=offs_m < N, other=0)  # (BLOCK_M,)

            # Unpack 64 bits
            bit_indices = tl.arange(0, BLOCK_N)
            mask = ((mask_word[:, None] >> bit_indices[None, :]) & 1) != 0

            # Apply bounds mask
            valid_mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
            mask = mask & valid_mask

            # Apply mask to attention scores
            # Avoid -inf since -inf - (-inf) = NaN
            qk = tl.where(mask, qk * qk_scale, -16384.0)

            # Online softmax update
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)

            alpha = tl.math.exp2(m_i - m_new)
            acc = acc * alpha[:, None]
            l_i = l_i * alpha

            p = tl.math.exp2(qk - m_new[:, None])
            l_ij = tl.sum(p, axis=1)
            l_i = l_i + l_ij

            p = p.to(v.dtype)
            acc = acc + tl.dot(p, v)

            m_i = m_new

        # Normalize
        acc = acc / l_i[:, None]

        # Store output
        o_ptrs = (
            o + qo_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
        )
        tl.store(o_ptrs, acc.to(o.dtype.element_ty), mask=offs_m[:, None] < N)

        # Store LSE for backward: lse = m + log2(l)
        lse = m_i + tl.math.log2(l_i)
        lse_ptrs = (
            LSE + off_b * stride_lse_b + off_h * stride_lse_h + offs_m * stride_lse_n
        )
        tl.store(lse_ptrs, lse, mask=offs_m < N)

    def tree_attention_triton(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        packed_mask: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_offsets: torch.Tensor,
        sm_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tree attention forward using precomputed dense bit-packed mask with block skipping.

        Returns output and LSE for backward pass.
        """
        B, H, N, D = query.shape
        # Handle GQA
        assert key.shape[0] == B and key.shape[2] == N and key.shape[3] == D
        assert value.shape == key.shape
        H_kv = key.shape[1]
        assert H % H_kv == 0, f"Query heads {H} must be divisible by KV heads {H_kv}"
        gqa_group_size = H // H_kv

        BLOCK_M = 128
        # BLOCK_N is fixed to 64 in the kernel

        if sm_scale is None:
            sm_scale = 1.0 / (D**0.5)

        packed_mask = packed_mask.to(query.device).contiguous()
        kv_indices = kv_indices.to(query.device).contiguous()
        kv_offsets = kv_offsets.to(query.device).contiguous()

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        output = torch.empty_like(query)
        lse = torch.empty(B, H, N, device=query.device, dtype=torch.float32)

        grid = (triton.cdiv(N, BLOCK_M), B * H)

        _tree_attn_fwd_triton[grid](
            query,
            key,
            value,
            output,
            lse,
            packed_mask,
            kv_indices,
            kv_offsets,
            sm_scale,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            packed_mask.stride(0),
            packed_mask.stride(1),
            packed_mask.stride(2),
            kv_offsets.stride(0),
            kv_offsets.stride(1),
            B=B,
            H=H,
            N=N,
            HEAD_DIM=D,
            GQA_GROUP_SIZE=gqa_group_size,
        )

        return output, lse

    @triton.jit
    def _tree_attn_bwd_preprocess(
        o,
        DO,
        Delta,
        stride_oh,
        stride_on,
        stride_od,
        N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        o_ptrs = (
            o
            + pid_bh * stride_oh
            + offs_m[:, None] * stride_on
            + offs_d[None, :] * stride_od
        )
        do_ptrs = (
            DO
            + pid_bh * stride_oh
            + offs_m[:, None] * stride_on
            + offs_d[None, :] * stride_od
        )

        o = tl.load(o_ptrs, mask=offs_m[:, None] < N, other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=offs_m[:, None] < N, other=0.0).to(tl.float32)

        delta = tl.sum(o * do, axis=1)

        delta_ptrs = Delta + pid_bh * N + offs_m
        tl.store(delta_ptrs, delta, mask=offs_m < N)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        ],
        key=["N", "HEAD_DIM"],
    )
    @triton.jit
    def _tree_attn_bwd_dq(
        Q,
        K,
        V,
        DO,
        DQ,
        LSE,
        Delta,
        packed_mask,
        kv_indices,
        kv_offsets,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_dob,
        stride_doh,
        stride_don,
        stride_dod,
        stride_dqb,
        stride_dqh,
        stride_dqn,
        stride_dqd,
        stride_lse_b,
        stride_lse_h,
        stride_lse_n,
        stride_mask_b,
        stride_mask_q,
        stride_mask_k,
        stride_off_b,
        stride_off_q,
        B: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        GQA_GROUP_SIZE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        off_b = pid_bh // H
        off_h = pid_bh % H

        qo_offset = off_b * stride_qb + off_h * stride_qh
        # Handle GQA mapping
        off_h_kv = off_h // GQA_GROUP_SIZE
        kv_offset = off_b * stride_kb + off_h_kv * stride_kh

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        q = tl.load(
            Q + qo_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
            mask=offs_m[:, None] < N,
            other=0.0,
        )
        do = tl.load(
            DO
            + off_b * stride_dob
            + off_h * stride_doh
            + offs_m[:, None] * stride_don
            + offs_d[None, :] * stride_dod,
            mask=offs_m[:, None] < N,
            other=0.0,
        )
        lse = tl.load(
            LSE + off_b * stride_lse_b + off_h * stride_lse_h + offs_m * stride_lse_n,
            mask=offs_m < N,
            other=0.0,
        )
        delta = tl.load(Delta + pid_bh * N + offs_m, mask=offs_m < N, other=0.0)

        kv_start = tl.load(kv_offsets + off_b * stride_off_b + pid_m * stride_off_q)
        kv_end = tl.load(kv_offsets + off_b * stride_off_b + (pid_m + 1) * stride_off_q)
        num_kv = kv_end - kv_start

        dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale * 1.44269504

        for i in range(num_kv):
            kv_block_idx = tl.load(kv_indices + kv_start + i)
            offs_n = kv_block_idx * BLOCK_N + tl.arange(0, BLOCK_N)

            k = tl.load(
                K
                + kv_offset
                + offs_n[:, None] * stride_kn
                + offs_d[None, :] * stride_kd,
                mask=offs_n[:, None] < N,
                other=0.0,
            )
            v = tl.load(
                V
                + kv_offset
                + offs_n[:, None] * stride_vn
                + offs_d[None, :] * stride_vd,
                mask=offs_n[:, None] < N,
                other=0.0,
            )

            mask_word = tl.load(
                packed_mask
                + off_b * stride_mask_b
                + offs_m * stride_mask_q
                + kv_block_idx * stride_mask_k,
                mask=offs_m < N,
                other=0,
            )
            bit_indices = tl.arange(0, BLOCK_N)
            mask = ((mask_word[:, None] >> bit_indices[None, :]) & 1) != 0
            mask = mask & ((offs_m[:, None] < N) & (offs_n[None, :] < N))

            qk = tl.dot(q, tl.trans(k))
            qk = tl.where(mask, qk * qk_scale, -16384.0)
            p = tl.math.exp2(qk - lse[:, None])
            p = tl.where(mask, p, 0.0)

            dp = tl.dot(do.to(v.dtype), tl.trans(v)).to(tl.float32)
            ds = p * (dp - delta[:, None])
            ds = tl.where(mask, ds, 0.0)

            dq += tl.dot(ds.to(k.dtype), k).to(tl.float32)

        dq *= sm_scale
        tl.store(
            DQ
            + off_b * stride_dqb
            + off_h * stride_dqh
            + offs_m[:, None] * stride_dqn
            + offs_d[None, :] * stride_dqd,
            dq.to(DQ.dtype.element_ty),
            mask=offs_m[:, None] < N,
        )

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        ],
        key=["N", "HEAD_DIM"],
    )
    @triton.jit
    def _tree_attn_bwd_dkdv(
        Q,
        K,
        V,
        DO,
        DK,
        DV,
        LSE,
        Delta,
        packed_mask,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_qn,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_dob,
        stride_doh,
        stride_don,
        stride_dod,
        stride_dkb,
        stride_dkh,
        stride_dkn,
        stride_dkd,
        stride_dvb,
        stride_dvh,
        stride_dvn,
        stride_dvd,
        stride_lse_b,
        stride_lse_h,
        stride_lse_n,
        stride_mask_b,
        stride_mask_q,
        stride_mask_k,
        B: tl.constexpr,
        H: tl.constexpr,
        H_KV: tl.constexpr,
        N: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_Q_BLOCKS: tl.constexpr,
        GQA_GROUP_SIZE: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_bh_kv = tl.program_id(1)
        off_b = pid_bh_kv // H_KV
        off_h_kv = pid_bh_kv % H_KV

        kv_offset = off_b * stride_kb + off_h_kv * stride_kh

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)

        k = tl.load(
            K + kv_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=offs_n[:, None] < N,
            other=0.0,
        )
        v = tl.load(
            V + kv_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=offs_n[:, None] < N,
            other=0.0,
        )

        dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
        qk_scale = sm_scale * 1.44269504

        # Loop over Q heads in the group
        for g in range(GQA_GROUP_SIZE):
            off_h = off_h_kv * GQA_GROUP_SIZE + g
            qo_offset = off_b * stride_qb + off_h * stride_qh

            # Loop over Q blocks
            for q_block_idx in range(NUM_Q_BLOCKS):
                offs_m = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)

                mask_word = tl.load(
                    packed_mask
                    + off_b * stride_mask_b
                    + offs_m * stride_mask_q
                    + pid_n * stride_mask_k,
                    mask=offs_m < N,
                    other=0,
                )
                has_nonzero = tl.sum(mask_word) != 0
                if has_nonzero:
                    q = tl.load(
                        Q
                        + qo_offset
                        + offs_m[:, None] * stride_qn
                        + offs_d[None, :] * stride_qd,
                        mask=offs_m[:, None] < N,
                        other=0.0,
                    )
                    do = tl.load(
                        DO
                        + off_b * stride_dob
                        + off_h * stride_doh
                        + offs_m[:, None] * stride_don
                        + offs_d[None, :] * stride_dod,
                        mask=offs_m[:, None] < N,
                        other=0.0,
                    )
                    lse = tl.load(
                        LSE
                        + off_b * stride_lse_b
                        + off_h * stride_lse_h
                        + offs_m * stride_lse_n,
                        mask=offs_m < N,
                        other=0.0,
                    )
                    delta = tl.load(
                        Delta + (off_b * H + off_h) * N + offs_m,
                        mask=offs_m < N,
                        other=0.0,
                    )

                    bit_indices = tl.arange(0, BLOCK_N)
                    mask = ((mask_word[:, None] >> bit_indices[None, :]) & 1) != 0
                    mask = mask & ((offs_m[:, None] < N) & (offs_n[None, :] < N))

                    qk = tl.dot(q, tl.trans(k))
                    qk = tl.where(mask, qk * qk_scale, -16384.0)
                    p = tl.math.exp2(qk - lse[:, None])
                    p = tl.where(mask, p, 0.0)

                    dv += tl.dot(tl.trans(p.to(do.dtype)), do).to(tl.float32)

                    dp = tl.dot(do.to(v.dtype), tl.trans(v)).to(tl.float32)
                    ds = p * (dp - delta[:, None])
                    ds = tl.where(mask, ds, 0.0)

                    dk += tl.dot(tl.trans(ds.to(q.dtype)), q).to(tl.float32)

        dk *= sm_scale
        tl.store(
            DK
            + off_b * stride_dkb
            + off_h_kv * stride_dkh
            + offs_n[:, None] * stride_dkn
            + offs_d[None, :] * stride_dkd,
            dk.to(DK.dtype.element_ty),
            mask=offs_n[:, None] < N,
        )
        tl.store(
            DV
            + off_b * stride_dvb
            + off_h_kv * stride_dvh
            + offs_n[:, None] * stride_dvn
            + offs_d[None, :] * stride_dvd,
            dv.to(DV.dtype.element_ty),
            mask=offs_n[:, None] < N,
        )

    def tree_attention_backward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
        dout: torch.Tensor,
        packed_mask: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_offsets: torch.Tensor,
        q_indices: torch.Tensor,
        q_offsets: torch.Tensor,
        sm_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H, N, D = query.shape
        H_kv = key.shape[1]
        gqa_group_size = H // H_kv
        BLOCK_M = 128
        BLOCK_N = 64

        dout = dout.contiguous()
        dq = torch.zeros_like(query)
        dk = torch.zeros_like(key)
        dv = torch.zeros_like(value)
        delta = torch.empty(B * H, N, device=query.device, dtype=torch.float32)

        pre_grid = (triton.cdiv(N, BLOCK_M), B * H)
        _tree_attn_bwd_preprocess[pre_grid](
            output,
            dout,
            delta,
            output.stride(1),
            output.stride(2),
            output.stride(3),
            N=N,
            HEAD_DIM=D,
            BLOCK_M=BLOCK_M,
            num_warps=4,
        )

        num_kv_blocks = triton.cdiv(N, BLOCK_N)
        num_q_blocks = triton.cdiv(N, BLOCK_M)

        dq_grid = (num_q_blocks, B * H)
        _tree_attn_bwd_dq[dq_grid](
            query,
            key,
            value,
            dout,
            dq,
            lse,
            delta,
            packed_mask,
            kv_indices,
            kv_offsets,
            sm_scale,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            packed_mask.stride(0),
            packed_mask.stride(1),
            packed_mask.stride(2),
            kv_offsets.stride(0),
            kv_offsets.stride(1),
            B=B,
            H=H,
            N=N,
            HEAD_DIM=D,
            GQA_GROUP_SIZE=gqa_group_size,
        )

        dkdv_grid = (num_kv_blocks, B * H_kv)
        _tree_attn_bwd_dkdv[dkdv_grid](
            query,
            key,
            value,
            dout,
            dk,
            dv,
            lse,
            delta,
            packed_mask,
            sm_scale,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            packed_mask.stride(0),
            packed_mask.stride(1),
            packed_mask.stride(2),
            B=B,
            H=H,
            H_KV=H_kv,
            N=N,
            HEAD_DIM=D,
            NUM_Q_BLOCKS=num_q_blocks,
            GQA_GROUP_SIZE=gqa_group_size,
        )

        return dq, dk, dv

    class TreeAttentionFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            query,
            key,
            value,
            packed_mask,
            kv_indices,
            kv_offsets,
            q_indices,
            q_offsets,
            sm_scale,
        ):
            output, lse = tree_attention_triton(
                query, key, value, packed_mask, kv_indices, kv_offsets, sm_scale
            )
            ctx.save_for_backward(
                query,
                key,
                value,
                output,
                lse,
                packed_mask,
                kv_indices,
                kv_offsets,
                q_indices,
                q_offsets,
            )
            ctx.sm_scale = sm_scale
            return output

        @staticmethod
        def backward(ctx, dout):
            (
                query,
                key,
                value,
                output,
                lse,
                packed_mask,
                kv_indices,
                kv_offsets,
                q_indices,
                q_offsets,
            ) = ctx.saved_tensors
            dq, dk, dv = tree_attention_backward(
                query,
                key,
                value,
                output,
                lse,
                dout,
                packed_mask,
                kv_indices,
                kv_offsets,
                q_indices,
                q_offsets,
                ctx.sm_scale,
            )
            return dq, dk, dv, None, None, None, None, None, None

    def tree_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        packed_mask: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_offsets: torch.Tensor,
        q_indices: torch.Tensor,
        q_offsets: torch.Tensor,
        sm_scale: float | None = None,
    ) -> torch.Tensor:
        """
        Tree attention with backward support.

        Args:
            query, key, value: (B, H, N, D) tensors
            packed_mask, kv_indices, kv_offsets, q_indices, q_offsets: from precompute_tree_attention_data
            sm_scale: softmax scale (default: 1/sqrt(D))
        """
        if sm_scale is None:
            sm_scale = 1.0 / (query.shape[-1] ** 0.5)
        packed_mask = packed_mask.to(query.device).contiguous()
        kv_indices = kv_indices.to(query.device).contiguous()
        kv_offsets = kv_offsets.to(query.device).contiguous()
        q_indices = q_indices.to(query.device).contiguous()
        q_offsets = q_offsets.to(query.device).contiguous()
        return TreeAttentionFunction.apply(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            packed_mask,
            kv_indices,
            kv_offsets,
            q_indices,
            q_offsets,
            sm_scale,
        )

else:

    def tree_attention(*args, **kwargs):
        raise ImportError(
            "Triton is not available. Please install triton to use tree_attention."
        )

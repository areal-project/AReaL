# SPDX-License-Identifier: Apache-2.0
"""AscendC forward for MindSpeed's GDN autograd and packed metadata contract."""

from collections.abc import Callable, Sequence
from types import ModuleType


def make_native_kkt_forward(owner: ModuleType, native: ModuleType) -> Callable:
    """Keep KKT and solve outputs head-first through the remaining operators.

    MindSpeed retains ownership of metadata preparation, autograd and CP.
    The existing backward consumes the same head-first saved matrix.
    """
    import torch

    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor | None,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_list: Sequence[int] | None,
        chunk_indices: Sequence[int] | None,
        chunk_indices_out: dict[str, torch.Tensor] | None,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if q.dtype not in (torch.float16, torch.bfloat16) or not (
            q.dtype == k.dtype == v.dtype
        ):
            raise ValueError("AscendC GDN requires matching FP16 or BF16 q, k and v.")
        if chunk_size not in (64, 128):
            raise ValueError("AscendC GDN requires chunk_size=64 or 128.")
        g_cum = owner.chunk_local_cumsum(
            g,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices_out=chunk_indices_out,
            head_first=False,
        )
        k_h = k.transpose(1, 2).contiguous()
        g_h = g_cum.transpose(1, 2).contiguous()
        beta_h = beta.transpose(1, 2).contiguous().float()
        A_h = native.npu_chunk_scaled_dot_kkt(
            k_h,
            g_h,
            beta_h,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )
        if cu_seqlens_list is None:
            A_h = owner.npu_solve_tri(A_h.to(k.dtype), layout="bnsd")
        else:
            A_h = owner.npu_solve_tri(
                A_h.squeeze(0).to(k.dtype),
                cu_seqlens=cu_seqlens_list,
                chunk_indices=chunk_indices,
                layout="ntd",
            ).unsqueeze(0)

        q_h = q.transpose(1, 2).contiguous()
        v_h = v.transpose(1, 2).contiguous()
        w, u = owner.npu_recompute_w_u_fwd(
            k_h,
            v_h,
            beta_h,
            A_h,
            chunk_size,
            g=g_h,
            gk=None,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_indices,
        )
        h, v_new, final_state = owner.npu_chunk_gated_delta_rule_fwd_h(
            k_h,
            w,
            u,
            g=g_h,
            gk=None,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_indices,
        )
        if not output_final_state:
            final_state = None
        o_h = owner.npu_chunk_fwd_o(
            q_h,
            k_h,
            v_new,
            h,
            scale,
            g=g_h,
            g_gamma=None,
            cu_seqlens=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
            transpose_state_layout=False,
        )
        return g_cum, o_h.transpose(1, 2).contiguous(), A_h, final_state

    forward._areal_native_kkt = True
    return forward

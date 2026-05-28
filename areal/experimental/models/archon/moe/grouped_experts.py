# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/models/moe/moe.py

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from areal.experimental.models.archon.moe.utils import indices_padding_wrapper


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Execute expert computation using for-loop (reference implementation).

    This implementation is slower but works without grouped_mm support.
    It processes each expert's tokens sequentially.

    Args:
        w1: Gate projection weights, shape (num_experts, hidden_dim, dim).
        w2: Down projection weights, shape (num_experts, dim, hidden_dim).
        w3: Up projection weights, shape (num_experts, hidden_dim, dim).
        x: Input tokens sorted by expert, shape (total_tokens, dim).
        num_tokens_per_expert: Number of tokens for each expert.

    Returns:
        Output tensor, shape (total_tokens, dim).
    """
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()
    total_tokens = sum(num_tokens_per_expert_list)

    # Split input by expert
    x_splits = torch.split(
        x[:total_tokens],
        split_size_or_sections=[int(n) for n in num_tokens_per_expert_list],
        dim=0,
    )

    out_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        if x_expert.shape[0] == 0:
            # Empty expert, skip
            out_splits.append(x_expert.new_empty(0, w2.shape[1]))
            continue

        # SwiGLU: silu(x @ w1.T) * (x @ w3.T) @ w2.T
        h = F.silu(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        out_splits.append(h)

    out = torch.cat(out_splits, dim=0)

    # Pad output if input was padded (handles alignment padding case)
    if x.shape[0] > total_tokens:
        padding = x.new_zeros((x.shape[0] - total_tokens, out.shape[-1]))
        out = torch.cat([out, padding], dim=0)

    return out


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Execute expert computation using torch._grouped_mm.

    This is the efficient implementation that uses grouped matrix multiplication
    to batch computation across all experts.

    Note: Requires torch._grouped_mm to be available (PyTorch 2.4+).

    Args:
        w1: Gate projection weights, shape (num_experts, hidden_dim, dim).
        w2: Down projection weights, shape (num_experts, dim, hidden_dim).
        w3: Up projection weights, shape (num_experts, hidden_dim, dim).
        x: Input tokens sorted by expert with alignment padding.
        num_tokens_per_expert: Aligned number of tokens for each expert.

    Returns:
        Output tensor with same shape as input x.
    """
    # Compute cumulative offsets for grouped_mm
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    # Cast to bfloat16 for grouped_mm.
    # Note: torch._grouped_mm currently only supports bfloat16 on CUDA.
    # bf16bf16_grouped_mm is optimized CUTLASS kernel.
    # NOTE: Upgrading PyTorch may resolve this in the future.
    x_bf16 = x.bfloat16()
    w1_bf16 = w1.bfloat16().transpose(-2, -1)
    w2_bf16 = w2.bfloat16().transpose(-2, -1)
    w3_bf16 = w3.bfloat16().transpose(-2, -1)

    # SwiGLU: silu(x @ w1.T) * (x @ w3.T) @ w2.T
    h = F.silu(torch._grouped_mm(x_bf16, w1_bf16, offs=offsets))
    h = h * torch._grouped_mm(x_bf16, w3_bf16, offs=offsets)
    out = torch._grouped_mm(h, w2_bf16, offs=offsets)

    return out.type_as(x)


def _run_experts_fp8_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    block_size: int = 128,
    use_triton: bool = True,
) -> torch.Tensor:
    """Per-expert FP8 for-loop fallback (no grouped_mm FP8 support yet).

    .. note:: Performance

       ``.tolist()`` forces a GPU→CPU sync every forward call.  The default
       BF16 path uses ``grouped_mm`` which keeps token counts on-device.
       This overhead is inherent to the per-expert for-loop approach and
       will be removed when ``grouped_mm`` gains FP8 support.
    """
    from torchao.prototype.blockwise_fp8_training.linear import fp8_blockwise_mm

    # GPU→CPU sync — see docstring.
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()
    total_tokens = sum(num_tokens_per_expert_list)

    x_splits = torch.split(
        x[:total_tokens],
        split_size_or_sections=[int(n) for n in num_tokens_per_expert_list],
        dim=0,
    )

    out_splits = []
    for expert_idx, x_expert in enumerate(x_splits):
        n_tokens = x_expert.shape[0]
        if n_tokens == 0:
            out_splits.append(x_expert.new_empty(0, w2.shape[1]))
            continue

        pad_size = (block_size - n_tokens % block_size) % block_size
        if pad_size > 0:
            x_expert = F.pad(x_expert, (0, 0, 0, pad_size))

        x_e = x_expert.contiguous()
        w1_e = w1[expert_idx].contiguous()
        w2_e = w2[expert_idx].contiguous()
        w3_e = w3[expert_idx].contiguous()

        # SwiGLU: silu(x @ w1.T) * (x @ w3.T) @ w2.T
        h1 = fp8_blockwise_mm.apply(x_e, w1_e, block_size, x_e.dtype, use_triton)
        h3 = fp8_blockwise_mm.apply(x_e, w3_e, block_size, x_e.dtype, use_triton)
        assert isinstance(h1, torch.Tensor)
        assert isinstance(h3, torch.Tensor)
        h = F.silu(h1) * h3
        h2 = fp8_blockwise_mm.apply(h, w2_e, block_size, h.dtype, use_triton)
        assert isinstance(h2, torch.Tensor)
        h = h2

        if pad_size > 0:
            h = h[:n_tokens]

        out_splits.append(h)

    out = torch.cat(out_splits, dim=0)

    if x.shape[0] > total_tokens:
        padding = x.new_zeros((x.shape[0] - total_tokens, out.shape[-1]))
        out = torch.cat([out, padding], dim=0)

    return out


def _check_grouped_mm_available() -> bool:
    """Check if torch._grouped_mm is available and functional.

    Note: grouped_mm requires CUDA. It exists in PyTorch 2.4+ but only works on CUDA.
    """
    if not hasattr(torch, "_grouped_mm"):
        return False
    # Also check if CUDA is available, as grouped_mm only works on CUDA
    return torch.cuda.is_available()


class GroupedExperts(nn.Module):
    """Grouped experts module with 3D weight tensors.

    This module stores expert weights in 3D tensors (num_experts, hidden_dim, dim)
    which enables efficient computation using grouped matrix multiplication.

    FP8 support is enabled externally via :func:`enable_fp8_experts` in
    ``areal.experimental.models.archon.fp8``, which patches :meth:`forward`
    with an FP8-aware implementation (following the same construct-then-transform
    pattern used by :func:`enable_fp8_linear` for ``nn.Linear``).

    Args:
        dim: Input/output dimension.
        hidden_dim: Hidden dimension of the feedforward network.
        num_experts: Number of experts.
        use_grouped_mm: Whether to use grouped_mm (True) or for-loop (False).
            Falls back to for-loop if grouped_mm is not available.

    Attributes:
        w1: Gate projection weights, shape (num_experts, hidden_dim, dim).
        w2: Down projection weights, shape (num_experts, dim, hidden_dim).
        w3: Up projection weights, shape (num_experts, hidden_dim, dim).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.use_grouped_mm = use_grouped_mm and _check_grouped_mm_available()

        # 3D weight tensors: (num_experts, out_features, in_features)
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))

    def _get_local_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (w1, w2, w3), converting from DTensor to local if needed."""
        if isinstance(self.w1, DTensor):
            return self.w1.to_local(), self.w2.to_local(), self.w3.to_local()
        return self.w1, self.w2, self.w3

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through grouped experts.

        Args:
            x: Input tokens sorted by expert assignment.
               Shape (total_tokens, dim) or (total_padded_tokens, dim) if using grouped_mm.
            num_tokens_per_expert: Number of tokens assigned to each expert.
               Shape (num_experts,). For grouped_mm, should be aligned to the required size.

        Returns:
            Output tensor with same shape as input x.
        """
        w1, w2, w3 = self._get_local_weights()

        if self.use_grouped_mm:
            # If EP is not used, apply padding wrapper;
            # otherwise, EP hooks already handle padding.
            if (
                not isinstance(self.w1, DTensor)
                or "ep" not in self.w1.device_mesh.mesh_dim_names
            ):
                run_experts_fn = indices_padding_wrapper(_run_experts_grouped_mm)
            else:
                run_experts_fn = _run_experts_grouped_mm
            return run_experts_fn(w1, w2, w3, x, num_tokens_per_expert)
        else:
            return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)

    def init_weights(self, init_std: float = 0.02):
        """Initialize weights using truncated normal distribution.

        Args:
            init_std: Standard deviation for w2 and w3. w1 uses 0.02.
        """
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


__all__ = ["GroupedExperts"]

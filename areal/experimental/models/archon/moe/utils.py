# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/models/moe/utils.py and moe.py

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

from areal.experimental.models.archon.moe.kernels import generate_permute_indices


def _round_up(x: int, multiple: int) -> int:
    """Round up x to the nearest multiple."""
    return ((x + multiple - 1) // multiple) * multiple


def permute_tokens(
    tokens: torch.Tensor,
    selected_experts_indices: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder tokens by expert assignment for efficient grouped computation.

    Given tokens and their assigned expert indices, this function:
    1. Sorts tokens so that all tokens for expert 0 come first, then expert 1, etc.
    2. Returns indices that can be used to restore the original order.

    Args:
        tokens: Input tokens with shape (num_tokens, dim).
        selected_experts_indices: Expert indices for each token, shape (num_tokens, top_k).
        num_experts: Total number of experts.

    Returns:
        tuple containing:
            - permuted_tokens: Tokens reordered by expert, shape (num_tokens * top_k, dim).
            - sorted_indices: Indices that map from original to sorted order.
            - num_tokens_per_expert: Number of tokens assigned to each expert.

    Example:
        >>> tokens = torch.randn(4, 64)  # 4 tokens, dim=64
        >>> indices = torch.tensor([[0, 1], [2, 0], [1, 2], [0, 1]])  # top_k=2
        >>> permuted, sorted_idx, counts = permute_tokens(tokens, indices, num_experts=3)
        >>> # counts might be [3, 3, 2] - experts 0,1,2 each get some tokens
    """
    top_k = selected_experts_indices.shape[1]

    # Flatten expert indices: (num_tokens, top_k) -> (num_tokens * top_k,)
    flat_indices = selected_experts_indices.view(-1)

    # Sort by expert index to group tokens by expert
    # stable=True ensures deterministic ordering for tokens assigned to same expert
    sorted_indices = torch.argsort(flat_indices, stable=True)

    # Expand tokens for top_k routing: each token is used top_k times
    # (num_tokens, dim) -> (num_tokens * top_k, dim)
    expanded_tokens = (
        tokens.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, tokens.shape[-1])
    )

    # Permute tokens according to sorted order
    permuted_tokens = expanded_tokens[sorted_indices]

    # Count tokens per expert using histogram
    num_tokens_per_expert = torch.histc(
        flat_indices,
        bins=num_experts,
        min=0,
        max=num_experts,
    )

    return permuted_tokens, sorted_indices, num_tokens_per_expert


def unpermute_tokens(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    num_tokens: int,
    top_k: int,
) -> torch.Tensor:
    """Restore tokens to their original order after expert computation.

    This is the inverse of permute_tokens. It takes the expert-sorted tokens
    and restores them to match the original token ordering.

    Args:
        permuted_tokens: Tokens in expert-sorted order, shape (num_tokens * top_k, dim).
        sorted_indices: Indices from permute_tokens that define the sorting.
        num_tokens: Original number of tokens (before top_k expansion).
        top_k: Number of experts each token was routed to.

    Returns:
        Tokens in original order, shape (num_tokens * top_k, dim).
    """
    # Create output tensor
    unpermuted = torch.empty_like(permuted_tokens)

    # Scatter back to original positions
    unpermuted[sorted_indices] = permuted_tokens

    return unpermuted


def merge_expert_outputs(
    unpermuted_tokens: torch.Tensor,
    scores: torch.Tensor,
    num_tokens: int,
    top_k: int,
    score_before_experts: bool = True,
) -> torch.Tensor:
    """Merge outputs from multiple experts for each token using routing scores.

    Args:
        unpermuted_tokens: Expert outputs in original order, shape (num_tokens * top_k, dim).
        scores: Routing scores for each (token, expert) pair, shape (num_tokens, top_k).
        num_tokens: Number of tokens.
        top_k: Number of experts each token was routed to.
        score_before_experts: If True, scores were applied before expert computation,
            so we just sum outputs. If False, apply scores now via weighted sum.

    Returns:
        Merged token outputs, shape (num_tokens, dim).
    """
    dim = unpermuted_tokens.shape[-1]

    # Reshape to (num_tokens, top_k, dim)
    reshaped = unpermuted_tokens.reshape(num_tokens, top_k, dim)

    if score_before_experts:
        # Scores were already applied, just sum
        return reshaped.sum(dim=1)
    else:
        # Apply scores now: weighted sum over top_k experts
        # scores: (num_tokens, top_k) -> (num_tokens, 1, top_k)
        # reshaped: (num_tokens, top_k, dim)
        # bmm result: (num_tokens, 1, dim) -> (num_tokens, dim)
        return (
            torch.bmm(
                scores.unsqueeze(1).float(),
                reshaped.float(),
            )
            .squeeze(1)
            .to(unpermuted_tokens.dtype)
        )


# Constants for grouped_mm alignment
TOKEN_GROUP_ALIGN_SIZE_M = 8
ValidTokenGroupAlignmentSize = Literal[8, 16, 32]


def set_token_group_alignment_size(
    alignment_size: ValidTokenGroupAlignmentSize,
) -> None:
    """Set the token group alignment size for grouped_mm.

    Valid values are: 8, 16, or 32.
    Different values are needed for different cases:

    * For bf16, 8 is enough (16 byte alignment / 2 bytes per elem = 8 elements).
    * For fp8, 16 byte alignment / 1 byte per elem = 16 elements.
    * For mxfp8, we need 32 (or block_size) because scaling block size is (1 x 32),
      so when doing per-token-group quantization on each logically distinct subtensor,
      we need to ensure the contracting dim is divisible by block_size.
      In the backward pass, grad_weight = (grad_output_t @ input).t() has gemm dims
      of (N, M) @ (M, K) so M is the contracting dim, and group offsets are along M,
      so we need 32 element alignment.

    Args:
        alignment_size: Alignment size in number of elements.
    """
    global TOKEN_GROUP_ALIGN_SIZE_M
    TOKEN_GROUP_ALIGN_SIZE_M = alignment_size


def _permute(
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    ep_degree: int,
    num_local_experts: int,
) -> tuple[tuple[int, ...], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute and pad tokens for grouped_mm alignment.

    Used by both EP (via ExpertParallel) and non-EP (via indices_padding_wrapper).
    This function reorders tokens by expert and adds alignment padding needed
    for torch._grouped_mm.

    Args:
        x: Input tokens, shape (num_tokens, dim).
        num_tokens_per_expert: Token counts per expert from all ranks,
            shape (ep_degree * num_local_experts,).
        ep_degree: Expert parallelism degree (number of EP ranks).
        num_local_experts: Number of experts on each EP rank.

    Returns:
        tuple containing:
            - input_shape: Original input shape with padding row.
            - x: Permuted and padded tokens.
            - permuted_indices: Indices for unpermuting.
            - num_tokens_per_expert: Aligned token counts per expert.
    """
    global TOKEN_GROUP_ALIGN_SIZE_M
    x_padded_per_expert = x.shape[0] + num_local_experts * TOKEN_GROUP_ALIGN_SIZE_M
    padded_max_len = _round_up(x_padded_per_expert, TOKEN_GROUP_ALIGN_SIZE_M)
    with torch.no_grad():
        (
            permuted_indices,
            num_tokens_per_expert,
            _offsets,
        ) = generate_permute_indices(
            num_tokens_per_expert,
            num_local_experts,
            ep_degree,
            padded_max_len,
            TOKEN_GROUP_ALIGN_SIZE_M,
        )

    x = torch.vstack((x, x.new_zeros(x.shape[-1])))
    input_shape = x.shape
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, num_tokens_per_expert


def _unpermute(
    out: torch.Tensor,
    input_shape: tuple[int, ...],
    permuted_indices: torch.Tensor,
) -> torch.Tensor:
    """Unpermute tokens after expert computation.

    Restores tokens to their original order after grouped_mm computation,
    removing the padding row.

    Args:
        out: Output from expert computation, shape (padded_num_tokens, dim).
        input_shape: Original input shape with padding row (from _permute).
        permuted_indices: Indices from _permute.

    Returns:
        Tokens in original order without padding, shape (num_tokens, dim).
    """
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    out = out_unpermuted[:-1]
    return out


def indices_padding_wrapper(func: Callable) -> Callable:
    """Wrapper for non-EP scenario to add padding for grouped_mm.

    In order to use torch._grouped_mm, we need to make sure the number of
    tokens each expert gets is a multiple of TOKEN_GROUP_ALIGN_SIZE_M. The
    generate_permute_indices kernel also helps achieve this via padding,
    without incurring synchronization between device and host.

    Args:
        func: Expert computation function with signature
            (w1, w2, w3, x, num_tokens_per_expert) -> output.

    Returns:
        Wrapped function that handles permutation and padding.
    """

    def wrapper(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        num_local_experts = w1.shape[0]
        ep_degree = num_tokens_per_expert.shape[0] // num_local_experts

        input_shape, x, permuted_indices, num_tokens_per_expert = _permute(
            x, num_tokens_per_expert, ep_degree, num_local_experts
        )

        out = func(w1, w2, w3, x, num_tokens_per_expert)

        out = _unpermute(out, input_shape, permuted_indices)

        return out

    return wrapper


__all__ = [
    "permute_tokens",
    "unpermute_tokens",
    "merge_expert_outputs",
    "set_token_group_alignment_size",
    "_permute",
    "_unpermute",
    "indices_padding_wrapper",
]

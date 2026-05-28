# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch


def infer_token_denominator(
    input_data: dict[str, Any],
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Infer the full token mask for stats logging.

    Context parallelism may slice intermediate tensors such as ``loss_mask`` or
    model outputs, while the original micro-batch metadata still describes the
    full logical sequence. Prefer that metadata for ``n_tokens`` so statistics
    stay consistent with and without context parallelism.
    """
    common_kwargs = {"dtype": torch.bool, "device": fallback.device}

    attention_mask = input_data.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        return torch.ones_like(attention_mask, **common_kwargs)

    cu_seqlens = input_data.get("cu_seqlens")
    if isinstance(cu_seqlens, torch.Tensor) and cu_seqlens.numel() > 0:
        return torch.ones(int(cu_seqlens[-1].item()), **common_kwargs)

    input_ids = input_data.get("input_ids")
    # Tree-packed batches keep input_ids padded to tree size while token-level
    # stats stay at packed-token length. Only reuse input_ids when it already
    # matches the stat tensor shape.
    if isinstance(input_ids, torch.Tensor) and input_ids.shape == fallback.shape:
        return torch.ones_like(input_ids, **common_kwargs)

    return torch.ones_like(fallback, **common_kwargs)

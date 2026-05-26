# SPDX-License-Identifier: Apache-2.0

"""R3 data-splitting helpers for the PPO actor.

Resolves and splits ``routed_experts`` tensors across mini-batches for
side-channel delivery to the training engine.
"""

from __future__ import annotations

from typing import Any

import torch

from areal.utils import logging

# NOTE: use areal.utils.logging.getLogger with a stable registered
# name so the logger survives the dictConfig(disable_existing_loggers=True) re-init path.
logger = logging.getLogger("R3/actor")


def _resolve_to_tensor(obj: Any) -> torch.Tensor | None:
    """Resolve *obj* to a ``torch.Tensor``, handling RTensor and numpy.

    Returns ``None`` if *obj* is ``None`` or cannot be converted.
    """
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj
    try:
        from areal.infra.rpc.rtensor import RTensor

        if isinstance(obj, RTensor):
            return obj.to_local()
    except ImportError:
        pass
    try:
        return torch.as_tensor(obj)
    except Exception:
        logger.warning(
            "[R3] Failed to resolve %s to torch.Tensor.",
            type(obj).__name__,
            exc_info=True,
        )
        return None


def split_routed_experts_for_minibatches(
    routed_experts: torch.Tensor,
    mb_list,
) -> list[torch.Tensor | None]:
    """Split ``routed_experts`` tensor for actor-level mini-batches.

    This handles the Level-1 split (actor._ppo_update splits into
    ppo_n_minibatches).  The tensor is reordered by ``forward_indices``
    and then sliced according to each mini-batch's sample count.

    Args:
        routed_experts: ``(bs, seq_len, num_moe_layers, topk)`` full batch tensor.
        mb_list: ``MicroBatchList`` from ``split_padded_tensor_dict_into_mb_list``.

    Returns:
        List of tensors, one per mini-batch, each of shape
        ``(mini_bs, seq_len, num_moe_layers, topk)``.
    """
    if routed_experts is None:
        return [None] * len(mb_list)

    forward_indices = mb_list.forward_indices
    n_mbs = len(mb_list)

    if forward_indices is None:
        reordered = routed_experts
    else:
        reordered = routed_experts[forward_indices]

    result = []
    offset = 0
    for i, mb_dict in enumerate(mb_list.mbs):
        n_samples = _infer_mb_sample_count_from_dict(
            mb_dict, routed_experts.shape[0], n_mbs
        )
        result.append(reordered[offset : offset + n_samples])
        offset += n_samples

    return result


def _infer_mb_sample_count_from_dict(
    mb_dict: dict,
    total_bs: int,
    n_mbs: int,
) -> int:
    """Infer sample count from a mini-batch dict."""
    if isinstance(mb_dict, dict):
        attn = mb_dict.get("attention_mask")
        if attn is not None and hasattr(attn, "shape"):
            return attn.shape[0]
        ids = mb_dict.get("input_ids")
        if ids is not None and hasattr(ids, "shape"):
            return ids.shape[0]
    return total_bs // n_mbs

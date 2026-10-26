# SPDX-License-Identifier: Apache-2.0
"""Avoid serializing a whole image batch for each split CPU tensor view."""

from dataclasses import replace
from typing import Any

import torch


def _compact_cpu_storage(value: Any, memo: dict[int, Any]) -> Any:
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, torch.Tensor):
        result = value
        if (
            value.device.type == "cpu"
            and value.layout == torch.strided
            and value.untyped_storage().nbytes() > value.numel() * value.element_size()
        ):
            result = value.clone()
    elif isinstance(value, dict):
        result = {key: _compact_cpu_storage(item, memo) for key, item in value.items()}
    elif isinstance(value, list):
        result = [_compact_cpu_storage(item, memo) for item in value]
    elif isinstance(value, tuple):
        result = tuple(_compact_cpu_storage(item, memo) for item in value)
    else:
        result = value
    memo[identity] = result
    return result


def compact_mm_item_storage(items: list[Any]) -> list[Any]:
    """Copy per-request metadata and compact CPU views without mutating cached items.

    Native preprocessing splits a batch into per-image views. Pickle serializes
    each view's entire backing storage, so N images can transmit N copies of the
    batch. Compact only those CPU views; retain tensor values, hashes, pad IDs,
    already compact tensors, and native CUDA/IPC objects.
    """
    memo: dict[int, Any] = {}
    return [
        replace(
            item,
            feature=_compact_cpu_storage(item.feature, memo),
            precomputed_embeddings=_compact_cpu_storage(
                item.precomputed_embeddings, memo
            ),
            model_specific_data=_compact_cpu_storage(item.model_specific_data, memo),
        )
        for item in items
    ]

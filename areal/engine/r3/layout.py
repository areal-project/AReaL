# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NativeReplaySlabs:
    slabs: list[torch.Tensor]
    skip_replay: bool
    reason: str | None = None


def slice_local_moe_layers(
    packed_routing: torch.Tensor,
    local_moe_indices: list[int] | tuple[int, ...],
) -> list[torch.Tensor]:
    """Return per-layer ``[tokens, topk]`` slabs from ``[tokens, layers, topk]``."""

    return [packed_routing[:, layer_idx, :].long() for layer_idx in local_moe_indices]

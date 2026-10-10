# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from areal.engine.megatron_utils.packed_context_parallel import (
    split_packed_seqs_for_context_parallel,
)
from areal.engine.r3.asserts import r3_error


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


def _pack_routing_to_padded_order(
    routed_experts: torch.Tensor,
    mb_input: Any,
) -> NativeReplaySlabs:
    """Pack ``[B, S, L, K]`` routing into Megatron's padded packed token order."""

    if routed_experts.ndim != 4:
        raise r3_error(
            "routed_experts must have shape [batch, seq_len, num_moe_layers, topk]",
            shape=tuple(routed_experts.shape),
        )
    orig_cu_seqlens = mb_input.orig_mb.get("cu_seqlens")
    padded_cu_seqlens = mb_input.padded_mb.get("cu_seqlens")
    if orig_cu_seqlens is None or padded_cu_seqlens is None:
        raise r3_error("R3 layout requires packed Megatron cu_seqlens")
    old_cu_seqlens = (
        mb_input.old_cu_seqlens
        if mb_input.old_cu_seqlens is not None
        else orig_cu_seqlens
    )

    old_cu_list = old_cu_seqlens.tolist()
    padded_cu_list = padded_cu_seqlens.tolist()

    batch_size = len(old_cu_list) - 1
    if routed_experts.shape[0] != batch_size:
        raise r3_error(
            "R3 micro-batch routed_experts batch size mismatch",
            routed_batch_size=routed_experts.shape[0],
            micro_batch_size=batch_size,
        )
    if len(padded_cu_list) < batch_size + 1:
        raise r3_error(
            "padded cu_seqlens is shorter than R3 micro-batch",
            padded_cu_len=len(padded_cu_list),
            micro_batch_size=batch_size,
        )

    total_padded_tokens = padded_cu_list[-1]
    packed = torch.empty(
        (total_padded_tokens, *routed_experts.shape[2:]),
        dtype=routed_experts.dtype,
        device=routed_experts.device,
    )

    fill_row: torch.Tensor | None = None
    for sample_idx in range(batch_size):
        real_start = old_cu_list[sample_idx]
        real_end = old_cu_list[sample_idx + 1]
        real_len = real_end - real_start
        padded_start = padded_cu_list[sample_idx]
        padded_end = padded_cu_list[sample_idx + 1]
        padded_len = padded_end - padded_start

        if real_len < 0 or padded_len < real_len:
            raise r3_error(
                "Invalid R3 cu_seqlens for routing layout",
                sample_idx=sample_idx,
                real_len=real_len,
                padded_len=padded_len,
            )
        if real_len > routed_experts.shape[1]:
            raise r3_error(
                "R3 routed_experts sequence dimension is shorter than micro-batch sequence",
                sample_idx=sample_idx,
                real_len=real_len,
                routed_seq_len=routed_experts.shape[1],
            )
        if real_len == 0:
            return NativeReplaySlabs(
                slabs=[],
                skip_replay=True,
                reason="empty_sequence",
            )

        rows = routed_experts[sample_idx, :real_len]
        fill_row = rows[-1]
        packed[padded_start : padded_start + real_len] = rows
        if padded_len > real_len:
            packed[padded_start + real_len : padded_end] = fill_row

    batch_padding_start = padded_cu_list[batch_size]
    if total_padded_tokens > batch_padding_start:
        if fill_row is None:
            return NativeReplaySlabs(
                slabs=[],
                skip_replay=True,
                reason="batch_padding_without_valid_routing",
            )
        packed[batch_padding_start:total_padded_tokens] = fill_row

    return NativeReplaySlabs(slabs=[packed], skip_replay=False)


def _scatter_to_sequence_parallel_region(tensor: torch.Tensor) -> torch.Tensor:
    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.mappings import (
        scatter_to_sequence_parallel_region,
    )

    if mpu.get_tensor_model_parallel_world_size() <= 1:
        return tensor
    return scatter_to_sequence_parallel_region(tensor)


def prepare_native_replay_slabs(
    routed_experts: torch.Tensor,
    routing_valid: torch.Tensor,
    mb_input: Any,
    local_moe_indices: list[int] | tuple[int, ...],
    *,
    sequence_parallel: bool = True,
) -> NativeReplaySlabs:
    """Build per-local-router native replay slabs for a Megatron micro-batch.

    ``routed_experts`` is the workflow tensor slice for the current micro-batch
    with shape ``[B, S, global_moe_layers, topk]``. The output list is ordered to
    match the local ``TopKRouter`` traversal order on the current PP/VPP stage.
    """

    if not local_moe_indices:
        return NativeReplaySlabs(slabs=[], skip_replay=False)
    num_moe_layers = routed_experts.shape[2] if routed_experts.ndim >= 3 else None
    invalid_indices = [
        idx
        for idx in local_moe_indices
        if num_moe_layers is None or idx < 0 or idx >= num_moe_layers
    ]
    if invalid_indices:
        raise r3_error(
            "R3 local MoE layer index is out of routed_experts range",
            local_moe_indices=list(local_moe_indices),
            num_moe_layers=num_moe_layers,
        )
    if routing_valid.ndim != 1:
        raise r3_error(
            "r3_routing_valid must have shape [batch]",
            shape=tuple(routing_valid.shape),
        )
    if routing_valid.shape[0] != routed_experts.shape[0]:
        raise r3_error(
            "r3_routing_valid batch size mismatch",
            valid_batch_size=routing_valid.shape[0],
            routed_batch_size=routed_experts.shape[0],
        )
    if not bool(torch.all(routing_valid).item()):
        return NativeReplaySlabs(
            slabs=[],
            skip_replay=True,
            reason="invalid_sample",
        )

    packed_result = _pack_routing_to_padded_order(routed_experts, mb_input)
    if packed_result.skip_replay:
        return packed_result
    packed_routing = packed_result.slabs[0]

    padded_cu_seqlens = mb_input.padded_mb["cu_seqlens"]
    packed_routing = split_packed_seqs_for_context_parallel(
        packed_routing,
        padded_cu_seqlens,
    )
    if sequence_parallel:
        packed_routing = _scatter_to_sequence_parallel_region(packed_routing)
    slabs = slice_local_moe_layers(packed_routing, local_moe_indices)
    topk = routed_experts.shape[3]
    for slab in slabs:
        if slab.ndim != 2 or slab.shape[1] != topk:
            raise r3_error(
                "R3 native replay slab has invalid shape",
                slab_shape=tuple(slab.shape),
                topk=topk,
            )
        if slab.dtype != torch.long:
            raise r3_error(
                "R3 native replay slab must be torch.long",
                slab_dtype=str(slab.dtype),
            )
    return NativeReplaySlabs(slabs=slabs, skip_replay=False)

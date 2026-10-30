# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from areal.engine.megatron_utils.packed_context_parallel import (
    split_packed_seqs_for_context_parallel,
)
from areal.engine.r3.asserts import r3_error
from areal.engine.r3.packed import PackedRouting


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

    return [
        packed_routing[:, layer_idx, :].to(dtype=torch.long).contiguous()
        for layer_idx in local_moe_indices
    ]


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
    routed_experts: torch.Tensor | PackedRouting,
    routing_valid: torch.Tensor,
    mb_input: Any,
    local_moe_indices: list[int] | tuple[int, ...],
    *,
    sequence_parallel: bool = True,
) -> NativeReplaySlabs:
    """Build per-local-router native replay slabs for a Megatron micro-batch.

    ``routed_experts`` is a packed CPU view with row/layer metadata, or a legacy
    ``[B, S, global_moe_layers, topk]`` tensor. The output list follows the local
    ``TopKRouter`` traversal order on the current PP/VPP stage.
    """

    if not local_moe_indices:
        return NativeReplaySlabs(slabs=[], skip_replay=False)
    if isinstance(routed_experts, PackedRouting):
        return _prepare_packed_replay_slabs(
            routed_experts,
            routing_valid,
            mb_input,
            local_moe_indices,
            sequence_parallel=sequence_parallel,
        )
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

    # Select PP/VPP layers before allocating the packed tensor. The engine
    # supplies CPU routing and metadata, so CP/SP slicing also stays on CPU.
    local_routing = routed_experts[:, :, list(local_moe_indices), :]
    packed_result = _pack_routing_to_padded_order(local_routing, mb_input)
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
    slabs = slice_local_moe_layers(packed_routing, list(range(len(local_moe_indices))))
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


def _prepare_packed_replay_slabs(
    routes: PackedRouting,
    valid: torch.Tensor,
    mb_input: Any,
    layers: list[int] | tuple[int, ...],
    *,
    sequence_parallel: bool,
) -> NativeReplaySlabs:
    """Gather only this rank's CP/SP tokens, without a full padded routing copy."""
    from megatron.core import parallel_state as mpu

    if valid.shape != torch.Size([len(routes)]):
        raise r3_error("R3 routing validity does not match packed rows")
    if not bool(valid.all()):
        return NativeReplaySlabs([], True, "invalid_sample")
    old_cu = mb_input.old_cu_seqlens
    if old_cu is None:
        old_cu = mb_input.orig_mb["cu_seqlens"]
    old = old_cu.tolist()
    padded = mb_input.padded_mb["cu_seqlens"].tolist()
    if len(old) != len(routes) + 1 or len(padded) < len(old):
        raise r3_error("Packed routing rows do not match microbatch cu_seqlens")
    if not len(routes):
        return NativeReplaySlabs([], True, "empty_sequence")
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank() if cp_size > 1 else 0
    token_indices = []
    for segment, (start, end) in enumerate(zip(padded[:-1], padded[1:])):
        row_index = min(segment, len(routes) - 1)
        source_row = int(routes.row_indices[row_index])
        real_len = old[row_index + 1] - old[row_index]
        if source_row < 0 or real_len == 0:
            return NativeReplaySlabs([], True, "empty_sequence")
        source_start, source_end = routes.cu_seqlens[
            source_row : source_row + 2
        ].tolist()
        if source_end - source_start != real_len:
            raise r3_error(
                "Packed routing token length does not match microbatch sequence"
            )
        size = end - start
        if size < 0 or (segment < len(routes) and size < real_len):
            raise r3_error("Invalid R3 padded sequence length")
        if cp_size > 1:
            if size % (2 * cp_size):
                raise r3_error("R3 padded length is not divisible by CP chunks")
            chunk = size // (2 * cp_size)
            positions = torch.cat(
                [
                    torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk),
                    torch.arange(size - (cp_rank + 1) * chunk, size - cp_rank * chunk),
                ]
            )
        else:
            positions = torch.arange(size)
        if segment >= len(routes):
            positions.fill_(real_len - 1)
        token_indices.append(positions.clamp_max(real_len - 1) + source_start)
    indices = torch.cat(token_indices)
    if sequence_parallel:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        if indices.numel() % tp_size:
            raise r3_error("R3 local token count is not divisible by SP size")
        if tp_size > 1:
            indices = indices.chunk(tp_size)[mpu.get_tensor_model_parallel_rank()]
    try:
        local_layers = [routes.layer_indices.index(layer) for layer in layers]
    except ValueError as exc:
        raise r3_error("R3 local MoE layer is absent from packed routing") from exc
    slabs = [
        routes.values[:, layer].index_select(0, indices).long().contiguous()
        for layer in local_layers
    ]
    return NativeReplaySlabs(slabs, False)

# SPDX-License-Identifier: Apache-2.0

"""Stream packed routing to pipeline stages without broadcasting global layers."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from areal.engine.r3.packed import (
    ROUTING_LAYERS_KEY,
    ROUTING_OFFSETS_KEY,
    PackedRouting,
)
from areal.utils.data import broadcast_tensor_container

_TRANSFER_BYTES = 4 * 1024 * 1024


def _extract_routes(data: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Copy containers, retaining normal tensor aliases and route descriptors."""
    descriptors: list[dict[str, Any]] = []

    def visit(value):
        if isinstance(value, dict):
            output = {key: visit(child) for key, child in value.items()}
            if ROUTING_OFFSETS_KEY in value:
                output["routed_experts"] = None
                descriptors.append({"source": value, "target": output})
            return output
        if isinstance(value, list):
            return [visit(child) for child in value]
        if isinstance(value, tuple):
            children = [visit(child) for child in value]
            return (
                type(value)(*children) if hasattr(value, "_fields") else tuple(children)
            )
        return value

    return visit(data), descriptors


def _route_targets(data: Any) -> list[dict[str, Any]]:
    targets = []
    if isinstance(data, dict):
        for child in data.values():
            targets.extend(_route_targets(child))
        if ROUTING_OFFSETS_KEY in data:
            targets.append(data)
    elif isinstance(data, (tuple, list)):
        for child in data:
            targets.extend(_route_targets(child))
    return targets


def broadcast_packed_routing(
    data: Any,
    *,
    src_rank: int,
    group: dist.ProcessGroup,
    stages: list[dict[str, Any]],
    preserve_tensor_aliases: bool = False,
    num_experts: int | None = None,
) -> Any:
    """Broadcast ordinary data and stream only each physical PP's routing.

    Every model-parallel rank calls this function in the same order. ``stages``
    contains the global ranks, Gloo subgroup and VPP layer union per physical
    stage. Sending one bounded row chunk at a time avoids a full PP temporary
    on the source. Receivers allocate only their own PP layer storage.
    """
    rank = dist.get_rank()
    descriptors = []
    header: list[Any] = [None]
    if rank == src_rank:
        try:
            ordinary, descriptors = _extract_routes(data)
            routes = []
            for descriptor in descriptors:
                source = descriptor["source"]
                layer_ids = source.get(ROUTING_LAYERS_KEY)
                route = PackedRouting.from_tensors(
                    source["routed_experts"],
                    source[ROUTING_OFFSETS_KEY],
                    layer_ids.tolist() if layer_ids is not None else None,
                )
                if route.values.dtype not in (torch.int16, torch.int32, torch.int64):
                    raise ValueError("R3 routing must use signed integer expert IDs")
                valid = source.get("r3_routing_valid")
                if not torch.is_tensor(valid) or valid.shape != torch.Size(
                    [len(route)]
                ):
                    raise ValueError("R3 validity does not match routing rows")
                mask = source.get("attention_mask")
                if torch.is_tensor(mask):
                    if mask.shape[0] != len(route) or not torch.equal(
                        mask.sum(-1).long(), route.cu_seqlens.diff()
                    ):
                        raise ValueError(
                            "R3 token offsets do not match trajectory lengths"
                        )
                if route.values.numel():
                    lo, hi = torch.aminmax(route.values)
                    if int(lo) < 0 or (
                        num_experts is not None and int(hi) >= num_experts
                    ):
                        raise ValueError("R3 expert ID outside model expert range")
                for stage in stages:
                    if any(
                        layer not in route.layer_indices for layer in stage["layers"]
                    ):
                        raise ValueError(
                            "R3 source is missing a pipeline stage's layers"
                        )
                descriptor["route"] = route
                routes.append(
                    (route.values.shape[0], route.values.shape[2], route.values.dtype)
                )
            header[0] = {"routes": routes, "error": None}
        except Exception as exc:
            header[0] = {"routes": [], "error": str(exc)}
    else:
        ordinary = None
    dist.broadcast_object_list(header, src=src_rank, group=group)
    if header[0]["error"] is not None:
        raise ValueError(f"Invalid R3 broadcast payload: {header[0]['error']}")
    ordinary = broadcast_tensor_container(
        ordinary,
        src_rank=src_rank,
        group=group,
        preserve_tensor_aliases=preserve_tensor_aliases,
    )
    targets = _route_targets(ordinary)
    for route_index, (tokens, topk, dtype) in enumerate(header[0]["routes"]):
        route = descriptors[route_index]["route"] if rank == src_rank else None
        for stage in stages:
            ranks, layers = stage["ranks"], stage["layers"]
            leader = ranks[0]
            member = rank in ranks
            if member:
                output = torch.empty(
                    (tokens, len(layers), topk), dtype=dtype, device="cpu"
                )
            elements_per_row = len(layers) * topk
            rows_per_chunk = max(
                1,
                _TRANSFER_BYTES
                // max(
                    1, elements_per_row * torch.empty((), dtype=dtype).element_size()
                ),
            )
            layer_index = (
                torch.tensor(
                    [route.layer_indices.index(layer) for layer in layers],
                    dtype=torch.long,
                )
                if route is not None
                else None
            )
            if elements_per_row:
                for start in range(0, tokens, rows_per_chunk):
                    end = min(start + rows_per_chunk, tokens)
                    if rank == src_rank:
                        chunk = route.values[start:end].index_select(1, layer_index)
                        if leader == src_rank:
                            output[start:end].copy_(chunk)
                        else:
                            dist.send(chunk.view(torch.uint8), dst=leader, group=group)
                        del chunk
                    elif rank == leader:
                        dist.recv(
                            output[start:end].view(torch.uint8),
                            src=src_rank,
                            group=group,
                        )
                    if member:
                        dist.broadcast(
                            output[start:end].view(torch.uint8),
                            src=leader,
                            group=stage["group"],
                        )
            if member:
                targets[route_index]["routed_experts"] = output
                targets[route_index][ROUTING_LAYERS_KEY] = torch.tensor(
                    layers, dtype=torch.long
                )
    return ordinary

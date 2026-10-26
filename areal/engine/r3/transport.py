# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

from areal.engine.r3.packed import (
    ROUTING_LAYERS_KEY,
    ROUTING_OFFSETS_KEY,
    PackedRouting,
)

R3_ROUTED_EXPERTS_KEY = "routed_experts"
R3_ROUTING_VALID_KEY = "r3_routing_valid"


def evict_routing_fetches(obj: Any) -> None:
    """Release local routing cache ownership after RPC localization.

    Source shards and other consumers remain unchanged. Localized arguments
    retain the routing until pipeline distribution completes.
    """
    from areal.infra.rpc.rtensor import RTensor, clear_fetch_buffer

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == R3_ROUTED_EXPERTS_KEY and isinstance(value, RTensor):
                clear_fetch_buffer([value.shard.shard_id])
                value.data = value.data.to(device="meta")
            else:
                evict_routing_fetches(value)
    elif isinstance(obj, (tuple, list)):
        for value in obj:
            evict_routing_fetches(value)


def without_r3_tensors(data: dict[str, Any]) -> dict[str, Any]:
    """Share trajectory tensors without actor-specific rollout routing.

    Reference, critic and teacher models must choose their own experts. A
    shallow copy isolates dictionary changes; tensor storage remains shared.
    """
    return {
        key: value
        for key, value in data.items()
        if key
        not in (
            R3_ROUTED_EXPERTS_KEY,
            R3_ROUTING_VALID_KEY,
            ROUTING_OFFSETS_KEY,
            ROUTING_LAYERS_KEY,
        )
    }


def pop_r3_tensors(data: dict[str, Any]) -> tuple[Any | None, Any | None]:
    routes = data.pop(R3_ROUTED_EXPERTS_KEY, None)
    valid = data.pop(R3_ROUTING_VALID_KEY, None)
    offsets = data.pop(ROUTING_OFFSETS_KEY, None)
    layers = data.pop(ROUTING_LAYERS_KEY, None)
    if offsets is not None:
        routes = PackedRouting.from_tensors(
            localize_r3_tensor(routes),
            localize_r3_tensor(offsets),
            localize_r3_tensor(layers).tolist() if layers is not None else None,
        )
    return routes, valid


def restore_r3_tensors(
    data: dict[str, Any], routes: Any | None, valid: Any | None
) -> None:
    if isinstance(routes, PackedRouting):
        data[R3_ROUTED_EXPERTS_KEY] = routes.values
        data[ROUTING_OFFSETS_KEY] = routes.cu_seqlens
        data[ROUTING_LAYERS_KEY] = torch.tensor(routes.layer_indices, dtype=torch.long)
    elif routes is not None:
        data[R3_ROUTED_EXPERTS_KEY] = routes
    if valid is not None:
        data[R3_ROUTING_VALID_KEY] = valid


def localize_r3_tensor(value: Any) -> Any:
    if hasattr(value, "localize"):
        return value.localize()
    return value


def set_engine_r3_side_channel(
    engine: Any,
    routed_experts: torch.Tensor | Any | None,
    routing_valid: torch.Tensor | Any | None,
) -> None:
    engine._r3_pending_routed_experts = localize_r3_tensor(routed_experts)
    engine._r3_pending_valid = localize_r3_tensor(routing_valid)


def clear_engine_r3_side_channel(engine: Any) -> None:
    engine._r3_pending_routed_experts = None
    engine._r3_pending_valid = None

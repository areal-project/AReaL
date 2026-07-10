# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from areal.engine.r3.asserts import r3_error


@dataclass(frozen=True)
class NativeRouterReplayRef:
    vp_stage: int
    name: str
    router: torch.nn.Module
    router_replay: Any
    layer_number: int | None = None


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    while hasattr(module, "module"):
        module = module.module
    return module


def _resolve_topk_router_type() -> type:
    try:
        from megatron.core.transformer.moe.router import TopKRouter
    except ImportError as exc:
        raise r3_error("Unable to import Megatron TopKRouter") from exc
    return TopKRouter


def _infer_layer_number(name: str, router: torch.nn.Module) -> int | None:
    layer_number = getattr(router, "layer_number", None)
    if layer_number is not None:
        return int(layer_number)

    match = re.search(r"(?:decoder|transformer|language_model).*layers\.(\d+)", name)
    if match is None:
        match = re.search(r"layers\.(\d+)", name)
    return int(match.group(1)) if match is not None else None


def discover_native_router_replay(
    model: torch.nn.Module | list[torch.nn.Module],
    *,
    require_router_replay: bool = True,
) -> dict[int, list[NativeRouterReplayRef]]:
    """Collect instance-local Megatron native RouterReplay objects.

    The returned refs are grouped by model chunk / virtual pipeline stage and
    never use Megatron's global RouterReplay registry.
    """

    topk_router_type = _resolve_topk_router_type()
    modules = model if isinstance(model, list) else [model]
    grouped: dict[int, list[NativeRouterReplayRef]] = {}

    for chunk_idx, module in enumerate(modules):
        unwrapped = _unwrap_module(module)
        vp_stage = int(
            getattr(module, "vp_stage", getattr(unwrapped, "vp_stage", chunk_idx)) or 0
        )
        refs: list[NativeRouterReplayRef] = []
        for name, child in unwrapped.named_modules():
            if not isinstance(child, topk_router_type):
                continue
            router_replay = getattr(child, "router_replay", None)
            if require_router_replay and router_replay is None:
                raise r3_error(
                    "Megatron TopKRouter is missing native router_replay; "
                    "moe_enable_routing_replay did not reach TransformerConfig",
                    vp_stage=vp_stage,
                    module_name=name,
                    router_type=type(child).__name__,
                )
            refs.append(
                NativeRouterReplayRef(
                    vp_stage=vp_stage,
                    name=name,
                    router=child,
                    router_replay=router_replay,
                    layer_number=_infer_layer_number(name, child),
                )
            )
        refs.sort(
            key=lambda ref: (
                ref.layer_number if ref.layer_number is not None else 10**9,
                ref.name,
            )
        )
        grouped[vp_stage] = refs

    return grouped

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from areal.engine.r3.asserts import r3_error
from areal.engine.r3.discovery import NativeRouterReplayRef


def get_router_replay_action(action_name: str) -> Any:
    try:
        from megatron.core.transformer.moe.router_replay import RouterReplayAction
    except ImportError:
        try:
            from megatron.core.transformer.moe.router import RouterReplayAction
        except ImportError as exc:
            raise r3_error("Unable to import Megatron RouterReplayAction") from exc
    return getattr(RouterReplayAction, action_name)


def set_router_replay_action(
    refs: Iterable[NativeRouterReplayRef],
    action_name: str,
) -> None:
    action = get_router_replay_action(action_name)
    for ref in refs:
        ref.router_replay.set_router_replay_action(action)


def clear_router_replay_action(refs: Iterable[NativeRouterReplayRef]) -> None:
    for ref in refs:
        ref.router_replay.clear_router_replay_action()


def set_target_indices(
    refs: Iterable[NativeRouterReplayRef],
    slabs: Iterable[torch.Tensor],
) -> None:
    for ref, slab in zip(refs, slabs, strict=True):
        if slab.dtype != torch.long:
            slab = slab.long()
        ref.router_replay.set_target_indices(slab)


def enqueue_recorded_indices(refs: Iterable[NativeRouterReplayRef]) -> None:
    for ref in refs:
        recorded = ref.router_replay.get_recorded_indices()
        if recorded is None:
            raise r3_error(
                "Megatron RouterReplay RECORD did not produce recorded indices",
                module_name=ref.name,
                vp_stage=ref.vp_stage,
            )
        if recorded.dtype != torch.long:
            recorded = recorded.long()
        ref.router_replay.set_target_indices(recorded)


def clear_router_replay_state(refs: Iterable[NativeRouterReplayRef]) -> None:
    for ref in refs:
        ref.router_replay.clear_indices()
        ref.router_replay.clear_router_replay_action()

# SPDX-License-Identifier: Apache-2.0

"""R3 native Megatron router replay helpers."""

from areal.engine.r3.config import R3MoEConfig, resolve_r3_moe_config
from areal.engine.r3.discovery import (
    NativeRouterReplayRef,
    discover_native_router_replay,
)
from areal.engine.r3.preprocess import (
    decode_sglang_routed_experts,
    preprocess_routed_experts_batch,
)

__all__ = [
    "NativeRouterReplayRef",
    "R3MoEConfig",
    "decode_sglang_routed_experts",
    "discover_native_router_replay",
    "preprocess_routed_experts_batch",
    "resolve_r3_moe_config",
]

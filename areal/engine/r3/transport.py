# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import torch

R3_ROUTED_EXPERTS_KEY = "routed_experts"
R3_ROUTING_VALID_KEY = "r3_routing_valid"


def pop_r3_tensors(data: dict[str, Any]) -> tuple[Any | None, Any | None]:
    return data.pop(R3_ROUTED_EXPERTS_KEY, None), data.pop(R3_ROUTING_VALID_KEY, None)


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

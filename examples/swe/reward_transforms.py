"""Reusable reward transforms for SWE Arena workflows."""

import math
from typing import Any


def identity_reward(
    reward: float,
    _data: dict[str, Any],
    reward_threshold: float | None = None,
) -> float:
    """Preserve the Arena reward without applying threshold-based scaling."""
    del reward_threshold
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError(f"Arena reward must be finite and within [0, 1], got {reward}")
    return float(reward)


def astra_partial_reward(
    reward: float,
    _data: dict[str, Any],
    reward_threshold: float = 0.98,
) -> float:
    """Keep Astra scores below the configured threshold at one tenth strength."""
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError(f"Astra reward must be finite and within [0, 1], got {reward}")
    if reward < reward_threshold:
        return reward * 0.1
    return 1.0

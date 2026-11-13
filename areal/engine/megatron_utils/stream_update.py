# SPDX-License-Identifier: Apache-2.0

"""State and normalization contract for one streamed, fixed-batch PPO update.

Each replica accumulates token-loss *numerator* gradients. No optimizer or DP
collective runs until commit. Legacy MCore DDP averages gradients; native
per-token DDP sums them. Commit compensates exactly once for that difference.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamUpdateState:
    update_id: str
    chunks: dict[int, dict[str, Any]] = field(default_factory=dict)
    local_mass: int = 0
    failed: bool = False

    def check_chunk(self, chunk_id: int) -> dict[str, Any] | None:
        if self.failed:
            raise RuntimeError("Stream update failed; abort it before retrying")
        if chunk_id < 0:
            raise ValueError("chunk_id must be nonnegative")
        return self.chunks.get(chunk_id)

    def complete_chunk(self, chunk_id: int, mass: int) -> dict[str, Any]:
        if chunk_id in self.chunks:
            raise RuntimeError("A streamed chunk cannot be accumulated twice")
        if mass < 0:
            raise ValueError("Token mass cannot be negative")
        result = dict(update_id=self.update_id, chunk_id=chunk_id, token_mass=mass)
        self.local_mass += mass
        self.chunks[chunk_id] = result
        return dict(result)


def stream_gradient_scale(
    global_mass: int, dp_size: int, *, per_token_loss: bool
) -> float:
    """Convert raw numerator grads to the ordinary token-mean PPO objective."""
    if global_mass <= 0:
        raise ValueError("Stream update requires a positive global token mass")
    if dp_size < 1:
        raise ValueError("Data parallel size must be positive")
    return (1.0 if per_token_loss else float(dp_size)) / global_mass

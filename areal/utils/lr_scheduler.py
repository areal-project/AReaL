# SPDX-License-Identifier: Apache-2.0

"""Learning-rate scheduler helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from areal.api.cli_args import OptimizerConfig


def get_num_warmup_steps(
    optimizer_config: OptimizerConfig,
    total_train_steps: int,
) -> int:
    """Resolve warmup steps, preferring fixed steps over proportions."""
    if optimizer_config.warmup_steps is not None:
        if optimizer_config.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {optimizer_config.warmup_steps}"
            )
        num_warmup_steps = optimizer_config.warmup_steps
    else:
        if optimizer_config.warmup_steps_proportion < 0:
            raise ValueError(
                "warmup_steps_proportion must be non-negative, "
                f"got {optimizer_config.warmup_steps_proportion}"
            )
        num_warmup_steps = int(
            optimizer_config.warmup_steps_proportion * total_train_steps
        )
    return num_warmup_steps

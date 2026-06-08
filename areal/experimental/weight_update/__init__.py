# SPDX-License-Identifier: Apache-2.0
"""Weight update protocol adapters for training and inference."""

from areal.experimental.weight_update.controller import (
    WeightUpdateController,
    WeightUpdateControllerConfig,
)

__all__ = [
    "WeightUpdateController",
    "WeightUpdateControllerConfig",
]

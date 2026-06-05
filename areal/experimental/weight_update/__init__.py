# SPDX-License-Identifier: Apache-2.0
"""Weight update protocol adapters for training and inference."""

from areal.experimental.weight_update.constants import (
    BACKEND_AWEX,
    BACKEND_DISK,
    BACKEND_RDT,
    WEIGHT_UPDATE_BACKEND_ENV,
    get_weight_update_backend,
)
from areal.experimental.weight_update.controller import (
    WeightUpdateController,
    WeightUpdateControllerConfig,
)

__all__ = [
    "WeightUpdateController",
    "WeightUpdateControllerConfig",
    "WEIGHT_UPDATE_BACKEND_ENV",
    "BACKEND_AWEX",
    "BACKEND_RDT",
    "BACKEND_DISK",
    "get_weight_update_backend",
]

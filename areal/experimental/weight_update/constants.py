# SPDX-License-Identifier: Apache-2.0
"""Constants and helpers for weight update backends."""

import os

WEIGHT_UPDATE_BACKEND_ENV = "AREAL_WEIGHT_UPDATE_BACKEND"
BACKEND_AWEX = "awex"
BACKEND_RDT = "rdt"
BACKEND_DISK = "disk"


def get_weight_update_backend() -> str:
    """Get weight update backend from env or default to awex."""
    backend = os.environ.get(WEIGHT_UPDATE_BACKEND_ENV, BACKEND_AWEX)
    if backend not in (BACKEND_AWEX, BACKEND_RDT):
        raise ValueError(f"Invalid backend: {backend}, must be awex or rdt")
    return backend

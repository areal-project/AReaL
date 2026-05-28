# SPDX-License-Identifier: Apache-2.0

"""Agent Service Guard backed by the shared guard infrastructure.

All core guard functionality (port allocation, process forking, health
checks, cleanup) is provided by ``areal.infra.rpc.guard``.  This module
creates and exposes the Flask app and shared state instance, following
the same pattern as ``areal.experimental.inference_service.guard``.
"""

from __future__ import annotations

from areal.infra.rpc.guard.app import (
    GuardState,
    create_app,
)
from areal.infra.rpc.guard.app import (
    cleanup_forked_children as _cleanup_impl,
)
from areal.utils import logging

logger = logging.getLogger("AgentGuard")

_state = GuardState()

app = create_app(_state)


def cleanup_forked_children() -> None:
    _cleanup_impl(_state)

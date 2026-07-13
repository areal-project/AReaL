# SPDX-License-Identifier: Apache-2.0

"""RPCGuard — inference service guard backed by the shared guard.

All guard functionality is now provided by ``areal.infra.rpc.guard``.
This module creates and exposes the Flask app and shared state instance
for backward compatibility with existing imports.
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

logger = logging.getLogger("RPCGuard")

_state = GuardState()

app = create_app(_state)


def cleanup_forked_children() -> None:
    _cleanup_impl(_state)

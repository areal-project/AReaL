# SPDX-License-Identifier: Apache-2.0

"""Training service guard backed by the shared RPC guard."""

from __future__ import annotations

from areal.infra.rpc.guard.app import GuardState, create_app
from areal.infra.rpc.guard.app import cleanup_forked_children as _cleanup_impl
from areal.utils import logging

logger = logging.getLogger("TrainRPCGuard")

_state = GuardState()

app = create_app(_state)


def cleanup_forked_children() -> None:
    _cleanup_impl(_state)

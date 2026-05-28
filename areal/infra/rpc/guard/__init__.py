# SPDX-License-Identifier: Apache-2.0

"""Shared Guard: reusable process management for RPC and inference services.

The Guard is the base process management layer shared between:

- ``areal.infra.rpc.rpc_server`` — RPC server (guard + data + engine)
- ``areal.experimental.inference_service.guard`` — inference service guard

Typical usage::

    from areal.infra.rpc.guard import GuardState, create_app, run_server

    state = GuardState()
    app = create_app(state)
    # Optionally register additional blueprints
    run_server(state, app, bind_host="0.0.0.0", port=0)
"""

from .app import (
    GuardState,
    cleanup_forked_children,
    configure_state_from_args,
    create_app,
    get_state,
    make_base_parser,
    run_server,
)

__all__ = [
    "GuardState",
    "cleanup_forked_children",
    "configure_state_from_args",
    "create_app",
    "get_state",
    "make_base_parser",
    "run_server",
]

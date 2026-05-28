# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint: ``python -m areal.infra.rpc.guard``

Starts a standalone Guard process with only process-management
endpoints (no engine, no data storage).
"""

from __future__ import annotations

from areal.infra.rpc.guard.app import (
    GuardState,
    configure_state_from_args,
    create_app,
    make_base_parser,
    run_server,
)


def main():
    """Main entry point for the standalone Guard service."""
    parser = make_base_parser(
        description=("AReaL Guard — HTTP gateway for coordinating forked workers")
    )
    args, _ = parser.parse_known_args()

    state = GuardState()
    bind_host = configure_state_from_args(state, args)
    app = create_app(state)

    run_server(state, app, bind_host, args.port)


if __name__ == "__main__":
    main()

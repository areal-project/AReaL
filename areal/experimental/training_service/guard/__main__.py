# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint: ``python -m areal.experimental.training_service.guard``"""

from __future__ import annotations

from areal.experimental.training_service.guard.app import _state, app
from areal.infra.rpc.guard.app import (
    configure_state_from_args,
    make_base_parser,
    run_server,
)


def main() -> None:
    parser = make_base_parser(
        description="AReaL Train RPCGuard — HTTP gateway for coordinating forked workers"
    )
    args, _ = parser.parse_known_args()
    bind_host = configure_state_from_args(_state, args)
    run_server(_state, app, bind_host, args.port)


if __name__ == "__main__":
    main()

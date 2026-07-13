# SPDX-License-Identifier: Apache-2.0

"""``python -m areal.v2.agent_service.worker``

Start a standalone Agent Worker process. The Controller forks this
via Guard to create Worker+DataProxy pairs.

    python -m areal.v2.agent_service.worker \
        --agent examples.agent_service.agent.ClaudeAgent \
        --host 127.0.0.1 --port 9000
"""

import argparse

import uvicorn

from .app import create_worker_app, create_worker_app_with_hop_auth


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Worker")
    parser.add_argument("--agent", required=True, help="Agent import path")
    parser.add_argument(
        "--worker-hop-api-key",
        default="",
        help=(
            "Dedicated DataProxy-to-Worker credential; empty preserves "
            "standalone compatibility"
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument(
        "--log-level", choices=["debug", "info", "warning", "error"], default="warning"
    )
    args = parser.parse_args()

    app = (
        create_worker_app_with_hop_auth(
            args.agent,
            args.worker_hop_api_key,
        )
        if args.worker_hop_api_key
        else create_worker_app(args.agent)
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()

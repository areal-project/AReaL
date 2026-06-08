# SPDX-License-Identifier: Apache-2.0

"""``python -m areal.experimental.agent_service.router``"""

import argparse

import uvicorn

from areal.infra.utils.http import validate_admin_api_key

from ..auth import DEFAULT_ADMIN_API_KEY
from .app import create_router_app
from .config import RouterConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Router")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--admin-api-key", default=DEFAULT_ADMIN_API_KEY)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--worker-health-timeout", type=float, default=2.0)
    parser.add_argument(
        "--log-level", choices=["debug", "info", "warning", "error"], default="warning"
    )
    args = parser.parse_args()

    validate_admin_api_key(
        args.host, args.admin_api_key, default_key=DEFAULT_ADMIN_API_KEY
    )

    config = RouterConfig(
        host=args.host,
        port=args.port,
        admin_api_key=args.admin_api_key,
        poll_interval=args.poll_interval,
        worker_health_timeout=args.worker_health_timeout,
        log_level=args.log_level,
    )
    uvicorn.run(
        create_router_app(config),
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()

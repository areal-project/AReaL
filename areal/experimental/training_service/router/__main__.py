# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib


def main():
    parser = argparse.ArgumentParser(description="AReaL Train Router")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=9081, help="Bind port")
    parser.add_argument(
        "--admin-api-key",
        default="areal-admin-key",
        help="Admin API key for privileged operations",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Seconds between model health polls",
    )
    parser.add_argument(
        "--worker-health-timeout",
        type=float,
        default=2.0,
        help="Timeout (seconds) per model health check",
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["debug", "info", "warning", "error"],
        help="Log level",
    )
    args, _ = parser.parse_known_args()

    from areal.experimental.training_service.router.app import create_app
    from areal.experimental.training_service.router.config import RouterConfig
    from areal.infra.utils.http import (
        get_default_uvicorn_kwargs,
        validate_admin_api_key,
    )
    from areal.utils.logging import suppress_http_loggers

    validate_admin_api_key(args.host, args.admin_api_key)

    config = RouterConfig(
        host=args.host,
        port=args.port,
        admin_api_key=args.admin_api_key,
        poll_interval=args.poll_interval,
        worker_health_timeout=args.worker_health_timeout,
        log_level=args.log_level,
    )

    suppress_http_loggers()
    app = create_app(config)
    uvicorn = importlib.import_module("uvicorn")
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        access_log=False,
        **get_default_uvicorn_kwargs(),
    )


if __name__ == "__main__":
    main()

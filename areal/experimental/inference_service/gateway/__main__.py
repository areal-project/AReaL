# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint for the inference gateway: ``python -m areal.experimental.inference_service.gateway``."""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="AReaL Inference Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument(
        "--admin-api-key",
        default="areal-admin-key",
        help="Admin API key for privileged operations",
    )
    parser.add_argument(
        "--router-addr",
        default="http://localhost:8081",
        help="Router service address",
    )
    parser.add_argument(
        "--router-timeout",
        type=float,
        default=2.0,
        help="Timeout (seconds) for router /route calls",
    )
    parser.add_argument(
        "--forward-timeout",
        type=float,
        default=120.0,
        help="Timeout (seconds) for forwarding requests to data proxies",
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["debug", "info", "warning", "error"],
        help="Log level",
    )
    args, _ = parser.parse_known_args()

    from areal.experimental.inference_service.gateway.app import create_app
    from areal.experimental.inference_service.gateway.config import GatewayConfig
    from areal.infra.utils.http import (
        get_default_uvicorn_kwargs,
        validate_admin_api_key,
    )
    from areal.utils.logging import suppress_http_loggers

    validate_admin_api_key(args.host, args.admin_api_key)

    config = GatewayConfig(
        host=args.host,
        port=args.port,
        admin_api_key=args.admin_api_key,
        router_addr=args.router_addr,
        router_timeout=args.router_timeout,
        forward_timeout=args.forward_timeout,
        log_level=args.log_level,
    )

    import uvicorn

    suppress_http_loggers()
    app = create_app(config)
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

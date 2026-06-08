# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint: ``python -m areal.experimental.weight_update.gateway``."""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="AReaL Weight Update Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=7080, help="Bind port")
    parser.add_argument(
        "--admin-api-key",
        default="areal-admin-key",
        help="Admin API key for privileged operations",
    )
    parser.add_argument(
        "--init-timeout",
        type=float,
        default=300.0,
        help="Timeout (seconds) for NCCL group initialization",
    )
    parser.add_argument(
        "--update-timeout",
        type=float,
        default=120.0,
        help="Timeout (seconds) for per-step weight updates",
    )
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["debug", "info", "warning", "error"],
        help="Log level",
    )
    args, _ = parser.parse_known_args()

    from areal.experimental.weight_update.gateway.app import create_app
    from areal.experimental.weight_update.gateway.config import WeightUpdateConfig
    from areal.infra.utils.http import (
        get_default_uvicorn_kwargs,
        validate_admin_api_key,
    )
    from areal.utils.logging import suppress_http_loggers

    validate_admin_api_key(args.host, args.admin_api_key)

    config = WeightUpdateConfig(
        host=args.host,
        gateway_port=args.port,
        admin_api_key=args.admin_api_key,
        log_level=args.log_level,
        init_timeout_s=args.init_timeout,
        update_timeout_s=args.update_timeout,
    )

    import uvicorn

    suppress_http_loggers()
    app = create_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.gateway_port,
        log_level=config.log_level,
        access_log=False,
        **get_default_uvicorn_kwargs(),
    )


if __name__ == "__main__":
    main()

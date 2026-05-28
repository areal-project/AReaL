# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

import uvicorn

from areal.experimental.training_service.data_proxy.app import create_app
from areal.experimental.training_service.data_proxy.config import TrainDataProxyConfig
from areal.infra.utils.http import (
    get_default_uvicorn_kwargs,
    validate_admin_api_key,
)
from areal.utils.logging import suppress_http_loggers


def main():
    parser = argparse.ArgumentParser(description="AReaL Train Data Proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9082)
    parser.add_argument("--worker-addrs", required=True)
    parser.add_argument("--admin-api-key", default="areal-admin-key")
    parser.add_argument("--idle-timeout", type=float, default=60.0)
    parser.add_argument("--warmup-timeout", type=float, default=120.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--log-level",
        default="warning",
        choices=["debug", "info", "warning", "error"],
    )
    args, _ = parser.parse_known_args()

    validate_admin_api_key(args.host, args.admin_api_key)

    worker_addrs = [
        addr.strip() for addr in args.worker_addrs.split(",") if addr.strip()
    ]

    config = TrainDataProxyConfig(
        host=args.host,
        port=args.port,
        worker_addrs=worker_addrs,
        admin_api_key=args.admin_api_key,
        log_level=args.log_level,
        request_timeout=args.request_timeout,
        warmup_timeout=args.warmup_timeout,
    )

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

# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint: python -m areal.infra.data_service.gateway"""

from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description="AReaL Data Service Gateway")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--admin-api-key", default="areal-data-admin")
    parser.add_argument("--router-addr", default="http://localhost:8091")
    parser.add_argument("--router-timeout", type=float, default=2.0)
    parser.add_argument("--forward-timeout", type=float, default=60.0)
    args, _ = parser.parse_known_args()

    from areal.infra.data_service.gateway.app import create_gateway_app
    from areal.infra.data_service.gateway.config import GatewayConfig
    from areal.utils.logging import suppress_http_loggers

    config = GatewayConfig(
        host=args.host,
        port=args.port,
        admin_api_key=args.admin_api_key,
        router_addr=args.router_addr,
        router_timeout=args.router_timeout,
        forward_timeout=args.forward_timeout,
    )

    import uvicorn

    suppress_http_loggers()
    app = create_gateway_app(config)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()

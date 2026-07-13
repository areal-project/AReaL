# SPDX-License-Identifier: Apache-2.0

"""``python -m areal.v2.agent_service.data_proxy``"""

import argparse

import uvicorn

from areal.infra.utils.http import validate_admin_api_key

from ..auth import DEFAULT_ADMIN_API_KEY
from .app import create_data_proxy_app
from .config import DataProxyConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent DataProxy")
    parser.add_argument("--worker-addr", required=True, help="Worker HTTP address")
    parser.add_argument("--admin-api-key", default=DEFAULT_ADMIN_API_KEY)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--session-timeout", type=int, default=3600)
    parser.add_argument(
        "--log-level", choices=["debug", "info", "warning", "error"], default="warning"
    )
    args = parser.parse_args()

    validate_admin_api_key(
        args.host,
        args.admin_api_key,
        default_key=DEFAULT_ADMIN_API_KEY,
    )

    config = DataProxyConfig(
        host=args.host,
        port=args.port,
        worker_addr=args.worker_addr,
        admin_api_key=args.admin_api_key,
        request_timeout=args.request_timeout,
        session_timeout=args.session_timeout,
        log_level=args.log_level,
    )
    uvicorn.run(
        create_data_proxy_app(config),
        host=config.host,
        port=config.port,
        log_level=config.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()

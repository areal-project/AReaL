# SPDX-License-Identifier: Apache-2.0

"""CLI entrypoint: python -m areal.experimental.inference_service.data_proxy"""

from __future__ import annotations

import argparse

import uvicorn

from areal.experimental.inference_service.data_proxy.app import create_app
from areal.experimental.inference_service.data_proxy.config import DataProxyConfig
from areal.infra.utils.http import (
    get_default_uvicorn_kwargs,
    validate_admin_api_key,
)
from areal.utils.logging import suppress_http_loggers
from areal.utils.network import format_hostport


def main():
    parser = argparse.ArgumentParser(description="AReaL Data Proxy")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument(
        "--backend-addr",
        default="http://localhost:30000",
    )
    parser.add_argument(
        "--backend-type",
        default="sglang",
        choices=("sglang", "vllm"),
    )
    parser.add_argument(
        "--tokenizer-path",
        required=True,
    )
    parser.add_argument(
        "--log-level",
        default="warning",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
    )
    parser.add_argument(
        "--set-reward-finish-timeout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--admin-api-key",
        default="areal-admin-key",
    )
    parser.add_argument(
        "--callback-server-addr",
        default="",
    )
    parser.add_argument(
        "--tool-call-parser",
        default="qwen",
    )
    parser.add_argument(
        "--reasoning-parser",
        default="qwen3",
    )
    parser.add_argument(
        "--engine-max-tokens",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--chat-template-type",
        default="hf",
        choices=("hf", "concat"),
    )
    args, _ = parser.parse_known_args()

    validate_admin_api_key(args.host, args.admin_api_key)

    # Resolve the actual serving host (replace 0.0.0.0 with real IP)
    from areal.utils.network import gethostip

    serving_host = args.host
    if serving_host == "0.0.0.0":
        serving_host = gethostip()

    config = DataProxyConfig(
        host=args.host,
        port=args.port,
        backend_addr=args.backend_addr,
        backend_type=args.backend_type,
        tokenizer_path=args.tokenizer_path,
        log_level=args.log_level,
        request_timeout=args.request_timeout,
        set_reward_finish_timeout=args.set_reward_finish_timeout,
        admin_api_key=args.admin_api_key,
        callback_server_addr=args.callback_server_addr,
        serving_addr=format_hostport(serving_host, args.port),
        tool_call_parser=args.tool_call_parser,
        reasoning_parser=args.reasoning_parser,
        engine_max_tokens=args.engine_max_tokens,
        chat_template_type=args.chat_template_type,
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

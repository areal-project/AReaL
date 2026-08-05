# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Adapted from sglang.srt.entrypoints.http_server.launch_server
# (SGLang commit pinned in this repo).
#
# AReaL additions are between # ---- BEGIN AREAL ---- / # ---- END AREAL ----
# markers. Everything else mirrors the upstream launch_server flow.
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import sys


def areal_launch_server(server_args) -> None:
    from sglang.srt.entrypoints.engine import init_tokenizer_manager
    from sglang.srt.entrypoints.http_server import (
        _execute_server_warmup,
        app,
        launch_server,
    )
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process

    # ---- BEGIN AREAL ----
    from areal.v2.inference_service.sglang.awex import (
        register_awex_endpoints,
    )
    from areal.v2.inference_service.sglang.rpc_proxy import RpcProxy
    from areal.v2.inference_service.sglang.scheduler import (
        areal_run_scheduler_process,
        create_result_ipc,
    )
    # ---- END AREAL ----

    # ---- BEGIN AREAL ----
    result_ipc = create_result_ipc()
    rpc_proxies = []
    # ---- END AREAL ----

    # ---- BEGIN AREAL ----
    def areal_init_tokenizer_manager(server_args, port_args):
        tokenizer_manager, template_manager = init_tokenizer_manager(
            server_args, port_args
        )
        rpc_proxy = RpcProxy(port_args, result_ipc)
        register_awex_endpoints(app, rpc_proxy)
        rpc_proxies.append(rpc_proxy)
        return tokenizer_manager, template_manager

    # ---- END AREAL ----

    try:
        launch_server(
            server_args,
            init_tokenizer_manager_func=areal_init_tokenizer_manager,
            run_scheduler_process_func=areal_run_scheduler_process,
            run_detokenizer_process_func=run_detokenizer_process,
            execute_warmup_func=_execute_server_warmup,
        )
    finally:
        # ---- BEGIN AREAL ----
        for rpc_proxy in rpc_proxies:
            rpc_proxy.close()
        # ---- END AREAL ----


if __name__ == "__main__":
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.srt.utils.common import suppress_noisy_warnings

    suppress_noisy_warnings()

    server_args = prepare_server_args(sys.argv[1:])

    try:
        areal_launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)

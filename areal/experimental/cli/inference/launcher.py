# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

from areal.experimental.cli.process import pick_free_port, spawn_process


def spawn_router(
    *,
    host: str,
    admin_api_key: str,
    routing_strategy: str,
    log_level: str,
    log_file: Path,
) -> tuple[int, int]:
    port = pick_free_port()
    cmd = [
        sys.executable,
        "-m",
        "areal.experimental.inference_service.router",
        "--host",
        host,
        "--port",
        str(port),
        "--admin-api-key",
        admin_api_key,
        "--routing-strategy",
        routing_strategy,
        "--log-level",
        log_level,
    ]
    return spawn_process(cmd, log_file), port


def spawn_gateway(
    *,
    host: str,
    port: int,
    admin_api_key: str,
    router_url: str,
    log_level: str,
    log_file: Path,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "areal.experimental.inference_service.gateway",
        "--host",
        host,
        "--port",
        str(port),
        "--admin-api-key",
        admin_api_key,
        "--router-addr",
        router_url,
        "--log-level",
        log_level,
    ]
    return spawn_process(cmd, log_file)


def spawn_sglang(
    *,
    model_path: str,
    host: str,
    port: int,
    tp: int,
    base_gpu_id: int,
    extra_args: list[str],
    log_file: Path,
) -> int:
    from areal.api.cli_args import SGLangConfig

    cfg = SGLangConfig(model_path=model_path, log_requests=True)
    cmd = list(
        SGLangConfig.build_cmd(
            sglang_config=cfg,
            tp_size=tp,
            base_gpu_id=base_gpu_id,
            host=host,
            port=port,
            n_nodes=1,
            node_rank=0,
            pp_size=1,
        )
    )
    cmd.extend(extra_args)
    return spawn_process(cmd, log_file)


def spawn_vllm(
    *,
    model_path: str,
    host: str,
    port: int,
    tp: int,
    pp: int,
    extra_args: list[str],
    log_file: Path,
) -> int:
    from areal.api.cli_args import vLLMConfig

    cfg = vLLMConfig(model=model_path)
    cmd = list(
        vLLMConfig.build_cmd(
            vllm_config=cfg,
            tp_size=tp,
            pp_size=pp,
            host=host,
            port=port,
        )
    )
    cmd.extend(extra_args)
    return spawn_process(cmd, log_file)


def spawn_data_proxy(
    *,
    host: str,
    port: int,
    backend_addr: str,
    backend_type: str,
    tokenizer_path: str,
    admin_api_key: str,
    log_level: str,
    extra_args: list[str],
    log_file: Path,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "areal.experimental.inference_service.data_proxy",
        "--host",
        host,
        "--port",
        str(port),
        "--backend-addr",
        backend_addr,
        "--backend-type",
        backend_type,
        "--tokenizer-path",
        tokenizer_path,
        "--admin-api-key",
        admin_api_key,
        "--log-level",
        log_level,
    ]
    cmd.extend(extra_args)
    return spawn_process(cmd, log_file)

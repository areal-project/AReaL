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


def physical_base_from_cvd(cvd: str) -> int:
    entries = [e.strip() for e in cvd.split(",") if e.strip()]
    if not entries:
        raise ValueError(f"CUDA_VISIBLE_DEVICES is empty: {cvd!r}")
    ids = sorted(int(e) for e in entries)
    if ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError(f"CUDA_VISIBLE_DEVICES is not contiguous: {cvd!r}")
    return ids[0]


def normalize_alloc_conf_for_inference(env: dict) -> None:
    """Force expandable_segments off for SGLang processes.

    The colocated guard env ships ``expandable_segments:True`` for the actor
    (fragmentation fix), and the forked SGLang server inherits it. SGLang's
    memory saver cannot physically unmap/remap expandable segments, so weight
    pages resumed after a release are invalid CUDA IPC/copy targets. Mirror of
    the donor's ``_normalize_cuda_alloc_conf_for_memory_saver``; must run
    before any CUDA initialization.
    """
    conf = env.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in conf.lower():
        return
    tokens = [
        t.strip()
        for t in conf.split(",")
        if t.strip() and not t.strip().lower().startswith("expandable_segments")
    ]
    tokens.append("expandable_segments:False")
    env["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(tokens)


def unmask_gpus_for_awex_colocate(server_args) -> None:
    """Restore the v1 GPU layout: full visibility + physical base_gpu_id.

    AWEX colocate keys every per-device MetaServer entry on
    ``torch.cuda.current_device()``, which must equal the physical GPU id
    (unique per host). A CVD mask makes it a logical id and collides across
    same-host servers, so translate the mask into ``base_gpu_id`` and drop it
    before SGLang forks its scheduler subprocesses.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return
    server_args.base_gpu_id += physical_base_from_cvd(cvd)
    del os.environ["CUDA_VISIBLE_DEVICES"]


def areal_launch_server(server_args) -> None:
    from sglang.srt.entrypoints.engine import Engine, init_tokenizer_manager
    from sglang.srt.entrypoints.http_server import (
        _execute_server_warmup,
        _setup_and_run_http_server,
        app,
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
    # ---- END AREAL ----

    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager,
        # ---- BEGIN AREAL ----
        run_scheduler_process_func=areal_run_scheduler_process,
        # ---- END AREAL ----
        run_detokenizer_process_func=run_detokenizer_process,
    )

    # ---- BEGIN AREAL ----
    if tokenizer_manager is None:
        return
    # ---- END AREAL ----

    # ---- BEGIN AREAL ----
    rpc_proxy = RpcProxy(port_args, result_ipc)
    register_awex_endpoints(app, rpc_proxy)
    # ---- END AREAL ----

    try:
        _setup_and_run_http_server(
            server_args,
            tokenizer_manager,
            template_manager,
            port_args,
            scheduler_init_result.scheduler_infos,
            subprocess_watchdog,
            execute_warmup_func=_execute_server_warmup,
        )
    finally:
        # ---- BEGIN AREAL ----
        rpc_proxy.close()
        # ---- END AREAL ----


if __name__ == "__main__":
    normalize_alloc_conf_for_inference(os.environ)

    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.srt.utils.common import suppress_noisy_warnings

    suppress_noisy_warnings()

    server_args = prepare_server_args(sys.argv[1:])

    if os.environ.get("AREAL_AWEX_COLOCATE") == "1":
        unmask_gpus_for_awex_colocate(server_args)

    try:
        areal_launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)

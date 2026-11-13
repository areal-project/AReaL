# SPDX-License-Identifier: Apache-2.0
"""HTTP bootstrap for SGLang 0.5.10, which predates native plugin discovery."""

import os
import sys


def run_scheduler_process(*args, **kwargs):
    """Pickleable spawn target: register before constructing the scheduler."""
    entrypoint = os.environ["AREAL_SAMPLING_MASK_ENTRYPOINT"]
    if entrypoint == "areal.engine.awex.plugins.sglang_plugin":
        # AWEX must establish its memory-saver hooks before SGLang imports.
        from areal.engine.awex.plugins.sglang_plugin import (
            awex_run_scheduler_process as run,
        )
    else:
        from sglang.srt.managers.scheduler import run_scheduler_process as run

    from areal.engine.sglang_sampling_mask_plugin import register

    register()
    return run(*args, **kwargs)


def main() -> None:
    entrypoint = os.environ["AREAL_SAMPLING_MASK_ENTRYPOINT"]
    if entrypoint == "areal.engine.awex.plugins.sglang_plugin":
        from areal.engine.awex.plugins.sglang_plugin import (
            _load_sglang_plugins_if_available,
        )

        _load_sglang_plugins_if_available()
    from areal.engine.sglang_sampling_mask_plugin import register

    register()
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    args = prepare_server_args(sys.argv[1:])
    if args.grpc_mode or args.encoder_only or args.use_ray:
        raise ValueError("Legacy sampling mask launcher requires the HTTP server")
    if args.tokenizer_worker_num != 1:
        raise ValueError(
            "Legacy sampling mask launcher requires tokenizer_worker_num=1; "
            "separate tokenizer workers do not run its metadata initializer"
        )
    if entrypoint == "areal.engine.awex.plugins.sglang_plugin":
        from areal.v2.inference_service.sglang.launch_server import (
            normalize_rocm_attention_backend,
        )

        normalize_rocm_attention_backend(args)
        args.skip_server_warmup = True
    try:
        launch_server(args, run_scheduler_process_func=run_scheduler_process)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()

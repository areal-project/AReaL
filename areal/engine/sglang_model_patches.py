# SPDX-License-Identifier: Apache-2.0
"""Apply validated model-specific patches before starting an SGLang server."""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from areal.engine._sglang_source_patch import (
    install_source_patches,
    installed_sglang_root,
)

_BOOTSTRAP = "areal.engine.sglang_model_patches"
_ENTRYPOINTS = (
    "areal.v2.inference_service.sglang.launch_server",
    "areal.engine.awex.plugins.sglang_plugin",
    "sglang.launch_server",
)


def wrap_server_command(command: list[str]) -> list[str]:
    """Run preparation in the server's interpreter, preserving its chosen entry."""
    if len(command) < 3 or command[1] != "-m" or command[2] not in _ENTRYPOINTS:
        return command
    return [
        *command[:2],
        _BOOTSTRAP,
        "--entrypoint",
        command[2],
        "--",
        *command[3:],
    ]


def apply_model_patches(server_args: Sequence[str]) -> None:
    """Apply model-specific preprocessing fixes, independent of model path."""
    if "--help" in server_args or "-h" in server_args:
        return
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model-path", "--model")
    parser.add_argument("--revision")
    args, _ = parser.parse_known_args(server_args)
    if not args.model_path:
        return  # Allow the original entry point to handle --help/argument errors.
    if args.model_path.lower() in ("none", "dummy"):
        return
    model_path = Path(args.model_path).expanduser()
    if model_path.is_file():
        if model_path.suffix != ".json":
            return  # GGUF and other binary formats use SGLang's own config loader.
        config_path = model_path
    else:
        config_path = model_path / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
    else:
        if "://" in args.model_path:
            return  # Connectors normalize remote storage paths inside SGLang.
        # Raw config access needs no custom model registration or remote code.
        from transformers import PretrainedConfig

        try:
            config, _ = PretrainedConfig.get_config_dict(
                args.model_path, revision=args.revision
            )
        except (OSError, ValueError):
            return  # Let the original entry point handle other config formats.
    from areal.engine import sglang_qwen_vl_cpu

    if config.get("model_type") in sglang_qwen_vl_cpu.MODEL_TYPES:
        # Let existing SGLang installations and third-party processors run
        # unchanged. Users explicitly select this integration in the server
        # environment; spawned tokenizer workers inherit that selection.
        package = "areal.engine.sglang_processors"
        if os.environ.get("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE") != package:
            return
        root = installed_sglang_root()
        install_source_patches(
            [
                (
                    root / sglang_qwen_vl_cpu.TARGET_RELATIVE,
                    sglang_qwen_vl_cpu.patched_source,
                ),
                (
                    root / sglang_qwen_vl_cpu.BASE_RELATIVE,
                    sglang_qwen_vl_cpu.patched_base_source,
                ),
            ],
            lock_path=root / ".areal-sglang-patches.lock",
        )
        return
    if config.get("model_type") != "qwen4_exp":
        return

    from areal.engine import (
        sglang_mm_executor,
        sglang_qsa_compress_gather,
        sglang_qwen4_vl,
    )

    root = installed_sglang_root()
    patches = (sglang_mm_executor, sglang_qsa_compress_gather, sglang_qwen4_vl)
    install_source_patches(
        [(root / patch.TARGET_RELATIVE, patch.patched_source) for patch in patches],
        lock_path=root / ".areal-sglang-patches.lock",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--entrypoint", required=True, choices=_ENTRYPOINTS)
    parser.add_argument("server_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    server_args = args.server_args
    if server_args[:1] == ["--"]:
        server_args = server_args[1:]
    apply_model_patches(server_args)
    # Exec preserves PID and the original __main__/multiprocessing spawn behavior.
    os.execv(sys.executable, [sys.executable, "-m", args.entrypoint, *server_args])


if __name__ == "__main__":
    main()

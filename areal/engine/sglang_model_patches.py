# SPDX-License-Identifier: Apache-2.0
"""Apply validated model-specific patches before starting an SGLang server."""

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from areal.engine._sglang_source_patch import (
    install_source_patches,
    installed_sglang_root,
    patch_source,
)

_BOOTSTRAP = "areal.engine.sglang_model_patches"
_ENTRYPOINTS = (
    "areal.v2.inference_service.sglang.launch_server",
    "areal.engine.awex.plugins.sglang_plugin",
    "sglang.launch_server",
)


# Qwen3.8-27B runtime patches ported from e055262d6d/d782649682 on
# SGLang afe21bcddb. They do not enable speculative decoding or training MTP.
# Use architecture fields rather than checkpoint names, which change on export.
_QWEN38_TEXT_SHAPE = {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "vocab_size": 248320,
}


def _matches_qwen38_27b(config: dict[str, Any]) -> bool:
    if config.get("model_type") != "qwen3_5":
        return False
    text = config.get("text_config", {})
    return isinstance(text, dict) and all(
        text.get(key) == value for key, value in _QWEN38_TEXT_SHAPE.items()
    )


_QWEN38_MROPE_OLD = "            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0).to(device=device)\n"
_QWEN38_MROPE_NEW = """            # The positions below consume this copy on the same stream. Avoid
            # waiting on earlier speculative work from the scheduling thread.
            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0).to(
                device=device, non_blocking=True
            )
"""
_QWEN38_ABORT_INIT_OLD = """class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__()
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type
"""
_QWEN38_ABORT_INIT_NEW = """class FINISH_ABORT(BaseFinishReason):
    def __init__(
        self, message=None, status_code=None, err_type=None, *, check_completion=False
    ):
        super().__init__()
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type
        # External cancellation may race with an in-flight decode completion.
        # Internal errors, including grammar failures, must still win immediately.
        self.check_completion = check_completion
"""
_QWEN38_FINISH_OLD = """        if self.to_finish:
            self.finished_reason = self.to_finish
            self.to_finish = None
            return
"""
_QWEN38_FINISH_NEW = """        pending_abort = None
        if self.to_finish:
            finish_reason = self.to_finish
            self.to_finish = None
            if (
                isinstance(finish_reason, FINISH_ABORT)
                and finish_reason.check_completion
                and new_accepted_len > 0
                and self.output_ids
            ):
                pending_abort = finish_reason
            else:
                self.finished_reason = finish_reason
                return
"""
_QWEN38_FINISH_END_OLD = """    def reset_for_retract(self):
"""
_QWEN38_FINISH_END_NEW = """        if pending_abort is not None and not self.finished():
            self.finished_reason = pending_abort

    def reset_for_retract(self):
"""
_QWEN38_SCHEDULER_OLD = "                    req.request_finish(FINISH_ABORT())\n"
_QWEN38_SCHEDULER_NEW = (
    "                    req.request_finish(FINISH_ABORT(check_completion=True))\n"
)

# Whole-file fingerprints prevent silently applying a partial fix to an
# incompatible fork. Both the unpatched baseline and exact patched files work.
# All four files are validated together by install_source_patches before writing.
_QWEN38_PATCHES = (
    (
        Path("srt/model_executor/forward_batch_info.py"),
        "864ef6b898468bd5fccdd29bb52c76fb9f469b87991017939a5bb3a5d4c1bc78",
        "ab9456594356ad525ecf035a2242b28147743f89151a75232419c7488fe04e61",
        [(_QWEN38_MROPE_OLD, _QWEN38_MROPE_NEW)],
    ),
    (
        Path("srt/managers/req_types.py"),
        "e9abafe50f787e28f6686502285ccd76a58cb39afbfe4ffcb155121869e7d5a3",
        "6f6df3a8d3e0a67007338cfb86bf42040443dec21ff27eac8da1c24c4342c4c0",
        [(_QWEN38_ABORT_INIT_OLD, _QWEN38_ABORT_INIT_NEW)],
    ),
    (
        Path("srt/managers/schedule_batch.py"),
        "65056becda1b1e4815035e68a2e102db5ed2eb72e2c94aee5a307fff295ed067",
        "52e98e70e17901c11466a001af70164d64115beb7d059b4eb1d0294984e0f518",
        [
            (_QWEN38_FINISH_OLD, _QWEN38_FINISH_NEW),
            (_QWEN38_FINISH_END_OLD, _QWEN38_FINISH_END_NEW),
        ],
    ),
    (
        Path("srt/managers/scheduler.py"),
        "d595f4d1575940cc13a472ec5d6631f1bdc6ca09f7785906d2995dad92a9b48a",
        "34e5c5316287b7f094673822cc71358c0cca49324faa41569c0f8d505fa0a551",
        [(_QWEN38_SCHEDULER_OLD, _QWEN38_SCHEDULER_NEW)],
    ),
)


def _qwen38_patch_targets(root: Path) -> list[tuple[Path, Callable[[bytes], bytes]]]:
    return [
        (
            root / relative,
            partial(
                patch_source,
                target_relative=relative,
                replacements_by_sha256={expected: replacements},
                patched_sha256=patched,
            ),
        )
        for relative, expected, patched, replacements in _QWEN38_PATCHES
    ]


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

    if _matches_qwen38_27b(config):
        root = installed_sglang_root()
        install_source_patches(
            _qwen38_patch_targets(root),
            lock_path=root / ".areal-sglang-patches.lock",
        )
        from areal.utils import logging

        logging.getLogger("SGLangWrapper").info(
            "Qwen3.8-27B runtime patches verified: nonblocking speculative "
            "mRoPE copies and external-abort completion handling."
        )

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

# SPDX-License-Identifier: Apache-2.0
"""Pinned SGLang runtime patches for the Qwen3.8-27B architecture.

Ported from SGLang commits e055262d6d and d782649682 on afe21bcddb.
These patches do not enable speculative decoding or training-side MTP.
"""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from areal.engine._sglang_source_patch import patch_source

# Use architecture fields rather than checkpoint names, which change on export.
_TEXT_SHAPE = {
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "num_hidden_layers": 64,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "vocab_size": 248320,
}


def matches_model(config: dict[str, Any]) -> bool:
    if config.get("model_type") != "qwen3_5":
        return False
    text = config.get("text_config", {})
    return isinstance(text, dict) and all(
        text.get(key) == value for key, value in _TEXT_SHAPE.items()
    )


MROPE_OLD = "            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0).to(device=device)\n"
MROPE_NEW = """            # The positions below consume this copy on the same stream. Avoid
            # waiting on earlier speculative work from the scheduling thread.
            mrope_delta_tensor = torch.stack(mrope_deltas, dim=0).to(
                device=device, non_blocking=True
            )
"""
ABORT_INIT_OLD = """class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__()
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type
"""
ABORT_INIT_NEW = """class FINISH_ABORT(BaseFinishReason):
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
FINISH_OLD = """        if self.to_finish:
            self.finished_reason = self.to_finish
            self.to_finish = None
            return
"""
FINISH_NEW = """        pending_abort = None
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
FINISH_END_OLD = """    def reset_for_retract(self):
"""
FINISH_END_NEW = """        if pending_abort is not None and not self.finished():
            self.finished_reason = pending_abort

    def reset_for_retract(self):
"""
SCHEDULER_OLD = "                    req.request_finish(FINISH_ABORT())\n"
SCHEDULER_NEW = (
    "                    req.request_finish(FINISH_ABORT(check_completion=True))\n"
)

# Whole-file fingerprints prevent silently applying a partial fix to an
# incompatible fork. Both the unpatched baseline and exact patched files work.
# All four files are validated together by install_source_patches before writing.
PATCHES = (
    (
        Path("srt/model_executor/forward_batch_info.py"),
        "864ef6b898468bd5fccdd29bb52c76fb9f469b87991017939a5bb3a5d4c1bc78",
        "ab9456594356ad525ecf035a2242b28147743f89151a75232419c7488fe04e61",
        [(MROPE_OLD, MROPE_NEW)],
    ),
    (
        Path("srt/managers/req_types.py"),
        "e9abafe50f787e28f6686502285ccd76a58cb39afbfe4ffcb155121869e7d5a3",
        "6f6df3a8d3e0a67007338cfb86bf42040443dec21ff27eac8da1c24c4342c4c0",
        [(ABORT_INIT_OLD, ABORT_INIT_NEW)],
    ),
    (
        Path("srt/managers/schedule_batch.py"),
        "65056becda1b1e4815035e68a2e102db5ed2eb72e2c94aee5a307fff295ed067",
        "52e98e70e17901c11466a001af70164d64115beb7d059b4eb1d0294984e0f518",
        [(FINISH_OLD, FINISH_NEW), (FINISH_END_OLD, FINISH_END_NEW)],
    ),
    (
        Path("srt/managers/scheduler.py"),
        "d595f4d1575940cc13a472ec5d6631f1bdc6ca09f7785906d2995dad92a9b48a",
        "34e5c5316287b7f094673822cc71358c0cca49324faa41569c0f8d505fa0a551",
        [(SCHEDULER_OLD, SCHEDULER_NEW)],
    ),
)


def patch_targets(root: Path) -> list[tuple[Path, Callable[[bytes], bytes]]]:
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
        for relative, expected, patched, replacements in PATCHES
    ]

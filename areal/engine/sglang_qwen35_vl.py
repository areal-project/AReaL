# SPDX-License-Identifier: Apache-2.0
"""Preserve the model image budget for dense Qwen3.5 in pinned SGLang."""

from pathlib import Path

from areal.engine._sglang_source_patch import patch_source

EXPECTED_SHA256 = "285fd0e0eef80d8c54db71724ef7cc369ae9379023437d3e2015deb65df69e17"
PATCHED_SHA256 = "778e63e389f0d96d0401217fcd82f207ab4a2a758a0dbaf82ec977d3dcfa882d"
TARGET_RELATIVE = Path("srt/multimodal/processors/qwen_vl.py")
CAP_OLD = '            if envs.SGLANG_VLM_QWEN_LIMIT_PIXEL.get() and self.model_type in (\n                "qwen3_vl",\n                "qwen3_vl_moe",\n                "qwen3_5",\n                "qwen3_5_moe",\n            ):\n'
CAP_NEW = '            if envs.SGLANG_VLM_QWEN_LIMIT_PIXEL.get() and self.model_type in (\n                "qwen3_vl",\n                "qwen3_vl_moe",\n                "qwen3_5_moe",\n            ):\n'
RESIZE_OLD = '            self.model_type != "paddleocr_vl"\n'
RESIZE_NEW = '            self.model_type not in ("paddleocr_vl", "qwen3_5")\n'


def patched_source(source: bytes) -> bytes:
    """Let HF perform the only resize, keeping its configured pixel bounds.

    HF retains its model-required resizing. The paired base patch runs that
    preprocessing on CPU.
    Other model types retain their existing SGLang preprocessing behavior.
    """
    return patch_source(
        source,
        target_relative=TARGET_RELATIVE,
        replacements_by_sha256={
            EXPECTED_SHA256: [(CAP_OLD, CAP_NEW), (RESIZE_OLD, RESIZE_NEW)],
        },
        patched_sha256=PATCHED_SHA256,
    )


BASE_RELATIVE = Path("srt/multimodal/processors/base_processor.py")
BASE_EXPECTED_SHA256 = (
    "2396ed6de130e17048eac64e97a2c665b7a33e8cdb42427b7ba5dd1e8be1ef61"
)
BASE_PATCHED_SHA256 = "29d5bef60a84ca2f8ac4b55d6f119a25a292b176328f7148c13dedd9ebac60fb"
DEVICE_OLD = "            if _is_cpu or get_global_server_args().rl_on_policy_target is not None:\n"
DEVICE_NEW = '            if (\n                _is_cpu\n                or get_global_server_args().rl_on_policy_target is not None\n                or getattr(self, "model_type", None) == "qwen3_5"\n            ):\n'


def patched_base_source(source: bytes) -> bytes:
    """Use CPU HF preprocessing for dense Qwen while retaining other devices."""
    return patch_source(
        source,
        target_relative=BASE_RELATIVE,
        replacements_by_sha256={BASE_EXPECTED_SHA256: [(DEVICE_OLD, DEVICE_NEW)]},
        patched_sha256=BASE_PATCHED_SHA256,
    )

# SPDX-License-Identifier: Apache-2.0
"""Use CPU/PIL Qwen decoding and preserve native SGLang HF device policy."""

from pathlib import Path

from areal.engine._sglang_source_patch import patch_source

MODEL_TYPES = ("qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe")
TARGET_RELATIVE = Path("srt/multimodal/processors/qwen_vl.py")
BASE_RELATIVE = Path("srt/multimodal/processors/base_processor.py")
EXPECTED_SHA256 = "285fd0e0eef80d8c54db71724ef7cc369ae9379023437d3e2015deb65df69e17"
PATCHED_SHA256 = "aa7980ea2dd72d699069e673a22a5c892e230b891de9047378e1a7fc1a1fe114"
BASE_EXPECTED_SHA256 = (
    "2396ed6de130e17048eac64e97a2c665b7a33e8cdb42427b7ba5dd1e8be1ef61"
)
LEGACY_CPU_SHA256 = "21fcee80ac708d3d8fabda2f476cb420b3b05504b1b13d5860654674b192e27b"
DECODE_OLD = "    supports_transformers_backend = True\n"
DECODE_NEW = DECODE_OLD + "    gpu_image_decode = False\n"
NATIVE_DEVICE = "            if _is_cpu or get_global_server_args().rl_on_policy_target is not None:\n"
LEGACY_CPU_DEVICE = (
    "            if (\n"
    "                _is_cpu\n"
    "                or get_global_server_args().rl_on_policy_target is not None\n"
    '                or getattr(self, "model_type", None) in (\n'
    '                    "qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"\n'
    "                )\n"
    "            ):\n"
)


def patched_source(source: bytes) -> bytes:
    """Decode JPEGs as PIL images; retain native pixel bounds and resize."""
    return patch_source(
        source,
        target_relative=TARGET_RELATIVE,
        replacements_by_sha256={EXPECTED_SHA256: [(DECODE_OLD, DECODE_NEW)]},
        patched_sha256=PATCHED_SHA256,
    )


def patched_base_source(source: bytes) -> bytes:
    """Restore native HF dispatch if the earlier CPU override was installed."""
    return patch_source(
        source,
        target_relative=BASE_RELATIVE,
        replacements_by_sha256={
            LEGACY_CPU_SHA256: [(LEGACY_CPU_DEVICE, NATIVE_DEVICE)]
        },
        patched_sha256=BASE_EXPECTED_SHA256,
    )

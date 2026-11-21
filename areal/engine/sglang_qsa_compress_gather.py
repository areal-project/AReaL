# SPDX-License-Identifier: Apache-2.0
"""Backport the QSA gather clamp when the installed SGLang lacks it."""

import hashlib
from pathlib import Path

from areal.engine._sglang_source_patch import patch_installed_module, patch_source

EXPECTED_SHA256 = "bb57ce1e9abc4fbfcba2c9aaaf125b9e625983966497df57165b6d4c6461afe2"
PATCHED_SHA256 = "72c5d471a67729446ec9b1c0d810bc87bc85149c381c213adfb4119eddde9a7c"
V0520_SHA256 = "1ca45f9ed46c088bd8ba734f34bc085d852c0141f5e7f5f0f933603b33e14b9c"
TARGET_RELATIVE = Path("srt/layers/attention/qsa/qsa_indexer.py")
OLD = "            source_keys = token_k\n            source_rope = metadata.extend_rope_matrix\n"
NEW = (
    "            source_keys = token_k\n"
    "            group_locs = group_locs.clamp_max(source_keys.shape[0] - 1)\n"
    "            source_rope = metadata.extend_rope_matrix\n"
)


def patched_source(source: bytes) -> bytes:
    """Clamp padded gathers, allowing repeat startup on the validated output."""
    if hashlib.sha256(source).hexdigest() == V0520_SHA256:
        # SGLang 0.5.20 already clamps group_locs before the gather.
        return source
    return patch_source(
        source,
        target_relative=TARGET_RELATIVE,
        replacements_by_sha256={EXPECTED_SHA256: [(OLD, NEW)]},
        patched_sha256=PATCHED_SHA256,
    )


def main() -> None:
    patch_installed_module(__doc__, TARGET_RELATIVE, patched_source)


if __name__ == "__main__":
    main()

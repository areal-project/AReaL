# SPDX-License-Identifier: Apache-2.0
"""Align Qwen4 image preprocessing in the pinned SGLang installation."""

import argparse
import hashlib
import importlib.util
from pathlib import Path

EXPECTED_SHA256 = "a78b01a05c387d04a185c631aa5037904c5e668d8f8217976b9b167aa1dd6666"
PATCHED_SHA256 = "7c5c20550fed1a385daa07ce4c9b5a0b74151cea22d7762299fae47a8169e189"
TARGET_RELATIVE = Path("srt/multimodal/processors/qwen_vl.py")
OLD = '        if self.model_type in ("qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"):\n'
NEW = (
    "        if self.model_type in (\n"
    '            "qwen3_vl",\n'
    '            "qwen3_vl_moe",\n'
    '            "qwen3_5",\n'
    '            "qwen3_5_moe",\n'
    '            "qwen4_exp",\n'
    "        ):\n"
)


def patched_source(source: bytes) -> bytes:
    """Use the model's image factor and pixel bounds, refusing unknown sources.

    Qwen4 uses patch_size=16 and merge_size=2. The legacy factor=28 followed
    by the HF processor's factor=32 changes image token counts. Preserve the
    model's pixel bounds as well; applying a separate Qwen3 pixel cap here
    would disagree with the training processor on larger images.
    """
    before = hashlib.sha256(source).hexdigest()
    if before == PATCHED_SHA256:
        return source
    if before != EXPECTED_SHA256:
        raise RuntimeError("Unknown SGLang image processor; refusing to modify")
    text = source.decode()
    if text.count(OLD) != 1:
        raise RuntimeError("Expected exactly one processor image-size branch")
    patched = text.replace(OLD, NEW).encode()
    if hashlib.sha256(patched).hexdigest() != PATCHED_SHA256:
        raise RuntimeError("Patched image processor does not match validated source")
    compile(patched, str(TARGET_RELATIVE), "exec")
    return patched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, help="Disposable qwen_vl.py copy")
    args = parser.parse_args()
    target = args.target
    if target is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None or spec.origin is None:
            raise RuntimeError("SGLang is not installed; pass --target")
        target = Path(spec.origin).parent / TARGET_RELATIVE
    source = target.read_bytes()
    patched = patched_source(source)
    if patched != source:
        target.write_bytes(patched)


if __name__ == "__main__":
    main()

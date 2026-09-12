# SPDX-License-Identifier: Apache-2.0

"""Backport Qwen3 incomplete-thinking tool-call parsing to SGLang 0.5.10."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import re
from pathlib import Path

EXPECTED_SGLANG_VERSION = "0.5.10.post1"
_QWEN3_CLASS = "class Qwen3Detector(BaseReasoningFormatDetector):"
_TOOL_START_ARGUMENT = '            tool_start_token="<tool_call>",\n'
_STREAM_ARGUMENT = "            stream_reasoning=stream_reasoning,\n"


def patch_qwen3_detector_source(source: str) -> tuple[str, bool]:
    """Add the 0.5.18 Qwen3 tool-call reasoning terminator to 0.5.10."""
    if "tool_start_token: Optional[str] = None" not in source:
        raise RuntimeError("SGLang reasoning base class lacks tool_start_token support")

    class_start = source.find(_QWEN3_CLASS)
    if class_start < 0:
        raise RuntimeError("SGLang Qwen3Detector definition was not found")
    next_class = re.search(r"\nclass \w+Detector\(", source[class_start + 1 :])
    class_end = (
        class_start + 1 + next_class.start() if next_class is not None else len(source)
    )
    qwen3_source = source[class_start:class_end]

    if _TOOL_START_ARGUMENT in qwen3_source:
        return source, False
    if qwen3_source.count(_STREAM_ARGUMENT) != 1:
        raise RuntimeError("Unexpected Qwen3Detector constructor shape")

    patched_qwen3_source = qwen3_source.replace(
        _STREAM_ARGUMENT,
        _STREAM_ARGUMENT + _TOOL_START_ARGUMENT,
        1,
    )
    return source[:class_start] + patched_qwen3_source + source[class_end:], True


def _verify_runtime_patch() -> None:
    module = importlib.import_module("sglang.srt.parser.reasoning_parser")
    module = importlib.reload(module)
    detector = module.Qwen3Detector(stream_reasoning=False, force_reasoning=True)
    result = detector.detect_and_parse(
        "inspect the model<tool_call><function=Read></function></tool_call>"
    )
    assert result.reasoning_text == "inspect the model", result.reasoning_text
    assert result.normal_text.startswith("<tool_call>"), result.normal_text


def main() -> None:
    version = importlib.metadata.version("sglang")
    if version != EXPECTED_SGLANG_VERSION:
        raise RuntimeError(
            f"Expected sglang {EXPECTED_SGLANG_VERSION}, found {version}"
        )

    spec = importlib.util.find_spec("sglang.srt.parser.reasoning_parser")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not locate SGLang reasoning_parser.py")
    path = Path(spec.origin)
    source = path.read_text(encoding="utf-8")
    patched_source, changed = patch_qwen3_detector_source(source)
    compile(patched_source, str(path), "exec")
    if changed:
        path.write_text(patched_source, encoding="utf-8")
    importlib.invalidate_caches()
    _verify_runtime_patch()
    state = "applied" if changed else "already-applied"
    print(f"sglang_qwen3_reasoning_patch {state} version={version} path={path}")


if __name__ == "__main__":
    main()

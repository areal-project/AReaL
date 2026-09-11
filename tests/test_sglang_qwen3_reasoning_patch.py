import pytest

from examples.swe.patch_sglang_qwen3_reasoning import (
    patch_qwen3_detector_source,
)

BASE_SOURCE = """
class BaseReasoningFormatDetector:
    def __init__(
        self,
        tool_start_token: Optional[str] = None,
    ):
        pass


class Qwen3Detector(BaseReasoningFormatDetector):
    def __init__(self, stream_reasoning=True):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            continue_final_message=continue_final_message,
        )


class KimiDetector(BaseReasoningFormatDetector):
    pass
"""


def test_patch_qwen3_detector_source_adds_tool_start_token():
    patched, changed = patch_qwen3_detector_source(BASE_SOURCE)

    assert changed is True
    assert patched.count('tool_start_token="<tool_call>"') == 1
    qwen3_source = patched.split("class Qwen3Detector", 1)[1].split(
        "class KimiDetector", 1
    )[0]
    assert 'tool_start_token="<tool_call>"' in qwen3_source


def test_patch_qwen3_detector_source_is_idempotent():
    patched, _ = patch_qwen3_detector_source(BASE_SOURCE)

    second_pass, changed = patch_qwen3_detector_source(patched)

    assert changed is False
    assert second_pass == patched


def test_patch_qwen3_detector_source_rejects_incompatible_base():
    with pytest.raises(RuntimeError, match="lacks tool_start_token support"):
        patch_qwen3_detector_source(BASE_SOURCE.replace("tool_start_token", "other"))

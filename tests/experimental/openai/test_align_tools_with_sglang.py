# SPDX-License-Identifier: Apache-2.0

import sys

import pytest
from pydantic import BaseModel

from areal.experimental.openai.client import _align_tools_with_sglang

pytest.importorskip(
    "sglang.srt.entrypoints.openai.protocol",
    reason="alignment target is sglang's own Tool model",
)

CHAT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}

FLAT_RESPONSES_TOOL = {
    "type": "function",
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def _sglang_rendering(tool: dict) -> dict:
    from sglang.srt.entrypoints.openai.protocol import Tool

    return Tool(**tool).model_dump()


def test_chat_tool_matches_sglang_native_rendering():
    assert _align_tools_with_sglang([CHAT_TOOL]) == [_sglang_rendering(CHAT_TOOL)]


def test_alignment_adds_the_defaults_that_caused_the_drift():
    function = _align_tools_with_sglang([CHAT_TOOL])[0]["function"]

    assert function["strict"] is False
    assert function["description"] is None
    assert "strict" not in CHAT_TOOL["function"]


def test_flat_responses_tool_is_normalized_to_the_chat_shape():
    assert _align_tools_with_sglang([FLAT_RESPONSES_TOOL]) == _align_tools_with_sglang(
        [CHAT_TOOL]
    )


def test_pydantic_tool_input_is_accepted():
    class Function(BaseModel):
        name: str
        parameters: dict

    class Tool(BaseModel):
        type: str
        function: Function

    aligned = _align_tools_with_sglang(
        [Tool(type="function", function=Function(name="get_weather", parameters={}))]
    )

    assert aligned[0]["function"]["name"] == "get_weather"


def test_alignment_is_idempotent():
    once = _align_tools_with_sglang([CHAT_TOOL])

    assert _align_tools_with_sglang(once) == once


def test_invalid_tool_is_passed_through_without_failing_the_batch():
    aligned = _align_tools_with_sglang([CHAT_TOOL, {"type": "function"}])

    assert aligned[0] == _sglang_rendering(CHAT_TOOL)
    assert aligned[1] == {"type": "function"}


def test_unsupported_tool_type_is_passed_through():
    assert _align_tools_with_sglang(["not-a-tool"]) == ["not-a-tool"]


def test_empty_list_is_unchanged():
    assert _align_tools_with_sglang([]) == []


@pytest.fixture
def sglang_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang.srt.entrypoints.openai.protocol", None)
    monkeypatch.delattr(_align_tools_with_sglang, "_warned_no_sglang", raising=False)


def test_flat_responses_tool_is_still_nested_without_sglang(sglang_missing):
    assert _align_tools_with_sglang([FLAT_RESPONSES_TOOL]) == [CHAT_TOOL]


def test_pydantic_tool_becomes_a_dict_without_sglang(sglang_missing):
    class Function(BaseModel):
        name: str
        parameters: dict

    class Tool(BaseModel):
        type: str
        function: Function

    tool = Tool(type="function", function=Function(**CHAT_TOOL["function"]))

    assert _align_tools_with_sglang([tool]) == [CHAT_TOOL]


def test_chat_tool_keeps_its_fields_without_sglang(sglang_missing):
    aligned = _align_tools_with_sglang([CHAT_TOOL])

    assert aligned == [CHAT_TOOL]
    assert "strict" not in aligned[0]["function"]

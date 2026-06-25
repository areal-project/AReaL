# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from typing import Any

import pytest

import areal.experimental.openai.client as client_module
from areal.api import ModelRequest, ModelResponse
from areal.experimental.openai import ArealOpenAI
from areal.experimental.openai.client import (
    _messages_for_hf_chat_template,
    concat_prompt_token_ids_with_parent,
)
from areal.experimental.openai.types import InteractionWithTokenLogpReward


class RecordingTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self):
        self.template_messages: list[list[dict[str, Any]]] = []

    def apply_chat_template(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> dict[str, list[int]]:
        self.template_messages.append(deepcopy(messages))
        return {"input_ids": [10, 11, 12]}

    def decode(self, token_ids: list[int]) -> str:
        return ""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]


class FakeEngine:
    def __init__(self):
        self.requests: list[ModelRequest] = []

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        self.requests.append(req)
        return ModelResponse(
            input_tokens=req.input_ids,
            output_tokens=[42, 0],
            output_logprobs=[0.0, 0.0],
            output_versions=[0, 0],
            stop_reason="stop",
            tokenizer=req.tokenizer,
        )


def _assistant_tool_message(arguments: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": arguments,
                },
            }
        ],
    }


def test_messages_for_hf_chat_template_decodes_tool_call_arguments_copy():
    messages = [
        {"role": "user", "content": "look this up"},
        _assistant_tool_message(json.dumps({"query": "AReaL", "limit": 2})),
        {"role": "tool", "tool_call_id": "call_123", "content": "done"},
    ]
    original = deepcopy(messages)

    rendered = _messages_for_hf_chat_template(messages)

    assert rendered[1]["tool_calls"][0]["function"]["arguments"] == {
        "query": "AReaL",
        "limit": 2,
    }
    assert messages == original


@pytest.mark.parametrize(
    ("raw_arguments", "rendered_arguments"),
    [
        ("not-json", {"arguments": "not-json"}),
        (json.dumps(["a", "b"]), {"arguments": ["a", "b"]}),
    ],
)
def test_messages_for_hf_chat_template_keeps_arguments_mapping_shaped(
    raw_arguments: str, rendered_arguments: dict[str, Any]
):
    rendered = _messages_for_hf_chat_template([_assistant_tool_message(raw_arguments)])

    assert rendered[0]["tool_calls"][0]["function"]["arguments"] == rendered_arguments


@pytest.mark.asyncio
async def test_chat_completion_templates_decoded_copy_but_caches_openai_messages():
    tokenizer = RecordingTokenizer()
    engine = FakeEngine()
    client = ArealOpenAI(
        engine=engine,
        tokenizer=tokenizer,
        api_key="test",
        base_url="http://localhost",
    )
    messages = [
        {"role": "user", "content": "look this up"},
        _assistant_tool_message(json.dumps({"query": "AReaL"})),
        {"role": "tool", "tool_call_id": "call_123", "content": "done"},
        {"role": "user", "content": "summarize it"},
    ]
    original = deepcopy(messages)

    completion = await client.chat.completions.create(
        messages=messages,
        max_completion_tokens=1,
    )

    rendered_arguments = tokenizer.template_messages[0][1]["tool_calls"][0]["function"][
        "arguments"
    ]
    cached_arguments = client.get_interaction(completion.id).messages[1]["tool_calls"][
        0
    ]["function"]["arguments"]

    assert rendered_arguments == {"query": "AReaL"}
    assert cached_arguments == json.dumps({"query": "AReaL"})
    assert messages == original


def test_concat_prompt_templates_decoded_copy(monkeypatch: pytest.MonkeyPatch):
    tokenizer = RecordingTokenizer()
    parent = InteractionWithTokenLogpReward(
        messages=[{"role": "user", "content": "look this up"}],
        model_response=ModelResponse(
            input_tokens=[10],
            output_tokens=[11, tokenizer.eos_token_id],
            output_logprobs=[0.0, 0.0],
            output_versions=[0, 0],
            stop_reason="stop",
            tokenizer=tokenizer,
        ),
        output_message_list=[_assistant_tool_message(json.dumps({"query": "AReaL"}))],
        chat_template_type="concat",
    )
    captured_messages: list[dict[str, Any]] | None = None

    def fake_apply_chat_template(
        tokenizer: RecordingTokenizer,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[int]:
        nonlocal captured_messages
        captured_messages = deepcopy(messages)
        return [99, tokenizer.eos_token_id, 100]

    monkeypatch.setattr(client_module, "apply_chat_template", fake_apply_chat_template)

    concat_prompt_token_ids_with_parent(
        message_list=[{"role": "tool", "tool_call_id": "call_123", "content": "done"}],
        parent=parent,
        tokenizer=tokenizer,
    )

    assert captured_messages is not None
    assert captured_messages[1]["tool_calls"][0]["function"]["arguments"] == {
        "query": "AReaL"
    }
    assert parent.output_message_list[0]["tool_calls"][0]["function"][
        "arguments"
    ] == json.dumps({"query": "AReaL"})

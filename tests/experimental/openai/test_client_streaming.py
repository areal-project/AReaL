# SPDX-License-Identifier: Apache-2.0

import pytest
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

import areal.experimental.openai.client as client_module
from areal.api import ModelResponse
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.client import AsyncCompletionsWithReward


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def decode(self, tokens):
        return "ok"


class _RecordingEngine:
    def __init__(self):
        self.request = None

    async def agenerate(self, req):
        self.request = req
        return ModelResponse(
            input_tokens=req.input_ids,
            output_tokens=[1],
            output_logprobs=[0.0],
            output_versions=[0],
            stop_reason="length",
        )


@pytest.mark.asyncio
async def test_create_stream_splits_tool_call_name_and_arguments():
    """Streaming tool calls should emit name/id before argument deltas."""
    tool_call = ChatCompletionMessageFunctionToolCall(
        id="call_123",
        type="function",
        function=Function(
            name="Bash",
            arguments='{"command": "ls -la /testbed"}',
        ),
    )
    response = ModelResponse(
        input_tokens=[1, 2, 3],
        output_tokens=[4, 5],
        output_logprobs=[0.0, 0.0],
        output_versions=[0, 0],
        stop_reason="tool_calls",
    )

    stream = AsyncCompletionsWithReward._create_stream(
        object(),
        completion_id="chatcmpl-test",
        current_time=123,
        output_text="",
        tool_calls=[tool_call],
        response=response,
    )
    chunks = [chunk async for chunk in stream]

    tool_chunks = [
        chunk
        for chunk in chunks
        if chunk.choices[0].delta.tool_calls is not None
        and len(chunk.choices[0].delta.tool_calls) > 0
    ]
    assert len(tool_chunks) == 2

    first_delta = tool_chunks[0].choices[0].delta.tool_calls[0]
    assert first_delta.index == 0
    assert first_delta.id == "call_123"
    assert first_delta.type == "function"
    assert first_delta.function is not None
    assert first_delta.function.name == "Bash"
    assert first_delta.function.arguments == ""

    second_delta = tool_chunks[1].choices[0].delta.tool_calls[0]
    assert second_delta.index == 0
    assert second_delta.id is None
    assert second_delta.type is None
    assert second_delta.function is not None
    assert second_delta.function.name is None
    assert second_delta.function.arguments == '{"command": "ls -la /testbed"}'

    assert chunks[-1].choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_chat_create_sets_generation_max_tokens_from_engine_limit(monkeypatch):
    prompt_tokens = list(range(33_000))
    monkeypatch.setattr(
        client_module,
        "apply_chat_template",
        lambda *args, **kwargs: prompt_tokens,
    )
    engine = _RecordingEngine()
    client = AsyncCompletionsWithReward.__new__(AsyncCompletionsWithReward)
    client.engine = engine
    client.tokenizer = _FakeTokenizer()
    client.tool_call_parser = "qwen3"
    client.reasoning_parser = "qwen3"
    client._cache = InteractionCache()
    client.engine_max_tokens = 131_071
    client.chat_template_type = "hf"
    client.lora_name = ""

    await client.create(
        messages=[{"role": "user", "content": "hello"}],
        max_completion_tokens=32_000,
        store=False,
    )

    assert engine.request is not None
    assert engine.request.gconfig.max_new_tokens == 32_000
    assert engine.request.gconfig.max_tokens == 131_071

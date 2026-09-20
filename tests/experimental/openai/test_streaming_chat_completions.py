"""Tests for streaming and non-streaming chat-completions behaviour.

The proxy rollout server's ``chat_completions`` handler must correctly return a
``StreamingResponse`` (SSE) when ``stream=True`` is requested, and a plain JSON
``ChatCompletion`` otherwise.

Ref: https://github.com/areal-project/AReaL/issues/1046
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_usage import CompletionUsage
from starlette.responses import StreamingResponse

from areal.api import ModelResponse
from areal.experimental.openai.client import ArealOpenAI, AsyncCompletionsWithReward
from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.types import InteractionWithTokenLogpReward

# ---------------------------------------------------------------------------
# Helpers (same pattern as test_proxy_rollout_server.py)
# ---------------------------------------------------------------------------

_ADMIN_KEY = "test-admin-key"

httpx = pytest.importorskip("httpx")

_transport = httpx.ASGITransport(app=srv.app)


def _client():
    return httpx.AsyncClient(transport=_transport, base_url="http://testserver")


def _admin_headers():
    return {"Authorization": f"Bearer {_ADMIN_KEY}"}


def _session_headers(api_key: str):
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


@pytest.fixture
def qwen_reasoning_parser(monkeypatch):
    """Model the pinned detector's handling of a prefilled, unclosed think block."""

    class ReasoningParser:
        def __init__(self, *args, force_reasoning=None, **kwargs):
            self.force_reasoning = force_reasoning
            self.detector = SimpleNamespace(think_start_token="<think>")

        def parse_non_stream(self, text):
            if text.startswith("<think>") or self.force_reasoning:
                text = text.removeprefix("<think>")
                if "</think>" in text:
                    return text.split("</think>", 1)
                return text, ""
            return "", text

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.parser.reasoning_parser",
        SimpleNamespace(ReasoningParser=ReasoningParser),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "text,force_reasoning,expected_reasoning,expected_content,has_tool",
    [
        ("<think>thinking</think>answer", False, "thinking", "answer", False),
        ("thinking</think>answer", True, "thinking", "answer", False),
        ("unfinished reasoning", True, "unfinished reasoning", "", False),
        ("plain answer", False, "", "plain answer", False),
        (
            "inspect first<tool_call><function=read><parameter=path>README.md"
            "</parameter></function></tool_call>",
            True,
            "inspect first",
            "",
            True,
        ),
        (
            "<think>thinking</think><tool_call><function=read>"
            "<parameter=path>README.md</parameter></function></tool_call>",
            False,
            "thinking",
            "",
            True,
        ),
    ],
)
async def test_create_preserves_reasoning_tools_and_cache(
    qwen_reasoning_parser,
    stream,
    text,
    force_reasoning,
    expected_reasoning,
    expected_content,
    has_tool,
):
    """Exercise create, real XML tool parsing, and cache before stream consumption."""
    tokenizer = MagicMock(eos_token_id=2, pad_token_id=0)
    tokenizer.apply_chat_template.return_value = {"input_ids": [1]}
    tokenizer.decode.return_value = text

    class Engine:
        async def agenerate(self, request):
            return ModelResponse(
                input_tokens=request.input_ids,
                output_tokens=[3, 2],
                output_logprobs=[-0.1, -0.1],
                output_versions=[0, 0],
                stop_reason="stop",
                tokenizer=request.tokenizer,
            )

    client = ArealOpenAI(
        engine=Engine(),
        tokenizer=tokenizer,
        api_key="test",
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]
    try:
        result = await client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "inspect"}],
            stream=stream,
            tools=tools if has_tool else [],
            max_tokens=32,
            extra_body={"chat_template_kwargs": {"enable_thinking": force_reasoning}},
        )
        interaction = next(iter(client._cache.values()))
        message = interaction.completion.choices[0].message
        assert message.content == expected_content
        assert getattr(message, "reasoning_content", "") == expected_reasoning
        assert interaction.output_message_list == [
            message.model_dump(exclude_none=True)
        ]
        assert bool(message.tool_calls) is has_tool
        if has_tool:
            assert message.tool_calls[0].function.name == "read"
            assert json.loads(message.tool_calls[0].function.arguments) == {
                "path": "README.md"
            }
            assert interaction.completion.choices[0].finish_reason == "tool_calls"
        if stream:
            chunks = [chunk async for chunk in result]
            assert (
                "".join(
                    getattr(c.choices[0].delta, "reasoning_content", "") or ""
                    for c in chunks
                )
                == expected_reasoning
            )
            assert (
                "".join(c.choices[0].delta.content or "" for c in chunks)
                == expected_content
            )
            assert (
                chunks[-1].choices[0].finish_reason
                == interaction.completion.choices[0].finish_reason
            )
            tool_deltas = [
                tc for c in chunks for tc in c.choices[0].delta.tool_calls or []
            ]
            if has_tool:
                assert tool_deltas[0].id == message.tool_calls[0].id
                assert (
                    "".join(tc.function.arguments or "" for tc in tool_deltas)
                    == message.tool_calls[0].function.arguments
                )
            else:
                assert not tool_deltas
        else:
            assert result == interaction.completion

        followup = await client.chat.completions.create(
            model="test-model",
            messages=[
                {"role": "user", "content": "inspect"},
                message.model_dump(exclude_none=True),
                {"role": "user", "content": "continue"},
            ],
            max_tokens=32,
            extra_body={"chat_template_kwargs": {"enable_thinking": force_reasoning}},
        )
        assert client.get_interaction(followup.id).parent is interaction
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_areal_stream_emits_reasoning_before_content():
    client = object.__new__(AsyncCompletionsWithReward)
    response = ModelResponse(input_tokens=[1], output_tokens=[2], stop_reason="stop")
    chunks = [
        chunk
        async for chunk in client._create_stream(
            completion_id="chatcmpl-test",
            current_time=0,
            model="test-model",
            reasoning_text="thinking",
            output_text="answer",
            tool_calls=None,
            response=response,
        )
    ]
    deltas = [chunk.choices[0].delta.model_dump(exclude_none=True) for chunk in chunks]
    assert deltas[1:3] == [
        {"reasoning_content": "thinking"},
        {"content": "answer"},
    ]


@pytest.mark.asyncio
async def test_areal_stream_reasoning_only_is_not_empty():
    client = object.__new__(AsyncCompletionsWithReward)
    response = ModelResponse(input_tokens=[1], output_tokens=[2], stop_reason="length")
    chunks = [
        chunk
        async for chunk in client._create_stream(
            completion_id="chatcmpl-test",
            current_time=0,
            model="test-model",
            reasoning_text="unfinished reasoning",
            output_text="",
            tool_calls=None,
            response=response,
        )
    ]
    assert chunks[1].choices[0].delta.model_dump(exclude_none=True) == {
        "reasoning_content": "unfinished reasoning"
    }


@pytest.fixture(autouse=True)
def _reset_server_globals(monkeypatch):
    """Reset all module-level globals before each test."""
    monkeypatch.setattr(srv, "_session_cache", {})
    monkeypatch.setattr(srv, "_api_key_to_session", {})
    monkeypatch.setattr(srv, "_session_to_api_key", {})
    monkeypatch.setattr(srv, "_capacity", 0)
    monkeypatch.setattr(srv, "_admin_api_key", _ADMIN_KEY)
    monkeypatch.setattr(srv, "_lock", threading.Lock())
    monkeypatch.setattr(srv, "_last_cleanup_time", 0.0)


# ---------------------------------------------------------------------------
# Fake create function (replaces _openai_client.chat.completions.create)
# ---------------------------------------------------------------------------


async def _fake_create(
    *,
    messages=None,
    model=None,
    stream=None,
    temperature=None,
    top_p=None,
    areal_cache=None,
    **kwargs,
):
    """Minimal stand-in for ``AsyncCompletionsWithReward.create``.

    Returns an ``AsyncGenerator[ChatCompletionChunk]`` when *stream* is truthy,
    or a ``ChatCompletion`` otherwise — mirroring the real client's behaviour.

    The explicit keyword parameters (``messages``, ``stream``, etc.) are
    required so that ``_call_client_create`` keeps them after its
    ``inspect.signature``-based filtering of request-body fields.
    """
    if stream:
        if areal_cache is not None:
            areal_cache["chatcmpl-test"] = InteractionWithTokenLogpReward(
                messages=list(messages or [])
            )

        async def _gen():
            yield ChatCompletionChunk(
                id="chatcmpl-test",
                choices=[
                    ChunkChoice(
                        delta=ChoiceDelta(role="assistant", content="hello"),
                        index=0,
                        finish_reason=None,
                    )
                ],
                created=0,
                model=model,
                object="chat.completion.chunk",
            )

        return _gen()

    return ChatCompletion(
        id="chatcmpl-test",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hello"),
            )
        ],
        created=0,
        model=model,
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


@pytest.fixture()
def _mock_openai_client(monkeypatch):
    """Inject a fake OpenAI client so no real inference engine is needed."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = _fake_create
    monkeypatch.setattr(srv, "_openai_client", mock_client)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
@pytest.mark.parametrize("stream", [False, True])
async def test_message_preprocessor_runs_once_per_request(monkeypatch, path, stream):
    """Non-idempotent preprocessors must not duplicate injected prompt content."""
    injected_message = {"role": "system", "content": "Injected instruction"}
    preprocessor = MagicMock(side_effect=lambda messages: [*messages, injected_message])
    captured_messages = []

    async def create(
        *, messages, model, stream=False, temperature=None, top_p=None, areal_cache=None
    ):
        captured_messages.append(messages)
        return await _fake_create(
            messages=messages, model=model, stream=stream, areal_cache=areal_cache
        )

    mock_client = MagicMock()
    mock_client.chat.completions.create = create
    monkeypatch.setattr(srv, "_openai_client", mock_client)
    monkeypatch.setattr(srv, "_message_preprocessors", [preprocessor])
    monkeypatch.setattr(srv, "_capacity", 1)

    async with _client() as client:
        start = await client.post(
            "/rl/start_session",
            headers=_admin_headers(),
            json={"task_id": "preprocess"},
        )
        response = await client.post(
            path,
            headers=_session_headers(start.json()["api_key"]),
            json={
                "model": "claude-compatible",
                "max_tokens": 16,
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hello"}]}
                ],
                "stream": stream,
            },
        )

    preprocessor.assert_called_once()
    assert captured_messages == [
        [{"role": "user", "content": "hello"}, injected_message]
    ]
    assert response.status_code == 200
    assert "hello" in response.text


class TestChatCompletionsEndpoint:
    """Verify both supported chat-completions paths."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["/chat/completions", "/v1/chat/completions"])
    async def test_streaming_returns_sse_response(
        self, monkeypatch, _mock_openai_client, path
    ):
        """``stream=True`` returns an SSE stream (``text/event-stream``)."""
        monkeypatch.setattr(srv, "_capacity", 1)

        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            api_key = resp.json()["api_key"]

            resp = await client.post(
                path,
                headers=_session_headers(api_key),
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "model": "test",
                    "stream": True,
                },
            )

            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            # Parse SSE body: data lines separated by blank lines
            events = [
                line
                for line in resp.text.strip().split("\n\n")
                if line.startswith("data: ")
            ]
            assert len(events) >= 3  # prelude + generated chunk + [DONE]
            assert events[-1] == "data: [DONE]"

            chunks = [json.loads(event.removeprefix("data: ")) for event in events[:-1]]
            assert chunks[0]["object"] == "chat.completion.chunk"
            assert chunks[0]["model"] == "test"
            assert chunks[0]["choices"][0]["delta"] == {
                "role": "assistant",
                "content": "",
            }
            assert chunks[1]["choices"][0]["delta"]["content"] == "hello"
            assert {chunk["id"] for chunk in chunks} == {chunks[0]["id"]}
            assert chunks[0]["id"] != "chatcmpl-test"

            resp = await client.post(
                "/rl/set_reward",
                headers=_session_headers(api_key),
                json={"interaction_id": chunks[0]["id"], "reward": 0.75},
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_streaming_starts_and_heartbeats_before_generation_finishes(
        self, monkeypatch, _mock_openai_client
    ):
        """The proxy must not wait for simulated streaming generation to finish."""
        generation_started = asyncio.Event()
        release_generation = asyncio.Event()

        async def slow_create(
            *,
            messages=None,
            model=None,
            stream=None,
            temperature=None,
            top_p=None,
            areal_cache=None,
            **kwargs,
        ):
            generation_started.set()
            await release_generation.wait()
            return await _fake_create(
                messages=messages,
                model=model,
                stream=stream,
                temperature=temperature,
                top_p=top_p,
                areal_cache=areal_cache,
                **kwargs,
            )

        srv._openai_client.chat.completions.create = slow_create
        monkeypatch.setattr(srv, "_STREAM_HEARTBEAT_INTERVAL_SECONDS", 0.01)
        srv._session_cache["slow-session"] = srv.SessionData(session_id="slow-session")

        response = await asyncio.wait_for(
            srv.chat_completions(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "model": "test",
                    "stream": True,
                },
                session_id="slow-session",
            ),
            timeout=0.1,
        )

        assert isinstance(response, StreamingResponse)
        iterator = response.body_iterator
        first = await asyncio.wait_for(anext(iterator), timeout=0.1)
        prelude = json.loads(first.removeprefix("data: "))
        assert prelude["choices"][0]["delta"]["content"] == ""

        next_event = asyncio.create_task(anext(iterator))
        await asyncio.wait_for(generation_started.wait(), timeout=0.1)
        heartbeat = await asyncio.wait_for(next_event, timeout=0.1)
        assert heartbeat == ": ping\n\n"

        release_generation.set()
        remaining = [chunk async for chunk in iterator]
        assert remaining[-1] == "data: [DONE]\n\n"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["/chat/completions", "/v1/chat/completions"])
    async def test_non_streaming_returns_json(
        self, monkeypatch, _mock_openai_client, path
    ):
        """Without ``stream``, returns a JSON ``ChatCompletion``."""
        monkeypatch.setattr(srv, "_capacity", 1)

        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            api_key = resp.json()["api_key"]

            resp = await client.post(
                path,
                headers=_session_headers(api_key),
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "model": "test",
                },
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "chat.completion"
            assert data["model"] == "test"
            assert data["choices"][0]["message"]["content"] == "hello"


@pytest.mark.asyncio
async def test_areal_completion_response_uses_request_model():
    """Both response modes retain the model required by Anthropic adapters."""
    client = object.__new__(AsyncCompletionsWithReward)
    model_response = ModelResponse(
        input_tokens=[1, 2],
        output_tokens=[3],
        stop_reason="stop",
    )

    completion, _ = client._build_chat_completion(
        completion_id="chatcmpl-test",
        current_time=0,
        model="claude-test-model",
        output_text="hello",
        tool_calls=None,
        response=model_response,
    )
    stream = client._create_stream(
        completion_id="chatcmpl-test",
        current_time=0,
        model="claude-test-model",
        output_text="hello",
        tool_calls=None,
        response=model_response,
    )
    chunks = [chunk async for chunk in stream]

    assert completion.model == "claude-test-model"
    assert chunks
    assert all(chunk.model == "claude-test-model" for chunk in chunks)

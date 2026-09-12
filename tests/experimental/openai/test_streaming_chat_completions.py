"""Tests for streaming and non-streaming chat-completions behaviour.

The proxy rollout server's ``chat_completions`` handler must correctly return a
``StreamingResponse`` (SSE) when ``stream=True`` is requested, and a plain JSON
``ChatCompletion`` otherwise.

Ref: https://github.com/areal-project/AReaL/issues/1046
"""

from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from starlette.responses import StreamingResponse

from areal.api import ModelResponse
from areal.experimental.openai.client import AsyncCompletionsWithReward
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

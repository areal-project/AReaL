"""Tests for streaming and non-streaming behaviour of the ``/chat/completions`` endpoint.

The proxy rollout server's ``chat_completions`` handler must correctly return a
``StreamingResponse`` (SSE) when ``stream=True`` is requested, and a plain JSON
``ChatCompletion`` otherwise.

Ref: https://github.com/areal-project/AReaL/issues/1046
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from areal.experimental.openai.proxy import proxy_rollout_server as srv

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
                model="test",
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
        model="test",
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
    """Verify streaming and non-streaming responses from ``/chat/completions``."""

    @pytest.mark.asyncio
    async def test_streaming_returns_sse_response(
        self, monkeypatch, _mock_openai_client
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
                "/chat/completions",
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
            assert len(events) >= 2  # at least one chunk + [DONE]
            assert events[-1] == "data: [DONE]"

            # Verify the data payload is a valid ChatCompletionChunk
            chunk = json.loads(events[0].removeprefix("data: "))
            assert chunk["object"] == "chat.completion.chunk"
            assert chunk["choices"][0]["delta"]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_non_streaming_returns_json(self, monkeypatch, _mock_openai_client):
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
                "/chat/completions",
                headers=_session_headers(api_key),
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "model": "test",
                },
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "chat.completion"
            assert data["choices"][0]["message"]["content"] == "hello"

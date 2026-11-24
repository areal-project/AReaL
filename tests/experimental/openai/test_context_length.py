# SPDX-License-Identifier: Apache-2.0

"""Context exhaustion is terminal and must not pollute captured trajectories."""

import asyncio
import json
import threading

import httpx
import pytest
import pytest_asyncio

from tests.experimental.openai.vision_stubs import StubTokenizer

from areal.api import ModelRequest, ModelResponse
from areal.experimental.openai.client import ArealOpenAI, ContextLengthExceededError
from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import SessionData
from areal.utils.hf_utils import apply_chat_template

_MESSAGES = [{"role": "user", "content": "private prompt text"}]


class _RecordingEngine:
    def __init__(self, tokenizer: StubTokenizer) -> None:
        self.tokenizer = tokenizer
        self.requests: list[ModelRequest] = []

    async def agenerate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            input_tokens=list(request.input_ids),
            output_tokens=[101],
            output_logprobs=[-0.25],
            output_versions=[7],
            stop_reason="length",
            tokenizer=self.tokenizer,
        )


@pytest.fixture
def client_and_engine():
    """Use the production client, tokenizer path and cache without model downloads."""
    tokenizer = StubTokenizer()
    engine = _RecordingEngine(tokenizer)
    client = ArealOpenAI(
        engine=engine, tokenizer=tokenizer, chat_template_type="hf", api_key="test"
    )
    return client, engine


def _prompt_length(client: ArealOpenAI, messages: list[dict[str, str]]) -> int:
    return len(
        apply_chat_template(
            client.chat.completions.tokenizer,
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
    )


@pytest.fixture
def proxy(monkeypatch, client_and_engine):
    """Use real HTTP validation, client tokenization and session storage."""
    client, engine = client_and_engine
    session = SessionData(session_id="episode")
    monkeypatch.setattr(srv, "_openai_client", client)
    monkeypatch.setattr(srv, "_session_cache", {"episode": session})
    monkeypatch.setattr(srv, "_api_key_to_session", {"episode-key": "episode"})
    monkeypatch.setattr(srv, "_session_to_api_key", {"episode": "episode-key"})
    monkeypatch.setattr(srv, "_lock", threading.Lock())
    return client, engine, session


@pytest_asyncio.fixture
async def paused_chat(proxy, monkeypatch):
    client, engine, session = proxy
    signal = asyncio.Event()
    monkeypatch.setattr(srv, "_resume_streams", signal)
    response = await srv.chat_completions(
        {"model": "policy", "messages": _MESSAGES, "stream": True},
        session_id=session.session_id,
    )
    events = response.body_iterator
    first = json.loads((await anext(events)).removeprefix("data: "))
    assert engine.requests == []
    assert len(session.completions) == 0

    def resume():
        monkeypatch.setattr(srv, "_resume_streams", None)
        signal.set()

    try:
        yield events, first["id"], resume
    finally:
        await events.aclose()
        await client.close()


@pytest.mark.asyncio
async def test_paused_chat_uses_public_completion_id_for_reward(paused_chat, proxy):
    events, public_id, resume = paused_chat
    _, engine, session = proxy
    resume()
    data = [
        event.removeprefix("data: ").strip()
        async for event in events
        if event.startswith("data: ")
    ]
    assert data[-1] == "[DONE]"
    assert {json.loads(chunk)["id"] for chunk in data[:-1]} == {public_id}
    assert len(engine.requests) == 1
    srv.set_reward(
        srv.SetRewardRequest(interaction_id=public_id, reward=0.625), session.session_id
    )
    assert session.completions[public_id].reward == 0.625


@pytest.mark.asyncio
async def test_paused_chat_reports_context_error_in_stream(paused_chat, proxy):
    events, _, resume = paused_chat
    client, engine, session = proxy
    client.chat.completions.engine_max_tokens = 2
    resume()
    (error,) = [
        event.removeprefix("data: ")
        async for event in events
        if event.startswith("data: ")
    ]
    assert json.loads(error)["error"]["code"] == "context_length_exceeded"
    assert _MESSAGES[0]["content"] not in error
    assert engine.requests == []
    assert len(session.completions) == 0


@pytest.mark.asyncio
async def test_paused_chat_does_not_generate_for_reused_session(
    paused_chat, proxy, monkeypatch
):
    events, _, resume = paused_chat
    _, engine, session = proxy
    replacement = SessionData(session.session_id)
    monkeypatch.setitem(srv._session_cache, session.session_id, replacement)
    resume()
    (error,) = [
        event.removeprefix("data: ")
        async for event in events
        if event.startswith("data: ")
    ]
    assert "error" in json.loads(error)
    assert engine.requests == []
    assert not replacement.completions


@pytest.mark.asyncio
async def test_disconnect_during_resumed_generation_removes_partial_interaction(
    paused_chat, proxy, monkeypatch
):
    events, public_id, resume = paused_chat
    _, engine, session = proxy
    started = asyncio.Event()

    async def wait_for_cancellation(request):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(engine, "agenerate", wait_for_cancellation)
    resume()
    next_chunk = asyncio.create_task(anext(events))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        assert public_id in session.completions
    finally:
        next_chunk.cancel()
        with pytest.raises(asyncio.CancelledError):
            await next_chunk
    assert public_id not in session.completions


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("excess_tokens", [0, 1])
@pytest.mark.asyncio
async def test_chat_http_context_error_returns_safe_openai_400(
    proxy, stream, excess_tokens
):
    """Both streaming setup and regular calls expose a non-retryable input error."""
    client, engine, session = proxy
    prompt_tokens = _prompt_length(client, _MESSAGES)
    max_tokens = prompt_tokens - excess_tokens
    client.chat.completions.engine_max_tokens = max_tokens
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=srv.app), base_url="http://proxy"
        ) as http:
            response = await http.post(
                "/chat/completions",
                headers={"Authorization": "Bearer episode-key"},
                json={"model": "policy", "messages": _MESSAGES, "stream": stream},
            )
        assert response.status_code == 400
        assert response.json() == {
            "error": {
                "message": str(ContextLengthExceededError(prompt_tokens, max_tokens)),
                "type": "invalid_request_error",
                "param": "messages",
                "code": "context_length_exceeded",
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
            }
        }
        assert _MESSAGES[0]["content"] not in response.text
        assert "episode-key" not in response.text
        assert engine.requests == []
        assert session.completions == {}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_non_context_value_error_remains_server_error(proxy):
    """An unrelated ValueError must not become a context termination signal."""
    client, engine, _ = proxy
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=srv.app), base_url="http://proxy"
        ) as http:
            response = await http.post(
                "/chat/completions",
                headers={"Authorization": "Bearer episode-key"},
                json={"model": "policy", "messages": []},
            )
        assert response.status_code == 500
        assert response.json() == {"detail": "messages cannot be empty"}
        assert engine.requests == []
    finally:
        await client.close()

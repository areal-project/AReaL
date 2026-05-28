"""Unit tests for data proxy pause/resume (Plan 3c)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from areal.experimental.inference_service.data_proxy.app import create_app
from areal.experimental.inference_service.data_proxy.config import DataProxyConfig
from areal.experimental.inference_service.data_proxy.pause import PauseState
from areal.experimental.inference_service.data_proxy.session import SessionStore

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    return DataProxyConfig(
        host="127.0.0.1",
        port=18083,
        backend_addr="http://mock-sglang:30000",
        tokenizer_path="mock-tokenizer",
        request_timeout=10.0,
        max_resubmit_retries=5,
        resubmit_wait=0.01,  # fast for tests
    )


@pytest.fixture
def mock_tokenizer():
    tok = MagicMock()
    tok.tokenize = AsyncMock(return_value=[101, 102, 103])
    tok.decode_token = MagicMock(side_effect=lambda tid: f"tok_{tid}")
    tok.decode_tokens = MagicMock(return_value="hello world")
    tok.apply_chat_template = AsyncMock(return_value=[100, 200, 300])
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    tok._tok = MagicMock()
    tok._tok.eos_token_id = 2
    tok._tok.pad_token_id = 0
    return tok


@pytest.fixture
def mock_areal_client():
    """Mock ArealOpenAI client that returns a valid ChatCompletion."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    mock_client = MagicMock()

    async def _mock_create(*, areal_cache=None, **kwargs):
        import torch

        completion = ChatCompletion(
            id="chatcmpl-mock",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(
                        content="mocked response", role="assistant"
                    ),
                )
            ],
            created=1700000000,
            model="sglang",
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=3, prompt_tokens=5, total_tokens=8),
        )
        if areal_cache is not None:
            from areal.experimental.openai.types import (
                InteractionWithTokenLogpReward,
            )

            interaction = InteractionWithTokenLogpReward(
                messages=[{"role": "user", "content": "test"}],
            )
            interaction._cache = {
                "input_ids": torch.tensor([100, 200, 300]),
                "output_tokens": torch.tensor([1234, 5678]),
            }
            interaction.completion = completion
            cid = completion.id
            areal_cache[cid] = interaction
        return completion

    mock_client.chat.completions.create = AsyncMock(side_effect=_mock_create)
    return mock_client


@pytest_asyncio.fixture
async def app_client(config, mock_tokenizer, mock_areal_client):
    """Create an ASGI test client with all app.state attributes injected."""
    from areal.experimental.inference_service.inf_bridge import InfBridge
    from areal.experimental.inference_service.sglang.bridge import SGLangBridgeBackend

    app = create_app(config)

    pause_state = PauseState()
    inf_bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr=config.backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
        max_resubmit_retries=config.max_resubmit_retries,
        resubmit_wait=config.resubmit_wait,
    )

    # Mock pause/resume so they don't make real HTTP calls to SGLang.
    # side_effect must be an async function (not a lambda returning a coroutine)
    # so that AsyncMock awaits the state change before the endpoint returns.
    async def _mock_pause():
        await _set_paused(pause_state, True)

    async def _mock_resume():
        await _set_paused(pause_state, False)

    inf_bridge.pause = AsyncMock(side_effect=_mock_pause)
    inf_bridge.resume = AsyncMock(side_effect=_mock_resume)

    app.state.tokenizer = mock_tokenizer
    app.state.inf_bridge = inf_bridge
    app.state.areal_client = mock_areal_client
    app.state.pause_state = pause_state
    app.state.config = config
    app.state.session_store = SessionStore()
    app.state.version = 0

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, app, pause_state


async def _set_paused(pause_state: PauseState, value: bool) -> None:
    """Helper to set pause state in async mock side effects."""
    await pause_state.set_paused(value)


# =============================================================================
# PauseState unit tests
# =============================================================================


class TestPauseState:
    """Test PauseState flag transitions."""

    @pytest.mark.asyncio
    async def test_default_not_paused(self):
        state = PauseState()
        assert not await state.is_paused()

    @pytest.mark.asyncio
    async def test_set_paused_true(self):
        state = PauseState()
        await state.set_paused(True)
        assert await state.is_paused()

    @pytest.mark.asyncio
    async def test_set_paused_false(self):
        state = PauseState()
        await state.set_paused(True)
        assert await state.is_paused()
        await state.set_paused(False)
        assert not await state.is_paused()

    @pytest.mark.asyncio
    async def test_multiple_transitions(self):
        state = PauseState()
        for _ in range(3):
            await state.set_paused(True)
            assert await state.is_paused()
            await state.set_paused(False)
            assert not await state.is_paused()


# =============================================================================
# Endpoint tests — /pause_generation and /continue_generation
# =============================================================================


class TestPauseResumeEndpoints:
    """Test POST /pause_generation and POST /continue_generation endpoints."""

    @pytest.mark.asyncio
    async def test_health_includes_paused_false(self, app_client):
        client, app, pause_state = app_client
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "paused" in data
        assert data["paused"] is False

    @pytest.mark.asyncio
    async def test_pause_endpoint(self, app_client):
        client, app, pause_state = app_client

        resp = await client.post("/pause_generation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["paused"] is True

        # Verify inf_bridge.pause() was called
        app.state.inf_bridge.pause.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_endpoint(self, app_client):
        client, app, pause_state = app_client

        # First pause
        await pause_state.set_paused(True)

        resp = await client.post("/continue_generation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["paused"] is False

        # Verify inf_bridge.resume() was called
        app.state.inf_bridge.resume.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_paused_true_after_pause(self, app_client):
        client, app, pause_state = app_client

        await client.post("/pause_generation")

        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["paused"] is True

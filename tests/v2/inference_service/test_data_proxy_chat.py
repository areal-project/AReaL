"""Unit tests for data proxy chat/session endpoints (Plan 3b)."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from areal.utils.seeding import derive_deterministic_seed
from areal.v2.inference_service.data_proxy.app import (
    _flush_ready_trajectories,
    _remotize_trajectory,
    create_app,
)
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.session import (
    EpisodeIdentity,
    SessionData,
    SessionStore,
)

# =============================================================================
# Fixtures
# =============================================================================

ADMIN_KEY = "areal-admin-key"


@pytest.fixture
def config():
    return DataProxyConfig(
        host="127.0.0.1",
        port=18082,
        backend_addr="http://mock-sglang:30000",
        tokenizer_path="mock-tokenizer",
        request_timeout=10.0,
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
    # Expose underlying _tok for ModelResponse.output_tokens_without_stop
    tok._tok = MagicMock()
    tok._tok.eos_token_id = 2
    tok._tok.pad_token_id = 0
    return tok


@pytest.fixture
def mock_areal_client():
    """Mock ArealOpenAI client that returns a valid ChatCompletion.
    Also stores the interaction in the session's InteractionCache.

    The mock has `.chat.completions.create()` as an AsyncMock to match
    the ArealOpenAI interface used by the data proxy app.
    """
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    mock_client = MagicMock()

    call_index = 0

    async def _mock_create(*, areal_cache=None, **kwargs):
        """Mock create that stores the interaction in session cache via areal_cache."""
        import torch

        nonlocal call_index

        completion = ChatCompletion(
            id=f"chatcmpl-test{call_index}",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(content="Hello!", role="assistant"),
                )
            ],
            created=1234567890 + call_index,
            model="sglang",
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=3, prompt_tokens=5, total_tokens=8),
        )
        call_index += 1

        messages = kwargs.get("messages", [])

        interaction = InteractionWithTokenLogpReward(
            messages=messages if isinstance(messages, list) else list(messages),
            completion=completion,
            output_message_list=[{"role": "assistant", "content": "Hello!"}],
        )
        # Pre-populate _cache so to_tensor_dict() works without ModelResponse
        interaction._cache = {
            "input_ids": torch.tensor([[100, 200, 300, 1234, 5678, 2]]),
            "loss_mask": torch.tensor([[0, 0, 0, 1, 1, 1]]),
            "logprobs": torch.tensor([[0.0, 0.0, 0.0, -0.5, -0.3, -0.1]]),
            "versions": torch.tensor([[-1, -1, -1, 0, 0, 0]]),
            "attention_mask": torch.ones(6, dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor([0.0]),
        }
        if areal_cache is not None:
            areal_cache[completion.id] = interaction
        return completion

    mock_client.chat.completions.create = AsyncMock(side_effect=_mock_create)
    return mock_client


@pytest_asyncio.fixture
async def client(config, mock_tokenizer, mock_areal_client):
    """Create app with mocked deps and yield an httpx async client."""
    from areal.v2.inference_service.data_proxy.pause import PauseState
    from areal.v2.inference_service.inf_bridge import InfBridge
    from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

    app = create_app(config)
    # Bypass lifespan — inject mocks directly into app.state
    pause_state = PauseState()
    inf_bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr=config.backend_addr,
        pause_state=pause_state,
        request_timeout=config.request_timeout,
        max_resubmit_retries=5,
        resubmit_wait=0.01,
    )
    app.state.tokenizer = mock_tokenizer
    app.state.inf_bridge = inf_bridge
    app.state.areal_client = mock_areal_client
    app.state.pause_state = pause_state
    app.state.config = config
    store = SessionStore()
    store.set_admin_key(config.admin_api_key)
    app.state.session_store = store
    app.state.version = 0
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def session_headers(api_key: str):
    return {"Authorization": f"Bearer {api_key}"}


def anthropic_headers(api_key: str):
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }


# =============================================================================
# SessionStore unit tests
# =============================================================================


class TestSessionStore:
    def test_start_session_returns_ids(self):
        store = SessionStore()
        session_id, api_key = store.start_session("task-1")
        assert session_id == "task-1-0"
        assert isinstance(api_key, str)
        assert len(api_key) > 0

    def test_get_session_by_api_key(self):
        store = SessionStore()
        session_id, api_key = store.start_session("task-1")
        session = store.get_session_by_api_key(api_key)
        assert session is not None
        assert session.session_id == session_id

    def test_get_session_by_api_key_not_found(self):
        store = SessionStore()
        assert store.get_session_by_api_key("nonexistent") is None

    def test_set_reward_marks_trajectory_ready(self):
        store = SessionStore()
        session_id, _ = store.start_session("task-1")
        session = store.get_session(session_id)
        assert isinstance(session, SessionData)
        assert not session.has_ready_trajectories
        # Populate with a fake interaction so set_reward succeeds
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        interaction = InteractionWithTokenLogpReward(
            messages=[{"role": "user", "content": "hi"}]
        )
        interaction.interaction_id = "fake-id"
        interaction.output_message_list = [{"role": "assistant", "content": "hello"}]
        session.active_completions["fake-id"] = interaction
        result = session.set_reward(interaction_id="fake-id", reward=1.0)
        assert session.has_ready_trajectories
        assert result.ready_transition is True

    def test_set_reward_waits_for_timeout_before_ready(self):
        store = SessionStore(set_reward_finish_timeout=5.0)
        session_id, _ = store.start_session("task-1")
        session = store.get_session(session_id)
        assert isinstance(session, SessionData)

        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        interaction = InteractionWithTokenLogpReward(
            messages=[{"role": "user", "content": "hi"}]
        )
        interaction.interaction_id = "fake-id"
        interaction.output_message_list = [{"role": "assistant", "content": "hello"}]
        session.active_completions["fake-id"] = interaction

        pending = session.set_reward(interaction_id="fake-id", reward=1.0)
        assert pending.ready_transition is False
        assert pending.trajectory_id is None
        assert not session.has_ready_trajectories

        not_ready = session.finalize_if_reward_timeout_elapsed()
        assert not_ready is None

        ready = session.finalize_if_reward_timeout_elapsed(
            now=session._last_access_time + 6.0
        )
        assert ready is not None
        assert ready.ready_transition is True
        assert ready.trajectory_id == 0
        assert session.has_ready_trajectories

    def test_multiple_set_reward_calls_update_same_trajectory_before_timeout(self):
        session = SessionData("task-1", set_reward_finish_timeout=5.0)

        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        interaction = InteractionWithTokenLogpReward(
            messages=[{"role": "user", "content": "hi"}]
        )
        interaction.interaction_id = "fake-id"
        interaction.output_message_list = [{"role": "assistant", "content": "hello"}]
        session.active_completions["fake-id"] = interaction

        first = session.set_reward(interaction_id="fake-id", reward=1.0)
        second = session.set_reward(interaction_id="fake-id", reward=2.0)

        assert first.ready_transition is False
        assert second.ready_transition is False
        assert second.trajectory_id is None

        ready = session.finalize_if_reward_timeout_elapsed(
            now=session._last_access_time + 6.0
        )
        assert ready is not None
        assert ready.trajectory_id == 0

        _, interactions = session.export_trajectory(discount=1.0, style="individual")
        assert interactions["fake-id"].reward == 2.0

    def test_finish_session_not_found(self):
        store = SessionStore()
        session = store.get_session("nonexistent")
        assert session is None

    def test_session_count(self):
        store = SessionStore()
        assert store.session_count == 0
        store.start_session("task-1")
        assert store.session_count == 1
        store.start_session("task-2")
        assert store.session_count == 2

    def test_remove_session(self):
        store = SessionStore()
        session_id, api_key = store.start_session("task-1")
        store.remove_session(session_id)
        assert store.get_session(session_id) is None
        assert store.get_session_by_api_key(api_key) is None

    def test_duplicate_session_ids_increment(self):
        store = SessionStore()
        sid1, _ = store.start_session("task-1")
        sid2, _ = store.start_session("task-1")
        assert sid1 == "task-1-0"
        assert sid2 == "task-1-1"

    def test_get_or_create_hitl_session_reuses_same_session(self):
        store = SessionStore()
        store.set_admin_key(ADMIN_KEY)
        online_session1 = store.get_or_create_hitl_session()
        online_session2 = store.get_or_create_hitl_session()
        assert online_session1 is online_session2
        assert online_session1.session_id == "__hitl__"

    def test_prefix_matcher_is_preserved_when_active_cache_rotates(self):
        from examples.swe.prefix_matchers import swe_prefix_matcher

        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        store = SessionStore(prefix_matcher=swe_prefix_matcher)
        session_id, _ = store.start_session("task-prefix")
        session = store.get_session(session_id)
        assert isinstance(session, SessionData)
        assert session.active_completions._prefix_matcher is swe_prefix_matcher

        interaction = InteractionWithTokenLogpReward(
            messages=[{"role": "user", "content": "hi"}],
        )
        interaction.interaction_id = "interaction-1"
        interaction.output_message_list = [{"role": "assistant", "content": "hello"}]
        session.active_completions["interaction-1"] = interaction

        session.set_reward("interaction-1", 1.0)

        assert session.active_completions._prefix_matcher is swe_prefix_matcher

    def test_online_session_count(self):
        store = SessionStore()
        store.set_admin_key(ADMIN_KEY)
        store.get_or_create_hitl_session()
        assert store.session_count == 1


def test_remotize_trajectory_keeps_reward_metadata_inline(monkeypatch):
    import torch

    rewards = torch.tensor([1.0])
    original_rewards = torch.tensor([3.0])
    captured = {}

    def _remotize(payload, node_addr):
        captured["payload"] = payload
        captured["node_addr"] = node_addr
        return {"input_ids": "remote-input-ids"}

    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.RTensor.remotize",
        _remotize,
    )

    result = _remotize_trajectory(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "rewards": rewards,
            "original_rewards": original_rewards,
        },
        node_addr="127.0.0.1:1234",
    )

    assert set(captured["payload"]) == {"input_ids"}
    assert captured["node_addr"] == "127.0.0.1:1234"
    assert result["input_ids"] == "remote-input-ids"
    assert result["rewards"] is rewards
    assert result["original_rewards"] is original_rewards


# =============================================================================
# Endpoint tests: /rl/start_session
# =============================================================================


@pytest.mark.asyncio
async def test_start_session_with_admin_key(client):
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "test-task"},
        headers=admin_headers(),
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "group_id" in data
    assert "sessions" in data
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["session_id"].startswith("test-task-")
    assert "session_api_key" in data["sessions"][0]


@pytest.mark.asyncio
async def test_start_session_without_admin_key(client):
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "test-task"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_anthropic_messages_applies_preprocessors_and_uses_session_cache(
    config,
    mock_areal_client,
):
    from examples.swe.prefix_matchers import swe_prefix_matcher

    cc_config = DataProxyConfig(
        **{
            **config.__dict__,
            "message_preprocessors": (
                "examples.swe.preprocessors.StripAnthropicBillingHeader",
                "examples.swe.preprocessors.StripAllSystemReminders",
            ),
            "prefix_matcher": "examples.swe.prefix_matchers.swe_prefix_matcher",
        }
    )
    app = create_app(cc_config)
    store = SessionStore(prefix_matcher=swe_prefix_matcher)
    store.set_admin_key(cc_config.admin_api_key)
    store.set_capacity(1)
    app.state.session_store = store
    app.state.areal_client = mock_areal_client

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        start = await c.post(
            "/rl/start_session",
            json={"task_id": "cc-task"},
            headers=admin_headers(),
        )
        session_key = start.json()["sessions"][0]["session_api_key"]
        response = await c.post(
            "/v1/messages",
            json={
                "model": "claude-compatible",
                "max_tokens": 64,
                "system": (
                    "x-anthropic-billing-header: volatile\n"
                    "stable<system-reminder>volatile</system-reminder>"
                ),
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ],
            },
            headers=anthropic_headers(session_key),
        )

    assert response.status_code == 200
    assert response.json()["type"] == "message"
    call_kwargs = mock_areal_client.chat.completions.create.await_args.kwargs
    messages = call_kwargs["messages"]
    serialized_messages = str(messages)
    assert "x-anthropic-billing-header" not in serialized_messages
    assert "system-reminder" not in serialized_messages
    assert "stable" in serialized_messages

    session = store.get_session_by_api_key(session_key)
    assert session is not None
    assert session.active_completions._prefix_matcher is swe_prefix_matcher
    assert len(session.active_completions) == 1


@pytest.mark.asyncio
async def test_anthropic_messages_streams_translated_events(
    config,
    mock_areal_client,
    monkeypatch,
):
    app = create_app(config)
    store = SessionStore()
    store.set_admin_key(config.admin_api_key)
    store.set_capacity(1)
    app.state.session_store = store
    app.state.areal_client = mock_areal_client

    async def _translated_stream():
        yield "event: message_start\ndata: {}\n\n"
        yield "event: message_stop\ndata: {}\n\n"

    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.translate_anthropic_stream",
        lambda _stream, model: _translated_stream(),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        start = await c.post(
            "/rl/start_session",
            json={"task_id": "cc-stream"},
            headers=admin_headers(),
        )
        session_key = start.json()["sessions"][0]["session_api_key"]
        response = await c.post(
            "/v1/messages",
            json={
                "model": "claude-compatible",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=anthropic_headers(session_key),
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "event: message_start" in response.text
    assert "event: message_stop" in response.text
    assert mock_areal_client.chat.completions.create.await_args.kwargs["stream"] is True


@pytest.mark.asyncio
async def test_start_session_wrong_admin_key(client):
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "test-task"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 403


# =============================================================================
# Endpoint tests: /chat/completions
# =============================================================================


@pytest.mark.asyncio
async def test_chat_completions_with_session_key(client, mock_areal_client):
    # Start session first
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "chat-test"},
        headers=admin_headers(),
    )
    api_key = resp.json()["sessions"][0]["session_api_key"]

    # Call chat/completions (OpenAI-compatible format)
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hello!"
    mock_areal_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_chat_completions_without_session_key(client):
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_completions_with_invalid_key(client):
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"Authorization": "Bearer fake-key"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_offline_chat_unknown_token_falls_through_to_standalone(client):
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(f"{ADMIN_KEY}:agent-online"),
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_chat_completions_passes_sampling_params(client, mock_areal_client):
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "sp-test"},
        headers=admin_headers(),
    )
    api_key = resp.json()["sessions"][0]["session_api_key"]

    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 100,
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200

    call_kwargs = mock_areal_client.chat.completions.create.call_args
    kw = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
    assert kw["temperature"] == 0.5
    assert kw["top_p"] == 0.9
    assert kw["max_tokens"] == 100
    assert "seed" not in kw


@pytest.mark.asyncio
async def test_chat_completions_deterministic_seed_distinguishes_group_sessions(
    client, config, mock_areal_client
):
    config.deterministic_sampling = True
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "seed-test", "group_size": 2},
        headers=admin_headers(),
    )
    sessions = resp.json()["sessions"]

    for session in sessions:
        resp = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=session_headers(session["session_api_key"]),
        )
        assert resp.status_code == 200

    seeds = [
        call.kwargs["seed"]
        for call in mock_areal_client.chat.completions.create.call_args_list
    ]
    assert seeds == [
        derive_deterministic_seed("seed-test:0", 0),
        derive_deterministic_seed("seed-test:1", 0),
    ]
    assert seeds[0] != seeds[1]


@pytest.mark.asyncio
async def test_chat_completions_preserves_explicit_seed(
    client, config, mock_areal_client
):
    config.deterministic_sampling = True
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "explicit-seed"},
        headers=admin_headers(),
    )
    api_key = resp.json()["sessions"][0]["session_api_key"]

    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
            "seed": 12345,
        },
        headers=session_headers(api_key),
    )

    assert resp.status_code == 200
    assert mock_areal_client.chat.completions.create.call_args.kwargs["seed"] == 12345

    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "again"}],
        },
        headers=session_headers(api_key),
    )

    assert resp.status_code == 200
    assert mock_areal_client.chat.completions.create.call_args.kwargs[
        "seed"
    ] == derive_deterministic_seed("explicit-seed", 1)


# =============================================================================
# Endpoint tests: /rl/set_reward
# =============================================================================


@pytest.mark.asyncio
async def test_set_reward_success(client):
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "reward-test"},
        headers=admin_headers(),
    )
    api_key = resp.json()["sessions"][0]["session_api_key"]

    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200

    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    assert resp.json()["message"] == "success"
    data = resp.json()
    assert data["interaction_count"] == 1
    assert data["session_id"].startswith("reward-test-")
    assert data["trajectory_id"] == 0
    assert data["trajectory_ready"] is True
    assert data["ready_transition"] is True


@pytest.mark.asyncio
async def test_set_reward_no_interactions(client):
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "reward-empty"},
        headers=admin_headers(),
    )
    api_key = resp.json()["sessions"][0]["session_api_key"]

    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 400
    assert "No interactions" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_set_reward_without_session_key(client):
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
    )
    assert resp.status_code == 401


# =============================================================================
# Endpoint tests: /rl/set_reward
# =============================================================================


@pytest.mark.asyncio
async def test_set_reward_auto_finishes(client):
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "end-test"},
        headers=admin_headers(),
    )
    api_key = resp.json()["sessions"][0]["session_api_key"]

    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200

    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 0.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "success"
    assert "interaction_count" in data
    assert data["session_id"].startswith("end-test-")
    assert data["trajectory_id"] == 0
    assert data["trajectory_ready"] is True
    assert data["ready_transition"] is True


@pytest.mark.asyncio
async def test_set_reward_only_once_allowed(client):
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "twice-test"},
        headers=admin_headers(),
    )
    api_key = resp.json()["sessions"][0]["session_api_key"]

    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200

    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    assert resp.json()["ready_transition"] is True

    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 2.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    assert resp.json()["ready_transition"] is False
    assert resp.json()["trajectory_ready"] is True


@pytest.mark.asyncio
async def test_set_reward_timeout_delays_readiness_and_direct_callback(
    client, monkeypatch
):
    app = client._transport.app
    app.state.config.set_reward_finish_timeout = 5.0
    app.state.config.callback_server_addr = "http://controller"
    app.state.session_store = SessionStore(set_reward_finish_timeout=5.0)
    app.state.session_store.set_admin_key(ADMIN_KEY)

    callback_calls = []

    async def _mock_post_callback(
        callback_server_addr, admin_api_key, notification, timeout, **kwargs
    ):
        callback_calls.append(
            (
                callback_server_addr,
                admin_api_key,
                notification.session_id,
                notification.trajectory_id,
            )
        )
        return True

    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app._post_online_ready_callback",
        _mock_post_callback,
    )

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "hi"}]},
        headers=session_headers(ADMIN_KEY),
    )

    reward_resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(ADMIN_KEY),
    )
    assert reward_resp.status_code == 200
    assert reward_resp.json()["ready_transition"] is False
    assert reward_resp.json()["trajectory_ready"] is False

    await _flush_ready_trajectories(app)
    assert callback_calls == []

    hitl_session = app.state.session_store.get_session("__hitl__")
    assert hitl_session is not None
    hitl_session.finalize_if_reward_timeout_elapsed(now=time.time() + 6.0)
    await _flush_ready_trajectories(app)

    assert callback_calls == [("http://controller", ADMIN_KEY, "__hitl__", 0)]


@pytest.mark.asyncio
async def test_set_reward_without_key(client):
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 0.0},
    )
    assert resp.status_code == 401


# =============================================================================
# HITL tests — admin key → single persistent session
# =============================================================================


@pytest.mark.asyncio
async def test_hitl_admin_key_creates_single_persistent_session(client):
    resp = await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "hi"}]},
        headers=session_headers(ADMIN_KEY),
    )
    assert resp.status_code == 200

    resp2 = await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "bye"}]},
        headers=session_headers(ADMIN_KEY),
    )
    assert resp2.status_code == 200

    health = await client.get("/health")
    assert health.json()["sessions"] == 1


@pytest.mark.asyncio
async def test_hitl_reuses_same_session_across_multiple_chat_requests(client):
    store: SessionStore = client._transport.app.state.session_store

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "a"}]},
        headers=session_headers(ADMIN_KEY),
    )
    session1 = store.get_session("__hitl__")
    assert session1 is not None

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "b"}]},
        headers=session_headers(ADMIN_KEY),
    )
    session2 = store.get_session("__hitl__")
    assert session1 is session2


@pytest.mark.asyncio
async def test_hitl_online_set_reward_separates_trajectories(client):
    token = ADMIN_KEY

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "t0"}]},
        headers=session_headers(token),
    )
    r0 = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(token),
    )
    assert r0.status_code == 200
    assert r0.json()["trajectory_id"] == 0

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "t1"}]},
        headers=session_headers(token),
    )
    r1 = await client.post(
        "/rl/set_reward",
        json={"reward": 2.0},
        headers=session_headers(token),
    )
    assert r1.status_code == 200
    assert r1.json()["trajectory_id"] == 1


@pytest.mark.asyncio
async def test_hitl_ready_transition_after_each_online_set_reward(client):
    token = ADMIN_KEY

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "hi"}]},
        headers=session_headers(token),
    )
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(token),
    )
    assert resp.json()["ready_transition"] is True
    assert resp.json()["trajectory_ready"] is True


# =============================================================================
# =============================================================================
@pytest.mark.asyncio
async def test_online_start_session_returns_generated_session_key(client):
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "batch-1"},
        headers=admin_headers(),
    )
    assert resp.status_code == 201
    data = resp.json()
    assert "group_id" in data
    assert "sessions" in data
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["session_id"].startswith("batch-1-")
    assert len(data["sessions"][0]["session_api_key"]) > 0


@pytest.mark.asyncio
async def test_batch_session_produces_single_trajectory(client):
    start = await client.post(
        "/rl/start_session",
        json={"task_id": "batch-traj"},
        headers=admin_headers(),
    )
    session_api_key = start.json()["sessions"][0]["session_api_key"]
    session_id = start.json()["sessions"][0]["session_id"]

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "hi"}]},
        headers=session_headers(session_api_key),
    )
    reward_resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(session_api_key),
    )
    assert reward_resp.status_code == 200
    assert reward_resp.json()["trajectory_id"] == 0
    assert reward_resp.json()["session_id"] == session_id
    assert reward_resp.json()["ready_transition"] is True


@pytest.mark.asyncio
async def test_batch_online_set_reward_completes_that_session(
    client, monkeypatch, tmp_path
):
    monkeypatch.setenv("AREAL_ROLLOUT_TRACE_DIR", str(tmp_path))
    start = await client.post(
        "/rl/start_session",
        json={"task_id": "batch-complete"},
        headers=admin_headers(),
    )
    session_api_key = start.json()["sessions"][0]["session_api_key"]
    session_id = start.json()["sessions"][0]["session_id"]

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "q"}]},
        headers=session_headers(session_api_key),
    )
    await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(session_api_key),
    )

    export_resp = await client.post(
        "/export_trajectories",
        json={
            "session_ids": [session_id],
            "trajectory_id": 0,
            "discount": 1.0,
            "style": "individual",
        },
        headers=admin_headers(),
    )
    assert export_resp.status_code == 200
    assert "traj" in export_resp.json()
    rows = [
        json.loads(line)
        for path in tmp_path.glob("exported-workloads-*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["sources"][0]["session_id"] == session_id
    assert row["sources"][0]["trajectory_id"] == 0
    assert row["interaction_ids"] == ["chatcmpl-test0"]
    assert row["workload"]["sequence_lengths"] == [6]
    assert row["workload"]["sequence_tokens"] == 6
    assert row["workload"]["loss_tokens"] == 3
    from areal.elastic.contracts import WorkloadStats

    assert WorkloadStats(**row["workload"]).n_seqs == 1
    assert session_api_key not in json.dumps(row)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_rewards", "normalized_rewards"),
    [
        ((1.0, 3.0), (-1.0, 1.0)),
        ((1.0, 1.0), (0.0, 0.0)),
        ((3.0,), (3.0,)),
    ],
)
async def test_export_trajectories_normalizes_rewards_within_group(
    client, monkeypatch, raw_rewards, normalized_rewards
):
    import torch

    from areal.infra.rpc.serialization import deserialize_value

    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.RTensor.remotize",
        lambda obj, node_addr: obj,
    )
    start = await client.post(
        "/rl/start_session",
        json={"task_id": "normalized-group", "group_size": len(raw_rewards)},
        headers=admin_headers(),
    )
    sessions = start.json()["sessions"]
    for session, reward in zip(sessions, raw_rewards):
        await client.post(
            "/chat/completions",
            json={"model": "sglang", "messages": [{"role": "user", "content": "q"}]},
            headers=session_headers(session["session_api_key"]),
        )
        reward_response = await client.post(
            "/rl/set_reward",
            json={"reward": reward},
            headers=session_headers(session["session_api_key"]),
        )
        assert reward_response.status_code == 200

    export_response = await client.post(
        "/export_trajectories",
        json={
            "session_ids": [session["session_id"] for session in sessions],
            "discount": 1.0,
            "style": "individual",
            "reward_normalization": True,
        },
        headers=admin_headers(),
    )

    assert export_response.status_code == 200
    trajectory = deserialize_value(export_response.json()["traj"])
    torch.testing.assert_close(
        trajectory["rewards"].flatten(),
        torch.tensor(normalized_rewards),
    )
    torch.testing.assert_close(
        trajectory["original_rewards"].flatten(),
        torch.tensor(raw_rewards),
    )


@pytest.mark.asyncio
async def test_export_trajectories_drops_incomplete_normalization_group(
    client, monkeypatch
):
    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.RTensor.remotize",
        lambda obj, node_addr: obj,
    )
    start = await client.post(
        "/rl/start_session",
        json={"task_id": "incomplete-group"},
        headers=admin_headers(),
    )
    session = start.json()["sessions"][0]
    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "q"}]},
        headers=session_headers(session["session_api_key"]),
    )
    await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(session["session_api_key"]),
    )

    export_response = await client.post(
        "/export_trajectories",
        json={
            "session_ids": [session["session_id"], "missing-session"],
            "reward_normalization": True,
        },
        headers=admin_headers(),
    )

    assert export_response.status_code == 200
    assert export_response.json()["traj"] == {}


# =============================================================================
# Coexistence tests — HITL and batch running simultaneously
# =============================================================================


@pytest.mark.asyncio
async def test_hitl_and_batch_can_run_simultaneously(client):
    start = await client.post(
        "/rl/start_session",
        json={"task_id": "coexist-batch"},
        headers=admin_headers(),
    )
    batch_key = start.json()["sessions"][0]["session_api_key"]

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "hitl"}]},
        headers=session_headers(ADMIN_KEY),
    )
    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "batch"}]},
        headers=session_headers(batch_key),
    )

    hitl_reward = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(ADMIN_KEY),
    )
    batch_reward = await client.post(
        "/rl/set_reward",
        json={"reward": 2.0},
        headers=session_headers(batch_key),
    )

    assert hitl_reward.status_code == 200
    assert batch_reward.status_code == 200
    assert hitl_reward.json()["session_id"] == "__hitl__"
    assert (
        batch_reward.json()["session_id"] == start.json()["sessions"][0]["session_id"]
    )

    health = await client.get("/health")
    assert health.json()["sessions"] == 2


@pytest.mark.asyncio
async def test_admin_key_still_only_maps_to_hitl_session_while_batch_uses_session_key(
    client,
):
    store: SessionStore = client._transport.app.state.session_store

    start = await client.post(
        "/rl/start_session",
        json={"task_id": "mapping"},
        headers=admin_headers(),
    )
    batch_key = start.json()["sessions"][0]["session_api_key"]
    batch_id = start.json()["sessions"][0]["session_id"]

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "hitl"}]},
        headers=session_headers(ADMIN_KEY),
    )

    hitl_session = store.get_session("__hitl__")
    batch_session = store.get_session(batch_id)
    assert hitl_session is not None
    assert batch_session is not None
    assert hitl_session is not batch_session

    batch_by_key = store.get_session_by_api_key(batch_key)
    assert batch_by_key is batch_session


# =============================================================================
# Negative tests
# =============================================================================


@pytest.mark.asyncio
async def test_online_start_session_rejects_wrong_admin_key(client):
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "reject"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_online_set_reward_rejects_unknown_session_key(client):
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers={"Authorization": "Bearer unknown-token-xyz"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_chat_completion_without_valid_token_falls_through_to_standalone(
    client,
):
    resp = await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer unknown-token-xyz"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_export_trajectories_not_found(client):
    resp = await client.post(
        "/export_trajectories",
        json={"session_ids": ["nonexistent"], "discount": 1.0, "style": "individual"},
        headers=admin_headers(),
    )
    assert resp.status_code == 200
    assert resp.json()["traj"] == {}


@pytest.mark.asyncio
async def test_export_trajectories_without_admin_key(client):
    resp = await client.post(
        "/export_trajectories",
        json={"session_ids": ["x"], "discount": 1.0, "style": "individual"},
    )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_export_trajectories_drops_retry_orphans_before_discount(
    client, monkeypatch
):
    from areal.infra.rpc.serialization import deserialize_value

    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.RTensor.remotize",
        lambda obj, node_addr: obj,
    )
    start = await client.post(
        "/rl/start_session",
        json={"task_id": "retry-orphan"},
        headers=admin_headers(),
    )
    session = start.json()["sessions"][0]
    headers = session_headers(session["session_api_key"])
    initial_messages = [{"role": "user", "content": "hi"}]

    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": initial_messages},
        headers=headers,
    )
    await client.post(
        "/chat/completions",
        json={"model": "sglang", "messages": initial_messages},
        headers=headers,
    )
    await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": initial_messages
            + [
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "more"},
            ],
        },
        headers=headers,
    )
    reward_response = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=headers,
    )
    assert reward_response.status_code == 200

    app = client._transport.app
    session_data = app.state.session_store.get_session(session["session_id"])
    assert session_data is not None
    ready = next(iter(session_data._ready_trajectories.values()))
    for interaction in ready.completions.values():
        interaction.chat_template_type = "concat"

    export_response = await client.post(
        "/export_trajectories",
        json={
            "session_ids": [session["session_id"]],
            "discount": 1.0,
            "style": "concat",
            "drop_retry_orphans": True,
        },
        headers=admin_headers(),
    )

    assert export_response.status_code == 200
    trajectory = deserialize_value(export_response.json()["traj"])
    assert trajectory["input_ids"].shape[0] == 1


# =============================================================================
# =============================================================================
@pytest.mark.asyncio
async def test_online_chat_completion_implicitly_binds_session(client):
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(ADMIN_KEY),
    )
    assert resp.status_code == 200

    health = await client.get("/health")
    assert health.status_code == 200
    assert health.json()["sessions"] == 1


@pytest.mark.asyncio
async def test_online_set_reward_returns_trajectory_metadata(client):
    token = ADMIN_KEY
    await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(token),
    )

    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(token),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "__hitl__"
    assert data["trajectory_id"] == 0
    assert data["trajectory_ready"] is True
    assert data["ready_transition"] is True


@pytest.mark.asyncio
async def test_online_set_reward_duplicate_is_idempotent(client):
    token = ADMIN_KEY
    await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=session_headers(token),
    )

    first = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(token),
    )
    second = await client.post(
        "/rl/set_reward",
        json={"reward": 2.0},
        headers=session_headers(token),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["trajectory_id"] == second.json()["trajectory_id"] == 0
    assert first.json()["ready_transition"] is True
    assert second.json()["ready_transition"] is False


@pytest.mark.skip(reason="pending /export_trajectories traj schema migration")
@pytest.mark.asyncio
async def test_online_export_latest_ready_without_trajectory_id(client):
    token = ADMIN_KEY

    await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "first"}],
        },
        headers=session_headers(token),
    )
    first_reward = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(token),
    )
    assert first_reward.status_code == 200

    await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "second"}],
        },
        headers=session_headers(token),
    )
    second_reward = await client.post(
        "/rl/set_reward",
        json={"reward": 2.0},
        headers=session_headers(token),
    )
    assert second_reward.status_code == 200
    assert second_reward.json()["trajectory_id"] == 1

    export_resp = await client.post(
        "/export_trajectories",
        json={"session_ids": ["__hitl__"], "discount": 1.0, "style": "individual"},
        headers=admin_headers(),
    )
    assert export_resp.status_code == 200
    interactions = export_resp.json()["interactions"]
    assert list(interactions) == ["chatcmpl-test1"]


@pytest.mark.skip(reason="pending /export_trajectories traj schema migration")
@pytest.mark.asyncio
async def test_online_export_explicit_trajectory_id(client):
    token = ADMIN_KEY

    await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "first"}],
        },
        headers=session_headers(token),
    )
    first_reward = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(token),
    )
    assert first_reward.status_code == 200

    await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "second"}],
        },
        headers=session_headers(token),
    )
    second_reward = await client.post(
        "/rl/set_reward",
        json={"reward": 2.0},
        headers=session_headers(token),
    )
    assert second_reward.status_code == 200

    export_resp = await client.post(
        "/export_trajectories",
        json={
            "session_ids": ["__hitl__"],
            "trajectory_id": 0,
            "discount": 1.0,
            "style": "individual",
            "remove_session": False,
        },
        headers=admin_headers(),
    )
    assert export_resp.status_code == 200
    interactions = export_resp.json()["interactions"]
    assert list(interactions) == ["chatcmpl-test0"]

    health = await client.get("/health")
    assert health.status_code == 200
    assert health.json()["sessions"] == 1


# =============================================================================
# Endpoint tests: /health (updated with sessions count)
# =============================================================================


@pytest.mark.asyncio
async def test_health_includes_sessions(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "sessions" in data
    assert data["sessions"] == 0


@pytest.mark.asyncio
async def test_health_sessions_count_after_start(client):
    # Start a session
    await client.post(
        "/rl/start_session",
        json={"task_id": "health-test"},
        headers=admin_headers(),
    )
    resp = await client.get("/health")
    assert resp.json()["sessions"] == 1


# =============================================================================
# Full lifecycle test
# =============================================================================


@pytest.mark.skip(reason="pending /export_trajectories traj schema migration")
@pytest.mark.asyncio
async def test_full_session_lifecycle(client, mock_areal_client):
    """Test the complete flow: start → chat → set_reward → export."""
    # 1. Start session
    resp = await client.post(
        "/rl/start_session",
        json={"task_id": "lifecycle"},
        headers=admin_headers(),
    )
    assert resp.status_code == 201
    session_id = resp.json()["sessions"][0]["session_id"]
    api_key = resp.json()["sessions"][0]["session_api_key"]

    # 2. Chat completion
    resp = await client.post(
        "/chat/completions",
        json={
            "model": "sglang",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        },
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    assert resp.json()["object"] == "chat.completion"

    # 3. Set reward (auto-finishes session)
    resp = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=session_headers(api_key),
    )
    assert resp.status_code == 200
    assert resp.json()["interaction_count"] == 1
    assert resp.json()["ready_transition"] is True

    # 4. Export trajectories
    resp = await client.post(
        "/export_trajectories",
        json={"session_ids": [session_id], "discount": 1.0, "style": "individual"},
        headers=admin_headers(),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "interactions" in data


class TestRetainedTrajectory:
    @staticmethod
    def ready_session():
        from areal.v2.inference_service.data_proxy.session import ReadyTrajectory

        session = SessionData("retained-session")
        from areal.experimental.openai.types import InteractionWithTokenLogpReward

        completions = MagicMock()
        completions.snapshot.return_value = completions
        completions.export_interactions.return_value = {
            "interaction": InteractionWithTokenLogpReward(reward=1.0)
        }
        session._ready_trajectories[0] = ReadyTrajectory(
            trajectory_id=0,
            interaction_id="interaction",
            completions=completions,
            created_at=time.time(),
        )
        return session, completions

    def test_repeated_peek_keeps_source_until_idempotent_ack(self):
        session, cache = self.ready_session()
        first = session.export_trajectory(
            1.0, "individual", trajectory_id=0, consume=False
        )
        second = session.export_trajectory(
            1.0, "individual", trajectory_id=0, consume=False
        )

        assert first == second
        assert session.has_ready_trajectories
        cache.export_interactions.assert_called_once()
        session.acknowledge_trajectory(0)
        session.acknowledge_trajectory(0)
        assert not session.has_ready_trajectories

    def test_failed_export_keeps_source_available_for_retry(self):
        session, cache = self.ready_session()
        cache.export_interactions.side_effect = RuntimeError("materialization failed")
        with pytest.raises(RuntimeError, match="materialization failed"):
            session.export_trajectory(1.0, "individual")
        assert session.has_ready_trajectories

        cache.export_interactions.side_effect = None
        session.export_trajectory(1.0, "individual", consume=False)
        session.acknowledge_trajectory(0)
        assert not session.has_ready_trajectories

    def test_ack_cannot_discard_unexported_or_unknown_trajectory(self):
        session, _ = self.ready_session()
        with pytest.raises(ValueError, match="before exporting"):
            session.acknowledge_trajectory(0)
        with pytest.raises(KeyError):
            session.acknowledge_trajectory(1)
        assert session.has_ready_trajectories

    def test_peek_retry_cannot_change_reward_export_options(self):
        session, cache = self.ready_session()
        session.export_trajectory(1.0, "individual", consume=False)
        with pytest.raises(ValueError, match="same export options"):
            session.export_trajectory(0.5, "individual", consume=False)
        cache.export_interactions.assert_called_once()

    def test_failed_concat_export_does_not_mutate_source_discount(self):
        import torch

        from areal.experimental.openai.types import InteractionWithTokenLogpReward
        from areal.v2.inference_service.data_proxy.session import ReadyTrajectory

        session = SessionData("failed-export")
        cache = session.active_completions
        for key, reward in (("first", 0.0), ("last", 2.0)):
            interaction = InteractionWithTokenLogpReward(
                reward=reward,
                output_message_list=[{"role": "assistant", "content": key}],
            )
            interaction.interaction_id = key
            interaction._cache = {"rewards": torch.tensor([reward])}
            cache[key] = interaction
        session._ready_trajectories[0] = ReadyTrajectory(0, "last", cache, time.time())
        with pytest.raises(ValueError, match="chat_template_type"):
            session.export_trajectory(0.5, "concat", consume=False)
        assert cache["first"].reward == 0.0
        assert cache._apply_reward_discount_called is False

        _, exported = session.export_trajectory(0.25, "individual", consume=False)
        assert exported["first"].reward == 0.5
        torch.testing.assert_close(
            exported["first"]._cache["rewards"], torch.tensor([0.5]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            cache["first"]._cache["rewards"], torch.tensor([0.0]), rtol=0, atol=0
        )
        exported["first"].reward = 999
        _, retry = session.export_trajectory(0.25, "individual", consume=False)
        assert retry["first"].reward == 0.5


@pytest.mark.asyncio
async def test_concurrent_generation_shares_credit_and_reward_waits_for_completion():
    import asyncio

    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    session = SessionData("concurrent-episode")

    async def admit(session_id, trajectory_id):
        await asyncio.sleep(0)
        return {"episode_id": "episode-0", "lease": "lease-0", "status": "admitted"}

    admission = AsyncMock(side_effect=admit)
    episode = EpisodeIdentity(incarnation_id=session.incarnation_id, trajectory_id=0)
    first, second = await asyncio.gather(
        session.begin_generation(admission, episode=episode),
        session.begin_generation(admission, episode=episode),
    )
    assert first is second
    admission.assert_awaited_once_with("concurrent-episode", 0)
    interaction = InteractionWithTokenLogpReward(
        output_message_list=[{"role": "assistant", "content": "answer"}]
    )
    interaction.interaction_id = "first"
    first["first"] = interaction
    assert session.set_reward(None, 1.0).trajectory_id is None
    session.finish_generation()
    assert session.finalize_if_reward_timeout_elapsed() is None
    session.finish_generation()
    assert session.finalize_if_reward_timeout_elapsed().trajectory_id == 0
    notification = session.pending_online_callbacks()[0]
    assert notification.episode_id == "episode-0"
    assert notification.lease == "lease-0"


@pytest.mark.asyncio
async def test_abort_lost_reply_fences_generation_reward_and_old_episode_requests():
    import asyncio

    from areal.experimental.openai.types import InteractionWithTokenLogpReward
    from areal.v2.inference_service.data_proxy.session import AbortEpisodeRequest

    session = SessionData("abort-race")
    identity = EpisodeIdentity(incarnation_id=session.incarnation_id, trajectory_id=0)
    admission = AsyncMock(return_value={"episode_id": "ep0", "lease": "lease0"})
    cache = await session.begin_generation(admission, episode=identity)
    interaction = InteractionWithTokenLogpReward(
        output_message_list=[{"role": "assistant", "content": "unrewarded"}]
    )
    interaction.interaction_id = "old"
    cache["old"] = interaction
    request = AbortEpisodeRequest(**identity.model_dump(), episode_id="ep0")
    cancel = AsyncMock(return_value={"status": "rejected", "episode_id": "ep0"})
    with pytest.raises(ValueError, match="in-flight"):
        await session.abort_episode(request, cancel)
    session.finish_generation()

    entered, finish = asyncio.Event(), asyncio.Event()

    async def lost_reply(intent):
        entered.set()
        await finish.wait()
        raise httpx.ReadError("Durably cancelled; lost response")

    pending = asyncio.create_task(session.abort_episode(request, lost_reply))
    await entered.wait()
    with pytest.raises(ValueError, match="abort"):
        await session.begin_generation(admission, episode=identity)
    with pytest.raises(ValueError, match="abort"):
        session.set_reward(None, 1.0, episode=identity, require_episode_identity=True)
    assert session.finalize_if_reward_timeout_elapsed() is None
    finish.set()
    with pytest.raises(httpx.ReadError):
        await pending
    assert session.current_episode()["state"] == "aborting"
    receipt = await session.abort_episode(request, cancel)
    assert receipt["status"] == "aborted"
    next_identity = EpisodeIdentity(
        incarnation_id=session.incarnation_id, trajectory_id=1
    )
    admission.return_value = {"episode_id": "ep1", "lease": "lease1"}
    next_cache = await session.begin_generation(admission, episode=next_identity)
    assert next_cache is not cache
    assert len(next_cache) == 0
    assert await session.abort_episode(request, cancel) == receipt
    assert session.current_episode()["episode_id"] == "ep1"
    with pytest.raises(ValueError, match="different or already sealed"):
        await session.begin_generation(admission, episode=identity)
    with pytest.raises(ValueError, match="different or already sealed"):
        session.set_reward(None, 1.0, episode=identity, require_episode_identity=True)
    session.finish_generation()
    assert admission.await_count == 2
    assert cancel.await_count == 1


@pytest.mark.asyncio
async def test_unknown_generation_retains_credit_and_session_until_terminal_proof():
    from areal.v2.inference_service.data_proxy.session import AbortEpisodeRequest

    store = SessionStore()
    sid, _ = store.start_session("unknown")
    session = store.get_session(sid)
    identity = EpisodeIdentity(incarnation_id=session.incarnation_id, trajectory_id=0)
    admission = AsyncMock(return_value={"episode_id": "ep0", "lease": "lease0"})
    await session.begin_generation(admission, episode=identity)
    session.finish_generation(uncertain=True)
    # Even a later successful response cannot prove the earlier backend stopped.
    assert session.current_episode()["state"] == "quarantined"
    with pytest.raises(ValueError, match="quarantined"):
        await session.begin_generation(admission, episode=identity)
    with pytest.raises(ValueError, match="quarantined"):
        session.set_reward(None, 1.0, episode=identity, require_episode_identity=True)
    cancel = AsyncMock()
    with pytest.raises(ValueError, match="unknown generation"):
        await session.abort_episode(
            AbortEpisodeRequest(**identity.model_dump(), episode_id="ep0"), cancel
        )
    session._last_access_time = 0
    store.cleanup_stale()
    assert store.get_session(sid) is session
    cancel.assert_not_awaited()
    admission.assert_awaited_once()
    # Resident requests never allocated online credits and retain their API.
    resident = SessionData("resident")
    await resident.begin_generation()
    resident.finish_generation(uncertain=True)
    await resident.begin_generation()
    resident.finish_generation()


@pytest.mark.asyncio
async def test_durable_abort_http_rejects_late_generation_and_reward(
    client, config, monkeypatch, tmp_path, mock_areal_client
):
    import json

    from areal.elastic.state.episode_inbox import OnlineInbox

    inbox = OnlineInbox(tmp_path, batch_size=1, max_staleness=0)
    config.online_training_admission = True
    config.callback_server_addr = "http://ledger"

    async def callback(request):
        assert request.headers["authorization"] == f"Bearer {ADMIN_KEY}"
        body = json.loads(request.content)
        if request.url.path.endswith("online_admit"):
            result = inbox.admit(
                body["source_id"], body["session_id"], body["trajectory_id"]
            )
        else:
            result = inbox.cancel_admission(**body)
        return httpx.Response(200, json=result)

    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.create_httpx_client",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(callback)),
    )
    started = await client.post(
        "/rl/start_session", json={"task_id": "abort-http"}, headers=admin_headers()
    )
    credentials = started.json()["sessions"][0]
    headers = session_headers(credentials["session_api_key"])
    identity = credentials["areal_episode"]
    body = {"messages": [{"role": "user", "content": "q"}], "areal_episode": identity}
    assert (
        await client.post("/chat/completions", json={"messages": []}, headers=headers)
    ).status_code == 409
    assert inbox.status()["active_credits"] == 0
    assert (
        await client.post("/chat/completions", json=body, headers=headers)
    ).status_code == 200
    current = (await client.post("/rl/current_episode", headers=headers)).json()
    assert current["state"] == "active"
    cancelled = await client.post("/rl/abort_episode", json=current, headers=headers)
    assert cancelled.status_code == 200
    assert inbox.status()["active_credits"] == 0
    next_identity = {**identity, "trajectory_id": 1}
    assert (
        await client.post(
            "/chat/completions",
            json={**body, "areal_episode": next_identity},
            headers=headers,
        )
    ).status_code == 200
    before = mock_areal_client.chat.completions.create.await_count
    assert (
        await client.post("/chat/completions", json=body, headers=headers)
    ).status_code == 409

    assert (
        await client.post(
            "/rl/set_reward",
            json={"reward": 1.0, "areal_episode": identity},
            headers=headers,
        )
    ).status_code == 409
    # A retry of the old abort cannot retire the new episode's credit.
    assert (
        await client.post("/rl/abort_episode", json=current, headers=headers)
    ).status_code == 200
    assert inbox.status()["active_credits"] == 1
    assert mock_areal_client.chat.completions.create.await_count == before
    rewarded = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0, "areal_episode": next_identity},
        headers=headers,
    )
    assert rewarded.status_code == 200
    assert rewarded.json()["trajectory_id"] == 1
    assert (
        await client.post(
            "/rl/set_reward",
            json={"reward": 2.0, "areal_episode": next_identity},
            headers=headers,
        )
    ).status_code == 409
    # The training ledger may explicitly reject the received sample. The
    # source still remembers its reward receipt after the next episode starts.
    ledger_episode = (await client.post("/rl/current_episode", headers=headers)).json()
    assert ledger_episode["trajectory_id"] == 2
    episode_id = (
        client._transport.app.state.session_store.get_session(credentials["session_id"])
        .pending_online_callbacks()[0]
        .episode_id
    )
    inbox.reject(episode_id, "test consumer rejection")
    third_identity = {**identity, "trajectory_id": 2}
    assert (
        await client.post(
            "/chat/completions",
            json={**body, "areal_episode": third_identity},
            headers=headers,
        )
    ).status_code == 200
    replayed = await client.post(
        "/rl/set_reward",
        json={"reward": 1.0, "areal_episode": next_identity},
        headers=headers,
    )
    assert replayed.status_code == 200
    assert replayed.json()["ready_transition"] is False
    assert (await client.post("/rl/current_episode", headers=headers)).json()[
        "trajectory_id"
    ] == 2
    assert inbox.status()["active_credits"] == 1
    # Key refresh must not strand a new episode behind an older ready sample.
    assert (
        await client.post(
            "/rl/start_session",
            json={"task_id": "replacement", "api_key": credentials["session_api_key"]},
            headers=admin_headers(),
        )
    ).status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["admission", "generation"])
async def test_lost_online_reply_only_allows_abort_before_backend_dispatch(
    client, config, monkeypatch, tmp_path, mock_areal_client, phase
):
    import json

    from areal.elastic.state.episode_inbox import OnlineInbox

    inbox = OnlineInbox(tmp_path, batch_size=1, max_staleness=0)
    config.online_training_admission = True
    config.callback_server_addr = "http://ledger"

    async def callback(request):
        body = json.loads(request.content)
        if request.url.path.endswith("online_admit"):
            result = inbox.admit(
                body["source_id"], body["session_id"], body["trajectory_id"]
            )
            if phase == "admission":
                raise httpx.ReadError("admitted, response lost", request=request)
        else:
            result = inbox.cancel_admission(**body)
        return httpx.Response(200, json=result)

    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.create_httpx_client",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(callback)),
    )
    if phase == "generation":
        mock_areal_client.chat.completions.create.side_effect = TimeoutError(
            "backend outcome unknown"
        )
    credentials = (
        await client.post(
            "/rl/start_session", json={"task_id": "lost-reply"}, headers=admin_headers()
        )
    ).json()["sessions"][0]
    headers = session_headers(credentials["session_api_key"])
    generated = await client.post(
        "/chat/completions",
        json={"messages": [], "areal_episode": credentials["areal_episode"]},
        headers=headers,
    )
    assert generated.status_code == (503 if phase == "admission" else 500)
    assert inbox.status()["active_credits"] == 1
    current = (await client.post("/rl/current_episode", headers=headers)).json()
    assert current["state"] == (
        "admission_unconfirmed" if phase == "admission" else "quarantined"
    )
    aborted = await client.post("/rl/abort_episode", json=current, headers=headers)
    assert aborted.status_code == (200 if phase == "admission" else 409)
    assert inbox.status()["active_credits"] == (0 if phase == "admission" else 1)
    assert mock_areal_client.chat.completions.create.await_count == (
        0 if phase == "admission" else 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("lose_ack", [False, True])
async def test_durable_online_handoff_retries_without_dummy_workflows(
    client, config, monkeypatch, tmp_path, lose_ack
):
    import json

    from areal.api.cli_args import AgentConfig, InferenceEngineConfig
    from areal.elastic.state.episode_inbox import OnlineInbox
    from areal.v2.inference_service.controller.controller import RolloutControllerV2

    monkeypatch.setattr("areal.utils.network.gethostip", lambda: "127.0.0.1")
    inbox = OnlineInbox(tmp_path, batch_size=4, max_staleness=2)
    controller = RolloutControllerV2(
        config=InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key=ADMIN_KEY,
            consumer_batch_size=4,
            max_head_offpolicyness=2,
            agent=AgentConfig(
                mode="online",
                agent_cls_path="areal.experimental.openai.proxy.online_agent._OnlineAgent",
            ),
        ),
        scheduler=MagicMock(n_gpus_per_node=8),
    )
    controller.configure_online_training(inbox)
    controller._gateway_addr = "http://proxy"
    app = client._transport.app
    exports, lost = [], False

    async def export(request):
        nonlocal lost
        body = json.loads(request.content)
        exports.append(body)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as proxy:
            response = await proxy.post(
                "/export_trajectories", json=body, headers=admin_headers()
            )
        if body.get("acknowledge_only") and lose_ack and not lost:
            lost = True
            raise httpx.ReadError(
                "Source accepted ack; response was lost", request=request
            )
        return httpx.Response(
            response.status_code, content=response.content, request=request
        )

    monkeypatch.setattr(
        "areal.v2.inference_service.controller.controller.create_httpx_client",
        lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(export)),
    )
    controller._start_online_callback_server()
    config.online_training_admission = True
    config.callback_server_addr = f"http://{controller.callback_addr}"
    try:
        for number in range(4):
            start = await client.post(
                "/rl/start_session",
                json={"task_id": f"durable-{number}"},
                headers=admin_headers(),
            )
            credentials = start.json()["sessions"][0]
            assert inbox.status()["active_credits"] == number
            headers = session_headers(credentials["session_api_key"])
            for _ in range(2):
                generated = await client.post(
                    "/chat/completions",
                    json={
                        "model": "sglang",
                        "areal_episode": credentials["areal_episode"],
                        "messages": [{"role": "user", "content": "q"}],
                    },
                    headers=headers,
                )
                assert generated.status_code == 200
                assert inbox.status()["active_credits"] == number + 1
            assert (
                await client.post(
                    "/rl/set_reward",
                    json={"reward": 1.0, "areal_episode": credentials["areal_episode"]},
                    headers=headers,
                )
            ).status_code == 200
            notifications = app.state.session_store.pending_online_callbacks()
            assert len(notifications) == 1
            assert (
                await client.post(
                    "/export_trajectories",
                    json={"session_ids": [credentials["session_id"]]},
                    headers=admin_headers(),
                )
            ).status_code == 400
            await _flush_ready_trajectories(app)
            if number == 0 and lose_ack:
                assert len(app.state.session_store.pending_online_callbacks()) == 1
                await _flush_ready_trajectories(app)
            assert app.state.session_store.pending_online_callbacks() == []
            notification = notifications[0]
            # Lost callback response causes a duplicate callback after source ack.
            assert (
                await controller._handle_online_ready_callback(
                    {
                        "session_id": notification.session_id,
                        "trajectory_id": notification.trajectory_id,
                        "episode_id": notification.episode_id,
                        "lease": notification.lease,
                    }
                )
            )["status"] == "ok"
        assert inbox.status()["active_credits"] == 4
        assert inbox.status()["counts"] == {"ready": 4}
        assert inbox.seal_batch(0) is not None
        assert len(exports) == 8 + int(lose_ack)
        assert not controller._completed_online_results
        assert controller._workflow_executor is None
        # A reused textual session ID denotes a new incarnation, never old data.
        old_notification = notification
        old_session_id = credentials["session_id"]
        app.state.session_store.remove_session(old_session_id)
        restarted = await client.post(
            "/rl/start_session", json={"task_id": "durable-3"}, headers=admin_headers()
        )
        credentials = restarted.json()["sessions"][0]
        assert credentials["session_id"] == old_session_id
        headers = session_headers(credentials["session_api_key"])
        assert (
            await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "areal_episode": credentials["areal_episode"],
                    "messages": [{"role": "user", "content": "new"}],
                },
                headers=headers,
            )
        ).status_code == 200
        assert (
            await client.post(
                "/rl/set_reward",
                json={"reward": 2.0, "areal_episode": credentials["areal_episode"]},
                headers=headers,
            )
        ).status_code == 200
        new_notification = app.state.session_store.pending_online_callbacks()[0]
        assert new_notification.episode_id != old_notification.episode_id
        assert inbox.status()["active_credits"] == 5
        stale_ack = await client.post(
            "/export_trajectories",
            json={
                "session_ids": [old_session_id],
                "trajectory_id": 0,
                "remove_session": False,
                "acknowledge_only": True,
                "episode_id": old_notification.episode_id,
                "lease": old_notification.lease,
                "handoff_digest": inbox.episode(old_notification.episode_id)["digest"],
            },
            headers=admin_headers(),
        )
        assert stale_ack.status_code == 409
        await _flush_ready_trajectories(app)
        assert inbox.status()["counts"] == {"reserved": 4, "ready": 1}
        # Explicit abort uses the same real authenticated callback server.
        async with httpx.AsyncClient() as callback_client:
            unauthorized = await callback_client.post(
                f"http://{controller.callback_addr}/callback/online_abort", json={}
            )
            assert unauthorized.status_code == 403
        abandoned = (
            await client.post(
                "/rl/start_session",
                json={"task_id": "abandoned"},
                headers=admin_headers(),
            )
        ).json()["sessions"][0]
        abandoned_headers = session_headers(abandoned["session_api_key"])
        assert (
            await client.post(
                "/chat/completions",
                json={"messages": [], "areal_episode": abandoned["areal_episode"]},
                headers=abandoned_headers,
            )
        ).status_code == 200
        assert inbox.status()["active_credits"] == 6
        identity = (
            await client.post("/rl/current_episode", headers=abandoned_headers)
        ).json()
        for _ in range(2):
            assert (
                await client.post(
                    "/rl/abort_episode", json=identity, headers=abandoned_headers
                )
            ).status_code == 200
            assert inbox.status()["active_credits"] == 5
    finally:
        controller._stop_online_callback_server()


@pytest.mark.asyncio
async def test_retained_export_requires_inline_payload_without_allocating_shards(
    client, monkeypatch
):
    remotize = MagicMock()
    monkeypatch.setattr(
        "areal.v2.inference_service.data_proxy.app.RTensor.remotize", remotize
    )
    response = await client.post(
        "/export_trajectories",
        json={
            "session_ids": ["missing"],
            "trajectory_id": 0,
            "remove_session": False,
            "consume": False,
        },
        headers=admin_headers(),
    )
    assert response.status_code == 400
    assert "inline_tensors" in response.text
    remotize.assert_not_called()


@pytest.mark.asyncio
async def test_retained_normalized_peek_preserves_original_rewards(client):
    import torch

    from areal.infra.rpc.serialization import deserialize_value

    start = await client.post(
        "/rl/start_session",
        json={"task_id": "retained-group", "group_size": 2},
        headers=admin_headers(),
    )
    sessions = start.json()["sessions"]
    for session, reward in zip(sessions, (1.0, 3.0)):
        headers = session_headers(session["session_api_key"])
        assert (
            await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "q"}],
                },
                headers=headers,
            )
        ).status_code == 200
        assert (
            await client.post(
                "/rl/set_reward", json={"reward": reward}, headers=headers
            )
        ).status_code == 200
    body = {
        "session_ids": [session["session_id"] for session in sessions],
        "trajectory_id": 0,
        "remove_session": False,
        "consume": False,
        "inline_tensors": True,
        "reward_normalization": True,
    }
    first = await client.post(
        "/export_trajectories", json=body, headers=admin_headers()
    )
    second = await client.post(
        "/export_trajectories", json=body, headers=admin_headers()
    )
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    normalized = deserialize_value(second.json()["traj"])
    torch.testing.assert_close(
        normalized["original_rewards"].flatten(),
        torch.tensor([1.0, 3.0]),
        rtol=0,
        atol=0,
    )
    raw = await client.post(
        "/export_trajectories",
        json={**body, "reward_normalization": False},
        headers=admin_headers(),
    )
    torch.testing.assert_close(
        deserialize_value(raw.json()["traj"])["rewards"].flatten(),
        torch.tensor([1.0, 3.0]),
        rtol=0,
        atol=0,
    )


@pytest.mark.asyncio
async def test_retained_export_roundtrip_keeps_session_for_next_trajectory(client):
    start = await client.post(
        "/rl/start_session",
        json={"task_id": "retained-roundtrip"},
        headers=admin_headers(),
    )
    credentials = start.json()["sessions"][0]
    key = credentials["session_api_key"]
    session_id = credentials["session_id"]
    for trajectory_id in range(2):
        response = await client.post(
            "/chat/completions",
            json={
                "model": "sglang",
                "messages": [{"role": "user", "content": "question"}],
            },
            headers=session_headers(key),
        )
        assert response.status_code == 200
        reward = await client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers=session_headers(key),
        )
        assert reward.json()["trajectory_id"] == trajectory_id
        export = {
            "session_ids": [session_id],
            "trajectory_id": trajectory_id,
            "style": "individual",
            "remove_session": False,
            "consume": False,
            "inline_tensors": True,
        }
        first = await client.post(
            "/export_trajectories", json=export, headers=admin_headers()
        )
        second = await client.post(
            "/export_trajectories", json=export, headers=admin_headers()
        )
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        assert first.json()["traj"]
        for _ in range(2):
            ack = await client.post(
                "/export_trajectories",
                json={**export, "acknowledge_only": True},
                headers=admin_headers(),
            )
            assert ack.status_code == 200


@pytest.mark.asyncio
async def test_resource_drain_rebind_preserves_session_and_sampling(
    config, mock_tokenizer, mock_areal_client, monkeypatch
):
    import asyncio

    from areal.v2.inference_service.data_proxy import app as proxy_app
    from areal.v2.inference_service.data_proxy.pause import PauseState

    config.deterministic_sampling = True
    app = create_app(config)
    store = SessionStore()
    store.set_admin_key(config.admin_api_key)
    session_id, key = store.start_session("retained")
    session = store.get_session(session_id)
    app.state.session_store, app.state.config = store, config
    app.state.tokenizer, app.state.pause_state = mock_tokenizer, PauseState()
    app.state.version = 7
    old_bridge = MagicMock(aclose=AsyncMock())
    app.state.inf_bridge = old_bridge
    app.state.areal_client = mock_areal_client
    entered, finish = asyncio.Event(), asyncio.Event()
    old_create = mock_areal_client.chat.completions.create.side_effect

    async def slow_create(**kwargs):
        entered.set()
        await finish.wait()
        return await old_create(**kwargs)

    mock_areal_client.chat.completions.create.side_effect = slow_create
    new_client = MagicMock()
    new_client.chat.completions.create = AsyncMock(side_effect=old_create)
    new_bridge = MagicMock(aclose=AsyncMock())
    monkeypatch.setattr(proxy_app, "_create_inf_bridge", lambda *args: new_bridge)
    monkeypatch.setattr(proxy_app, "_create_areal_client", lambda *args: new_client)
    body = {"model": "math", "messages": [{"role": "user", "content": "1+1?"}]}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = asyncio.create_task(
            client.post("/chat/completions", headers=session_headers(key), json=body)
        )
        await entered.wait()
        drain = asyncio.create_task(
            client.post(
                "/reconfiguration/drain", headers=admin_headers(), json={"timeout": 5}
            )
        )
        await asyncio.sleep(0)
        assert app.state.generation_gate.closed and not drain.done()
        second = asyncio.create_task(
            client.post("/chat/completions", headers=session_headers(key), json=body)
        )
        await asyncio.sleep(0)
        finish.set()
        assert (await first).status_code == 200
        assert (await drain).json() == {"status": "drained", "active": 0}
        assert not second.done()
        assert session._next_sampling_request_index == 1
        response = await client.post(
            "/configure_backend",
            headers=admin_headers(),
            json={"backend_addr": "http://new-backend"},
        )
        assert response.status_code == 200
        new_bridge.set_version.assert_called_once_with(7)
        old_bridge.aclose.assert_awaited_once()
        await client.post("/reconfiguration/open", headers=admin_headers())
        assert (await second).status_code == 200
    assert store.get_session_by_api_key(key) is session
    assert session._next_sampling_request_index == 2
    assert len(session.active_completions) == 2
    assert mock_areal_client.chat.completions.create.await_count == 1
    assert new_client.chat.completions.create.await_count == 1
    assert app.state.generation_gate.active == 0


@pytest.mark.asyncio
async def test_resource_drain_waits_for_admin_stream_completion(config):
    import asyncio

    app = create_app(config)
    store = SessionStore()
    store.set_admin_key(config.admin_api_key)
    app.state.session_store = store
    entered, finish = asyncio.Event(), asyncio.Event()

    async def chunks():
        entered.set()
        await finish.wait()
        yield MagicMock(model_dump_json=lambda: '{"choices":[]}')

    app.state.areal_client = MagicMock()
    app.state.areal_client.chat.completions.create = AsyncMock(return_value=chunks())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        stream = asyncio.create_task(
            client.post(
                "/chat/completions",
                headers=admin_headers(),
                json={"model": "math", "messages": [], "stream": True},
            )
        )
        await entered.wait()
        drain = asyncio.create_task(
            client.post(
                "/reconfiguration/drain", headers=admin_headers(), json={"timeout": 5}
            )
        )
        await asyncio.sleep(0)
        assert app.state.generation_gate.active == 1 and not drain.done()
        finish.set()
        assert (await stream).status_code == 200
        assert (await drain).json() == {"status": "drained", "active": 0}

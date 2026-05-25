"""Unit tests for data proxy chat/session endpoints (Plan 3b)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from areal.experimental.inference_service.data_proxy.app import (
    _flush_ready_trajectories,
    create_app,
)
from areal.experimental.inference_service.data_proxy.config import DataProxyConfig
from areal.experimental.inference_service.data_proxy.session import (
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
    from areal.experimental.inference_service.data_proxy.pause import PauseState
    from areal.experimental.inference_service.inf_bridge import InfBridge
    from areal.experimental.inference_service.sglang.bridge import SGLangBridgeBackend

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

    def test_online_session_count(self):
        store = SessionStore()
        store.set_admin_key(ADMIN_KEY)
        store.get_or_create_hitl_session()
        assert store.session_count == 1


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
        "areal.experimental.inference_service.data_proxy.app._post_online_ready_callback",
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
async def test_batch_online_set_reward_completes_that_session(client):
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
    assert "interactions" in export_resp.json()


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
    assert resp.json()["interactions"] == {}


@pytest.mark.asyncio
async def test_export_trajectories_without_admin_key(client):
    resp = await client.post(
        "/export_trajectories",
        json={"session_ids": ["x"], "discount": 1.0, "style": "individual"},
    )
    assert resp.status_code == 401


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

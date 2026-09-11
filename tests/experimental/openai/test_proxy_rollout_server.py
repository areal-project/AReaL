"""Unit tests for the proxy rollout server's session key handling."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from areal.experimental.openai.client import ContextLengthExceededError
from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import (
    SessionData,
    derive_session_gateway_api_key,
    derive_session_gateway_token,
)
from areal.experimental.openai.proxy.tensor_reference import GroupTensorStoreRegistry
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.infra.processor_cache import ProcessorCacheRegistry
from areal.infra.rpc.serialization import deserialize_value

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ADMIN_KEY = "test-admin-key"


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
    monkeypatch.setattr(srv, "_worker_role", None)
    monkeypatch.setattr(srv, "_worker_index", None)
    monkeypatch.setattr(srv, "_engine", None)
    monkeypatch.setattr(srv, "_openai_client", None)
    monkeypatch.setattr(srv, "_processor_cache_registry", ProcessorCacheRegistry())
    monkeypatch.setattr(srv, "_group_tensor_store_registry", GroupTensorStoreRegistry())


httpx = pytest.importorskip("httpx")

_transport = httpx.ASGITransport(app=srv.app)


def _client():
    return httpx.AsyncClient(transport=_transport, base_url="http://testserver")


def _admin_headers():
    return {"Authorization": f"Bearer {_ADMIN_KEY}"}


class _Request:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers
        self.url = SimpleNamespace(path="/v1/chat/completions")


def test_require_session_key_accepts_active_session_key():
    srv._api_key_to_session["session-key"] = "task-1-0"

    session_id = srv._require_session_key(
        _Request({"authorization": "Bearer session-key"})
    )

    assert session_id == "task-1-0"


def test_require_session_key_accepts_gateway_capability_for_active_session():
    srv._session_to_api_key["task-1-0"] = "session-key"
    gateway_key = derive_session_gateway_api_key(_ADMIN_KEY)
    session_token = derive_session_gateway_token(_ADMIN_KEY, "task-1-0")

    session_id = srv._require_session_key(
        _Request(
            {
                "authorization": "Bearer arena-gateway-key",
                "x-api-key": gateway_key,
                "x-session-id": "task-1-0",
                "x-session-token": session_token,
            }
        )
    )

    assert session_id == "task-1-0"


@pytest.mark.parametrize("session_id", [None, "missing-session"])
def test_require_session_key_rejects_gateway_without_active_session(session_id):
    headers = {
        "authorization": (f"Bearer {derive_session_gateway_api_key(_ADMIN_KEY)}")
    }
    if session_id is not None:
        headers["x-session-id"] = session_id
        headers["x-session-token"] = derive_session_gateway_token(
            _ADMIN_KEY, session_id
        )

    with pytest.raises(srv.HTTPException) as exc_info:
        srv._require_session_key(_Request(headers))

    assert exc_info.value.status_code == 401


def test_require_session_key_rejects_capability_for_another_active_session():
    srv._session_to_api_key["task-1-0"] = "session-key-1"
    srv._session_to_api_key["task-2-0"] = "session-key-2"

    with pytest.raises(srv.HTTPException) as exc_info:
        srv._require_session_key(
            _Request(
                {
                    "x-api-key": derive_session_gateway_api_key(_ADMIN_KEY),
                    "x-session-id": "task-2-0",
                    "x-session-token": derive_session_gateway_token(
                        _ADMIN_KEY, "task-1-0"
                    ),
                }
            )
        )

    assert exc_info.value.status_code == 401


@pytest.mark.parametrize("path", ["/rl/set_reward", "/rl/end_session"])
def test_gateway_capability_cannot_mutate_session(path):
    """A public gateway must not assign rewards or close a training session."""
    srv._session_to_api_key["task-1-0"] = "session-key"
    request = _Request(
        {
            "x-api-key": derive_session_gateway_api_key(_ADMIN_KEY),
            "x-session-id": "task-1-0",
            "x-session-token": derive_session_gateway_token(_ADMIN_KEY, "task-1-0"),
        }
    )
    request.url = SimpleNamespace(path=path)

    with pytest.raises(srv.HTTPException) as exc_info:
        srv._require_session_key(request)

    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# Tests: health reports forked worker identity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_with_worker_identity_returns_exact_role_and_index(monkeypatch):
    """Health identifies the forked worker that owns the listening port."""
    monkeypatch.setattr(srv, "_worker_role", "proxy-rollout")
    monkeypatch.setattr(srv, "_worker_index", 7)

    async with _client() as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "initialized": False,
        "role": "proxy-rollout",
        "worker_index": 7,
    }


def test_explicit_proxy_worker_index_wins_over_stale_slurm_env(monkeypatch):
    """A local proxy keeps the identity supplied by its scheduler."""
    monkeypatch.setenv("SLURM_PROCID", "0")

    assert srv._resolve_worker_index(7) == 7


def test_proxy_worker_index_falls_back_to_slurm_env(monkeypatch):
    """A Slurm proxy can still obtain its identity from the task environment."""
    monkeypatch.setenv("SLURM_PROCID", "5")

    assert srv._resolve_worker_index(-1) == 5


# ---------------------------------------------------------------------------
# Tests: message preprocessing
# ---------------------------------------------------------------------------


def test_preprocess_messages_flattens_text_blocks_and_preserves_images(monkeypatch):
    """OpenAI-routed Claude text blocks should be flattened before inference."""

    class RemoveReminder:
        def __call__(self, messages):
            for message in messages:
                if isinstance(message.get("content"), str):
                    message["content"] = message["content"].replace(
                        "reminder", "processed"
                    )
            return messages

    monkeypatch.setattr(srv, "_message_preprocessors", [RemoveReminder()])
    image_content = [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "https://example/image.png"}},
    ]
    messages = [
        {"role": "system", "content": image_content},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {
                    "type": "tool_result",
                    "content": [{"type": "text", "text": "reminder"}],
                },
            ],
        },
        {"role": "user", "content": image_content},
    ]

    result = srv._preprocess_messages(messages)

    assert isinstance(result[0]["content"], str)
    assert result[1]["content"] == "hello\nprocessed"
    assert result[2]["content"] == image_content


def test_prepare_request_messages_normalizes_tuple_and_system_content(monkeypatch):
    """Tuple-backed request messages should not bypass system normalization."""
    monkeypatch.setattr(srv, "_message_preprocessors", [])
    messages = (
        {
            "role": "system",
            "content": ({"type": "text", "text": "system prompt"},),
        },
        {"role": "user", "content": "hello"},
    )

    result = srv._prepare_request_messages(messages)

    assert isinstance(result, list)
    assert result[0]["content"] == "system prompt"


def test_prepare_request_messages_preserves_generator_after_unsupported_item(
    monkeypatch,
):
    """Unsupported generator items must not leave a partially consumed iterator."""
    monkeypatch.setattr(srv, "_message_preprocessors", [])
    unsupported = object()

    def message_generator():
        yield {"role": "user", "content": "hello"}
        yield unsupported

    result = srv._prepare_request_messages(message_generator())

    assert isinstance(result, list)
    assert result == [{"role": "user", "content": "hello"}, unsupported]


@pytest.mark.asyncio
async def test_chat_completions_accepts_image_content_in_tool_message(monkeypatch):
    """FastAPI must not lazily validate gateway image blocks as text-only."""
    captured_messages = None

    async def create(
        *, messages, areal_cache, model="areal", temperature=1.0, top_p=1.0
    ):
        nonlocal captured_messages
        del areal_cache, model, temperature, top_p
        captured_messages = messages
        return {"ok": True}

    monkeypatch.setattr(
        srv,
        "_openai_client",
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    srv._session_cache["image-session"] = SessionData(session_id="image-session")
    srv._api_key_to_session["image-key"] = "image-session"
    image_part = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
    }

    async with _client() as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer image-key"},
            json={
                "model": "areal",
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "tool-1",
                        "content": [image_part],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert captured_messages[0]["content"] == [image_part]


@pytest.mark.asyncio
async def test_internal_generation_failure_is_reported_when_session_ends(monkeypatch):
    """A proxy-side 500 must remain distinguishable from model behavior."""

    async def create(
        *, messages, areal_cache, model="areal", temperature=1.0, top_p=1.0
    ):
        del messages, areal_cache, model, temperature, top_p
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(
        srv,
        "_openai_client",
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    srv._session_cache["failed-session"] = SessionData(session_id="failed-session")
    srv._api_key_to_session["failed-key"] = "failed-session"

    async with _client() as client:
        generation = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer failed-key"},
            json={
                "model": "areal",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        ended = await client.post(
            "/rl/end_session",
            headers={"Authorization": "Bearer failed-key"},
            json={},
        )

    assert generation.status_code == 500
    assert ended.json()["system_error"] is True
    assert "backend unavailable" in ended.json()["system_error_message"]


# ---------------------------------------------------------------------------
# Tests: start_session with provided api_key
# ---------------------------------------------------------------------------


class TestStartSessionApiKey:
    @pytest.mark.asyncio
    async def test_uses_provided_api_key(self, monkeypatch):
        """Worker returns the caller-provided key instead of generating one."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "my-preferred-key"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key"] == "my-preferred-key"
        assert srv._api_key_to_session["my-preferred-key"] == data["session_id"]

    @pytest.mark.asyncio
    async def test_generates_key_when_none(self, monkeypatch):
        """No api_key → worker generates a random key (current behaviour)."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
        assert resp.status_code == 200
        key = resp.json()["api_key"]
        assert key != _ADMIN_KEY
        assert len(key) > 10  # random token

    @pytest.mark.asyncio
    async def test_rejects_admin_key_as_session_key(self, monkeypatch):
        """Cannot use the admin key as a session key."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": _ADMIN_KEY},
            )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_cleans_up_finished_session_conflict(self, monkeypatch):
        """Key reuse after a finished session cleans up old mappings."""
        # Pre-seed a finished session with the same key.
        sid_old = "old-session"
        old_session = SessionData(session_id=sid_old)
        old_session.finish()
        srv._session_cache[sid_old] = old_session
        srv._api_key_to_session["reuse-me"] = sid_old
        srv._session_to_api_key[sid_old] = "reuse-me"
        monkeypatch.setattr(srv, "_capacity", 1)

        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "reuse-me"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["api_key"] == "reuse-me"
        # Old session mapping should be gone; new one present.
        assert srv._api_key_to_session["reuse-me"] == data["session_id"]
        assert data["session_id"] != sid_old

    @pytest.mark.asyncio
    async def test_rejects_active_session_conflict(self, monkeypatch):
        """Key bound to an active (unfinished) session → 409."""
        sid_active = "active-session"
        active_session = SessionData(session_id=sid_active)
        srv._session_cache[sid_active] = active_session
        srv._api_key_to_session["busy-key"] = sid_active
        srv._session_to_api_key[sid_active] = "busy-key"
        monkeypatch.setattr(srv, "_capacity", 1)

        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t", "api_key": "busy-key"},
            )
        assert resp.status_code == 409


# ---------------------------------------------------------------------------
# Tests: end_session returns interaction_count
# ---------------------------------------------------------------------------


class TestEndSessionInteractionCount:
    @pytest.mark.asyncio
    async def test_end_session_returns_interaction_count(self, monkeypatch):
        """end_session response includes interaction_count field."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            # Start a session.
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            api_key = resp.json()["api_key"]

            # End it immediately (0 interactions).
            resp_end = await client.post(
                "/rl/end_session",
                headers={"Authorization": f"Bearer {api_key}"},
                json={},
            )
            assert resp_end.status_code == 200
            data = resp_end.json()
            assert data["interaction_count"] == 0

    @pytest.mark.asyncio
    async def test_context_overflow_is_reported_when_session_ends(self, monkeypatch):
        """A context error should persist on the session for workflow recovery."""

        async def overflow_create(*, areal_cache, **_kwargs):
            del areal_cache
            raise ContextLengthExceededError("prompt exceeds context window")

        monkeypatch.setattr(
            srv,
            "_openai_client",
            SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=overflow_create)
                )
            ),
        )
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            start = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "overflow"},
            )
            api_key = start.json()["api_key"]
            headers = {"Authorization": f"Bearer {api_key}"}

            generation = await client.post(
                "/chat/completions",
                headers=headers,
                json={
                    "model": "areal",
                    "messages": [{"role": "user", "content": "long prompt"}],
                },
            )
            ended = await client.post("/rl/end_session", headers=headers, json={})

        assert generation.status_code == 400
        assert generation.json()["detail"]["type"] == "context_length_exceeded"
        assert ended.json()["context_overflow"] is True
        assert ended.json()["context_overflow_message"] == (
            "prompt exceeds context window"
        )

    @pytest.mark.asyncio
    async def test_anthropic_streaming_context_overflow_returns_400(self, monkeypatch):
        """Anthropic streaming must preserve the structured context error."""

        async def overflow_create(*, areal_cache, **_kwargs):
            del areal_cache
            raise ContextLengthExceededError("prompt exceeds context window")

        monkeypatch.setattr(
            srv,
            "_openai_client",
            SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=overflow_create)
                )
            ),
        )
        monkeypatch.setattr(
            srv,
            "_translate_anthropic_to_openai_request",
            lambda _request: {
                "model": "areal",
                "messages": [{"role": "user", "content": "long prompt"}],
            },
        )
        monkeypatch.setattr(srv, "_capacity", 1)

        async with _client() as client:
            start = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "anthropic-stream-overflow"},
            )
            api_key = start.json()["api_key"]
            response = await client.post(
                "/v1/messages",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "areal",
                    "messages": [{"role": "user", "content": "long prompt"}],
                    "stream": True,
                },
            )

        assert response.status_code == 400
        assert response.json()["detail"] == {
            "type": "context_length_exceeded",
            "message": "prompt exceeds context window",
        }


def test_setup_openai_client_loads_and_passes_vlm_processor(monkeypatch):
    """The v1 proxy should reuse the model processor for trajectory export."""
    processor = SimpleNamespace(image_processor=object())
    tokenizer = object()
    agent_config = SimpleNamespace(
        tool_call_parser="qwen25",
        reasoning_parser="qwen3",
        engine_max_tokens=4096,
        chat_template_type="concat",
        session_timeout_seconds=60,
        admin_api_key="test-admin-key",
        message_preprocessors=[],
        prefix_matcher=None,
    )
    engine_config = SimpleNamespace(
        tokenizer_path="test-vlm",
        agent=agent_config,
        lora_name="",
    )
    monkeypatch.setattr(srv, "_engine", SimpleNamespace(config=engine_config))
    monkeypatch.setattr(
        srv,
        "load_hf_processor_and_tokenizer",
        lambda _path: (processor, tokenizer),
    )
    client_cls = MagicMock()
    monkeypatch.setattr(srv, "ArealOpenAI", client_cls)
    monkeypatch.setattr(srv, "validate_admin_api_key", lambda *_args, **_kwargs: None)

    srv._setup_openai_client()

    assert client_cls.call_args.kwargs["processor"] is processor
    assert client_cls.call_args.kwargs["tokenizer"] is tokenizer


# ---------------------------------------------------------------------------
# Tests: export_trajectories (requires session_id + admin auth)
# ---------------------------------------------------------------------------


class TestExportTrajectories:
    """Tests for the export_trajectories endpoint.

    The endpoint requires an explicit ``session_id`` in the request body
    and admin-key authentication.  This eliminates routing ambiguity when
    an API key has been reused across sessions.
    """

    @pytest.mark.asyncio
    async def test_export_with_session_id_and_admin_auth(self, monkeypatch):
        """Export succeeds with required session_id + admin key."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            session_id = resp.json()["session_id"]
            api_key = resp.json()["api_key"]

            # End the session so export doesn't block.
            await client.post(
                "/rl/end_session",
                headers={"Authorization": f"Bearer {api_key}"},
                json={},
            )

            resp_export = await client.post(
                "/export_trajectories",
                headers=_admin_headers(),
                json={"session_id": session_id},
            )
            assert resp_export.status_code == 200
            assert "interactions" in resp_export.json()

    @pytest.mark.asyncio
    async def test_group_exports_share_multimodal_tensor_reference(self):
        """Two grouped sessions should export refs and fetch one tensor payload."""
        group_id = "train:task-1"
        pixel_values = torch.arange(12).reshape(1, 3, 2, 2)

        for sample_idx in range(2):
            session_id = f"task-1:{sample_idx}-0"
            session = SessionData(
                session_id=session_id,
                processor_cache_group_id=group_id,
            )
            interaction = InteractionWithTokenLogpReward()
            interaction.interaction_id = f"interaction-{sample_idx}"
            interaction.messages = [{"role": "user", "content": "question"}]
            interaction.output_message_list = [
                {"role": "assistant", "content": "answer"}
            ]
            interaction.reward = 1.0
            interaction._cache = {
                "input_ids": torch.tensor([[sample_idx, 2]]),
                "multi_modal_input": [{"pixel_values": pixel_values}],
            }
            session.completions[interaction.interaction_id] = interaction
            session.finish()
            srv._session_cache[session_id] = session

        async with _client() as client:
            responses = []
            for sample_idx in range(2):
                response = await client.post(
                    "/export_trajectories",
                    headers=_admin_headers(),
                    json={
                        "session_id": f"task-1:{sample_idx}-0",
                        "supports_shared_tensor_references": True,
                    },
                )
                assert response.status_code == 200
                responses.append(response.json())

            first_item = next(iter(responses[0]["interactions"].values()))
            second_item = next(iter(responses[1]["interactions"].values()))
            first_ref = first_item["tensor_dict"]["multi_modal_input"][0][
                "pixel_values"
            ]
            second_ref = second_item["tensor_dict"]["multi_modal_input"][0][
                "pixel_values"
            ]
            assert first_ref == second_ref
            assert responses[0]["tensor_reference_group_id"] == group_id
            assert "data" not in first_ref

            fetch_response = await client.post(
                "/rl/fetch_shared_tensors",
                headers=_admin_headers(),
                json={"group_id": group_id, "ref_ids": [first_ref["ref_id"]]},
            )

        assert fetch_response.status_code == 200
        fetched = deserialize_value(fetch_response.json()["tensors"])
        assert list(fetched) == [first_ref["ref_id"]]
        torch.testing.assert_close(
            fetched[first_ref["ref_id"]], pixel_values, rtol=0, atol=0
        )

    @pytest.mark.asyncio
    async def test_group_export_without_reference_capability_keeps_inline_tensor(self):
        """Legacy clients should continue receiving inline multimodal tensors."""
        pixel_values = torch.arange(4)
        session = SessionData(
            session_id="legacy-session",
            processor_cache_group_id="train:legacy-task",
        )
        interaction = InteractionWithTokenLogpReward()
        interaction.interaction_id = "legacy-interaction"
        interaction.messages = [{"role": "user", "content": "question"}]
        interaction.output_message_list = [{"role": "assistant", "content": "answer"}]
        interaction.reward = 1.0
        interaction._cache = {
            "input_ids": torch.tensor([[1, 2]]),
            "multi_modal_input": [{"pixel_values": pixel_values}],
        }
        session.completions[interaction.interaction_id] = interaction
        session.finish()
        srv._session_cache[session.session_id] = session

        async with _client() as client:
            response = await client.post(
                "/export_trajectories",
                headers=_admin_headers(),
                json={"session_id": session.session_id},
            )

        assert response.status_code == 200
        data = response.json()
        item = next(iter(data["interactions"].values()))
        serialized_image = item["tensor_dict"]["multi_modal_input"][0]["pixel_values"]
        assert serialized_image["type"] == "tensor"
        assert serialized_image["data"] is not None
        assert data["tensor_reference_group_id"] is None

    @pytest.mark.asyncio
    async def test_export_rejects_non_admin_key(self, monkeypatch):
        """Export requires admin auth; a session key must be rejected."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "t"},
            )
            assert resp.status_code == 200
            session_id = resp.json()["session_id"]
            api_key = resp.json()["api_key"]

            await client.post(
                "/rl/end_session",
                headers={"Authorization": f"Bearer {api_key}"},
                json={},
            )

            resp_export = await client.post(
                "/export_trajectories",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"session_id": session_id},
            )
            assert resp_export.status_code == 403

    @pytest.mark.asyncio
    async def test_export_rejects_missing_session_id(self, monkeypatch):
        """Omitting session_id from body triggers a validation error (422)."""
        monkeypatch.setattr(srv, "_capacity", 1)
        async with _client() as client:
            resp_export = await client.post(
                "/export_trajectories",
                headers=_admin_headers(),
                json={},
            )
            assert resp_export.status_code == 422

    @pytest.mark.asyncio
    async def test_export_survives_key_remap(self, monkeypatch):
        """Explicit session_id resolves correctly even after key remapping.

        After a session refresh the API key maps to the NEW session.
        Because export uses the explicit session_id, it still targets
        the OLD (completed) session without blocking.
        """
        monkeypatch.setattr(srv, "_capacity", 2)
        async with _client() as client:
            # Start first session with a specific key.
            resp1 = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "task-0", "api_key": "shared-key"},
            )
            assert resp1.status_code == 200
            session_id_old = resp1.json()["session_id"]

            # End the first session.
            await client.post(
                "/rl/end_session",
                headers={"Authorization": "Bearer shared-key"},
                json={},
            )

            # Start a second session reusing the same key.
            resp2 = await client.post(
                "/rl/start_session",
                headers=_admin_headers(),
                json={"task_id": "task-1", "api_key": "shared-key"},
            )
            assert resp2.status_code == 200
            session_id_new = resp2.json()["session_id"]
            assert session_id_new != session_id_old

            # API key now points to the NEW session.
            assert srv._api_key_to_session["shared-key"] == session_id_new

            # Export the OLD session by session_id — unaffected by the remap.
            resp_export = await client.post(
                "/export_trajectories",
                headers=_admin_headers(),
                json={"session_id": session_id_old},
            )
            assert resp_export.status_code == 200
            assert "interactions" in resp_export.json()

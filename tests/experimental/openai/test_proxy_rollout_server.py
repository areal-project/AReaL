"""Unit tests for the proxy rollout server's session key handling."""

from __future__ import annotations

import subprocess
import sys
import threading
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import SessionData, deserialize_interactions
from areal.experimental.openai.proxy.tensor_reference import (
    GroupTensorStoreRegistry,
    SharedTensorResolver,
)
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
    monkeypatch.setattr(srv, "_processor_cache_registry", ProcessorCacheRegistry())
    monkeypatch.setattr(srv, "_group_tensor_store_registry", GroupTensorStoreRegistry())


httpx = pytest.importorskip("httpx")

_transport = httpx.ASGITransport(app=srv.app)


def _client():
    return httpx.AsyncClient(transport=_transport, base_url="http://testserver")


def _admin_headers():
    return {"Authorization": f"Bearer {_ADMIN_KEY}"}


def test_openai_proxy_without_litellm():
    # Start with a fresh module cache so earlier tests cannot mask eager imports.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import asyncio, sys\n"
            "sys.modules['litellm'] = None\n"
            f"from {__name__} import _exercise_openai_routes\n"
            "asyncio.run(_exercise_openai_routes())",
        ],
        check=True,
        timeout=90,
    )


async def _exercise_openai_routes():
    srv._openai_client = Mock()
    srv._openai_client.chat.completions.create = AsyncMock(return_value={"id": "chat"})
    srv._openai_client.responses.create = AsyncMock(
        return_value={
            "id": "resp",
            "object": "response",
            "created_at": 0,
            "model": "test",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )
    srv._api_key_to_session["test-key"] = "session"
    srv._session_cache["session"] = SessionData("session")
    headers = {"Authorization": "Bearer test-key"}
    async with _client() as client:
        chat = await client.post(
            "/chat/completions",
            headers=headers,
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
        response = await client.post(
            "/responses", headers=headers, json={"model": "test", "input": "hi"}
        )
    assert chat.status_code == 200, chat.text
    assert response.status_code == 200, response.text
    assert chat.json()["id"] == "chat"
    assert response.json()["id"] == "resp"


def test_anthropic_request_loads_litellm_adapter():
    request = {
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
    }
    translated = srv._translate_anthropic_to_openai_request(request)
    assert translated["messages"] == request["messages"]
    assert translated["max_tokens"] == 8


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

            first_item = next(
                iter(responses[0]["interactions"]["interactions"].values())
            )
            second_item = next(
                iter(responses[1]["interactions"]["interactions"].values())
            )
            first_ref = first_item["multi_modal_input"][0]["pixel_values"]
            second_ref = second_item["multi_modal_input"][0]["pixel_values"]
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
        resolver = SharedTensorResolver()
        fetches = []

        async def fetch(ref_ids):
            fetches.append(ref_ids)
            return fetched

        restored = [
            deserialize_interactions(
                await resolver.resolve(
                    response["interactions"], group_id=group_id, fetch=fetch
                )
            )
            for response in responses
        ]
        images = [
            next(iter(items.values())).to_tensor_dict()["multi_modal_input"][0][
                "pixel_values"
            ]
            for items in restored
        ]
        assert images[0] is images[1]
        assert len(fetches) == 1

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
        assert data["interactions"]["__format__"] == "mm-blobs-v1"
        serialized_image = next(iter(data["interactions"]["blobs"].values()))
        assert serialized_image["type"] == "tensor"
        assert serialized_image["data"] is not None
        assert data["tensor_reference_group_id"] is None
        restored = deserialize_interactions(data["interactions"])
        torch.testing.assert_close(
            restored["legacy-interaction"].to_tensor_dict()["multi_modal_input"][0][
                "pixel_values"
            ],
            pixel_values,
            rtol=0,
            atol=0,
        )

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

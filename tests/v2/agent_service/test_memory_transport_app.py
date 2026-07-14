# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from hashlib import sha256
from unittest.mock import patch

import pytest

from areal.v2.agent_service.auth import DEFAULT_ADMIN_API_KEY, admin_headers
from areal.v2.agent_service.memory import MemoryAgentSessionPinV1
from areal.v2.agent_service.memory_transport import (
    AREAL_INFERENCE_METADATA_KEY,
    AREAL_MEMORY_METADATA_KEY,
    CHAT_REQUEST_METADATA_KEY,
    MEMORY_ASSIGNMENT_PIN_FIELD,
    MEMORY_CONTROL_AUTHORIZED_FIELD,
    MemoryAssignmentPinWireV1,
    parse_memory_assignment_pin_metadata,
)
from areal.v2.agent_service.session_keys import session_key_sha256
from areal.v2.memory_service.types import MemoryScope

httpx = pytest.importorskip("httpx")
data_proxy_app = pytest.importorskip("areal.v2.agent_service.data_proxy.app")
data_proxy_client = pytest.importorskip("areal.v2.agent_service.data_proxy.client")
data_proxy_config = pytest.importorskip("areal.v2.agent_service.data_proxy.config")
gateway_bridge = pytest.importorskip("areal.v2.agent_service.gateway.bridge")
gateway_config = pytest.importorskip("areal.v2.agent_service.gateway.config")
fastapi = pytest.importorskip("fastapi")

_MEMORY_CONTROL_API_KEY = "test-memory-control-hop-key"
_MEMORY_CONTROL_HEADERS = admin_headers(_MEMORY_CONTROL_API_KEY)
_WORKER_HOP_API_KEY = "test-worker-hop-key"
_WORKER_HOP_HEADERS = admin_headers(_WORKER_HOP_API_KEY)
_EXTERNAL_ADMIN_API_KEY = "test-external-admin-key"


def _wire(suffix: str = "a") -> dict[str, object]:
    assignment_hash = sha256(f"assignment-{suffix}".encode()).hexdigest()
    incarnation_hash = sha256(f"incarnation-{suffix}".encode()).hexdigest()
    return MemoryAssignmentPinWireV1.from_runtime_pin(
        MemoryAgentSessionPinV1(
            scope=MemoryScope(
                tenant_id="tenant-1",
                namespace="agent-long-term-memory",
                subject_id="subject-1",
            ),
            rollout_group_id="rollout-group-1",
            rollout_group_incarnation_sha256=incarnation_hash,
            assignment_id=f"masn_{assignment_hash[:24]}",
            assignment_content_sha256=assignment_hash,
        )
    ).to_wire()


def _authorized(body: dict[str, object]) -> dict[str, object]:
    result = dict(body)
    result[MEMORY_CONTROL_AUTHORIZED_FIELD] = True
    return result


def _internal_request():
    return fastapi.Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/session/s1/turn",
            "headers": [
                (
                    b"authorization",
                    f"Bearer {_MEMORY_CONTROL_API_KEY}".encode(),
                )
            ],
        }
    )


def _worker_close_session_key(request) -> str:
    assert request.url.path == "/sessions/close"
    payload = json.loads(request.content)
    assert set(payload) == {"session_key"}
    return payload["session_key"]


def _worker_close_receipt(request) -> dict[str, str]:
    session_key = _worker_close_session_key(request)
    return {
        "status": "ok",
        "session_key_sha256": session_key_sha256(session_key),
    }


def _patched_worker_send(worker_bodies, worker_closes):
    original_send = httpx.AsyncClient.send

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path == "/run":
            worker_bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"summary": "ok", "events": [], "metadata": {}},
                request=request,
            )
        if request.url.path == "/sessions/close":
            worker_closes.append(_worker_close_session_key(request))
            return httpx.Response(
                200, json=_worker_close_receipt(request), request=request
            )
        raise AssertionError(f"unexpected worker request: {request.url}")

    return patched_send


def _proxy_app():
    return data_proxy_app.create_data_proxy_app(
        data_proxy_config.DataProxyConfig(
            worker_addr="http://worker",
            worker_hop_api_key=_WORKER_HOP_API_KEY,
            memory_control_api_key=_MEMORY_CONTROL_API_KEY,
        )
    )


def test_gateway_requires_a_distinct_memory_hop_key() -> None:
    for blank_admin in ("", "   "):
        with pytest.raises(ValueError, match="admin_api_key must not be blank"):
            gateway_config.GatewayConfig(
                admin_api_key=blank_admin,
                memory_control_api_key=_MEMORY_CONTROL_API_KEY,
            )
        with pytest.raises(ValueError, match="admin_api_key must not be blank"):
            gateway_bridge.ChatCompletionsBridge(
                router_addr="http://router",
                admin_api_key=blank_admin,
                memory_control_api_key=_MEMORY_CONTROL_API_KEY,
            )
    with pytest.raises(ValueError, match="must differ"):
        gateway_config.GatewayConfig(
            admin_api_key="same-public-key",
            memory_control_api_key="same-public-key",
        )
    with pytest.raises(ValueError, match="must differ"):
        gateway_bridge.ChatCompletionsBridge(
            router_addr="http://router",
            admin_api_key="same-public-key",
            memory_control_api_key="same-public-key",
        )
    for source_visible_default in (DEFAULT_ADMIN_API_KEY, "areal-admin-key"):
        assert not gateway_config.GatewayConfig(
            admin_api_key=source_visible_default,
            memory_control_api_key=_MEMORY_CONTROL_API_KEY,
        ).memory_control_enabled
        with pytest.raises(ValueError, match="source-visible default"):
            gateway_config.GatewayConfig(
                admin_api_key=_EXTERNAL_ADMIN_API_KEY,
                memory_control_api_key=source_visible_default,
            )
        with pytest.raises(ValueError, match="source-visible default"):
            gateway_bridge.ChatCompletionsBridge(
                router_addr="http://router",
                admin_api_key=_EXTERNAL_ADMIN_API_KEY,
                memory_control_api_key=source_visible_default,
            )
        with pytest.raises(ValueError, match="source-visible default"):
            data_proxy_config.DataProxyConfig(
                memory_control_api_key=source_visible_default,
            )
        with pytest.raises(ValueError, match="source-visible default"):
            data_proxy_client.DataProxyClient(
                "http://proxy",
                memory_control_api_key=source_visible_default,
            )
    assert gateway_config.GatewayConfig(
        admin_api_key=_EXTERNAL_ADMIN_API_KEY,
        memory_control_api_key=_MEMORY_CONTROL_API_KEY,
    ).memory_control_enabled


def test_data_proxy_requires_independent_memory_and_worker_hops() -> None:
    with pytest.raises(ValueError, match="requires an independent"):
        data_proxy_config.DataProxyConfig(
            memory_control_api_key=_MEMORY_CONTROL_API_KEY,
        )
    with pytest.raises(ValueError, match="must differ"):
        data_proxy_config.DataProxyConfig(
            memory_control_api_key="same-hop-key",
            worker_hop_api_key="same-hop-key",
        )
    for source_visible_default in (DEFAULT_ADMIN_API_KEY, "areal-admin-key"):
        with pytest.raises(ValueError, match="source-visible default"):
            data_proxy_config.DataProxyConfig(
                worker_hop_api_key=source_visible_default,
            )


@pytest.mark.asyncio
async def test_data_proxy_replaces_inbound_auth_with_worker_pair_auth() -> None:
    forwarded_authorization: list[str | None] = []
    original_send = httpx.AsyncClient.send

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        forwarded_authorization.append(request.headers.get("Authorization"))
        if request.url.path == "/run":
            return httpx.Response(
                200,
                json={"summary": "ok", "events": [], "metadata": {}},
                request=request,
            )
        if request.url.path == "/sessions/close":
            return httpx.Response(
                200, json=_worker_close_receipt(request), request=request
            )
        raise AssertionError(f"unexpected Worker request: {request.url}")

    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(httpx.AsyncClient, "send", patched_send):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            turn = await client.post(
                "/session/s1/turn",
                json={"message": "hello", "run_id": "r1"},
            )
            assert turn.status_code == 200
            closed = await client.post(
                "/sessions/close",
                json={"session_key": "s1"},
            )
            assert closed.status_code == 200

    assert forwarded_authorization == [
        _WORKER_HOP_HEADERS["Authorization"],
        _WORKER_HOP_HEADERS["Authorization"],
    ]
    assert _WORKER_HOP_API_KEY != _MEMORY_CONTROL_API_KEY


@pytest.mark.parametrize(
    ("worker_status", "worker_body", "expected_status"),
    (
        (200, b"<html>not json</html>", 502),
        (503, b"upstream unavailable", 503),
        (200, b"[]", 502),
    ),
)
@pytest.mark.asyncio
async def test_invalid_structured_worker_response_preserves_status_without_history(
    worker_status: int,
    worker_body: bytes,
    expected_status: int,
) -> None:
    original_send = httpx.AsyncClient.send

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        assert request.url.path == "/run"
        return httpx.Response(
            worker_status,
            content=worker_body,
            request=request,
        )

    app = _proxy_app()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    with patch.object(httpx.AsyncClient, "send", patched_send):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
        ) as client:
            response = await client.post(
                "/session/s1/turn",
                json={"message": "must not enter history"},
            )
            history = await client.get(
                "/session/s1/history", headers=_MEMORY_CONTROL_HEADERS
            )

    assert response.status_code == expected_status
    assert response.json() == {
        "detail": "worker returned an invalid structured response"
    }
    assert history.json() == {"history": []}


@pytest.mark.parametrize(
    "payload",
    (
        {"status": "ok"},
        {"status": "ok", "worker_hop_auth": 1},
        ["ok", True],
    ),
)
@pytest.mark.asyncio
async def test_data_proxy_startup_rejects_malformed_worker_auth_receipt(
    payload: object,
) -> None:
    forwarded_authorization: list[str | None] = []
    original_send = httpx.AsyncClient.send

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        assert request.method == "GET"
        assert request.url.path == "/internal/auth-check"
        forwarded_authorization.append(request.headers.get("Authorization"))
        return httpx.Response(200, json=payload, request=request)

    app = _proxy_app()
    with (
        patch.object(httpx.AsyncClient, "send", patched_send),
        pytest.raises(
            RuntimeError,
            match="Worker hop authentication check failed",
        ) as error,
    ):
        await app.router.startup()

    assert isinstance(error.value.__cause__, ValueError)
    assert forwarded_authorization == [_WORKER_HOP_HEADERS["Authorization"]]


@pytest.mark.parametrize(
    "source_visible_default",
    (DEFAULT_ADMIN_API_KEY, "areal-admin-key"),
)
@pytest.mark.asyncio
async def test_source_visible_admin_defaults_cannot_enable_memory_control(
    source_visible_default: str,
) -> None:
    bridge = gateway_bridge.ChatCompletionsBridge(
        router_addr="http://router",
        admin_api_key=source_visible_default,
        memory_control_api_key=_MEMORY_CONTROL_API_KEY,
    )
    app = fastapi.FastAPI()
    gateway_bridge.mount_chat_bridge(app, bridge)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://gateway",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {source_visible_default}",
                    "X-AReaL-Session-Key": "default-key-session",
                },
                json={
                    "model": "model-1",
                    "messages": [{"role": "user", "content": "hello"}],
                    MEMORY_ASSIGNMENT_PIN_FIELD: _wire(),
                },
            )
    finally:
        await bridge.close()

    assert response.status_code == 503
    assert "non-default external admin key" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_chat_bridge_rejects_ambiguous_key_before_router_reservation() -> None:
    forwarded = []

    def handler(request):
        forwarded.append(request)
        return httpx.Response(500, request=request)

    bridge = gateway_bridge.ChatCompletionsBridge(router_addr="http://router")
    await bridge._http.aclose()
    bridge._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = fastapi.FastAPI()
    gateway_bridge.mount_chat_bridge(app, bridge)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://gateway",
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                headers={"X-AReaL-Session-Key": "s%252Fb"},
                json={
                    "model": "model-1",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
    finally:
        await bridge.close()

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request"
    assert forwarded == []


@pytest.mark.asyncio
async def test_first_turn_pin_is_reused_and_forwarded_as_reserved_metadata() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            first = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {
                        "message": "first",
                        "run_id": "r1",
                        "metadata": {"caller": "value"},
                        MEMORY_ASSIGNMENT_PIN_FIELD: _wire(),
                    }
                ),
            )
            second = await client.post(
                "/session/s1/turn",
                json=_authorized({"message": "second", "run_id": "r2"}),
            )

    assert first.status_code == second.status_code == 200
    assert len(worker_bodies) == 2
    first_metadata = worker_bodies[0]["metadata"]
    second_metadata = worker_bodies[1]["metadata"]
    assert first_metadata["caller"] == "value"
    assert (
        first_metadata[AREAL_MEMORY_METADATA_KEY]
        == second_metadata[AREAL_MEMORY_METADATA_KEY]
    )
    parsed = parse_memory_assignment_pin_metadata(first_metadata)
    assert parsed == MemoryAssignmentPinWireV1.from_wire(_wire()).to_runtime_pin()
    assert AREAL_MEMORY_METADATA_KEY not in first.json().get("metadata", {})


@pytest.mark.asyncio
async def test_session_mode_blocks_ordinary_to_memory_upgrade_until_close() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            ordinary = await client.post(
                "/session/s1/turn",
                json={
                    "message": "ordinary",
                    "inf_base_url": "http://trusted-inference",
                    "session_api_key": "sk-sess-trusted",
                },
            )
            untrusted_upgrade = await client.post(
                "/session/s1/turn",
                json={
                    "message": "untrusted upgrade",
                    MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a"),
                },
            )
            upgrade = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {
                        "message": "upgrade",
                        MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a"),
                        "inf_base_url": "http://attacker-inference",
                        "session_api_key": "sk-sess-attacker",
                    }
                ),
            )
            assert app.state.memory_pin_cache.resolve("s1") is None
            retry = await client.post(
                "/session/s1/turn",
                json={"message": "still ordinary"},
            )
            closed = await client.post(
                "/sessions/close",
                json={"session_key": "s1"},
            )
            replacement = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {
                        "message": "new Memory incarnation",
                        MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a"),
                    }
                ),
            )

    assert ordinary.status_code == retry.status_code == 200
    assert untrusted_upgrade.status_code == 403
    assert upgrade.status_code == 409
    assert "security mode cannot change" in upgrade.json()["detail"]
    assert closed.status_code == replacement.status_code == 200
    assert worker_closes == ["s1"]
    assert len(worker_bodies) == 3
    assert AREAL_MEMORY_METADATA_KEY not in worker_bodies[0]["metadata"]
    assert AREAL_MEMORY_METADATA_KEY not in worker_bodies[1]["metadata"]
    assert worker_bodies[1]["metadata"][AREAL_INFERENCE_METADATA_KEY] == {
        "base_url": "http://trusted-inference",
        "api_key": "sk-sess-trusted",
        "model": "",
    }
    assert parse_memory_assignment_pin_metadata(worker_bodies[2]["metadata"]) == (
        MemoryAssignmentPinWireV1.from_wire(_wire("a")).to_runtime_pin()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ("worker_500", "send_error"))
async def test_admitted_failed_turn_still_fixes_session_mode(failure_mode) -> None:
    worker_bodies: list[dict] = []
    original_send = httpx.AsyncClient.send

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path != "/run":
            raise AssertionError(f"unexpected worker request: {request.url}")
        worker_bodies.append(json.loads(request.content))
        if len(worker_bodies) == 1:
            if failure_mode == "worker_500":
                return httpx.Response(
                    500,
                    json={"error": {"message": "injected", "type": "test"}},
                    request=request,
                )
            raise httpx.ConnectError("injected send failure", request=request)
        return httpx.Response(
            200,
            json={"summary": "ok", "events": [], "metadata": {}},
            request=request,
        )

    app = _proxy_app()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    with patch.object(httpx.AsyncClient, "send", patched_send):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            failed = await client.post(
                "/session/s1/turn",
                json={
                    "message": "admitted ordinary failure",
                    "inf_base_url": "http://trusted-inference",
                    "session_api_key": "sk-sess-trusted",
                },
            )
            upgrade = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {
                        "message": "unsafe retry as Memory",
                        MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a"),
                        "inf_base_url": "http://attacker-inference",
                        "session_api_key": "sk-sess-attacker",
                    }
                ),
            )
            retry = await client.post(
                "/session/s1/turn",
                json={"message": "retry in the original mode"},
            )

    assert failed.status_code == 500
    assert upgrade.status_code == 409
    assert retry.status_code == 200
    assert app.state.memory_pin_cache.resolve("s1") is None
    assert len(worker_bodies) == 2
    assert AREAL_MEMORY_METADATA_KEY not in worker_bodies[1]["metadata"]
    assert worker_bodies[1]["metadata"][AREAL_INFERENCE_METADATA_KEY] == {
        "base_url": "http://trusted-inference",
        "api_key": "sk-sess-trusted",
        "model": "",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ("worker_500", "send_error"))
async def test_admitted_failed_memory_turn_preserves_pin_for_trusted_reuse(
    failure_mode,
) -> None:
    worker_bodies: list[dict] = []
    original_send = httpx.AsyncClient.send

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path != "/run":
            raise AssertionError(f"unexpected worker request: {request.url}")
        worker_bodies.append(json.loads(request.content))
        if len(worker_bodies) == 1:
            if failure_mode == "worker_500":
                return httpx.Response(
                    500,
                    json={"error": {"message": "injected", "type": "test"}},
                    request=request,
                )
            raise httpx.ConnectError("injected send failure", request=request)
        return httpx.Response(
            200,
            json={"summary": "ok", "events": [], "metadata": {}},
            request=request,
        )

    app = _proxy_app()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    with patch.object(httpx.AsyncClient, "send", patched_send):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            failed = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {
                        "message": "admitted Memory failure",
                        MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a"),
                    }
                ),
            )
            untrusted_reuse = await client.post(
                "/session/s1/turn",
                headers={"Authorization": ""},
                json={"message": "untrusted retry cannot reuse the retained pin"},
            )
            retry = await client.post(
                "/session/s1/turn",
                json=_authorized({"message": "trusted retry reuses the retained pin"}),
            )

    assert failed.status_code == 500
    assert untrusted_reuse.status_code == 403
    assert retry.status_code == 200
    assert app.state.memory_pin_cache.resolve("s1") is not None
    assert len(worker_bodies) == 2
    first_pin = parse_memory_assignment_pin_metadata(worker_bodies[0]["metadata"])
    retry_pin = parse_memory_assignment_pin_metadata(worker_bodies[1]["metadata"])
    assert first_pin == retry_pin
    assert first_pin == MemoryAssignmentPinWireV1.from_wire(_wire("a")).to_runtime_pin()


@pytest.mark.asyncio
async def test_reusing_pinned_session_requires_trusted_ingress_on_every_turn() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            missing_hop_auth = await client.post(
                "/session/missing-hop/turn",
                headers={"Authorization": ""},
                json=_authorized(
                    {"message": "forged marker", MEMORY_ASSIGNMENT_PIN_FIELD: _wire()}
                ),
            )
            wrong_hop_auth = await client.post(
                "/session/wrong-hop/turn",
                headers={"Authorization": "Bearer wrong-key"},
                json=_authorized(
                    {"message": "forged marker", MEMORY_ASSIGNMENT_PIN_FIELD: _wire()}
                ),
            )
            external_admin_is_not_hop_auth = await client.post(
                "/session/external-admin/turn",
                headers={"Authorization": f"Bearer {DEFAULT_ADMIN_API_KEY}"},
                json=_authorized(
                    {
                        "message": "wrong capability",
                        MEMORY_ASSIGNMENT_PIN_FIELD: _wire(),
                    }
                ),
            )
            unauthorized_bind = await client.post(
                "/session/untrusted/turn",
                json={
                    "message": "steal by binding",
                    MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a"),
                },
            )
            health_after_rejection = await client.get("/health")
            bound = await client.post(
                "/session/shared/turn",
                json=_authorized(
                    {"message": "bind", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a")}
                ),
            )
            unauthorized_reuse = await client.post(
                "/session/shared/turn",
                json={"message": "steal by omitting the pin"},
            )
            authorized_reuse = await client.post(
                "/session/shared/turn",
                json=_authorized({"message": "reuse"}),
            )

    assert bound.status_code == authorized_reuse.status_code == 200
    assert (
        missing_hop_auth.status_code
        == wrong_hop_auth.status_code
        == external_admin_is_not_hop_auth.status_code
        == 401
    )
    assert unauthorized_bind.status_code == unauthorized_reuse.status_code == 403
    assert health_after_rejection.json()["active_sessions"] == 0
    assert len(worker_bodies) == 2
    first_pin = parse_memory_assignment_pin_metadata(worker_bodies[0]["metadata"])
    second_pin = parse_memory_assignment_pin_metadata(worker_bodies[1]["metadata"])
    assert first_pin == second_pin


@pytest.mark.asyncio
async def test_unconfigured_memory_hop_allows_plain_turns_but_rejects_pins() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = data_proxy_app.create_data_proxy_app(
        data_proxy_config.DataProxyConfig(worker_addr="http://worker")
    )
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            # A source-visible Agent admin default must not substitute for the
            # dedicated, controller-generated Memory hop capability.
            headers=admin_headers("areal-admin-key"),
        ) as client:
            ordinary = await client.post(
                "/session/plain/turn",
                json={"message": "plain"},
            )
            disabled = await client.post(
                "/session/pinned/turn",
                json=_authorized(
                    {"message": "pin", MEMORY_ASSIGNMENT_PIN_FIELD: _wire()}
                ),
            )
            health = await client.get("/health")

    assert ordinary.status_code == 200
    assert disabled.status_code == 503
    assert "not configured" in disabled.json()["detail"]
    assert health.json()["active_sessions"] == 1
    assert len(worker_bodies) == 1
    assert AREAL_MEMORY_METADATA_KEY not in worker_bodies[0]["metadata"]


@pytest.mark.asyncio
async def test_data_proxy_client_sends_hop_key_only_for_memory_control() -> None:
    forwarded = []

    def handler(request):
        forwarded.append(request)
        return httpx.Response(
            200,
            json={"summary": "ok", "events": [], "metadata": {}},
            request=request,
        )

    client = data_proxy_client.DataProxyClient("http://proxy")
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await client.turn("plain", "hello")
        with pytest.raises(ValueError, match="dedicated Memory control key"):
            await client.turn(
                "pinned",
                "hello",
                memory_control_authorized=True,
                memory_assignment_pin=MemoryAssignmentPinWireV1.from_wire(_wire()),
            )
    finally:
        await client.close()

    assert len(forwarded) == 1
    assert "Authorization" not in forwarded[0].headers
    assert MEMORY_CONTROL_AUTHORIZED_FIELD not in json.loads(forwarded[0].content)

    controlled = data_proxy_client.DataProxyClient(
        "http://proxy",
        memory_control_api_key=_MEMORY_CONTROL_API_KEY,
    )
    await controlled._http.aclose()
    controlled._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await controlled.turn(
            "pinned",
            "hello",
            memory_control_authorized=True,
            memory_assignment_pin=MemoryAssignmentPinWireV1.from_wire(_wire()),
        )
    finally:
        await controlled.close()

    assert len(forwarded) == 2
    assert (
        forwarded[1].headers["Authorization"]
        == (_MEMORY_CONTROL_HEADERS["Authorization"])
    )
    controlled_body = json.loads(forwarded[1].content)
    assert controlled_body[MEMORY_CONTROL_AUTHORIZED_FIELD] is True
    assert controlled_body[MEMORY_ASSIGNMENT_PIN_FIELD] == _wire()


@pytest.mark.asyncio
async def test_data_proxy_client_rejects_ambiguous_key_before_network_io() -> None:
    forwarded = []

    def handler(request):
        forwarded.append(request)
        return httpx.Response(500, request=request)

    client = data_proxy_client.DataProxyClient("http://proxy")
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ValueError):
            await client.turn("s%252Fb", "hello")
        with pytest.raises(ValueError):
            await client.close_session("s/b")
        with pytest.raises(ValueError):
            await client.get_history("s#fragment")
    finally:
        await client.close()

    assert forwarded == []


@pytest.mark.asyncio
async def test_data_proxy_client_can_close_on_old_proxy_during_upgrade() -> None:
    forwarded = []

    def handler(request):
        forwarded.append(request)
        if request.url.path == "/sessions/close":
            return httpx.Response(404, request=request)
        if request.url.path == "/session/chat:model:user/close":
            return httpx.Response(200, json={"status": "ok"}, request=request)
        raise AssertionError(f"unexpected request: {request.url}")

    client = data_proxy_client.DataProxyClient("http://proxy")
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await client.close_session("chat:model:user")
    finally:
        await client.close()

    assert [request.url.path for request in forwarded] == [
        "/sessions/close",
        "/session/chat:model:user/close",
    ]
    assert json.loads(forwarded[0].content) == {"session_key": "chat:model:user"}
    assert forwarded[1].content == b""


@pytest.mark.asyncio
async def test_controlled_data_proxy_client_authenticates_both_close_routes() -> None:
    forwarded = []

    def handler(request):
        forwarded.append(request)
        if request.url.path == "/sessions/close":
            return httpx.Response(404, request=request)
        if request.url.path == "/session/chat:model:user/close":
            return httpx.Response(200, json={"status": "ok"}, request=request)
        raise AssertionError(f"unexpected request: {request.url}")

    client = data_proxy_client.DataProxyClient(
        "http://proxy",
        memory_control_api_key=_MEMORY_CONTROL_API_KEY,
    )
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await client.close_session("chat:model:user")
    finally:
        await client.close()

    assert [request.url.path for request in forwarded] == [
        "/sessions/close",
        "/session/chat:model:user/close",
    ]
    assert {request.headers["Authorization"] for request in forwarded} == {
        _MEMORY_CONTROL_HEADERS["Authorization"]
    }


@pytest.mark.asyncio
async def test_controlled_data_proxy_client_authenticates_history_read() -> None:
    forwarded = []
    expected_history = [{"role": "user", "content": "hello"}]

    def handler(request):
        forwarded.append(request)
        return httpx.Response(
            200,
            json={"history": expected_history},
            request=request,
        )

    client = data_proxy_client.DataProxyClient(
        "http://proxy",
        memory_control_api_key=_MEMORY_CONTROL_API_KEY,
    )
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        history = await client.get_history("chat:model:user")
    finally:
        await client.close()

    assert history == expected_history
    assert len(forwarded) == 1
    assert forwarded[0].method == "GET"
    assert forwarded[0].url.path == "/session/chat:model:user/history"
    assert (
        forwarded[0].headers["Authorization"]
        == _MEMORY_CONTROL_HEADERS["Authorization"]
    )


@pytest.mark.asyncio
async def test_conflicting_pin_and_reserved_metadata_fail_before_worker() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            first = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {"message": "first", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a")}
                ),
            )
            conflict = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {"message": "second", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("b")}
                ),
            )
            forged = await client.post(
                "/session/s2/turn",
                json={
                    "message": "forged",
                    "metadata": {AREAL_MEMORY_METADATA_KEY: {"exposure": True}},
                },
            )
            nested_transport = await client.post(
                "/session/s3/turn",
                json={
                    "message": "nested",
                    "metadata": {MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a")},
                },
            )
            forged_inference = await client.post(
                "/session/s4/turn",
                json={
                    "message": "forged inference",
                    "metadata": {
                        AREAL_INFERENCE_METADATA_KEY: {
                            "base_url": "http://attacker",
                            "api_key": "stolen",
                            "model": "",
                        }
                    },
                },
            )
            forged_chat = await client.post(
                "/session/s5/turn",
                json={
                    "message": "forged chat",
                    "metadata": {CHAT_REQUEST_METADATA_KEY: {"model": "attacker"}},
                },
            )

    assert first.status_code == 200
    assert conflict.status_code == 409
    assert forged.status_code == 400
    assert nested_transport.status_code == 400
    assert forged_inference.status_code == forged_chat.status_code == 400
    assert len(worker_bodies) == 1


@pytest.mark.asyncio
async def test_concurrent_first_pin_is_compare_and_set() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            responses = await asyncio.gather(
                client.post(
                    "/session/s1/turn",
                    json=_authorized(
                        {"message": "a", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a")}
                    ),
                ),
                client.post(
                    "/session/s1/turn",
                    json=_authorized(
                        {"message": "b", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("b")}
                    ),
                ),
            )

    assert {response.status_code for response in responses} == {200, 409}
    assert len(worker_bodies) == 1


@pytest.mark.asyncio
async def test_concurrent_first_turn_atomically_selects_one_security_mode() -> None:
    worker_bodies: list[dict] = []
    ordinary_worker_request_started = asyncio.Event()
    release_ordinary_worker_request = asyncio.Event()
    original_send = httpx.AsyncClient.send

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path != "/run":
            raise AssertionError(f"unexpected worker request: {request.url}")
        worker_bodies.append(json.loads(request.content))
        if len(worker_bodies) == 1:
            ordinary_worker_request_started.set()
            await release_ordinary_worker_request.wait()
        return httpx.Response(
            200,
            json={"summary": "ok", "events": [], "metadata": {}},
            request=request,
        )

    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(httpx.AsyncClient, "send", patched_send):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            ordinary = asyncio.create_task(
                client.post(
                    "/session/race/turn",
                    json={"message": "ordinary fixes the session mode"},
                )
            )
            memory = None
            try:
                await asyncio.wait_for(
                    ordinary_worker_request_started.wait(), timeout=1
                )
                assert len(worker_bodies) == 1
                assert (
                    parse_memory_assignment_pin_metadata(worker_bodies[0]["metadata"])
                    is None
                )

                memory = asyncio.create_task(
                    client.post(
                        "/session/race/turn",
                        json=_authorized(
                            {
                                "message": "Memory cannot replace ordinary",
                                MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a"),
                            }
                        ),
                    )
                )
                rejected = await asyncio.wait_for(asyncio.shield(memory), timeout=1)

                # The ordinary request is still blocked inside Worker I/O, but
                # its mode stamp is already visible to the overlapping Memory
                # request.  The rejection must not reach Worker.
                assert not ordinary.done()
                assert rejected.status_code == 409
                assert len(worker_bodies) == 1

                release_ordinary_worker_request.set()
                accepted = await asyncio.wait_for(ordinary, timeout=1)
                assert accepted.status_code == 200

                follow_up = await client.post(
                    "/session/race/turn",
                    json={"message": "reuse ordinary"},
                )
            finally:
                release_ordinary_worker_request.set()
                requests = [task for task in (ordinary, memory) if task is not None]
                for task in requests:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*requests, return_exceptions=True)

    assert follow_up.status_code == 200
    assert len(worker_bodies) == 2
    assert parse_memory_assignment_pin_metadata(worker_bodies[1]["metadata"]) is None
    assert app.state.memory_pin_cache.resolve("race") is None


@pytest.mark.asyncio
async def test_close_clears_pin_and_allows_new_incarnation() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            first = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {"message": "a", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a")}
                ),
            )
            closed = await client.post(
                "/sessions/close",
                json={"session_key": "s1"},
            )
            replacement = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {"message": "b", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("b")}
                ),
            )

    assert first.status_code == closed.status_code == replacement.status_code == 200
    assert worker_closes == ["s1"]
    pins = [
        parse_memory_assignment_pin_metadata(body["metadata"]) for body in worker_bodies
    ]
    assert pins == [
        MemoryAssignmentPinWireV1.from_wire(_wire("a")).to_runtime_pin(),
        MemoryAssignmentPinWireV1.from_wire(_wire("b")).to_runtime_pin(),
    ]


@pytest.mark.asyncio
async def test_memory_capable_data_proxy_authenticates_close_before_state_oracle() -> (
    None
):
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
        ) as client:
            ordinary = await client.post(
                "/session/ordinary/turn",
                json={"message": "ordinary"},
            )
            memory = await client.post(
                "/session/memory/turn",
                headers=_MEMORY_CONTROL_HEADERS,
                json=_authorized(
                    {"message": "memory", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a")}
                ),
            )

            denied = [
                await client.post(
                    "/sessions/close",
                    json={"session_key": "unknown"},
                ),
                await client.post(
                    "/session/ordinary/close",
                    headers=admin_headers("wrong-memory-control-key"),
                ),
                await client.post(
                    "/sessions/close",
                    headers=admin_headers(_EXTERNAL_ADMIN_API_KEY),
                    json={"session_key": "memory"},
                ),
                await client.post(
                    "/sessions/close",
                    json={"session_key": "s%252Fb"},
                ),
            ]

            ordinary_retry = await client.post(
                "/session/ordinary/turn",
                json={"message": "still ordinary"},
            )
            memory_retry = await client.post(
                "/session/memory/turn",
                headers=_MEMORY_CONTROL_HEADERS,
                json=_authorized({"message": "same pin"}),
            )
            authorized_close = await client.post(
                "/sessions/close",
                headers=_MEMORY_CONTROL_HEADERS,
                json={"session_key": "memory"},
            )
            replacement = await client.post(
                "/session/memory/turn",
                json={"message": "new ordinary incarnation"},
            )
            authorized_legacy_close = await client.post(
                "/session/ordinary/close",
                headers=_MEMORY_CONTROL_HEADERS,
            )

    assert ordinary.status_code == memory.status_code == 200
    assert [response.status_code for response in denied] == [401, 401, 401, 401]
    assert ordinary_retry.status_code == memory_retry.status_code == 200
    assert authorized_close.status_code == replacement.status_code == 200
    assert authorized_legacy_close.status_code == 200
    assert worker_closes == ["memory", "ordinary"]
    assert app.state.memory_pin_cache.resolve("memory") is None


@pytest.mark.asyncio
async def test_memory_capable_data_proxy_authenticates_history_before_state_oracle() -> (
    None
):
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
        ) as client:
            ordinary = await client.post(
                "/session/ordinary/turn",
                json={"message": "ordinary"},
            )
            memory = await client.post(
                "/session/memory/turn",
                headers=_MEMORY_CONTROL_HEADERS,
                json=_authorized(
                    {"message": "memory", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a")}
                ),
            )

            denied = []
            for target in ("ordinary", "memory", "unknown", "s%252Fb"):
                for headers in (
                    None,
                    admin_headers("wrong-memory-control-key"),
                    admin_headers(_EXTERNAL_ADMIN_API_KEY),
                ):
                    denied.append(
                        await client.get(
                            f"/session/{target}/history",
                            headers=headers,
                        )
                    )

            ordinary_history = await client.get(
                "/session/ordinary/history",
                headers=_MEMORY_CONTROL_HEADERS,
            )
            memory_history = await client.get(
                "/session/memory/history",
                headers=_MEMORY_CONTROL_HEADERS,
            )
            unknown_history = await client.get(
                "/session/unknown/history",
                headers=_MEMORY_CONTROL_HEADERS,
            )
            invalid_key = await client.get(
                "/session/s%252Fb/history",
                headers=_MEMORY_CONTROL_HEADERS,
            )

    assert ordinary.status_code == memory.status_code == 200
    assert [(response.status_code, response.json()) for response in denied] == [
        (401, {"detail": "Invalid admin key"})
    ] * 12
    assert ordinary_history.status_code == memory_history.status_code == 200
    assert ordinary_history.json() == {
        "history": [
            {"role": "user", "content": "ordinary"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    assert memory_history.json() == {
        "history": [
            {"role": "user", "content": "memory"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    assert unknown_history.status_code == 200
    assert unknown_history.json() == {"history": []}
    assert invalid_key.status_code == 400
    assert worker_closes == []


@pytest.mark.asyncio
async def test_standalone_data_proxy_preserves_anonymous_close_compatibility() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = data_proxy_app.create_data_proxy_app(
        data_proxy_config.DataProxyConfig(worker_addr="http://worker")
    )
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
        ) as client:
            first = await client.post(
                "/session/s1/turn",
                json={"message": "first"},
            )
            fixed_close = await client.post(
                "/sessions/close",
                json={"session_key": "s1"},
            )
            replacement = await client.post(
                "/session/s1/turn",
                json={"message": "replacement"},
            )
            legacy_close = await client.post("/session/s1/close")

    assert first.status_code == fixed_close.status_code == 200
    assert replacement.status_code == legacy_close.status_code == 200
    assert worker_closes == ["s1", "s1"]


@pytest.mark.asyncio
async def test_standalone_data_proxy_preserves_anonymous_history_compatibility() -> (
    None
):
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = data_proxy_app.create_data_proxy_app(
        data_proxy_config.DataProxyConfig(worker_addr="http://worker")
    )
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
        ) as client:
            turn = await client.post(
                "/session/s1/turn",
                json={"message": "standalone"},
            )
            history = await client.get("/session/s1/history")
            unknown = await client.get("/session/unknown/history")

    assert turn.status_code == history.status_code == unknown.status_code == 200
    assert history.json() == {
        "history": [
            {"role": "user", "content": "standalone"},
            {"role": "assistant", "content": "ok"},
        ]
    }
    assert unknown.json() == {"history": []}
    assert worker_closes == []


@pytest.mark.asyncio
async def test_memory_close_allows_ordinary_new_incarnation() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            memory_turn = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {"message": "Memory", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a")}
                ),
            )
            closed = await client.post(
                "/sessions/close",
                json={"session_key": "s1"},
            )
            ordinary_turn = await client.post(
                "/session/s1/turn",
                json={"message": "ordinary new incarnation"},
            )
            rejected_upgrade = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {
                        "message": "cannot silently restore Memory",
                        MEMORY_ASSIGNMENT_PIN_FIELD: _wire("b"),
                    }
                ),
            )

    assert (
        memory_turn.status_code
        == closed.status_code
        == ordinary_turn.status_code
        == 200
    )
    assert rejected_upgrade.status_code == 409
    assert worker_closes == ["s1"]
    assert len(worker_bodies) == 2
    assert parse_memory_assignment_pin_metadata(worker_bodies[0]["metadata"]) == (
        MemoryAssignmentPinWireV1.from_wire(_wire("a")).to_runtime_pin()
    )
    assert AREAL_MEMORY_METADATA_KEY not in worker_bodies[1]["metadata"]
    assert app.state.memory_pin_cache.resolve("s1") is None


@pytest.mark.asyncio
async def test_canonical_session_identity_is_exact_at_worker_run_and_close() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            turn = await client.post(
                "/session/chat:model:user/turn",
                json={"message": "hello"},
            )
            closed = await client.post(
                "/sessions/close",
                json={"session_key": "chat:model:user"},
            )

    assert turn.status_code == closed.status_code == 200
    assert [body["session_key"] for body in worker_bodies] == ["chat:model:user"]
    assert worker_closes == ["chat:model:user"]


@pytest.mark.asyncio
async def test_new_data_proxy_can_close_session_on_old_worker_during_upgrade() -> None:
    worker_run_keys: list[str] = []
    worker_close_paths: list[str] = []
    original_send = httpx.AsyncClient.send

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path == "/run":
            worker_run_keys.append(json.loads(request.content)["session_key"])
            return httpx.Response(
                200,
                json={"summary": "ok", "events": [], "metadata": {}},
                request=request,
            )
        if request.url.path == "/sessions/close":
            worker_close_paths.append(request.url.path)
            return httpx.Response(
                404,
                json={"detail": "Not Found"},
                request=request,
            )
        if request.url.path == "/session/chat:model:user/close":
            worker_close_paths.append(request.url.path)
            return httpx.Response(200, json={"status": "ok"}, request=request)
        raise AssertionError(f"unexpected worker request: {request.url}")

    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(httpx.AsyncClient, "send", patched_send):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            first = await client.post(
                "/session/chat:model:user/turn",
                json={"message": "first"},
            )
            closed = await client.post(
                "/sessions/close",
                json={"session_key": "chat:model:user"},
            )
            replacement = await client.post(
                "/session/chat:model:user/turn",
                json={"message": "replacement incarnation"},
            )

    assert first.status_code == closed.status_code == replacement.status_code == 200
    assert worker_run_keys == ["chat:model:user", "chat:model:user"]
    assert worker_close_paths == [
        "/sessions/close",
        "/session/chat:model:user/close",
    ]


@pytest.mark.asyncio
async def test_shutdown_closes_pinned_worker_session_before_clearing_pin() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            response = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {"message": "first", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a")}
                ),
            )
        for shutdown_handler in app.router.on_shutdown:
            await shutdown_handler()

    assert response.status_code == 200
    assert worker_closes == ["s1"]
    assert app.state.memory_pin_cache.resolve("s1") is None


@pytest.mark.asyncio
async def test_idle_reaper_closes_sessions_concurrently_without_duplicate_tasks() -> (
    None
):
    worker_close_keys: list[str] = []
    close_started = {"s1": asyncio.Event(), "s2": asyncio.Event()}
    release_close = {"s1": asyncio.Event(), "s2": asyncio.Event()}
    never_resume = asyncio.Event()
    sleep_calls = 0
    original_send = httpx.AsyncClient.send
    real_sleep = asyncio.sleep

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path == "/run":
            return httpx.Response(
                200,
                json={"summary": "ok", "events": [], "metadata": {}},
                request=request,
            )
        if request.url.path == "/sessions/close":
            session_key = _worker_close_session_key(request)
            worker_close_keys.append(session_key)
            close_started[session_key].set()
            await release_close[session_key].wait()
            return httpx.Response(
                200, json=_worker_close_receipt(request), request=request
            )
        raise AssertionError(f"unexpected worker request: {request.url}")

    async def run_one_scan(delay: float) -> None:
        nonlocal sleep_calls
        assert delay == 60
        sleep_calls += 1
        if sleep_calls == 1:
            return
        await never_resume.wait()

    app = data_proxy_app.create_data_proxy_app(
        data_proxy_config.DataProxyConfig(
            worker_addr="http://worker",
            session_timeout=0,
        )
    )
    transport = httpx.ASGITransport(app=app)
    with patch.object(httpx.AsyncClient, "send", patched_send):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
        ) as client:
            for session_key in ("s1", "s2"):
                turn = await client.post(
                    f"/session/{session_key}/turn",
                    json={"message": session_key},
                )
                assert turn.status_code == 200

            with patch.object(data_proxy_app.asyncio, "sleep", run_one_scan):
                for startup_handler in app.router.on_startup:
                    await startup_handler()
                shutdown_task = None
                try:
                    # Both Worker closes must start even though the first one
                    # remains blocked.  The old serial loop times out waiting
                    # for s2 here.
                    await asyncio.wait_for(close_started["s1"].wait(), timeout=1)
                    await asyncio.wait_for(close_started["s2"].wait(), timeout=1)

                    # Join the exact task already installed by the reaper,
                    # then let only s2 complete.  This also proves that a
                    # concurrent explicit close does not duplicate Worker I/O.
                    s2_waiter = asyncio.create_task(
                        client.post(
                            "/sessions/close",
                            json={"session_key": "s2"},
                        )
                    )
                    await real_sleep(0)
                    release_close["s2"].set()
                    s2_closed = await asyncio.wait_for(s2_waiter, timeout=1)
                    assert s2_closed.status_code == 200
                    assert worker_close_keys.count("s2") == 1
                    assert not release_close["s1"].is_set()
                    health_while_s1_blocked = await client.get("/health")
                    assert health_while_s1_blocked.json()["active_sessions"] == 1

                    async def run_shutdown() -> None:
                        for shutdown_handler in app.router.on_shutdown:
                            await shutdown_handler()

                    # Shutdown cancels only the scanning coroutine.  The
                    # shielded exact s1 close must keep running and shutdown
                    # must join it instead of declaring cleanup complete.
                    shutdown_task = asyncio.create_task(run_shutdown())
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(app.state.reaper_task, timeout=1)
                    assert not shutdown_task.done()
                    assert worker_close_keys.count("s1") == 1

                    release_close["s1"].set()
                    await asyncio.wait_for(shutdown_task, timeout=1)
                    health_after_scan = await client.get("/health")
                    assert health_after_scan.json()["active_sessions"] == 0
                    assert sorted(worker_close_keys) == ["s1", "s2"]
                finally:
                    release_close["s1"].set()
                    release_close["s2"].set()
                    if shutdown_task is None:
                        app.state.reaper_task.cancel()
                        await asyncio.gather(
                            app.state.reaper_task,
                            return_exceptions=True,
                        )
                        for shutdown_handler in app.router.on_shutdown:
                            await shutdown_handler()
                    else:
                        await asyncio.gather(shutdown_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_idle_reaper_skips_session_removed_after_scan_snapshot() -> None:
    worker_close_keys: list[str] = []
    created_sessions: list[data_proxy_app._SessionData] = []
    first_begin_blocked = asyncio.Event()
    next_scan_started = asyncio.Event()
    never_resume = asyncio.Event()
    sleep_calls = 0
    original_send = httpx.AsyncClient.send
    original_session_data = data_proxy_app._SessionData

    class ObservedLock(asyncio.Lock):
        async def acquire(self) -> bool:
            if self.locked():
                first_begin_blocked.set()
            return await super().acquire()

    def tracking_session_data() -> data_proxy_app._SessionData:
        session = original_session_data()
        created_sessions.append(session)
        return session

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path == "/run":
            return httpx.Response(
                200,
                json={"summary": "ok", "events": [], "metadata": {}},
                request=request,
            )
        if request.url.path == "/sessions/close":
            worker_close_keys.append(_worker_close_session_key(request))
            return httpx.Response(
                200, json=_worker_close_receipt(request), request=request
            )
        raise AssertionError(f"unexpected worker request: {request.url}")

    async def run_one_scan(delay: float) -> None:
        nonlocal sleep_calls
        assert delay == 60
        sleep_calls += 1
        if sleep_calls == 1:
            return
        next_scan_started.set()
        await never_resume.wait()

    with (
        patch.object(data_proxy_app, "_SessionData", tracking_session_data),
        patch.object(httpx.AsyncClient, "send", patched_send),
    ):
        app = data_proxy_app.create_data_proxy_app(
            data_proxy_config.DataProxyConfig(
                worker_addr="http://worker",
                session_timeout=0,
            )
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
        ) as client:
            for session_key in ("s1", "s2"):
                turn = await client.post(
                    f"/session/{session_key}/turn",
                    json={"message": session_key},
                )
                assert turn.status_code == 200

            assert len(created_sessions) == 2
            observed_lock = ObservedLock()
            await observed_lock.acquire()
            created_sessions[0].lifecycle_lock = observed_lock

            with patch.object(data_proxy_app.asyncio, "sleep", run_one_scan):
                for startup_handler in app.router.on_startup:
                    await startup_handler()
                try:
                    # The scan has snapshotted both keys and is blocked before
                    # beginning s1.  Retire s2 through an independent close so
                    # the later snapshot entry is stale when the scan reaches
                    # it.
                    await asyncio.wait_for(first_begin_blocked.wait(), timeout=1)
                    closed = await client.post(
                        "/sessions/close",
                        json={"session_key": "s2"},
                    )
                    assert closed.status_code == 200
                    health_after_s2 = await client.get("/health")
                    assert health_after_s2.json()["active_sessions"] == 1

                    observed_lock.release()
                    await asyncio.wait_for(next_scan_started.wait(), timeout=1)

                    # A stale idle-only lookup must neither reconstruct a
                    # dummy tombstone nor issue a second Worker close for s2.
                    assert len(created_sessions) == 2
                    assert worker_close_keys.count("s2") == 1
                    assert worker_close_keys.count("s1") == 1
                    health_after_scan = await client.get("/health")
                    assert health_after_scan.json()["active_sessions"] == 0
                finally:
                    if observed_lock.locked():
                        observed_lock.release()
                    app.state.reaper_task.cancel()
                    await asyncio.gather(
                        app.state.reaper_task,
                        return_exceptions=True,
                    )
                    for shutdown_handler in app.router.on_shutdown:
                        await shutdown_handler()


@pytest.mark.asyncio
async def test_idle_reaper_isolates_worker_close_failure_and_preserves_retry() -> None:
    worker_close_keys: list[str] = []
    s1_attempts = 0
    next_scan_started = asyncio.Event()
    never_resume = asyncio.Event()
    sleep_calls = 0
    original_send = httpx.AsyncClient.send

    async def patched_send(self, request, **kwargs):
        nonlocal s1_attempts
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path == "/run":
            return httpx.Response(
                200,
                json={"summary": "ok", "events": [], "metadata": {}},
                request=request,
            )
        if request.url.path == "/sessions/close":
            session_key = _worker_close_session_key(request)
            worker_close_keys.append(session_key)
            if session_key == "s1":
                s1_attempts += 1
                if s1_attempts == 1:
                    raise RuntimeError("injected Worker close failure")
            return httpx.Response(
                200, json=_worker_close_receipt(request), request=request
            )
        raise AssertionError(f"unexpected worker request: {request.url}")

    async def run_one_scan(delay: float) -> None:
        nonlocal sleep_calls
        assert delay == 60
        sleep_calls += 1
        if sleep_calls == 1:
            return
        next_scan_started.set()
        await never_resume.wait()

    app = data_proxy_app.create_data_proxy_app(
        data_proxy_config.DataProxyConfig(
            worker_addr="http://worker",
            session_timeout=0,
        )
    )
    transport = httpx.ASGITransport(app=app)
    with patch.object(httpx.AsyncClient, "send", patched_send):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
        ) as client:
            for session_key in ("s1", "s2"):
                turn = await client.post(
                    f"/session/{session_key}/turn",
                    json={"message": session_key},
                )
                assert turn.status_code == 200

            with (
                patch.object(data_proxy_app.asyncio, "sleep", run_one_scan),
                patch.object(data_proxy_app.logger, "info") as log_info,
            ):
                for startup_handler in app.router.on_startup:
                    await startup_handler()
                try:
                    await asyncio.wait_for(next_scan_started.wait(), timeout=1)

                    # s2 succeeds in the same scan even though s1 raises.  The
                    # failed s1 incarnation remains a closing tombstone until
                    # an explicit retry confirms Worker cleanup.
                    assert sorted(worker_close_keys) == ["s1", "s2"]
                    health_after_scan = await client.get("/health")
                    assert health_after_scan.json()["active_sessions"] == 1
                    blocked = await client.post(
                        "/session/s1/turn",
                        json={"message": "must remain tombstoned"},
                    )
                    assert blocked.status_code == 409
                    log_info.assert_called_once_with("Reaped %d idle sessions", 1)

                    retried = await client.post(
                        "/sessions/close",
                        json={"session_key": "s1"},
                    )
                    assert retried.status_code == 200
                    assert worker_close_keys.count("s1") == 2
                    assert worker_close_keys.count("s2") == 1
                    health_after_retry = await client.get("/health")
                    assert health_after_retry.json()["active_sessions"] == 0
                finally:
                    app.state.reaper_task.cancel()
                    await asyncio.gather(
                        app.state.reaper_task,
                        return_exceptions=True,
                    )

        for shutdown_handler in app.router.on_shutdown:
            await shutdown_handler()


@pytest.mark.asyncio
async def test_malformed_pin_fails_without_reserving_session() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    malformed = _wire("a")
    malformed["exposure"] = {"status": "delivered"}
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            rejected = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {"message": "bad", MEMORY_ASSIGNMENT_PIN_FIELD: malformed}
                ),
            )
            health_after_rejection = await client.get("/health")
            accepted = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {"message": "good", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("b")}
                ),
            )

    assert rejected.status_code == 400
    assert health_after_rejection.json()["active_sessions"] == 0
    assert accepted.status_code == 200
    assert len(worker_bodies) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", (None, False, 0, [], {}))
async def test_falsy_non_string_inference_fields_fail_before_session_mutation(
    bad_value,
) -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            rejected = await client.post(
                "/session/s1/turn",
                json={
                    "message": "bad routing",
                    "inf_base_url": bad_value,
                    "session_api_key": "sk-sess-valid",
                },
            )
            health = await client.get("/health")

    assert rejected.status_code == 400
    assert health.json()["active_sessions"] == 0
    assert worker_bodies == []


@pytest.mark.asyncio
async def test_blank_session_key_fails_without_reserving_session() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            rejected = await client.post(
                "/session/%20/turn",
                json={"message": "blank"},
            )
            health = await client.get("/health")

    assert rejected.status_code == 400
    assert health.json()["active_sessions"] == 0
    assert worker_bodies == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoded_key",
    (
        "s%252Fb",
        "s%25252Fb",
        "s%23fragment",
        "s%3Fquery",
        "s%5Cb",
        "s%00b",
    ),
)
async def test_ambiguous_session_key_fails_before_worker_or_state_mutation(
    encoded_key: str,
) -> None:
    """Recursive URL decoding must not change a session's identity by hop.

    The request path is decoded once by the DataProxy.  Before this regression
    was fixed, the resulting key still contained ``%25`` and was accepted into
    local state; interpolating it into the Worker close URL decoded it again,
    so ``run`` and ``close_session`` could address different agent sessions.
    """

    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            rejected = await client.post(
                f"/session/{encoded_key}/turn",
                json={"message": "must not reserve an ambiguous identity"},
            )
            health = await client.get("/health")

    assert rejected.status_code == 400
    assert health.json()["active_sessions"] == 0
    assert worker_bodies == []
    assert worker_closes == []


@pytest.mark.asyncio
async def test_fixed_close_rejects_ambiguous_key_before_tombstone_or_worker() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            rejected = await client.post(
                "/sessions/close",
                json={"session_key": "s%252Fb"},
            )
            health = await client.get("/health")

    assert rejected.status_code == 400
    assert health.json()["active_sessions"] == 0
    assert worker_bodies == []
    assert worker_closes == []


@pytest.mark.asyncio
async def test_conflicting_pin_cannot_mutate_cached_inference_routing() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(
        httpx.AsyncClient,
        "send",
        _patched_worker_send(worker_bodies, worker_closes),
    ):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            first = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {
                        "message": "first",
                        MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a"),
                        "inf_base_url": "http://trusted-inference",
                        "session_api_key": "sk-sess-trusted",
                    }
                ),
            )
            conflict = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {
                        "message": "conflict",
                        MEMORY_ASSIGNMENT_PIN_FIELD: _wire("b"),
                        "inf_base_url": "http://attacker-inference",
                        "session_api_key": "sk-sess-attacker",
                    }
                ),
            )
            retry = await client.post(
                "/session/s1/turn",
                json=_authorized({"message": "retry"}),
            )

    assert first.status_code == retry.status_code == 200
    assert conflict.status_code == 409
    assert len(worker_bodies) == 2
    assert worker_bodies[1]["metadata"]["areal_inference"] == {
        "base_url": "http://trusted-inference",
        "api_key": "sk-sess-trusted",
        "model": "",
    }


@pytest.mark.asyncio
async def test_close_tombstone_prevents_rebind_before_worker_cleanup() -> None:
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    original_send = httpx.AsyncClient.send

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path == "/run":
            worker_bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"summary": "ok", "events": [], "metadata": {}},
                request=request,
            )
        if request.url.path == "/sessions/close":
            worker_closes.append(_worker_close_session_key(request))
            close_started.set()
            await release_close.wait()
            return httpx.Response(
                200, json=_worker_close_receipt(request), request=request
            )
        raise AssertionError(f"unexpected worker request: {request.url}")

    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(httpx.AsyncClient, "send", patched_send):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            first = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {"message": "first", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a")}
                ),
            )
            closing = asyncio.create_task(client.post("/session/s1/close"))
            await asyncio.wait_for(close_started.wait(), timeout=2)
            during_close = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {"message": "too early", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("b")}
                ),
            )
            release_close.set()
            closed = await closing
            replacement = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {
                        "message": "replacement",
                        MEMORY_ASSIGNMENT_PIN_FIELD: _wire("b"),
                    }
                ),
            )

    assert first.status_code == closed.status_code == replacement.status_code == 200
    assert during_close.status_code == 409
    assert len(worker_bodies) == 2
    assert worker_closes == ["s1"]
    assert parse_memory_assignment_pin_metadata(worker_bodies[0]["metadata"]) != (
        parse_memory_assignment_pin_metadata(worker_bodies[1]["metadata"])
    )


@pytest.mark.asyncio
async def test_stream_body_holds_turn_lease_until_iterator_finishes() -> None:
    stream_release = asyncio.Event()
    worker_closes: list[str] = []
    original_send = httpx.AsyncClient.send

    class BlockingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"first"
            await stream_release.wait()
            yield b"second"

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path == "/run":
            return httpx.Response(
                200,
                headers={"x-areal-passthrough": "1"},
                stream=BlockingStream(),
                request=request,
            )
        if request.url.path == "/sessions/close":
            worker_closes.append(_worker_close_session_key(request))
            return httpx.Response(
                200, json=_worker_close_receipt(request), request=request
            )
        raise AssertionError(f"unexpected worker request: {request.url}")

    app = _proxy_app()
    turn_endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/session/{session_key}/turn"
    )
    close_endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/session/{session_key}/close"
    )
    with patch.object(httpx.AsyncClient, "send", patched_send):
        response = await turn_endpoint(
            "s1",
            _authorized({"message": "stream", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a")}),
            _internal_request(),
        )
        close_task = asyncio.create_task(close_endpoint("s1", _internal_request()))
        await asyncio.sleep(0)
        assert worker_closes == []

        iterator = response.body_iterator.__aiter__()
        assert await anext(iterator) == b"first"
        assert worker_closes == []
        stream_release.set()
        assert await anext(iterator) == b"second"
        with pytest.raises(StopAsyncIteration):
            await anext(iterator)
        assert await close_task == {"status": "ok"}

    assert worker_closes == ["s1"]


@pytest.mark.asyncio
async def test_stream_close_before_first_byte_releases_turn_lease() -> None:
    upstream_closed = asyncio.Event()
    worker_closes: list[str] = []
    original_send = httpx.AsyncClient.send

    class NeverStartedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.Event().wait()
            yield b"unreachable"

        async def aclose(self):
            upstream_closed.set()

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path == "/run":
            return httpx.Response(
                200,
                headers={"x-areal-passthrough": "1"},
                stream=NeverStartedStream(),
                request=request,
            )
        if request.url.path == "/sessions/close":
            worker_closes.append(_worker_close_session_key(request))
            return httpx.Response(
                200, json=_worker_close_receipt(request), request=request
            )
        raise AssertionError(f"unexpected worker request: {request.url}")

    app = _proxy_app()
    turn_endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/session/{session_key}/turn"
    )
    close_endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/session/{session_key}/close"
    )
    with patch.object(httpx.AsyncClient, "send", patched_send):
        response = await turn_endpoint(
            "s1",
            _authorized({"message": "stream", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a")}),
            _internal_request(),
        )
        await response.body_iterator.aclose()
        await asyncio.wait_for(upstream_closed.wait(), timeout=1)
        assert await asyncio.wait_for(
            close_endpoint("s1", _internal_request()), timeout=1
        ) == {"status": "ok"}

    assert worker_closes == ["s1"]


@pytest.mark.asyncio
async def test_stream_close_failure_cannot_leak_turn_lease() -> None:
    worker_closes: list[str] = []
    original_send = httpx.AsyncClient.send

    class FailingCloseStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"first"
            await asyncio.Event().wait()

        async def aclose(self):
            raise RuntimeError("injected upstream close failure")

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path == "/run":
            return httpx.Response(
                200,
                headers={"x-areal-passthrough": "1"},
                stream=FailingCloseStream(),
                request=request,
            )
        if request.url.path == "/sessions/close":
            worker_closes.append(_worker_close_session_key(request))
            return httpx.Response(
                200, json=_worker_close_receipt(request), request=request
            )
        raise AssertionError(f"unexpected worker request: {request.url}")

    app = _proxy_app()
    turn_endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/session/{session_key}/turn"
    )
    close_endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/session/{session_key}/close"
    )
    with patch.object(httpx.AsyncClient, "send", patched_send):
        response = await turn_endpoint(
            "s1",
            _authorized({"message": "stream", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a")}),
            _internal_request(),
        )
        iterator = response.body_iterator.__aiter__()
        assert await anext(iterator) == b"first"
        with pytest.raises(RuntimeError, match="injected upstream close failure"):
            await iterator.aclose()

        assert await asyncio.wait_for(
            close_endpoint("s1", _internal_request()), timeout=1
        ) == {"status": "ok"}

    assert worker_closes == ["s1"]


@pytest.mark.asyncio
async def test_stream_cancellation_releases_lease_but_preserves_ordinary_mode() -> None:
    second_read_started = asyncio.Event()
    release_second_read = asyncio.Event()
    upstream_closed = asyncio.Event()
    worker_bodies: list[dict] = []
    worker_closes: list[str] = []
    original_send = httpx.AsyncClient.send

    class CancellableStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"first"
            second_read_started.set()
            await release_second_read.wait()
            yield b"unreachable"

        async def aclose(self):
            upstream_closed.set()

    async def patched_send(self, request, **kwargs):
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path == "/run":
            worker_bodies.append(json.loads(request.content))
            if len(worker_bodies) == 1:
                return httpx.Response(
                    200,
                    headers={"x-areal-passthrough": "1"},
                    stream=CancellableStream(),
                    request=request,
                )
            return httpx.Response(
                200,
                json={"summary": "ok", "events": [], "metadata": {}},
                request=request,
            )
        if request.url.path == "/sessions/close":
            worker_closes.append(_worker_close_session_key(request))
            return httpx.Response(
                200, json=_worker_close_receipt(request), request=request
            )
        raise AssertionError(f"unexpected worker request: {request.url}")

    app = _proxy_app()
    turn_endpoint = next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", None) == "/session/{session_key}/turn"
    )
    transport = httpx.ASGITransport(app=app)
    with patch.object(httpx.AsyncClient, "send", patched_send):
        blocked_read = None
        try:
            response = await turn_endpoint(
                "s1",
                {"message": "ordinary stream"},
                _internal_request(),
            )
            iterator = response.body_iterator.__aiter__()
            assert await anext(iterator) == b"first"

            blocked_read = asyncio.create_task(anext(iterator))
            await asyncio.wait_for(second_read_started.wait(), timeout=1)
            blocked_read.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(blocked_read, timeout=1)
            await asyncio.wait_for(upstream_closed.wait(), timeout=1)

            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://proxy",
                headers=_MEMORY_CONTROL_HEADERS,
            ) as client:
                rejected_upgrade = await client.post(
                    "/session/s1/turn",
                    json=_authorized(
                        {
                            "message": "stream cancellation is not a close",
                            MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a"),
                        }
                    ),
                )
                ordinary_retry = await client.post(
                    "/session/s1/turn",
                    json={"message": "ordinary retry"},
                )
                closed = await asyncio.wait_for(
                    client.post(
                        "/sessions/close",
                        json={"session_key": "s1"},
                    ),
                    timeout=1,
                )
        finally:
            release_second_read.set()
            if blocked_read is not None and not blocked_read.done():
                blocked_read.cancel()
            if blocked_read is not None:
                await asyncio.gather(blocked_read, return_exceptions=True)

    assert rejected_upgrade.status_code == 409
    assert ordinary_retry.status_code == closed.status_code == 200
    assert len(worker_bodies) == 2
    assert all(
        AREAL_MEMORY_METADATA_KEY not in body["metadata"] for body in worker_bodies
    )
    assert worker_closes == ["s1"]


@pytest.mark.asyncio
async def test_chat_downstream_close_survives_caller_cancellation() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    close_completed = asyncio.Event()

    class BlockingCloseStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"first"

        async def aclose(self):
            close_started.set()
            await allow_close.wait()
            close_completed.set()

    request = httpx.Request("POST", "http://proxy/session/s1/turn")
    upstream = httpx.Response(200, stream=BlockingCloseStream(), request=request)
    response = gateway_bridge.CleanupStreamingResponse(
        upstream.aiter_raw(),
        cleanup=upstream.aclose,
        cleanup_task_name="test-chat-downstream-cleanup",
    )
    closing = asyncio.create_task(response.body_iterator.aclose())
    await asyncio.wait_for(close_started.wait(), timeout=1)
    closing.cancel()
    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    await asyncio.wait_for(close_completed.wait(), timeout=1)


@pytest.mark.asyncio
async def test_failed_memory_close_retry_allows_ordinary() -> None:
    worker_bodies: list[dict] = []
    worker_close_keys: list[str] = []
    close_attempts = 0
    original_send = httpx.AsyncClient.send

    async def patched_send(self, request, **kwargs):
        nonlocal close_attempts
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path == "/run":
            worker_bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"summary": "ok", "events": [], "metadata": {}},
                request=request,
            )
        if request.url.path == "/sessions/close":
            close_attempts += 1
            worker_close_keys.append(_worker_close_session_key(request))
            return httpx.Response(
                503 if close_attempts == 1 else 200,
                json=(
                    {"status": "retry"}
                    if close_attempts == 1
                    else _worker_close_receipt(request)
                ),
                request=request,
            )
        raise AssertionError(f"unexpected worker request: {request.url}")

    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(httpx.AsyncClient, "send", patched_send):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            memory_turn = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {"message": "first", MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a")}
                ),
            )
            failed_close = await client.post(
                "/sessions/close",
                json={"session_key": "s1"},
            )
            blocked = await client.post(
                "/session/s1/turn",
                json={"message": "ordinary blocked by close tombstone"},
            )
            successful_close = await client.post(
                "/sessions/close",
                json={"session_key": "s1"},
            )
            replacement = await client.post(
                "/session/s1/turn",
                json={"message": "ordinary replacement"},
            )
            rejected_upgrade = await client.post(
                "/session/s1/turn",
                json=_authorized(
                    {
                        "message": "new incarnation remains ordinary",
                        MEMORY_ASSIGNMENT_PIN_FIELD: _wire("b"),
                    }
                ),
            )

    assert memory_turn.status_code == 200
    assert failed_close.status_code == 503
    assert blocked.status_code == 409
    assert successful_close.status_code == replacement.status_code == 200
    assert rejected_upgrade.status_code == 409
    assert close_attempts == 2
    assert worker_close_keys == ["s1", "s1"]
    assert len(worker_bodies) == 2
    assert parse_memory_assignment_pin_metadata(worker_bodies[0]["metadata"]) == (
        MemoryAssignmentPinWireV1.from_wire(_wire("a")).to_runtime_pin()
    )
    assert AREAL_MEMORY_METADATA_KEY not in worker_bodies[1]["metadata"]
    assert app.state.memory_pin_cache.resolve("s1") is None


@pytest.mark.asyncio
async def test_mismatched_worker_close_receipt_keeps_tombstone() -> None:
    worker_bodies: list[dict] = []
    worker_close_keys: list[str] = []
    close_attempts = 0
    original_send = httpx.AsyncClient.send

    async def patched_send(self, request, **kwargs):
        nonlocal close_attempts
        if request.url.host != "worker":
            return await original_send(self, request, **kwargs)
        if request.url.path == "/run":
            worker_bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"summary": "ok", "events": [], "metadata": {}},
                request=request,
            )
        if request.url.path == "/sessions/close":
            close_attempts += 1
            worker_close_keys.append(_worker_close_session_key(request))
            return httpx.Response(
                200,
                json=(
                    {
                        "status": "ok",
                        "session_key_sha256": session_key_sha256("other-session"),
                    }
                    if close_attempts == 1
                    else _worker_close_receipt(request)
                ),
                request=request,
            )
        raise AssertionError(f"unexpected worker request: {request.url}")

    app = _proxy_app()
    transport = httpx.ASGITransport(app=app)
    with patch.object(httpx.AsyncClient, "send", patched_send):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://proxy",
            headers=_MEMORY_CONTROL_HEADERS,
        ) as client:
            first = await client.post(
                "/session/s1/turn",
                json={"message": "first"},
            )
            rejected_close = await client.post(
                "/sessions/close",
                json={"session_key": "s1"},
            )
            blocked = await client.post(
                "/session/s1/turn",
                json={"message": "must remain closed"},
            )
            health = await client.get("/health")
            successful_close = await client.post(
                "/sessions/close",
                json={"session_key": "s1"},
            )
            replacement = await client.post(
                "/session/s1/turn",
                json={"message": "replacement incarnation"},
            )

    assert first.status_code == 200
    assert rejected_close.status_code == 503
    assert blocked.status_code == 409
    assert health.json()["active_sessions"] == 1
    assert successful_close.status_code == replacement.status_code == 200
    assert worker_close_keys == ["s1", "s1"]
    assert len(worker_bodies) == 2


@pytest.mark.asyncio
async def test_chat_bridge_separates_memory_control_from_upstream_request() -> None:
    forwarded: list[dict] = []
    routed = 0

    def handler(request):
        nonlocal routed
        if request.url.host == "router" and request.url.path == "/route":
            assert request.headers["Authorization"] == (
                f"Bearer {_EXTERNAL_ADMIN_API_KEY}"
            )
            routed += 1
            return httpx.Response(
                200,
                json={"data_proxy_addr": "http://proxy"},
                request=request,
            )
        if request.url.host == "proxy" and request.url.path.endswith("/turn"):
            turn = json.loads(request.content)
            if turn[MEMORY_CONTROL_AUTHORIZED_FIELD]:
                assert (
                    request.headers["Authorization"]
                    == (_MEMORY_CONTROL_HEADERS["Authorization"])
                )
            else:
                assert "Authorization" not in request.headers
            forwarded.append(turn)
            return httpx.Response(
                200,
                stream=httpx.ByteStream(b'{"id":"chatcmpl-test"}'),
                headers={"content-type": "application/json"},
                request=request,
            )
        raise AssertionError(f"unexpected bridge request: {request.url}")

    bridge = gateway_bridge.ChatCompletionsBridge(
        router_addr="http://router",
        admin_api_key=_EXTERNAL_ADMIN_API_KEY,
        memory_control_api_key=_MEMORY_CONTROL_API_KEY,
    )
    await bridge._http.aclose()
    bridge._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = fastapi.FastAPI()
    gateway_bridge.mount_chat_bridge(app, bridge)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://gateway",
        ) as client:
            ordinary = await client.post(
                "/v1/chat/completions",
                headers={"X-AReaL-Session-Key": "ordinary-session"},
                json={
                    "model": "model-1",
                    "messages": [{"role": "user", "content": "ordinary"}],
                },
            )
            missing_auth = await client.post(
                "/v1/chat/completions",
                headers={"X-AReaL-Session-Key": "missing-auth"},
                json={
                    "model": "model-1",
                    "messages": [{"role": "user", "content": "hello"}],
                    MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a"),
                },
            )
            wrong_auth = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": "Bearer wrong-key",
                    "X-AReaL-Session-Key": "wrong-auth",
                },
                json={
                    "model": "model-1",
                    "messages": [{"role": "user", "content": "hello"}],
                    MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a"),
                },
            )
            response = await client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {_EXTERNAL_ADMIN_API_KEY}",
                    "X-AReaL-Session-Key": "session-1",
                },
                json={
                    "model": "model-1",
                    "messages": [{"role": "user", "content": "hello"}],
                    MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a"),
                    "inf_base_url": "http://inference",
                    "inf_model": "model-1",
                    "session_api_key": "sk-sess-secret",
                },
            )
    finally:
        await bridge.close()

    assert ordinary.status_code == response.status_code == 200
    assert missing_auth.status_code == wrong_auth.status_code == 401
    assert routed == len(forwarded) == 2
    assert MEMORY_ASSIGNMENT_PIN_FIELD not in forwarded[0]
    assert forwarded[0][MEMORY_CONTROL_AUTHORIZED_FIELD] is False
    turn = forwarded[1]
    assert turn[MEMORY_ASSIGNMENT_PIN_FIELD] == _wire("a")
    assert turn[MEMORY_CONTROL_AUTHORIZED_FIELD] is True
    assert turn["metadata"] == {}
    assert turn["inf_base_url"] == "http://inference"
    assert turn["session_api_key"] == "sk-sess-secret"
    replayed = turn[CHAT_REQUEST_METADATA_KEY]
    for key in (
        MEMORY_ASSIGNMENT_PIN_FIELD,
        MEMORY_CONTROL_AUTHORIZED_FIELD,
        "inf_base_url",
        "inf_model",
        "session_api_key",
    ):
        assert key not in replayed


@pytest.mark.asyncio
async def test_responses_bridge_forwards_trusted_pin_and_preserves_conflict() -> None:
    forwarded: list[dict] = []

    def handler(request):
        if request.url.host == "router" and request.url.path == "/route":
            assert request.headers["Authorization"] == (
                f"Bearer {_EXTERNAL_ADMIN_API_KEY}"
            )
            return httpx.Response(
                200,
                json={"data_proxy_addr": "http://proxy"},
                request=request,
            )
        if request.url.host == "proxy" and request.url.path.endswith("/turn"):
            assert (
                request.headers["Authorization"]
                == _MEMORY_CONTROL_HEADERS["Authorization"]
            )
            forwarded.append(json.loads(request.content))
            return httpx.Response(
                409,
                json={"detail": "session is already bound"},
                request=request,
            )
        raise AssertionError(f"unexpected bridge request: {request.url}")

    bridge = gateway_bridge.OpenResponsesBridge(
        router_addr="http://router",
        admin_api_key=_EXTERNAL_ADMIN_API_KEY,
        memory_control_api_key=_MEMORY_CONTROL_API_KEY,
    )
    await bridge._http.aclose()
    bridge._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = fastapi.FastAPI()
    gateway_bridge.mount_bridge(
        app,
        bridge,
        admin_api_key=_EXTERNAL_ADMIN_API_KEY,
    )
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://gateway",
        ) as client:
            response = await client.post(
                "/v1/responses",
                headers={
                    "Authorization": f"Bearer {_EXTERNAL_ADMIN_API_KEY}",
                    "X-AReaL-Session-Key": "session-1",
                },
                json={
                    "model": "model-1",
                    "input": [
                        {
                            "type": "message",
                            "content": [{"type": "input_text", "text": "hello"}],
                        }
                    ],
                    MEMORY_ASSIGNMENT_PIN_FIELD: _wire("a"),
                },
            )
    finally:
        await bridge.close()

    assert response.status_code == 409
    assert response.json() == {"detail": "session is already bound"}
    assert len(forwarded) == 1
    assert forwarded[0][MEMORY_ASSIGNMENT_PIN_FIELD] == _wire("a")
    assert forwarded[0][MEMORY_CONTROL_AUTHORIZED_FIELD] is True

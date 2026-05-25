from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from areal.experimental.inference_service.data_proxy.app import (
    create_app as create_data_proxy_app,
)
from areal.experimental.inference_service.data_proxy.config import DataProxyConfig
from areal.experimental.inference_service.data_proxy.session import SessionStore
from areal.experimental.inference_service.gateway.app import (
    create_app as create_gateway_app,
)
from areal.experimental.inference_service.gateway.config import GatewayConfig
from areal.experimental.inference_service.gateway.streaming import (
    RouterKeyRejectedError,
)
from areal.experimental.inference_service.router.app import (
    create_app as create_router_app,
)
from areal.experimental.inference_service.router.config import RouterConfig
from areal.experimental.inference_service.router.state import ModelRegistry

ADMIN_KEY = "test-admin-key"
SESSION_KEY = "session-key-abc123"
WORKER_ADDR = "http://worker-1:18082"
ROUTER_MODULE = "areal.experimental.inference_service.gateway.app"


def admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def session_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {SESSION_KEY}"}


@pytest.fixture
def router_config() -> RouterConfig:
    return RouterConfig(
        host="127.0.0.1",
        port=18081,
        admin_api_key=ADMIN_KEY,
        poll_interval=999,
        routing_strategy="round_robin",
    )


@pytest_asyncio.fixture
async def router_client(router_config: RouterConfig):
    app = create_router_app(router_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestModelRegistry:
    @pytest.mark.asyncio
    async def test_model_registry_register_get_list_remove(self):
        reg = ModelRegistry()
        await reg.register("ext-a", "http://api", None, [WORKER_ADDR])

        info = await reg.get("ext-a")
        assert info is not None
        assert info.name == "ext-a"
        assert info.url == "http://api"
        assert info.data_proxy_addrs == [WORKER_ADDR]

        names = await reg.list_names()
        assert names == ["ext-a"]

        removed = await reg.remove("ext-a")
        assert removed is True
        assert await reg.get("ext-a") is None


class TestRouterExternalEndpoints:
    @pytest.mark.asyncio
    async def test_register_model_success(self, router_client):
        await router_client.post(
            "/register",
            json={"worker_addr": WORKER_ADDR},
            headers=admin_headers(),
        )

        resp = await router_client.post(
            "/register_model",
            json={
                "model": "ext-1",
                "url": "http://ext-api",
                "data_proxy_addrs": [WORKER_ADDR],
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model"] == "ext-1"
        assert data["data_proxy_addrs"] == [WORKER_ADDR]

    @pytest.mark.asyncio
    async def test_register_model_no_workers_503(self, router_client):
        resp = await router_client.post(
            "/register_model",
            json={"model": "ext-1", "url": "http://ext-api"},
            headers=admin_headers(),
        )
        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_register_model_no_auth_401(self, router_client):
        resp = await router_client.post(
            "/register_model",
            json={"model": "ext-1", "url": "http://ext-api"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_route_model_success(self, router_client):
        await router_client.post(
            "/register",
            json={"worker_addr": WORKER_ADDR},
            headers=admin_headers(),
        )
        await router_client.post(
            "/register_model",
            json={
                "model": "ext-1",
                "url": "http://ext-api",
                "data_proxy_addrs": [WORKER_ADDR],
            },
            headers=admin_headers(),
        )

        resp = await router_client.post(
            "/route",
            json={"model": "ext-1"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["worker_addr"] == WORKER_ADDR
        assert data["url"] == "http://ext-api"

    @pytest.mark.asyncio
    async def test_route_model_not_found_404(self, router_client):
        resp = await router_client.post(
            "/route",
            json={"model": "nope"},
            headers=admin_headers(),
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_models_empty(self, router_client):
        resp = await router_client.get("/models", headers=admin_headers())
        assert resp.status_code == 200
        assert resp.json()["models"] == []

    @pytest.mark.asyncio
    async def test_list_models_after_registration(self, router_client):
        await router_client.post(
            "/register",
            json={"worker_addr": WORKER_ADDR},
            headers=admin_headers(),
        )
        await router_client.post(
            "/register_model",
            json={
                "model": "ext-1",
                "url": "http://ext-api",
                "data_proxy_addrs": [WORKER_ADDR],
            },
            headers=admin_headers(),
        )

        resp = await router_client.get("/models", headers=admin_headers())
        assert resp.status_code == 200
        assert resp.json()["models"] == ["ext-1"]


@pytest.fixture
def gateway_config() -> GatewayConfig:
    return GatewayConfig(
        host="127.0.0.1",
        port=18080,
        admin_api_key=ADMIN_KEY,
        router_addr="http://mock-router:8081",
        router_timeout=2.0,
        forward_timeout=30.0,
    )


@pytest_asyncio.fixture
async def gateway_client(gateway_config: GatewayConfig):
    app = create_gateway_app(gateway_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestGatewayExternalEndpoints:
    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.register_model_in_router", new_callable=AsyncMock)
    async def test_register_model_gateway_full_flow(
        self,
        mock_register_model,
        mock_forward,
        gateway_client,
    ):
        mock_register_model.return_value = {
            "status": "ok",
            "model": "ext-1",
            "data_proxy_addrs": [WORKER_ADDR],
        }
        mock_forward.return_value = httpx.Response(200, json={"status": "ok"})

        resp = await gateway_client.post(
            "/register_model",
            json={"model": "ext-1", "url": "http://ext-api"},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["data_proxy_addrs"] == [WORKER_ADDR]
        mock_register_model.assert_called_once()
        mock_forward.assert_called_once()

    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.query_router", new_callable=AsyncMock)
    async def test_chat_completions_external_model(
        self,
        mock_query_router,
        mock_forward,
        gateway_client,
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"id": "ext-chat-1"})

        resp = await gateway_client.post(
            "/chat/completions",
            json={"model": "ext-1", "messages": [{"role": "user", "content": "hi"}]},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "ext-chat-1"
        assert "/chat/completions" in mock_forward.call_args.args[0]

    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.forward_sse_stream")
    @patch(f"{ROUTER_MODULE}.query_router", new_callable=AsyncMock)
    async def test_chat_completions_external_model_streaming(
        self,
        mock_query_router,
        mock_forward_sse,
        gateway_client,
    ):
        mock_query_router.return_value = WORKER_ADDR

        async def _stream() -> AsyncGenerator[bytes, None]:
            yield b"data: hello\n\n"
            yield b"data: [DONE]\n\n"

        mock_forward_sse.return_value = _stream()

        resp = await gateway_client.post(
            "/chat/completions",
            json={
                "model": "ext-1",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.query_router", new_callable=AsyncMock)
    async def test_chat_completions_unregistered_model_falls_back(
        self,
        mock_query_router,
        mock_forward,
        gateway_client,
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"id": "internal-chat"})

        resp = await gateway_client.post(
            "/chat/completions",
            json={
                "model": "missing",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=session_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "internal-chat"
        assert "/chat/completions" in mock_forward.call_args.args[0]

    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.query_router", new_callable=AsyncMock)
    async def test_chat_completions_no_model_internal_path(
        self,
        mock_query_router,
        mock_forward,
        gateway_client,
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"id": "internal-chat"})

        resp = await gateway_client.post(
            "/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=session_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "internal-chat"

    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.list_models_from_router", new_callable=AsyncMock)
    async def test_list_models_gateway(self, mock_list_models, gateway_client):
        mock_list_models.return_value = ["ext-1", "ext-2"]

        resp = await gateway_client.get("/models", headers=admin_headers())
        assert resp.status_code == 200
        assert resp.json()["models"] == ["ext-1", "ext-2"]

    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.query_router", new_callable=AsyncMock)
    async def test_export_trajectories_routes_external_by_session_id(
        self,
        mock_query_router,
        mock_forward,
        gateway_client,
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(
            200,
            json={
                "interactions": {"id-1": {"messages": [], "reward": 0.0}},
            },
        )

        resp = await gateway_client.post(
            "/export_trajectories",
            json={"session_ids": ["ext-1"]},
            headers=admin_headers(),
        )
        assert resp.status_code == 200
        assert "/export_trajectories" in mock_forward.call_args.args[0]


@pytest.fixture
def data_proxy_config() -> DataProxyConfig:
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
    tok._tok = MagicMock()
    tok._tok.eos_token_id = 2
    tok._tok.pad_token_id = 0
    return tok


@pytest.fixture
def mock_areal_client():
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=ChatCompletion(
            id="chatcmpl-mock",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(content="Hello!", role="assistant"),
                )
            ],
            created=1234567890,
            model="sglang",
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=3, prompt_tokens=5, total_tokens=8),
        )
    )
    return mock_client


@pytest_asyncio.fixture
async def data_proxy_app(data_proxy_config, mock_tokenizer, mock_areal_client):
    from areal.experimental.inference_service.data_proxy.pause import PauseState
    from areal.experimental.inference_service.inf_bridge import InfBridge
    from areal.experimental.inference_service.sglang.bridge import SGLangBridgeBackend

    app = create_data_proxy_app(data_proxy_config)
    pause_state = PauseState()
    inf_bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr=data_proxy_config.backend_addr,
        pause_state=pause_state,
        request_timeout=data_proxy_config.request_timeout,
        max_resubmit_retries=5,
        resubmit_wait=0.01,
    )
    app.state.tokenizer = mock_tokenizer
    app.state.inf_bridge = inf_bridge
    app.state.areal_client = mock_areal_client
    app.state.pause_state = pause_state
    app.state.config = data_proxy_config
    store = SessionStore()
    store.set_admin_key(data_proxy_config.admin_api_key)
    app.state.session_store = store
    app.state.http_client = MagicMock()
    app.state.version = 0
    http_client = httpx.AsyncClient(timeout=10.0)
    app.state.http_client = http_client
    yield app
    await http_client.aclose()


@pytest_asyncio.fixture
async def data_proxy_client(data_proxy_app):
    transport = httpx.ASGITransport(app=data_proxy_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestDataProxyExternalEndpoints:
    @pytest.mark.asyncio
    async def test_register_external_model(self, data_proxy_client):
        resp = await data_proxy_client.post(
            "/register_model",
            json={"name": "ext-1", "url": "http://ext-api", "model": "gpt-4o"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_external_chat_completions_non_streaming(
        self,
        data_proxy_client,
        data_proxy_app,
        monkeypatch,
    ):
        await data_proxy_client.post(
            "/register_model",
            json={"name": "ext-1", "url": "http://ext-api", "model": "gpt-4o"},
        )

        class _FakeClient:
            async def post(self, url, json=None, headers=None):
                assert url == "http://ext-api/chat/completions"
                assert json["model"] == "gpt-4o"
                return httpx.Response(200, json={"id": "ext-1-response"})

        data_proxy_app.state.http_client = _FakeClient()

        resp = await data_proxy_client.post(
            "/chat/completions",
            json={"model": "ext-1", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "ext-1-response"

        not_ready = await data_proxy_client.post(
            "/export_trajectories",
            json={"session_ids": ["__hitl__"], "remove_session": False},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert not_ready.status_code == 200
        assert not_ready.json()["interactions"] == {}

        set_reward = await data_proxy_client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert set_reward.status_code == 200
        assert set_reward.json()["trajectory_ready"] is True

        exported = await data_proxy_client.post(
            "/export_trajectories",
            json={"session_ids": ["__hitl__"]},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert exported.status_code == 200
        payload = exported.json()
        assert len(payload["interactions"]) == 1

    @pytest.mark.asyncio
    async def test_external_chat_completions_streaming(
        self,
        data_proxy_client,
        monkeypatch,
    ):
        await data_proxy_client.post(
            "/register_model",
            json={"name": "ext-1", "url": "http://ext-api", "model": "gpt-4o"},
        )

        class _FakeStreamResponse:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def aread(self):
                return b""

            async def aiter_bytes(self):
                yield b"data: chunk-1\n\n"
                yield b"data: [DONE]\n\n"

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            def stream(self, method, url, json=None, headers=None):
                assert method == "POST"
                assert url == "http://ext-api/chat/completions"
                assert json["model"] == "gpt-4o"
                return _FakeStreamResponse()

        monkeypatch.setattr(
            "areal.experimental.inference_service.data_proxy.app.httpx.AsyncClient",
            _FakeClient,
        )

        resp = await data_proxy_client.post(
            "/chat/completions",
            json={
                "model": "ext-1",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        set_reward = await data_proxy_client.post(
            "/rl/set_reward",
            json={"reward": 1.0},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert set_reward.status_code == 200
        assert set_reward.json()["trajectory_ready"] is True

        exported = await data_proxy_client.post(
            "/export_trajectories",
            json={"session_ids": ["__hitl__"]},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert exported.status_code == 200
        payload = exported.json()
        assert len(payload["interactions"]) == 1

        exported_again = await data_proxy_client.post(
            "/export_trajectories",
            json={"session_ids": ["__hitl__"]},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
        assert exported_again.status_code == 200
        assert exported_again.json()["interactions"] == {}

    @pytest.mark.asyncio
    async def test_unregistered_model_falls_through_to_internal(
        self, data_proxy_client
    ):
        resp = await data_proxy_client.post(
            "/chat/completions",
            json={
                "model": "missing",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_external_chat_uses_stored_provider_api_key(
        self,
        data_proxy_client,
        data_proxy_app,
        monkeypatch,
    ):
        await data_proxy_client.post(
            "/register_model",
            json={
                "name": "ext-1",
                "url": "http://ext-api",
                "model": "gpt-4o",
                "api_key": "sk-provider-key-99",
            },
        )

        captured_headers: dict[str, str] = {}

        class _FakeClient:
            async def post(self, url, json=None, headers=None):
                if headers:
                    captured_headers.update(headers)
                return httpx.Response(200, json={"id": "ext-1-response"})

        data_proxy_app.state.http_client = _FakeClient()

        resp = await data_proxy_client.post(
            "/chat/completions",
            json={"model": "ext-1", "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sk-session-key"},
        )
        assert resp.status_code == 200
        assert captured_headers.get("authorization") == "Bearer sk-provider-key-99"


@pytest.mark.asyncio
async def test_external_model_end_to_end_register_then_chat(router_config):
    router_app = create_router_app(router_config)
    router_transport = httpx.ASGITransport(app=router_app)
    async with httpx.AsyncClient(
        transport=router_transport,
        base_url="http://router",
    ) as router_client:
        await router_client.post(
            "/register",
            json={"worker_addr": WORKER_ADDR},
            headers=admin_headers(),
        )

        gateway_config = GatewayConfig(
            host="127.0.0.1",
            port=18080,
            admin_api_key=ADMIN_KEY,
            router_addr="http://router",
            router_timeout=2.0,
            forward_timeout=30.0,
        )
        gateway_app = create_gateway_app(gateway_config)
        gateway_transport = httpx.ASGITransport(app=gateway_app)

        proxy_state: dict[str, dict[str, str | None]] = {}

        async def _register_model_in_router(
            router_addr: str,
            model: str,
            url: str,
            api_key: str | None,
            data_proxy_addrs: list[str],
            admin_api_key: str,
            timeout: float,
            *,
            client: httpx.AsyncClient | None = None,
        ) -> dict:
            del router_addr, admin_api_key, timeout, client
            resp = await router_client.post(
                "/register_model",
                json={
                    "model": model,
                    "url": url,
                    "api_key": api_key,
                    "data_proxy_addrs": data_proxy_addrs,
                },
                headers=admin_headers(),
            )
            resp.raise_for_status()
            return resp.json()

        async def _query_router(
            router_addr: str,
            api_key: str | None = None,
            path: str | None = None,
            timeout: float = 2.0,
            *,
            session_id: str | None = None,
            admin_api_key: str | None = None,
            model: str | None = None,
            client: httpx.AsyncClient | None = None,
        ) -> str:
            del router_addr, path, timeout, admin_api_key, client
            payload: dict[str, str] = {}
            if model is not None:
                payload["model"] = model
            if api_key is not None:
                payload["api_key"] = api_key
            if session_id is not None:
                payload["session_id"] = session_id
            resp = await router_client.post(
                "/route",
                json=payload,
                headers=admin_headers(),
            )
            if model is not None:
                if resp.status_code == 404:
                    raise RouterKeyRejectedError("not found", 404)
                if resp.status_code == 503:
                    raise RouterKeyRejectedError("no healthy workers", 503)
                resp.raise_for_status()
                return resp.json()["worker_addr"]
            resp.raise_for_status()
            return resp.json()["worker_addr"]

        async def _forward_request(
            upstream_url: str,
            body: bytes,
            headers: dict[str, str],
            timeout: float,
            *,
            client: httpx.AsyncClient | None = None,
        ) -> httpx.Response:
            del timeout, client
            if upstream_url == f"{WORKER_ADDR}/register_model":
                data = json.loads(body)
                proxy_state[data["name"]] = {
                    "url": data["url"],
                    "model": data.get("model"),
                }
                return httpx.Response(200, json={"status": "ok"})
            if upstream_url == f"{WORKER_ADDR}/chat/completions":
                data = json.loads(body)
                assert data["model"] in proxy_state
                return httpx.Response(200, json={"id": "ext-e2e"})
            return httpx.Response(500, json={"error": "unexpected"})

        with (
            patch(
                f"{ROUTER_MODULE}.register_model_in_router",
                new=AsyncMock(side_effect=_register_model_in_router),
            ),
            patch(
                f"{ROUTER_MODULE}.query_router",
                new=AsyncMock(side_effect=_query_router),
            ),
            patch(
                f"{ROUTER_MODULE}.forward_request",
                new=AsyncMock(side_effect=_forward_request),
            ),
        ):
            async with httpx.AsyncClient(
                transport=gateway_transport,
                base_url="http://gateway",
            ) as gateway_client:
                reg = await gateway_client.post(
                    "/register_model",
                    json={
                        "model": "ext-1",
                        "url": "http://ext-api",
                    },
                    headers=admin_headers(),
                )
                assert reg.status_code == 200

                chat = await gateway_client.post(
                    "/chat/completions",
                    json={
                        "model": "ext-1",
                        "messages": [{"role": "user", "content": "hello"}],
                    },
                    headers=admin_headers(),
                )
                assert chat.status_code == 200
                assert chat.json()["id"] == "ext-e2e"

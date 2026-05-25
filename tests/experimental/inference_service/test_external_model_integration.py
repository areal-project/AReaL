from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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

ADMIN_KEY = "test-admin-key"
WORKER_ADDR = "http://worker-1:18082"
ROUTER_MODULE = "areal.experimental.inference_service.gateway.app"


def admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


@pytest.fixture
def router_config() -> RouterConfig:
    return RouterConfig(
        host="127.0.0.1",
        port=18081,
        admin_api_key=ADMIN_KEY,
        poll_interval=999,
        routing_strategy="round_robin",
    )


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


class TestGatewayUnifiedExportTrajectories:
    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.query_router", new_callable=AsyncMock)
    async def test_export_trajectories_with_session_id(
        self,
        mock_query_router,
        mock_forward,
        mock_revoke,
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
        mock_query_router.assert_called_once()
        assert "/export_trajectories" in mock_forward.call_args.args[0]

    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.query_router", new_callable=AsyncMock)
    async def test_export_trajectories_internal_session(
        self,
        mock_query_router,
        mock_forward,
        mock_revoke,
        gateway_client,
    ):
        mock_query_router.return_value = WORKER_ADDR
        mock_forward.return_value = httpx.Response(200, json={"interactions": []})

        resp = await gateway_client.post(
            "/export_trajectories",
            json={
                "session_ids": ["ses-1"],
                "group_id": "grp-test",
                "discount": 1.0,
                "style": "sft",
            },
            headers=admin_headers(),
        )

        assert resp.status_code == 200
        mock_query_router.assert_called_once()
        mock_revoke.assert_called_once()
        assert "/export_trajectories" in mock_forward.call_args.args[0]

    @pytest.mark.asyncio
    @patch(f"{ROUTER_MODULE}.revoke_session_in_router", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.forward_request", new_callable=AsyncMock)
    @patch(f"{ROUTER_MODULE}.query_router", new_callable=AsyncMock)
    async def test_export_trajectories_without_session_id_returns_400(
        self,
        mock_query_router,
        mock_forward,
        mock_revoke,
        gateway_client,
    ):
        resp = await gateway_client.post(
            "/export_trajectories",
            json={"discount": 1.0},
            headers=admin_headers(),
        )

        assert resp.status_code == 400
        assert "session_ids is required" in resp.json()["error"]
        mock_query_router.assert_not_called()
        mock_forward.assert_not_called()
        mock_revoke.assert_not_called()


@pytest.mark.asyncio
async def test_external_model_flow_end_to_end_gateway_router_data_proxy(router_config):
    mock_external_app = FastAPI()

    @mock_external_app.post("/chat/completions")
    async def mock_chat(request: Request):
        body = await request.json()
        return JSONResponse(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 1234567890,
                "model": body.get("model", "mock-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Mock response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

    router_app = create_router_app(router_config)
    data_proxy_app = create_data_proxy_app(
        DataProxyConfig(
            host="127.0.0.1",
            port=18082,
            backend_addr="http://mock-sglang:30000",
            tokenizer_path="mock-tokenizer",
            request_timeout=10.0,
            admin_api_key=ADMIN_KEY,
        )
    )
    data_proxy_app.state.config = DataProxyConfig(
        host="127.0.0.1",
        port=18082,
        backend_addr="http://mock-sglang:30000",
        tokenizer_path="mock-tokenizer",
        request_timeout=10.0,
        admin_api_key=ADMIN_KEY,
    )
    store = SessionStore()
    store.set_admin_key(ADMIN_KEY)
    data_proxy_app.state.session_store = store
    data_proxy_app.state.version = 0
    data_proxy_app.state.http_client = httpx.AsyncClient(timeout=10.0)
    gateway_app = create_gateway_app(
        GatewayConfig(
            host="127.0.0.1",
            port=18080,
            admin_api_key=ADMIN_KEY,
            router_addr="http://router",
            router_timeout=2.0,
            forward_timeout=30.0,
        )
    )

    router_transport = httpx.ASGITransport(app=router_app)
    proxy_transport = httpx.ASGITransport(app=data_proxy_app)
    gateway_transport = httpx.ASGITransport(app=gateway_app)
    external_transport = httpx.ASGITransport(app=mock_external_app)

    async with (
        httpx.AsyncClient(
            transport=router_transport, base_url="http://router"
        ) as router_client,
        httpx.AsyncClient(
            transport=proxy_transport, base_url="http://worker-1:18082"
        ) as data_proxy_client,
        httpx.AsyncClient(
            transport=gateway_transport, base_url="http://gateway"
        ) as gateway_client,
        httpx.AsyncClient(
            transport=external_transport, base_url="http://mock-external"
        ) as external_client,
    ):
        data_proxy_app.state.http_client = external_client
        await router_client.post(
            "/register",
            json={"worker_addr": WORKER_ADDR},
            headers=admin_headers(),
        )

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
            payload: dict = {}
            if model is not None:
                payload["model"] = model
            if session_id is not None:
                payload["session_id"] = session_id
            elif api_key is not None:
                payload["api_key"] = api_key
            resp = await router_client.post(
                "/route",
                json=payload,
                headers=admin_headers(),
            )
            if resp.status_code in (404, 503):
                raise RouterKeyRejectedError("routing failed", resp.status_code)
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
            if upstream_url.startswith(WORKER_ADDR):
                path = upstream_url.removeprefix(WORKER_ADDR)
                return await data_proxy_client.post(path, content=body, headers=headers)
            return httpx.Response(500, json={"error": "unexpected upstream"})

        class _ExternalClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, url, json=None, headers=None):
                assert url == "http://mock-external/chat/completions"
                return await external_client.post(
                    "/chat/completions",
                    json=json,
                    headers=headers,
                )

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
            patch(
                "areal.experimental.inference_service.data_proxy.app.httpx.AsyncClient",
                _ExternalClient,
            ),
        ):
            # Set the shared HTTP client to the fake external client so
            # non-streaming external model requests go through it.
            data_proxy_app.state.http_client = _ExternalClient()

            reg = await gateway_client.post(
                "/register_model",
                json={
                    "model": "ext-1",
                    "url": "http://mock-external",
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
            assert chat.json()["id"] == "chatcmpl-mock"

            set_reward = await gateway_client.post(
                "/rl/set_reward",
                json={"reward": 1.0},
                headers=admin_headers(),
            )
            assert set_reward.status_code == 200
            assert set_reward.json()["trajectory_ready"] is True

            exported = await gateway_client.post(
                "/export_trajectories",
                json={"session_ids": ["__hitl__"]},
                headers=admin_headers(),
            )
            assert exported.status_code == 200
            payload = exported.json()
            assert len(payload["interactions"]) == 1

            interaction = next(iter(payload["interactions"].values()))
            assert interaction["messages"][0]["content"] == "hello"
            cached_response = json.loads(
                interaction["output_message_list"][0]["content"]
            )
            assert (
                cached_response["choices"][0]["message"]["content"] == "Mock response"
            )

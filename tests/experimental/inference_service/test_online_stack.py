"""Lightweight online-stack integration tests.

These tests exercise the Gateway -> Router -> Data Proxy online flow in-process
using ASGI transports and mocked model responses, so they do not require the
GPU-backed serving stack used by the slow integration suites.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from areal.experimental.inference_service.data_proxy.app import (
    create_app as create_dp_app,
)
from areal.experimental.inference_service.data_proxy.config import DataProxyConfig
from areal.experimental.inference_service.data_proxy.session import SessionStore
from areal.experimental.inference_service.gateway.app import create_app as create_gw_app
from areal.experimental.inference_service.gateway.config import GatewayConfig
from areal.experimental.inference_service.gateway.streaming import (
    RouterKeyRejectedError,
    RouterUnreachableError,
)
from areal.experimental.inference_service.router.app import (
    create_app as create_router_app,
)
from areal.experimental.inference_service.router.config import RouterConfig

ADMIN_KEY = "areal-admin-key"
DATA_PROXY_ADDR = "http://data-proxy"
ROUTER_ADDR = "http://router"


def _make_mock_areal_client():
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.completion_usage import CompletionUsage

    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    mock_client = MagicMock()
    call_index = 0

    async def _mock_create(*, areal_cache=None, **kwargs):
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

        interaction = InteractionWithTokenLogpReward(
            messages=kwargs.get("messages", []),
            completion=completion,
            output_message_list=[{"role": "assistant", "content": "Hello!"}],
        )
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
async def online_stack(monkeypatch):
    import areal.experimental.inference_service.gateway.app as gateway_app_module
    from areal.experimental.inference_service.data_proxy.pause import PauseState
    from areal.experimental.inference_service.inf_bridge import InfBridge
    from areal.experimental.inference_service.sglang.bridge import SGLangBridgeBackend

    dp_config = DataProxyConfig(
        host="127.0.0.1",
        port=18082,
        backend_addr="http://mock-sglang:30000",
        tokenizer_path="mock-tokenizer",
        request_timeout=10.0,
        admin_api_key=ADMIN_KEY,
    )
    router_config = RouterConfig(
        host="127.0.0.1",
        port=18081,
        admin_api_key=ADMIN_KEY,
        poll_interval=999,
        routing_strategy="round_robin",
    )
    gw_config = GatewayConfig(
        host="127.0.0.1",
        port=18080,
        admin_api_key=ADMIN_KEY,
        router_addr=ROUTER_ADDR,
        router_timeout=2.0,
        forward_timeout=10.0,
    )

    dp_app = create_dp_app(dp_config)
    pause_state = PauseState()
    inf_bridge = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr=dp_config.backend_addr,
        pause_state=pause_state,
        request_timeout=dp_config.request_timeout,
        max_resubmit_retries=5,
        resubmit_wait=0.01,
    )
    store = SessionStore()
    store.set_admin_key(ADMIN_KEY)
    mock_tokenizer = MagicMock()
    mock_tokenizer._tok = MagicMock()
    mock_tokenizer._tok.eos_token_id = 2
    mock_tokenizer._tok.pad_token_id = 0
    dp_app.state.tokenizer = mock_tokenizer
    dp_app.state.inf_bridge = inf_bridge
    dp_app.state.areal_client = _make_mock_areal_client()
    dp_app.state.pause_state = pause_state
    dp_app.state.config = dp_config
    dp_app.state.session_store = store
    dp_app.state.http_client = MagicMock()
    dp_app.state.version = 0

    router_app = create_router_app(router_config)
    gw_app = create_gw_app(gw_config)

    dp_transport = httpx.ASGITransport(app=dp_app)
    router_transport = httpx.ASGITransport(app=router_app)
    gw_transport = httpx.ASGITransport(app=gw_app)

    async with (
        httpx.AsyncClient(
            transport=dp_transport, base_url=DATA_PROXY_ADDR
        ) as dp_client,
        httpx.AsyncClient(
            transport=router_transport, base_url=ROUTER_ADDR
        ) as router_client,
        httpx.AsyncClient(
            transport=gw_transport, base_url="http://gateway"
        ) as gw_client,
    ):
        reg_resp = await router_client.post(
            "/register",
            json={"worker_addr": DATA_PROXY_ADDR},
            headers={"Authorization": f"Bearer {ADMIN_KEY}"},
        )
        assert reg_resp.status_code == 200

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
            del router_addr, timeout, client
            payload: dict[str, str] = {}
            if model is not None:
                payload["model"] = model
            if session_id is not None:
                payload["session_id"] = session_id
            else:
                if api_key is not None:
                    payload["api_key"] = api_key
                if path is not None:
                    payload["path"] = path
            headers = {}
            if admin_api_key is not None:
                headers["Authorization"] = f"Bearer {admin_api_key}"
            resp = await router_client.post("/route", json=payload, headers=headers)
            if resp.status_code in {404, 503}:
                detail = resp.json().get("detail", "route failed")
                raise RouterKeyRejectedError(detail, resp.status_code)
            if resp.status_code >= 400:
                raise RouterUnreachableError(resp.text)
            return resp.json()["worker_addr"]

        async def _forward_request(
            url: str,
            body: bytes,
            headers: dict[str, str],
            timeout: float,
            *,
            client: httpx.AsyncClient | None = None,
        ):
            del timeout, client
            path = url.replace(DATA_PROXY_ADDR, "")
            return await dp_client.post(path, content=body, headers=headers)

        async def _revoke_session_in_router(
            router_addr: str,
            admin_api_key: str,
            group_id: str,
            timeout: float = 2.0,
            *,
            client: httpx.AsyncClient | None = None,
        ) -> None:
            del router_addr, timeout, client
            resp = await router_client.post(
                "/remove_session",
                json={"group_id": group_id},
                headers={"Authorization": f"Bearer {admin_api_key}"},
            )
            assert resp.status_code == 200

        monkeypatch.setattr(gateway_app_module, "query_router", _query_router)
        monkeypatch.setattr(gateway_app_module, "forward_request", _forward_request)
        monkeypatch.setattr(
            gateway_app_module,
            "revoke_session_in_router",
            _revoke_session_in_router,
        )

        yield {
            "gateway_client": gw_client,
            "router_app": router_app,
            "data_proxy_app": dp_app,
        }


def _admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


def _hitl_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


@pytest.mark.asyncio
async def test_online_stack_latest_ready_export_keeps_session_pinned(online_stack):
    gw = online_stack["gateway_client"]
    router_app = online_stack["router_app"]

    await gw.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "first"}]},
        headers=_hitl_headers(),
    )
    await gw.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=_hitl_headers(),
    )

    await gw.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "second"}]},
        headers=_hitl_headers(),
    )
    reward_resp = await gw.post(
        "/rl/set_reward",
        json={"reward": 2.0},
        headers=_hitl_headers(),
    )
    assert reward_resp.status_code == 200
    assert reward_resp.json()["trajectory_id"] == 1

    export_resp = await gw.post(
        "/export_trajectories",
        json={"session_ids": ["__hitl__"], "discount": 1.0, "style": "individual"},
        headers=_admin_headers(),
    )
    assert export_resp.status_code == 200
    assert list(export_resp.json()["interactions"]) == ["chatcmpl-test1"]

    session_registry = router_app.state.session_registry
    assert await session_registry.lookup_by_id("__hitl__") == DATA_PROXY_ADDR


@pytest.mark.asyncio
async def test_online_stack_explicit_then_latest_export(online_stack):
    gw = online_stack["gateway_client"]

    await gw.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "first"}]},
        headers=_hitl_headers(),
    )
    first_reward = await gw.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=_hitl_headers(),
    )
    assert first_reward.status_code == 200
    assert first_reward.json()["trajectory_id"] == 0

    await gw.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "second"}]},
        headers=_hitl_headers(),
    )
    second_reward = await gw.post(
        "/rl/set_reward",
        json={"reward": 2.0},
        headers=_hitl_headers(),
    )
    assert second_reward.status_code == 200
    assert second_reward.json()["trajectory_id"] == 1

    explicit_resp = await gw.post(
        "/export_trajectories",
        json={
            "session_ids": ["__hitl__"],
            "trajectory_id": 0,
            "discount": 1.0,
            "style": "individual",
            "remove_session": False,
        },
        headers=_admin_headers(),
    )
    assert explicit_resp.status_code == 200
    assert list(explicit_resp.json()["interactions"]) == ["chatcmpl-test0"]

    latest_resp = await gw.post(
        "/export_trajectories",
        json={
            "session_ids": ["__hitl__"],
            "discount": 1.0,
            "style": "individual",
            "remove_session": False,
        },
        headers=_admin_headers(),
    )
    assert latest_resp.status_code == 200
    assert list(latest_resp.json()["interactions"]) == ["chatcmpl-test1"]


@pytest.mark.asyncio
async def test_online_stack_reward_creates_pending_ready_event_without_callback(
    online_stack,
):
    gw = online_stack["gateway_client"]
    data_proxy_app = online_stack["data_proxy_app"]

    await gw.post(
        "/chat/completions",
        json={"model": "sglang", "messages": [{"role": "user", "content": "first"}]},
        headers=_hitl_headers(),
    )
    resp = await gw.post(
        "/rl/set_reward",
        json={"reward": 1.0},
        headers=_hitl_headers(),
    )
    assert resp.status_code == 200

    notifications = data_proxy_app.state.session_store.pending_online_callbacks()
    assert len(notifications) == 1
    assert notifications[0].session_id == "__hitl__"
    assert notifications[0].trajectory_id == 0

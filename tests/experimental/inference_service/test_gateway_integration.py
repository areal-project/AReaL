"""Full-stack integration test: client → Gateway → Router → Data Proxy → SGLang.

Requires GPU and a model. Run with:
    uv run pytest tests/experimental/inference_service/test_gateway_integration.py -v -s

The test launches:
  1. A real SGLang server (GPU subprocess)
  2. A Data Proxy FastAPI server (uvicorn, thread)
  3. A Router FastAPI server (uvicorn, thread)
  4. A Gateway FastAPI server (uvicorn, thread)

Then exercises the full request path through the gateway.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time

import httpx
import pytest
import torch
import uvicorn

from tests.utils import get_model_path

from areal.api.cli_args import SGLangConfig
from areal.utils import network

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
HF_MODEL_ID = "Qwen/Qwen3-0.6B"
SERVER_STARTUP_TIMEOUT = 180  # seconds
SERVICE_STARTUP_TIMEOUT = 10  # seconds for gateway/router/data-proxy

ADMIN_KEY = "integ-admin-key"


def _get_test_model_path() -> str:
    return get_model_path(LOCAL_MODEL_PATH, HF_MODEL_ID)


def _has_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _check_health(base_url: str, timeout: float = 5.0) -> bool:
    try:
        resp = httpx.get(f"{base_url}/health", timeout=timeout)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


def _wait_for_health(base_url: str, timeout: float, label: str) -> None:
    """Block until the service at base_url/health returns 200."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _check_health(base_url, timeout=2.0):
            return
        time.sleep(0.3)
    raise RuntimeError(
        f"{label} at {base_url} did not become healthy within {timeout}s"
    )


def _run_uvicorn(app, host: str, port: int) -> None:
    """Run uvicorn in the current thread (blocking). Used as a thread target."""
    uvicorn.run(app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sglang_server():
    """Launch an SGLang server and yield its ``(host, port, base_url)``."""
    if not _has_gpu():
        pytest.skip("GPU required for SGLang server")

    from areal.infra.utils.proc import kill_process_tree

    host = network.gethostip()
    port, dist_port = network.find_free_ports(2)

    cmd = SGLangConfig.build_cmd(
        sglang_config=SGLangConfig(
            skip_tokenizer_init=True,
            model_path=_get_test_model_path(),
            mem_fraction_static=0.3,
        ),
        host=host,
        port=port,
        tp_size=1,
        base_gpu_id=0,
        dist_init_addr=f"{host}:{dist_port}",
    )

    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
    base_url = f"http://{host}:{port}"

    t0 = time.time()
    while time.time() - t0 < SERVER_STARTUP_TIMEOUT:
        if _check_health(base_url):
            break
        time.sleep(1)

    if time.time() - t0 >= SERVER_STARTUP_TIMEOUT:
        kill_process_tree(process.pid, graceful=True)
        pytest.fail("SGLang server did not become healthy within timeout")

    yield {"host": host, "port": port, "base_url": base_url}

    kill_process_tree(process.pid, graceful=True)


@pytest.fixture(scope="module")
def vllm_server():
    """Launch a vLLM server and yield its ``(host, port, base_url)``."""
    if not _has_gpu():
        pytest.skip("GPU required for vLLM server")

    from tests.experimental.inference_service.integration_utils import (
        launch_vllm_server,
    )

    from areal.infra.utils.proc import kill_process_tree

    process, info = launch_vllm_server(_get_test_model_path())
    yield info
    kill_process_tree(process.pid, graceful=True)


@pytest.fixture(scope="module")
def model_path() -> str:
    return _get_test_model_path()


@pytest.fixture(scope="module")
def gateway_stack(sglang_server, model_path):
    """Launch the full gateway stack (data proxy + router + gateway) on free ports.

    Yields a dict with addresses and ports for all services.
    """
    # --- Allocate ports ---
    data_proxy_port = _find_free_port()
    router_port = _find_free_port()
    gateway_port = _find_free_port()

    bind_host = "127.0.0.1"
    data_proxy_addr = f"http://{bind_host}:{data_proxy_port}"
    router_addr = f"http://{bind_host}:{router_port}"
    gateway_addr = f"http://{bind_host}:{gateway_port}"

    # --- Create Data Proxy app ---
    from areal.experimental.inference_service.data_proxy.app import (
        create_app as create_dp_app,
    )
    from areal.experimental.inference_service.data_proxy.config import DataProxyConfig

    dp_config = DataProxyConfig(
        host=bind_host,
        port=data_proxy_port,
        backend_addr=sglang_server["base_url"],
        tokenizer_path=model_path,
        request_timeout=60.0,
        admin_api_key=ADMIN_KEY,
    )
    dp_app = create_dp_app(dp_config)

    # --- Create Router app ---
    from areal.experimental.inference_service.router.app import (
        create_app as create_router_app,
    )
    from areal.experimental.inference_service.router.config import RouterConfig

    router_config = RouterConfig(
        host=bind_host,
        port=router_port,
        admin_api_key=ADMIN_KEY,
        poll_interval=2.0,
        worker_health_timeout=2.0,
        routing_strategy="round_robin",
    )
    router_app = create_router_app(router_config)

    # --- Create Gateway app ---
    from areal.experimental.inference_service.gateway.app import (
        create_app as create_gw_app,
    )
    from areal.experimental.inference_service.gateway.config import GatewayConfig

    gw_config = GatewayConfig(
        host=bind_host,
        port=gateway_port,
        admin_api_key=ADMIN_KEY,
        router_addr=router_addr,
        router_timeout=5.0,
        forward_timeout=60.0,
    )
    gw_app = create_gw_app(gw_config)

    # --- Start all three as daemon threads ---
    threads = []
    for app, port in [
        (dp_app, data_proxy_port),
        (router_app, router_port),
        (gw_app, gateway_port),
    ]:
        t = threading.Thread(
            target=_run_uvicorn,
            args=(app, bind_host, port),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # --- Wait for all services to be healthy ---
    _wait_for_health(data_proxy_addr, SERVICE_STARTUP_TIMEOUT, "Data Proxy")
    _wait_for_health(router_addr, SERVICE_STARTUP_TIMEOUT, "Router")
    _wait_for_health(gateway_addr, SERVICE_STARTUP_TIMEOUT, "Gateway")

    # --- Register the data proxy worker in the router ---
    resp = httpx.post(
        f"{router_addr}/register",
        json={"worker_addr": data_proxy_addr},
        headers={"Authorization": f"Bearer {ADMIN_KEY}"},
        timeout=5.0,
    )
    assert resp.status_code == 200, f"Failed to register worker: {resp.text}"
    worker_id = resp.json()["worker_id"]

    # Wait briefly for router health poller to mark worker healthy
    time.sleep(3)

    yield {
        "gateway_addr": gateway_addr,
        "router_addr": router_addr,
        "data_proxy_addr": data_proxy_addr,
        "admin_key": ADMIN_KEY,
        "worker_id": worker_id,
    }

    # Daemon threads die with the process — no explicit cleanup needed.


# ---------------------------------------------------------------------------
# Tests — full stack via gateway
# ---------------------------------------------------------------------------


@pytest.mark.sglang
@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestGatewayStackHealth:
    """Verify all services are healthy after stack launch."""

    @pytest.mark.asyncio
    async def test_all_services_healthy(self, gateway_stack):
        """All three services report healthy."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Gateway health
            resp = await client.get(f"{gateway_stack['gateway_addr']}/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

            # Router health (should show 1 worker)
            resp = await client.get(f"{gateway_stack['router_addr']}/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["workers"] == 1

            # Data Proxy health
            resp = await client.get(f"{gateway_stack['data_proxy_addr']}/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"


@pytest.mark.sglang
@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestGatewayChatCompletions:
    """Test /chat/completions endpoint through the full stack."""

    @pytest.mark.asyncio
    async def test_admin_non_streaming_chat(self, gateway_stack):
        """Admin key → Gateway /chat/completions (non-streaming, standalone mode)."""
        gw = gateway_stack["gateway_addr"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{gw}/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "max_completion_tokens": 32,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 200
            data = resp.json()

            assert data["object"] == "chat.completion"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert len(data["choices"][0]["message"]["content"]) > 0
            assert data["choices"][0]["finish_reason"] in ("stop", "length")

    @pytest.mark.asyncio
    async def test_session_non_streaming_chat(self, gateway_stack):
        """Session key → Gateway /chat/completions (non-streaming, with caching)."""
        gw = gateway_stack["gateway_addr"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Start session
            resp = await client.post(
                f"{gw}/rl/start_session",
                json={"task_id": "integ-gw-chat-ns"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 201
            session_api_key = resp.json()["sessions"][0]["session_api_key"]

            # Chat completion with session key
            resp = await client.post(
                f"{gw}/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "What is 3+5?"}],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
            )
            assert resp.status_code == 200
            completion = resp.json()
            assert completion["object"] == "chat.completion"
            assert len(completion["choices"][0]["message"]["content"]) > 0

            # Validate usage
            assert "usage" in completion
            assert completion["usage"]["prompt_tokens"] > 0
            assert completion["usage"]["completion_tokens"] > 0

            # Finish session via set_reward
            resp = await client.post(
                f"{gw}/rl/set_reward",
                json={
                    "reward": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
            )
            assert resp.status_code == 200
            assert resp.json()["interaction_count"] == 1
            assert resp.json()["ready_transition"] is True

    @pytest.mark.asyncio
    async def test_session_streaming_chat(self, gateway_stack):
        """Session key → Gateway /chat/completions stream=True → SSE chunks."""
        gw = gateway_stack["gateway_addr"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Start session
            resp = await client.post(
                f"{gw}/rl/start_session",
                json={"task_id": "integ-gw-chat-stream"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 201
            session_api_key = resp.json()["sessions"][0]["session_api_key"]

            # Streaming chat completion
            resp = await client.post(
                f"{gw}/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_completion_tokens": 32,
                    "temperature": 0.0,
                    "stream": True,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            # Parse SSE chunks
            chunks = []
            for line in resp.content.decode().strip().split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    chunks.append(json.loads(payload))

            assert len(chunks) >= 2  # role chunk + content/finish chunk

            # First chunk: role
            first = chunks[0]
            assert first["object"] == "chat.completion.chunk"
            assert first["choices"][0]["delta"].get("role") == "assistant"

            # Last chunk: finish_reason
            last = chunks[-1]
            assert last["choices"][0]["finish_reason"] in ("stop", "length")

            # At least one content chunk
            content_chunks = [
                c for c in chunks if c["choices"][0]["delta"].get("content")
            ]
            assert len(content_chunks) >= 1

            # All chunks share the same completion ID
            ids = {c["id"] for c in chunks}
            assert len(ids) == 1

            # Finish session via set_reward
            resp = await client.post(
                f"{gw}/rl/set_reward",
                json={
                    "reward": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
            )
            assert resp.status_code == 200
            assert resp.json()["ready_transition"] is True

    @pytest.mark.asyncio
    async def test_multi_turn_session_chat(self, gateway_stack):
        """Two sequential chat completions in the same session through the gateway."""
        gw = gateway_stack["gateway_addr"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Start session
            resp = await client.post(
                f"{gw}/rl/start_session",
                json={"task_id": "integ-gw-multi"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 201
            session_api_key = resp.json()["sessions"][0]["session_api_key"]

            # Turn 1
            resp1 = await client.post(
                f"{gw}/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "What is 3+5?"}],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
            )
            assert resp1.status_code == 200
            turn1_content = resp1.json()["choices"][0]["message"]["content"]
            assert len(turn1_content) > 0

            # Turn 2 (builds on first)
            resp2 = await client.post(
                f"{gw}/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [
                        {"role": "user", "content": "What is 3+5?"},
                        {"role": "assistant", "content": turn1_content},
                        {"role": "user", "content": "Now add 2 to that."},
                    ],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
            )
            assert resp2.status_code == 200
            assert len(resp2.json()["choices"][0]["message"]["content"]) > 0

            # Different completion IDs
            assert resp1.json()["id"] != resp2.json()["id"]

            # Finish session via set_reward — should report 2 interactions
            resp = await client.post(
                f"{gw}/rl/set_reward",
                json={
                    "reward": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
            )
            assert resp.status_code == 200
            assert resp.json()["interaction_count"] == 2
            assert resp.json()["ready_transition"] is True


@pytest.mark.sglang
@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestGatewaySessionLifecycle:
    """Test full RL session lifecycle through the gateway stack."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, gateway_stack):
        """start_session → chat → set_reward(finish=True) → export_trajectories."""
        gw = gateway_stack["gateway_addr"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            # --- start session ---
            resp = await client.post(
                f"{gw}/rl/start_session",
                json={"task_id": "integ-gw-lifecycle"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 201, resp.text
            session = resp.json()
            session_api_key = session["sessions"][0]["session_api_key"]
            session_id = session["sessions"][0]["session_id"]

            # --- chat completions (with session key) ---
            resp = await client.post(
                f"{gw}/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "What is 10-3?"}],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
            )
            assert resp.status_code == 200, resp.text

            # --- set reward and finish session ---
            resp = await client.post(
                f"{gw}/rl/set_reward",
                json={
                    "reward": 1.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["message"] == "success"
            assert resp.json()["interaction_count"] == 1
            assert resp.json()["ready_transition"] is True

            # --- export trajectories ---
            resp = await client.post(
                f"{gw}/export_trajectories",
                json={"session_ids": [session_id]},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 200, resp.text
            export_data = resp.json()
            assert "interactions" in export_data
            interactions = export_data["interactions"]
            assert len(interactions) == 1

            for _iid, item in interactions.items():
                assert "tensor_dict" in item
                assert "reward" in item
                assert item["reward"] == 1.0
                assert "interaction_id" in item

    @pytest.mark.asyncio
    async def test_session_pinning(self, gateway_stack):
        """Verify that session key routes to the same data proxy (pinned worker)."""
        gw = gateway_stack["gateway_addr"]
        router = gateway_stack["router_addr"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Start session
            resp = await client.post(
                f"{gw}/rl/start_session",
                json={"task_id": "integ-gw-pinning"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 201
            session = resp.json()
            session_api_key = session["sessions"][0]["session_api_key"]
            session_id = session["sessions"][0]["session_id"]

            # Query router directly for session pinning
            resp = await client.post(
                f"{router}/route",
                json={"session_id": session_id},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 200
            pinned_worker = resp.json()["worker_addr"]
            assert pinned_worker == gateway_stack["data_proxy_addr"]

            # Session key should route to the pinned worker via /route
            resp = await client.post(
                f"{router}/route",
                json={"api_key": session_api_key, "path": "/chat/completions"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 200
            assert resp.json()["worker_addr"] == pinned_worker

            resp = await client.post(
                f"{gw}/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_completion_tokens": 8,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
            )
            assert resp.status_code == 200

            # Finish session via set_reward
            resp = await client.post(
                f"{gw}/rl/set_reward",
                json={
                    "reward": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
            )
            assert resp.status_code == 200
            assert resp.json()["ready_transition"] is True


@pytest.mark.sglang
@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestGatewayAuth:
    """Test authentication enforcement through the gateway."""

    @pytest.mark.asyncio
    async def test_missing_auth_rejected(self, gateway_stack):
        """Requests without Authorization header are rejected."""
        gw = gateway_stack["gateway_addr"]
        async with httpx.AsyncClient(timeout=10.0) as client:
            # /chat/completions without auth → 401
            resp = await client.post(
                f"{gw}/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_non_admin_on_admin_endpoint(self, gateway_stack):
        """Non-admin key on admin-only endpoints is rejected with 403."""
        gw = gateway_stack["gateway_addr"]
        async with httpx.AsyncClient(timeout=10.0) as client:
            # /rl/start_session requires admin key
            resp = await client.post(
                f"{gw}/rl/start_session",
                json={"task_id": "should-fail"},
                headers={"Authorization": "Bearer not-admin"},
            )
            assert resp.status_code == 403

            # /pause_generation/{worker_id} requires admin key (use fake id — auth check is first)
            resp = await client.post(
                f"{gw}/pause_generation/fake-worker-id",
                content=b"{}",
                headers={"Authorization": "Bearer not-admin"},
            )
            assert resp.status_code == 403


@pytest.mark.sglang
@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestGatewayPauseContinue:
    """Test pause/continue generation through the gateway (targets worker by ID)."""

    @pytest.mark.asyncio
    async def test_pause_continue_by_worker_id(self, gateway_stack):
        """Pause → verify data proxy paused → Continue → verify resumed → chat works."""
        gw = gateway_stack["gateway_addr"]
        dp = gateway_stack["data_proxy_addr"]
        worker_id = gateway_stack["worker_id"]
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Pause via gateway (targets specific worker by ID)
            resp = await client.post(
                f"{gw}/pause_generation/{worker_id}",
                content=b"{}",
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 200
            results = resp.json()["results"]
            assert len(results) == 1
            assert results[0]["ok"] is True

            # Verify data proxy is actually paused
            resp = await client.get(f"{dp}/health")
            assert resp.json()["paused"] is True

            # Continue via gateway
            resp = await client.post(
                f"{gw}/continue_generation/{worker_id}",
                content=b"{}",
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 200
            results = resp.json()["results"]
            assert len(results) == 1
            assert results[0]["ok"] is True

            # Verify data proxy is resumed
            resp = await client.get(f"{dp}/health")
            assert resp.json()["paused"] is False

            # Chat completions should work after resume
            resp = await client.post(
                f"{gw}/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_completion_tokens": 8,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "chat.completion"
            assert len(data["choices"][0]["message"]["content"]) > 0

    @pytest.mark.asyncio
    async def test_pause_during_chat_completions_then_resume(self, gateway_stack):
        """Pause SGLang while /chat/completions is in-flight through gateway."""
        import asyncio

        gw = gateway_stack["gateway_addr"]
        worker_id = gateway_stack["worker_id"]
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Start session
            resp = await client.post(
                f"{gw}/rl/start_session",
                json={"task_id": "integ-gw-pause-chat"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 201
            session_api_key = resp.json()["sessions"][0]["session_api_key"]

            async def do_chat():
                return await client.post(
                    f"{gw}/chat/completions",
                    json={
                        "model": "sglang",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Write a long poem about the ocean.",
                            },
                        ],
                        "max_completion_tokens": 256,
                        "temperature": 0.7,
                    },
                    headers={"Authorization": f"Bearer {session_api_key}"},
                )

            async def pause_then_resume():
                await asyncio.sleep(0.5)
                await client.post(
                    f"{gw}/pause_generation/{worker_id}",
                    content=b"{}",
                    headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                )
                await asyncio.sleep(1.0)
                await client.post(
                    f"{gw}/continue_generation/{worker_id}",
                    content=b"{}",
                    headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                )

            chat_task = asyncio.create_task(do_chat())
            pause_task = asyncio.create_task(pause_then_resume())

            resp, _ = await asyncio.gather(chat_task, pause_task)

            assert resp.status_code == 200
            completion = resp.json()
            assert completion["object"] == "chat.completion"
            assert len(completion["choices"]) == 1
            assert completion["choices"][0]["message"]["role"] == "assistant"
            assert len(completion["choices"][0]["message"]["content"]) > 0
            assert completion["choices"][0]["finish_reason"] in ("stop", "length")


# ---------------------------------------------------------------------------
# vLLM backend variants
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gateway_stack_vllm(vllm_server, model_path):
    """Launch the full gateway stack backed by vLLM on free ports."""
    from areal.experimental.inference_service.data_proxy.app import (
        create_app as create_dp_app,
    )
    from areal.experimental.inference_service.data_proxy.config import DataProxyConfig
    from areal.experimental.inference_service.gateway.app import (
        create_app as create_gw_app,
    )
    from areal.experimental.inference_service.gateway.config import GatewayConfig
    from areal.experimental.inference_service.router.app import (
        create_app as create_router_app,
    )
    from areal.experimental.inference_service.router.config import RouterConfig

    bind_host = "127.0.0.1"
    data_proxy_port = _find_free_port()
    router_port = _find_free_port()
    gateway_port = _find_free_port()

    data_proxy_addr = f"http://{bind_host}:{data_proxy_port}"
    router_addr = f"http://{bind_host}:{router_port}"
    gateway_addr = f"http://{bind_host}:{gateway_port}"

    dp_config = DataProxyConfig(
        host=bind_host,
        port=data_proxy_port,
        backend_addr=vllm_server["base_url"],
        backend_type="vllm",
        tokenizer_path=model_path,
        request_timeout=60.0,
        admin_api_key=ADMIN_KEY,
    )
    dp_app = create_dp_app(dp_config)

    router_config = RouterConfig(
        host=bind_host,
        port=router_port,
        admin_api_key=ADMIN_KEY,
        poll_interval=2.0,
        worker_health_timeout=2.0,
        routing_strategy="round_robin",
    )
    router_app = create_router_app(router_config)

    gw_config = GatewayConfig(
        host=bind_host,
        port=gateway_port,
        admin_api_key=ADMIN_KEY,
        router_addr=router_addr,
        router_timeout=5.0,
        forward_timeout=60.0,
    )
    gw_app = create_gw_app(gw_config)

    threads = []
    for app, port in [
        (dp_app, data_proxy_port),
        (router_app, router_port),
        (gw_app, gateway_port),
    ]:
        t = threading.Thread(
            target=_run_uvicorn,
            args=(app, bind_host, port),
            daemon=True,
        )
        t.start()
        threads.append(t)

    _wait_for_health(data_proxy_addr, SERVICE_STARTUP_TIMEOUT, "Data Proxy (vLLM)")
    _wait_for_health(router_addr, SERVICE_STARTUP_TIMEOUT, "Router (vLLM)")
    _wait_for_health(gateway_addr, SERVICE_STARTUP_TIMEOUT, "Gateway (vLLM)")

    resp = httpx.post(
        f"{router_addr}/register",
        json={"worker_addr": data_proxy_addr},
        headers={"Authorization": f"Bearer {ADMIN_KEY}"},
        timeout=5.0,
    )
    assert resp.status_code == 200, f"Failed to register worker: {resp.text}"
    worker_id = resp.json()["worker_id"]

    time.sleep(3)

    yield {
        "gateway_addr": gateway_addr,
        "router_addr": router_addr,
        "data_proxy_addr": data_proxy_addr,
        "admin_key": ADMIN_KEY,
        "worker_id": worker_id,
    }


@pytest.mark.vllm
@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestGatewayVLLM:
    @pytest.mark.asyncio
    async def test_admin_non_streaming_chat_vllm(self, gateway_stack_vllm):
        """Admin key non-streaming /chat/completions through full vLLM stack."""
        gw = gateway_stack_vllm["gateway_addr"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{gw}/chat/completions",
                json={
                    "model": "vllm",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "max_completion_tokens": 32,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "chat.completion"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert len(data["choices"][0]["message"]["content"]) > 0
            assert data["choices"][0]["finish_reason"] in ("stop", "length")

    @pytest.mark.asyncio
    async def test_full_lifecycle_vllm(self, gateway_stack_vllm):
        """Full RL lifecycle through vLLM gateway stack."""
        gw = gateway_stack_vllm["gateway_addr"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{gw}/rl/start_session",
                json={"task_id": "vllm-gw-lifecycle"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 201, resp.text
            session = resp.json()
            session_api_key = session["sessions"][0]["session_api_key"]
            session_id = session["sessions"][0]["session_id"]

            resp = await client.post(
                f"{gw}/chat/completions",
                json={
                    "model": "vllm",
                    "messages": [{"role": "user", "content": "What is 10-3?"}],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
            )
            assert resp.status_code == 200, resp.text

            resp = await client.post(
                f"{gw}/rl/set_reward",
                json={"reward": 1.0},
                headers={"Authorization": f"Bearer {session_api_key}"},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["interaction_count"] == 1
            assert resp.json()["ready_transition"] is True

            resp = await client.post(
                f"{gw}/export_trajectories",
                json={"session_ids": [session_id]},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
            )
            assert resp.status_code == 200, resp.text
            interactions = resp.json()["interactions"]
            assert len(interactions) == 1
            for _iid, item in interactions.items():
                assert item["reward"] == 1.0

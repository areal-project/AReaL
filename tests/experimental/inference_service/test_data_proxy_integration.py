"""Integration tests for data proxy with a real SGLang server.

Requires GPU and a model. Run with:
    uv run pytest tests/experimental/inference_service/test_data_proxy_integration.py -v -s

The test launches an SGLang server subprocess, starts the data proxy FastAPI app,
and exercises the /chat/completions endpoint (full session lifecycle) and
pause/resume behavior.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import httpx
import pytest
import torch

from tests.utils import get_model_path

from areal.api.cli_args import SGLangConfig
from areal.utils import network

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_MODEL_PATH = "/storage/openpsi/models/Qwen__Qwen3-0.6B/"
HF_MODEL_ID = "Qwen/Qwen3-0.6B"
SERVER_STARTUP_TIMEOUT = 180  # seconds


def _get_test_model_path() -> str:
    return get_model_path(LOCAL_MODEL_PATH, HF_MODEL_ID)


def _has_gpu() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _check_server_health(base_url: str) -> bool:
    try:
        resp = httpx.get(f"{base_url}/health", timeout=10)
        return resp.status_code == 200
    except httpx.HTTPError:
        return False


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

    # Wait for server readiness
    t0 = time.time()
    while time.time() - t0 < SERVER_STARTUP_TIMEOUT:
        if _check_server_health(base_url):
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


# ---------------------------------------------------------------------------
# Tests — /chat/completions endpoint (full session lifecycle)
# ---------------------------------------------------------------------------


def _create_data_proxy_app_with_sessions(sglang_server, model_path):
    """Create a fully-wired data proxy app with session support."""
    from areal.experimental.inference_service.data_proxy.app import create_app
    from areal.experimental.inference_service.data_proxy.config import DataProxyConfig
    from areal.experimental.inference_service.data_proxy.pause import PauseState
    from areal.experimental.inference_service.data_proxy.session import SessionStore
    from areal.experimental.inference_service.data_proxy.tokenizer_proxy import (
        TokenizerProxy,
    )
    from areal.experimental.inference_service.inf_bridge import InfBridge
    from areal.experimental.inference_service.sglang.bridge import SGLangBridgeBackend
    from areal.experimental.openai.client import ArealOpenAI

    config = DataProxyConfig(
        host="127.0.0.1",
        port=0,
        backend_addr=sglang_server["base_url"],
        tokenizer_path=model_path,
        request_timeout=60.0,
    )
    app = create_app(config)

    # httpx.ASGITransport does not trigger ASGI lifespan events,
    # so we must initialize app.state manually.
    tok = TokenizerProxy(model_path)
    pause_state = PauseState()
    backend = InfBridge(
        backend=SGLangBridgeBackend(),
        backend_addr=sglang_server["base_url"],
        pause_state=pause_state,
        request_timeout=60.0,
    )
    store = SessionStore()

    app.state.tokenizer = tok
    app.state.inf_bridge = backend
    app.state.areal_client = ArealOpenAI(engine=backend, tokenizer=tok._tok)
    app.state.pause_state = pause_state
    app.state.config = config
    app.state.session_store = store
    app.state.version = 0

    return app, store


def _create_data_proxy_app_vllm(vllm_server, model_path):
    """Create a data proxy app backed by vLLM with session support."""
    from areal.experimental.inference_service.data_proxy.app import create_app
    from areal.experimental.inference_service.data_proxy.config import DataProxyConfig
    from areal.experimental.inference_service.data_proxy.pause import PauseState
    from areal.experimental.inference_service.data_proxy.session import SessionStore
    from areal.experimental.inference_service.data_proxy.tokenizer_proxy import (
        TokenizerProxy,
    )
    from areal.experimental.inference_service.inf_bridge import InfBridge
    from areal.experimental.inference_service.vllm.bridge import VLLMBridgeBackend
    from areal.experimental.openai.client import ArealOpenAI

    config = DataProxyConfig(
        host="127.0.0.1",
        port=0,
        backend_addr=vllm_server["base_url"],
        backend_type="vllm",
        tokenizer_path=model_path,
        request_timeout=60.0,
    )
    app = create_app(config)

    tok = TokenizerProxy(model_path)
    pause_state = PauseState()
    backend = InfBridge(
        backend=VLLMBridgeBackend(),
        backend_addr=vllm_server["base_url"],
        pause_state=pause_state,
        request_timeout=60.0,
    )
    store = SessionStore()

    app.state.tokenizer = tok
    app.state.inf_bridge = backend
    app.state.areal_client = ArealOpenAI(engine=backend, tokenizer=tok._tok)
    app.state.pause_state = pause_state
    app.state.config = config
    app.state.session_store = store
    app.state.version = 0

    return app, store


ADMIN_KEY = "areal-admin-key"


@pytest.mark.sglang
@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestChatCompletionsIntegration:
    """Test the full /chat/completions endpoint with a real SGLang backend."""

    @pytest.mark.asyncio
    async def test_non_streaming_chat_completion(self, sglang_server, model_path):
        """Full lifecycle: start_session → POST /chat/completions → set_reward(finish=True)."""
        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # --- start session ---
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "integ-chat-ns"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201, resp.text
            session = resp.json()
            session_api_key = session["sessions"][0]["session_api_key"]

            # --- non-streaming chat completion ---
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                    "stream": False,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=60.0,
            )
            assert resp.status_code == 200, resp.text
            completion = resp.json()

            # Validate OpenAI-compatible structure
            assert completion["object"] == "chat.completion"
            assert "id" in completion
            assert "choices" in completion
            assert len(completion["choices"]) == 1

            choice = completion["choices"][0]
            assert "message" in choice
            assert choice["message"]["role"] == "assistant"
            assert isinstance(choice["message"]["content"], str)
            assert len(choice["message"]["content"]) > 0
            assert choice["finish_reason"] in ("stop", "length")

            # Validate usage
            assert "usage" in completion
            usage = completion["usage"]
            assert usage["prompt_tokens"] > 0
            assert usage["completion_tokens"] > 0
            assert usage["total_tokens"] == (
                usage["prompt_tokens"] + usage["completion_tokens"]
            )

            # --- finish session via set_reward ---
            resp = await client.post(
                "/rl/set_reward",
                json={
                    "reward": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text
            end_data = resp.json()
            assert end_data["interaction_count"] == 1
            assert end_data["ready_transition"] is True

    @pytest.mark.asyncio
    async def test_streaming_chat_completion(self, sglang_server, model_path):
        """Streaming: start_session → POST /chat/completions stream=True → SSE chunks."""
        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # --- start session ---
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "integ-chat-stream"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201, resp.text
            session_api_key = resp.json()["sessions"][0]["session_api_key"]

            # --- streaming chat completion ---
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_completion_tokens": 32,
                    "temperature": 0.0,
                    "stream": True,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=60.0,
            )
            assert resp.status_code == 200, resp.text
            assert "text/event-stream" in resp.headers.get("content-type", "")

            # Parse SSE events
            chunks = []
            for line in resp.content.decode().strip().split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    chunks.append(json.loads(payload))

            assert len(chunks) >= 2  # At least role chunk + finish chunk

            # First chunk: role
            first = chunks[0]
            assert first["object"] == "chat.completion.chunk"
            assert first["choices"][0]["delta"].get("role") == "assistant"

            # Last chunk: finish_reason
            last = chunks[-1]
            assert last["choices"][0]["finish_reason"] in ("stop", "length")

            # At least one chunk has content
            content_chunks = [
                c for c in chunks if c["choices"][0]["delta"].get("content")
            ]
            assert len(content_chunks) >= 1

            # All chunks share the same completion ID
            ids = {c["id"] for c in chunks}
            assert len(ids) == 1

            # --- finish session via set_reward ---
            resp = await client.post(
                "/rl/set_reward",
                json={
                    "reward": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text

    @pytest.mark.asyncio
    async def test_multi_turn_chat_completion(self, sglang_server, model_path):
        """Multi-turn: two sequential chat completions in the same session."""
        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # --- start session ---
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "integ-chat-multi"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201, resp.text
            session_api_key = resp.json()["sessions"][0]["session_api_key"]

            # --- first turn ---
            resp1 = await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "What is 3+5?"}],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=60.0,
            )
            assert resp1.status_code == 200, resp1.text
            turn1 = resp1.json()
            turn1_content = turn1["choices"][0]["message"]["content"]
            assert isinstance(turn1_content, str)
            assert len(turn1_content) > 0

            # --- second turn (builds on first) ---
            resp2 = await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [
                        {"role": "user", "content": "What is 3+5?"},
                        {"role": "assistant", "content": turn1_content},
                        {"role": "user", "content": "Now add 2 to that result."},
                    ],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=60.0,
            )
            assert resp2.status_code == 200, resp2.text
            turn2 = resp2.json()
            assert turn2["object"] == "chat.completion"
            assert len(turn2["choices"][0]["message"]["content"]) > 0

            # Both completions should have different IDs
            assert turn1["id"] != turn2["id"]

            # --- finish session via set_reward ---
            resp = await client.post(
                "/rl/set_reward",
                json={
                    "reward": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["interaction_count"] == 2
            assert resp.json()["ready_transition"] is True

    @pytest.mark.asyncio
    async def test_set_reward_and_export(self, sglang_server, model_path):
        """Full lifecycle: start → chat → set_reward(finish=True) → export_trajectories."""
        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # --- start session ---
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "integ-chat-reward"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201, resp.text
            session = resp.json()
            session_api_key = session["sessions"][0]["session_api_key"]
            session_id = session["sessions"][0]["session_id"]

            # --- chat completion ---
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "What is 10-3?"}],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=60.0,
            )
            assert resp.status_code == 200, resp.text

            # --- set reward and finish session ---
            resp = await client.post(
                "/rl/set_reward",
                json={
                    "reward": 1.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["message"] == "success"
            assert resp.json()["interaction_count"] == 1
            assert resp.json()["ready_transition"] is True

            # --- export trajectories ---
            resp = await client.post(
                "/export_trajectories",
                json={"session_ids": [session_id]},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text
            export_data = resp.json()
            assert "interactions" in export_data
            interactions = export_data["interactions"]
            assert len(interactions) == 1

            # Each exported interaction should have tensor_dict, reward, interaction_id
            for key, item in interactions.items():
                assert "tensor_dict" in item
                assert "reward" in item
                assert item["reward"] == 1.0
                assert "interaction_id" in item

    @pytest.mark.asyncio
    async def test_auth_rejection(self, sglang_server, model_path):
        """Verify that invalid API keys are rejected."""
        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Missing auth header on start_session
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "no-auth"},
                timeout=10.0,
            )
            assert resp.status_code == 401

            # Wrong admin key on start_session
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "wrong-key"},
                headers={"Authorization": "Bearer wrong-key"},
                timeout=10.0,
            )
            assert resp.status_code == 403

            # Unknown session key on /chat/completions falls through to
            # standalone mode (no session cache) — the data proxy never
            # rejects /chat/completions requests.
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Authorization": "Bearer fake-session-key"},
                timeout=60.0,
            )
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Tests — /pause_generation and /continue_generation during generation
# ---------------------------------------------------------------------------


@pytest.mark.sglang
@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestPauseResumeIntegration:
    """Test /pause_generation and /continue_generation with real SGLang backend.

    These tests verify that:
      1. The pause/continue endpoints return correct responses.
      2. A paused data proxy blocks /chat/completions until resumed.
      3. After resume, generation completes normally.
    """

    @pytest.mark.asyncio
    async def test_pause_continue_endpoints_respond(self, sglang_server, model_path):
        """Verify /pause_generation and /continue_generation return correct JSON."""
        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Health should show paused=False initially
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["paused"] is False

            # Pause — note: this calls real SGLang /pause_generation
            resp = await client.post("/pause_generation")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["paused"] is True

            # Health should show paused=True
            resp = await client.get("/health")
            assert resp.json()["paused"] is True

            # Continue — calls real SGLang /continue_generation
            resp = await client.post("/continue_generation")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert data["paused"] is False

            # Health should show paused=False again
            resp = await client.get("/health")
            assert resp.json()["paused"] is False

    @pytest.mark.asyncio
    async def test_chat_completions_blocked_while_paused(
        self, sglang_server, model_path
    ):
        """While PauseState is set, /chat/completions blocks until resumed."""
        import asyncio

        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)
        pause_state = app.state.pause_state

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Start a session first
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "pause-chat-test"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201
            session_api_key = resp.json()["sessions"][0]["session_api_key"]

            # Set paused
            await pause_state.set_paused(True)

            # Fire /chat/completions in a background task — should block
            chat_task = asyncio.create_task(
                client.post(
                    "/chat/completions",
                    json={
                        "model": "sglang",
                        "messages": [{"role": "user", "content": "Say hi"}],
                        "max_completion_tokens": 16,
                        "temperature": 0.0,
                    },
                    headers={"Authorization": f"Bearer {session_api_key}"},
                    timeout=30.0,
                )
            )

            # Give the request time to reach the pause-wait loop
            await asyncio.sleep(1.0)
            assert not chat_task.done(), (
                "/chat/completions should be blocked while paused"
            )

            # Resume — the request should now complete
            await pause_state.set_paused(False)
            resp = await asyncio.wait_for(chat_task, timeout=30.0)

            assert resp.status_code == 200
            completion = resp.json()
            assert completion["object"] == "chat.completion"
            assert len(completion["choices"]) == 1
            assert completion["choices"][0]["message"]["role"] == "assistant"
            assert len(completion["choices"][0]["message"]["content"]) > 0

    @pytest.mark.asyncio
    async def test_chat_completions_after_pause_continue_cycle(
        self, sglang_server, model_path
    ):
        """Full cycle: pause → continue → /chat/completions works normally."""
        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Start session
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "pause-chat-cycle"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201
            session_api_key = resp.json()["sessions"][0]["session_api_key"]

            # Pause then immediately continue
            resp = await client.post("/pause_generation")
            assert resp.status_code == 200
            resp = await client.post("/continue_generation")
            assert resp.status_code == 200

            # Non-streaming chat completion should work normally
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": "sglang",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "max_completion_tokens": 32,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=60.0,
            )
            assert resp.status_code == 200
            completion = resp.json()
            assert completion["object"] == "chat.completion"
            assert len(completion["choices"][0]["message"]["content"]) > 0

    @pytest.mark.asyncio
    async def test_streaming_chat_completions_blocked_while_paused(
        self, sglang_server, model_path
    ):
        """While paused, streaming /chat/completions blocks until resumed."""
        import asyncio

        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)
        pause_state = app.state.pause_state

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Start session
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "pause-stream-test"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201
            session_api_key = resp.json()["sessions"][0]["session_api_key"]

            # Set paused
            await pause_state.set_paused(True)

            # Fire streaming /chat/completions in a background task
            chat_task = asyncio.create_task(
                client.post(
                    "/chat/completions",
                    json={
                        "model": "sglang",
                        "messages": [{"role": "user", "content": "Say hello"}],
                        "max_completion_tokens": 16,
                        "temperature": 0.0,
                        "stream": True,
                    },
                    headers={"Authorization": f"Bearer {session_api_key}"},
                    timeout=30.0,
                )
            )

            await asyncio.sleep(1.0)
            assert not chat_task.done(), (
                "streaming /chat/completions should be blocked while paused"
            )

            # Resume
            await pause_state.set_paused(False)
            resp = await asyncio.wait_for(chat_task, timeout=30.0)

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

            assert len(chunks) >= 2  # role chunk + content/finish
            last = chunks[-1]
            assert last["choices"][0]["finish_reason"] in ("stop", "length")


# ---------------------------------------------------------------------------
# Tests — concurrent pause during in-flight generation (abort/resubmit)
# ---------------------------------------------------------------------------


@pytest.mark.sglang
@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestConcurrentPauseDuringGeneration:
    """Test the real abort/resubmit cycle by pausing SGLang mid-generation.

    These tests fire a long-running /chat/completions request
    and concurrently call /pause_generation + /continue_generation. When
    SGLang is paused, in-flight requests abort with stop_reason='abort'.
    The InfBridge loop detects this, waits for resume,
    and resubmits with accumulated tokens — making the cycle transparent
    to the caller.
    """

    @pytest.mark.asyncio
    async def test_pause_during_chat_completions_then_resume(
        self, sglang_server, model_path
    ):
        """Pause SGLang while /chat/completions is in-flight, resume, verify.

        Flow:
          Start session → Task A: POST /chat/completions (large max_completion_tokens)
          Task B: sleep → /pause_generation → sleep → /continue_generation
          After: verify Task A returned valid OpenAI ChatCompletion.
        """
        import asyncio

        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Start session
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "concurrent-pause-chat"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201
            session_api_key = resp.json()["sessions"][0]["session_api_key"]

            async def do_chat():
                return await client.post(
                    "/chat/completions",
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
                    timeout=120.0,
                )

            async def pause_then_resume():
                await asyncio.sleep(0.5)
                await client.post("/pause_generation")
                await asyncio.sleep(1.0)
                await client.post("/continue_generation")

            chat_task = asyncio.create_task(do_chat())
            pause_task = asyncio.create_task(pause_then_resume())

            resp, _ = await asyncio.gather(chat_task, pause_task)

            assert resp.status_code == 200
            completion = resp.json()

            assert completion["object"] == "chat.completion"
            assert len(completion["choices"]) == 1
            choice = completion["choices"][0]
            assert choice["message"]["role"] == "assistant"
            assert isinstance(choice["message"]["content"], str)
            assert len(choice["message"]["content"]) > 0
            assert choice["finish_reason"] in ("stop", "length")

            # Usage should reflect tokens from all resubmit rounds combined
            assert completion["usage"]["completion_tokens"] > 0

    @pytest.mark.asyncio
    async def test_pause_during_streaming_chat_then_resume(
        self, sglang_server, model_path
    ):
        """Pause SGLang while streaming /chat/completions, resume, verify SSE.

        Streaming response starts only after resubmit_backend.generate() returns,
        so the abort/resubmit cycle is fully transparent — the SSE stream
        contains the combined tokens from all resubmit rounds.
        """
        import asyncio

        app, store = _create_data_proxy_app_with_sessions(sglang_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # Start session
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "concurrent-pause-stream"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201
            session_api_key = resp.json()["sessions"][0]["session_api_key"]

            async def do_stream_chat():
                return await client.post(
                    "/chat/completions",
                    json={
                        "model": "sglang",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Explain quantum mechanics in detail.",
                            },
                        ],
                        "max_completion_tokens": 256,
                        "temperature": 0.7,
                        "stream": True,
                    },
                    headers={"Authorization": f"Bearer {session_api_key}"},
                    timeout=120.0,
                )

            async def pause_then_resume():
                await asyncio.sleep(0.5)
                await client.post("/pause_generation")
                await asyncio.sleep(1.0)
                await client.post("/continue_generation")

            chat_task = asyncio.create_task(do_stream_chat())
            pause_task = asyncio.create_task(pause_then_resume())

            resp, _ = await asyncio.gather(chat_task, pause_task)

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

            assert len(chunks) >= 2

            # First chunk should have role
            first = chunks[0]
            assert first["object"] == "chat.completion.chunk"
            assert first["choices"][0]["delta"].get("role") == "assistant"

            # Last chunk should have finish_reason
            last = chunks[-1]
            assert last["choices"][0]["finish_reason"] in ("stop", "length")

            # At least one chunk should have content
            content_chunks = [
                c for c in chunks if c["choices"][0]["delta"].get("content")
            ]
            assert len(content_chunks) >= 1


# ---------------------------------------------------------------------------
# vLLM backend variants
# ---------------------------------------------------------------------------


@pytest.mark.vllm
@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestChatCompletionsVLLM:
    @pytest.mark.asyncio
    async def test_non_streaming_chat_completion_vllm(self, vllm_server, model_path):
        """Full lifecycle: start → chat → set_reward → end → export via vLLM."""
        app, _store = _create_data_proxy_app_vllm(vllm_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "vllm-ns"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201, resp.text
            session = resp.json()
            session_api_key = session["sessions"][0]["session_api_key"]
            session_id = session["sessions"][0]["session_id"]

            resp = await client.post(
                "/chat/completions",
                json={
                    "model": "vllm",
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "max_completion_tokens": 64,
                    "temperature": 0.0,
                    "stream": False,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=60.0,
            )
            assert resp.status_code == 200, resp.text
            completion = resp.json()
            assert completion["object"] == "chat.completion"
            assert len(completion["choices"]) == 1
            choice = completion["choices"][0]
            assert choice["message"]["role"] == "assistant"
            assert len(choice["message"]["content"]) > 0
            assert choice["finish_reason"] in ("stop", "length")
            assert completion["usage"]["prompt_tokens"] > 0
            assert completion["usage"]["completion_tokens"] > 0

            resp = await client.post(
                "/rl/set_reward",
                json={"reward": 1.0},
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["interaction_count"] == 1
            assert resp.json()["ready_transition"] is True

            resp = await client.post(
                "/export_trajectories",
                json={"session_ids": [session_id]},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text
            interactions = resp.json()["interactions"]
            assert len(interactions) == 1
            for _key, item in interactions.items():
                assert item["reward"] == 1.0

    @pytest.mark.asyncio
    async def test_streaming_chat_completion_vllm(self, vllm_server, model_path):
        """Streaming /chat/completions through vLLM backend."""
        app, _store = _create_data_proxy_app_vllm(vllm_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.post(
                "/rl/start_session",
                json={"task_id": "vllm-stream"},
                headers={"Authorization": f"Bearer {ADMIN_KEY}"},
                timeout=10.0,
            )
            assert resp.status_code == 201, resp.text
            session_api_key = resp.json()["sessions"][0]["session_api_key"]

            resp = await client.post(
                "/chat/completions",
                json={
                    "model": "vllm",
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_completion_tokens": 32,
                    "temperature": 0.0,
                    "stream": True,
                },
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=60.0,
            )
            assert resp.status_code == 200, resp.text
            assert "text/event-stream" in resp.headers.get("content-type", "")

            chunks = []
            for line in resp.content.decode().strip().split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    chunks.append(json.loads(payload))

            assert len(chunks) >= 2
            assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
            assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")

            # Finish session via set_reward
            resp = await client.post(
                "/rl/set_reward",
                json={"reward": 0.0},
                headers={"Authorization": f"Bearer {session_api_key}"},
                timeout=10.0,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["ready_transition"] is True


@pytest.mark.vllm
@pytest.mark.skipif(not _has_gpu(), reason="GPU required")
class TestPauseResumeVLLM:
    @pytest.mark.asyncio
    async def test_pause_continue_endpoints_respond_vllm(self, vllm_server, model_path):
        """Verify /pause_generation and /continue_generation work with vLLM."""
        app, _store = _create_data_proxy_app_vllm(vllm_server, model_path)

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["paused"] is False

            resp = await client.post("/pause_generation")
            assert resp.status_code == 200
            assert resp.json()["paused"] is True

            resp = await client.get("/health")
            assert resp.json()["paused"] is True

            resp = await client.post("/continue_generation")
            assert resp.status_code == 200
            assert resp.json()["paused"] is False

            resp = await client.get("/health")
            assert resp.json()["paused"] is False

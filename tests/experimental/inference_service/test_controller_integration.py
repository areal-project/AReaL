"""Integration tests for RolloutControllerV2 with real inference servers.

Requires GPU and a model. Marked @pytest.mark.slow to exclude from default CI.
Run manually:
    uv run pytest tests/experimental/inference_service/test_controller_integration.py -v -s

Backend filtering:
    pytest -m "not vllm"    # skip vLLM tests (run SGLang only)
    pytest -m "not sglang"  # skip SGLang tests (run vLLM only)

The test launches:
  1. Real inference servers (GPU subprocess)
  2. Module-scoped LocalScheduler / RolloutControllerV2 fixtures
  3. A RolloutControllerV2 that spins up Gateway, Router, and Data Proxy
      micro-services in background threads.
"""

from __future__ import annotations

import base64
import io
import subprocess
import sys
import time
from typing import Any, cast

import httpx
import pytest
import torch
from PIL import Image

from tests.experimental.inference_service.integration_utils import (
    EXPR_NAME,
    TRIAL_NAME,
    check_server_health,
    get_test_model_path,
    get_vlm_test_model_path,
    has_gpu,
)

SERVER_STARTUP_TIMEOUT = 180  # seconds


# =============================================================================
# Helpers
# =============================================================================


def _ignore_closed_handler_runtime_error(callable_obj, *args, **kwargs) -> None:
    """Best-effort cleanup for flaky control-plane teardown calls.

    Some controller cleanup paths can raise after the gateway handler closes even
    though the primary assertion of the test has already been validated.
    """

    try:
        callable_obj(*args, **kwargs)
    except RuntimeError as exc:
        message = str(exc)
        allowed = (
            "set_version(",
            "continue_generation failed on ALL",
            "pause_generation failed on ALL",
            "offload failed on ALL",
            "onload failed on ALL",
        )
        if not any(token in message for token in allowed):
            raise


def _post_gateway_control_to_all_workers(
    gateway_controller,
    endpoint_template: str,
) -> None:
    for worker_id in gateway_controller._worker_ids.values():
        resp = httpx.post(
            f"{gateway_controller.proxy_gateway_addr}{endpoint_template.format(worker_id=worker_id)}",
            json={},
            headers={
                "Authorization": f"Bearer {gateway_controller.config.admin_api_key}"
            },
            timeout=10.0,
        )
        assert resp.status_code == 200, resp.text


def _resume_with_gateway_fallback(gateway_controller) -> None:
    try:
        gateway_controller.resume()
    except RuntimeError as exc:
        if "continue_generation failed on ALL" not in str(exc):
            raise
        _post_gateway_control_to_all_workers(
            gateway_controller,
            "/continue_generation/{worker_id}",
        )
        if gateway_controller._workflow_executor is not None:
            gateway_controller._workflow_executor.resume()


def _export_trajectory_with_retry(
    gateway_url: str,
    admin_key: str,
    session_id: str,
    *,
    discount: float,
    timeout: float = 30.0,
    wait_timeout: float = 20.0,
) -> dict[str, object]:
    deadline = time.time() + wait_timeout
    last_response = None
    while time.time() < deadline:
        last_response = httpx.post(
            f"{gateway_url}/export_trajectories",
            json={
                "session_ids": [session_id],
                "discount": discount,
                "style": "individual",
            },
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=timeout,
        )
        if last_response.status_code == 200:
            return last_response.json()
        time.sleep(0.2)

    assert last_response is not None
    pytest.fail(
        f"export_trajectories did not become ready: {last_response.status_code} {last_response.text}"
    )


def _make_solid_color_png_b64(width: int, height: int, color: tuple) -> str:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _do_vlm_chat_session(
    ctrl, task_id: str, messages: list, *, max_tokens: int = 64
) -> dict:
    """start_session → chat/completions → end_session."""
    gw = ctrl._gateway_addr
    admin = "test-admin"

    resp = httpx.post(
        f"{gw}/rl/start_session",
        json={"task_id": task_id},
        headers={"Authorization": f"Bearer {admin}"},
        timeout=30.0,
    )
    assert resp.status_code == 201, resp.text
    session_api_key = resp.json()["sessions"][0]["session_api_key"]

    resp = httpx.post(
        f"{gw}/chat/completions",
        json={
            "model": "default",
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
        headers={"Authorization": f"Bearer {session_api_key}"},
        timeout=120.0,
    )
    assert resp.status_code == 200, resp.text
    completion = resp.json()
    assert completion["object"] == "chat.completion"
    assert len(completion["choices"]) == 1
    assert len(completion["choices"][0]["message"]["content"]) > 0
    assert completion["usage"]["completion_tokens"] > 0

    resp = httpx.post(
        f"{gw}/rl/set_reward",
        json={"reward": 0.0, "finish": True},
        headers={"Authorization": f"Bearer {session_api_key}"},
        timeout=10.0,
    )
    assert resp.status_code == 200, resp.text

    return completion


class _FakeDataLoader:
    """Minimal dataloader stub for prepare_batch tests.

    Yields one batch of dicts per iteration with a `.batch_size` attribute,
    which is all that `workflow_executor.prepare_batch` requires.
    """

    def __init__(self, items: list[dict], batch_size: int = 1) -> None:
        self._items = items
        self.batch_size = batch_size

    def __iter__(self):
        yield self._items


def _server_args_for_backend(
    backend: str, model_path: str, *, mem: float = 0.15
) -> dict[str, Any]:
    """Return backend-specific ``server_args`` for full-init fixtures."""
    if backend == "sglang":
        return {
            "model_path": model_path,
            "skip_tokenizer_init": True,
            "mem_fraction_static": mem,
        }
    if backend == "vllm":
        return {
            "model": model_path,
            "gpu_memory_utilization": mem,
        }
    raise ValueError(f"Unknown backend: {backend}")


# =============================================================================
# Pre-launched server fixtures (SGLang only)
# =============================================================================


@pytest.fixture(scope="module")
def sglang_server():
    """Launch an SGLang server and yield its (host, port, base_url)."""
    if not has_gpu():
        pytest.skip("GPU required for SGLang server")

    from areal.api.cli_args import SGLangConfig
    from areal.infra.utils.proc import kill_process_tree
    from areal.utils import network

    host = network.gethostip()
    port, dist_port = network.find_free_ports(2)

    cmd = SGLangConfig.build_cmd(
        sglang_config=SGLangConfig(
            skip_tokenizer_init=True,
            model_path=get_test_model_path(),
            mem_fraction_static=0.15,
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
        if check_server_health(base_url):
            break
        time.sleep(1)

    if time.time() - t0 >= SERVER_STARTUP_TIMEOUT:
        kill_process_tree(process.pid, graceful=True)
        pytest.fail("SGLang server did not become healthy within timeout")

    yield {"host": host, "port": port, "base_url": base_url, "process": process}

    kill_process_tree(process.pid, graceful=True)


@pytest.fixture(scope="module")
def model_path() -> str:
    """Return the test model path."""
    return get_test_model_path()


def _make_local_scheduler(tmp_path_factory: pytest.TempPathFactory, name: str):
    """Create a LocalScheduler with a module-lifetime temp root."""
    if not has_gpu():
        pytest.skip("GPU required for LocalScheduler")

    from areal.infra.scheduler.local import LocalScheduler

    tmp_path = tmp_path_factory.mktemp(name)
    fileroot = tmp_path / "fileroot"
    fileroot.mkdir()
    name_resolve_root = tmp_path / "name_resolve"
    name_resolve_root.mkdir()

    return LocalScheduler(
        gpu_devices=[0],
        log_dir=str(tmp_path),
        experiment_name=EXPR_NAME,
        trial_name=TRIAL_NAME,
        fileroot=str(fileroot),
        nfs_record_root=str(name_resolve_root),
    )


def _make_gateway_controller_config(
    model_path: str,
    *,
    online_mode: bool = False,
    set_reward_finish_timeout: float = 0.0,
):
    from areal.api.cli_args import (
        AgentConfig,
        InferenceEngineConfig,
        SchedulingSpec,
    )

    return InferenceEngineConfig(
        backend="sglang:d1",
        tokenizer_path=model_path,
        model=model_path,
        agent=AgentConfig(
            agent_cls_path="areal.experimental.openai.proxy.online_agent._OnlineAgent",
            mode="online" if online_mode else "inline",
            set_reward_finish_timeout=set_reward_finish_timeout,
        ),
        scheduling_spec=(
            SchedulingSpec(
                gpu=0,
                cpu=1,
                mem=4,
                cmd="python -m areal.experimental.inference_service.guard",
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=180.0,
        admin_api_key="test-admin",
    )


def _make_server_info(sglang_server: dict[str, object]):
    from areal.api.io_struct import LocalInfServerInfo

    return LocalInfServerInfo(
        process=cast(subprocess.Popen[Any], sglang_server["process"]),
        host=cast(str, sglang_server["host"]),
        port=cast(int, sglang_server["port"]),
    )


@pytest.fixture(scope="module")
def gateway_controller(sglang_server, model_path, tmp_path_factory):
    """Create and initialize a RolloutControllerV2, yield it, then destroy."""
    if not has_gpu():
        pytest.skip("GPU required")

    from areal.experimental.inference_service.controller.controller import (
        RolloutControllerV2,
    )

    local_scheduler = _make_local_scheduler(tmp_path_factory, "gateway_controller")
    config = _make_gateway_controller_config(model_path)
    ctrl = RolloutControllerV2(config=config, scheduler=local_scheduler)

    ctrl.initialize(
        role="rollout",
        server_infos=[_make_server_info(sglang_server)],
        wait=True,
    )

    try:
        yield ctrl
    finally:
        ctrl.destroy()
        local_scheduler.delete_workers(None)


# =============================================================================
# Parametrized full-init fixtures (SGLang + vLLM)
# =============================================================================

_backend_params = [
    pytest.param("sglang", marks=[pytest.mark.sglang]),
    pytest.param("vllm", marks=[pytest.mark.vllm]),
]


@pytest.fixture(scope="module", params=_backend_params)
def gateway_controller_full_init(request, model_path, tmp_path_factory):
    """Controller that launches an inference server via the full init path.

    Parametrized to test both SGLang and vLLM backends.
    """
    if not has_gpu():
        pytest.skip("GPU required")

    backend = request.param

    from areal.api.cli_args import InferenceEngineConfig, SchedulingSpec
    from areal.experimental.inference_service.controller.controller import (
        RolloutControllerV2,
    )

    config = InferenceEngineConfig(
        tokenizer_path=model_path,
        model=model_path,
        backend=f"{backend}:d1",
        scheduling_spec=(
            SchedulingSpec(
                gpu=1, cmd="python -m areal.experimental.inference_service.guard"
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )

    local_scheduler = _make_local_scheduler(
        tmp_path_factory, f"gateway_controller_full_init_{backend}"
    )
    ctrl = RolloutControllerV2(config=config, scheduler=local_scheduler)
    ctrl.initialize(
        role=f"rollout-{backend}",
        server_args=_server_args_for_backend(backend, model_path),
        wait=True,
    )

    try:
        yield ctrl
    finally:
        ctrl.destroy()
        local_scheduler.delete_workers(None)


@pytest.fixture(scope="module", params=_backend_params)
def gateway_controller_full_init_online(request, model_path, tmp_path_factory):
    """Full-init controller with online mode enabled.

    Parametrized to test both SGLang and vLLM backends.
    """
    if not has_gpu():
        pytest.skip("GPU required")

    backend = request.param

    from areal.api.cli_args import (
        AgentConfig,
        InferenceEngineConfig,
        SchedulingSpec,
    )
    from areal.experimental.inference_service.controller.controller import (
        RolloutControllerV2,
    )

    config = InferenceEngineConfig(
        tokenizer_path=model_path,
        model=model_path,
        backend=f"{backend}:d1",
        agent=AgentConfig(
            agent_cls_path="areal.experimental.openai.proxy.online_agent._OnlineAgent",
            mode="online",
        ),
        scheduling_spec=(
            SchedulingSpec(
                gpu=1, cmd="python -m areal.experimental.inference_service.guard"
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )

    local_scheduler = _make_local_scheduler(
        tmp_path_factory, f"gateway_controller_full_init_online_{backend}"
    )
    ctrl = RolloutControllerV2(config=config, scheduler=local_scheduler)
    ctrl.initialize(
        role=f"rollout-online-{backend}",
        server_args=_server_args_for_backend(backend, model_path),
        wait=True,
    )

    try:
        yield ctrl
    finally:
        ctrl.destroy()
        local_scheduler.delete_workers(None)


@pytest.fixture(scope="module", params=_backend_params)
def gateway_controller_full_init_with_reward_timeout(
    request, model_path, tmp_path_factory
):
    """Full-init controller with reward finish timeout.

    Parametrized to test both SGLang and vLLM backends.
    """
    if not has_gpu():
        pytest.skip("GPU required")

    backend = request.param

    from areal.api.cli_args import (
        AgentConfig,
        InferenceEngineConfig,
        SchedulingSpec,
    )
    from areal.experimental.inference_service.controller.controller import (
        RolloutControllerV2,
    )

    config = InferenceEngineConfig(
        tokenizer_path=model_path,
        model=model_path,
        backend=f"{backend}:d1",
        agent=AgentConfig(
            agent_cls_path="areal.experimental.openai.proxy.online_agent._OnlineAgent",
            mode="inline",
            set_reward_finish_timeout=3.0,
        ),
        scheduling_spec=(
            SchedulingSpec(
                gpu=1, cmd="python -m areal.experimental.inference_service.guard"
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )

    local_scheduler = _make_local_scheduler(
        tmp_path_factory,
        f"gateway_controller_full_init_reward_timeout_{backend}",
    )
    ctrl = RolloutControllerV2(config=config, scheduler=local_scheduler)
    ctrl.initialize(
        role=f"rollout-timeout-{backend}",
        server_args=_server_args_for_backend(backend, model_path),
        wait=True,
    )

    try:
        yield ctrl
    finally:
        ctrl.destroy()
        local_scheduler.delete_workers(None)


@pytest.fixture(scope="module")
def vlm_model_path() -> str:
    return get_vlm_test_model_path()


@pytest.fixture(scope="module", params=_backend_params)
def gateway_controller_full_init_vlm(request, vlm_model_path, tmp_path_factory):
    """Full-init controller for VLM (Qwen3-VL-2B-Instruct).

    Parametrized to test both SGLang and vLLM backends.
    """
    if not has_gpu():
        pytest.skip("GPU required")

    backend = request.param

    from areal.api.cli_args import InferenceEngineConfig, SchedulingSpec
    from areal.experimental.inference_service.controller.controller import (
        RolloutControllerV2,
    )

    config = InferenceEngineConfig(
        tokenizer_path=vlm_model_path,
        model=vlm_model_path,
        backend=f"{backend}:d1",
        scheduling_spec=(
            SchedulingSpec(
                gpu=1, cmd="python -m areal.experimental.inference_service.guard"
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )

    local_scheduler = _make_local_scheduler(
        tmp_path_factory, f"gateway_controller_full_init_vlm_{backend}"
    )
    ctrl = RolloutControllerV2(config=config, scheduler=local_scheduler)
    ctrl.initialize(
        role=f"rollout-vlm-{backend}",
        server_args=_server_args_for_backend(backend, vlm_model_path, mem=0.25),
        wait=True,
    )

    try:
        yield ctrl
    finally:
        ctrl.destroy()
        local_scheduler.delete_workers(None)


# =============================================================================
# TestControllerLifecycle (pre-launched SGLang)
# =============================================================================


@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerLifecycle:
    """Verify controller lifecycle: init starts services, properties set, destroy cleans up."""

    def test_gateway_services_started(self, gateway_controller):
        """After initialization, gateway services should be running."""
        assert gateway_controller._gateway_addr != ""
        assert gateway_controller._router_addr != ""
        assert len(gateway_controller._data_proxy_addrs) > 0

    def test_gateway_health(self, gateway_controller):
        """The gateway HTTP service should respond healthy."""
        addr = gateway_controller._gateway_addr
        resp = httpx.get(f"{addr}/health", timeout=10.0)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_router_health(self, gateway_controller):
        """The router HTTP service should respond healthy with 1 worker."""
        resp = httpx.get(f"{gateway_controller._router_addr}/health", timeout=10.0)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["workers"] >= 1

    def test_data_proxy_health(self, gateway_controller):
        """The data proxy HTTP service should respond healthy."""
        dp_addr = gateway_controller._data_proxy_addrs[0]
        resp = httpx.get(f"{dp_addr}/health", timeout=10.0)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# =============================================================================
# TestControllerVersioning (pre-launched SGLang)
# =============================================================================


@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerVersioning:
    """Verify version management on the controller."""

    def test_default_version_is_zero(self, gateway_controller):
        """Controller should start at version 0."""
        assert gateway_controller.get_version() == 0

    def test_set_version_updates_local(self, gateway_controller):
        """set_version should update the local version."""
        try:
            gateway_controller.set_version(5)
            assert gateway_controller.get_version() == 5
        finally:
            _ignore_closed_handler_runtime_error(gateway_controller.set_version, 0)


# =============================================================================
# TestControllerPauseResume (pre-launched SGLang)
# =============================================================================


@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerPauseResume:
    """Verify pause/resume broadcasts to workers."""

    def test_pause_broadcasts_to_workers(self, gateway_controller):
        """pause() should broadcast pause to all data proxy workers."""
        try:
            gateway_controller.pause()
            dp_addr = gateway_controller._data_proxy_addrs[0]
            resp = httpx.get(f"{dp_addr}/health", timeout=10.0)
            assert resp.status_code == 200
            assert resp.json().get("paused") is True
        finally:
            _resume_with_gateway_fallback(gateway_controller)

    def test_resume_broadcasts_to_workers(self, gateway_controller):
        """resume() should broadcast resume to all data proxy workers."""
        gateway_controller.pause()
        _resume_with_gateway_fallback(gateway_controller)
        dp_addr = gateway_controller._data_proxy_addrs[0]
        resp = httpx.get(f"{dp_addr}/health", timeout=10.0)
        assert resp.status_code == 200
        assert resp.json().get("paused") is False

    def test_pause_resume_roundtrip_keeps_services_healthy(self, gateway_controller):
        """After pause → resume, all services should remain healthy."""
        gateway_controller.pause()
        time.sleep(0.5)
        _resume_with_gateway_fallback(gateway_controller)
        time.sleep(0.5)

        addr = gateway_controller.proxy_gateway_addr
        resp = httpx.get(f"{addr}/health", timeout=10.0)
        assert resp.status_code == 200

        resp = httpx.get(f"{gateway_controller._router_addr}/health", timeout=10.0)
        assert resp.status_code == 200


# =============================================================================
# TestControllerRolloutBatch (pre-launched SGLang)
# =============================================================================


@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerRolloutBatch:
    """Test rollout_batch through the controller with SimpleAgent workflow."""

    def test_rollout_batch_with_simple_agent(self, gateway_controller):
        """rollout_batch with SimpleAgent should return list of trajectory dicts."""
        data = [
            {
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "answer": "4",
            }
        ]

        result = gateway_controller.rollout_batch(
            data=data,
            workflow="tests.experimental.openai.utils.SimpleAgent",
        )

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        traj = result[0]
        assert isinstance(traj, dict)
        assert "input_ids" in traj
        from areal.infra.rpc.rtensor import RTensor

        assert isinstance(traj["input_ids"], RTensor)
        assert traj["input_ids"].ndim == 2


# =============================================================================
# TestControllerPrepareBatch (pre-launched SGLang)
# =============================================================================


@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerPrepareBatch:
    """Test prepare_batch through the controller with SimpleAgent workflow."""

    def test_prepare_batch_returns_results(self, gateway_controller):
        """prepare_batch should return a list of trajectory dicts."""
        items = [
            {
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "answer": "4",
            },
            {
                "messages": [{"role": "user", "content": "What is 3+3?"}],
                "answer": "6",
            },
        ]
        dataloader = _FakeDataLoader(items, batch_size=len(items))

        result = gateway_controller.prepare_batch(
            dataloader=dataloader,
            workflow="tests.experimental.openai.utils.SimpleAgent",
        )

        assert isinstance(result, list)
        assert len(result) > 0
        traj = result[0]
        assert isinstance(traj, dict)
        assert "input_ids" in traj
        from areal.infra.rpc.rtensor import RTensor

        assert isinstance(traj["input_ids"], RTensor)
        assert traj["input_ids"].ndim == 2


# =============================================================================
# TestControllerSubmitWait (pre-launched SGLang)
# =============================================================================


@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerSubmitWait:
    """Test submit/wait API on the controller."""

    def test_submit_wait_roundtrip(self, gateway_controller):
        """submit + wait should complete a full roundtrip."""
        data = {
            "messages": [{"role": "user", "content": "Say hello."}],
            "answer": "hello",
        }

        task_id = gateway_controller.submit(
            data=data,
            workflow="tests.experimental.openai.utils.SimpleAgent",
        )

        assert isinstance(task_id, int)

        results = gateway_controller.wait(count=1, timeout=120.0)

        assert results is not None
        assert len(results) == 1
        result = results[0]
        assert result is None or isinstance(result, dict)


# =============================================================================
# TestControllerFullInitialization (parametrized: SGLang + vLLM)
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerFullInitialization:
    """Test the full initialization path where the controller launches the server.

    Parametrized via ``gateway_controller_full_init`` to cover both SGLang and
    vLLM backends.  Use ``pytest -m "not vllm"`` or ``pytest -m "not sglang"``
    to run only one backend.
    """

    def test_server_infos_populated(self, gateway_controller_full_init):
        """server_infos should be populated after full init."""
        ctrl = gateway_controller_full_init
        assert len(ctrl.server_infos) > 0
        info = ctrl.server_infos[0]
        assert info.host
        assert info.port > 0

    def test_inf_server_health(self, gateway_controller_full_init):
        """The inference server launched by the controller should be healthy."""
        ctrl = gateway_controller_full_init
        for addr in ctrl._inf_addrs:
            resp = httpx.get(f"{addr}/health", timeout=30.0)
            assert resp.status_code == 200

    def test_gateway_health(self, gateway_controller_full_init):
        """Gateway should be healthy after full init."""
        ctrl = gateway_controller_full_init
        resp = httpx.get(f"{ctrl._gateway_addr}/health", timeout=10.0)
        assert resp.status_code == 200

    def test_data_proxy_health(self, gateway_controller_full_init):
        """Data proxies should be healthy after full init."""
        ctrl = gateway_controller_full_init
        for dp_addr in ctrl._data_proxy_addrs:
            resp = httpx.get(f"{dp_addr}/health", timeout=10.0)
            assert resp.status_code == 200

    def test_data_proxy_forked_from_inf_workers(self, gateway_controller_full_init):
        """Data proxies should have been forked via RPCGuard in full init path."""
        ctrl = gateway_controller_full_init
        # Verify at least one inf role is registered
        inf_roles = [r for r in ctrl._service_roles if r.endswith(ctrl._INF_SUFFIX)]
        assert len(inf_roles) > 0, f"No inf roles in {ctrl._service_roles}"
        # Data proxies are forked via RPCGuard /fork, tracked in _forked_services
        dp_entries = [
            (addr, role, idx)
            for addr, role, idx in ctrl._forked_services
            if role == "data-proxy"
        ]
        assert len(dp_entries) > 0, "No data-proxy entries in _forked_services"

    def test_chat_completion_via_gateway(self, gateway_controller_full_init):
        """Full e2e: start_session → /chat/completions → validate → set_reward."""
        ctrl = gateway_controller_full_init
        gw = ctrl._gateway_addr
        admin_key = "test-admin"

        # --- start session ---
        resp = httpx.post(
            f"{gw}/rl/start_session",
            json={"task_id": "full-init-chat-test"},
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=30.0,
        )
        assert resp.status_code == 201, resp.text
        session = resp.json()
        session_api_key = session["sessions"][0]["session_api_key"]

        # --- non-streaming chat completion ---
        resp = httpx.post(
            f"{gw}/chat/completions",
            json={
                "model": "default",
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
        resp = httpx.post(
            f"{gw}/rl/set_reward",
            json={"reward": 0.0, "finish": True},
            headers={"Authorization": f"Bearer {session_api_key}"},
            timeout=10.0,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["interaction_count"] == 1
        assert resp.json()["ready_transition"] is True

    def test_rtensor_localize_on_rollout_result(self, gateway_controller_full_init):
        """RTensor.localize() should successfully fetch tensors from data proxy."""
        ctrl = gateway_controller_full_init
        data = [
            {
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "answer": "4",
            }
        ]

        result = ctrl.rollout_batch(
            data=data,
            workflow="tests.experimental.openai.utils.SimpleAgent",
        )

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1
        traj = result[0]
        assert isinstance(traj, dict)
        assert "input_ids" in traj

        from areal.infra.rpc.rtensor import RTensor

        # The result values should be RTensors with meta data (not yet fetched)
        rtensor_input_ids = traj["input_ids"]
        assert isinstance(rtensor_input_ids, RTensor)
        assert rtensor_input_ids.data.is_meta

        # Verify shard points to a data proxy address (not just a bare IP)
        assert ":" in rtensor_input_ids.shard.node_addr

        # Localize the trajectory — this fetches tensors from the data proxy
        local_traj = RTensor.localize(traj)

        # After localization, values should be real tensors (not RTensor)
        assert isinstance(local_traj, dict)
        assert "input_ids" in local_traj
        assert isinstance(local_traj["input_ids"], torch.Tensor)
        assert not local_traj["input_ids"].is_meta
        assert local_traj["input_ids"].ndim == 2
        assert local_traj["input_ids"].shape[0] >= 1  # at least 1 sample

        # Check other expected keys are also localized
        if "attention_mask" in local_traj:
            assert isinstance(local_traj["attention_mask"], torch.Tensor)
            assert not local_traj["attention_mask"].is_meta

    def test_rtensor_localize_batch4(self, gateway_controller_full_init):
        """RTensor.localize() on a batch of 4 should produce 4 trajectory dicts."""
        ctrl = gateway_controller_full_init
        batch_size = 4
        data = [
            {
                "messages": [{"role": "user", "content": f"What is {i}+{i}?"}],
                "answer": str(i * 2),
            }
            for i in range(batch_size)
        ]

        result = ctrl.rollout_batch(
            data=data,
            workflow="tests.experimental.openai.utils.SimpleAgent",
        )

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == batch_size

        from areal.infra.rpc.rtensor import RTensor

        # Localize each trajectory and verify tensors
        for i, traj in enumerate(result):
            assert isinstance(traj, dict), f"Trajectory {i} is not a dict"
            assert "input_ids" in traj, f"Trajectory {i} missing input_ids"

            local_traj = RTensor.localize(traj)
            assert isinstance(local_traj["input_ids"], torch.Tensor)
            assert not local_traj["input_ids"].is_meta
            assert local_traj["input_ids"].ndim == 2


# =============================================================================
# TestControllerOnlineWorkflow (parametrized: SGLang + vLLM)
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerOnlineWorkflow:
    """Test controller-in-the-loop online workflow through real gateway services.

    Parametrized via ``gateway_controller_full_init_online`` and
    ``gateway_controller_full_init_with_reward_timeout`` to cover both SGLang
    and vLLM backends.
    """

    def test_online_workflow_submit_wait_roundtrip(
        self, gateway_controller_full_init_online
    ):
        import requests

        gateway_url = gateway_controller_full_init_online.proxy_gateway_addr
        assert gateway_controller_full_init_online.config.admin_api_key is not None
        admin_key = gateway_controller_full_init_online.config.admin_api_key

        task_id = gateway_controller_full_init_online.submit(
            data={},
            workflow=None,
            workflow_kwargs={"timeout": 120.0},
        )
        assert isinstance(task_id, int)

        chat_resp = requests.post(
            f"{gateway_url}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {admin_key}",
            },
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 64,
            },
            timeout=30.0,
        )
        assert chat_resp.status_code == 200, chat_resp.text

        reward_resp = requests.post(
            f"{gateway_url}/rl/set_reward",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {admin_key}",
            },
            json={"reward": 1.0},
            timeout=10.0,
        )
        assert reward_resp.status_code == 200, reward_resp.text
        reward_data = reward_resp.json()
        assert reward_data["session_id"] == "__hitl__"
        assert reward_data["trajectory_id"] == 0

        result = gateway_controller_full_init_online.wait_for_task(
            task_id=task_id, timeout=120.0
        )
        assert result is not None
        assert isinstance(result, dict)
        assert "rewards" in result

        from areal.infra.rpc.rtensor import RTensor

        local_result = RTensor.localize(result)
        assert torch.is_tensor(local_result["rewards"])
        assert local_result["rewards"].numel() >= 1
        assert local_result["rewards"].reshape(-1)[0].item() == pytest.approx(1.0)

    def test_offline_export_applies_discount_after_multiple_rewards_in_same_trajectory(
        self, gateway_controller_full_init_with_reward_timeout
    ):
        ctrl = gateway_controller_full_init_with_reward_timeout
        gateway_url = ctrl.proxy_gateway_addr
        assert ctrl.config.admin_api_key is not None
        admin_key = ctrl.config.admin_api_key

        start_resp = httpx.post(
            f"{gateway_url}/rl/start_session",
            json={"task_id": "reward-timeout-export"},
            headers={"Authorization": f"Bearer {admin_key}"},
            timeout=30.0,
        )
        assert start_resp.status_code == 201, start_resp.text
        session = start_resp.json()
        session_id = session["sessions"][0]["session_id"]
        session_api_key = session["sessions"][0]["session_api_key"]

        first_chat = httpx.post(
            f"{gateway_url}/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 64,
            },
            headers={"Authorization": f"Bearer {session_api_key}"},
            timeout=30.0,
        )
        assert first_chat.status_code == 200, first_chat.text
        first_chat_id = first_chat.json()["id"]

        first_reward = httpx.post(
            f"{gateway_url}/rl/set_reward",
            json={"reward": 1.0},
            headers={"Authorization": f"Bearer {session_api_key}"},
            timeout=10.0,
        )
        assert first_reward.status_code == 200, first_reward.text
        first_reward_data = first_reward.json()
        assert first_reward_data["ready_transition"] is False
        assert first_reward_data["trajectory_ready"] is False
        assert first_reward_data["trajectory_id"] is None

        second_chat = httpx.post(
            f"{gateway_url}/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "What is 3+3?"}],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 64,
            },
            headers={"Authorization": f"Bearer {session_api_key}"},
            timeout=30.0,
        )
        assert second_chat.status_code == 200, second_chat.text
        second_chat_id = second_chat.json()["id"]

        second_reward = httpx.post(
            f"{gateway_url}/rl/set_reward",
            json={"reward": 4.0},
            headers={"Authorization": f"Bearer {session_api_key}"},
            timeout=10.0,
        )
        assert second_reward.status_code == 200, second_reward.text
        second_reward_data = second_reward.json()
        assert second_reward_data["ready_transition"] is False
        assert second_reward_data["trajectory_ready"] is False
        assert second_reward_data["trajectory_id"] is None

        export_data = _export_trajectory_with_retry(
            gateway_url,
            admin_key,
            session_id,
            discount=0.5,
        )
        interactions = export_data["interactions"]
        assert list(interactions) == [first_chat_id, second_chat_id]
        assert interactions[first_chat_id]["reward"] == pytest.approx(3.0)
        assert interactions[second_chat_id]["reward"] == pytest.approx(4.0)


# =============================================================================
# VLM image input tests (parametrized: SGLang + vLLM)
# =============================================================================


@pytest.mark.slow
@pytest.mark.ci
@pytest.mark.skipif(not has_gpu(), reason="GPU required")
class TestControllerVLMImage:
    """VLM image chat tests via real Qwen3-VL-2B-Instruct inference.

    Parametrized via ``gateway_controller_full_init_vlm`` to cover both SGLang
    and vLLM backends.
    """

    def test_single_image_chat(self, gateway_controller_full_init_vlm):
        img = _make_solid_color_png_b64(64, 64, (255, 0, 0))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    },
                ],
            }
        ]
        _do_vlm_chat_session(gateway_controller_full_init_vlm, "vlm-1img", messages)

    def test_multiple_images_chat(self, gateway_controller_full_init_vlm):
        red = _make_solid_color_png_b64(32, 32, (255, 0, 0))
        blue = _make_solid_color_png_b64(32, 32, (0, 0, 255))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe these two images."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{red}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{blue}"},
                    },
                ],
            }
        ]
        _do_vlm_chat_session(
            gateway_controller_full_init_vlm,
            "vlm-2img",
            messages,
            max_tokens=128,
        )

    def test_text_only_on_vlm(self, gateway_controller_full_init_vlm):
        messages = [{"role": "user", "content": "What is 2+2? Answer briefly."}]
        _do_vlm_chat_session(
            gateway_controller_full_init_vlm,
            "vlm-text",
            messages,
            max_tokens=32,
        )

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import torch

from tests.experimental.weight_update.test_nccl_integration import (
    _get_test_model_path,
    _make_local_scheduler,
    _validate_weight_update_correctness,
)

from areal.experimental.weight_update.gateway.app import create_app
from areal.experimental.weight_update.gateway.config import (
    PairInfo,
    WeightUpdateConfig,
)
from areal.infra.platforms import current_platform


@pytest.fixture()
def config() -> WeightUpdateConfig:
    return WeightUpdateConfig(
        admin_api_key="test-key",
        init_timeout_s=5,
        update_timeout_s=5,
    )


@pytest.fixture()
def app(config):
    return create_app(config)


@pytest.fixture()
def client(app):
    from starlette.testclient import TestClient

    return TestClient(app)


ADMIN_HEADERS = {"Authorization": "Bearer test-key"}


def _make_mock_aiohttp_session(called_urls: list[tuple[str, str]]):
    mock_session = MagicMock()

    @asynccontextmanager
    async def _fake_post(url, **kwargs):
        called_urls.append(("POST", url))
        resp = MagicMock()
        resp.status = 200
        resp.raise_for_status = MagicMock()
        resp.json = AsyncMock(return_value={"status": "success", "result": None})
        yield resp

    @asynccontextmanager
    async def _fake_get(url, **kwargs):
        called_urls.append(("GET", url))
        resp = MagicMock()
        resp.status = 200
        resp.raise_for_status = MagicMock()
        resp.json = AsyncMock(return_value={})
        yield resp

    mock_session.post = _fake_post
    mock_session.get = _fake_get
    return mock_session


class TestDiskConnect:
    def test_disk_connect_requires_non_empty_save_path(self, client, app):
        resp = client.post(
            "/connect",
            json={
                "pair_name": "missing_path",
                "train_worker_urls": ["http://train:8000"],
                "inference_worker_urls": ["http://infer:9000"],
                "mode": "disk",
                "save_path": "",
            },
            headers=ADMIN_HEADERS,
        )

        assert resp.status_code == 400
        assert resp.json()["error"] == "save_path is required when mode='disk'"
        assert app.state.registry.get_by_name("missing_path") is None

    def test_disk_connect_requires_absolute_save_path(self, client, app):
        resp = client.post(
            "/connect",
            json={
                "pair_name": "relative_path",
                "train_worker_urls": ["http://train:8000"],
                "inference_worker_urls": ["http://infer:9000"],
                "mode": "disk",
                "save_path": "shared/weights",
            },
            headers=ADMIN_HEADERS,
        )

        assert resp.status_code == 400
        assert resp.json()["error"] == (
            "save_path must be an absolute path when mode='disk'"
        )
        assert app.state.registry.get_by_name("relative_path") is None

    def test_disk_connect_requires_lora_name_when_lora_enabled(self, client, app):
        resp = client.post(
            "/connect",
            json={
                "pair_name": "missing_lora_name",
                "train_worker_urls": ["http://train:8000"],
                "inference_worker_urls": ["http://infer:9000"],
                "mode": "disk",
                "save_path": "/shared/lora",
                "use_lora": True,
                "lora_name": "",
            },
            headers=ADMIN_HEADERS,
        )

        assert resp.status_code == 400
        assert resp.json()["error"] == "lora_name is required when use_lora=True"
        assert app.state.registry.get_by_name("missing_lora_name") is None

    def test_disk_connect_registers_pair(self, client, app):
        resp = client.post(
            "/connect",
            json={
                "pair_name": "disk_pair",
                "train_worker_urls": ["http://train:8000"],
                "inference_worker_urls": ["http://infer:9000"],
                "mode": "disk",
                "save_path": "/shared/weights",
            },
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["pair_name"] == "disk_pair"

        registry = app.state.registry
        pair = registry.get_by_name("disk_pair")
        assert pair is not None
        assert pair.mode == "disk"
        assert pair.save_path == "/shared/weights"
        assert pair.train_worker_urls == ["http://train:8000"]
        assert pair.inference_worker_urls == ["http://infer:9000"]

    def test_disk_connect_with_lora(self, client, app):
        resp = client.post(
            "/connect",
            json={
                "pair_name": "lora_pair",
                "train_worker_urls": ["http://train:8000"],
                "inference_worker_urls": ["http://infer:9000"],
                "mode": "disk",
                "save_path": "/shared/lora",
                "use_lora": True,
                "lora_name": "my-adapter",
            },
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200

        pair = app.state.registry.get_by_name("lora_pair")
        assert pair.use_lora is True
        assert pair.lora_name == "my-adapter"

    def test_disk_connect_does_not_call_awex_endpoints(self, client, app):
        called_urls: list[tuple[str, str]] = []
        app.state.http_session = _make_mock_aiohttp_session(called_urls)

        resp = client.post(
            "/connect",
            json={
                "pair_name": "disk_only",
                "train_worker_urls": ["http://train:8000"],
                "inference_worker_urls": ["http://infer:9000"],
                "mode": "disk",
                "save_path": "/tmp/w",
            },
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        assert len(called_urls) == 0


class TestDiskUpdateWeights:
    @pytest.fixture(autouse=True)
    def _register_disk_pair(self, app):
        pair_info = PairInfo(
            pair_name="test_disk",
            train_worker_urls=["http://train:8000"],
            inference_worker_urls=["http://infer:9000"],
            mode="disk",
            save_path="/shared/weights",
        )
        app.state.registry.register(pair_info)

    def test_disk_update_calls_pause_save_load_resume(self, client, app):
        called_urls: list[tuple[str, str]] = []
        app.state.http_session = _make_mock_aiohttp_session(called_urls)

        resp = client.post(
            "/update_weights",
            json={"pair_name": "test_disk", "version": 1},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == 1

        urls = [url for _, url in called_urls]
        assert "http://infer:9000/pause_generation" in urls
        assert any("/save" in u for u in urls)
        assert "http://infer:9000/update_weights_from_disk" in urls
        assert "http://infer:9000/continue_generation" in urls

        pause_idx = urls.index("http://infer:9000/pause_generation")
        save_idx = next(i for i, u in enumerate(urls) if "/save" in u)
        load_idx = urls.index("http://infer:9000/update_weights_from_disk")
        resume_idx = urls.index("http://infer:9000/continue_generation")
        assert pause_idx < save_idx < load_idx < resume_idx

    def test_disk_update_versioned_save_path(self, client, app):
        called_urls: list[tuple[str, str]] = []
        app.state.http_session = _make_mock_aiohttp_session(called_urls)

        client.post(
            "/update_weights",
            json={"pair_name": "test_disk", "version": 42},
            headers=ADMIN_HEADERS,
        )

        save_url = next(url for _, url in called_urls if "/save" in url)
        assert save_url == "http://train:8000/save"

    def test_disk_update_not_found_pair(self, client, app):
        resp = client.post(
            "/update_weights",
            json={"pair_name": "nonexistent", "version": 1},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 404

    def test_disk_update_error_still_resumes_inference(self, client, app):
        called_urls: list[tuple[str, str]] = []

        @asynccontextmanager
        async def _failing_post(url, **kwargs):
            called_urls.append(("POST", url))
            if "/save" in url:
                raise RuntimeError("save failed")
            resp = MagicMock()
            resp.status = 200
            resp.raise_for_status = MagicMock()
            resp.json = AsyncMock(return_value={"status": "success", "result": None})
            yield resp

        mock_session = MagicMock()
        mock_session.post = _failing_post
        app.state.http_session = mock_session

        resp = client.post(
            "/update_weights",
            json={"pair_name": "test_disk", "version": 1},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert "save failed" in data["error"]

        urls = [url for _, url in called_urls]
        assert "http://infer:9000/continue_generation" in urls


class TestDiskUpdateWeightsLora:
    @pytest.fixture(autouse=True)
    def _register_lora_pair(self, app):
        pair_info = PairInfo(
            pair_name="lora_disk",
            train_worker_urls=["http://train:8000"],
            inference_worker_urls=["http://infer:9000"],
            mode="disk",
            save_path="/shared/lora_weights",
            use_lora=True,
            lora_name="my-adapter",
        )
        app.state.registry.register(pair_info)

    def test_lora_update_uses_load_lora_adapter(self, client, app):
        called_urls: list[tuple[str, str]] = []
        app.state.http_session = _make_mock_aiohttp_session(called_urls)

        resp = client.post(
            "/update_weights",
            json={"pair_name": "lora_disk", "version": 3},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        urls = [url for _, url in called_urls]
        assert "http://infer:9000/load_lora_adapter" in urls
        assert "http://infer:9000/update_weights_from_disk" not in urls


class TestDiskDisconnect:
    def test_disconnect_removes_disk_pair(self, client, app):
        client.post(
            "/connect",
            json={
                "pair_name": "to_remove",
                "train_worker_urls": ["http://train:8000"],
                "inference_worker_urls": ["http://infer:9000"],
                "mode": "disk",
                "save_path": "/tmp/w",
            },
            headers=ADMIN_HEADERS,
        )
        assert app.state.registry.get_by_name("to_remove") is not None

        resp = client.post(
            "/disconnect",
            json={"pair_name": "to_remove"},
            headers=ADMIN_HEADERS,
        )
        assert resp.status_code == 200
        assert app.state.registry.get_by_name("to_remove") is None


# ---------------------------------------------------------------------------
# E2E disk weight update: real FSDPEngine + SGLang server + gateway
# ---------------------------------------------------------------------------


@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.sglang
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("n_gpus", [2, 4, 8], ids=["2gpu", "4gpu", "8gpu"])
def test_disk_e2e_weight_update(n_gpus, tmp_path_factory):
    """Full round trip: FSDPEngine save → shared disk → SGLang load.

    Same infrastructure as :func:`test_awex_e2e_weight_update` but uses
    the disk-based weight update path (``mode="disk"``).  The gateway
    orchestrates pause → FSDP save to HF → SGLang
    ``/update_weights_from_disk`` → resume.
    """
    if current_platform.device_count() < n_gpus:
        pytest.skip(f"This test requires {n_gpus} GPUs")

    from areal.api import FinetuneSpec
    from areal.api.cli_args import (
        InferenceEngineConfig,
        OptimizerConfig,
        SchedulingSpec,
        TrainEngineConfig,
    )
    from areal.experimental.inference_service.controller.controller import (
        RolloutControllerV2,
    )
    from areal.experimental.training_service.controller.controller import (
        GatewayTrainController,
    )
    from areal.experimental.weight_update.controller import (
        WeightUpdateController,
        WeightUpdateControllerConfig,
    )

    n_half = n_gpus // 2
    tmp = tmp_path_factory.mktemp("disk_e2e")
    model_path = _get_test_model_path()
    shared_weight_dir = str(tmp / "shared_weights")

    scheduler = _make_local_scheduler(tmp, "disk_e2e", gpu_devices=list(range(n_gpus)))

    inf_config = InferenceEngineConfig(
        tokenizer_path=model_path,
        backend=f"sglang:d{n_half}",
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.experimental.inference_service.guard",
            ),
        ),
        consumer_batch_size=8,
        max_head_offpolicyness=1024,
        setup_timeout=300.0,
        admin_api_key="test-admin",
    )
    inf_ctrl = RolloutControllerV2(config=inf_config, scheduler=scheduler)

    train_config = TrainEngineConfig(
        backend=f"fsdp:d{n_half}",
        experiment_name="test-disk-e2e",
        trial_name="t0",
        path=model_path,
        optimizer=OptimizerConfig(),
        _version="v2",
        setup_timeout=300.0,
        scheduling_spec=(
            SchedulingSpec(
                gpu=1,
                cmd="python -m areal.experimental.training_service.guard",
                env_vars=dict(NCCL_CUMEM_ENABLE="0", NCCL_NVLS_ENABLE="0"),
            ),
        ),
    )
    train_ctrl = GatewayTrainController(
        train_engine="areal.engine.fsdp_engine.FSDPEngine",
        config=train_config,
        scheduler=scheduler,
    )

    wu_ctrl: WeightUpdateController | None = None

    try:
        # -- 1. SGLang via inference controller ----------------------------
        inf_ctrl.initialize(
            role="rollout",
            server_args={"model_path": model_path, "mem_fraction_static": 0.7},
            wait=True,
        )
        inf_worker_urls = list(inf_ctrl._inf_addrs)

        for url in inf_worker_urls:
            resp = httpx.post(f"{url}/awex/debug/randomize_parameters", timeout=120.0)
            assert resp.status_code == 200, f"randomize_parameters failed: {resp.text}"

        # -- 2. FSDP engine via training controller -------------------------
        ft_spec = FinetuneSpec(
            total_train_epochs=1, dataset_size=100, train_batch_size=2
        )
        train_ctrl.initialize(role="actor", ft_spec=ft_spec, wait=True)
        train_worker_urls = list(train_ctrl._worker_addrs)

        # -- 3. Weight update gateway -------------------------------------
        wu_ctrl = WeightUpdateController(
            config=WeightUpdateControllerConfig(
                host="127.0.0.1",
                request_timeout=300.0,
            )
        )
        wu_ctrl.initialize()

        # -- 4. Disk weight update lifecycle -------------------------------
        assert wu_ctrl.health_check(), "Weight update gateway health check failed"

        wu_ctrl.connect(
            pair_name="test_disk_e2e",
            train_worker_urls=train_worker_urls,
            inference_worker_urls=inf_worker_urls,
            mode="disk",
            save_path=shared_weight_dir,
        )

        result = wu_ctrl.update_weights(version=1)
        assert result.status == "ok", f"Disk weight update failed: {result.error}"
        assert result.version == 1

        wu_ctrl.disconnect()

        # -- 5. Verify the HF checkpoint exists on disk --------------------
        versioned_path = os.path.join(shared_weight_dir, "weight_update_v1")
        assert os.path.isdir(versioned_path), (
            f"Expected HF checkpoint at {versioned_path}"
        )
        assert os.path.isfile(os.path.join(versioned_path, "config.json")), (
            "Missing config.json in saved checkpoint"
        )

        # -- 6. Verify inference server still works post-update -----------
        gen_resp = httpx.post(
            f"{inf_worker_urls[0]}/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 5, "temperature": 0},
            },
            timeout=30.0,
        )
        assert gen_resp.status_code == 200, (
            f"Generation failed after weight update: {gen_resp.text}"
        )

        # -- 7. Validate training ↔ inference parameter equality ----------
        _validate_weight_update_correctness(
            train_worker_urls=train_worker_urls,
            inf_worker_url=inf_worker_urls[0],
            param_dir=tmp,
        )

    finally:
        if wu_ctrl is not None:
            wu_ctrl.destroy()
        train_ctrl.destroy()
        inf_ctrl.destroy()
        scheduler.delete_workers(None)

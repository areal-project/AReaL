# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
import requests

from areal.v2.weight_update.controller.config import (
    WeightUpdateControllerConfig,
)
from areal.v2.weight_update.controller.controller import (
    WeightUpdateController,
)
from areal.v2.weight_update.gateway.config import WeightUpdateResult

GATEWAY_URL = "http://localhost:7080"


@pytest.fixture()
def ctrl() -> WeightUpdateController:
    c = WeightUpdateController(
        config=WeightUpdateControllerConfig(
            admin_api_key="test-admin-key",
            request_timeout=10.0,
        )
    )
    c._gateway_url = GATEWAY_URL
    c._session = MagicMock(spec=requests.Session)
    return c


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            response=resp,
        )
    return resp


class TestHealthCheck:
    def test_health_check_success_returns_true(self, ctrl):
        ctrl._session.get.return_value = _mock_response(200, {"status": "healthy"})

        assert ctrl.health_check() is True
        ctrl._session.get.assert_called_once_with(f"{GATEWAY_URL}/health", timeout=10.0)

    def test_health_check_connect_error_returns_false(self, ctrl):
        ctrl._session.get.side_effect = httpx.ConnectError("refused")

        assert ctrl.health_check() is False


class TestConnect:
    @pytest.mark.parametrize(
        "train_urls,mode", [([], "awex"), (["http://train"], "unknown")]
    )
    def test_validation_rejection_does_not_leave_uncertain_pending_pair(
        self, ctrl, train_urls, mode
    ):
        with pytest.raises(ValueError):
            ctrl.connect("pair", train_urls, ["http://infer"], mode=mode)
        ctrl._session.post.assert_not_called()
        assert ctrl._pending_pair_name is None
        ctrl.destroy()

    def test_connect_failure_does_not_claim_connection(self, ctrl):
        ctrl._session.post.return_value = _mock_response(503)

        with pytest.raises(requests.HTTPError):
            ctrl.connect("pair0", ["http://t:8000"], ["http://i:8000"])

        assert ctrl._pair_name is None

    def test_connect_stores_pair_name(self, ctrl):
        ctrl._session.post.return_value = _mock_response(200, {"pair_name": "pair0"})

        ctrl.connect("pair0", ["http://t:8000"], ["http://i:8000"])

        assert ctrl._pair_name == "pair0"

    def test_connect_sends_correct_request(self, ctrl):
        ctrl._session.post.return_value = _mock_response(200, {"pair_name": "pair0"})
        train_urls = ["http://train1:8000", "http://train2:8000"]
        infer_urls = ["http://infer1:8000"]

        ctrl.connect("pair0", train_urls, infer_urls)

        ctrl._session.post.assert_called_once_with(
            f"{GATEWAY_URL}/connect",
            json={
                "pair_name": "pair0",
                "train_worker_urls": train_urls,
                "inference_worker_urls": infer_urls,
                "mode": "awex",
                "save_path": "",
                "use_lora": False,
                "lora_name": "",
                "lora_keep_versions": 0,
                "colocate": False,
                "nccl_master_addr": "",
                "nccl_master_port": 0,
            },
            timeout=10.0,
        )

    def test_connect_disk_mode_sends_disk_fields(self, ctrl):
        ctrl._session.post.return_value = _mock_response(200, {"pair_name": "pair0"})
        train_urls = ["http://train1:8000"]
        infer_urls = ["http://infer1:8000"]

        ctrl.connect(
            "pair0",
            train_urls,
            infer_urls,
            mode="disk",
            save_path="/shared/weights",
            use_lora=True,
            lora_name="my-lora",
        )

        ctrl._session.post.assert_called_once_with(
            f"{GATEWAY_URL}/connect",
            json={
                "pair_name": "pair0",
                "train_worker_urls": train_urls,
                "inference_worker_urls": infer_urls,
                "mode": "disk",
                "save_path": "/shared/weights",
                "use_lora": True,
                "lora_name": "my-lora",
                "lora_keep_versions": 0,
                "colocate": False,
                "nccl_master_addr": "",
                "nccl_master_port": 0,
            },
            timeout=10.0,
        )


class TestUpdateWeights:
    def test_update_weights_returns_result(self, ctrl):
        ctrl._pair_name = "pair0"
        ctrl._session.post.return_value = _mock_response(
            200,
            {"status": "ok", "version": 5, "duration_ms": 123.4, "error": None},
        )

        result = ctrl.update_weights(version=5)

        assert isinstance(result, WeightUpdateResult)
        assert result.status == "ok"
        assert result.version == 5
        assert result.duration_ms == 123.4
        assert result.error is None
        ctrl._session.post.assert_called_once_with(
            f"{GATEWAY_URL}/update_weights",
            json={"pair_name": "pair0", "version": 5},
            timeout=10.0,
        )

    def test_update_weights_raises_when_not_connected(self, ctrl):
        with pytest.raises(RuntimeError, match="Not connected"):
            ctrl.update_weights(version=1)


class TestDisconnect:
    def test_disconnect_failure_retains_connection_for_cleanup_retry(self, ctrl):
        ctrl._pair_name = "pair0"
        ctrl._session.post.return_value = _mock_response(503)

        with pytest.raises(requests.HTTPError):
            ctrl.disconnect()

        assert ctrl._pair_name == "pair0"

    def test_disconnect_clears_state(self, ctrl):
        ctrl._pair_name = "pair0"
        ctrl._session.post.return_value = _mock_response(
            200, {"status": "ok", "pair_name": "pair0"}
        )

        ctrl.disconnect()

        assert ctrl._pair_name is None
        ctrl._session.post.assert_called_once_with(
            f"{GATEWAY_URL}/disconnect",
            json={"pair_name": "pair0"},
            timeout=10.0,
        )

    def test_disconnect_noop_when_not_connected(self, ctrl):
        ctrl.disconnect()
        assert ctrl._pair_name is None


class TestLifecycle:
    def test_full_lifecycle(self, ctrl):
        connect_resp = _mock_response(200, {"pair_name": "pair0"})
        update_resp = _mock_response(
            200, {"status": "ok", "version": 1, "duration_ms": 50.0, "error": None}
        )
        disconnect_resp = _mock_response(200, {"status": "ok", "pair_name": "pair0"})
        ctrl._session.post.side_effect = [connect_resp, update_resp, disconnect_resp]

        ctrl.connect("pair0", ["http://t:8000"], ["http://i:8000"])
        assert ctrl._pair_name == "pair0"

        result = ctrl.update_weights(version=1)
        assert result.status == "ok"
        assert result.version == 1

        ctrl.disconnect()
        assert ctrl._pair_name is None

    def test_gateway_error_raises_http_error(self, ctrl):
        ctrl._pair_name = "pair0"
        ctrl._session.post.return_value = _mock_response(500, {"error": "internal"})

        with pytest.raises(requests.HTTPError):
            ctrl.update_weights(version=1)


class TestGatewayDisconnect:
    @pytest.mark.parametrize("failed", [False, True])
    def test_stopped_training_only_tears_down_inference_and_fences_pair(
        self, monkeypatch, failed
    ):
        from unittest.mock import AsyncMock

        from starlette.testclient import TestClient

        from areal.v2.weight_update.gateway import app as gateway
        from areal.v2.weight_update.gateway.config import PairInfo, WeightUpdateConfig

        app = gateway.create_app(WeightUpdateConfig(admin_api_key="test-key"))
        app.state.http_session = object()
        pair = PairInfo("dead-generation", ["http://dead-train"], ["http://infer"])
        app.state.registry.register(pair)
        teardown = AsyncMock(
            side_effect=RuntimeError("inference unknown") if failed else None
        )
        monkeypatch.setattr(gateway, "_post", teardown)
        client = TestClient(app)
        response = client.post(
            "/disconnect",
            json={"pair_name": pair.pair_name, "training_stopped": True},
            headers={"Authorization": "Bearer test-key"},
        )
        assert teardown.await_count == 1
        assert teardown.await_args.args[1] == "http://infer/awex/teardown"
        if failed:
            assert response.status_code == 503
            assert app.state.registry.get_by_name(pair.pair_name) is pair
            teardown.side_effect = None
            response = client.post(
                "/disconnect",
                json={"pair_name": pair.pair_name, "training_stopped": True},
                headers={"Authorization": "Bearer test-key"},
            )
        assert response.status_code == 200
        assert pair.pair_name in app.state.retired_pairs
        assert app.state.registry.get_by_name(pair.pair_name) is None

    @pytest.mark.parametrize("failed", [False, True])
    def test_teardown_reaches_both_sides_and_retains_uncertain_pair(
        self, monkeypatch, failed
    ):
        from unittest.mock import AsyncMock

        from starlette.testclient import TestClient

        from areal.v2.weight_update.gateway import app as gateway
        from areal.v2.weight_update.gateway.config import PairInfo, WeightUpdateConfig

        app = gateway.create_app(WeightUpdateConfig(admin_api_key="test-key"))
        app.state.http_session = object()
        pair = PairInfo("pair", ["http://train"], ["http://infer"])
        app.state.registry.register(pair)
        app.state.kv_store.put("pair", "metadata", [1])
        teardown = AsyncMock(
            side_effect=[RuntimeError("peer unavailable") if failed else None, None]
        )
        monkeypatch.setattr(gateway, "_post", teardown)

        response = TestClient(app).post(
            "/disconnect",
            json={"pair_name": "pair"},
            headers={"Authorization": "Bearer test-key"},
        )

        assert teardown.await_count == 2
        assert {call.args[1] for call in teardown.await_args_list} == {
            "http://train/awex/teardown",
            "http://infer/awex/teardown",
        }
        if failed:
            assert response.status_code == 503
            assert app.state.registry.get_by_name("pair") is pair
            assert app.state.kv_store.get("pair", "metadata") == [1]
            teardown.side_effect = None
            response = TestClient(app).post(
                "/disconnect",
                json={"pair_name": "pair"},
                headers={"Authorization": "Bearer test-key"},
            )
        assert response.status_code == 200
        assert app.state.registry.get_by_name("pair") is None
        assert app.state.kv_store.get("pair", "metadata") is None

    @pytest.mark.asyncio
    async def test_collective_post_timeout_is_not_retried(self):
        from contextlib import asynccontextmanager

        from areal.v2.weight_update.gateway.app import _post

        calls = []

        @asynccontextmanager
        async def post(url, **kwargs):
            calls.append(url)
            raise TimeoutError("collective outcome unknown")
            yield  # pragma: no cover

        session = MagicMock()
        session.post = post
        with pytest.raises(TimeoutError):
            await _post(session, "http://worker/awex/init_weights_update_group", 1)
        assert calls == ["http://worker/awex/init_weights_update_group"]

    def test_sglang_teardown_ack_waits_for_collective(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient

        from areal.v2.inference_service.sglang.awex import register_awex_endpoints

        app, proxy = FastAPI(), MagicMock()
        register_awex_endpoints(app, proxy)
        client = TestClient(app)
        assert client.post("/awex/teardown").status_code == 200
        proxy.collective_rpc.assert_called_once_with("awex_teardown")
        proxy.collective_rpc.side_effect = RuntimeError("still connected")
        assert client.post("/awex/teardown").status_code == 500


class TestGatewayTransactions:
    @pytest.mark.parametrize("colocate", [False, True])
    def test_dtype_mismatch_rejects_connect_before_group_init(
        self, monkeypatch, colocate
    ):
        from unittest.mock import AsyncMock

        from starlette.testclient import TestClient

        from areal.v2.weight_update.gateway import app as gateway

        app, calls = self.app(), []
        monkeypatch.setattr(
            gateway,
            "_discover_parallelism",
            AsyncMock(return_value=({"world_size": 1}, {"world_size": 1})),
        )

        async def metadata(session, url, timeout_s, json_data=None):
            calls.append(url)
            dtype = (
                "torch.float32" if url.startswith("http://train/") else "torch.bfloat16"
            )
            return {
                "result": [
                    {
                        "data": {
                            "name": "layer.weight",
                            "dtype": {"type": "torch_dtype", "value": dtype},
                        }
                    }
                ]
            }

        cleanup = AsyncMock()
        monkeypatch.setattr(gateway, "_post_json", metadata)
        monkeypatch.setattr(gateway, "_post", cleanup)
        response = TestClient(app).post(
            "/connect",
            json={**self.body(), "colocate": colocate},
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 503
        assert "wire dtype mismatch" in response.json()["error"]
        assert all(url.endswith("/report_weight_meta") for url in calls)
        assert {call.args[1] for call in cleanup.call_args_list} == {
            "http://train/awex/teardown",
            "http://infer/awex/teardown",
        }

    @staticmethod
    def app():
        from areal.v2.weight_update.gateway.app import create_app
        from areal.v2.weight_update.gateway.config import WeightUpdateConfig

        app = create_app(WeightUpdateConfig(admin_api_key="test-key"))
        app.state.http_session = object()
        return app

    @staticmethod
    def body():
        return {
            "pair_name": "incarnation",
            "train_worker_urls": ["http://train"],
            "inference_worker_urls": ["http://infer"],
            "nccl_master_addr": "127.0.0.1",
            "nccl_master_port": 30000,
        }

    def test_validation_rejection_can_be_cleaned_up_after_lost_response(self):
        from starlette.testclient import TestClient

        app = self.app()
        client, headers = TestClient(app), {"Authorization": "Bearer test-key"}
        body = {**self.body(), "mode": "disk", "save_path": ""}
        assert client.post("/connect", json=body, headers=headers).status_code == 400
        assert app.state.registry.get_by_name("incarnation") is None
        assert (
            client.post(
                "/disconnect", json={"pair_name": "incarnation"}, headers=headers
            ).status_code
            == 200
        )

    @pytest.mark.asyncio
    async def test_metadata_timeout_does_not_retry_one_rank_collective(self):
        from areal.v2.weight_update.gateway.app import _post_json

        session = MagicMock()
        session.post.side_effect = TimeoutError("metadata response lost")
        with pytest.raises(TimeoutError):
            await _post_json(session, "http://rank/awex/report_weight_meta", 1)
        session.post.assert_called_once()

    @pytest.mark.parametrize("colocate", [False, True])
    def test_metadata_failure_waits_for_peers_and_cleans_both_sides(
        self, monkeypatch, colocate
    ):
        import asyncio
        from unittest.mock import AsyncMock

        from starlette.testclient import TestClient

        from areal.v2.weight_update.gateway import app as gateway

        app, seen = self.app(), []
        monkeypatch.setattr(
            gateway,
            "_discover_parallelism",
            AsyncMock(return_value=({"world_size": 1}, {"world_size": 1})),
        )

        async def metadata(session, url, timeout_s, json_data=None):
            seen.append(url)
            if url.startswith("http://train/"):
                raise TimeoutError("metadata response lost")
            await asyncio.sleep(0)
            seen.append("inference finished")
            return {"result": []}

        async def cleanup(session, url, timeout_s, json_data=None):
            assert "inference finished" in seen
            seen.append(url)

        monkeypatch.setattr(gateway, "_post_json", metadata)
        monkeypatch.setattr(gateway, "_post", cleanup)
        response = TestClient(app).post(
            "/connect",
            json={**self.body(), "colocate": colocate},
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 503
        assert seen.count("http://train/awex/report_weight_meta") == 1
        assert seen.count("http://infer/awex/report_weight_meta") == 1
        assert set(seen[-2:]) == {
            "http://train/awex/teardown",
            "http://infer/awex/teardown",
        }

    def test_partial_init_waits_for_other_peer_then_preserves_failed_pair(
        self, monkeypatch
    ):
        import asyncio
        from unittest.mock import AsyncMock

        from starlette.testclient import TestClient

        from areal.v2.weight_update.gateway import app as gateway

        app, events = self.app(), []
        monkeypatch.setattr(
            gateway,
            "_discover_parallelism",
            AsyncMock(return_value=({"world_size": 1}, {"world_size": 1})),
        )
        monkeypatch.setattr(
            gateway, "_post_json", AsyncMock(return_value={"result": []})
        )
        # These tests isolate connection transactions; metadata planning has
        # real AWEX coverage in test_metadata.py.
        monkeypatch.setattr(gateway, "merge_parameter_metadata", lambda _: [])
        monkeypatch.setattr(gateway, "merge_inference_metadata", lambda _: [])

        async def post(session, url, timeout_s, json_data=None):
            if url.endswith("init_weights_update_group"):
                assert (
                    app.state.registry.get_by_name("incarnation").state == "connecting"
                )
                if url.startswith("http://infer/"):
                    events.append("init_failed")
                    raise RuntimeError("init failed")
                await asyncio.sleep(0)
                events.append("train_finished")
            else:
                assert "train_finished" in events
                events.append(url)

        monkeypatch.setattr(gateway, "_post", post)
        client = TestClient(app)
        response = client.post(
            "/connect", json=self.body(), headers={"Authorization": "Bearer test-key"}
        )
        assert response.status_code == 503
        assert app.state.registry.get_by_name("incarnation").state == "failed"
        assert events[:2] == ["init_failed", "train_finished"]
        assert {event for event in events[2:]} == {
            "http://train/awex/teardown",
            "http://infer/awex/teardown",
        }
        assert (
            client.get(
                "/pairs/incarnation", headers={"Authorization": "Bearer test-key"}
            ).json()["state"]
            == "failed"
        )
        assert (
            client.post(
                "/connect",
                json=self.body(),
                headers={"Authorization": "Bearer test-key"},
            ).status_code
            == 409
        )

    def test_successful_connect_and_disconnect_are_idempotent_and_fenced(
        self, monkeypatch
    ):
        from unittest.mock import AsyncMock

        from starlette.testclient import TestClient

        from areal.v2.weight_update.gateway import app as gateway

        app = self.app()
        monkeypatch.setattr(
            gateway,
            "_discover_parallelism",
            AsyncMock(return_value=({"world_size": 1}, {"world_size": 1})),
        )
        monkeypatch.setattr(
            gateway, "_post_json", AsyncMock(return_value={"result": []})
        )
        # These tests isolate connection transactions; metadata planning has
        # real AWEX coverage in test_metadata.py.
        monkeypatch.setattr(gateway, "merge_parameter_metadata", lambda _: [])
        monkeypatch.setattr(gateway, "merge_inference_metadata", lambda _: [])
        post = AsyncMock()
        monkeypatch.setattr(gateway, "_post", post)
        client, headers = TestClient(app), {"Authorization": "Bearer test-key"}
        for _ in range(2):
            assert (
                client.post("/connect", json=self.body(), headers=headers).status_code
                == 200
            )
        assert post.await_count == 4  # two init requests and two liveness probes
        for call in post.await_args_list[:2]:
            assert call.kwargs["json_data"]["fenced"] is True
        for _ in range(2):
            assert (
                client.post(
                    "/update_weights",
                    json={"pair_name": "incarnation", "version": 1},
                    headers=headers,
                ).json()["status"]
                == "ok"
            )
        assert post.await_count == 6
        for _ in range(2):
            assert (
                client.post(
                    "/disconnect", json={"pair_name": "incarnation"}, headers=headers
                ).status_code
                == 200
            )
        assert post.await_count == 8
        assert (
            client.post("/connect", json=self.body(), headers=headers).status_code
            == 409
        )
        assert post.await_count == 8

    @pytest.mark.asyncio
    async def test_disconnect_waits_for_running_update(self, monkeypatch):
        import asyncio

        from areal.v2.weight_update.gateway import app as gateway
        from areal.v2.weight_update.gateway.config import PairInfo

        app, events = self.app(), []
        app.state.registry.register(
            PairInfo("pair", ["http://train"], ["http://infer"])
        )
        started, release = asyncio.Event(), asyncio.Event()

        async def post(session, url, timeout_s, json_data=None):
            if url.endswith("update_weights"):
                started.set()
                await release.wait()
                events.append("updated")
            else:
                events.append("teardown")

        monkeypatch.setattr(gateway, "_post", post)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway",
            headers={"Authorization": "Bearer test-key"},
        ) as client:
            update = asyncio.create_task(
                client.post("/update_weights", json={"pair_name": "pair", "version": 1})
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            disconnect = asyncio.create_task(
                client.post("/disconnect", json={"pair_name": "pair"})
            )
            await asyncio.sleep(0)
            assert events == []
            release.set()
            assert (await update).status_code == (await disconnect).status_code == 200
        assert events == ["updated", "updated", "teardown", "teardown"]

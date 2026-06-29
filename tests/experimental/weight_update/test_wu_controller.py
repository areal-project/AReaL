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

from __future__ import annotations

# pyright: reportMissingImports=false
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio

from areal.infra.data_service.gateway.app import create_gateway_app
from areal.infra.data_service.gateway.config import GatewayConfig

ADMIN_KEY = "test-admin-key"
WORKER_ADDR = "http://worker-1:8000"
WORKER_ADDR_2 = "http://worker-2:8000"
MODULE = "areal.infra.data_service.gateway.app"


@pytest.fixture
def config():
    return GatewayConfig(
        host="127.0.0.1",
        port=18090,
        admin_api_key=ADMIN_KEY,
        router_addr="http://mock-router:8091",
    )


@pytest_asyncio.fixture
async def client(config):
    app = create_gateway_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def admin_headers():
    return {"Authorization": f"Bearer {ADMIN_KEY}"}


async def _register_dataset(client, dataset_id: str = "train-sample") -> dict:
    with (
        patch(
            f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
        ) as mock_workers,
        patch(
            f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
        ) as mock_broadcast,
    ):
        mock_workers.return_value = [WORKER_ADDR]
        mock_broadcast.return_value = [
            {
                "addr": WORKER_ADDR,
                "status": 200,
                "data": {"steps_per_epoch": 10, "dataset_size": 100},
            }
        ]
        resp = await client.post(
            "/v1/datasets/register",
            json={"dataset_id": dataset_id, "dataset_path": "/tmp/sample.jsonl"},
            headers=admin_headers(),
        )
    assert resp.status_code == 200
    return resp.json()


class TestGatewayHealth:
    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_returns_router_addr(self, client, config):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["router_addr"] == config.router_addr


class TestGatewayAuth:
    @pytest.mark.asyncio
    async def test_fetch_samples_no_auth_401(self, client):
        resp = await client.post("/v1/samples/fetch", json={"indices": [0]})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_fetch_samples_bad_key_401(self, client):
        resp = await client.post(
            "/v1/samples/fetch",
            json={"indices": [0]},
            headers={"Authorization": "Bearer unknown-key"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_admin_endpoint_with_dataset_key_403(self, client):
        resp = await client.post(
            "/v1/datasets/register",
            json={"dataset_id": "d1", "dataset_path": "/tmp/data.jsonl"},
            headers={"Authorization": "Bearer ds-not-admin"},
        )
        assert resp.status_code == 403


class TestDatasetRegistration:
    @pytest.mark.asyncio
    async def test_register_dataset_returns_api_key(self, client):
        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.return_value = [
                {
                    "addr": WORKER_ADDR,
                    "status": 200,
                    "data": {"steps_per_epoch": 12, "dataset_size": 120},
                },
                {
                    "addr": WORKER_ADDR_2,
                    "status": 200,
                    "data": {"steps_per_epoch": 12, "dataset_size": 120},
                },
            ]

            resp = await client.post(
                "/v1/datasets/register",
                json={
                    "dataset_id": "dataset-a",
                    "dataset_path": "/tmp/a.jsonl",
                },
                headers=admin_headers(),
            )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["api_key"].startswith("ds-")
        assert payload["dataset_id"] == "dataset-a"
        assert payload["dataset_size"] == 240

    @pytest.mark.asyncio
    async def test_register_then_fetch_uses_dataset_key(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-b")
        dataset_key = reg_payload["api_key"]

        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"samples": [{"text": "hello"}]})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.post = lambda *args, **kwargs: (
            setattr(mock_session, "_post_args", (args, kwargs)) or mock_resp
        )
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(f"{MODULE}._query_router", new_callable=AsyncMock) as mock_route,
            patch(f"{MODULE}.aiohttp.ClientSession", return_value=mock_session),
        ):
            mock_route.return_value = WORKER_ADDR
            resp = await client.post(
                "/v1/samples/fetch",
                json={"indices": [0]},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        post_args, post_kwargs = mock_session._post_args
        assert post_args[0] == f"{WORKER_ADDR}/v1/samples/fetch"
        assert post_kwargs["json"]["dataset_id"] == "dataset-b"

    @pytest.mark.asyncio
    async def test_unregister_revokes_key(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-c")
        dataset_key = reg_payload["api_key"]

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR]
            mock_broadcast.return_value = [
                {"addr": WORKER_ADDR, "status": 200, "data": {}}
            ]
            resp = await client.post(
                "/v1/datasets/unregister",
                json={"dataset_id": "dataset-c"},
                headers=admin_headers(),
            )
        assert resp.status_code == 200

        after_revoke = await client.post(
            "/v1/samples/fetch",
            json={"indices": [0]},
            headers={"Authorization": f"Bearer {dataset_key}"},
        )
        assert after_revoke.status_code == 401

    @pytest.mark.asyncio
    async def test_register_failure_rolls_back_successful_workers(self, client):
        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.side_effect = [
                [
                    {"addr": WORKER_ADDR, "status": 200, "data": {}},
                    {"addr": WORKER_ADDR_2, "status": 500, "error": "boom"},
                ],
                [
                    {"addr": WORKER_ADDR, "status": 200, "data": {}},
                ],
            ]

            resp = await client.post(
                "/v1/datasets/register",
                json={
                    "dataset_id": "dataset-rollback",
                    "dataset_path": "/tmp/a.jsonl",
                },
                headers=admin_headers(),
            )

        assert resp.status_code == 502
        assert mock_broadcast.await_count == 2
        rollback_call = mock_broadcast.await_args_list[1]
        assert rollback_call.args[0] == [WORKER_ADDR]
        assert rollback_call.args[1] == "/datasets/unload"
        assert rollback_call.args[2] == {"dataset_id": "dataset-rollback"}

    @pytest.mark.asyncio
    async def test_reregister_revokes_old_key(self, client):
        first = await _register_dataset(client, dataset_id="dataset-rekey")

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR]
            mock_broadcast.return_value = [
                {
                    "addr": WORKER_ADDR,
                    "status": 200,
                    "data": {"steps_per_epoch": 10, "dataset_size": 100},
                }
            ]
            second_resp = await client.post(
                "/v1/datasets/register",
                json={
                    "dataset_id": "dataset-rekey",
                    "dataset_path": "/tmp/sample.jsonl",
                },
                headers=admin_headers(),
            )

        assert second_resp.status_code == 200
        second = second_resp.json()
        assert first["api_key"] != second["api_key"]

        old_key_resp = await client.post(
            "/v1/samples/fetch",
            json={"indices": [0]},
            headers={"Authorization": f"Bearer {first['api_key']}"},
        )
        assert old_key_resp.status_code == 401


class TestBroadcastEndpoints:
    @pytest.mark.asyncio
    async def test_epoch_advance_broadcasts_to_all_workers(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-f")
        dataset_key = reg_payload["api_key"]

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.return_value = [
                {"addr": WORKER_ADDR, "status": 200, "data": {}},
                {"addr": WORKER_ADDR_2, "status": 200, "data": {}},
            ]
            resp = await client.post(
                "/v1/epochs/advance",
                json={"epoch": 7},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        assert resp.json()["workers_reset"] == 2

    @pytest.mark.asyncio
    async def test_epoch_advance_no_workers_returns_503(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-noworkers")
        dataset_key = reg_payload["api_key"]

        with patch(
            f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
        ) as mock_workers:
            mock_workers.return_value = []
            resp = await client.post(
                "/v1/epochs/advance",
                json={"epoch": 7},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 503

    @pytest.mark.asyncio
    async def test_epoch_advance_worker_failure_returns_502(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-partial")
        dataset_key = reg_payload["api_key"]

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.return_value = [
                {"addr": WORKER_ADDR, "status": 200, "data": {}},
                {"addr": WORKER_ADDR_2, "status": 500, "error": "boom"},
            ]
            resp = await client.post(
                "/v1/epochs/advance",
                json={"epoch": 7},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_state_save_broadcasts_to_all_workers(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-g")
        dataset_key = reg_payload["api_key"]

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.return_value = [
                {"addr": WORKER_ADDR, "status": 200, "data": {}},
                {"addr": WORKER_ADDR_2, "status": 200, "data": {}},
            ]
            resp = await client.post(
                "/v1/state/save",
                json={"path": "/tmp/ckpt"},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "path": "/tmp/ckpt"}

    @pytest.mark.asyncio
    async def test_state_load_broadcasts_to_all_workers(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-h")
        dataset_key = reg_payload["api_key"]

        with (
            patch(
                f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
            ) as mock_workers,
            patch(
                f"{MODULE}._broadcast_to_workers", new_callable=AsyncMock
            ) as mock_broadcast,
        ):
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            mock_broadcast.return_value = [
                {"addr": WORKER_ADDR, "status": 200, "data": {}},
                {"addr": WORKER_ADDR_2, "status": 200, "data": {}},
            ]
            resp = await client.post(
                "/v1/state/load",
                json={"path": "/tmp/ckpt"},
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestStatusAndWorkers:
    @pytest.mark.asyncio
    async def test_workers_returns_router_workers(self, client):
        with patch(
            f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
        ) as mock_workers:
            mock_workers.return_value = [WORKER_ADDR, WORKER_ADDR_2]
            resp = await client.get("/v1/workers", headers=admin_headers())

        assert resp.status_code == 200
        assert resp.json() == {
            "workers": [{"addr": WORKER_ADDR}, {"addr": WORKER_ADDR_2}]
        }

    @pytest.mark.asyncio
    async def test_status_returns_dataset_id(self, client):
        reg_payload = await _register_dataset(client, dataset_id="dataset-status")
        dataset_key = reg_payload["api_key"]

        with patch(
            f"{MODULE}._query_router", new_callable=AsyncMock
        ) as mock_query_router:
            mock_query_router.side_effect = RuntimeError("router unavailable")
            resp = await client.get(
                "/v1/status",
                headers={"Authorization": f"Bearer {dataset_key}"},
            )

        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "ok"
        assert payload["dataset_id"] == "dataset-status"


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_requires_admin_key(self, client):
        resp = await client.post(
            "/v1/shutdown",
            headers={"Authorization": "Bearer ds-not-admin"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_shutdown_returns_ok(self, client):
        with patch(
            f"{MODULE}._get_all_worker_addrs", new_callable=AsyncMock
        ) as mock_workers:
            mock_workers.return_value = []
            resp = await client.post("/v1/shutdown", headers=admin_headers())
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

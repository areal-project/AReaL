# SPDX-License-Identifier: Apache-2.0

"""Rollout resource reconfiguration beneath a persistent CPU service plane."""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import nullcontext
from pathlib import Path

from areal.infra.utils.process import (
    OWNER_ENV,
    owned_processes,
    stop_owned_processes,
    wait_for_gpu_release,
)
from areal.utils.atomic import atomic_json
from areal.utils.network import format_hostport


class ManagedReplicas:
    def __init__(self, controller, reservation, devices, state_root: Path):
        self.controller = controller
        self.reservation = tuple(reservation)
        self.devices = tuple(devices)
        self.path = state_root / "rollout-owners.json"
        self.records: list[dict] = []
        self.server_args = None
        if self.path.exists():
            old = json.loads(self.path.read_text())
            if old["reservation"] != list(self.reservation) or old["records"]:
                raise RuntimeError(
                    "Previous Rollout ownership is unconfirmed; preserve its journal for cleanup"
                )

    def guard(self, device: int) -> str:
        worker = self.controller.workers[self.reservation.index(device)]
        return f"http://{format_hostport(worker.ip, int(worker.worker_ports[0]))}"

    def persist(self) -> None:
        atomic_json(
            self.path, {"reservation": list(self.reservation), "records": self.records}
        )

    async def launch(self, devices) -> None:
        from areal.api.cli_args import SGLangConfig
        from areal.api.io_struct import LocalInfServerInfo
        from areal.infra.utils.launcher import TRITON_CACHE_PATH

        if self.records:
            raise RuntimeError("Stop the previous GPU engines before rebuilding")
        wait_for_gpu_release(devices, timeout=0)
        controller = self.controller
        client = await controller._get_async_client()
        controller._inf_addrs = []
        controller._server_infos = []
        for device in devices:
            guard = self.guard(device)
            index = self.reservation.index(device)
            response = await client.post(
                f"{guard}/alloc_ports", json={"count": 1}, timeout=30
            )
            response.raise_for_status()
            host, port = response.json()["host"], response.json()["ports"][0]
            owner = uuid.uuid4().hex
            # Persist ownership BEFORE launch, including ambiguous RPC outcomes.
            self.records.append(
                {"device": device, "guard": guard, "index": index, "owner": owner}
            )
            self.persist()
            args = {
                **self.server_args,
                "host": host,
                "port": port,
                "nnodes": 1,
                "node_rank": 0,
                "dist_init_addr": None,
            }
            response = await client.post(
                f"{guard}/fork",
                json={
                    "role": "inf-server",
                    "worker_index": index,
                    "raw_cmd": SGLangConfig.build_cmd_from_args(args),
                    "env": {
                        "CUDA_VISIBLE_DEVICES": str(device),
                        OWNER_ENV: owner,
                        "TRITON_CACHE_DIR": f"{TRITON_CACHE_PATH}/elastic-{owner}",
                        "TRITON_CACHE_PATH": f"{TRITON_CACHE_PATH}/elastic-{owner}",
                    },
                },
                timeout=30,
            )
            response.raise_for_status()
            address = f"http://{format_hostport(host, port)}"
            controller._inf_addrs.append(address)
            controller._server_infos.append(
                LocalInfServerInfo(host=host, port=port, process=None)
            )
        for index, address in enumerate(controller._inf_addrs):
            await controller._async_wait_for_service(
                f"{address}/health",
                f"InfServer-{index}",
                timeout=controller.config.setup_timeout,
            )
        self.devices = tuple(devices)

    def stop(self) -> None:
        for record in list(self.records):
            response = self.controller._sync_client.post(
                f"{record['guard']}/kill_forked_worker",
                json={"role": "inf-server", "worker_index": record["index"]},
                timeout=30,
            )
            if response.status_code not in (200, 404):
                response.raise_for_status()
                raise RuntimeError("Inference guard did not confirm termination")
            stop_owned_processes(record["owner"])
            if owned_processes(record["owner"]):
                raise RuntimeError("Inference descendants remain alive")
            wait_for_gpu_release(
                [record["device"]], timeout=self.controller.config.setup_timeout
            )
            self.records.remove(record)
            self.persist()
        self.controller._inf_addrs = []
        self.controller._server_infos = []


def drain_model_calls(controller) -> None:
    controller._ensure_initialized()
    controller.pause()
    from areal.infra.utils.concurrent import run_async_task

    async def drain_all():
        client = await controller._get_async_client()

        async def drain(address):
            response = await client.post(
                f"{address}/reconfiguration/drain",
                json={"timeout": controller.config.request_timeout},
                headers={"Authorization": f"Bearer {controller.config.admin_api_key}"},
                timeout=controller.config.request_timeout + 10,
            )
            response.raise_for_status()
            if response.json() != {"status": "drained", "active": 0}:
                raise RuntimeError("Data proxy did not confirm model-call drain")

        results = await asyncio.gather(
            *[drain(address) for address in controller._data_proxy_addrs],
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, Exception)]
        if errors:
            raise ExceptionGroup("Model-call drain incomplete", errors)

    run_async_task(drain_all)


def open_model_calls(controller) -> None:
    for address in controller._data_proxy_addrs:
        response = controller._sync_client.post(
            f"{address}/reconfiguration/open",
            headers={"Authorization": f"Bearer {controller.config.admin_api_key}"},
        )
        response.raise_for_status()
        if response.json() != {"status": "open"}:
            raise RuntimeError("Data proxy did not acknowledge model-call admission")
    controller.resume()


def rebuild_inference_replicas(controller, devices, *, timing=None) -> None:
    from areal.infra.utils.concurrent import run_async_task

    if controller._managed_replicas is None:
        raise RuntimeError("Rollout was not initialized with a resource pool")
    phase = timing or (lambda name: nullcontext())
    with phase("rollout_start"):
        run_async_task(controller._managed_replicas.launch, tuple(devices))
    # Preserve every session store and router assignment; only replace bridges.
    with phase("proxy_rebind"):
        for index, address in enumerate(controller._data_proxy_addrs):
            response = controller._sync_client.post(
                f"{address}/configure_backend",
                json={
                    "backend_addr": controller._inf_addrs[
                        index % len(controller._inf_addrs)
                    ]
                },
                headers={"Authorization": f"Bearer {controller.config.admin_api_key}"},
            )
            response.raise_for_status()
            if (
                response.json().get("backend_addr")
                != controller._inf_addrs[index % len(controller._inf_addrs)]
            ):
                raise RuntimeError(
                    "Data proxy did not acknowledge its replacement backend"
                )


def arm_reconfiguration_observation(controller, transition_id: str) -> None:
    for address in controller._data_proxy_addrs:
        response = controller._sync_client.post(
            f"{address}/reconfiguration/observe",
            json={"transition_id": transition_id},
            headers={"Authorization": f"Bearer {controller.config.admin_api_key}"},
        )
        response.raise_for_status()
        if response.json() != {"status": "armed", "transition_id": transition_id}:
            raise RuntimeError("Data proxy did not acknowledge observation marker")


def reconfiguration_metrics(controller) -> list[dict]:
    rows = []
    for address in controller._data_proxy_addrs:
        response = controller._sync_client.get(
            f"{address}/reconfiguration/metrics",
            headers={"Authorization": f"Bearer {controller.config.admin_api_key}"},
        )
        response.raise_for_status()
        rows.append({"proxy": address, **response.json()})
    return rows


async def set_generation_paused(controller, paused: bool) -> None:
    client = await controller._get_async_client()

    async def post(address, endpoint):
        response = await client.post(
            f"{address}{endpoint}",
            json={"paused": paused},
            headers={"Authorization": f"Bearer {controller.config.admin_api_key}"},
        )
        response.raise_for_status()
        if response.json() != {"status": "ok", "paused": paused}:
            raise RuntimeError("Every proxy/engine must acknowledge generation control")

    async def local_state():
        await asyncio.gather(
            *[
                post(address, "/reconfiguration/pause_state")
                for address in controller._data_proxy_addrs
            ]
        )

    # Pause all bridges before aborting a backend; resume each unique engine
    # before releasing its proxy waiters. Require EVERY acknowledgment.
    if paused:
        await local_state()
    await asyncio.gather(
        *[
            post(address, "/pause_generation" if paused else "/continue_generation")
            for address in controller._data_proxy_addrs[: len(controller._inf_addrs)]
        ]
    )
    if not paused:
        await local_state()

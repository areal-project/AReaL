# SPDX-License-Identifier: Apache-2.0

"""Local TP1 SGLang GPU engines beneath a persistent CPU service plane."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from areal.elastic.control.processes import (
    OWNER_ENV,
    gpu_processes,
    owned_processes,
    resolve_gpu_uuids,
    stop_owned_processes,
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
        present = gpu_processes()
        if any(present.get(gpu) for gpu in resolve_gpu_uuids(list(devices))):
            raise RuntimeError("Target inference GPUs are occupied")
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
            present = gpu_processes()
            if any(present.get(gpu) for gpu in resolve_gpu_uuids([record["device"]])):
                raise RuntimeError(
                    "Inference GPU is still occupied; do not transfer it"
                )
            self.records.remove(record)
            self.persist()
        self.controller._inf_addrs = []
        self.controller._server_infos = []

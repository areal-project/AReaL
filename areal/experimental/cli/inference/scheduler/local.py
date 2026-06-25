# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

from areal.experimental.cli.inference.scheduler.base import (
    Scheduler,
    SchedulerError,
    TaskAllocation,
    TaskHandle,
    TaskSpec,
)
from areal.experimental.cli.process import spawn_process
from areal.utils.logging import getLogger
from areal.utils.network import find_free_ports

logger = getLogger("InfLocalScheduler")


def _detect_gpus() -> list[int]:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if vis:
        try:
            return [int(x) for x in vis.split(",") if x.strip()]
        except ValueError:
            logger.warning(
                "ignoring malformed CUDA_VISIBLE_DEVICES=%r, falling back to detection",
                vis,
            )
    try:
        nvidia = [
            d
            for d in os.listdir("/dev")
            if d.startswith("nvidia") and d[len("nvidia") :].isdigit()
        ]
    except OSError:
        return [0]
    return sorted(int(d[len("nvidia") :]) for d in nvidia) or [0]


class LocalScheduler(Scheduler):
    """Ephemeral local subprocess scheduler for ``areal inf``.

    The instance is discarded after each CLI command — Popen handles are
    not retained, only the PID written into the returned TaskHandle. GPUs
    are masked via ``CUDA_VISIBLE_DEVICES`` so the spawned process always
    sees its devices as ``0..N-1``, matching K8s device-plugin semantics.
    """

    def __init__(
        self,
        *,
        all_gpus: list[int] | None = None,
        occupied_gpus: set[int] | None = None,
        host: str = "127.0.0.1",
    ) -> None:
        self.all_gpus = list(all_gpus) if all_gpus is not None else _detect_gpus()
        self.occupied_gpus: set[int] = set(occupied_gpus or ())
        self.host = host
        # Ports already handed out by this scheduler instance — passed as
        # exclude_ports so the next find_free_ports call cannot return the
        # same port between subsequent submits within one CLI command.
        self._allocated_ports: set[int] = set()

    def submit(self, spec: TaskSpec) -> TaskHandle:
        gpus = self._allocate_gpus(spec.resources.gpu)
        ports = self._allocate_ports(spec.resources.ports)
        alloc = TaskAllocation(host=self.host, ports=ports, gpu_devices=gpus)
        cmd = spec.cmd_builder(alloc)

        env = dict(spec.env)
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)

        try:
            pid = spawn_process(cmd, spec.log_file, env=env)
        except OSError as exc:
            raise SchedulerError(f"failed to spawn task {spec.name!r}: {exc}") from exc

        self.occupied_gpus.update(gpus)
        logger.info(
            "submitted %s pid=%d host=%s ports=%s gpus=%s",
            spec.name,
            pid,
            alloc.host,
            ports,
            gpus,
        )
        return TaskHandle(
            host=alloc.host,
            ports=ports,
            gpu_devices=list(gpus),
            ref={"pid": pid},
        )

    def _allocate_gpus(self, n: int) -> list[int]:
        if n <= 0:
            return []
        free = [g for g in self.all_gpus if g not in self.occupied_gpus]
        if len(free) < n:
            raise SchedulerError(
                f"need {n} GPUs but only {len(free)} free "
                f"(total={self.all_gpus}, occupied={sorted(self.occupied_gpus)})"
            )
        return free[:n]

    def _allocate_ports(self, n: int) -> list[int]:
        ports = find_free_ports(n, exclude_ports=self._allocated_ports)
        self._allocated_ports.update(ports)
        return ports

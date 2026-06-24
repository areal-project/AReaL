# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


class SchedulerError(Exception):
    pass


@dataclass
class TaskResources:
    gpu: int = 0
    ports: int = 1


@dataclass
class TaskAllocation:
    host: str
    ports: list[int]
    gpu_devices: list[int]


@dataclass
class TaskSpec:
    """cmd_builder receives the TaskAllocation so host/port/GPU can be filled
    in at placement time — required for K8s/Slurm where the host is only
    known after scheduling."""

    name: str
    cmd_builder: Callable[[TaskAllocation], list[str]]
    log_file: Path
    resources: TaskResources = field(default_factory=TaskResources)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class TaskHandle:
    """Backend-agnostic identity of a running task.

    ``ref`` carries backend-specific identity payload — ``{"pid": int}`` for
    Local, ``{"pod_name", "namespace"}`` for K8s, ``{"job_id"}`` for Slurm.
    """

    host: str
    ports: list[int]
    gpu_devices: list[int] = field(default_factory=list)
    ref: dict = field(default_factory=dict)

    @property
    def addr(self) -> str:
        if not self.host or not self.ports:
            return ""
        return f"http://{self.host}:{self.ports[0]}"

    @property
    def pid(self) -> int:
        return int(self.ref.get("pid", 0) or 0)


class Scheduler(ABC):
    @abstractmethod
    def submit(self, spec: TaskSpec) -> TaskHandle: ...


def build_scheduler(backend: str, **kwargs) -> Scheduler:
    if backend == "local":
        from areal.experimental.cli.inference.scheduler.local import LocalScheduler

        return LocalScheduler(**kwargs)
    raise SchedulerError(f"unknown scheduler backend: {backend!r}")

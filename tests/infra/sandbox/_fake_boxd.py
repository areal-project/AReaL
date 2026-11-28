# SPDX-License-Identifier: Apache-2.0

"""A small in-memory stand-in for ``boxd.AsyncBoxd`` used by the unit tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from boxd.errors import ConflictError, NotFoundError


@dataclass
class FakeExecResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


@dataclass
class FakeMachines:
    exec_handler: object = None
    created: list[dict] = field(default_factory=list)
    execs: list[dict] = field(default_factory=list)
    uploads: dict[str, str] = field(default_factory=dict)
    deleted: list[str] = field(default_factory=list)
    delete_errors: list[Exception] = field(default_factory=list)
    ready_errors: list[Exception] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    started: list[str] = field(default_factory=list)
    stopped: set[str] = field(default_factory=set)
    gone: set[str] = field(default_factory=set)

    def __post_init__(self):
        self.files = SimpleNamespace(upload=self._upload, delete=self._remove)

    async def create(self, name, **kwargs):
        if kwargs.get("from_snapshot") and kwargs.get("env"):
            # Same rule as the real SDK.
            raise ValueError("create(from_snapshot=...) does not accept env")
        machine_id = f"vm-{len(self.created) + 1}"
        self.created.append({"id": machine_id, "name": name, **kwargs})
        return SimpleNamespace(id=machine_id, name=name)

    async def wait_until_ready(self, machine_id, **kwargs):
        if self.ready_errors:
            raise self.ready_errors.pop(0)
        return SimpleNamespace(id=machine_id, status="running")

    async def exec(self, machine_id, command, **kwargs):
        if machine_id in self.gone:
            raise NotFoundError("machine not found")
        self.execs.append({"id": machine_id, "command": command, **kwargs})
        if machine_id in self.stopped:
            raise ConflictError("machine is stopped")
        if self.exec_handler is None:
            return FakeExecResult()
        return self.exec_handler(command, kwargs)

    async def get(self, machine_id, **kwargs):
        if machine_id in self.gone:
            raise NotFoundError("machine not found")
        status = "stopped" if machine_id in self.stopped else "running"
        return SimpleNamespace(id=machine_id, status=status)

    async def start(self, machine_id, **kwargs):
        self.started.append(machine_id)
        self.stopped.discard(machine_id)

    async def delete(self, machine_id, **kwargs):
        if machine_id in self.gone:
            raise NotFoundError("machine not found")
        if self.delete_errors:
            raise self.delete_errors.pop(0)
        self.deleted.append(machine_id)
        self.gone.add(machine_id)

    async def _upload(self, machine_id, path, data, **kwargs):
        if machine_id in self.gone:
            raise NotFoundError("machine not found")
        self.uploads[path] = data
        return len(data)

    async def _remove(self, machine_id, path, **kwargs):
        self.removed.append(path)


class FakeAsyncBoxd:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.machines = FakeMachines()
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1

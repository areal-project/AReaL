# SPDX-License-Identifier: Apache-2.0

"""An in-memory fake of the ``opensandbox`` SDK for offline tests.

``install`` registers fake modules under the ``opensandbox`` package names
through pytest's ``monkeypatch``, so the lazy loader in
``_opensandbox_session`` imports them without the real dependency and pytest
restores both ``sys.modules`` and the loader cache at teardown. The fake
sandbox records what was created and answers command runs from a scriptable
queue of results.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from datetime import timedelta

MODULE_NAMES = (
    "opensandbox",
    "opensandbox.sync",
    "opensandbox.config",
    "opensandbox.exceptions",
    "opensandbox.models",
    "opensandbox.models.execd",
    "opensandbox.models.sandboxes",
)


class FakeSandboxException(Exception):
    pass


class FakeConnectionConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeConnectionConfigSync:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeRunCommandOpts:
    def __init__(self, timeout=None, **kwargs):
        self.timeout = timeout
        self.kwargs = kwargs


class FakeNetworkPolicy:
    def __init__(self, default_action=None, egress=None):
        self.default_action = default_action
        self.egress = egress


class FakeOutputMessage:
    def __init__(self, text):
        self.text = text


class FakeExecutionLogs:
    def __init__(self, stdout="", stderr=""):
        self.stdout = [FakeOutputMessage(stdout)] if stdout else []
        self.stderr = [FakeOutputMessage(stderr)] if stderr else []


class FakeExecutionError:
    def __init__(self, name, value, exit_code=None):
        self.name = name
        self.value = value
        self.exit_code = exit_code


class FakeExecution:
    def __init__(self, stdout="", stderr="", exit_code=0, error=None):
        self.logs = FakeExecutionLogs(stdout, stderr)
        self.exit_code = exit_code
        self.error = error


class FakeSandboxStatus:
    def __init__(self, state):
        self.state = state


class FakeSandboxInfo:
    def __init__(self, state):
        self.status = FakeSandboxStatus(state)


class _Files:
    def __init__(self, state):
        self._state = state

    def write(self, path, data):
        if self._state.write_errors:
            raise self._state.write_errors.pop(0)
        self._state.files[path] = data

    def delete(self, paths):
        for path in paths:
            self._state.files.pop(path, None)


class FakeAsyncFiles(_Files):
    async def write_file(self, path, data, **kwargs):
        self.write(path, data)

    async def delete_files(self, paths):
        self.delete(paths)


class FakeSyncFiles(_Files):
    def write_file(self, path, data, **kwargs):
        self.write(path, data)

    def delete_files(self, paths):
        self.delete(paths)


class _Commands:
    def __init__(self, state):
        self._state = state


class FakeAsyncCommands(_Commands):
    async def run(self, command, opts=None):
        return self._state.next_execution(command, opts)


class FakeSyncCommands(_Commands):
    def run(self, command, opts=None):
        return self._state.next_execution(command, opts)


class FakeSandboxBase:
    def __init__(self, state, sandbox_id):
        self.id = sandbox_id
        self._state = state
        self.renew_calls: list[timedelta] = []
        self.destroy_calls = 0
        self.get_info_results: list[FakeSandboxInfo | Exception] = []
        self.get_info_default = FakeSandboxInfo("Running")


class FakeSandbox(FakeSandboxBase):
    async def get_info(self):
        result = self._state.pop_get_info(self)
        if isinstance(result, Exception):
            raise result
        return result

    async def renew(self, timeout):
        self.renew_calls.append(timeout)
        return {"expires_at": timeout}

    async def destroy(self):
        self.destroy_calls += 1
        self._state.destroyed_ids.append(self.id)


class FakeSandboxSync(FakeSandboxBase):
    def get_info(self):
        result = self._state.pop_get_info(self)
        if isinstance(result, Exception):
            raise result
        return result

    def renew(self, timeout):
        self.renew_calls.append(timeout)
        return {"expires_at": timeout}

    def destroy(self):
        self.destroy_calls += 1
        self._state.destroyed_ids.append(self.id)


class FakeOpensandboxState:
    """Shared scriptable state for the fake SDK modules."""

    def __init__(self):
        self.create_calls: list[dict] = []
        self.create_delays: list[float] = []
        self.executions: list[FakeExecution | Exception] = []
        self.executed_commands: list[list[str]] = []
        self.last_opts = None
        self.files: dict[str, str] = {}
        self.write_errors: list[Exception] = []
        self.destroyed_ids: list[str] = []
        self.sandbox_counter = 0
        self.sandboxes: list[FakeSandboxBase] = []

    def queue_execution(self, result):
        self.executions.append(result)

    def next_execution(self, command, opts):
        self.executed_commands.append(list(command))
        self.last_opts = opts
        if not self.executions:
            return FakeExecution(stdout="")
        result = self.executions.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def pop_get_info(self, sandbox):
        if sandbox.get_info_results:
            return sandbox.get_info_results.pop(0)
        return sandbox.get_info_default

    def make_sandbox(self, sync: bool):
        self.sandbox_counter += 1
        sandbox_id = f"sb-{self.sandbox_counter}"
        files_cls = FakeSyncFiles if sync else FakeAsyncFiles
        commands_cls = FakeSyncCommands if sync else FakeAsyncCommands
        sandbox_cls = FakeSandboxSync if sync else FakeSandbox
        sandbox = sandbox_cls(self, sandbox_id)
        sandbox.files = files_cls(self)
        sandbox.commands = commands_cls(self)
        self.sandboxes.append(sandbox)
        return sandbox


def _build_modules(state: FakeOpensandboxState) -> dict[str, types.ModuleType]:
    import importlib.machinery

    async def async_create(**kwargs):
        if state.create_delays:
            import asyncio

            await asyncio.sleep(state.create_delays.pop(0))
        state.create_calls.append(kwargs)
        return state.make_sandbox(sync=False)

    def sync_create(**kwargs):
        state.create_calls.append(kwargs)
        return state.make_sandbox(sync=True)

    opensandbox = types.ModuleType("opensandbox")
    opensandbox.Sandbox = type("Sandbox", (), {"create": staticmethod(async_create)})

    sync_module = types.ModuleType("opensandbox.sync")
    sync_module.SandboxSync = type(
        "SandboxSync", (), {"create": staticmethod(sync_create)}
    )

    config_module = types.ModuleType("opensandbox.config")
    config_module.ConnectionConfig = FakeConnectionConfig
    config_module.ConnectionConfigSync = FakeConnectionConfigSync

    exceptions_module = types.ModuleType("opensandbox.exceptions")
    exceptions_module.SandboxException = FakeSandboxException

    execd_module = types.ModuleType("opensandbox.models.execd")
    execd_module.RunCommandOpts = FakeRunCommandOpts

    sandboxes_module = types.ModuleType("opensandbox.models.sandboxes")
    sandboxes_module.NetworkPolicy = FakeNetworkPolicy

    models_module = types.ModuleType("opensandbox.models")
    models_module.execd = execd_module
    models_module.sandboxes = sandboxes_module

    modules = {
        "opensandbox": opensandbox,
        "opensandbox.sync": sync_module,
        "opensandbox.config": config_module,
        "opensandbox.exceptions": exceptions_module,
        "opensandbox.models": models_module,
        "opensandbox.models.execd": execd_module,
        "opensandbox.models.sandboxes": sandboxes_module,
    }
    for name, module in modules.items():
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return modules


@dataclass
class InstalledFake:
    """Handle returned by ``install``; teardown is owned by ``monkeypatch``."""

    state: FakeOpensandboxState


def install(monkeypatch) -> InstalledFake:
    """Install the fake modules into ``sys.modules`` and reset the loader cache.

    Everything is registered through ``monkeypatch`` so pytest restores the
    original modules and cached SDK classes even if a test fails.
    """
    import areal.infra.sandbox._opensandbox_session as session_module

    for name in MODULE_NAMES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    state = FakeOpensandboxState()
    for name, module in _build_modules(state).items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in (
        "Sandbox",
        "SandboxSync",
        "RunCommandOpts",
        "NetworkPolicy",
        "ConnectionConfig",
        "ConnectionConfigSync",
        "SandboxException",
    ):
        monkeypatch.setattr(session_module, name, None)
    return InstalledFake(state)

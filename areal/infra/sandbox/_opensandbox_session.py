# SPDX-License-Identifier: Apache-2.0

"""One OpenSandbox sandbox used as a sandbox: create, run code, delete.

Shared by ``OpenSandboxRunner`` (one fresh Python process per run) and the TIR
``OpenSandboxPythonTool`` (a persistent interpreter whose state survives
across calls).

Every sandbox is created with a server-side expiration (``ttl_seconds``, one
hour by default): if the AReaL process dies before cleanup runs, the platform
deletes the sandbox on its own instead of leaking it. While the process is
alive and the sandbox is in use, the lease is renewed after each run, so an
actively used sandbox is not expired underneath a rollout.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areal.utils import logging

logger = logging.getLogger("OpenSandboxSession")

SANDBOX_DIR = "/tmp/.areal-sandbox"
REPL_PATH = f"{SANDBOX_DIR}/repl.py"
REPL_SOCKET = f"{SANDBOX_DIR}/repl.sock"
REPL_PID = f"{SANDBOX_DIR}/repl.pid"
PYTHON_BIN = "python3"
TIMEOUT_EXIT_CODE = 124
DEFAULT_METADATA = {"areal-sandbox": "true"}
# Server-side sandbox lifetime in seconds. A killed process stops renewing
# the lease and the platform deletes the sandbox at expiry, so a failed run
# does not consume the provider's concurrency budget. ``ttl_seconds=None``
# requires explicit cleanup instead.
DEFAULT_TTL_SECONDS = 3600
# Extra seconds the exec call waits beyond the code's own ``timeout``, so the
# in-sandbox timeout (coreutils ``timeout`` / the interpreter's SIGALRM)
# fires first and reports a clean result.
_EXEC_GRACE_SECONDS = 15.0
_CREATE_ATTEMPTS_CANCEL_WAIT = 30.0

Sandbox = None
SandboxSync = None
RunCommandOpts = None
NetworkPolicy = None
ConnectionConfig = None
SandboxException = None


def _load_opensandbox_sdk() -> dict[str, Any]:
    global Sandbox, SandboxSync, RunCommandOpts, NetworkPolicy, ConnectionConfig
    global SandboxException

    if all(
        value is not None
        for value in (Sandbox, SandboxSync, RunCommandOpts, NetworkPolicy, ConnectionConfig)
    ):
        return {
            "Sandbox": Sandbox,
            "SandboxSync": SandboxSync,
            "RunCommandOpts": RunCommandOpts,
            "NetworkPolicy": NetworkPolicy,
            "ConnectionConfig": ConnectionConfig,
            "SandboxException": SandboxException,
        }

    try:
        from opensandbox import Sandbox as ImportedSandbox
        from opensandbox.config import ConnectionConfig as ImportedConnectionConfig
        from opensandbox.exceptions import (
            SandboxException as ImportedSandboxException,
        )
        from opensandbox.models.execd import RunCommandOpts as ImportedRunCommandOpts
        from opensandbox.models.sandboxes import NetworkPolicy as ImportedNetworkPolicy
        from opensandbox.sync import SandboxSync as ImportedSandboxSync
    except ImportError as exc:
        raise ImportError(
            "OpenSandboxRunner requires the optional 'opensandbox' dependency. "
            "Install it with `uv sync --extra sandbox`."
        ) from exc

    Sandbox = ImportedSandbox
    SandboxSync = ImportedSandboxSync
    RunCommandOpts = ImportedRunCommandOpts
    NetworkPolicy = ImportedNetworkPolicy
    ConnectionConfig = ImportedConnectionConfig
    SandboxException = ImportedSandboxException

    return {
        "Sandbox": Sandbox,
        "SandboxSync": SandboxSync,
        "RunCommandOpts": RunCommandOpts,
        "NetworkPolicy": NetworkPolicy,
        "ConnectionConfig": ConnectionConfig,
        "SandboxException": SandboxException,
    }


@dataclass
class OpenSandboxCellError:
    name: str
    value: str
    traceback: str


@dataclass
class OpenSandboxCellResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: OpenSandboxCellError | None = None


@dataclass
class OpenSandboxExecResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: str | None = None


class OpenSandboxSession:
    """Owns one OpenSandbox sandbox.

    The sandbox is created with a server-side TTL (``ttl_seconds``). If this
    process never gets to delete it, the platform still reclaims it at
    expiry. If the sandbox is already gone when the next piece of code
    arrives, a new one is created.
    """

    def __init__(
        self,
        *,
        image: str | None = None,
        snapshot_id: str | None = None,
        cpu: str | None = None,
        memory: str | None = None,
        env_vars: Mapping[str, str] | None = None,
        metadata: Mapping[str, str] | None = None,
        network_block_all: bool = False,
        ttl_seconds: int | None = DEFAULT_TTL_SECONDS,
        api_key: str | None = None,
        domain: str | None = None,
        create_timeout: float = 120.0,
        default_timeout: int = 30,
    ):
        if image is None and snapshot_id is None:
            image = "python:3.11"
        if image is not None and snapshot_id is not None:
            raise ValueError("Pass either image or snapshot_id, not both")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive or None, got {ttl_seconds}")
        self._sdk = _load_opensandbox_sdk()
        self.image = image
        self.snapshot_id = snapshot_id
        self.cpu = cpu
        self.memory = memory
        self.env_vars = dict(env_vars or {})
        self.metadata = {**DEFAULT_METADATA, **(metadata or {})}
        self.network_block_all = network_block_all
        self.ttl_seconds = ttl_seconds
        self.api_key = api_key
        self.domain = domain
        self.create_timeout = create_timeout
        self.default_timeout = default_timeout
        self.sandbox_id: str | None = None
        self._sandbox = None
        self._repl_started = False
        self._lock = asyncio.Lock()

    def _build_create_kwargs(self) -> dict[str, Any]:
        return build_create_kwargs(
            self._sdk,
            image=self.image,
            snapshot_id=self.snapshot_id,
            cpu=self.cpu,
            memory=self.memory,
            env_vars=self.env_vars,
            metadata=self.metadata,
            network_block_all=self.network_block_all,
            ttl_seconds=self.ttl_seconds,
            api_key=self.api_key,
            domain=self.domain,
        )

    async def astart(self) -> Any:
        """Create the sandbox and wait until it is ready."""
        if self._sandbox is not None:
            return self._sandbox

        create = asyncio.ensure_future(
            self._sdk["Sandbox"].create(**self._build_create_kwargs())
        )
        try:
            sandbox = await asyncio.shield(create)
        except asyncio.CancelledError:
            # The create may still finish on the server. Wait for it briefly
            # and delete the sandbox, so a cancelled rollout does not leave
            # one behind on top of the TTL.
            try:
                sandbox = await asyncio.wait_for(
                    asyncio.shield(create), timeout=_CREATE_ATTEMPTS_CANCEL_WAIT
                )
                self._sandbox = sandbox
                self.sandbox_id = sandbox.id
                await self.adelete()
            except Exception as exc:
                logger.warning(
                    "Cleanup after a cancelled OpenSandbox create failed: %s", exc
                )
            raise

        self._sandbox = sandbox
        self.sandbox_id = sandbox.id
        logger.debug("Created OpenSandbox sandbox %s", sandbox.id)
        return sandbox

    async def _aexec(self, command: list[str], *, timeout: float) -> OpenSandboxExecResult:
        sandbox = await self.astart()
        run_command_opts = self._sdk["RunCommandOpts"]
        opts = run_command_opts(timeout=timedelta_of(timeout))
        try:
            execution = await sandbox.commands.run(command, opts=opts)
        except self._sdk["SandboxException"] as exc:
            if await self._ais_gone(exc):
                raise _SandboxGoneError(str(exc)) from exc
            raise
        return _to_exec_result(execution)

    async def _ais_gone(self, exc: Exception) -> bool:
        """Probe the sandbox after a failed exec.

        Returns True when the sandbox no longer exists or has reached a
        terminal state, so the caller can recreate it and run the command
        again. The failed exec never ran code in that case.
        """
        del exc
        sandbox = self._sandbox
        if sandbox is None:
            return False
        try:
            info = await sandbox.get_info()
        except self._sdk["SandboxException"]:
            logger.warning(
                "OpenSandbox sandbox %s is no longer reachable; recreating it",
                self.sandbox_id,
            )
            self._forget()
            return True
        state = getattr(getattr(info, "status", None), "state", "") or ""
        if is_terminal_state(state):
            logger.warning(
                "OpenSandbox sandbox %s is %s; recreating it", self.sandbox_id, state
            )
            self._forget()
            return True
        return False

    def _forget(self) -> None:
        self._sandbox = None
        self.sandbox_id = None
        self._repl_started = False

    async def arun(self, command: list[str], *, timeout: float) -> OpenSandboxExecResult:
        """Run a command in the sandbox.

        Raises ``_SandboxGoneError`` when the sandbox no longer exists, so the
        caller decides whether rerunning the command in a fresh sandbox is
        safe (the caller's command may have state effects).
        """
        return await self._aexec(command, timeout=timeout)

    async def arun_script(self, code: str, *, timeout: float) -> OpenSandboxExecResult:
        """Run ``code`` in a fresh Python process.

        Exit code 124 means the code ran past ``timeout`` and was killed. The
        code is written to a file first, so its size is not bounded by the
        command line. If the sandbox was deleted since the last call (its TTL
        passed), the code runs in a new one: the failed attempt never
        delivered the code, so rerunning it is safe.
        """
        _check_timeout(timeout)
        script_path = f"{SANDBOX_DIR}/code-{uuid.uuid4().hex}.py"
        try:
            result = await self._run_script_once(script_path, code, timeout=timeout)
        except _SandboxGoneError:
            result = await self._run_script_once(script_path, code, timeout=timeout)
        await self._arenew_lease()
        return result

    async def _run_script_once(
        self, script_path: str, code: str, *, timeout: float
    ) -> OpenSandboxExecResult:
        sandbox = await self.astart()
        await sandbox.files.write_file(script_path, code)
        try:
            command = [
                "timeout",
                "-k",
                "2",
                str(timeout),
                PYTHON_BIN,
                script_path,
            ]
            return await self.arun(command, timeout=timeout + _EXEC_GRACE_SECONDS)
        finally:
            await self._aremove(script_path)

    async def _aremove(self, path: str) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            return
        try:
            await sandbox.files.delete_files([path])
        except Exception as exc:
            logger.debug("Could not remove %s: %s", path, exc)

    async def _arenew_lease(self) -> None:
        """Best-effort lease renewal; expiry still reaps a lost sandbox."""
        if self.ttl_seconds is None or self._sandbox is None:
            return
        try:
            await self._sandbox.renew(timeout=timedelta_of(self.ttl_seconds))
        except Exception as exc:
            logger.debug(
                "Could not renew the OpenSandbox lease for %s: %s",
                self.sandbox_id,
                exc,
            )

    async def _astart_repl(self) -> None:
        sandbox = await self.astart()
        source = Path(__file__).with_name("_opensandbox_repl.py").read_text(
            encoding="utf-8"
        )
        await sandbox.files.write_file(REPL_PATH, source)
        start = (
            f"if [ -f {REPL_PID} ]; then kill -9 $(cat {REPL_PID}) 2>/dev/null; fi; "
            f"rm -f {REPL_SOCKET}; "
            f"setsid nohup {PYTHON_BIN} -u {REPL_PATH} serve {REPL_SOCKET} "
            f"> {SANDBOX_DIR}/repl.log 2>&1 < /dev/null & echo $! > {REPL_PID}; "
            f"for i in $(seq 100); do [ -S {REPL_SOCKET} ] && exit 0; sleep 0.1; done; "
            "exit 1"
        )
        result = await self.arun(["bash", "-c", start], timeout=60)
        if result.exit_code != 0:
            raise RuntimeError(
                "Starting the Python interpreter in the OpenSandbox sandbox "
                f"failed: {result.stderr.strip()}"
            )
        self._repl_started = True

    async def arun_cell(self, code: str, *, timeout: float) -> OpenSandboxCellResult:
        """Run ``code`` in the persistent interpreter.

        Variables defined by earlier cells stay available. If the interpreter
        is gone (it crashed, or the cell hung past its timeout), it is
        restarted and the result says that the earlier state was lost. The
        same happens in a new sandbox if this one was deleted since the last
        call. A cell is never sent twice.
        """
        _check_timeout(timeout)
        async with self._lock:
            try:
                result = await self._arun_cell_once(code, timeout=timeout)
            except _SandboxGoneError:
                result = _restarted_result(
                    "RuntimeError",
                    "The sandbox no longer exists, so a new one will be created",
                )
                await self.astart()
                await self._astart_repl()
            await self._arenew_lease()
            return result

    async def _arun_cell_once(
        self, code: str, *, timeout: float
    ) -> OpenSandboxCellResult:
        await self.astart()
        if not self._repl_started:
            await self._astart_repl()

        prefix = [PYTHON_BIN, REPL_PATH, "call", REPL_SOCKET, str(timeout)]
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        result = await self.arun(
            [*prefix, encoded], timeout=timeout + _EXEC_GRACE_SECONDS
        )

        if self._repl_started and result.exit_code == 0:
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                error = payload.get("error")
                return OpenSandboxCellResult(
                    stdout=payload.get("stdout", ""),
                    stderr=payload.get("stderr", ""),
                    error=OpenSandboxCellError(**error) if error else None,
                )

        logger.warning(
            "Python interpreter in OpenSandbox sandbox %s stopped (exit code %s): %s",
            self.sandbox_id,
            result.exit_code,
            result.stderr.strip(),
        )
        await self._astart_repl()
        return _restarted_result(
            "RuntimeError", "The Python interpreter stopped unexpectedly"
        )

    async def adelete(self) -> None:
        """Delete the sandbox. The id is kept until the delete succeeds."""
        sandbox = self._sandbox
        self._repl_started = False
        if sandbox is None:
            return
        try:
            await sandbox.destroy()
            logger.debug("Deleted OpenSandbox sandbox %s", self.sandbox_id)
        except self._sdk["SandboxException"] as exc:
            logger.debug("OpenSandbox sandbox %s delete failed: %s", self.sandbox_id, exc)
        finally:
            self._sandbox = None
            self.sandbox_id = None


class _SandboxGoneError(RuntimeError):
    """The sandbox disappeared before or while a command ran."""


def build_create_kwargs(
    sdk: dict[str, Any],
    *,
    image: str | None,
    snapshot_id: str | None,
    cpu: str | None,
    memory: str | None,
    env_vars: Mapping[str, str],
    metadata: Mapping[str, str],
    network_block_all: bool,
    ttl_seconds: int | None,
    api_key: str | None,
    domain: str | None,
) -> dict[str, Any]:
    """Shared create-call arguments for the async and sync SDK entry points."""
    from datetime import timedelta

    config_cls = sdk["ConnectionConfig"]
    connection_kwargs: dict[str, Any] = {}
    if api_key is not None:
        connection_kwargs["api_key"] = api_key
    if domain is not None:
        connection_kwargs["domain"] = domain

    kwargs: dict[str, Any] = {
        "env": dict(env_vars) or None,
        "metadata": dict(metadata) or None,
        "connection_config": config_cls(**connection_kwargs),
    }
    if image is not None:
        kwargs["image"] = image
    else:
        kwargs["snapshot_id"] = snapshot_id
    resource: dict[str, str] = {}
    if cpu is not None:
        resource["cpu"] = cpu
    if memory is not None:
        resource["memory"] = memory
    if resource:
        kwargs["resource"] = resource
    if ttl_seconds is not None:
        kwargs["timeout"] = timedelta(seconds=ttl_seconds)
    if network_block_all:
        kwargs["network_policy"] = sdk["NetworkPolicy"](
            default_action="deny", egress=[]
        )
    return kwargs


def is_terminal_state(state: str | None) -> bool:
    """True when a sandbox state means it can no longer run commands."""
    return (state or "") in ("Stopping", "Terminated", "Failed")


def _to_exec_result(execution) -> OpenSandboxExecResult:
    stdout = _format_messages(getattr(getattr(execution, "logs", None), "stdout", []))
    stderr = _format_messages(getattr(getattr(execution, "logs", None), "stderr", []))
    exit_code = getattr(execution, "exit_code", None)
    error = None
    execution_error = getattr(execution, "error", None)
    if execution_error is not None:
        error = f"{execution_error.name}: {execution_error.value}"
        if exit_code is None:
            exit_code = getattr(execution_error, "exit_code", None) or 1
    if exit_code is None:
        exit_code = 0 if error is None else 1
    return OpenSandboxExecResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=int(exit_code),
        error=error,
    )


def _format_messages(messages) -> str:
    parts = []
    for message in messages or []:
        text = getattr(message, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def timedelta_of(seconds: float):
    from datetime import timedelta

    return timedelta(seconds=seconds)


def _check_timeout(timeout: float) -> None:
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")


def _restarted_result(name: str, value: str) -> OpenSandboxCellResult:
    message = (
        f"{value}. The interpreter was restarted; earlier variables are gone."
    )
    return OpenSandboxCellResult(
        error=OpenSandboxCellError(
            name=name,
            value=message,
            traceback=f"Traceback (most recent call last):\n{name}: {message}\n",
        )
    )

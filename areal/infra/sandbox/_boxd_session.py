# SPDX-License-Identifier: Apache-2.0

"""One boxd machine used as a sandbox: create, run code, delete.

Shared by ``BoxdRunner`` (one fresh Python process per run) and the TIR
``BoxdPythonTool`` (a persistent interpreter whose state survives across calls).
"""

from __future__ import annotations

import asyncio
import base64
import json
import shlex
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areal.utils import logging

logger = logging.getLogger("BoxdSession")

SANDBOX_DIR = "/home/boxd/.areal-sandbox"
REPL_PATH = f"{SANDBOX_DIR}/repl.py"
REPL_SOCKET = f"{SANDBOX_DIR}/repl.sock"
REPL_PID = f"{SANDBOX_DIR}/repl.pid"
VENV_DIR = f"{SANDBOX_DIR}/venv"
SYSTEM_PYTHON = "python3"
TIMEOUT_EXIT_CODE = 124
# The command runs as one shell line, and a single argv string is capped at
# 128 KiB on Linux. Commands longer than this send the code as a file instead.
_MAX_COMMAND_BYTES = 96 * 1024
# Extra seconds the exec call waits beyond the code's own timeout, so the
# in-machine timeout fires first and reports a clean result.
_EXEC_GRACE_SECONDS = 15.0
_EXEC_TIMEOUT_PREFIX = "exec timed out"
DEFAULT_LABELS = {"areal-sandbox": "true"}
# Seconds after creation at which boxd deletes the machine on its own, so a
# killed process does not leave machines behind. ``delete_after=None`` turns
# the timer off.
DEFAULT_DELETE_AFTER = 6 * 3600
_DELETE_ATTEMPTS = 3

BoxdError = None
ConflictError = None
NotFoundError = None


def _load_boxd_errors() -> dict[str, type[Exception]]:
    global BoxdError, ConflictError, NotFoundError

    if BoxdError is None or ConflictError is None or NotFoundError is None:
        try:
            from boxd.errors import BoxdError as ImportedBoxdError
            from boxd.errors import ConflictError as ImportedConflictError
            from boxd.errors import NotFoundError as ImportedNotFoundError
        except ImportError as exc:
            raise ImportError(
                "boxd sandboxes require the optional 'boxd' dependency. Install it with `uv sync --extra sandbox`."
            ) from exc

        BoxdError = ImportedBoxdError
        ConflictError = ImportedConflictError
        NotFoundError = ImportedNotFoundError

    return {
        "BoxdError": BoxdError,
        "ConflictError": ConflictError,
        "NotFoundError": NotFoundError,
    }


@dataclass
class BoxdCellError:
    name: str
    value: str
    traceback: str


@dataclass
class BoxdCellResult:
    stdout: str = ""
    stderr: str = ""
    error: BoxdCellError | None = None


class BoxdSession:
    """Owns one isolated boxd machine.

    The machine is always created with ``isolated=True``: it joins no private
    network and holds no credentials, which suits model-generated code. The
    API key stays on the AReaL side and never enters the machine.

    ``delete_after`` (seconds after creation) has boxd delete the machine even
    if this process never gets to. If the machine is gone when the next piece
    of code arrives, a new one is created.
    """

    def __init__(
        self,
        *,
        name_prefix: str = "areal-sandbox",
        org: str | None = None,
        region: str | None = None,
        snapshot: str | None = None,
        vcpu: int | None = None,
        memory: int | str | None = None,
        env_vars: Mapping[str, str] | None = None,
        labels: Mapping[str, str] | None = None,
        python_packages: Sequence[str] | None = None,
        delete_after: int | None = DEFAULT_DELETE_AFTER,
        create_timeout: float = 120.0,
    ):
        if delete_after is not None and delete_after <= 0:
            raise ValueError(
                f"delete_after must be positive or None, got {delete_after}"
            )
        self._errors = _load_boxd_errors()
        self.name_prefix = name_prefix
        self.org = org
        self.region = region
        self.snapshot = snapshot
        self.vcpu = vcpu
        self.memory = memory
        self.env_vars = dict(env_vars or {})
        self.labels = {**DEFAULT_LABELS, **(labels or {})}
        self.python_packages = list(python_packages or [])
        self.delete_after = delete_after
        self.create_timeout = create_timeout
        self.machine_id: str | None = None
        self._repl_started = False

    @property
    def python(self) -> str:
        if self.python_packages:
            return f"{VENV_DIR}/bin/python"
        return SYSTEM_PYTHON

    def _create_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"isolated": True, "labels": self.labels}
        optional = {
            "org": self.org,
            "region": self.region,
            "from_snapshot": self.snapshot,
            "vcpu": self.vcpu,
            "memory": self.memory,
            "delete_after": self.delete_after,
        }
        # A machine restored from a snapshot keeps the snapshot's environment.
        # env_vars still reach the code: they go with every command.
        if self.snapshot is None and self.env_vars:
            kwargs["env"] = self.env_vars
        kwargs.update({k: v for k, v in optional.items() if v is not None})
        return kwargs

    async def astart(self, client) -> str:
        """Create the machine and wait until it runs commands."""
        if self.machine_id is not None:
            return self.machine_id

        name = f"{self.name_prefix}-{uuid.uuid4().hex[:8]}"
        create = asyncio.ensure_future(
            client.machines.create(name, **self._create_kwargs())
        )
        try:
            machine = await asyncio.shield(create)
        except asyncio.CancelledError:
            # The create may still finish on the server. Wait for it and delete
            # the machine, so a cancelled rollout does not leave one behind.
            try:
                machine = await create
                self.machine_id = machine.id
                await self.adelete(client)
            except Exception as exc:
                logger.warning("Cleanup after a cancelled create failed: %s", exc)
            raise
        self.machine_id = machine.id
        logger.debug("Created boxd machine %s (%s)", name, machine.id)
        try:
            await client.machines.wait_until_ready(
                machine.id, timeout=self.create_timeout, region=self.region
            )
            if self.python_packages:
                await self._ainstall_packages(client)
        except BaseException:
            try:
                await self.adelete(client)
            except Exception as exc:
                logger.warning("Deleting boxd machine %s failed: %s", machine.id, exc)
            raise
        return machine.id

    async def _ainstall_packages(self, client) -> None:
        packages = " ".join(shlex.quote(p) for p in self.python_packages)
        command = (
            f"uv venv -q {VENV_DIR} && "
            f"uv pip install -q --python {VENV_DIR}/bin/python {packages}"
        )
        result = await self.aexec(client, ["bash", "-c", command], timeout=600)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Installing python_packages in the boxd machine failed: {result.stderr.strip()}"
            )

    async def aexec(self, client, command, *, timeout: float):
        """Run a command, starting the machine first if it was stopped."""
        assert self.machine_id is not None
        env = self.env_vars or None
        try:
            return await client.machines.exec(
                self.machine_id, command, env=env, timeout=timeout, region=self.region
            )
        except self._errors["ConflictError"]:
            # Only a stopped machine is started and the command run again: then
            # the first attempt never ran it.
            machine = await client.machines.get(self.machine_id, region=self.region)
            if str(machine.status) != "stopped":
                raise
            logger.debug("boxd machine %s is stopped; starting it", self.machine_id)
            await client.machines.start(self.machine_id, region=self.region)
            await client.machines.wait_until_ready(
                self.machine_id, timeout=self.create_timeout, region=self.region
            )
            self._repl_started = False
            return await client.machines.exec(
                self.machine_id, command, env=env, timeout=timeout, region=self.region
            )

    def _fits_inline(self, command: Sequence[str]) -> bool:
        # The limit is in bytes, and non-ASCII code takes several per character.
        size = _byte_len(shlex.join(command))
        size += sum(
            _byte_len(k) + _byte_len(shlex.quote(v)) + 2
            for k, v in self.env_vars.items()
        )
        return size <= _MAX_COMMAND_BYTES

    async def _aforget_if_gone(self, client) -> bool:
        """Drop the machine id if boxd no longer has the machine."""
        try:
            await client.machines.get(self.machine_id, region=self.region)
        except self._errors["NotFoundError"]:
            logger.warning(
                "boxd machine %s no longer exists; creating a new one",
                self.machine_id,
            )
            self.machine_id = None
            self._repl_started = False
            return True
        return False

    async def _aupload(self, client, suffix: str, data: str) -> str:
        path = f"{SANDBOX_DIR}/code-{uuid.uuid4().hex}{suffix}"
        await client.machines.files.upload(
            self.machine_id, path, data, region=self.region
        )
        return path

    async def arun_script(self, client, code: str, *, timeout: float):
        """Run ``code`` in a fresh Python process.

        Exit code 124 means the code ran past ``timeout`` and was killed. If
        the machine was deleted since the last call (its ``delete_after`` time
        passed), the code runs in a new one.
        """
        _check_timeout(timeout)
        if self.machine_id is None:
            return await self._arun_script_once(client, code, timeout=timeout)
        try:
            return await self._arun_script_once(client, code, timeout=timeout)
        except self._errors["NotFoundError"]:
            if not await self._aforget_if_gone(client):
                raise
            return await self._arun_script_once(client, code, timeout=timeout)

    async def _arun_script_once(self, client, code: str, *, timeout: float):
        await self.astart(client)
        prefix = ["timeout", "-k", "2", str(timeout), self.python]
        command = [*prefix, "-c", code]
        uploaded = None
        if not self._fits_inline(command):
            uploaded = await self._aupload(client, ".py", code)
            command = [*prefix, uploaded]
        try:
            return await self.aexec(
                client, command, timeout=timeout + _EXEC_GRACE_SECONDS
            )
        finally:
            if uploaded is not None:
                await self._aremove(client, uploaded)

    async def _aremove(self, client, path: str) -> None:
        try:
            await client.machines.files.delete(
                self.machine_id, path, region=self.region
            )
        except Exception as exc:
            logger.debug("Could not remove %s: %s", path, exc)

    async def _astart_repl(self, client) -> None:
        source = Path(__file__).with_name("_boxd_repl.py").read_text(encoding="utf-8")
        await client.machines.files.upload(
            self.machine_id, REPL_PATH, source, region=self.region
        )
        start = (
            f"if [ -f {REPL_PID} ]; then kill -9 $(cat {REPL_PID}) 2>/dev/null; fi; "
            f"rm -f {REPL_SOCKET}; cd /home/boxd && "
            f"setsid nohup {self.python} -u {REPL_PATH} serve {REPL_SOCKET} "
            f"> {SANDBOX_DIR}/repl.log 2>&1 < /dev/null & echo $! > {REPL_PID}; "
            f"for i in $(seq 100); do [ -S {REPL_SOCKET} ] && exit 0; sleep 0.1; done; "
            "exit 1"
        )
        result = await self.aexec(client, ["bash", "-c", start], timeout=60)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Starting the Python interpreter in the boxd machine failed: {result.stderr.strip()}"
            )
        self._repl_started = True

    async def arun_cell(self, client, code: str, *, timeout: float) -> BoxdCellResult:
        """Run ``code`` in the persistent interpreter.

        Variables defined by earlier cells stay available. If the interpreter
        is gone (it crashed, the machine was stopped, or the cell hung past
        its timeout), it is restarted and the result says that the earlier
        state was lost. The same happens in a new machine if this one was
        deleted since the last call. A cell is never sent twice.
        """
        _check_timeout(timeout)
        if self.machine_id is None:
            return await self._arun_cell_once(client, code, timeout=timeout)
        try:
            return await self._arun_cell_once(client, code, timeout=timeout)
        except self._errors["NotFoundError"]:
            if not await self._aforget_if_gone(client):
                raise
            await self.astart(client)
            await self._astart_repl(client)
            return _restarted_result(
                "RuntimeError",
                "The sandbox machine no longer exists, so a new one was created",
            )

    async def _arun_cell_once(
        self, client, code: str, *, timeout: float
    ) -> BoxdCellResult:
        await self.astart(client)
        if not self._repl_started:
            await self._astart_repl(client)

        prefix = [self.python, REPL_PATH, "call", REPL_SOCKET, str(timeout)]
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        command = [*prefix, encoded]
        if not self._fits_inline(command):
            # The interpreter's client removes this file after reading it.
            command = [*prefix, "@" + await self._aupload(client, ".b64", encoded)]
        try:
            result = await self.aexec(
                client, command, timeout=timeout + _EXEC_GRACE_SECONDS
            )
        except self._errors["BoxdError"] as exc:
            if not str(exc).startswith(_EXEC_TIMEOUT_PREFIX):
                raise
            await self._astart_repl(client)
            return _restarted_result(
                "TimeoutError", f"Execution timed out after {timeout}s"
            )

        if self._repl_started and result.exit_code == 0:
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                error = payload.get("error")
                return BoxdCellResult(
                    stdout=payload.get("stdout", ""),
                    stderr=payload.get("stderr", ""),
                    error=BoxdCellError(**error) if error else None,
                )

        logger.warning(
            "Python interpreter in boxd machine %s stopped (exit code %s): %s",
            self.machine_id,
            result.exit_code,
            result.stderr.strip(),
        )
        await self._astart_repl(client)
        return _restarted_result(
            "RuntimeError", "The Python interpreter stopped unexpectedly"
        )

    async def adelete(self, client) -> None:
        """Delete the machine. The id is kept until the delete succeeds."""
        machine_id = self.machine_id
        self._repl_started = False
        if machine_id is None:
            return
        for attempt in range(_DELETE_ATTEMPTS):
            try:
                await client.machines.delete(machine_id, region=self.region)
                logger.debug("Deleted boxd machine %s", machine_id)
                break
            except self._errors["NotFoundError"]:
                logger.debug("boxd machine %s already deleted", machine_id)
                break
            except self._errors["BoxdError"]:
                if attempt == _DELETE_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(1.0 + attempt)
        self.machine_id = None


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8", "surrogatepass"))


def _check_timeout(timeout: float) -> None:
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")


def _restarted_result(name: str, value: str) -> BoxdCellResult:
    message = f"{value}. The interpreter was restarted; earlier variables are gone."
    return BoxdCellResult(
        error=BoxdCellError(
            name=name,
            value=message,
            traceback=f"Traceback (most recent call last):\n{name}: {message}\n",
        )
    )

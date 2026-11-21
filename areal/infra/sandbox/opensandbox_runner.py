# SPDX-License-Identifier: Apache-2.0

"""Synchronous OpenSandbox runner for non-async user code.

Uses the SDK's blocking ``SandboxSync`` API directly, so no background event
loop is needed. Every sandbox is created with a server-side expiration
(``ttl_seconds``, one hour by default): if this process dies before cleanup,
the platform deletes the sandbox instead of leaking it.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from areal.utils import logging

from ._opensandbox_session import (
    DEFAULT_TTL_SECONDS,
    PYTHON_BIN,
    SANDBOX_DIR,
    TIMEOUT_EXIT_CODE,
    _EXEC_GRACE_SECONDS,
    _check_timeout,
    _load_opensandbox_sdk,
    _to_exec_result,
    build_create_kwargs,
    is_terminal_state,
    timedelta_of,
)

logger = logging.getLogger("OpenSandboxRunner")


@dataclass
class OpenSandboxRunResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: str | None = None


class OpenSandboxRunner:
    def __init__(
        self,
        *,
        image: str | None = None,
        snapshot_id: str | None = None,
        cpu: str | None = None,
        memory: str | None = None,
        env_vars: dict[str, str] | None = None,
        metadata: dict[str, str] | None = None,
        network_block_all: bool = False,
        ttl_seconds: int | None = DEFAULT_TTL_SECONDS,
        api_key: str | None = None,
        domain: str | None = None,
        create_timeout: float = 120.0,
        default_timeout: int = 30,
    ):
        self._sdk = _load_opensandbox_sdk()
        if image is None and snapshot_id is None:
            image = "python:3.11"
        if image is not None and snapshot_id is not None:
            raise ValueError("Pass either image or snapshot_id, not both")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive or None, got {ttl_seconds}")

        self.image = image
        self.snapshot_id = snapshot_id
        self.cpu = cpu
        self.memory = memory
        self.env_vars = dict(env_vars or {})
        self.metadata = dict(metadata or {})
        self.network_block_all = network_block_all
        self.ttl_seconds = ttl_seconds
        self.api_key = api_key
        self.domain = domain
        self.create_timeout = create_timeout
        self.default_timeout = default_timeout
        self.sandbox_id: str | None = None
        self._sandbox = None
        self._closed = False

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

    def _start(self):
        if self._sandbox is not None:
            return self._sandbox
        sandbox = self._sdk["SandboxSync"].create(**self._build_create_kwargs())
        self._sandbox = sandbox
        self.sandbox_id = sandbox.id
        logger.debug("Created OpenSandbox sandbox %s", sandbox.id)
        return sandbox

    def start(self) -> OpenSandboxRunner:
        try:
            self._start()
            return self
        except BaseException:
            self.close()
            raise

    def run(
        self,
        code: str,
        *,
        timeout: int | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> OpenSandboxRunResult:
        timeout = timeout if timeout is not None else self.default_timeout
        _check_timeout(timeout)
        try:
            result = self._run_once(code, timeout=timeout)
        except self._sdk["SandboxException"] as exc:
            if not self._is_gone(exc):
                raise
            logger.warning(
                "OpenSandbox sandbox %s is gone; creating a new one",
                self.sandbox_id,
            )
            result = self._run_once(code, timeout=timeout)

        if on_stdout is not None and result.stdout:
            on_stdout(result.stdout)
        if on_stderr is not None and (result.stderr or result.error):
            on_stderr(result.stderr or result.error)
        self._renew_lease()
        return result

    def _run_once(self, code: str, *, timeout: float) -> OpenSandboxRunResult:
        sandbox = self._start()
        script_path = f"{SANDBOX_DIR}/code-{uuid.uuid4().hex}.py"
        sandbox.files.write_file(script_path, code)
        try:
            command = [
                "timeout",
                "-k",
                "2",
                str(timeout),
                PYTHON_BIN,
                script_path,
            ]
            execution = sandbox.commands.run(
                command,
                opts=self._sdk["RunCommandOpts"](
                    timeout=timedelta_of(timeout + _EXEC_GRACE_SECONDS)
                ),
            )
        finally:
            self._remove(script_path)
        result = _to_run_result(execution)
        return result

    def _remove(self, path: str) -> None:
        if self._sandbox is None:
            return
        try:
            self._sandbox.files.delete_files([path])
        except Exception as exc:
            logger.debug("Could not remove %s: %s", path, exc)

    def _is_gone(self, exc: Exception) -> bool:
        """Probe the sandbox after a failed exec.

        The failed exec never ran code, so recreating and running once more
        is safe. Returns True when the sandbox should be recreated.
        """
        del exc
        sandbox = self._sandbox
        if sandbox is None:
            return False
        try:
            info = sandbox.get_info()
        except self._sdk["SandboxException"]:
            self._forget()
            return True
        if is_terminal_state(getattr(getattr(info, "status", None), "state", "")):
            self._forget()
            return True
        return False

    def _forget(self) -> None:
        self._sandbox = None
        self.sandbox_id = None

    def _renew_lease(self) -> None:
        """Best-effort lease renewal; expiry still reaps a lost sandbox."""
        if self.ttl_seconds is None or self._sandbox is None:
            return
        try:
            self._sandbox.renew(timeout=timedelta_of(self.ttl_seconds))
        except Exception as exc:
            logger.debug(
                "Could not renew the OpenSandbox lease for %s: %s",
                self.sandbox_id,
                exc,
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        sandbox = self._sandbox
        self._sandbox = None
        if sandbox is None:
            return
        try:
            sandbox.destroy()
            logger.debug("Deleted OpenSandbox sandbox %s", self.sandbox_id)
        except self._sdk["SandboxException"] as exc:
            logger.debug(
                "OpenSandbox sandbox %s delete failed: %s", self.sandbox_id, exc
            )

    def __enter__(self) -> OpenSandboxRunner:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()


def _to_run_result(execution) -> OpenSandboxRunResult:
    exec_result = _to_exec_result(execution)
    error = exec_result.error
    if exec_result.exit_code == TIMEOUT_EXIT_CODE and not error:
        error = (
            f"Execution timed out after the configured limit "
            f"(exit code {TIMEOUT_EXIT_CODE})"
        )
    elif exec_result.exit_code != 0 and not error:
        error = f"Process exited with code {exec_result.exit_code}"
    return OpenSandboxRunResult(
        stdout=exec_result.stdout,
        stderr=exec_result.stderr,
        exit_code=exec_result.exit_code,
        error=error,
    )


__all__ = ["OpenSandboxRunResult", "OpenSandboxRunner"]

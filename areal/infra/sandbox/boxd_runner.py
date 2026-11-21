# SPDX-License-Identifier: Apache-2.0

"""Synchronous boxd sandbox runner for non-async user code."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from areal.utils import logging

from ._boxd_client import BoxdClientManager
from ._boxd_client import _load_boxd_sdk as _load_boxd_client_sdk
from ._boxd_session import (
    _EXEC_TIMEOUT_PREFIX,
    DEFAULT_DELETE_AFTER,
    TIMEOUT_EXIT_CODE,
    BoxdSession,
    _load_boxd_errors,
)

logger = logging.getLogger("BoxdRunner")


def _last_non_empty_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


@dataclass
class BoxdRunResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: str | None = None


class BoxdRunner:
    """Run Python code in one isolated boxd machine.

    Each runner creates one machine on first use and keeps it warm across
    ``run()`` calls. Every ``run()`` starts a fresh Python process, so nothing
    carries over between calls. ``close()`` deletes the machine. Machines carry
    the label ``areal-sandbox=true``. ``delete_after`` (seconds after creation,
    six hours by default) has boxd delete the machine even if ``close()`` never
    runs; a runner that outlives it creates a new machine on the next
    ``run()``. ``delete_after=None`` turns the timer off. A machine that does
    not become ready, or any other boxd error, comes back as a result with
    ``error`` set. A non-positive ``timeout`` raises ``ValueError``.
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
        default_timeout: int = 30,
    ):
        _load_boxd_client_sdk()
        self._errors = _load_boxd_errors()
        self._session = BoxdSession(
            name_prefix=name_prefix,
            org=org,
            region=region,
            snapshot=snapshot,
            vcpu=vcpu,
            memory=memory,
            env_vars=env_vars,
            labels=labels,
            python_packages=python_packages,
            delete_after=delete_after,
            create_timeout=create_timeout,
        )
        self.default_timeout = default_timeout
        self._client = None
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()
        self._start_lock: asyncio.Lock | None = None

    @property
    def machine_id(self) -> str | None:
        return self._session.machine_id

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._closed:
            raise RuntimeError("BoxdRunner is closed")

        with self._loop_lock:
            if self._loop is not None:
                return self._loop

            loop = asyncio.new_event_loop()
            ready = threading.Event()

            def run_loop() -> None:
                asyncio.set_event_loop(loop)
                ready.set()
                loop.run_forever()
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

            self._loop = loop
            self._thread = threading.Thread(
                target=run_loop,
                name="BoxdRunnerLoop",
                daemon=True,
            )
            self._thread.start()
            ready.wait()
            return loop

    def _run(self, coro):
        try:
            loop = self._ensure_loop()
        except BaseException:
            coro.close()
            raise

        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result()
        except BaseException:
            future.cancel()
            raise

    async def _aget_client(self):
        if self._client is None:
            async_boxd_cls = _load_boxd_client_sdk()
            self._client = async_boxd_cls(**BoxdClientManager.config_overrides())
            logger.debug("Initialized BoxdRunner AsyncBoxd client")
        return self._client

    async def _astart(self) -> str:
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            return await self._session.astart(await self._aget_client())

    def start(self) -> BoxdRunner:
        try:
            self._run(self._astart())
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
    ) -> BoxdRunResult:
        return self._run(
            self._arun(code, timeout=timeout, on_stdout=on_stdout, on_stderr=on_stderr)
        )

    async def _arun(
        self,
        code: str,
        *,
        timeout: int | None = None,
        on_stdout: Callable[[str], None] | None = None,
        on_stderr: Callable[[str], None] | None = None,
    ) -> BoxdRunResult:
        timeout = timeout if timeout is not None else self.default_timeout

        try:
            await self._astart()
            response = await self._session.arun_script(
                await self._aget_client(), code, timeout=timeout
            )
        except (self._errors["BoxdError"], RuntimeError, TimeoutError) as exc:
            # TimeoutError: the machine did not become ready in create_timeout.
            error = str(exc) or type(exc).__name__
            exit_code = (
                TIMEOUT_EXIT_CODE if error.startswith(_EXEC_TIMEOUT_PREFIX) else 1
            )
            if on_stderr is not None:
                on_stderr(error)
            return BoxdRunResult(stderr=error, exit_code=exit_code, error=error)

        stdout = response.stdout or ""
        stderr = response.stderr or ""
        if on_stdout is not None and stdout:
            on_stdout(stdout)
        if on_stderr is not None and stderr:
            on_stderr(stderr)

        if response.exit_code == 0:
            error = None
        elif response.exit_code == TIMEOUT_EXIT_CODE:
            error = f"Execution timed out after {timeout}s"
        else:
            error = (
                _last_non_empty_line(stderr)
                or f"Process exited with code {response.exit_code}"
            )
        return BoxdRunResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=response.exit_code,
            error=error,
        )

    def close(self) -> None:
        if self._loop is None:
            self._closed = True
            return

        try:
            self._run(self._aclose())
        finally:
            self._closed = True
            self._stop_loop()

    async def _aclose(self) -> None:
        try:
            if self._session.machine_id is not None:
                await self._session.adelete(await self._aget_client())
        finally:
            if self._client is not None:
                client = self._client
                self._client = None
                await client.close()

    def _stop_loop(self) -> None:
        loop = self._loop
        thread = self._thread
        self._loop = None
        self._thread = None

        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            thread.join()

    def __enter__(self) -> BoxdRunner:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

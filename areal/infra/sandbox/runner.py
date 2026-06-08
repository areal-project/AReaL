# SPDX-License-Identifier: Apache-2.0

"""Synchronous Daytona sandbox runner for non-async user code."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from areal.utils import logging

from ._client import DaytonaClientManager
from ._client import _load_daytona_sdk as _load_daytona_client_sdk

if TYPE_CHECKING:
    from daytona import Chart, Image, Resources

logger = logging.getLogger("DaytonaRunner")

CreateSandboxFromImageParams = None
CreateSandboxFromSnapshotParams = None
DaytonaError = None
DaytonaNotFoundError = None
DaytonaTimeoutError = None


def _load_daytona_sdk() -> dict[str, Any]:
    global CreateSandboxFromImageParams
    global CreateSandboxFromSnapshotParams
    global DaytonaError
    global DaytonaNotFoundError
    global DaytonaTimeoutError

    if all(
        value is not None
        for value in (
            CreateSandboxFromImageParams,
            CreateSandboxFromSnapshotParams,
            DaytonaError,
            DaytonaNotFoundError,
            DaytonaTimeoutError,
        )
    ):
        return {
            "CreateSandboxFromImageParams": CreateSandboxFromImageParams,
            "CreateSandboxFromSnapshotParams": CreateSandboxFromSnapshotParams,
            "DaytonaError": DaytonaError,
            "DaytonaNotFoundError": DaytonaNotFoundError,
            "DaytonaTimeoutError": DaytonaTimeoutError,
        }

    try:
        from daytona import (
            CreateSandboxFromImageParams as ImportedCreateSandboxFromImageParams,
        )
        from daytona import (
            CreateSandboxFromSnapshotParams as ImportedCreateSandboxFromSnapshotParams,
        )
        from daytona import DaytonaError as ImportedDaytonaError
        from daytona import DaytonaNotFoundError as ImportedDaytonaNotFoundError
        from daytona import DaytonaTimeoutError as ImportedDaytonaTimeoutError
    except ImportError as exc:
        raise ImportError(
            "DaytonaRunner requires the optional 'daytona' dependency. Install it with `uv sync --extra sandbox`."
        ) from exc

    CreateSandboxFromImageParams = ImportedCreateSandboxFromImageParams
    CreateSandboxFromSnapshotParams = ImportedCreateSandboxFromSnapshotParams
    DaytonaError = ImportedDaytonaError
    DaytonaNotFoundError = ImportedDaytonaNotFoundError
    DaytonaTimeoutError = ImportedDaytonaTimeoutError

    return {
        "CreateSandboxFromImageParams": CreateSandboxFromImageParams,
        "CreateSandboxFromSnapshotParams": CreateSandboxFromSnapshotParams,
        "DaytonaError": DaytonaError,
        "DaytonaNotFoundError": DaytonaNotFoundError,
        "DaytonaTimeoutError": DaytonaTimeoutError,
    }


def _last_non_empty_line(text: str) -> str | None:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


@dataclass
class DaytonaRunResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    charts: list[Chart] = field(default_factory=list)
    error: str | None = None


class DaytonaRunner:
    def __init__(
        self,
        *,
        snapshot: str | None = None,
        image: str | Image | None = None,
        resources: Resources | None = None,
        env_vars: dict[str, str] | None = None,
        network_block_all: bool = False,
        labels: dict[str, str] | None = None,
        create_timeout: float = 60.0,
        default_timeout: int = 30,
        language: str = "python",
        ephemeral: bool = True,
    ):
        self._sdk = _load_daytona_sdk()
        if image is not None and snapshot is not None:
            raise ValueError("Pass either snapshot or image, not both")
        if resources is not None and image is None:
            raise ValueError(
                "Daytona resources require image-based sandbox creation in SDK 0.167.0"
            )

        self.snapshot = snapshot
        self.image = image
        self.resources = resources
        self.env_vars = dict(env_vars or {})
        self.network_block_all = network_block_all
        self.labels = dict(labels or {})
        self.create_timeout = create_timeout
        self.default_timeout = default_timeout
        self.language = language
        self.ephemeral = ephemeral
        self._sandbox_id: str | None = None
        self._client = None
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_lock = threading.Lock()
        self._sandbox_lock: asyncio.Lock | None = None

    def _build_create_params(self):
        if self.image is not None:
            return self._sdk["CreateSandboxFromImageParams"](
                image=self.image,
                resources=self.resources,
                language=self.language,
                env_vars=self.env_vars or None,
                network_block_all=self.network_block_all,
                labels=self.labels or None,
                ephemeral=self.ephemeral,
            )

        return self._sdk["CreateSandboxFromSnapshotParams"](
            snapshot=self.snapshot,
            language=self.language,
            env_vars=self.env_vars or None,
            network_block_all=self.network_block_all,
            labels=self.labels or None,
            ephemeral=self.ephemeral,
        )

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._closed:
            raise RuntimeError("DaytonaRunner is closed")

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
                name="DaytonaRunnerLoop",
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
            async_daytona_cls, daytona_config_cls = _load_daytona_client_sdk()
            config = daytona_config_cls(
                connection_pool_maxsize=None,
                **DaytonaClientManager.config_overrides(),
            )
            self._client = async_daytona_cls(config)
            logger.debug("Initialized DaytonaRunner AsyncDaytona client")
        return self._client

    async def _acreate(self):
        client = await self._aget_client()
        sandbox = await client.create(
            self._build_create_params(),
            timeout=self.create_timeout,
        )
        self._sandbox_id = sandbox.id
        logger.debug("Created Daytona sandbox %s", self._sandbox_id)
        return sandbox

    async def _aget_sandbox(self):
        if self._sandbox_id is None:
            if self._sandbox_lock is None:
                self._sandbox_lock = asyncio.Lock()
            async with self._sandbox_lock:
                if self._sandbox_id is None:
                    return await self._acreate()

        client = await self._aget_client()
        return await client.get(self._sandbox_id)

    def start(self) -> DaytonaRunner:
        try:
            self._run(self._aget_sandbox())
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
    ) -> DaytonaRunResult:
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
    ) -> DaytonaRunResult:
        timeout = timeout if timeout is not None else self.default_timeout

        try:
            sandbox = await self._aget_sandbox()
            response = await sandbox.process.code_run(code, timeout=timeout)
        except self._sdk["DaytonaTimeoutError"] as exc:
            error = str(exc)
            if on_stderr is not None:
                on_stderr(error)
            return DaytonaRunResult(stderr=error, exit_code=124, charts=[], error=error)
        except self._sdk["DaytonaError"] as exc:
            error = str(exc)
            if on_stderr is not None:
                on_stderr(error)
            return DaytonaRunResult(stderr=error, exit_code=1, charts=[], error=error)
        except Exception as exc:
            status_code = getattr(exc, "status", None)
            error_text = str(exc)
            if status_code == 408 or "timeout" in error_text.lower():
                if on_stderr is not None:
                    on_stderr(error_text)
                error = _last_non_empty_line(error_text) or error_text
                return DaytonaRunResult(
                    stderr=error_text,
                    exit_code=124,
                    charts=[],
                    error=error,
                )
            raise

        output = response.result or ""
        charts = (
            response.artifacts.charts
            if response.artifacts and response.artifacts.charts
            else []
        )

        if response.exit_code == 0:
            if on_stdout is not None and output:
                on_stdout(output)
            return DaytonaRunResult(
                stdout=output,
                stderr="",
                exit_code=response.exit_code,
                charts=charts,
                error=None,
            )

        error = (
            _last_non_empty_line(output)
            or f"Process exited with code {response.exit_code}"
        )
        if on_stderr is not None and output:
            on_stderr(output)
        return DaytonaRunResult(
            stdout="",
            stderr=output,
            exit_code=response.exit_code,
            charts=charts,
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
        sandbox_id = self._sandbox_id
        self._sandbox_id = None

        try:
            if sandbox_id is not None:
                client = await self._aget_client()
                sandbox = await client.get(sandbox_id)
                await sandbox.delete(timeout=self.create_timeout)
        except self._sdk["DaytonaNotFoundError"]:
            logger.debug("Daytona sandbox already deleted")
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

    def __enter__(self) -> DaytonaRunner:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

"""Daytona-backed Python tool for stateful TIR code execution."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from areal.infra.sandbox import DaytonaClientManager
from areal.utils import logging

from .base import BaseTool, ToolCallStatus, ToolDescription, ToolMarkers, ToolType
from .python_tool import extract_python_code

if TYPE_CHECKING:
    from daytona import Image, Resources

logger = logging.getLogger("DaytonaPythonTool")

CreateSandboxFromImageParams = None
CreateSandboxFromSnapshotParams = None
DaytonaError = None
DaytonaNotFoundError = None


def _load_daytona_sdk() -> dict[str, Any]:
    global CreateSandboxFromImageParams
    global CreateSandboxFromSnapshotParams
    global DaytonaError
    global DaytonaNotFoundError

    if all(
        value is not None
        for value in (
            CreateSandboxFromImageParams,
            CreateSandboxFromSnapshotParams,
            DaytonaError,
            DaytonaNotFoundError,
        )
    ):
        return {
            "CreateSandboxFromImageParams": CreateSandboxFromImageParams,
            "CreateSandboxFromSnapshotParams": CreateSandboxFromSnapshotParams,
            "DaytonaError": DaytonaError,
            "DaytonaNotFoundError": DaytonaNotFoundError,
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
    except ImportError as exc:
        raise ImportError(
            "DaytonaPythonTool requires the optional 'daytona' dependency. Install it with `uv sync --extra sandbox`."
        ) from exc

    CreateSandboxFromImageParams = ImportedCreateSandboxFromImageParams
    CreateSandboxFromSnapshotParams = ImportedCreateSandboxFromSnapshotParams
    DaytonaError = ImportedDaytonaError
    DaytonaNotFoundError = ImportedDaytonaNotFoundError
    return {
        "CreateSandboxFromImageParams": CreateSandboxFromImageParams,
        "CreateSandboxFromSnapshotParams": CreateSandboxFromSnapshotParams,
        "DaytonaError": DaytonaError,
        "DaytonaNotFoundError": DaytonaNotFoundError,
    }


class DaytonaPythonTool(BaseTool):
    def __init__(
        self,
        timeout: int = 30,
        debug_mode: bool = False,
        *,
        snapshot: str | None = None,
        image: str | Image | None = None,
        resources: Resources | None = None,
        env_vars: dict[str, str] | None = None,
        network_block_all: bool = False,
        labels: dict[str, str] | None = None,
        create_timeout: float = 60.0,
    ):
        super().__init__(timeout, debug_mode)
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
        self._sandbox_id: str | None = None
        self._sandbox_lock = asyncio.Lock()
        self._sdk: dict[str, Any] | None = None

    @property
    def tool_type(self) -> ToolType:
        return ToolType.DAYTONA_PYTHON

    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="daytona_python_executor",
            description="Execute Python code in a Daytona cloud sandbox with state preserved across calls.",
            parameters={"code": "The Python code string to execute"},
            parameter_prompt="Please provide the Python code to execute inside the Daytona sandbox.",
            example="```python\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n```\n or \n<python>\na=1\nb=1\nprint(f'The a+b result is {a+b}')\n</python>",
        )

    @property
    def markers(self) -> ToolMarkers:
        return ToolMarkers(
            start_markers=["```python", "<python>"],
            end_markers=["```", "</python>"],
        )

    def parse_parameters(self, text: str) -> dict[str, Any]:
        return {"code": extract_python_code(text)}

    def execute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        raise NotImplementedError("DaytonaPythonTool is async-only; use aexecute().")

    def _build_create_params(self):
        assert self._sdk is not None

        if self.image is not None:
            return self._sdk["CreateSandboxFromImageParams"](
                image=self.image,
                resources=self.resources,
                language="python",
                env_vars=self.env_vars or None,
                network_block_all=self.network_block_all,
                labels=self.labels or None,
                ephemeral=True,
            )

        return self._sdk["CreateSandboxFromSnapshotParams"](
            snapshot=self.snapshot,
            language="python",
            env_vars=self.env_vars or None,
            network_block_all=self.network_block_all,
            labels=self.labels or None,
            ephemeral=True,
        )

    async def _aget_sandbox(self):
        self._sdk = _load_daytona_sdk()
        client = await DaytonaClientManager.get_client()

        if self._sandbox_id is not None:
            return await client.get(self._sandbox_id)

        async with self._sandbox_lock:
            if self._sandbox_id is not None:
                return await client.get(self._sandbox_id)

            sandbox = await client.create(
                self._build_create_params(),
                timeout=self.create_timeout,
            )
            self._sandbox_id = sandbox.id
            logger.debug("Created Daytona sandbox %s", self._sandbox_id)
            return sandbox

    async def aexecute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        code = parameters.get("code", "")
        if not code:
            return "Error: No code provided", ToolCallStatus.ERROR

        if self.debug_mode:
            logger.debug("[FAKE] Executing Daytona Python code: %s", code[:100])
            return "dummy python output", ToolCallStatus.SUCCESS

        try:
            sandbox = await self._aget_sandbox()
            result = await sandbox.code_interpreter.run_code(code, timeout=self.timeout)
        except Exception as exc:
            daytona_error = _load_daytona_sdk()["DaytonaError"]
            if isinstance(exc, daytona_error):
                return str(exc), ToolCallStatus.ERROR
            raise

        if result.error is not None:
            error_lines = []
            if result.stdout.strip():
                error_lines.append(result.stdout.strip())
            if result.stderr.strip():
                error_lines.append(result.stderr.strip())
            if result.error.traceback.strip():
                traceback_text = result.error.traceback.strip()
                if not traceback_text.startswith("Traceback"):
                    error_lines.append("Traceback (most recent call last):")
                error_lines.append(traceback_text)
            else:
                error_lines.append("Traceback (most recent call last):")
                error_lines.append(f"{result.error.name}: {result.error.value}")

            return "\n".join(error_lines), ToolCallStatus.ERROR

        output = "\n".join(
            part.rstrip()
            for part in (result.stdout, result.stderr)
            if part and part.rstrip()
        )
        return output, ToolCallStatus.SUCCESS

    async def _aclose(self) -> None:
        if self._sandbox_id is None:
            return

        sandbox_id = self._sandbox_id
        self._sandbox_id = None

        try:
            self._sdk = _load_daytona_sdk()
            client = await DaytonaClientManager.get_client()
            sandbox = await client.get(sandbox_id)
            await sandbox.delete(timeout=self.create_timeout)
        except self._sdk["DaytonaNotFoundError"]:
            logger.debug("Daytona sandbox already deleted")
        except Exception as exc:
            logger.warning("DaytonaPythonTool cleanup error: %s", exc)

    async def aclose(self) -> None:
        await self._aclose()

    def close(self) -> None:
        if self._sandbox_id is None:
            return

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = DaytonaClientManager.active_loop()
            if loop is not None:
                asyncio.run_coroutine_threadsafe(self._aclose(), loop).result()
                return

            try:
                asyncio.run(self._aclose())
            finally:
                asyncio.run(DaytonaClientManager.close())
            return

        logger.warning(
            "DaytonaPythonTool.close() was called from an active event loop; use asyncio.to_thread(tool.close)."
        )

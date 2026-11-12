"""boxd-backed Python tool for stateful TIR code execution."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from areal.infra.sandbox import BoxdClientManager
from areal.infra.sandbox._boxd_session import BoxdSession, _load_boxd_errors
from areal.utils import logging

from .base import BaseTool, ToolCallStatus, ToolDescription, ToolMarkers, ToolType
from .python_tool import extract_python_code

logger = logging.getLogger("BoxdPythonTool")


class BoxdPythonTool(BaseTool):
    """Run model-generated Python in an isolated boxd machine.

    One machine per tool instance (one per trajectory). A persistent interpreter
    runs inside the machine, so variables survive across tool calls.
    """

    def __init__(
        self,
        timeout: int = 30,
        debug_mode: bool = False,
        *,
        name_prefix: str = "areal-tir",
        org: str | None = None,
        region: str | None = None,
        snapshot: str | None = None,
        vcpu: int | None = None,
        memory: int | str | None = None,
        env_vars: Mapping[str, str] | None = None,
        labels: Mapping[str, str] | None = None,
        python_packages: Sequence[str] | None = None,
        delete_after: int | None = None,
        create_timeout: float = 120.0,
    ):
        super().__init__(timeout, debug_mode)
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
        self._lock = asyncio.Lock()

    @property
    def machine_id(self) -> str | None:
        return self._session.machine_id

    @property
    def tool_type(self) -> ToolType:
        return ToolType.BOXD_PYTHON

    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="boxd_python_executor",
            description="Execute Python code in a boxd sandbox machine with state preserved across calls.",
            parameters={"code": "The Python code string to execute"},
            parameter_prompt="Please provide the Python code to execute inside the boxd sandbox.",
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
        raise NotImplementedError("BoxdPythonTool is async-only; use aexecute().")

    async def aexecute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        code = parameters.get("code", "")
        if not code:
            return "Error: No code provided", ToolCallStatus.ERROR

        if self.debug_mode:
            logger.debug("[FAKE] Executing boxd Python code: %s", code[:100])
            return "dummy python output", ToolCallStatus.SUCCESS

        try:
            client = await BoxdClientManager.get_client()
            async with self._lock:
                result = await self._session.arun_cell(
                    client, code, timeout=self.timeout
                )
        except (self._errors["BoxdError"], RuntimeError) as exc:
            return str(exc), ToolCallStatus.ERROR

        if result.error is not None:
            error_lines = []
            if result.stdout.strip():
                error_lines.append(result.stdout.strip())
            if result.stderr.strip():
                error_lines.append(result.stderr.strip())
            traceback_text = result.error.traceback.strip()
            if traceback_text:
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
        if self._session.machine_id is None:
            return

        try:
            client = await BoxdClientManager.get_client()
            await self._session.adelete(client)
        except Exception as exc:
            logger.warning("BoxdPythonTool cleanup error: %s", exc)

    async def aclose(self) -> None:
        await self._aclose()

    def close(self) -> None:
        if self._session.machine_id is None:
            return

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = BoxdClientManager.active_loop()
            if loop is not None:
                asyncio.run_coroutine_threadsafe(self._aclose(), loop).result()
                return

            try:
                asyncio.run(self._aclose())
            finally:
                asyncio.run(BoxdClientManager.close())
            return

        logger.warning(
            "BoxdPythonTool.close() was called from an active event loop; use asyncio.to_thread(tool.close)."
        )

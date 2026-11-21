"""OpenSandbox-backed Python tool for stateful TIR code execution."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from areal.infra.sandbox._opensandbox_session import (
    DEFAULT_TTL_SECONDS,
    OpenSandboxSession,
    _load_opensandbox_sdk,
)
from areal.utils import logging

from .base import BaseTool, ToolCallStatus, ToolDescription, ToolMarkers, ToolType
from .python_tool import extract_python_code

logger = logging.getLogger("OpenSandboxPythonTool")


class OpenSandboxPythonTool(BaseTool):
    """Run model-generated Python in an isolated OpenSandbox sandbox.

    One sandbox per tool instance (one per trajectory). A persistent
    interpreter runs inside the sandbox, so variables survive across tool
    calls.

    ``ttl_seconds`` (one hour by default) is the sandbox's server-side
    expiration: if cleanup never runs (for example the training process was
    killed), the platform deletes the sandbox at expiry instead of leaking
    it. ``None`` turns the TTL off and requires explicit cleanup.
    """

    def __init__(
        self,
        timeout: int = 30,
        debug_mode: bool = False,
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
    ):
        super().__init__(timeout, debug_mode)
        self._errors = _load_opensandbox_sdk()
        self._session = OpenSandboxSession(
            image=image,
            snapshot_id=snapshot_id,
            cpu=cpu,
            memory=memory,
            env_vars=env_vars,
            metadata=metadata,
            network_block_all=network_block_all,
            ttl_seconds=ttl_seconds,
            api_key=api_key,
            domain=domain,
            create_timeout=create_timeout,
            default_timeout=timeout,
        )

    @property
    def sandbox_id(self) -> str | None:
        return self._session.sandbox_id

    @property
    def tool_type(self) -> ToolType:
        return ToolType.OPENSANDBOX_PYTHON

    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="opensandbox_python_executor",
            description="Execute Python code in an OpenSandbox sandbox with state preserved across calls.",
            parameters={"code": "The Python code string to execute"},
            parameter_prompt="Please provide the Python code to execute inside the OpenSandbox sandbox.",
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
        raise NotImplementedError(
            "OpenSandboxPythonTool is async-only; use aexecute()."
        )

    async def aexecute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        code = parameters.get("code", "")
        if not code:
            return "Error: No code provided", ToolCallStatus.ERROR

        if self.debug_mode:
            logger.debug("[FAKE] Executing OpenSandbox Python code: %s", code[:100])
            return "dummy python output", ToolCallStatus.SUCCESS

        try:
            result = await self._session.arun_cell(code, timeout=self.timeout)
        except (self._errors["SandboxException"], RuntimeError, TimeoutError) as exc:
            # TimeoutError: the sandbox did not become ready in create_timeout.
            return str(exc) or type(exc).__name__, ToolCallStatus.ERROR

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

    async def aclose(self) -> None:
        try:
            await self._session.adelete()
        except Exception as exc:
            logger.warning("OpenSandboxPythonTool cleanup error: %s", exc)

    def close(self) -> None:
        if self._session.sandbox_id is None:
            return

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(self._session.adelete())
            except Exception as exc:
                logger.warning("OpenSandboxPythonTool cleanup error: %s", exc)
            return

        logger.warning(
            "OpenSandboxPythonTool.close() was called from an active event loop; "
            "use asyncio.to_thread(tool.close) or await aclose()."
        )

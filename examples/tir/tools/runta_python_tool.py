"""Runta-backed Python tool for sandboxed TIR code execution."""

from __future__ import annotations

from typing import Any

from areal.infra.sandbox import AsyncRuntaRunner
from areal.utils import logging

from .base import BaseTool, ToolCallStatus, ToolDescription, ToolMarkers, ToolType
from .python_tool import extract_python_code

logger = logging.getLogger("RuntaPythonTool")


class RuntaPythonTool(BaseTool):
    """Run each code block in one Runta runtime owned by this tool instance.

    The runtime is created on the first call and deleted by `aclose()`. Every
    call starts a fresh interpreter, so variables do not carry over; files
    written inside the sandbox do.

    Without `image` or `checkpoint_id` the sandbox has only the Python
    standard library: `import numpy` fails with `ModuleNotFoundError`, and
    nothing is installed for you. If the model's code needs packages, pass
    `image` (a Docker image that already contains them) or `checkpoint_id`
    (a Runta checkpoint of a runtime prepared with them installed).
    """

    def __init__(
        self,
        timeout: int = 30,
        debug_mode: bool = False,
        *,
        image: str | None = None,
        checkpoint_id: str | None = None,
        vcpus: int = 1,
        memory_mib: int = 1024,
        env_vars: dict[str, str] | None = None,
        network_block_all: bool = False,
        create_timeout: float = 240.0,
    ):
        super().__init__(timeout, debug_mode)
        self._runner = AsyncRuntaRunner(
            image=image,
            checkpoint_id=checkpoint_id,
            vcpus=vcpus,
            memory_mib=memory_mib,
            env_vars=env_vars,
            network_block_all=network_block_all,
            create_timeout=create_timeout,
            default_timeout=timeout,
        )
        self._started = False

    @property
    def tool_type(self) -> ToolType:
        return ToolType.RUNTA_PYTHON

    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="runta_python_executor",
            description="Execute Python code in a Runta cloud sandbox. Each call runs a fresh interpreter; files persist across calls. Only the Python standard library is guaranteed: third-party packages such as numpy are missing unless the sandbox was set up with an image or checkpoint that provides them.",
            parameters={"code": "The Python code string to execute"},
            parameter_prompt="Please provide the Python code to execute inside the Runta sandbox.",
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
        raise NotImplementedError("RuntaPythonTool is async-only; use aexecute().")

    async def aexecute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        code = parameters.get("code", "")
        if not code:
            return "Error: No code provided", ToolCallStatus.ERROR

        if self.debug_mode:
            logger.debug("[FAKE] Executing Runta Python code: %s", code[:100])
            return "dummy python output", ToolCallStatus.SUCCESS

        self._started = True
        result = await self._runner.arun(code, timeout=self.timeout)

        output = "\n".join(
            part.rstrip()
            for part in (result.stdout, result.stderr)
            if part and part.rstrip()
        )
        if result.exit_code != 0:
            return output or result.error or "", ToolCallStatus.ERROR
        return output, ToolCallStatus.SUCCESS

    async def aclose(self) -> None:
        if not self._started:
            return
        self._started = False
        await self._runner.aclose()

    def close(self) -> None:
        if not self._started:
            return
        # The client is bound to the event loop that created the runtime, so a
        # synchronous close cannot delete it; the runtime's server-side lifetime
        # reclaims it if aclose() is never awaited.
        logger.warning(
            "RuntaPythonTool.close() cannot delete the runtime; use aclose()."
        )

"""Tool registry, routing, and execution helpers for the TIR example."""

from __future__ import annotations

import importlib.util
import re

from areal.utils import logging

from tools import (  # isort: skip
    BaseTool,
    CalculatorTool,
    PythonTool,
    ToolCallStatus,
    ToolType,
)

logger = logging.getLogger("Tool Manager")


def _build_daytona_python_tool(timeout: int, debug_mode: bool) -> BaseTool:
    if importlib.util.find_spec("daytona") is None:
        raise ImportError(
            "daytona_python requires the optional 'daytona' dependency. Install it with `uv sync --extra sandbox`."
        )

    from tools.daytona_python_tool import DaytonaPythonTool

    return DaytonaPythonTool(timeout, debug_mode)


class ToolRegistry:
    """Tool registry that manages all available tools."""

    TOOL_NAMES = {
        "python": ToolType.PYTHON,
        "calculator": ToolType.CALCULATOR,
        "daytona_python": ToolType.DAYTONA_PYTHON,
    }
    DEFAULT_TOOL_NAMES = ("python", "calculator")

    def __init__(
        self,
        timeout: int = 30,
        enabled_tools: str = "python;calculator",
        debug_mode: bool = False,
    ):
        self._tool_factories = {
            ToolType.PYTHON: lambda: PythonTool(timeout, debug_mode),
            ToolType.CALCULATOR: lambda: CalculatorTool(timeout, debug_mode),
            ToolType.DAYTONA_PYTHON: lambda: _build_daytona_python_tool(
                timeout, debug_mode
            ),
        }

        if enabled_tools is None:
            requested_tool_names = list(self.DEFAULT_TOOL_NAMES)
        else:
            requested_tool_names = enabled_tools.split(";")

        self.enabled_tools = []
        for tool_name in requested_tool_names:
            if tool_name in self.TOOL_NAMES:
                self.enabled_tools.append(self.TOOL_NAMES[tool_name])
            else:
                logger.warning(f"Unknown tool type: {tool_name}, skipping")

        if (
            ToolType.PYTHON in self.enabled_tools
            and ToolType.DAYTONA_PYTHON in self.enabled_tools
        ):
            raise ValueError(
                "python and daytona_python use the same markers; enable only one Python backend"
            )

        self.tools = {
            tool_type: self._tool_factories[tool_type]()
            for tool_type in self.enabled_tools
        }

        logger.info(
            f"ToolRegistry initialized with enabled tools: {[t.value for t in self.enabled_tools]}"
        )

    def get_tool(self, tool_type: ToolType) -> BaseTool | None:
        """Get tool instance."""
        return self.tools.get(tool_type)

    def get_all_tools(self) -> dict[ToolType, BaseTool]:
        """Get all tool instances."""
        return self.tools

    def get_tool_markers(self) -> dict[ToolType, tuple[list[str], list[str]]]:
        """Get marker information for enabled tools only.

        Returns:
            Dict[ToolType, Tuple[List[str], List[str]]]: Tool type ->
                (start markers list, end markers list)
        """
        return {
            tool_type: (tool.markers.start_markers, tool.markers.end_markers)
            for tool_type, tool in self.tools.items()
        }

    def get_all_start_markers(self) -> list[str]:
        """Get all start markers for enabled tools only.

        Returns:
            List[str]: List of all start markers.
        """
        start_markers = []
        for tool in self.tools.values():
            start_markers.extend(tool.markers.start_markers)
        return start_markers

    def get_all_end_markers(self) -> list[str]:
        """Get all end markers for enabled tools only.

        Returns:
            List[str]: List of all end markers.
        """
        end_markers = []
        for tool in self.tools.values():
            end_markers.extend(tool.markers.end_markers)
        return end_markers

    def get_all_markers(self) -> list[str]:
        """Get all markers (start and end) for enabled tools only.

        Returns:
            List[str]: List of all markers.
        """
        all_markers = []
        all_markers.extend(self.get_all_start_markers())
        all_markers.extend(self.get_all_end_markers())
        return all_markers

    def get_tool_descriptions_prompt(self) -> str:
        """Generate tool description prompt text for external calls."""
        prompt_parts = ["Tools List:\n"]
        for tool in self.tools.values():
            desc = tool.description
            prompt_parts.append(f"Tool Name: {desc.name}")
            prompt_parts.append(f"Description: {desc.description}")
            prompt_parts.append(f"Parameter Description: {desc.parameter_prompt}")
            prompt_parts.append(f"Usage Example: {desc.example}")
            prompt_parts.append("---")
        return "\n".join(prompt_parts)

    def get_enabled_tools(self) -> list[ToolType]:
        """Get list of enabled tools.

        Returns:
            List[ToolType]: List of enabled tool types.
        """
        return self.enabled_tools.copy()


class ToolRouter:
    """Tool router that determines which tool to call based on markers."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.tool_markers = self._build_tool_markers()

    def _build_tool_markers(self) -> list[tuple[ToolType, str]]:
        """Build tool markers based on enabled tools."""
        markers = []
        for tool_type, tool in self.registry.tools.items():
            for start_marker in tool.markers.start_markers:
                for end_marker in tool.markers.end_markers:
                    escaped_start = re.escape(start_marker)
                    escaped_end = re.escape(end_marker)
                    pattern = f"{escaped_start}(.*?){escaped_end}"
                    markers.append((tool_type, pattern))
        return markers

    def route(self, text: str) -> ToolType | None:
        """Determine tool type to call based on markers."""
        text = text.strip()
        for tool_type, pattern in self.tool_markers:
            if re.search(pattern, text, re.DOTALL | re.IGNORECASE):
                return tool_type
        return None


class ToolManager:
    """General tool manager responsible for coordinating tool calls."""

    def __init__(
        self,
        timeout: int = 30,
        enabled_tools: str = "python;calculator",
        debug_mode: bool = False,
    ):
        self.timeout = timeout
        self.debug_mode = debug_mode
        self.registry = ToolRegistry(timeout, enabled_tools, debug_mode)
        self.router = ToolRouter(self.registry)

        logger.info(
            f"Initialized ToolManager (debug_mode={debug_mode}, enabled_tools={[t.value for t in self.registry.get_enabled_tools()]})"
        )

    def get_tool_descriptions_prompt(self) -> str:
        """Get tool description prompt text for external calls."""
        return self.registry.get_tool_descriptions_prompt()

    def get_tool_markers(self) -> dict[ToolType, tuple[list[str], list[str]]]:
        """Get marker information for all tools.

        Returns:
            Dict[ToolType, Tuple[List[str], List[str]]]: Tool type ->
                (start markers list, end markers list)
        """
        return self.registry.get_tool_markers()

    def get_all_start_markers(self) -> list[str]:
        """Get all start markers for setting stop tokens.

        Returns:
            List[str]: List of all start markers, e.g. ['```python\\n', '<calculator>']
        """
        return self.registry.get_all_start_markers()

    def get_all_end_markers(self) -> list[str]:
        """Get all end markers for setting stop tokens.

        Returns:
            List[str]: List of all end markers, e.g. ['\\n```', '</calculator>']
        """
        return self.registry.get_all_end_markers()

    def get_all_markers(self) -> list[str]:
        """Get all markers (start and end) for setting stop tokens.

        Returns:
            List[str]: List of all markers, e.g. ['```python\\n', '\\n```', '<calculator>', '</calculator>']
        """
        return self.registry.get_all_markers()

    def _prepare_tool_call(
        self, text: str
    ) -> tuple[
        BaseTool | None, dict[str, str] | None, tuple[str, ToolCallStatus] | None
    ]:
        tool_type = self.router.route(text)
        if not tool_type:
            return (
                None,
                None,
                (
                    "Error: No suitable tool found for the given text",
                    ToolCallStatus.NOT_FOUND,
                ),
            )

        tool = self.registry.get_tool(tool_type)
        if not tool:
            return (
                None,
                None,
                (
                    f"Error: Tool {tool_type.value} not found",
                    ToolCallStatus.NOT_FOUND,
                ),
            )

        try:
            parameters = tool.parse_parameters(text)
            logger.debug(f"Parsed parameters: {parameters}")
        except Exception as exc:
            logger.error(f"Parameter parsing error: {exc}")
            return (
                None,
                None,
                (
                    f"Error: Failed to parse parameters - {exc}",
                    ToolCallStatus.ERROR,
                ),
            )

        return tool, parameters, None

    @staticmethod
    def _finalize_tool_result(
        result: str, status: ToolCallStatus
    ) -> tuple[str, ToolCallStatus]:
        if status == ToolCallStatus.SUCCESS:
            logger.debug(f"Tool execution completed: {result}")
            return result, status

        logger.error(f"Tool execution error: {result}")
        return f"Error: Tool execution failed - {result}", status

    def execute_tool_call(self, text: str) -> tuple[str, ToolCallStatus]:
        """Unified synchronous tool call interface.

        Returns:
            Tuple[str, ToolCallStatus]: (result, status)
        """
        tool, parameters, error = self._prepare_tool_call(text)
        if error is not None:
            return error

        assert tool is not None
        assert parameters is not None
        result, status = tool.execute(parameters)
        return self._finalize_tool_result(result, status)

    async def aexecute_tool_call(self, text: str) -> tuple[str, ToolCallStatus]:
        """Unified asynchronous tool call interface.

        Returns:
            Tuple[str, ToolCallStatus]: (result, status)
        """
        tool, parameters, error = self._prepare_tool_call(text)
        if error is not None:
            return error

        assert tool is not None
        assert parameters is not None
        result, status = await tool.aexecute(parameters)
        return self._finalize_tool_result(result, status)

    def cleanup(self) -> None:
        """Release any resources held by enabled tools."""
        for tool in self.registry.get_all_tools().values():
            tool.close()

    async def acleanup(self) -> None:
        """Asynchronously release any resources held by enabled tools."""
        for tool in self.registry.get_all_tools().values():
            await tool.aclose()

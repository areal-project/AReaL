from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from areal.utils import logging

logger = logging.getLogger("Tool Base")


class ToolCallStatus(Enum):
    """Tool call status enumeration"""

    SUCCESS = "success"
    ERROR = "error"
    NOT_FOUND = "not_found"


class ToolType(Enum):
    """Tool type enumeration"""

    PYTHON = "python"
    CALCULATOR = "calculator"
    DAYTONA_PYTHON = "daytona_python"


@dataclass
class ToolCall:
    """Tool call data structure"""

    tool_type: ToolType
    parameters: dict[str, Any]
    raw_text: str


@dataclass
class ToolDescription:
    """Tool description data structure"""

    name: str
    description: str
    parameters: dict[str, str]  # Parameter name -> parameter description
    parameter_prompt: str  # Prompt for parameter parsing
    example: str


@dataclass
class ToolMarkers:
    """Tool markers data structure"""

    start_markers: list[str]  # Start markers for the tool
    end_markers: list[str]  # End markers for the tool


class BaseTool(ABC):
    """Base tool abstract class"""

    def __init__(self, timeout: int = 30, debug_mode: bool = False):
        self.timeout = timeout
        self.debug_mode = debug_mode

    @property
    @abstractmethod
    def tool_type(self) -> ToolType:
        """Tool type"""

    @property
    @abstractmethod
    def description(self) -> ToolDescription:
        """Tool description"""

    @property
    @abstractmethod
    def markers(self) -> ToolMarkers:
        """Tool markers for parsing"""

    @abstractmethod
    def parse_parameters(self, text: str) -> dict[str, Any]:
        """Parse parameters"""

    @abstractmethod
    def execute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        """Execute tool.

        Returns:
            Tuple[str, ToolCallStatus]: (result, status)
        """

    async def aexecute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        """Async version of execute.

        Override directly for tools that are natively async. The default keeps
        existing synchronous tools working unchanged.
        """

        return await asyncio.to_thread(self.execute, parameters)

    async def aclose(self) -> None:
        """Async cleanup hook."""

        return None

    def close(self) -> None:
        """Release any tool resources.

        Most tools are stateless and do not need cleanup.
        """

        return None

import re
from typing import Any

from areal.utils import logging

from .base import BaseTool, ToolCallStatus, ToolDescription, ToolMarkers, ToolType

logger = logging.getLogger("Calculator Tool")


class CalculatorTool(BaseTool):
    """Basic calculator tool"""

    @property
    def tool_type(self) -> ToolType:
        return ToolType.CALCULATOR

    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="python calculator",
            description="Perform basic mathematical calculations, supporting addition, subtraction, multiplication, division, and parentheses.",
            parameters={"expression": "Mathematical expression string"},
            parameter_prompt="Please provide a mathematical expression. Supports addition, subtraction, multiplication, division, and parentheses.",
            example="<calculator>1 + 2 * 3</calculator>",
        )

    @property
    def markers(self) -> ToolMarkers:
        return ToolMarkers(
            start_markers=["<calculator>"], end_markers=["</calculator>"]
        )

    def parse_parameters(self, text: str) -> dict[str, Any]:
        """Extract mathematical expression from <calculator> tags"""
        pattern = r"<calculator>(.*?)</calculator>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

        if match:
            expression = match.group(1).strip()
            logger.debug(f"Extracted expression: {expression}")
            return {"expression": expression}
        else:
            logger.warning("No <calculator> tag found")
            return {"expression": ""}

    def execute(self, parameters: dict[str, Any]) -> tuple[str, ToolCallStatus]:
        """Execute mathematical calculation"""
        expression = parameters.get("expression", "")
        if not expression:
            return "Error: No expression provided", ToolCallStatus.ERROR

        if self.debug_mode:
            logger.debug(f"[FAKE] Executing calculator: {expression}")
            return "dummy calculator output", ToolCallStatus.SUCCESS

        try:
            # Simple mathematical expression calculation
            safe_pattern = r"^[0-9+\-*/().\s]+$"
            if not re.match(safe_pattern, expression):
                return "Error: Invalid expression", ToolCallStatus.ERROR

            # Use eval for calculation (in controlled environment)
            result = eval(expression)
            return str(result), ToolCallStatus.SUCCESS

        except Exception as e:
            return f"Error: {str(e)}", ToolCallStatus.ERROR

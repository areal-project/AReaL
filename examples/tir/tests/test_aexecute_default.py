# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from tools import BaseTool, ToolCallStatus, ToolDescription, ToolMarkers, ToolType


class DummyTool(BaseTool):
    @property
    def tool_type(self) -> ToolType:
        return ToolType.CALCULATOR

    @property
    def description(self) -> ToolDescription:
        return ToolDescription(
            name="dummy",
            description="dummy",
            parameters={},
            parameter_prompt="",
            example="",
        )

    @property
    def markers(self) -> ToolMarkers:
        return ToolMarkers(start_markers=["<dummy>"], end_markers=["</dummy>"])

    def parse_parameters(self, text: str) -> dict[str, str]:
        return {"text": text}

    def execute(self, parameters: dict[str, str]) -> tuple[str, ToolCallStatus]:
        return "ok", ToolCallStatus.SUCCESS


@pytest.mark.asyncio
async def test_base_tool_aexecute_delegates_to_execute_by_default():
    tool = DummyTool()

    assert await tool.aexecute({}) == ("ok", ToolCallStatus.SUCCESS)

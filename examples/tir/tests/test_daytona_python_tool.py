# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import importlib.util
import os

import pytest
from tool_manager import ToolRegistry
from tools import ToolCallStatus, ToolType
from tools.daytona_python_tool import DaytonaPythonTool


def _has_daytona_sdk() -> bool:
    return importlib.util.find_spec("daytona") is not None


def _has_daytona_credentials() -> bool:
    return _has_daytona_sdk() and bool(os.environ.get("DAYTONA_API_KEY"))


def test_daytona_python_tool_execute_raises_not_implemented():
    tool = DaytonaPythonTool(debug_mode=True)

    with pytest.raises(NotImplementedError, match="aexecute"):
        tool.execute({"code": "print(1)"})


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
@pytest.mark.asyncio
async def test_daytona_python_tool_simple_run():
    tool = DaytonaPythonTool()

    try:
        result, status = await tool.aexecute({"code": "print(2 + 2)"})
    finally:
        await asyncio.to_thread(tool.close)

    assert status == ToolCallStatus.SUCCESS
    assert "4" in result


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
@pytest.mark.asyncio
async def test_daytona_python_tool_preserves_state():
    tool = DaytonaPythonTool()

    try:
        first_result, first_status = await tool.aexecute({"code": "x = 5"})
        second_result, second_status = await tool.aexecute({"code": "print(x)"})
    finally:
        await asyncio.to_thread(tool.close)

    assert first_status == ToolCallStatus.SUCCESS
    assert first_result == ""
    assert second_status == ToolCallStatus.SUCCESS
    assert "5" in second_result


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_daytona_credentials(), reason="DAYTONA_API_KEY is required"
)
@pytest.mark.asyncio
async def test_daytona_python_tool_syntax_error():
    tool = DaytonaPythonTool()

    try:
        result, status = await tool.aexecute({"code": "def f(: pass"})
    finally:
        await asyncio.to_thread(tool.close)

    assert status == ToolCallStatus.ERROR
    assert "Traceback" in result


@pytest.mark.skipif(not _has_daytona_sdk(), reason="daytona package is required")
def test_tool_registry_enables_daytona_python():
    registry = ToolRegistry(enabled_tools="daytona_python", debug_mode=True)

    assert registry.get_enabled_tools() == [ToolType.DAYTONA_PYTHON]
    assert ToolType.DAYTONA_PYTHON in registry.get_tool_markers()

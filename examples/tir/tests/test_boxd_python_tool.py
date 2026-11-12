# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import importlib.util
import os

import pytest

pytest.importorskip("boxd")

from tool_manager import ToolManager, ToolRegistry  # noqa: E402
from tools import ToolCallStatus, ToolType  # noqa: E402
from tools.boxd_python_tool import BoxdPythonTool  # noqa: E402

from areal.infra.sandbox import BoxdClientManager  # noqa: E402
from areal.infra.sandbox._boxd_session import (  # noqa: E402
    BoxdCellError,
    BoxdCellResult,
)


def _has_boxd_credentials() -> bool:
    return importlib.util.find_spec("boxd") is not None and bool(
        os.environ.get("BOXD_API_KEY")
    )


@pytest.fixture
def fake_cells(monkeypatch):
    calls: list[str] = []
    results: list[BoxdCellResult] = []

    async def fake_get_client():
        return object()

    async def fake_arun_cell(self, client, code, *, timeout):
        calls.append(code)
        return results.pop(0)

    monkeypatch.setattr(BoxdClientManager, "get_client", fake_get_client)
    monkeypatch.setattr(
        "areal.infra.sandbox._boxd_session.BoxdSession.arun_cell", fake_arun_cell
    )
    return calls, results


def test_boxd_python_tool_execute_raises_not_implemented():
    tool = BoxdPythonTool(debug_mode=True)

    with pytest.raises(NotImplementedError, match="aexecute"):
        tool.execute({"code": "print(1)"})


@pytest.mark.asyncio
async def test_boxd_python_tool_joins_stdout_and_stderr(fake_cells):
    calls, results = fake_cells
    results.append(BoxdCellResult(stdout="4\n", stderr="warning\n"))
    tool = BoxdPythonTool()

    output, status = await tool.aexecute({"code": "print(2 + 2)"})

    assert calls == ["print(2 + 2)"]
    assert status == ToolCallStatus.SUCCESS
    assert output == "4\nwarning"


@pytest.mark.asyncio
async def test_boxd_python_tool_formats_errors_with_traceback(fake_cells):
    _, results = fake_cells
    results.append(
        BoxdCellResult(
            stdout="partial\n",
            error=BoxdCellError(
                name="ZeroDivisionError",
                value="division by zero",
                traceback='Traceback (most recent call last):\n  File "<cell>", line 1\nZeroDivisionError: division by zero\n',
            ),
        )
    )
    tool = BoxdPythonTool()

    output, status = await tool.aexecute({"code": "print('partial'); 1/0"})

    assert status == ToolCallStatus.ERROR
    assert output.startswith("partial\nTraceback")
    assert output.endswith("ZeroDivisionError: division by zero")


@pytest.mark.asyncio
async def test_boxd_python_tool_rejects_empty_code():
    tool = BoxdPythonTool()

    output, status = await tool.aexecute({"code": ""})

    assert status == ToolCallStatus.ERROR
    assert "No code" in output


def test_tool_registry_enables_boxd_python():
    registry = ToolRegistry(enabled_tools="boxd_python", debug_mode=True)

    assert registry.get_enabled_tools() == [ToolType.BOXD_PYTHON]
    assert ToolType.BOXD_PYTHON in registry.get_tool_markers()


@pytest.mark.parametrize("tools", ["python;boxd_python", "daytona_python;boxd_python"])
def test_tool_registry_rejects_boxd_with_another_python_backend(tools):
    with pytest.raises(ValueError, match="enable only one Python backend"):
        ToolRegistry(enabled_tools=tools, debug_mode=True)


@pytest.mark.integration
@pytest.mark.skipif(not _has_boxd_credentials(), reason="BOXD_API_KEY is required")
@pytest.mark.asyncio
async def test_boxd_python_tool_preserves_state():
    tool = BoxdPythonTool()

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
@pytest.mark.skipif(not _has_boxd_credentials(), reason="BOXD_API_KEY is required")
@pytest.mark.asyncio
async def test_boxd_python_tool_syntax_error_and_timeout():
    tool = BoxdPythonTool(timeout=2)

    try:
        _, _ = await tool.aexecute({"code": "y = 7"})
        syntax, syntax_status = await tool.aexecute({"code": "def f(: pass"})
        slow, slow_status = await tool.aexecute({"code": "import time; time.sleep(10)"})
        after, after_status = await tool.aexecute({"code": "print(y)"})
    finally:
        await asyncio.to_thread(tool.close)

    assert syntax_status == ToolCallStatus.ERROR
    assert "SyntaxError" in syntax
    assert slow_status == ToolCallStatus.ERROR
    assert "TimeoutError" in slow
    assert after_status == ToolCallStatus.SUCCESS
    assert after == "7"


@pytest.mark.integration
@pytest.mark.skipif(not _has_boxd_credentials(), reason="BOXD_API_KEY is required")
@pytest.mark.asyncio
async def test_tool_manager_async_dispatch_boxd():
    manager = ToolManager(enabled_tools="boxd_python")

    try:
        result, status = await manager.aexecute_tool_call(
            "```python\nimport math\nprint(math.sqrt(16))\n```"
        )
    finally:
        await asyncio.to_thread(manager.cleanup)

    assert status == ToolCallStatus.SUCCESS
    assert result == "4.0"

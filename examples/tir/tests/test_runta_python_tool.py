# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace

import pytest
import tool_manager as tool_manager_module
from tool_manager import ToolRegistry
from tools import ToolCallStatus, ToolType
from tools.runta_python_tool import RuntaPythonTool

import areal.infra.sandbox.runta as runta_module
from areal.infra.sandbox import RuntaRunResult


def _has_runta_credentials() -> bool:
    return importlib.util.find_spec("runta") is not None and bool(
        os.environ.get("RUNTA_TOKEN")
    )


class FakeRunner:
    """Stands in for AsyncRuntaRunner; records how the tool drives it."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[tuple[str, int | None]] = []
        self.results: list[RuntaRunResult] = []
        self.closes = 0

    async def arun(self, code, *, timeout=None):
        self.calls.append((code, timeout))
        return self.results.pop(0) if self.results else RuntaRunResult()

    async def aclose(self):
        self.closes += 1


@pytest.fixture
def fake_runner(monkeypatch):
    import tools.runta_python_tool as tool_module

    monkeypatch.setattr(tool_module, "AsyncRuntaRunner", FakeRunner)


def test_runta_python_tool_execute_raises_not_implemented(fake_runner):
    tool = RuntaPythonTool(debug_mode=True)

    with pytest.raises(NotImplementedError, match="aexecute"):
        tool.execute({"code": "print(1)"})


def test_runta_python_tool_forwards_sandbox_options(fake_runner):
    tool = RuntaPythonTool(
        timeout=12, image="python:3.12-slim", memory_mib=2048, network_block_all=True
    )

    assert tool.tool_type == ToolType.RUNTA_PYTHON
    assert tool._runner.kwargs["image"] == "python:3.12-slim"
    assert tool._runner.kwargs["memory_mib"] == 2048
    assert tool._runner.kwargs["network_block_all"] is True
    assert tool._runner.kwargs["default_timeout"] == 12


@pytest.mark.asyncio
async def test_runta_python_tool_empty_code_is_error_without_sandbox(fake_runner):
    tool = RuntaPythonTool()

    result, status = await tool.aexecute({"code": ""})
    await tool.aclose()

    assert status == ToolCallStatus.ERROR
    assert "No code" in result
    assert tool._runner.calls == []
    assert tool._runner.closes == 0


@pytest.mark.asyncio
async def test_runta_python_tool_debug_mode_does_not_touch_sandbox(fake_runner):
    tool = RuntaPythonTool(debug_mode=True)

    result, status = await tool.aexecute({"code": "print(1)"})

    assert (result, status) == ("dummy python output", ToolCallStatus.SUCCESS)
    assert tool._runner.calls == []


@pytest.mark.asyncio
async def test_runta_python_tool_joins_streams_on_success(fake_runner):
    tool = RuntaPythonTool(timeout=9)
    tool._runner.results = [RuntaRunResult(stdout="4\n", stderr="warning: x\n")]

    result, status = await tool.aexecute({"code": "print(2 + 2)"})

    assert status == ToolCallStatus.SUCCESS
    assert result == "4\nwarning: x"
    assert tool._runner.calls == [("print(2 + 2)", 9)]


@pytest.mark.asyncio
async def test_runta_python_tool_returns_traceback_on_failure(fake_runner):
    tool = RuntaPythonTool()
    traceback = (
        "Traceback (most recent call last):\nZeroDivisionError: division by zero"
    )
    tool._runner.results = [
        RuntaRunResult(
            stdout="before\n",
            stderr=traceback + "\n",
            exit_code=1,
            error="ZeroDivisionError: division by zero",
        )
    ]

    result, status = await tool.aexecute({"code": "print('before'); 1/0"})

    assert status == ToolCallStatus.ERROR
    assert result == "before\n" + traceback


@pytest.mark.asyncio
async def test_runta_python_tool_reports_error_when_failure_has_no_output(fake_runner):
    tool = RuntaPythonTool()
    tool._runner.results = [
        RuntaRunResult(exit_code=1, error="runtime unavailable", stderr="")
    ]

    result, status = await tool.aexecute({"code": "print(1)"})

    assert (result, status) == ("runtime unavailable", ToolCallStatus.ERROR)


@pytest.mark.asyncio
async def test_runta_python_tool_aclose_deletes_sandbox_once(fake_runner):
    tool = RuntaPythonTool()
    await tool.aexecute({"code": "print(1)"})

    await tool.aclose()
    await tool.aclose()

    assert tool._runner.closes == 1


def test_tool_registry_builds_runta_python_lazily(monkeypatch):
    built = []
    monkeypatch.setattr(
        tool_manager_module,
        "_build_runta_python_tool",
        lambda timeout, debug_mode: built.append((timeout, debug_mode))
        or SimpleNamespace(),
    )

    ToolRegistry(enabled_tools="calculator")
    assert built == []

    registry = ToolRegistry(timeout=11, enabled_tools="runta_python;calculator")
    assert built == [(11, False)]
    assert registry.get_enabled_tools() == [ToolType.RUNTA_PYTHON, ToolType.CALCULATOR]


@pytest.mark.parametrize(
    "enabled", ["python;runta_python", "daytona_python;runta_python"]
)
def test_tool_registry_rejects_runta_with_another_python_backend(enabled):
    with pytest.raises(ValueError, match="enable only one Python backend"):
        ToolRegistry(enabled_tools=enabled)


def test_runta_factory_explains_missing_sdk(monkeypatch):
    monkeypatch.setattr(
        tool_manager_module.importlib.util, "find_spec", lambda name: None
    )

    with pytest.raises(ImportError, match="pip install"):
        tool_manager_module._build_runta_python_tool(30, False)


def test_runta_python_tool_requires_sdk(monkeypatch):
    def raise_import_error():
        raise ImportError("runta-sdk is not installed")

    monkeypatch.setattr(runta_module, "_load_runta_sdk", raise_import_error)

    with pytest.raises(ImportError, match="runta-sdk"):
        RuntaPythonTool()


@pytest.mark.integration
@pytest.mark.skipif(not _has_runta_credentials(), reason="RUNTA_TOKEN is required")
@pytest.mark.asyncio
async def test_runta_python_tool_runs_code_and_keeps_files_live():
    tool = RuntaPythonTool()

    try:
        result, status = await tool.aexecute({"code": "print(2 + 2)"})
        await tool.aexecute({"code": "open('/tmp/x.txt', 'w').write('kept')"})
        kept, kept_status = await tool.aexecute(
            {"code": "print(open('/tmp/x.txt').read())"}
        )
        failed, failed_status = await tool.aexecute({"code": "1/0"})
    finally:
        await tool.aclose()

    assert (result, status) == ("4", ToolCallStatus.SUCCESS)
    assert (kept, kept_status) == ("kept", ToolCallStatus.SUCCESS)
    assert failed_status == ToolCallStatus.ERROR
    assert "ZeroDivisionError" in failed

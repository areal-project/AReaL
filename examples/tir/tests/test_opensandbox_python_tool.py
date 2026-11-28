# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import importlib.util
import json
import os

import pytest
from tool_manager import ToolRegistry
from tools import ToolCallStatus, ToolType
from tools.opensandbox_python_tool import OpenSandboxPythonTool

from tests.infra.sandbox._fake_opensandbox import (
    FakeExecution,
    install,
)


def _has_opensandbox_sdk() -> bool:
    return importlib.util.find_spec("opensandbox") is not None


def _has_opensandbox_endpoint() -> bool:
    return _has_opensandbox_sdk() and bool(os.environ.get("OPEN_SANDBOX_DOMAIN"))


@pytest.fixture
def fake_sdk(monkeypatch):
    return install(monkeypatch)


def _queue_cell(fake_sdk, stdout="", error=None, exit_code=0):
    fake_sdk.state.queue_execution(FakeExecution(stdout="0\n", exit_code=0))  # start
    payload = json.dumps({"stdout": stdout, "stderr": "", "error": error})
    fake_sdk.state.queue_execution(FakeExecution(stdout=payload, exit_code=exit_code))


def test_tool_metadata(fake_sdk):
    tool = OpenSandboxPythonTool(debug_mode=True)

    assert tool.tool_type is ToolType.OPENSANDBOX_PYTHON
    assert tool.description.name == "opensandbox_python_executor"
    assert tool.markers.start_markers == ["```python", "<python>"]
    assert tool.markers.end_markers == ["```", "</python>"]
    assert tool.parse_parameters("```python\nprint(1)\n```") == {"code": "print(1)"}


def test_tool_execute_raises_not_implemented(fake_sdk):
    tool = OpenSandboxPythonTool(debug_mode=True)

    with pytest.raises(NotImplementedError, match="aexecute"):
        tool.execute({"code": "print(1)"})


@pytest.mark.asyncio
async def test_tool_debug_mode_returns_dummy(fake_sdk):
    tool = OpenSandboxPythonTool(debug_mode=True)

    result, status = await tool.aexecute({"code": "print(1)"})

    assert status == ToolCallStatus.SUCCESS
    assert result == "dummy python output"


@pytest.mark.asyncio
async def test_tool_empty_code_is_error(fake_sdk):
    tool = OpenSandboxPythonTool()

    result, status = await tool.aexecute({"code": ""})

    assert status == ToolCallStatus.ERROR
    assert "No code provided" in result


@pytest.mark.asyncio
async def test_tool_runs_code_and_returns_output(fake_sdk):
    _queue_cell(fake_sdk, stdout="4\n")
    tool = OpenSandboxPythonTool()

    result, status = await tool.aexecute({"code": "print(2 + 2)"})

    assert status == ToolCallStatus.SUCCESS
    assert "4" in result


@pytest.mark.asyncio
async def test_tool_formats_cell_error(fake_sdk):
    error = {
        "name": "ValueError",
        "value": "x",
        "traceback": "Traceback (most recent call last):\nValueError: x\n",
    }
    _queue_cell(fake_sdk, error=error)
    tool = OpenSandboxPythonTool()

    result, status = await tool.aexecute({"code": "raise ValueError('x')"})

    assert status == ToolCallStatus.ERROR
    assert "ValueError" in result
    assert "Traceback" in result


@pytest.mark.asyncio
async def test_tool_close_deletes_sandbox(fake_sdk):
    _queue_cell(fake_sdk, stdout="")
    tool = OpenSandboxPythonTool()
    await tool.aexecute({"code": "x = 1"})
    sandbox = fake_sdk.state.sandboxes[0]

    await asyncio.to_thread(tool.close)

    assert sandbox.destroy_calls == 1
    assert tool.sandbox_id is None


@pytest.mark.asyncio
async def test_tool_aclose_deletes_sandbox(fake_sdk):
    _queue_cell(fake_sdk, stdout="")
    tool = OpenSandboxPythonTool()
    await tool.aexecute({"code": "x = 1"})
    sandbox = fake_sdk.state.sandboxes[0]

    await tool.aclose()

    assert sandbox.destroy_calls == 1


def test_tool_close_before_first_use_is_noop(fake_sdk):
    tool = OpenSandboxPythonTool()

    tool.close()

    assert fake_sdk.state.sandboxes == []


@pytest.mark.asyncio
async def test_tool_ttl_passed_to_sandbox(fake_sdk):
    _queue_cell(fake_sdk, stdout="")
    tool = OpenSandboxPythonTool(ttl_seconds=1800)

    await tool.aexecute({"code": "x = 1"})

    from datetime import timedelta

    assert fake_sdk.state.create_calls[0]["timeout"] == timedelta(seconds=1800)


def test_tool_registry_enables_opensandbox_python(fake_sdk):
    registry = ToolRegistry(enabled_tools="opensandbox_python", debug_mode=True)

    assert registry.get_enabled_tools() == [ToolType.OPENSANDBOX_PYTHON]
    assert ToolType.OPENSANDBOX_PYTHON in registry.get_tool_markers()


def test_tool_registry_rejects_two_python_backends(fake_sdk):
    with pytest.raises(ValueError, match="only one Python backend"):
        ToolRegistry(enabled_tools="python;opensandbox_python")
    with pytest.raises(ValueError, match="only one Python backend"):
        ToolRegistry(enabled_tools="daytona_python;opensandbox_python")


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_opensandbox_endpoint(), reason="OPEN_SANDBOX_DOMAIN is required"
)
@pytest.mark.asyncio
async def test_tool_simple_run_live():
    tool = OpenSandboxPythonTool()

    try:
        result, status = await tool.aexecute({"code": "print(2 + 2)"})
    finally:
        await tool.aclose()

    assert status == ToolCallStatus.SUCCESS
    assert "4" in result


@pytest.mark.integration
@pytest.mark.skipif(
    not _has_opensandbox_endpoint(), reason="OPEN_SANDBOX_DOMAIN is required"
)
@pytest.mark.asyncio
async def test_tool_preserves_state_live():
    tool = OpenSandboxPythonTool()

    try:
        first_result, first_status = await tool.aexecute({"code": "x = 5"})
        second_result, second_status = await tool.aexecute({"code": "print(x)"})
    finally:
        await tool.aclose()

    assert first_status == ToolCallStatus.SUCCESS
    assert first_result == ""
    assert second_status == ToolCallStatus.SUCCESS
    assert "5" in second_result

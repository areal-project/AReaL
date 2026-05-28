# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tool_manager as tool_manager_module
from tool_manager import ToolRegistry
from tools import ToolType
from tools.python_tool import PythonTool


def test_tool_registry_only_constructs_enabled_tools(monkeypatch):
    def fail_python_tool_init(*args, **kwargs):
        raise RuntimeError("should-not-be-called")

    monkeypatch.setattr(PythonTool, "__init__", fail_python_tool_init)

    registry = ToolRegistry(enabled_tools="calculator")

    assert registry.get_enabled_tools() == [ToolType.CALCULATOR]
    assert list(registry.get_all_tools()) == [ToolType.CALCULATOR]


def test_tool_registry_none_uses_builtin_defaults(monkeypatch):
    def fail_daytona_factory(*args, **kwargs):
        raise RuntimeError("daytona factory should not be called")

    monkeypatch.setattr(
        tool_manager_module,
        "_build_daytona_python_tool",
        fail_daytona_factory,
    )

    registry = ToolRegistry(enabled_tools=None, debug_mode=True)

    assert registry.get_enabled_tools() == [ToolType.PYTHON, ToolType.CALCULATOR]


def test_tool_registry_rejects_two_python_backends():
    try:
        ToolRegistry(enabled_tools="python;daytona_python")
    except ImportError:
        return
    except ValueError as exc:
        assert "enable only one Python backend" in str(exc)
        return

    raise AssertionError("expected ToolRegistry to reject multiple Python backends")

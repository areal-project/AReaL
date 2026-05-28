# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import sys

import tools as tools_module

from areal.infra.sandbox import DaytonaRunner


def test_tools_import_without_daytona():
    original_find_spec = importlib.util.find_spec

    with __import__("pytest").MonkeyPatch.context() as monkeypatch:

        def fake_find_spec(name, *args, **kwargs):
            if name == "daytona":
                return None
            return original_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        sys.modules.pop("tools.daytona_python_tool", None)
        reloaded = importlib.reload(tools_module)

        assert hasattr(reloaded, "PythonTool")
        assert hasattr(reloaded, "CalculatorTool")
        assert not hasattr(reloaded, "DaytonaPythonTool")

    importlib.reload(tools_module)


def test_sandbox_import_is_safe_without_daytona():
    assert DaytonaRunner is not None

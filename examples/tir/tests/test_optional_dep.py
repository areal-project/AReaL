# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys

import tools as tools_module

from areal.infra.sandbox import BoxdRunner, DaytonaRunner


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


def test_tools_import_without_boxd():
    original_find_spec = importlib.util.find_spec

    with __import__("pytest").MonkeyPatch.context() as monkeypatch:

        def fake_find_spec(name, *args, **kwargs):
            if name == "boxd":
                return None
            return original_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
        sys.modules.pop("tools.boxd_python_tool", None)
        reloaded = importlib.reload(tools_module)

        assert hasattr(reloaded, "PythonTool")
        assert not hasattr(reloaded, "BoxdPythonTool")

    importlib.reload(tools_module)


def test_sandbox_import_is_safe_without_boxd():
    code = (
        "import sys\n"
        "sys.modules['boxd'] = None\n"
        "from areal.infra.sandbox import BoxdRunner\n"
        "try:\n"
        "    BoxdRunner()\n"
        "except ImportError as exc:\n"
        "    print('ImportError:', exc)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert BoxdRunner is not None
    assert "uv sync --extra sandbox" in out.stdout

# SPDX-License-Identifier: Apache-2.0
"""Workspace state and instructions shared by native Hermes task workers."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any


class TaskWorkspace:
    kind = "python"
    native_tools = True

    def __init__(
        self,
        output: Path,
        task_id: str,
        *,
        workspace: Path | None = None,
        tool_path: str | None = None,
        python_path: str | None = None,
        instructions: str = "",
    ) -> None:
        self.task_id = task_id
        self.directory = output.resolve() / "tools" / task_id
        self.directory.mkdir(parents=True, exist_ok=False)
        if workspace is None:
            workspace = self.directory / "workspace"
            workspace.mkdir()
        self.workdir = str(workspace.resolve(strict=True))
        if not Path(self.workdir).is_dir() or self.workdir == "/":
            raise ValueError("Supply a prepared task workspace directory")
        # Select the task's interpreter independently of the bundled Agent Python.
        self.tool_path = tool_path or os.environ.get("PATH", os.defpath)
        self.python_path = python_path
        self.instructions = instructions
        self.records: list[dict[str, Any]] = []
        self.call_timings: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "backend": "hermes-native-workspace",
            "task_id": self.task_id,
            "workspace": self.workdir,
            "calls": list(self.records),
            "call_timings": list(self.call_timings),
        }

    def close(self) -> None:
        # Native tool lifecycle owns process cleanup; keep files as evidence.
        pass


def workspace_instructions(runtime: TaskWorkspace) -> str:
    return f"""Answer-producing task
Your task workspace is {runtime.workdir}. Use the native terminal and file tools
to compute, inspect inputs, write scripts, and check results when helpful.
To run Python, execute Python code through terminal (for example a Python script
or a shell heredoc). Each new Python process has fresh interpreter state; files
persist within this task.
Follow the task's requested final-answer format. Your submission is the final
answer text, not a repository patch. Report only checks supported by tool output.
{runtime.instructions}
"""

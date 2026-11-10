# SPDX-License-Identifier: Apache-2.0
"""Public SWE task inputs; grading stays in the pinned Arena Reward sandbox."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.v2_online_workflows.types import EnvironmentSpec, TaskInput, TaskResult


@dataclass(frozen=True)
class SWEBenchTask:
    task_id: str
    prompt: str
    repo: str
    base_commit: str
    image: str
    workdir: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.task_id):
            raise ValueError("Expected a path-safe SWE task identifier")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repo):
            raise ValueError("Expected owner/repository")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", self.base_commit):
            raise ValueError("Expected the original full base commit")
        if (
            not self.prompt.strip()
            or not self.image
            or not Path(self.workdir).is_absolute()
        ):
            raise ValueError("SWE requires a prompt, image and absolute workdir")


class SWEBenchBenchmark:
    name = "swe_bench"
    request_prefix = "swe"

    @staticmethod
    def restore_task(value: dict[str, Any]) -> SWEBenchTask:
        return SWEBenchTask(**value)

    @staticmethod
    def load_tasks(path: Path) -> tuple[list[SWEBenchTask], dict[str, Any]]:
        manifest = json.loads((path.parent / "manifest.json").read_text())
        payload = path.read_bytes()
        split = manifest["splits"][path.name]
        if (
            manifest.get("format_version") != 1
            or manifest.get("benchmark") != "swe_bench"
            or hashlib.sha256(payload).hexdigest() != split["sha256"]
        ):
            raise ValueError("SWE inputs differ from their frozen manifest")
        tasks = [
            SWEBenchTask(**json.loads(line))
            for line in payload.splitlines()
            if line.strip()
        ]
        if len(tasks) != split["count"] or len({t.task_id for t in tasks}) != len(
            tasks
        ):
            raise ValueError("SWE task count or identities differ from the manifest")
        return tasks, manifest

    def prepare(self, task: SWEBenchTask) -> TaskInput:
        return TaskInput(
            task.task_id,
            task.prompt,
            EnvironmentSpec(
                "repository",
                task.image,
                {
                    "repo": task.repo,
                    "base_commit": task.base_commit,
                    "workdir": task.workdir,
                },
            ),
        )

    def grade(self, task: SWEBenchTask, result: TaskResult, output: Path):
        raise RuntimeError(
            "SWE-bench uses the configured Arena Reward; no local grading"
        )

    def result_metadata(self, task: SWEBenchTask) -> dict[str, Any]:
        return {"repo": task.repo, "base_commit": task.base_commit}

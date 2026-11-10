# SPDX-License-Identifier: Apache-2.0

"""Task-local Docker repository execution and patch export, without grading."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.v2_online_workflows.environments.python import run_limited
from examples.v2_online_workflows.trace import mark_tool_execution


@dataclass(frozen=True)
class RepositorySettings:
    image: str
    root: Path
    timeout: float = 120.0
    max_output_bytes: int = 65536
    memory_mb: int = 4096
    cpus: float = 2.0
    max_patch_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        if not self.image or not self.root.is_absolute():
            raise ValueError(
                "A repository image and absolute tool log root are required"
            )
        if (
            min(
                self.timeout,
                self.max_output_bytes,
                self.memory_mb,
                self.cpus,
                self.max_patch_bytes,
            )
            <= 0
        ):
            raise ValueError("Repository resource limits must be positive")

    @classmethod
    def from_env(cls, image: str) -> RepositorySettings:
        return cls(
            image=image,
            root=Path(os.environ["AREAL_TASK_TOOL_ROOT"]),
            timeout=float(os.environ.get("AREAL_TASK_REPO_TIMEOUT", "120")),
            memory_mb=int(os.environ.get("AREAL_TASK_REPO_MEMORY_MB", "4096")),
            cpus=float(os.environ.get("AREAL_TASK_REPO_CPUS", "2")),
        )


class RepositoryEnvironment:
    """Only public task initialization enters the agent's repository container."""

    kind = "repository"
    toolset = "areal_repository"
    tool_schemas = [
        {
            "name": "workspace_exec",
            "description": (
                "Run a bash command in this task's repository. Use it to inspect files, "
                "edit source code and run tests. The working directory resets to the "
                "repository root on each call; filesystem changes persist within this task."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        }
    ]

    def __init__(
        self,
        settings: RepositorySettings,
        task_id: str,
        *,
        image_id: str,
        repo: str,
        base_commit: str,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("Repository must be owner/name")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", base_commit):
            raise ValueError("Repository base_commit must be a full commit hash")
        self.settings, self.task_id, self.image_id = settings, task_id, image_id
        self.repo, self.base_commit = repo, base_commit
        self.workdir = "/" + repo.split("/")[1]
        self.directory = settings.root / task_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.container: str | None = None
        self.records: list[dict[str, Any]] = []
        self.call_timings: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _control(self, *args: str) -> str:
        return subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout.strip()

    def _command(self, *args: str) -> list[str]:
        return ["docker", "exec", self.container, *args]

    def _start(self) -> None:
        if self.container is not None:
            return
        self.container = "areal-repository-" + uuid.uuid4().hex
        self._control(
            "create",
            "--name",
            self.container,
            "--network=none",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=256",
            f"--memory={self.settings.memory_mb}m",
            f"--memory-swap={self.settings.memory_mb}m",
            f"--cpus={self.settings.cpus}",
            "--workdir",
            self.workdir,
            "--entrypoint=/bin/bash",
            self.image_id,
            "-c",
            "while :; do sleep 3600; done",
        )
        self._control("start", self.container)
        self._control(
            "exec", self.container, "bash", "-lc", "command -v timeout >/dev/null"
        )
        self._control(
            "exec", self.container, "git", "reset", "--hard", self.base_commit
        )
        self._control("exec", self.container, "git", "clean", "-ffd")
        head = self._control("exec", self.container, "git", "rev-parse", "HEAD")
        if head != self.base_commit:
            raise RuntimeError(
                "Repository image did not reset to the frozen base commit"
            )

    def execute(self, command: str) -> str:
        with self._lock:
            if (
                not isinstance(command, str)
                or not command.strip()
                or len(command.encode()) > 65536
            ):
                return json.dumps(
                    {
                        "status": "invalid_command",
                        "error": "A bounded command is required",
                    }
                )
            started = time.time()
            mark_tool_execution(started)
            try:
                self._start()
                result = run_limited(
                    self._command(
                        "timeout",
                        "--kill-after=5s",
                        str(self.settings.timeout),
                        "bash",
                        "-lc",
                        command,
                    ),
                    timeout=self.settings.timeout + 10,
                    output_bytes=self.settings.max_output_bytes,
                    truncate=True,
                    separate_stderr=True,
                    preserve_tail=True,
                    process_group=getattr(self, "host_process_group", False),
                    env=getattr(self, "host_env", None),
                )
                if result["exit_code"] == 124:
                    result["status"] = "timeout"
                if result["exit_code"] == 125:
                    result["status"] = "infrastructure_error"
            except (OSError, subprocess.SubprocessError) as exc:
                result = {"status": "infrastructure_error", "error": str(exc)}
            record = {
                "task_id": self.task_id,
                "call": len(self.records) + 1,
                "started": started,
                "finished": time.time(),
                "image_id": self.image_id,
                "container": self.container,
                "command": command,
                **result,
            }
            self.records.append(record)
            with (self.directory / "tools.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            return json.dumps(result, ensure_ascii=False)

    def invoke(self, name: str, arguments: dict[str, Any]) -> str:
        clock = time.perf_counter()
        record = {
            "name": name,
            "started": time.time(),
            "finished": None,
            "seconds": None,
        }
        self.call_timings.append(record)
        try:
            return self.execute(arguments.get("command"))
        finally:
            record.update(finished=time.time(), seconds=time.perf_counter() - clock)

    def export_artifacts(self) -> dict[str, Any]:
        self._start()
        self._control("exec", self.container, "git", "add", "--intent-to-add", "--all")
        result = run_limited(
            self._command(
                "git",
                "diff",
                "--binary",
                self.base_commit,
                "--",
            ),
            timeout=60,
            output_bytes=self.settings.max_patch_bytes,
            process_group=getattr(self, "host_process_group", False),
            env=getattr(self, "host_env", None),
            separate_stderr=True,
            keep_stdout_bytes=True,
        )
        if result["status"] != "completed" or result["exit_code"] != 0:
            raise RuntimeError("Cannot export the complete candidate patch")
        raw_patch = result["stdout_bytes"]
        patch = raw_patch.decode("utf-8", errors="replace")
        (self.directory / "candidate.patch").write_bytes(raw_patch)
        exported = {"patch": patch, "base_commit": self.base_commit}
        if patch.encode("utf-8") != raw_patch:
            exported["patch_base64"] = base64.b64encode(raw_patch).decode("ascii")
        return exported

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "runtime_task_id": self.task_id,
            "image_id": self.image_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "calls": self.records,
            "timing_calls": self.call_timings,
        }

    def close(self) -> None:
        if self.container is not None:
            name, self.container = self.container, None
            self._control("rm", "--force", name)

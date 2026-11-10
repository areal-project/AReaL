# SPDX-License-Identifier: Apache-2.0

"""A task-local Python execution environment, independent of the agent harness."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.v2_online_workflows.environments.process import run_limited as run_limited
from examples.v2_online_workflows.trace import mark_tool_execution


@dataclass(frozen=True)
class ToolSettings:
    image: str
    root: Path
    timeout: float = 15.0
    max_output_bytes: int = 65536
    max_code_bytes: int = 16384
    memory_mb: int = 512
    max_calls: int = 16
    cpus: float = 1.0

    def __post_init__(self) -> None:
        if not self.image or not self.root.is_absolute():
            raise ValueError("A tool image and an absolute tool log root are required")
        if (
            min(
                self.timeout,
                self.max_output_bytes,
                self.max_code_bytes,
                self.memory_mb,
                self.max_calls,
                self.cpus,
            )
            <= 0
        ):
            raise ValueError("Tool resource limits must be positive")

    @classmethod
    def from_env(cls) -> ToolSettings:
        return cls(
            image=os.environ.get("AREAL_TASK_PYTHON_IMAGE")
            or os.environ["HERMES_MATH_IMAGE"],
            root=Path(
                os.environ.get("AREAL_TASK_TOOL_ROOT")
                or os.environ["HERMES_MATH_TOOL_ROOT"]
            ),
            timeout=float(
                os.environ.get(
                    "AREAL_TASK_TOOL_TIMEOUT",
                    os.environ.get("HERMES_MATH_TOOL_TIMEOUT", "15"),
                )
            ),
            max_output_bytes=int(
                os.environ.get("HERMES_MATH_TOOL_OUTPUT_BYTES", "65536")
            ),
            max_calls=int(os.environ.get("HERMES_MATH_TOOL_MAX_CALLS", "16")),
        )


class PythonSandbox:
    """Only model-generated code enters Docker; no host-execution fallback exists."""

    kind = "python"
    toolset = "areal_math"
    tool_schemas = [
        {
            "name": "math_python",
            "description": (
                "Execute Python to calculate, enumerate or check a mathematical result. "
                "Use print() to return results. Each call starts a fresh interpreter; "
                "files in /work persist only within this task. No network access."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        }
    ]

    def __init__(self, settings: ToolSettings, task_id: str, *, image_id: str) -> None:
        self.settings, self.task_id, self.image_id = settings, task_id, image_id
        self.directory = settings.root / task_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.container: str | None = None
        self.records: list[dict[str, Any]] = []
        self.call_timings: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def _control(self, *args: str) -> str:
        result = subprocess.run(
            ["docker", *args], capture_output=True, text=True, timeout=30, check=True
        )
        return result.stdout.strip()

    def _command(self, *args: str) -> list[str]:
        return ["docker", "exec", self.container, *args]

    def _run_code(self, code: str) -> dict[str, Any]:
        return run_limited(
            self._command("python", "-I", "-u", "-c", code),
            timeout=self.settings.timeout,
            output_bytes=self.settings.max_output_bytes,
            process_group=getattr(self, "host_process_group", False),
            env=getattr(self, "host_env", None),
            separate_stderr=True,
            truncate=True,
            preserve_tail=True,
        )

    def _start(self) -> None:
        if self.container is not None:
            return
        name = "areal-math-" + uuid.uuid4().hex
        # No host mounts, credentials, Docker socket, network, or GPU devices.
        self.container = name
        self._control(
            "create",
            "--name",
            name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=64",
            f"--memory={self.settings.memory_mb}m",
            "--memory-swap",
            f"{self.settings.memory_mb}m",
            f"--cpus={self.settings.cpus}",
            "--user=65534:65534",
            "--tmpfs=/work:rw,nosuid,nodev,size=64m,mode=1777",
            "--tmpfs=/tmp:rw,nosuid,nodev,size=32m,mode=1777",
            "--workdir=/work",
            "--env=PYTHONDONTWRITEBYTECODE=1",
            "--entrypoint=python",
            self.image_id,
            "-I",
            "-u",
            "-c",
            "import time; time.sleep(86400)",
        )
        self._control("start", name)

    def close(self) -> None:
        if self.container is not None:
            name, self.container = self.container, None
            self._control("rm", "--force", name)

    def execute(self, code: str) -> str:
        with self._lock:
            started = time.time()
            mark_tool_execution(started)
            if (
                not isinstance(code, str)
                or len(code.encode()) > self.settings.max_code_bytes
            ):
                return json.dumps(
                    {
                        "status": "invalid_code",
                        "error": "Python code is missing or too large",
                    }
                )
            if len(self.records) >= self.settings.max_calls:
                return json.dumps(
                    {"status": "call_limit", "error": "Task tool budget exhausted"}
                )
            name = None
            try:
                self._start()
                name = self.container
                result = self._run_code(code)
                if result["exit_code"] in (125, 126, 127):
                    result["status"] = "infrastructure_error"
                if result["status"] != "completed":
                    self.close()
                    result["environment_reset"] = True
            except (OSError, subprocess.SubprocessError) as exc:
                result = {"status": "infrastructure_error", "error": str(exc)}
            record = {
                "task_id": self.task_id,
                "call": len(self.records) + 1,
                "started": started,
                "finished": time.time(),
                "image_id": self.image_id,
                "container": name,
                "code": code,
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
            return self.execute(arguments.get("code"))
        finally:
            record.update(finished=time.time(), seconds=time.perf_counter() - clock)

    def export_artifacts(self) -> dict[str, Any]:
        return {}

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "runtime_task_id": self.task_id,
            "image_id": self.image_id,
            "calls": self.records,
            "timing_calls": self.call_timings,
        }


def resolve_image(settings: ToolSettings) -> str:
    backend = os.environ.get("AREAL_TASK_RUNTIME", "docker")
    if backend == "singularity":
        from examples.v2_online_workflows.environments.singularity import rpc

        return rpc("resolve", image=settings.image)
    if backend != "docker":
        raise ValueError(f"Unknown task runtime: {backend}")
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for the math Python tool")
    image_id = subprocess.run(
        ["docker", "image", "inspect", "--format={{.Id}}", settings.image],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    if not image_id.startswith("sha256:"):
        raise RuntimeError("Cannot resolve the pre-pulled tool image to a local digest")
    return image_id


def main() -> None:
    """Normal-path Docker preflight; no model or training service is started."""
    import sys

    settings = ToolSettings.from_env()
    from examples.v2_online_workflows.environments.singularity import create_environment

    runtime = create_environment(
        PythonSandbox,
        settings,
        "preflight-" + uuid.uuid4().hex,
        image_id=resolve_image(settings),
    )
    try:
        result = json.loads(runtime.execute("print(6 * 7)"))
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        if (
            result.get("status") != "completed"
            or result.get("output", "").strip() != "42"
        ):
            raise SystemExit(1)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0

"""Frozen SWE-rebench V2 tasks and an adapter to its pinned upstream evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.v2_online_workflows.benchmarks.rebench_evaluator import load_evaluator
from examples.v2_online_workflows.environments.python import resolve_image, run_limited
from examples.v2_online_workflows.environments.repository import RepositorySettings
from examples.v2_online_workflows.types import (
    EnvironmentSpec,
    GradeResult,
    TaskInput,
    TaskResult,
)

DATASET = "nebius/SWE-rebench-V2"
EVALUATOR_URL = "https://github.com/SWE-rebench/SWE-rebench-V2"
EVALUATOR_REVISION = "c71902a8cf8d2b725f63d51f199f4d3e56f68d2d"


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SWERebenchTask:
    record: dict[str, Any]

    def __post_init__(self) -> None:
        for key in (
            "instance_id",
            "repo",
            "base_commit",
            "problem_statement",
            "image_name",
            "test_patch",
        ):
            if (
                not isinstance(self.record.get(key), str)
                or not self.record[key].strip()
            ):
                raise ValueError(f"SWE task requires {key}")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.task_id):
            raise ValueError("SWE instance_id must be a path-safe identifier")
        if not self.record.get("FAIL_TO_PASS") or not isinstance(
            self.record["FAIL_TO_PASS"], list
        ):
            raise ValueError("SWE task requires nonempty FAIL_TO_PASS tests")
        if not isinstance(self.record.get("PASS_TO_PASS"), list):
            raise ValueError("SWE task requires PASS_TO_PASS tests")
        install = self.record.get("install_config", {})
        if not install.get("test_cmd") or not install.get("log_parser"):
            raise ValueError(
                "SWE task requires its upstream test command and log parser"
            )

    @property
    def task_id(self) -> str:
        return self.record["instance_id"]


def load_tasks(path: Path) -> tuple[list[SWERebenchTask], dict[str, Any]]:
    manifest = json.loads((path.parent / "manifest.json").read_text())
    payload = path.read_bytes()
    expected = manifest["splits"][path.name]
    if (
        manifest.get("format_version") != 1
        or manifest.get("benchmark") != "swe_rebench"
        or manifest.get("evaluator")
        != {"repository": EVALUATOR_URL, "revision": EVALUATOR_REVISION}
        or _digest(payload) != expected["sha256"]
    ):
        raise ValueError("SWE task file differs from its frozen manifest")
    tasks = [
        SWERebenchTask(json.loads(line))
        for line in payload.decode().splitlines()
        if line.strip()
    ]
    if len(tasks) != expected["count"] or len({task.task_id for task in tasks}) != len(
        tasks
    ):
        raise ValueError("SWE task count or identity differs from the manifest")
    return tasks, manifest


def write_tasks(records: list[dict], output: Path, source: dict) -> dict:
    tasks = [SWERebenchTask(record) for record in records]
    if not tasks or len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("A nonempty unique SWE task list is required")
    output.mkdir(parents=True, exist_ok=False)
    payload = "".join(
        json.dumps(task.record, ensure_ascii=False) + "\n" for task in tasks
    ).encode()
    (output / "tasks.jsonl").write_bytes(payload)
    manifest = {
        "format_version": 1,
        "benchmark": "swe_rebench",
        "source": source,
        "evaluator": {"repository": EVALUATOR_URL, "revision": EVALUATOR_REVISION},
        "splits": {"tasks.jsonl": {"count": len(tasks), "sha256": _digest(payload)}},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _check_evaluator(root: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if revision != EVALUATOR_REVISION:
        raise ValueError(f"SWE evaluator must be pinned to {EVALUATOR_REVISION}")
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "scripts/eval.py",
            "lib",
        ],
        check=True,
        capture_output=True,
        timeout=10,
    )


class SWERebenchBenchmark:
    @staticmethod
    def restore_task(value: dict[str, Any]) -> SWERebenchTask:
        return SWERebenchTask(**value)

    name = "swe_rebench"
    request_prefix = "swe"
    load_tasks = staticmethod(load_tasks)

    def prepare(self, task: SWERebenchTask) -> TaskInput:
        record = task.record
        return TaskInput(
            task.task_id,
            "Fix the following issue in the provided repository. Inspect and edit source "
            "files using the repository tools; run relevant tests when helpful. "
            "Leave your changes in the working tree. Do not change tests to hide a failure.\n\n"
            + record["problem_statement"],
            EnvironmentSpec(
                "repository",
                record["image_name"],
                {
                    "repo": record["repo"],
                    "base_commit": record["base_commit"],
                },
            ),
        )

    def result_metadata(self, task: SWERebenchTask) -> dict[str, Any]:
        return {"repo": task.record["repo"], "base_commit": task.record["base_commit"]}

    def grade(
        self, task: SWERebenchTask, result: TaskResult, output: Path
    ) -> GradeResult:
        artifacts = result.environment.get("artifacts", {})
        patch = artifacts.get("patch")
        if (
            not isinstance(patch, str)
            or artifacts.get("base_commit") != task.record["base_commit"]
        ):
            raise RuntimeError(
                "SWE grading requires a complete patch from the frozen base commit"
            )
        if not patch.strip():
            # Upstream falls back to the reference patch for missing overrides.
            # A model that produced no patch is an ordinary unsuccessful attempt.
            return GradeResult(0.0, "empty_patch")
        image = result.environment.get("image_id", "")
        if not image.startswith("sha256:"):
            raise RuntimeError(
                "SWE grading requires the agent's resolved image identity"
            )
        root_value = os.environ.get("AREAL_SWE_EVALUATOR_ROOT")
        if not root_value:
            raise ValueError(
                "Set AREAL_SWE_EVALUATOR_ROOT to the pinned upstream checkout"
            )
        root = Path(root_value).resolve()
        _check_evaluator(root)
        output.mkdir(parents=True, exist_ok=True)
        timeout = float(os.environ.get("AREAL_SWE_GRADING_TIMEOUT", "600"))
        if timeout <= 0:
            raise ValueError("SWE grading timeout must be positive")
        name = "areal-swe-grade-" + uuid.uuid4().hex
        job = {
            "task": task.record,
            "patch": patch,
            "image": image,
            "evaluator_root": str(root),
            "output": str(output.resolve()),
            "timeout": timeout,
            "container": name,
        }
        job_path = output / "grade-job.json"
        job_path.write_text(json.dumps(job))
        try:
            env = dict(os.environ)
            env["PYTHONPATH"] = (
                str(Path(__file__).resolve().parents[3])
                + os.pathsep
                + env.get("PYTHONPATH", "")
            )
            with (output / "grade-worker.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "examples.v2_online_workflows.benchmarks.swe_rebench",
                        "grade-worker",
                        str(job_path.resolve()),
                    ],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout + 120,
                    env=env,
                )
        except subprocess.TimeoutExpired:
            if os.environ.get("AREAL_TASK_RUNTIME", "docker") == "docker":
                subprocess.run(
                    ["docker", "rm", "--force", name], capture_output=True, timeout=30
                )
            raise
        report = json.loads((output / "grade-report.json").read_text())
        return GradeResult(
            float(report["passed_match"]),
            "resolved" if report["passed_match"] else "unresolved",
            details=report,
        )


def _grade_worker(job_path: Path) -> None:
    """Use upstream parsing/scoring in a child process; bound only its Docker IO."""
    job = json.loads(job_path.read_text())
    root, output = Path(job["evaluator_root"]), Path(job["output"])
    _check_evaluator(root)
    module = load_evaluator(root)
    module.LOGS_DIR = output / "logs"

    def run_in_container(
        image, workdir, patch_dir, patch_name, test_patch_name, test_cmds
    ):
        if os.environ.get("AREAL_TASK_RUNTIME", "docker") == "singularity":
            from examples.v2_online_workflows.environments.singularity import rpc

            return rpc(
                "grade",
                task_id=job["container"],
                image=image,
                repo=job["task"]["repo"],
                base_commit=job["task"]["base_commit"],
                patch=(patch_dir / patch_name).read_text(),
                test_patch=(patch_dir / test_patch_name).read_text(),
                test_cmds=test_cmds,
                timeout=job["timeout"],
            )
        name = job["container"]

        def docker(*args):
            return subprocess.run(
                ["docker", *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

        try:
            docker(
                "create",
                "--name",
                name,
                "--network=none",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=512",
                "--memory=" + os.environ.get("AREAL_TASK_REPO_MEMORY_MB", "4096") + "m",
                "--cpus=" + os.environ.get("AREAL_TASK_REPO_CPUS", "2"),
                "-e",
                "_JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false",
                "--workdir",
                workdir,
                "--entrypoint=/bin/bash",
                image,
                "-c",
                "while :; do sleep 3600; done",
            )
            docker("start", name)
            docker("exec", name, "mkdir", "-p", "/patches")
            # docker cp also works when the daemon is outside the client container.
            docker("cp", str(patch_dir) + "/.", name + ":/patches")
            script = "\n".join(
                [
                    "set -e",
                    "git reset --hard " + job["task"]["base_commit"],
                    f"git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/{patch_name}",
                    f"git apply -v --3way --recount --ignore-space-change --whitespace=nowarn /patches/{test_patch_name}",
                    *test_cmds,
                ]
            )
            result = run_limited(
                [
                    "docker",
                    "exec",
                    name,
                    "timeout",
                    "--kill-after=5s",
                    str(job["timeout"]),
                    "bash",
                    "-c",
                    script,
                ],
                timeout=job["timeout"] + 10,
                output_bytes=64 * 1024 * 1024,
                truncate=True,
            )
            if result["status"] != "completed" or result["exit_code"] in (124, 125):
                raise RuntimeError("SWE test execution could not be scored completely")
            return result["exit_code"], result["output"]
        finally:
            subprocess.run(
                ["docker", "rm", "--force", name], capture_output=True, timeout=30
            )

    module.run_in_container = run_in_container
    report = module.evaluate_instance(
        job["task"], job["image"], {"patch": job["patch"]}
    )
    (output / "grade-report.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--revision", default="main")
    prepare.add_argument("--language", default="python")
    prepare.add_argument("--count", type=int, default=16)
    prepare.add_argument("--seed", type=int, default=1)
    prepare.add_argument("--source-jsonl", type=Path)
    worker = commands.add_parser("grade-worker")
    worker.add_argument("job", type=Path)
    reference = commands.add_parser("grade-reference")
    reference.add_argument("--tasks", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    images = commands.add_parser("pull-images")
    images.add_argument("--tasks", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "grade-worker":
        _grade_worker(args.job)
        return
    if args.command == "pull-images":
        tasks, _ = load_tasks(args.tasks)
        for image in sorted({task.record["image_name"] for task in tasks}):
            subprocess.run(["docker", "pull", image], check=True)
        return
    if args.command == "grade-reference":
        tasks, _ = load_tasks(args.tasks)
        benchmark = SWERebenchBenchmark()
        for task in tasks:
            image = resolve_image(
                RepositorySettings.from_env(task.record["image_name"])
            )
            result = TaskResult(
                "",
                True,
                {
                    "image_id": image,
                    "artifacts": {
                        "patch": task.record["patch"],
                        "base_commit": task.record["base_commit"],
                    },
                },
            )
            grade = benchmark.grade(task, result, args.output / task.task_id)
            sys.stdout.write(
                json.dumps({"task_id": task.task_id, "reward": grade.reward}) + "\n"
            )
        return
    if args.count <= 0:
        raise ValueError("Task count must be positive")
    if args.source_jsonl:
        payload = args.source_jsonl.read_bytes()
        rows = [
            json.loads(line) for line in payload.decode().splitlines() if line.strip()
        ]
        source = {"file_sha256": _digest(payload)}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi

        revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
        if not revision:
            raise RuntimeError("Cannot resolve an immutable SWE dataset revision")
        rows = load_dataset(DATASET, split="train", revision=revision, streaming=True)
        source = {"dataset": DATASET, "revision": revision, "split": "train"}
    # Keep the lowest seeded hashes, independent of input iteration order.
    selected = []
    excluded = 0
    for record in rows:
        if record.get("language") != args.language:
            continue
        try:
            task = SWERebenchTask(record)
        except ValueError:
            excluded += 1
            continue
        rank = _digest(f"{args.seed}:{task.task_id}".encode())
        selected.append((rank, task.task_id, record))
        selected.sort(key=lambda item: (item[0], item[1]))
        del selected[args.count :]
    if len(selected) != args.count:
        raise ValueError("Requested task count exceeds the selected language pool")
    source.update(
        language=args.language,
        seed=args.seed,
        selection="lowest_seeded_task_hash",
        schema_excluded=excluded,
    )
    manifest = write_tasks([item[2] for item in selected], args.output, source)
    sys.stdout.write(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

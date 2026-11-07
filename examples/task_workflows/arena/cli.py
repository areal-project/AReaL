# SPDX-License-Identifier: Apache-2.0
"""Check a pinned combination, run episodes, or resume their existing identities."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import shutil
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

import httpx

from examples.task_workflows.arena.backend import (
    ArenaCLI,
    ArenaExecutor,
    check_combination,
    write_private,
)
from examples.task_workflows.arena.config import ArenaConfig, digest
from examples.task_workflows.benchmarks.dapo_math import DapoBenchmark, MathTask
from examples.task_workflows.client import ClientSettings, EpisodeClient


def selection(config: ArenaConfig, tasks: Path, limit: int | None):
    benchmark = DapoBenchmark()
    rows, manifest = benchmark.load_tasks(tasks)
    for key in ("dataset", "revision", "config", "split"):
        if manifest["source"].get(key) != config.value["dataset"]["source"].get(key):
            raise ValueError("Task source differs from the published pool")
    if limit is not None:
        if limit < 1 or limit > len(rows):
            raise ValueError("Limit must fit the frozen task file")
        rows = rows[:limit]
    for task in rows:
        config.match(task.task_id, task.prompt())
    return rows


async def run(args: argparse.Namespace) -> int:
    benchmark = DapoBenchmark()
    if args.command == "resume":
        output = args.run_dir.resolve()
        run_info = json.loads((output / "run.json").read_text())
        config = ArenaConfig.load(
            output / "config.json", task_index=output / "task-index.jsonl"
        )
        rows = json.loads((output / "tasks.json").read_text())
        if (
            digest(rows) != run_info["tasks_sha256"]
            or config.fingerprint != run_info["configuration_sha256"]
        ):
            raise ValueError("Saved run inputs differ from their digests")
        tasks = [MathTask(**row) for row in rows]
        options = run_info["options"]
    else:
        config = ArenaConfig.load(args.config)
        tasks = selection(config, args.tasks, args.limit) if args.tasks else []
        if args.command == "check":
            result = await check_combination(
                config, ArenaCLI(config), [benchmark.prepare(t) for t in tasks]
            )
            sys.stdout.write(json.dumps(result, indent=2) + "\n")
            return 0
        if not tasks:
            raise ValueError("Run requires a frozen task file")
        if args.mode != "eval" and config.value["model_mode"] != "external":
            raise ValueError("Mock configurations are evaluation-only")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        options = {
            "mode": args.mode,
            "inference_url": args.inference_url.rstrip("/"),
            "model": args.model,
            "inference_key_env": args.inference_key_env,
            "concurrency": args.concurrency,
            "request_timeout": args.request_timeout,
        }
        run_info = {
            "format_version": 1,
            "configuration_sha256": config.fingerprint,
            "options": options,
            "tasks_sha256": digest([asdict(t) for t in tasks]),
            "request_ids": [
                benchmark.request_prefix + "-" + uuid.uuid4().hex for _ in tasks
            ],
        }
        write_private(output / "run.json", run_info)
        write_private(output / "config.json", config.value)
        write_private(output / "tasks.json", [asdict(t) for t in tasks])
        shutil.copyfile(
            config.path.parent / config.value["dataset"]["task_index"],
            output / "task-index.jsonl",
        )
        (output / "task-index.jsonl").chmod(0o600)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if options["mode"] != "eval" and config.value["model_mode"] != "external":
            raise ValueError("Mock samples cannot enter training")
        settings = ClientSettings(
            agent_url="",
            agent_key="",
            inference_url=options["inference_url"],
            inference_key=os.environ[options["inference_key_env"]],
            model=options["model"],
            mode=options["mode"],
            output=output,
            concurrency=options["concurrency"],
            request_timeout=options["request_timeout"],
            backend="arena",
        )
        api = ArenaCLI(config)
        check = await check_combination(
            config, api, [benchmark.prepare(t) for t in tasks]
        )
        write_private(output / "check.json", check)
        executor = ArenaExecutor(
            config, api, output, identity=options, timeout=settings.request_timeout
        )
        async with httpx.AsyncClient(timeout=settings.request_timeout) as http:
            summary = await EpisodeClient(
                settings, http, benchmark, executor=executor
            ).run(tasks, request_ids=run_info["request_ids"])
        summary.update(
            configuration_sha256=config.fingerprint,
            model_mode=config.value["model_mode"],
            validation_scope="Execution and protocol receipts; Actor consumption requires independent verification",
        )
        write_private(output / "summary.json", summary)
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
        return int(bool(summary["infrastructure_errors"] or summary["not_submitted"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "run"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--tasks", type=Path)
        command.add_argument("--limit", type=int)
    command = commands.choices["run"]
    command.add_argument("--output", type=Path, required=True)
    command.add_argument(
        "--inference-url", required=True, help="Model/gateway origin, without /v1"
    )
    command.add_argument("--model", required=True)
    command.add_argument("--inference-key-env", default="AREAL_INFERENCE_ADMIN_KEY")
    command.add_argument("--mode", choices=("eval", "resident"), default="eval")
    command.add_argument("--concurrency", type=int, default=4)
    command.add_argument("--request-timeout", type=float, default=900)
    command = commands.add_parser("resume")
    command.add_argument("--run-dir", type=Path, required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))


if __name__ == "__main__":
    main()

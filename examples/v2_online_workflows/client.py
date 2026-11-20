# SPDX-License-Identifier: Apache-2.0

"""Send benchmark tasks through a harness with configurable concurrency."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import multiprocessing
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from examples.v2_online_workflows.artifacts import write_json
from examples.v2_online_workflows.types import Harness, get_benchmark


class EpisodeClient:
    def __init__(
        self,
        *,
        http: httpx.AsyncClient,
        harness: Harness,
        benchmark: Any,
        inference_url: str,
        inference_key: str,
        model: str,
        output: Path,
        concurrency: int,
        grade_pool=None,
    ):
        if concurrency < 1:
            raise ValueError("Concurrency must be positive")
        self.http, self.harness, self.benchmark = http, harness, benchmark
        self.inference_url = inference_url.rstrip("/")
        self.inference_key, self.model = inference_key, model
        self.output, self.concurrency = output, concurrency
        self.grade_pool = grade_pool

    async def _start_session(self, request_id: str) -> dict:
        while True:
            response = await self.http.post(
                f"{self.inference_url}/rl/start_session",
                headers={"Authorization": f"Bearer {self.inference_key}"},
                json={"task_id": request_id},
            )
            if response.status_code == 429 and response.json().get("detail") == {
                "code": "online_capacity_exhausted"
            }:
                await asyncio.sleep(1.0)
                continue
            response.raise_for_status()
            sessions = response.json()["sessions"]
            if len(sessions) != 1:
                raise RuntimeError("Each task requires exactly one RL session")
            return sessions[0]

    async def _report_execution_failure(self, session: dict, row: dict) -> None:
        try:
            response = await self.http.post(
                f"{self.inference_url}/rl/set_reward",
                headers={"Authorization": f"Bearer {session['session_api_key']}"},
                json={"execution_failed": True},
            )
            response.raise_for_status()
            receipt = response.json()
            if (
                receipt.get("message") != "execution_failed"
                or receipt.get("session_id") != session["session_id"]
            ):
                raise RuntimeError("Failure receipt does not confirm this session")
            row["failure_receipt"] = receipt
        except Exception as exc:
            row["failure_report_error_type"] = type(exc).__name__

    async def run_task(self, task, request_id: str) -> dict:
        row = {
            "task_id": task.task_id,
            "request_id": request_id,
            "started": time.time(),
        }
        agent_returned = False
        try:
            row["phase"] = "start_session"
            session = await self._start_session(request_id)
            row["session_id"] = session["session_id"]
            row["phase"] = "agent"
            result = await self.harness.run(
                self.benchmark.prepare(task),
                request_id=request_id,
                inference_url=self.inference_url,
                model=self.model,
                session_api_key=session["session_api_key"],
            )
            agent_returned = True
            row.update(
                answer=result.answer, metadata=result.metadata, answered=time.time()
            )
            if not result.completed:
                row["status"] = "execution_failed"
                await self._report_execution_failure(session, row)
                return row
            row["phase"] = "grade"
            score = await asyncio.get_running_loop().run_in_executor(
                self.grade_pool,
                self.benchmark.grade,
                task,
                result,
                self.output / request_id,
            )
            if not math.isfinite(score.reward):
                raise ValueError("Benchmark returned a nonfinite reward")
            row["score"] = asdict(score)
            row["phase"] = "reward"
            response = await self.http.post(
                f"{self.inference_url}/rl/set_reward",
                headers={"Authorization": f"Bearer {session['session_api_key']}"},
                json={"reward": score.reward},
            )
            response.raise_for_status()
            receipt = response.json()
            if (
                receipt.get("message") != "success"
                or receipt.get("session_id") != row["session_id"]
            ):
                raise RuntimeError("Reward receipt does not confirm this session")
            row.update(reward_receipt=receipt, status="completed", phase="finished")
        except Exception as exc:
            # Record the operation and exception type without credential-bearing bodies.
            if agent_returned and row["phase"] == "grade":
                await self._report_execution_failure(session, row)
            row.update(status="infrastructure_error", error_type=type(exc).__name__)
            if isinstance(exc, httpx.HTTPStatusError):
                row["http_status"] = exc.response.status_code
        finally:
            if agent_returned:
                try:
                    await self.harness.close(request_id)
                except Exception as exc:
                    row["cleanup_error_type"] = type(exc).__name__
            row["finished"] = time.time()
        return row

    async def run(self, tasks: list) -> dict:
        self.output.mkdir(parents=True, exist_ok=False, mode=0o700)
        available_at = time.time()
        requests = [
            {"request_id": uuid.uuid4().hex, "task_id": task.task_id} for task in tasks
        ]
        write_json(
            self.output / "requests.json",
            {
                "available_at": available_at,
                "concurrency": self.concurrency,
                "requests": requests,
            },
        )
        pending = iter(zip(tasks, requests))
        rows = []

        async def worker():
            for task, request in pending:
                row = await self.run_task(task, request["request_id"])
                row["available_at"] = available_at
                row["queue_seconds"] = row["started"] - available_at
                # Each append completes on the event-loop thread before yielding.
                with (self.output / "episodes.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)

        await asyncio.gather(
            *(worker() for _ in range(min(self.concurrency, len(tasks))))
        )
        completed = [row for row in rows if row["status"] == "completed"]
        summary = {
            "tasks": len(rows),
            "completed": len(completed),
            "correct": sum(row["score"]["reward"] == 1 for row in completed),
            "execution_failed": sum(
                row["status"] == "execution_failed" for row in rows
            ),
            "infrastructure_errors": sum(
                row["status"] == "infrastructure_error" for row in rows
            ),
            "cleanup_errors": sum("cleanup_error_type" in row for row in rows),
        }
        write_json(self.output / "summary.json", summary)
        return summary


async def _main(args):
    benchmark = get_benchmark(args.benchmark)
    tasks, manifest = benchmark.load_tasks(args.tasks)
    if not tasks or not manifest.get("training_allowed", True):
        raise ValueError("Provide a nonempty training split")
    if args.output.exists():
        raise FileExistsError("Choose a new output directory for each submission")
    if args.request_timeout is not None and (
        not math.isfinite(args.request_timeout) or args.request_timeout <= 0
    ):
        raise ValueError("Request timeout must be positive when supplied")
    agent_key = os.environ["AREAL_AGENT_ADMIN_KEY"]
    inference_key = os.environ["AREAL_INFERENCE_ADMIN_KEY"]
    module_name, class_name = args.harness.rsplit(".", 1)
    harness_class = getattr(importlib.import_module(module_name), class_name)
    async with httpx.AsyncClient(
        timeout=args.request_timeout,
        trust_env=False,
        limits=httpx.Limits(max_connections=2 * args.concurrency),
    ) as http:
        harness = harness_class(
            http=http, agent_url=args.agent_url, agent_key=agent_key
        )
        with ProcessPoolExecutor(
            max_workers=args.concurrency,
            mp_context=multiprocessing.get_context("spawn"),
        ) as grade_pool:
            client = EpisodeClient(
                http=http,
                harness=harness,
                benchmark=benchmark,
                inference_url=args.inference_url,
                inference_key=inference_key,
                model=args.model,
                output=args.output,
                concurrency=args.concurrency,
                grade_pool=grade_pool,
            )
            summary = await client.run(tasks)
    print(json.dumps(summary))
    return int(
        bool(
            summary["infrastructure_errors"]
            or summary["execution_failed"]
            or summary["cleanup_errors"]
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("gsm8k", "dapo"), required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-url", required=True)
    parser.add_argument("--inference-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--request-timeout", type=float)
    parser.add_argument(
        "--harness",
        default="examples.v2_online_workflows.harnesses.hermes.HermesHarness",
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    raise SystemExit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()

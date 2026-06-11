#!/usr/bin/env python3
"""
Online Trajectory Collector

Runs tau2-bench tasks through AReaL Inference Service, collecting trajectories
with logprobs for RL training. Each concurrent worker gets its own IS session.

Flow per worker:
  1. POST /rl/start_session → session_api_key
  2. Run tau2 task (OpenClaw CLI with dynamic api_key → IS Gateway → DataProxy → SGLang)
  3. POST /rl/set_reward {reward}
  4. POST /export_trajectories → save trajectory to disk

Usage:
  python collect_trajectories.py \
      --gateway-url http://127.0.0.1:30098 \
      --admin-api-key "dummy:0" \
      --user-endpoint http://<node>:30001/v1 \
      --domain airline \
      --concurrency 5 \
      --num-tasks 50 \
      --max-steps 200 \
      --output-dir ./trajectories
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

try:
    from tau2.data_model.tasks import Task  # noqa: F401
    from tau2.evaluator.evaluator import EvaluationType  # noqa: F401
    from tau2.run import load_tasks
except ImportError:
    print("ERROR: tau2 not installed. Run: pip install -e /path/to/tau2-bench")
    sys.exit(1)

WORKER_SCRIPT = str(Path(__file__).resolve().parent / "worker.py")


async def grant_capacity(
    client: httpx.AsyncClient,
    gateway_url: str,
    admin_key: str,
) -> None:
    resp = await client.post(
        f"{gateway_url}/grant_capacity",
        headers={"Authorization": f"Bearer {admin_key}"},
    )
    resp.raise_for_status()


async def start_session(
    client: httpx.AsyncClient,
    gateway_url: str,
    admin_key: str,
    task_id: str,
) -> dict:
    await grant_capacity(client, gateway_url, admin_key)
    resp = await client.post(
        f"{gateway_url}/rl/start_session",
        json={"task_id": task_id},
        headers={"Authorization": f"Bearer {admin_key}"},
    )
    resp.raise_for_status()
    return resp.json()


async def set_reward(
    client: httpx.AsyncClient,
    gateway_url: str,
    session_api_key: str,
    reward: float,
) -> dict:
    resp = await client.post(
        f"{gateway_url}/rl/set_reward",
        json={"reward": reward},
        headers={"Authorization": f"Bearer {session_api_key}"},
    )
    resp.raise_for_status()
    return resp.json()


async def export_trajectories(
    client: httpx.AsyncClient,
    gateway_url: str,
    admin_key: str,
    session_id: str,
) -> dict:
    resp = await client.post(
        f"{gateway_url}/export_trajectories",
        json={"session_id": session_id},
        headers={"Authorization": f"Bearer {admin_key}"},
    )
    resp.raise_for_status()
    return resp.json()


def _ensure_agent_home(agent_home: Path, gateway_url: str, api_key: str, model: str):
    oc_dir = agent_home / ".openclaw"
    (oc_dir / "workspace").mkdir(parents=True, exist_ok=True)
    (oc_dir / "agents" / "main" / "agent").mkdir(parents=True, exist_ok=True)
    oc_config = {
        "models": {
            "providers": {
                "sglang": {
                    "baseUrl": gateway_url,
                    "apiKey": api_key,
                    "api": "openai-completions",
                    "models": [{"id": model, "name": model}],
                }
            }
        }
    }
    with open(oc_dir / "openclaw.json", "w") as f:
        json.dump(oc_config, f, indent=2)


def run_tau2_task_subprocess(
    domain: str,
    task_index: int,
    gateway_url: str,
    session_api_key: str,
    user_endpoint: str,
    model_name: str,
    max_steps: int,
    max_errors: int,
    seed: int | None,
    openclaw_cli: str,
    openclaw_timeout: int,
    worker_id: int,
    work_dir: Path,
) -> dict:
    """Run tau2 task via run_single_worker.py subprocess (same pattern as v2).

    HOME is set via env prefix so each worker gets an isolated OpenClaw home.
    """
    agent_home = work_dir / f"agent_{worker_id}"
    results_dir = work_dir / f"results_{worker_id}"
    log_file = work_dir / f"worker_{worker_id}_task_{task_index}.log"
    results_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HOME"] = str(agent_home)
    env["OPENCLAW_CLI_COMMAND"] = openclaw_cli
    env["OPENCLAW_API_BASE"] = gateway_url
    env["OPENCLAW_API_KEY"] = session_api_key
    env["OPENCLAW_MODEL"] = model_name
    env["OPENCLAW_TIMEOUT"] = str(openclaw_timeout)
    env["OPENAI_API_BASE"] = user_endpoint
    env["OPENAI_API_KEY"] = "dummy"

    cmd = [
        sys.executable,
        WORKER_SCRIPT,
        "--domain",
        domain,
        "--task-index",
        str(task_index),
        "--agent-endpoint",
        gateway_url,
        "--user-endpoint",
        user_endpoint,
        "--model",
        model_name,
        "--user-llm",
        f"openai/{model_name}",
        "--max-steps",
        str(max_steps),
        "--max-errors",
        str(max_errors),
        "--output-dir",
        str(results_dir),
        "--worker-id",
        str(worker_id),
        "--user-llm-args",
        json.dumps({"temperature": 0.0}),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    with open(log_file, "a") as lf:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=openclaw_timeout + 120,
            env=env,
        )
        lf.write(result.stderr)

    if not result.stdout.strip():
        raise RuntimeError(
            f"Worker-{worker_id} returned no stdout (exit={result.returncode}). "
            f"Check {log_file}"
        )

    return json.loads(result.stdout.strip())


async def worker(
    worker_id: int,
    task_queue: asyncio.Queue,
    gateway_url: str,
    admin_key: str,
    user_endpoint: str,
    model_name: str,
    domain: str,
    max_steps: int,
    max_errors: int,
    seed: int | None,
    openclaw_cli: str,
    openclaw_timeout: int,
    output_dir: Path,
    work_dir: Path,
    results: list,
):
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0)) as client:
        while True:
            try:
                task_idx, task = task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            task_start = time.time()
            session_info = None

            try:
                # 1. Start IS session
                print(
                    f"  [worker-{worker_id}] task={task.id} (idx={task_idx}) → start_session"
                )
                session_info = await start_session(
                    client, gateway_url, admin_key, task.id
                )
                session_api_key = session_info["api_key"]
                session_id = session_info["session_id"]
                print(f"  [worker-{worker_id}] session={session_id[:12]}...")

                _ensure_agent_home(
                    work_dir / f"agent_{worker_id}",
                    gateway_url,
                    session_api_key,
                    model_name,
                )

                loop = asyncio.get_event_loop()
                task_result = await loop.run_in_executor(
                    None,
                    run_tau2_task_subprocess,
                    domain,
                    task_idx,
                    gateway_url,
                    session_api_key,
                    user_endpoint,
                    model_name,
                    max_steps,
                    max_errors,
                    seed,
                    openclaw_cli,
                    openclaw_timeout,
                    worker_id,
                    work_dir,
                )

                reward = task_result.get("reward", 0.0)

                print(
                    f"  [worker-{worker_id}] task={task.id} reward={reward} → set_reward"
                )
                sr_resp = await set_reward(client, gateway_url, session_api_key, reward)
                print(
                    f"  [worker-{worker_id}] set_reward → {sr_resp.get('message', 'ok')}"
                )

                print(f"  [worker-{worker_id}] task={task.id} → export_trajectories")
                trajectory_data = await export_trajectories(
                    client, gateway_url, admin_key, session_id
                )

                traj_file = output_dir / f"task_{task.id}_session_{session_id}.json"
                with open(traj_file, "w") as f:
                    json.dump(
                        {
                            "task_id": task.id,
                            "session_id": session_id,
                            "reward": reward,
                            "duration": time.time() - task_start,
                            "num_turns": task_result.get("num_steps", 0),
                            "termination_reason": task_result.get("termination_reason"),
                            "trajectory": trajectory_data,
                        },
                        f,
                        indent=2,
                        default=str,
                    )

                symbol = "✔" if reward > 0 else "✘"
                elapsed = time.time() - task_start
                print(
                    f"  {symbol} [worker-{worker_id}] task={task.id} reward={reward} dur={elapsed:.1f}s saved={traj_file.name}"
                )

                results.append(
                    {
                        "task_id": task.id,
                        "session_id": session_id,
                        "reward": reward,
                        "duration": elapsed,
                        "status": "ok",
                    }
                )

            except Exception as e:
                elapsed = time.time() - task_start
                print(
                    f"  ✗ [worker-{worker_id}] task={task.id} ERROR: {e} dur={elapsed:.1f}s"
                )
                results.append(
                    {
                        "task_id": task.id,
                        "session_id": session_info["session_id"]
                        if session_info
                        else None,
                        "reward": 0.0,
                        "duration": elapsed,
                        "status": f"error: {e}",
                    }
                )

            finally:
                task_queue.task_done()


async def run_collection(args):
    """Main collection loop."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "workdir"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Load tau2 tasks
    print(f"Loading {args.domain} tasks...")
    tasks = load_tasks(task_set_name=args.domain)
    if args.num_tasks != "all":
        tasks = tasks[: int(args.num_tasks)]
    print(f"  {len(tasks)} tasks loaded")

    # Build task queue
    task_queue: asyncio.Queue = asyncio.Queue()
    for idx, task in enumerate(tasks):
        task_queue.put_nowait((idx, task))

    results: list = []

    print("\nStarting trajectory collection:")
    print(f"  Gateway: {args.gateway_url}")
    print(f"  Domain: {args.domain}")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Output: {output_dir}")
    print()

    start_time = time.time()

    # Launch N workers
    workers = [
        asyncio.create_task(
            worker(
                worker_id=i,
                task_queue=task_queue,
                gateway_url=args.gateway_url,
                admin_key=args.admin_api_key,
                user_endpoint=args.user_endpoint,
                model_name=args.model,
                domain=args.domain,
                max_steps=args.max_steps,
                max_errors=args.max_errors,
                seed=args.seed,
                openclaw_cli=args.openclaw_cli,
                openclaw_timeout=args.openclaw_timeout,
                output_dir=output_dir,
                work_dir=work_dir,
                results=results,
            )
        )
        for i in range(args.concurrency)
    ]

    await asyncio.gather(*workers)

    total_time = time.time() - start_time
    n_pass = sum(1 for r in results if r["reward"] > 0)
    n_fail = sum(1 for r in results if r["reward"] == 0 and r["status"] == "ok")
    n_error = sum(1 for r in results if "error" in r["status"])

    # Save summary
    summary = {
        "domain": args.domain,
        "concurrency": args.concurrency,
        "total_tasks": len(tasks),
        "completed": len(results),
        "passed": n_pass,
        "failed": n_fail,
        "errors": n_error,
        "pass_rate": n_pass / max(len(results), 1),
        "total_time_s": total_time,
        "tasks_per_min": len(results) / (total_time / 60) if total_time > 0 else 0,
        "avg_duration_s": sum(r["duration"] for r in results) / max(len(results), 1),
        "results": results,
    }

    summary_file = output_dir / "collection_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Collection Complete")
    print(f"{'=' * 60}")
    print(f"  Tasks: {len(results)}/{len(tasks)}")
    print(f"  Pass:  {n_pass} ({summary['pass_rate']:.1%})")
    print(f"  Fail:  {n_fail}")
    print(f"  Error: {n_error}")
    print(f"  Time:  {total_time:.0f}s ({summary['tasks_per_min']:.1f} tasks/min)")
    print(f"  Output: {output_dir}")
    print(f"  Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect trajectories via AReaL Inference Service"
    )
    parser.add_argument(
        "--gateway-url",
        required=True,
        help="IS Gateway URL (e.g. http://127.0.0.1:30098)",
    )
    parser.add_argument("--admin-api-key", default="dummy:0", help="IS admin API key")
    parser.add_argument("--user-endpoint", required=True, help="User sim LLM endpoint")
    parser.add_argument(
        "--model", default="Qwen3-235B-A22B-Instruct-2507", help="Model name"
    )
    parser.add_argument("--domain", default="airline", help="tau2 domain")
    parser.add_argument(
        "--concurrency", type=int, default=5, help="Number of concurrent workers"
    )
    parser.add_argument("--num-tasks", default="all", help="Number of tasks (or 'all')")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per task")
    parser.add_argument(
        "--max-errors", type=int, default=10, help="Max errors per task"
    )
    parser.add_argument("--seed", type=int, default=300, help="Random seed")
    parser.add_argument("--openclaw-cli", default="openclaw", help="OpenClaw CLI path")
    parser.add_argument(
        "--openclaw-timeout",
        type=int,
        default=600,
        help="OpenClaw subprocess timeout (s)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save trajectories"
    )
    args = parser.parse_args()

    asyncio.run(run_collection(args))


if __name__ == "__main__":
    main()

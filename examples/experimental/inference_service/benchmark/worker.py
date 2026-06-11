#!/usr/bin/env python3
"""
Single TAU²-Bench worker — runs one task with OpenClaw agent via socket server.

Usage:
    python run_single_worker.py \
        --domain retail \
        --task-index 0 \
        --agent-endpoint http://127.0.0.1:30000/v1 \
        --user-endpoint http://<node>:30001/v1 \
        --model Qwen3-235B-A22B-Instruct-2507 \
        --output-dir /tmp/results

Output: writes <output-dir>/task_<id>.json with trajectory + reward.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run single TAU²-Bench task with OpenClaw"
    )
    parser.add_argument("--domain", type=str, default="retail")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--agent-endpoint", type=str, required=True)
    parser.add_argument("--user-endpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen3-235B-A22B-Instruct-2507")
    parser.add_argument("--user-llm", type=str, default="")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--user-llm-args",
        type=str,
        default="",
        help="JSON string of extra llm_args for user simulator, e.g. '{\"temperature\":0.0}'",
    )
    args = parser.parse_args()

    os.environ["OPENCLAW_CLI_COMMAND"] = "openclaw"
    os.environ["OPENCLAW_API_BASE"] = args.agent_endpoint
    os.environ["OPENCLAW_API_KEY"] = os.environ.get("OPENCLAW_API_KEY", "dummy")
    os.environ["OPENCLAW_MODEL"] = args.model
    os.environ["OPENAI_API_BASE"] = args.user_endpoint
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy")

    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    import litellm

    litellm.drop_params = True
    litellm.suppress_debug_info = True

    src_path = str(Path(__file__).resolve().parent.parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    tau2_src = os.environ.get("TAU2_SRC_PATH", "${TAU2_DIR}/src")
    if os.path.isdir(tau2_src) and tau2_src not in sys.path:
        sys.path.insert(0, tau2_src)

    import subprocess

    from openclaw_tau2 import run_task_with_socket_server
    from tau2.registry import registry
    from tau2.run import load_tasks

    openclaw_version = "unknown"
    try:
        r = subprocess.run(
            ["openclaw", "--version"], capture_output=True, text=True, timeout=5
        )
        openclaw_version = (
            r.stdout.strip() if r.returncode == 0 else f"exit={r.returncode}"
        )
    except Exception as e:
        openclaw_version = f"not found: {e}"

    agent_constructor = registry.get_agent_constructor("openclaw_agent")
    agent_class_name = f"{agent_constructor.__module__}.{agent_constructor.__name__}"

    print("=" * 72, file=sys.stderr)
    print("  AGENT VERIFICATION", file=sys.stderr)
    print("  Agent type:       openclaw_agent", file=sys.stderr)
    print(f"  Agent class:      {agent_class_name}", file=sys.stderr)
    print(f"  OpenClaw CLI:     {openclaw_version}", file=sys.stderr)
    print(f"  Agent endpoint:   {args.agent_endpoint}", file=sys.stderr)
    print(f"  Agent model:      {args.model}", file=sys.stderr)
    print(f"  HOME:             {os.environ.get('HOME', '?')}", file=sys.stderr)
    print(f"  Worker ID:        {args.worker_id}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    tasks = load_tasks(task_set_name=args.domain)
    if args.task_index >= len(tasks):
        print(
            json.dumps({"error": f"task_index {args.task_index} >= {len(tasks)} tasks"})
        )
        sys.exit(1)

    task = tasks[args.task_index]
    user_llm = args.user_llm or f"openai/{args.model}"

    llm_args_user = None
    if args.user_llm_args:
        try:
            llm_args_user = json.loads(args.user_llm_args)
        except json.JSONDecodeError as e:
            print(f"WARNING: Failed to parse --user-llm-args: {e}", file=sys.stderr)

    tag = f"[worker-{args.worker_id}][task-{task.id}]"
    print(f"{tag} Starting: domain={args.domain} task={task.id}", file=sys.stderr)

    start = time.time()
    try:
        simulation, environment = run_task_with_socket_server(
            domain=args.domain,
            task=task,
            agent="openclaw_agent",
            user="user_simulator",
            llm_user=user_llm,
            llm_args_user=llm_args_user,
            max_steps=args.max_steps,
            max_errors=args.max_errors,
            seed=args.seed,
            socket_port=None,
        )

        duration = time.time() - start
        reward = simulation.reward_info.reward if simulation.reward_info else 0.0

        messages = []
        for msg in simulation.messages:
            messages.append(
                {
                    "role": msg.role
                    if hasattr(msg, "role")
                    else str(type(msg).__name__),
                    "content": msg.content if msg.content else "",
                }
            )

        reward_detail = {}
        if simulation.reward_info:
            reward_detail["reward"] = simulation.reward_info.reward
            if simulation.reward_info.db_check:
                reward_detail["db_match"] = simulation.reward_info.db_check.db_match
            if simulation.reward_info.action_checks:
                matched = sum(
                    1 for ac in simulation.reward_info.action_checks if ac.action_match
                )
                reward_detail["action_checks"] = (
                    f"{matched}/{len(simulation.reward_info.action_checks)}"
                )

        result = {
            "task_id": str(task.id),
            "task_index": args.task_index,
            "domain": args.domain,
            "worker_id": args.worker_id,
            "model": args.model,
            "reward": reward,
            "reward_detail": reward_detail,
            "num_steps": len(simulation.messages),
            "termination_reason": simulation.termination_reason.value,
            "duration_seconds": round(duration, 1),
            "messages": messages,
        }

        status = "✓" if reward > 0 else "✗"
        print(
            f"{tag} {status} reward={reward:.2f} steps={len(simulation.messages)} dur={duration:.1f}s term={simulation.termination_reason.value}",
            file=sys.stderr,
        )

    except Exception as e:
        import traceback

        duration = time.time() - start
        error_type = type(e).__name__
        result = {
            "task_id": str(task.id),
            "task_index": args.task_index,
            "domain": args.domain,
            "worker_id": args.worker_id,
            "model": args.model,
            "reward": 0.0,
            "num_steps": 0,
            "termination_reason": "error",
            "error_type": error_type,
            "duration_seconds": round(duration, 1),
            "error": traceback.format_exc(),
        }
        print(f"{tag} ✗ ERROR ({error_type}) dur={duration:.1f}s: {e}", file=sys.stderr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"task_{task.id}.json"
    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    real_stdout.write(
        json.dumps(
            {
                "task_id": str(task.id),
                "reward": result["reward"],
                "duration": result["duration_seconds"],
            }
        )
        + "\n"
    )
    real_stdout.flush()


if __name__ == "__main__":
    main()

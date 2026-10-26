# SPDX-License-Identifier: Apache-2.0
"""Isolated, single-node FSDP train-step measurements (not an RL benchmark)."""

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path", required=True, help="HF model ID or local checkpoint"
    )
    p.add_argument("--revision", help="HF revision, resolved once before the sweep")
    p.add_argument(
        "--seq-lens", required=True, help="Comma-separated GLOBAL batch lengths"
    )
    p.add_argument(
        "--token-budgets", required=True, help="max_tokens_per_mb candidates"
    )
    p.add_argument(
        "--gradient-checkpointing", choices=("off", "on", "both"), default="both"
    )
    p.add_argument("--nproc-per-node", type=int, default=1)
    p.add_argument("--prompt-tokens", type=int, default=1)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--timeout",
        type=float,
        default=1800,
        help="Seconds per trial, including initialization",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory; existing paths are rejected",
    )
    p.add_argument(
        "--init-from-scratch",
        action="store_true",
        help="Random initialization from a local model config (smoke tests)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the plan without loading a model or starting CUDA",
    )
    return p


def create_plan(args: argparse.Namespace) -> dict[str, Any]:
    lengths = [int(x) for x in args.seq_lens.split(",")]
    budgets = sorted(set(int(x) for x in args.token_budgets.split(",")))
    if not lengths or min(lengths) < 2:
        raise ValueError("Each sequence length must be at least 2")
    if args.nproc_per_node < 1 or len(lengths) % args.nproc_per_node:
        raise ValueError("Global batch size must be divisible by nproc-per-node")
    if min(budgets) < max(lengths):
        raise ValueError("Every token budget must fit the longest sequence")
    if not 1 <= args.prompt_tokens < min(lengths):
        raise ValueError("prompt-tokens must be >= 1 and smaller than every sequence")
    if args.warmup_steps < 1 or args.steps < 1:
        raise ValueError("warmup-steps and steps must both be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    checkpointing = {"off": [False], "on": [True], "both": [False, True]}[
        args.gradient_checkpointing
    ]
    return {
        "schema_version": 1,
        "model": args.model_path,
        "revision": args.revision,
        "init_from_scratch": args.init_from_scratch,
        "sequence_lengths": lengths,
        "global_batch_size": len(lengths),
        "input_tokens_per_step": sum(lengths),
        "loss_tokens_per_step": sum(n - args.prompt_tokens for n in lengths),
        "prompt_tokens": args.prompt_tokens,
        "world_size": args.nproc_per_node,
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "timeout_seconds": args.timeout,
        "trials": [
            {
                "id": f"tokens-{budget}-gc-{int(gc)}",
                "max_tokens_per_mb": budget,
                "gradient_checkpointing": gc,
            }
            for budget, gc in itertools.product(budgets, checkpointing)
        ],
    }


def checkpoint_manifest(path: Path) -> list[dict[str, Any]]:
    """Identify local artifacts without hashing multi-GB weights on every trial."""
    return [
        {
            "name": p.relative_to(path).as_posix(),
            "bytes": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
            **(
                {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                if p.suffix == ".json"
                else {}
            ),
        }
        for p in sorted(path.rglob("*"))
        if p.is_file()
    ]


def resolve_model(plan: dict[str, Any]) -> None:
    path = Path(plan["model"]).expanduser()
    if path.is_dir():
        if plan["revision"]:
            raise ValueError("--revision only applies to a Hugging Face model ID")
    else:
        if plan["init_from_scratch"]:
            raise ValueError(
                "--init-from-scratch requires a local model config/tokenizer"
            )
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                plan["model"],
                revision=plan["revision"],
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.bin",
                    "*.model",
                    "*.txt",
                    "*.tiktoken",
                ],
            )
        )
        plan["resolved_revision"] = path.name
    plan["resolved_model_path"] = str(path.resolve())
    plan["checkpoint_manifest"] = checkpoint_manifest(path)


def run_process(command: list[str], log: Path, timeout: float) -> dict[str, Any]:
    """Bound a trial using the existing Linux subreaper, including detached workers."""
    supervisor = (
        Path(__file__).resolve().parents[1] / "infra/utils/process_supervisor.py"
    )
    supervised_command = [sys.executable, str(supervisor), "--", *command]
    started = time.perf_counter()
    with log.open("w") as output:
        process = subprocess.Popen(
            supervised_command,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        status = "completed"
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status = "timeout"
        finally:
            # torchrun workers start their own sessions. Let the subreaper
            # finish TERM/KILL/reaping; killing only torchrun's group leaks them.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        "Trial supervisor could not complete cleanup; stopping sweep"
                    ) from exc
    return {
        "status": status,
        "returncode": process.returncode,
        "wall_seconds_including_startup": time.perf_counter() - started,
        "supervised_command": supervised_command,
    }


def summarize_trial(
    plan: dict[str, Any], ranks: list[dict[str, Any]]
) -> dict[str, Any]:
    ranks = sorted(ranks, key=lambda r: r["rank"])
    if [r["rank"] for r in ranks] != list(range(plan["world_size"])):
        raise ValueError("Missing or duplicate rank results")
    if any(len(r["samples"]) != plan["steps"] for r in ranks):
        raise ValueError("Incomplete step samples")
    if len({r["global_batch_sha256"] for r in ranks}) != 1:
        raise ValueError("Ranks generated different global batches")
    if any(
        not math.isfinite(s["seconds"]) or s["seconds"] <= 0
        for r in ranks
        for s in r["samples"]
    ):
        raise ValueError("Invalid step duration")
    for key, expected in (
        ("input_tokens", plan["input_tokens_per_step"]),
        ("loss_tokens", plan["loss_tokens_per_step"]),
    ):
        if sum(r[key] for r in ranks) != expected:
            raise ValueError(f"Rank {key} do not match the fixed global workload")
    durations = [
        max(r["samples"][i]["seconds"] for r in ranks) for i in range(plan["steps"])
    ]
    if any(not math.isfinite(t) or t <= 0 for t in durations):
        raise ValueError("Invalid step duration")
    prepared = [
        [r["samples"][i]["prepared_microbatches"] for r in ranks]
        for i in range(plan["steps"])
    ]
    return {
        "step_seconds": durations,
        "latency_seconds": {
            "mean": statistics.mean(durations),
            "median": statistics.median(durations),
            "min": min(durations),
            "max": max(durations),
            "p95_nearest_rank": sorted(durations)[math.ceil(0.95 * len(durations)) - 1],
        },
        "input_tokens_per_second": plan["input_tokens_per_step"]
        * plan["steps"]
        / sum(durations),
        "loss_tokens_per_second": plan["loss_tokens_per_step"]
        * plan["steps"]
        / sum(durations),
        "prepared_microbatches_per_step_per_rank": prepared,
        "executed_microbatches_per_step_per_rank": [
            [max(row)] * len(row) for row in prepared
        ],
        "peak_allocated_bytes_per_rank": [
            max(s["peak_allocated_bytes"] for s in r["samples"]) for r in ranks
        ],
        "peak_reserved_bytes_per_rank": [
            max(s["peak_reserved_bytes"] for s in r["samples"]) for r in ranks
        ],
        "ranks": ranks,
    }


def render_report(report: dict[str, Any]) -> str:
    plan = report["plan"]
    lines = [
        "# FSDP train-step benchmark",
        "",
        f"Model: `{plan['model']}`; world size: {plan['world_size']}; seed: {plan['seed']}.",
        f"Global batch: {plan['global_batch_size']} sequences, {plan['input_tokens_per_step']} input tokens, "
        f"{plan['loss_tokens_per_step']} loss tokens per step.",
        "",
        "Synthetic tokens; GPU-resident inputs. Synchronized wall time includes split/pack, "
        "SFT loss/monitoring, forward/backward and optimizer step. Initialization, IO, "
        "input staging and stats export are excluded. Each step uses the slowest rank.",
        "",
        "Reserved memory includes allocator caches from warmup/stats export. "
        "Results depend on workload and GPU contention; they are not general tuning advice.",
        "",
        "| Token budget | Checkpointing | Status | Mean step (s) | Input tokens/s | Loss tokens/s | Peak allocated GiB/rank |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for trial in report["trials"]:
        columns = [
            str(trial["max_tokens_per_mb"]),
            str(trial["gradient_checkpointing"]),
            trial["status"],
        ]
        if trial["status"] == "ok":
            columns += [
                f"{trial['latency_seconds']['mean']:.4f}",
                f"{trial['input_tokens_per_second']:.1f}",
                f"{trial['loss_tokens_per_second']:.1f}",
                ", ".join(
                    f"{x / 2**30:.3f}" for x in trial["peak_allocated_bytes_per_rank"]
                ),
            ]
        else:
            columns += ["—"] * 4
        lines.append("| " + " | ".join(columns) + " |")
    lines += [
        "",
        "See report.json and each trial's rank JSON/log for samples, provenance and failures.",
        "",
    ]
    if report.get("fatal_error"):
        lines += [f"Sweep stopped: {report['fatal_error']}", ""]
    return "\n".join(lines)


def run_trial(
    plan: dict[str, Any], trial: dict[str, Any], root: Path
) -> dict[str, Any]:
    directory = root / trial["id"]
    directory.mkdir()
    config = {**plan, "trial": trial, "trial_dir": str(directory)}
    write_json(directory / "config.json", config)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={plan['world_size']}",
        "--max-restarts=0",
        "--module",
        "areal.tools.fsdp_benchmark_worker",
        str(directory / "config.json"),
    ]
    outcome = run_process(command, directory / "output.log", plan["timeout_seconds"])
    result = {
        **trial,
        **outcome,
        "command": command,
        "log": str(directory / "output.log"),
    }
    failures = [
        json.loads(p.read_text()) for p in sorted(directory.glob("failure-*.json"))
    ]
    result["failures"] = failures
    if outcome["status"] == "timeout":
        return result
    if outcome["returncode"] != 0:
        result["status"] = (
            "oom"
            if any(f["type"] == "OutOfMemoryError" for f in failures)
            else "failed"
        )
        return result
    try:
        ranks = [
            json.loads(p.read_text()) for p in sorted(directory.glob("rank-*.json"))
        ]
        result.update(summarize_trial(plan, ranks))
        result["status"] = "ok"
    except (ValueError, KeyError, TypeError) as exc:
        result.update(status="failed", report_error=str(exc))
    return result


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    try:
        plan = create_plan(args)
        root = args.output_dir.expanduser().resolve()
        model = Path(args.model_path).expanduser()
        if model.is_dir() and root.is_relative_to(model.resolve()):
            raise ValueError("output-dir must be outside the local model directory")
        root.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    plan["created_at"] = datetime.now(UTC).isoformat()
    plan["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime_overrides": {
            key: os.environ[key]
            for key in (
                "OMP_NUM_THREADS",
                "NCCL_P2P_DISABLE",
                "NCCL_IB_DISABLE",
                "NCCL_SHM_DISABLE",
                "NCCL_NET",
                "CUBLAS_WORKSPACE_CONFIG",
                "PYTORCH_ALLOC_CONF",
                "PYTORCH_CUDA_ALLOC_CONF",
            )
            if key in os.environ
        },
    }
    write_json(root / "plan.json", plan)
    if args.dry_run:
        return 0
    try:
        resolve_model(plan)
    except Exception as exc:
        report = {
            "plan": plan,
            "trials": [],
            "fatal_error": f"{type(exc).__name__}: {exc}",
        }
        write_json(root / "report.json", report)
        (root / "report.md").write_text(render_report(report))
        return 1
    write_json(root / "plan.json", plan)
    report = {"plan": plan, "trials": []}
    received_signal = signal.SIGINT

    def interrupt(signum: int, frame: Any) -> None:
        nonlocal received_signal
        received_signal = signum
        raise KeyboardInterrupt

    previous_handlers = {
        s: signal.signal(s, interrupt)
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        for trial in plan["trials"]:
            if (
                checkpoint_manifest(Path(plan["resolved_model_path"]))
                != plan["checkpoint_manifest"]
            ):
                report["fatal_error"] = "Checkpoint files changed during the sweep"
                return 1
            try:
                result = run_trial(plan, trial, root)
            except KeyboardInterrupt:
                report["trials"].append(
                    {
                        **trial,
                        "status": "interrupted",
                        "signal": received_signal,
                        "log": str(root / trial["id"] / "output.log"),
                    }
                )
                return 128 + received_signal
            except Exception as exc:
                report["fatal_error"] = f"{type(exc).__name__}: {exc}"
                report["trials"].append(
                    {**trial, "status": "failed", "report_error": report["fatal_error"]}
                )
                return 1
            report["trials"].append(result)
            if (
                checkpoint_manifest(Path(plan["resolved_model_path"]))
                != plan["checkpoint_manifest"]
            ):
                result.update(
                    status="failed",
                    report_error="Checkpoint files changed during the trial",
                )
                report["fatal_error"] = result["report_error"]
                return 1
            if result["status"] == "ok":
                digest = result["ranks"][0]["global_batch_sha256"]
                if report.setdefault("global_batch_sha256", digest) != digest:
                    result.update(
                        status="failed",
                        report_error="Global data changed between trials",
                    )
                    report["fatal_error"] = result["report_error"]
                    return 1
            write_json(root / "report.json", report)
            (root / "report.md").write_text(render_report(report))
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        write_json(root / "report.json", report)
        (root / "report.md").write_text(render_report(report))
    return int(any(t["status"] != "ok" for t in report["trials"]))


if __name__ == "__main__":
    raise SystemExit(main())

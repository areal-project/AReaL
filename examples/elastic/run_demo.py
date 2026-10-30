# SPDX-License-Identifier: Apache-2.0

"""Native inline SWE training with event-driven Actor lifecycle control."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path


def redact(value):
    if isinstance(value, dict):
        return {
            key: "<redacted>"
            if "api" in key.lower() and "key" in key.lower()
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [redact(item) for item in value]
    return value


def prepare(args):
    from omegaconf import OmegaConf

    from areal.api.cli_args import PPOConfig, parse_cli_args, to_structured_cfg
    from areal.utils.atomic import atomic_json

    from .swe.sandbox import load_runtime
    from .swe.workflow import load_tasks

    if not args.swe_runtime:
        raise ValueError("SWE requires --swe-runtime or AREAL_SWE_CONFIG")
    load_runtime(str(args.swe_runtime))

    output, model, dataset_path = (
        args.output.resolve(),
        args.model.resolve(),
        args.dataset.resolve(),
    )
    if output.exists():
        raise ValueError(
            "Use a new output directory; existing runs are never overwritten"
        )
    if not args.prepare_only and not model.is_dir():
        raise ValueError("Model must be a complete local model directory")
    if len(args.devices) != 8 or len(set(args.devices)) != 8 or min(args.devices) < 0:
        raise ValueError("Provide eight distinct physical GPU indices")
    dataset = load_tasks(dataset_path)
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.export_style != "concat":
        raise ValueError("Grouped SWE requires linear concat export")
    os.environ.update(
        AREAL_DEMO_ROOT=str(output),
        AREAL_MODEL_PATH=str(model),
        AREAL_DATASET_PATH=str(dataset_path),
        AREAL_DEMO_POLICY=args.policy,
    )
    for name in ("AREAL_INFERENCE_ADMIN_KEY", "AREAL_ACTOR_ADMIN_KEY"):
        os.environ.setdefault(name, os.urandom(32).hex())
    overrides = {
        "trial_name": output.name,
        "total_train_steps": args.steps,
        "actor_lifecycle.actor_devices": args.devices[:4],
        "actor_lifecycle.rollout_devices": args.devices[4:],
        "actor_lifecycle.idle_timeout": args.idle_seconds,
        "actor_lifecycle.checkpoint_estimated_bytes": args.checkpoint_gib * 1024**3,
        "gconfig.max_new_tokens": args.max_tokens,
        "actor.max_new_tokens": args.max_tokens,
        "rollout.max_concurrent_rollouts": args.concurrency,
        "rollout.agent.export_style": args.export_style,
        "rollout.agent.chat_template_type": "concat"
        if args.export_style == "concat"
        else "hf",
        "rollout.agent.engine_max_tokens": args.context_tokens,
        "sglang.context_length": args.context_tokens,
        "vllm.max_model_len": args.context_tokens,
        "actor.mb_spec.max_tokens_per_mb": args.context_tokens,
    }
    raw, _ = parse_cli_args(
        [
            "--config",
            str(Path(__file__).with_name("config.yaml")),
            *[f"++{key}={json.dumps(value)}" for key, value in overrides.items()],
        ]
    )
    config = OmegaConf.to_object(to_structured_cfg(raw, PPOConfig))
    if args.steps * config.train_dataset.batch_size > len(dataset):
        raise ValueError("Provide at least steps * task-group batch size dataset rows")
    output.mkdir(parents=True)
    Path(config.cluster.fileroot).mkdir(parents=True, exist_ok=True)
    Path(config.cluster.name_resolve.nfs_record_root).mkdir(parents=True, exist_ok=True)
    atomic_json(output / "config.json", redact(dataclasses.asdict(config)))
    (output / "tasks.jsonl").write_bytes(dataset_path.read_bytes())
    manifest_path = dataset_path.with_suffix(".manifest.json")
    if manifest_path.exists():
        (output / "tasks.manifest.json").write_bytes(manifest_path.read_bytes())
    images_path = dataset_path.with_suffix(".images.json")
    images_sha256 = None
    if images_path.exists():
        images_sha256 = hashlib.sha256(images_path.read_bytes()).hexdigest()
        if manifest_path.exists():
            recorded = (
                json.loads(manifest_path.read_text()).get("images", {}).get("sha256")
            )
            if recorded is not None and recorded != images_sha256:
                raise ValueError(
                    "Resolved image manifest changed after dataset preparation"
                )
        (output / "tasks.images.json").write_bytes(images_path.read_bytes())
    atomic_json(
        output / "plan.json",
        {
            "base": "v2.1.0/ecc8b0e4",
            "mode": "inline",
            "policy": args.policy,
            "actor_startup": "first_collection"
            if args.policy == "resident"
            else "policy_controlled",
            "actor_backend": config.actor.backend,
            "rollout_backend": config.rollout.backend,
            "max_head_offpolicyness": config.rollout.max_head_offpolicyness,
            "updates": args.steps,
            "batch_size": config.train_dataset.batch_size,
            "count_unit": "task_groups",
            "n_samples": config.gconfig.n_samples,
            "trajectories_per_batch": config.train_dataset.batch_size
            * config.gconfig.n_samples,
            "dataset_rows": len(dataset),
            "unique_tasks": len({row["instance_id"] for row in dataset}),
            "supply": "native dataset/workflow with original staleness limits",
            "workload": "swe",
            "context_tokens": args.context_tokens,
            "max_turns": args.max_turns,
            "export_style": args.export_style,
            "concurrency": args.concurrency,
            "max_concurrent_episodes": args.concurrency * config.gconfig.n_samples,
            "actor_devices": args.devices[:4],
            "rollout_devices": args.devices[4:],
            "model": str(model),
            "model_present": model.is_dir(),
            "tasks_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "images_sha256": images_sha256,
            "gpu_acceptance": "not_run",
            "release_is_guaranteed": False,
            "agent_timing": {
                "triggers": "collection/update/release/progress/timer/batch_ready",
                "startup_estimate_seconds": getattr(
                    args, "startup_estimate_seconds", 90
                ),
                "checkpoint_estimate_seconds": getattr(
                    args, "checkpoint_estimate_seconds", 40
                ),
                "release_estimate_seconds": getattr(
                    args, "release_estimate_seconds", 15
                ),
                "restore_margin_seconds": getattr(args, "restore_margin_seconds", 15),
                "progress_debounce_seconds": 5,
            },
        },
    )
    return config, output, dataset


def summarize(output: Path, manager, expected_steps: int):
    from .plot import read_jsonl

    events = read_jsonl(output / "state/events.jsonl")
    observations = read_jsonl(output / "observations.jsonl")
    episodes = read_jsonl(output / "episodes.jsonl")
    releases = [
        e
        for e in events
        if e["state"] == "STOPPED"
        and e["applied_version"] < expected_steps
        and e.get("actor_gpu_seconds") is not None
    ]
    restores = [e for e in events if e["state"] == "RESTORED"]
    progress_while_off = False
    for release in releases:
        later_start = next(
            (
                e
                for e in events
                if e["state"] == "STARTING" and e["monotonic"] > release["monotonic"]
            ),
            None,
        )
        if later_start is None:
            continue
        window = [
            o
            for o in observations
            if release["monotonic"] <= o["monotonic"] < later_start["monotonic"]
        ]
        # Lifecycle snapshots bracket the exact off interval; a batch can arrive
        # between the final 0.5s observer sample and the STARTING event.
        accepted = [
            o["rollout"]["accepted"]
            for o in [release, *window, later_start]
            if "accepted" in o.get("rollout", {})
        ]
        progress_while_off |= bool(accepted and max(accepted) > min(accepted))
    generations = {e["generation"] for e in events if e["state"] == "TRAINING"}
    versions = manager.rollout.serving_versions()
    if manager.applied_version != expected_steps or any(
        v != expected_steps for v in versions
    ):
        raise RuntimeError(
            "Final native updates/serving acknowledgements are incomplete"
        )
    return {
        "status": "training_completed",
        "policy": manager.policy_adapter.mode,
        "updates": manager.applied_version,
        "serving_versions": versions,
        "data_progress": manager.data_progress,
        "mid_run_releases": len(releases),
        "restores": len(restores),
        "trained_generations": len(generations),
        "rollout_progress_while_actor_off": progress_while_off,
        "elastic_cycle_observed": bool(
            releases and restores and len(generations) > 1 and progress_while_off
        ),
        "max_actor_idle_seconds": max(
            (o["idle_seconds"] for o in observations), default=0
        ),
        "idle_metric_note": "Idle clock includes LLM/checkpoint work; use ready-count observations to measure missing-batch intervals.",
        "completed_workflow_episodes": len(episodes),
        "completed_workflow_grading_exception_zeros": sum(
            e.get("grading_kind") == "exception_zero" for e in episodes
        ),
        "tool_calls": sum(e["tool_calls"] for e in episodes),
        "average_turns": sum(e["turns"] for e in episodes) / max(1, len(episodes)),
        "restore_tensor_equality": "native DCP restore; not independently tensor-compared",
        "coordinator_recovery": "unsupported; coordinator stays resident",
    }


def close_trainer(trainer, output: Path) -> None:
    """Close training and await cleanup of prefetched SWE sandboxes."""
    from areal.utils.atomic import atomic_json

    executor = getattr(getattr(trainer, "rollout", None), "workflow_executor", None)
    dispatcher = getattr(executor, "_dispatcher", None)
    runner = getattr(dispatcher, "runner", None)
    try:
        trainer.close()
    finally:
        if runner is not None and runner.thread is not None:
            # Native shutdown only joins for 30s; SWE pod deletion can exceed it.
            runner.thread.join(timeout=180)
            if runner.thread.is_alive():
                raise RuntimeError("SWE native workflow cleanup did not finish")
        incomplete = [
            str(directory)
            for directory in (output / "swe-episodes").glob("*")
            if not (directory / "report.json").exists()
            or not json.loads((directory / "report.json").read_text()).get(
                "cleanup_verified"
            )
        ]
        if incomplete:
            atomic_json(
                output / "cleanup-error.json", {"incomplete_swe_episodes": incomplete}
            )
            raise RuntimeError("SWE episode cleanup was not independently verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--swe-runtime", type=Path, default=os.environ.get("AREAL_SWE_CONFIG")
    )
    parser.add_argument("--context-tokens", type=int, default=16384)
    parser.add_argument(
        "--policy", choices=("agent", "manual", "idle", "resident"), default="agent"
    )
    parser.add_argument("--devices", nargs=8, type=int, default=list(range(8)))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Concurrent task groups (4 SWE episodes each)",
    )
    parser.add_argument(
        "--export-style", choices=("individual", "concat"), default="concat"
    )
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=40,
        help="Threshold for policy=idle; agent decisions do not wait for it",
    )
    parser.add_argument("--startup-estimate-seconds", type=float, default=90)
    parser.add_argument("--checkpoint-estimate-seconds", type=float, default=40)
    parser.add_argument("--release-estimate-seconds", type=float, default=15)
    parser.add_argument("--restore-margin-seconds", type=float, default=15)
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--checkpoint-gib", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if any(
        not math.isfinite(value) or value < 0
        for value in (
            args.startup_estimate_seconds,
            args.checkpoint_estimate_seconds,
            args.release_estimate_seconds,
            args.restore_margin_seconds,
        )
    ):
        parser.error("Lifecycle estimates and margin must be finite and nonnegative")
    if min(args.max_turns, args.max_tokens, args.concurrency, args.timeout) <= 0:
        parser.error("turns, tokens, concurrency and timeout must be positive")
    if args.max_tokens + 1024 >= args.context_tokens:
        parser.error(
            "context must leave at least 1024 prompt tokens beyond completion budget"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["AREAL_IS_SINGLE_CONTROLLER"] = "1"
    os.environ["AREAL_SPMD_MODE"] = "0"
    repo = str(Path(__file__).resolve().parents[2])
    os.environ["PYTHONPATH"] = repo + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    from areal.elastic.policy import LifecyclePolicy
    from areal.utils.atomic import atomic_json

    from .agent import ControlLLM
    from .plot import plot_run
    from .swe.progress import SWEProgress
    from .telemetry import GPUMonitor

    config, output, dataset = prepare(args)
    if args.prepare_only:
        return
    from areal import PPOTrainer

    llm = ControlLLM() if args.policy == "agent" else None
    progress = SWEProgress(
        output,
        startup_seconds=args.startup_estimate_seconds,
        checkpoint_seconds=args.checkpoint_estimate_seconds,
        release_seconds=args.release_estimate_seconds,
        margin_seconds=args.restore_margin_seconds,
    )
    policy = LifecyclePolicy(
        mode=args.policy,
        decide=None if llm is None else llm.decide,
        observe=progress if llm is not None else None,
    )

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    trainer = None
    finished = threading.Event()
    observer = None
    result = None
    gpu_monitor = GPUMonitor(output / "gpu-memory.jsonl", args.devices)
    gpu_monitor.start()
    try:
        trainer = PPOTrainer(config, train_dataset=dataset, elastic_policy=policy)
        manager = trainer._on_demand
        if llm is not None:
            llm.observe = policy.observation
        start = time.monotonic()

        def observe():
            try:
                with (output / "observations.jsonl").open("a", buffering=1) as stream:
                    while not finished.is_set():
                        stream.write(
                            json.dumps(
                                {
                                    "wall_time": time.time(),
                                    "monotonic": time.monotonic(),
                                    **policy.observation(),
                                }
                            )
                            + "\n"
                        )
                        stats = manager.get_status()["rollout"]
                        if (
                            stats.get("rejected", 0) >= 32
                            and stats.get("accepted", 0) == 0
                        ):
                            atomic_json(
                                output / "observer-error.json",
                                {
                                    "error": "32 native workflow rejections without an accepted episode; inspect service logs"
                                },
                            )
                            manager.cancelled.set()
                        if time.monotonic() - start > args.timeout:
                            manager.cancelled.set()
                        finished.wait(0.5)
            except BaseException:
                manager.cancelled.set()
                raise

        observer = threading.Thread(
            target=observe, name="elastic-observer", daemon=True
        )
        observer.start()
        workflow_kwargs = {
            "log_path": str(output / "episodes.jsonl"),
            "max_turns": args.max_turns,
            "max_tokens": args.max_tokens,
            "runtime_config": str(args.swe_runtime.resolve()),
            "context_tokens": args.context_tokens,
        }
        trainer.train(
            workflow="examples.elastic.swe.workflow.SWEAgent",
            workflow_kwargs=workflow_kwargs,
        )
        result = summarize(output, manager, args.steps)
    except BaseException as error:
        atomic_json(
            output / "result.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        finished.set()
        if observer is not None:
            observer.join(timeout=5)
        try:
            try:
                if trainer is not None:
                    close_trainer(trainer, output)
            finally:
                if llm is not None:
                    llm.close()
        except BaseException as error:
            atomic_json(output / "cleanup-error.json", {"error": str(error)})
            atomic_json(
                output / "result.json",
                {
                    "status": "failed",
                    "stage": "cleanup",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "training_result_before_cleanup": result,
                },
            )
            raise
        finally:
            gpu_monitor.close()
    plot_run(output)
    atomic_json(output / "result.json", result)


if __name__ == "__main__":
    main()

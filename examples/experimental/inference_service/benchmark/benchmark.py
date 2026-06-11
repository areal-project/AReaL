"""Inference service benchmark via RolloutControllerV2.

Launches the full IS stack (SGLang → DataProxy → Router → Gateway) through
RolloutControllerV2, then sweeps concurrency levels using
Tau2AgentWorkflow to measure end-to-end throughput and pass rate.

Usage (controller launches SGLang):
    python3 examples/experimental/inference_service/benchmark/benchmark.py \
        --config examples/experimental/inference_service/benchmark/benchmark.yaml \
        --model-path /models/Qwen3-235B \
        --user-endpoint http://user-node:30001/v1 \
        --concurrencies 5,10 --num-tasks 2 --num-trials 1

Usage (pre-existing SGLang):
    python3 ... --agent-endpoint http://agent-node:30000 \
        --user-endpoint http://user-node:30001/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from areal.api.alloc_mode import ModelAllocation
from areal.api.cli_args import (
    BaseExperimentConfig,
    GenerationHyperparameters,
    InferenceEngineConfig,
    SGLangConfig,
    TrainDatasetConfig,
    load_expr_config,
)
from areal.experimental.inference_service.controller.controller import (
    RolloutControllerV2,
)
from areal.utils import logging

logger = logging.getLogger("ISBenchmark")


@dataclass
class Tau2EnvConfig:
    domain: str = field(
        default="airline",
        metadata={
            "help": "The tau2 domain name, e.g., 'retail', 'airline', 'telecom'."
        },
    )
    max_steps: int = field(default=200, metadata={"help": "Maximum turns per task."})
    add_thinking_tool: bool = field(default=False)
    solo_mode: bool = field(default=False)
    user_llm_base_url: str | None = field(
        default=None, metadata={"help": "Base URL of the user simulator LLM."}
    )
    user_llm: str | None = field(default=None)
    user_llm_args: dict | None = field(default=None)
    turn_discount: float = 1.0
    invalid_format_penalty: float = 0.1


@dataclass
class BenchmarkConfig(BaseExperimentConfig):
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters
    )
    rollout: InferenceEngineConfig = field(default_factory=InferenceEngineConfig)
    model_path: str = ""
    econfig: Tau2EnvConfig = field(default_factory=Tau2EnvConfig)
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    train_dataset: TrainDatasetConfig = field(default_factory=TrainDatasetConfig)


def get_tau2_tasks(domain: str, num_tasks: int, seed: int) -> list[dict[str, Any]]:
    """Load tau2 task IDs and return as a list of data dicts."""
    import random

    from tau2.registry import registry

    tasks = registry.get_tasks_loader(domain)("test")
    rng = random.Random(seed)
    selected = [tasks[i % len(tasks)] for i in range(num_tasks)]
    rng.shuffle(selected)
    return [{"task_id": t.id, "split": "test"} for t in selected]


def _wait_workers_healthy(
    router_addr: str,
    admin_api_key: str,
    timeout_s: float = 60.0,
    poll_interval_s: float = 1.0,
) -> None:
    import urllib.request

    deadline = time.monotonic() + timeout_s
    last_state = "no-response"
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(
                f"{router_addr}/workers",
                headers={"Authorization": f"Bearer {admin_api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            workers = data.get("workers", [])
            healthy = [w for w in workers if w.get("healthy")]
            last_state = f"total={len(workers)} healthy={len(healthy)}"
            if healthy:
                return
        except Exception as exc:
            last_state = f"poll-error: {exc!r}"
        time.sleep(poll_interval_s)
    raise RuntimeError(f"No healthy router worker after {timeout_s}s ({last_state})")


def run_trial(
    ctrl: RolloutControllerV2,
    data: list[dict[str, Any]],
    workflow: str,
    workflow_kwargs: dict[str, Any],
    concurrency: int,
) -> dict[str, Any]:
    """Run one trial at a given concurrency and return summary stats."""
    batch = data[:concurrency]
    t0 = time.monotonic()
    results = ctrl.rollout_batch(
        data=batch,
        workflow=workflow,
        workflow_kwargs=workflow_kwargs,
    )
    elapsed = time.monotonic() - t0

    import torch

    from areal.infra.rpc.rtensor import RTensor

    rewards = []
    for traj in results:
        local_traj = RTensor.localize(traj)
        if "rewards" in local_traj:
            rewards.append(local_traj["rewards"])

    if rewards:
        all_rewards = torch.cat(rewards, dim=0)
        passed = int((all_rewards > 0).sum().item())
    else:
        all_rewards = torch.tensor([])
        passed = 0

    completed = len(results)
    return {
        "completed": completed,
        "passed": passed,
        "failed": completed - passed,
        "pass_rate": passed / completed if completed > 0 else 0.0,
        "total_time_s": elapsed,
        "tasks_per_min": completed / elapsed * 60 if elapsed > 0 else 0.0,
    }


def main(argv: list[str]) -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--user-endpoint", type=str, default=None)
    parser.add_argument(
        "--agent-endpoint",
        type=str,
        default=None,
        help="Use a pre-existing Agent SGLang (e.g. http://slurmd-85:30000). "
        "Skips launching a new SGLang; only starts IS services (Router/DataProxy/Gateway).",
    )
    parser.add_argument("--concurrencies", type=str, default="5,10,15,20,25,30")
    parser.add_argument("--num-tasks", type=int, default=50)
    parser.add_argument("--num-trials", type=int, default=4)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="./trajectories")
    bench_args, remaining = parser.parse_known_args(argv)

    if bench_args.model_path:
        model_name = os.path.basename(bench_args.model_path)
        remaining += [
            f"model_path={bench_args.model_path}",
            f"sglang.model_path={bench_args.model_path}",
            f"tokenizer_path={bench_args.model_path}",
        ]
    else:
        model_name = None
    if bench_args.user_endpoint:
        remaining.append(f"econfig.user_llm_base_url={bench_args.user_endpoint}")
        if model_name and not any(r.startswith("econfig.user_llm=") for r in remaining):
            remaining.append(f"econfig.user_llm=openai/{model_name}")

    concurrencies = [int(c) for c in bench_args.concurrencies.split(",")]
    num_tasks = bench_args.num_tasks
    num_trials = bench_args.num_trials
    output_dir = Path(bench_args.output_dir)

    config, _ = load_expr_config(remaining, BenchmarkConfig)
    econfig = config.econfig
    rollout_cfg = config.rollout
    rollout_cfg.consumer_batch_size = max(concurrencies)

    from areal.infra.scheduler.local import LocalScheduler

    scheduler = LocalScheduler(exp_config=config)

    ctrl = RolloutControllerV2(config=rollout_cfg, scheduler=scheduler)
    try:
        if bench_args.agent_endpoint:
            from urllib.parse import urlparse

            from areal.api.io_struct import LocalInfServerInfo

            parsed = urlparse(bench_args.agent_endpoint)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 30000
            server_infos = [LocalInfServerInfo(host=host, port=port, process=None)]
            ctrl.initialize(role="rollout", server_args=None, server_infos=server_infos)
        else:
            rollout_alloc = ModelAllocation.from_str(
                config.rollout.backend, name="rollout"
            )
            if rollout_alloc.backend == "sglang":
                server_args = asdict(config.sglang)
            else:
                raise ValueError(f"Unsupported backend: {rollout_alloc.backend}")
            ctrl.initialize(role="rollout", server_args=server_args)
        logger.info("IS stack ready at %s", ctrl.proxy_gateway_addr)

        econfig_dict = asdict(econfig)
        workflow_kwargs: dict[str, Any] = dict(
            econfig=econfig_dict,
            gen_args=dict(
                temperature=config.gconfig.temperature,
                max_completion_tokens=config.gconfig.max_new_tokens,
            ),
            timeout=600.0,
        )
        workflow = "examples.tau2.agent.Tau2AgentWorkflow"

        data = get_tau2_tasks(econfig.domain, num_tasks, bench_args.seed)
        all_summaries: list[dict[str, Any]] = []

        logger.info(
            "Starting sweep: concurrencies=%s, tasks=%d, trials=%d",
            concurrencies,
            num_tasks,
            num_trials,
        )

        for c in concurrencies:
            for trial in range(1, num_trials + 1):
                logger.info("Concurrency=%d  Trial=%d", c, trial)
                _wait_workers_healthy(ctrl._router_addr, ctrl.config.admin_api_key)
                summary = run_trial(ctrl, data, workflow, workflow_kwargs, c)
                summary["concurrency"] = c
                summary["trial"] = trial
                all_summaries.append(summary)

                run_dir = output_dir / f"c{c}" / f"trial_{trial}"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
                logger.info(
                    "  pass_rate=%.1f%%  time=%.0fs  tasks/min=%.1f",
                    summary["pass_rate"] * 100,
                    summary["total_time_s"],
                    summary["tasks_per_min"],
                )

        print()
        print("Concurrency | Trial | Tasks | Pass | Fail | Rate   | Dur(s) | tasks/min")
        print("-" * 80)
        for s in all_summaries:
            print(
                f"  c{s['concurrency']:>8} | {s['trial']:>5} | {s['completed']:>5} | "
                f"{s['passed']:>4} | {s['failed']:>4} | {s['pass_rate']:>5.1%} | "
                f"{s['total_time_s']:>6.0f} | {s['tasks_per_min']:>9.1f}"
            )
        print()

    finally:
        ctrl.destroy()
        scheduler.delete_workers(None)


if __name__ == "__main__":
    main(sys.argv[1:])

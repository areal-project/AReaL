# SPDX-License-Identifier: Apache-2.0

"""CPU-only, causal fixed-versus-online simulation with explicitly supplied fits."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path


def cpu_math_imports():
    """Do not initialize areal's GPU runtime in this standalone math runner."""
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for name in (
        "areal",
        "areal.autopilot",
        "areal.autopilot.domains",
        "areal.autopilot.domains.elastic",
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(root.joinpath(*name.split(".")))]
            sys.modules[name] = package
    return root


def load_workload(path, tool_seconds_per_unit):
    from areal.elastic.policy.basic.forecast import EpisodeSample, Turn

    result = []
    for line in Path(path).read_text().splitlines():
        raw = json.loads(line)
        nodes = raw["nodes"]
        turns = []
        for index in range(0, len(nodes), 2):
            request = nodes[index]
            previous = () if index == 0 else (nodes[index - 1]["id"],)
            if (
                request["kind"] != "inference"
                or tuple(request["predecessors"]) != previous
            ):
                raise ValueError(
                    "Online workload adapter requires a strict inference/tool chain"
                )
            tool = nodes[index + 1] if index + 1 < len(nodes) else None
            if tool is not None:
                if (
                    tool["kind"] != "tool"
                    or tuple(tool["predecessors"]) != (request["id"],)
                    or index + 2 >= len(nodes)
                ):
                    raise ValueError(
                        "An intermediate tool must have one successor inference"
                    )
                seconds = (
                    tool["work_units"] * tool_seconds_per_unit[tool["work_reference"]]
                )
                overhead = (
                    nodes[index + 2]["input_tokens"]
                    - request["input_tokens"]
                    - request["output_tokens"]
                    - tool["output_tokens"]
                )
                turns.append(
                    Turn(
                        request["input_tokens"],
                        request["output_tokens"],
                        seconds,
                        tool["output_tokens"],
                        tool["tool_type"],
                        overhead,
                    )
                )
            else:
                turns.append(Turn(request["input_tokens"], request["output_tokens"]))
        result.append(
            EpisodeSample(
                raw["episode_id"], raw["family"], raw["available_at"], tuple(turns)
            )
        )
    return tuple(result)


def candidates(batch_size, *, steady_gpu_budget=16, role_gpu_limit=8):
    from areal.elastic.policy.basic.forecast import OnlineLayout

    if not 2 <= steady_gpu_budget <= 16 or not 1 <= role_gpu_limit <= 16:
        raise ValueError("Search requires a2–16GPU steady budget and1–16GPU role limit")
    return tuple(
        sorted(
            (
                OnlineLayout(dp, tp, replicas, rtp)
                for tp in (1, 2, 4, 8)
                for dp in range(1, role_gpu_limit // tp + 1)
                if batch_size % dp == 0
                for rtp in (1, 2, 4, 8)
                for replicas in range(1, role_gpu_limit // rtp + 1)
                if dp * tp + replicas * rtp <= steady_gpu_budget
            ),
            key=lambda c: (c.actor_gpus + c.rollout_gpus, c),
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--mechanism-fits", type=Path, required=True)
    parser.add_argument("--publication-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("fixed", "dynamic", "both"), default="both")
    parser.add_argument("--search-workers", type=int, default=4)
    parser.add_argument(
        "--event-detail", choices=("batch", "structural"), default="batch"
    )
    parser.add_argument(
        "--decision-clock", choices=("measured", "fixed"), default="measured"
    )
    args = parser.parse_args()
    if args.search_workers < 1:
        parser.error("search-workers must be positive")
    repo = cpu_math_imports()
    from examples.autopilot.elastic.simulate_online import (
        cpu_math_imports as initialize_worker,
    )

    from areal.autopilot.domains.elastic.online_models import CalibratedOnlineModels
    from areal.autopilot.domains.elastic.online_simulation import (
        OnlineRunConfig,
        OnlineSimulation,
    )
    from areal.autopilot.domains.elastic.online_transitions import HandoffConfig
    from areal.elastic.policy.basic.forecast import (
        ForecastConfig,
        HistoricalForecaster,
        OnlineLayout,
    )
    from areal.elastic.policy.basic.history import HistoricalReplay, HistoryReplayConfig
    from areal.elastic.policy.basic.online import OnlinePolicy, OnlinePolicyConfig
    from areal.elastic.policy.basic.serving import ServingSLO

    raw = json.loads(args.config.read_text())
    run_raw = dict(raw["run"])
    run_raw["tool_capacities"] = tuple(tuple(row) for row in run_raw["tool_capacities"])
    run = OnlineRunConfig(**run_raw)
    handoff_raw = dict(raw["handoff"])
    handoff_raw["node_gpus"] = tuple(handoff_raw["node_gpus"])
    handoff = HandoffConfig(**handoff_raw)
    initial = OnlineLayout(**raw["initial_layout"])
    slo = ServingSLO(**raw["slo"])
    workload_config = raw.get("replay", raw.get("forecast"))
    if workload_config is None:
        parser.error("Supply an explicit replay configuration")
    if workload_config["max_new_tokens"] != run.max_new_tokens:
        parser.error("Forecast and serving must share the declared generation budget")
    models = CalibratedOnlineModels(
        args.bundle,
        args.mechanism_fits,
        args.publication_metadata,
        handoff.effective_bandwidth_GBps,
        inter_node_dp_GBps=raw.get("model_assumptions", {}).get("inter_node_dp_GBps"),
    )
    if models.weight_bytes != handoff.weight_bytes:
        parser.error(
            "Explicit handoff weight_bytes differs from model publication metadata"
        )
    workload = load_workload(args.workload, raw["tool_seconds_per_unit"])
    configurations = (
        tuple(OnlineLayout(**c) for c in raw["candidates"])
        if "candidates" in raw
        else candidates(run.batch_size, **raw.get("search_space", {}))
    )
    args.output.mkdir(parents=True, exist_ok=False)
    files = [
        args.workload,
        args.config,
        args.mechanism_fits,
        args.publication_metadata,
        args.bundle / "manifest.json",
    ]
    sources = [Path(__file__)] + [
        repo / p
        for p in (
            "areal/elastic/policy/basic/forecast.py",
            "areal/elastic/policy/basic/history.py",
            "areal/elastic/policy/basic/online.py",
            "areal/elastic/policy/basic/search.py",
            "areal/elastic/policy/basic/serving.py",
            "areal/autopilot/domains/elastic/online_models.py",
            "areal/autopilot/domains/elastic/online_decode.py",
            "areal/autopilot/domains/elastic/online_inputs.py",
            "areal/autopilot/domains/elastic/online_control.py",
            "areal/autopilot/domains/elastic/online_state.py",
            "areal/autopilot/domains/elastic/online_parallel.py",
            "areal/autopilot/domains/elastic/online_simulation.py",
            "areal/autopilot/domains/elastic/online_transitions.py",
        )
    ]
    identity = dict(
        inputs={
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        sources={
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        models=models.source_ids,
        config=raw,
        candidate_count=len(configurations),
        search_workers=args.search_workers,
        decision_clock=args.decision_clock,
        event_detail=args.event_detail,
        environment=sys.version,
    )
    (args.output / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    results = {}
    for mode in ("fixed", "dynamic") if args.mode == "both" else (args.mode,):
        policy = None
        if mode == "dynamic":
            policy = OnlinePolicy(
                configurations,
                HistoricalReplay(HistoryReplayConfig(**raw["replay"]))
                if "replay" in raw
                else HistoricalForecaster(ForecastConfig(**raw["forecast"])),
                OnlinePolicyConfig(**raw["policy"]),
                slo,
            )
        path = args.output / mode
        path.mkdir()
        started = time.perf_counter()
        with (
            gzip.open(path / "events.jsonl.gz", "wt") as stream,
            (
                ProcessPoolExecutor(
                    max_workers=args.search_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_worker,
                )
                if mode == "dynamic" and args.search_workers > 1
                else nullcontext(None)
            ) as candidate_executor,
        ):
            simulation = OnlineSimulation(
                workload,
                initial,
                models,
                run,
                handoff,
                slo,
                input_closes_at=raw["input_closes_at"],
                policy=policy,
                candidate_executor=candidate_executor,
                measured_search_time=args.decision_clock == "measured",
                event_sink=(
                    lambda event: stream.write(
                        json.dumps(event, separators=(",", ":")) + "\n"
                    )
                )
                if args.event_detail == "batch"
                else None,
            )
            result = simulation.run()
            if args.event_detail == "structural":
                for event in simulation.events:
                    stream.write(json.dumps(event, separators=(",", ":")) + "\n")
        result["program_wall_seconds"] = time.perf_counter() - started
        result["slo_met"] = (
            result["status"] == "complete"
            and result["joint_request_fraction"] is not None
            and result["joint_request_fraction"] >= slo.minimum_request_fraction
        )
        for name, value in (
            ("summary", result),
            ("decisions", simulation.decisions),
            ("training", simulation.training_records),
            ("requests", [asdict(r.view) for r in simulation.requests.values()]),
        ):
            (path / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
        results[mode] = result
        print(json.dumps(dict(mode=mode, **result)), flush=True)
    if len(results) == 2 and all(r["status"] == "complete" for r in results.values()):
        for key in (
            "arrived_episodes",
            "completed_episodes",
            "completed_updates",
            "trained_episodes",
            "retained_episodes",
            "completed_requests",
        ):
            if results["fixed"][key] != results["dynamic"][key]:
                raise RuntimeError(f"Fixed/dynamic work conservation failed: {key}")
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(r["status"] != "complete" for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

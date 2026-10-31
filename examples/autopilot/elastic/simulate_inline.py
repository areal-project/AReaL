# SPDX-License-Identifier: Apache-2.0

"""Run equal-work fixed/dynamic inline comparisons with frozen calibrated models."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Direct script invocation and module invocation share the existing CPU bootstrap.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.autopilot.elastic.simulate_online import cpu_math_imports, load_workload


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--workload", required=True, type=Path, help="Task-count generation spec"
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--mechanism-fits", required=True, type=Path)
    parser.add_argument("--publication-metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--decision-clock", choices=("measured", "fixed"), default="measured"
    )
    parser.add_argument(
        "--all-fixed",
        action="store_true",
        help="Also run every fixed candidate; default compares only initial fixed and dynamic",
    )
    parser.add_argument(
        "--progress", action="store_true", help="Print full-search progress"
    )
    args = parser.parse_args()
    repo = cpu_math_imports()
    from areal.elastic.policy.basic.inline import InlinePolicy, InlinePolicyConfig
    from areal.elastic.policy.basic.inline_forecast import (
        InlineForecastConfig,
        InlineForecaster,
    )
    from areal.elastic.policy.basic.observations import OnlineLayout
    from areal.elastic.simulation.inline_simulation import InlineSimulation
    from areal.elastic.simulation.online_models import CalibratedOnlineModels
    from areal.elastic.simulation.online_transitions import HandoffConfig
    from areal.elastic.simulation.simulation_core import SimulationConfig
    from areal.elastic.workload import TaskCountSpec, canonical, generate_count

    raw = json.loads(args.config.read_text())
    spec = TaskCountSpec.decode(json.loads(args.workload.read_text()))
    config = SimulationConfig(
        **{
            **raw["run"],
            "tool_capacities": tuple(tuple(v) for v in raw["run"]["tool_capacities"]),
        }
    )
    handoff = HandoffConfig(
        **{**raw["handoff"], "node_gpus": tuple(raw["handoff"]["node_gpus"])}
    )
    initial = OnlineLayout(**raw["initial_layout"])
    layouts = tuple(
        dict.fromkeys((initial,) + tuple(OnlineLayout(**v) for v in raw["candidates"]))
    )
    policy_config = InlinePolicyConfig(**raw["policy"])
    forecast_config = InlineForecastConfig(**raw["forecast"])
    if config.max_new_tokens != forecast_config.max_new_tokens:
        parser.error("Forecast and execution must use the same generation budget")
    if spec.count % config.batch_size:
        parser.error(
            "The first inline example requires a whole number of training batches"
        )
    models = CalibratedOnlineModels(
        args.bundle,
        args.mechanism_fits,
        args.publication_metadata,
        handoff.effective_bandwidth_GBps,
        inter_node_dp_GBps=raw.get("model_assumptions", {}).get("inter_node_dp_GBps"),
    )
    if models.weight_bytes != handoff.weight_bytes:
        parser.error("Handoff weight size disagrees with frozen publication metadata")
    args.output.mkdir(parents=True, exist_ok=False)
    trace = args.output / "workload.jsonl"
    trace.write_text("".join(canonical(e) + "\n" for e in generate_count(spec)))
    workload = load_workload(trace, raw["tool_seconds_per_unit"])
    write_json(args.output / "config.json", raw)
    write_json(
        args.output / "workload-spec.json", json.loads(args.workload.read_text())
    )
    source_paths = (
        list((repo / "areal/elastic/simulation").glob("*.py"))
        + [
            repo / "areal/elastic/policy/basic" / name
            for name in (
                "inline.py",
                "inline_forecast.py",
                "inline_trigger.py",
                "forecast_workload.py",
                "observations.py",
                "online_forecast.py",
            )
        ]
        + [repo / "areal/elastic/workload.py", Path(__file__).resolve()]
    )
    inputs = (
        args.config,
        args.workload,
        args.bundle / "manifest.json",
        args.mechanism_fits,
        args.publication_metadata,
        trace,
    )
    write_json(
        args.output / "identity.json",
        dict(
            sources={
                str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source_paths
            },
            inputs={
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in inputs
            },
            models=models.source_ids,
            decision_clock=args.decision_clock,
            candidates=[asdict(c) for c in layouts],
        ),
    )
    runs = (
        [("fixed", initial, None)]
        + [("fixed-" + c.key, c, None) for c in layouts if c != initial]
        + [
            (
                "dynamic",
                initial,
                InlinePolicy(layouts, InlineForecaster(forecast_config), policy_config),
            )
        ]
    )
    if not args.all_fixed:
        runs = [run for run in runs if run[0] in {"fixed", "dynamic"}]
    results = {}

    def report_search(record):
        print(
            json.dumps(
                {
                    "event": "search",
                    "update": record["observed_version"],
                    "trigger": record["trigger"],
                    "seconds": record["search_seconds"],
                    "target": record["target"],
                    "gain": record["relative_improvement"],
                }
            ),
            flush=True,
        )

    for name, layout, policy in runs:
        out = args.output / name
        out.mkdir()
        started = time.perf_counter()
        sim = InlineSimulation(
            workload,
            layout,
            models,
            config,
            handoff,
            policy=policy,
            gpu_budget=raw["gpu_budget"],
            coalesce_decode=True,
            measured_search_time=args.decision_clock == "measured",
            search_callback=report_search if args.progress else None,
        )
        result = sim.run()
        result["program_wall_seconds"] = time.perf_counter() - started
        records = [e for r in sim.training_records for e in r["episodes"]]
        expected = [e.episode_id for e in workload]
        expected_requests = sum(len(e.turns) for e in workload)
        expected_tokens = sum(t.output_tokens for e in workload for t in e.turns)
        audit = dict(
            ordered_training=records == expected,
            requests=result["completed_requests"] == expected_requests,
            output_tokens=sum(r.view.generated for r in sim.requests.values())
            == expected_tokens,
            versions=sim.training_version
            == sim.provider_version
            == spec.count // config.batch_size,
            whole_run_budget=sim.pool.peak_gpus <= raw["gpu_budget"],
            released_at_end=not sim.pool.leases,
            token_staleness=all(
                r.get("maximum_version_lag", 0) <= config.staleness
                for r in sim.training_records
            ),
        )
        result["audit"] = audit
        write_json(out / "summary.json", result)
        write_json(out / "decisions.json", sim.decisions)
        write_json(out / "trigger-checks.json", sim.trigger_checks)
        write_json(out / "training.json", sim.training_records)
        write_json(
            out / "requests.json", [asdict(r.view) for r in sim.requests.values()]
        )
        with gzip.open(out / "events.jsonl.gz", "wt") as stream:
            for event in sim.events:
                stream.write(json.dumps(event) + "\n")
        results[name] = result
        print(
            json.dumps(
                dict(
                    run=name,
                    status=result["status"],
                    seconds=result["seconds"],
                    updates=result["completed_updates"],
                    peak_gpus=result["peak_gpus"],
                    audit=audit,
                )
            ),
            flush=True,
        )
        if result["status"] == "complete" and not all(audit.values()):
            raise RuntimeError(f"Completed run failed training/budget audit: {name}")
    feasible_fixed = [
        name
        for name, r in results.items()
        if name != "dynamic" and r["status"] == "complete"
    ]
    best = (
        min(feasible_fixed, key=lambda name: results[name]["seconds"])
        if feasible_fixed
        else None
    )
    comparison = dict(
        results=results,
        baseline="initial_fixed",
        improvement_vs_initial=1
        - results["dynamic"]["seconds"] / results["fixed"]["seconds"],
    )
    if args.all_fixed:
        comparison.update(
            fastest_fixed=best,
            improvement_vs_fastest_fixed=(
                1 - results["dynamic"]["seconds"] / results[best]["seconds"]
            )
            if best
            else None,
            fastest_fixed_semantics="Ex-post enumeration only; not visible to the policy",
        )
    write_json(args.output / "comparison.json", comparison)
    if any(results[name]["status"] != "complete" for name in ("fixed", "dynamic")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0

"""Run workload-triggered Optimizer revision in the outer CPU Simulation."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from examples.autopilot.elastic.simulate_online import cpu_math_imports


def run_revision_simulation(
    simulation,
    agent,
    output,
    *,
    assembly,
    initial_source,
    trigger_config=None,
    evaluation=None,
    revision=None,
    runner=None,
):
    from areal.autopilot.domains.elastic.cycle import (
        optimizer_simulation_cases,
        revise_optimizer,
    )
    from areal.autopilot.domains.elastic.evaluation.optimizer import (
        OptimizerSimulationBenchmark,
        OptimizerSimulationCriteria,
        VerifiedOnlineOptimizer,
    )
    from areal.autopilot.domains.elastic.trigger import RevisionTrigger
    from areal.autopilot.domains.elastic.versions import load_accepted_optimizer
    from areal.autopilot.revision.contracts import source_digest
    from areal.utils.atomic import atomic_json

    if simulation.control is None:
        raise ValueError("Revision requires a running Reconfigurator")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evaluation, revision = dict(evaluation or {}), dict(revision or {})
    limits = dict(
        max_queries=revision.get("max_queries", 64),
        max_seconds=revision.get("max_seconds", 30),
    )
    current_source = initial_source
    optimizer_settings = dict(revision.get("optimizer_settings", {}))
    simulation.control.optimizer = VerifiedOnlineOptimizer(
        current_source, optimizer_settings, **limits
    )
    trigger = RevisionTrigger(trigger_config)
    pending = None
    records = []
    while not simulation.done and simulation.status == "running":
        simulation.run(until=simulation.now + trigger.config.check_interval_seconds)
        view = simulation.observe()
        change = trigger.observe(view)
        if change is not None:
            pending = change
            simulation.emit("revision_triggered", features=change.features)
        if pending is None or view.busy or view.input_closed or simulation.done:
            continue
        cases = optimizer_simulation_cases(
            view, max_new_tokens=simulation.config.max_new_tokens, **evaluation
        )
        if not cases:
            continue
        started = time.perf_counter()
        cycle_root = output / "revision" / f"cycle-{len(records) + 1:03d}" / "optimizer"
        benchmark = OptimizerSimulationBenchmark(
            cases,
            assembly=assembly,
            cost_models=simulation.cost_models,
            prediction_cost_models=simulation.prediction_cost_models,
            settings=optimizer_settings,
            criteria=OptimizerSimulationCriteria(**revision.get("criteria", {})),
            runner=runner,
            **limits,
        )
        cycle_root.mkdir(parents=True, exist_ok=True)
        atomic_json(
            cycle_root / "inputs.json",
            dict(
                change=asdict(pending),
                cases=[asdict(case) for case in cases],
                assembly=assembly,
                initial_source_sha256=source_digest(current_source),
            ),
        )
        try:
            result = revise_optimizer(
                agent,
                benchmark,
                cycle_root,
                initial_source=current_source,
                workload_change=pending,
                history_root=output / "revision",
                max_rounds=revision.get("max_rounds", 1),
            )
            applied = result["accepted"] > 0
            if applied:
                optimizer = load_accepted_optimizer(cycle_root, benchmark)
                simulation.control.optimizer = optimizer
                simulation.control.search_invalidated()
                current_source = optimizer.source
                simulation.emit(
                    "revision_applied", source_sha256=source_digest(current_source)
                )
            record = dict(
                triggered_at=pending.observed_at,
                evaluated_at=view.now,
                features=pending.features,
                applied=applied,
                source_sha256=source_digest(current_source),
                benchmark_sha256=benchmark.benchmark_sha256,
                revision_wall_seconds=time.perf_counter() - started,
                result=result,
            )
            records.append(record)
            atomic_json(cycle_root / "application.json", record)
        finally:
            trigger.finish()
        pending = None
    result = dict(
        simulation=simulation.result(),
        revisions=records,
        optimizer_sha256=source_digest(current_source),
        trigger=asdict(trigger.config),
        timing="Revision runs synchronously at a stable simulation boundary; its wall time is not added to simulated GPU seconds",
    )
    atomic_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("revision-example.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--model-settings", type=Path)
    parser.add_argument("--agent-command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cpu_math_imports()

    from examples.autopilot.elastic.revision_demo import (
        GOOD_EVALUATOR_SOURCE,
        REDUNDANT_OPTIMIZER_SOURCE,
        demo_inputs,
    )
    from examples.autopilot.elastic.simulate_online import (
        load_workload,
        workload_samples,
    )

    from areal.autopilot.domains.elastic.contracts import BASELINE_SOURCE, decode
    from areal.autopilot.domains.elastic.evaluation.evaluator import (
        historical_case_from_dict,
    )
    from areal.autopilot.domains.elastic.trigger import RevisionTriggerConfig
    from areal.autopilot.revision import (
        CommandRevisionAgent,
        LLMRevisionAgent,
        Proposal,
    )
    from areal.autopilot.revision.execution import load_source
    from areal.elastic.contracts import HandoffConfig, OnlineLayout, SimulationConfig
    from areal.elastic.reconfigurator import build_reconfigurator
    from areal.elastic.serving import ServingSLO
    from areal.elastic.simulation.simulation import OnlineSimulation

    raw = json.loads(args.config.read_text())
    if args.demo:
        from areal.elastic.simulation.workload import WorkloadSpec, generate

        _, _, assembly = demo_inputs()
        assembly.update(raw.get("assembly", {}))
        if "replay" in raw.get("assembly", {}):
            assembly.pop("forecast", None)
        source = GOOD_EVALUATOR_SOURCE
        models = load_source(source).build_cost_models((), {})
        workload = workload_samples(
            generate(WorkloadSpec.decode(raw["demo_workload"])),
            raw.get("tool_seconds_per_unit", {"synthetic-python-work-v1": 1.0}),
        )
        initial_source = REDUNDANT_OPTIMIZER_SOURCE

        class DemoAgent:
            def propose(self, request):
                return Proposal(
                    BASELINE_SOURCE["optimizer"],
                    "Controlled CPU demonstration removes repeated Evaluator queries",
                )

        agent = DemoAgent()
    else:
        if args.profile is None or args.workload is None:
            parser.error("Supply --profile and --workload, or use --demo")
        profile = json.loads(args.profile.read_text())
        assembly = profile["assembly"]
        models = load_source(profile["evaluator_source"]).build_cost_models(
            tuple(
                historical_case_from_dict(case)
                for case in profile.get("calibration_cases", ())
            ),
            profile.get("cost_model_settings", {}),
        )
        workload = load_workload(args.workload, raw.get("tool_seconds_per_unit", {}))
        initial_source = profile.get("optimizer_source", BASELINE_SOURCE["optimizer"])
        agent = None
    if args.model_settings is not None:
        agent = LLMRevisionAgent(**json.loads(args.model_settings.read_text()))
    elif args.agent_command:
        agent = CommandRevisionAgent(tuple(args.agent_command))
    if agent is None:
        parser.error("Supply --model-settings or --agent-command")
    args.output.mkdir(parents=True, exist_ok=False)
    run = decode(SimulationConfig, assembly["run"])
    slo = decode(ServingSLO, assembly["slo"])
    initial = decode(OnlineLayout, raw["initial_layout"])
    reconfigurator = build_reconfigurator(
        "reference",
        candidates=tuple(
            decode(OnlineLayout, layout) for layout in assembly["candidates"]
        ),
        settings=assembly.get("reconfigurator"),
        replay=assembly.get("replay"),
        forecast=assembly.get("forecast"),
        slo=slo,
    )
    simulation = OnlineSimulation(
        workload,
        initial,
        models,
        run,
        decode(HandoffConfig, assembly["handoff"]),
        slo,
        input_closes_at=raw["input_closes_at"],
        reconfigurator=reconfigurator,
        coalesce_decode=True,
    )
    run_revision_simulation(
        simulation,
        agent,
        args.output,
        assembly=assembly,
        initial_source=initial_source,
        trigger_config=RevisionTriggerConfig(**raw.get("trigger", {})),
        evaluation=raw.get("evaluation"),
        revision=raw.get("revision"),
    )


if __name__ == "__main__":
    main()

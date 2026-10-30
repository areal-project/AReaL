# SPDX-License-Identifier: Apache-2.0

"""Analyze AReaL training traces and produce offline tuning guidance."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import yaml

from areal.api.alloc_mode import AllocationValidationError, _AllocationMode

_PHASES: dict[str, tuple[str, ...]] = {
    "rollout": ("train.rollout",),
    "data": ("train.load_bcast",),
    "training": (
        "train.sft_step",
        "train.compute_values",
        "train.ref_logp",
        "train.teacher_logp",
        "train.recompute_logp",
        "train.compute_advantage",
        "train.ppo_update",
        "train.critic_ppo_update",
    ),
    "weight_update": ("train.update_weights",),
    "checkpoint": ("train.save", "train.checkpoint"),
    "memory_transfer": (
        "train.actor_onload",
        "train.actor_offload",
        "train.critic_onload",
        "train.critic_offload",
        "train.ref_onload",
        "train.ref_offload",
        "train.teacher_onload",
        "train.teacher_offload",
        "train.rollout_pause",
        "train.rollout_pause_generation",
        "train.rollout_offload",
        "train.rollout_onload",
        "train.rollout_continue_generation",
        "train.rollout_resume",
    ),
    "evaluation": ("train.eval",),
    "instrumentation": (
        "train.clear_batches",
        "train.log_stats",
    ),
}
_SPAN_TO_PHASE = {span: phase for phase, spans in _PHASES.items() for span in spans}
_EXPECTED_SPANS = frozenset(_SPAN_TO_PHASE)
_RANK_SUFFIX = re.compile(r"-r(\d+)\.jsonl$")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            if isinstance(event, dict):
                yield event


def _worker(event: dict[str, Any], path: Path) -> tuple[str, str]:
    args = event.get("args")
    rank = args.get("rank") if isinstance(args, dict) else None
    if not (isinstance(rank, int | str) and not isinstance(rank, bool)):
        match = _RANK_SUFFIX.search(path.name)
        rank = match.group(1) if match else path.stem
    role = args.get("role", "") if isinstance(args, dict) else ""
    return str(role), str(rank)


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(math.ceil(percentile * len(ordered)) - 1, 0)
    return ordered[index]


def analyze_traces(trace_paths: Sequence[Path]) -> dict[str, Any]:
    """Summarize known trainer spans using the slowest observed worker per span."""
    step_spans: dict[int, dict[str, dict[tuple[str, str], float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    observed_span_roles: dict[str, set[str]] = defaultdict(set)
    files: set[Path] = set()
    observed_spans: set[str] = set()
    event_count = 0

    for raw_path in trace_paths:
        path = raw_path.expanduser().resolve()
        if path in files:
            continue
        files.add(path)
        for event in _read_jsonl(path):
            event_count += 1
            name = event.get("name")
            args = event.get("args")
            if (
                event.get("ph") != "X"
                or not isinstance(name, str)
                or name not in _SPAN_TO_PHASE
                or not isinstance(args, dict)
            ):
                continue
            step = args.get("global_step")
            duration = event.get("dur")
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or step < 0
                or isinstance(duration, bool)
                or not isinstance(duration, int | float)
                or not math.isfinite(duration)
                or duration <= 0
            ):
                continue
            role_rank = _worker(event, path)
            step_spans[step][name][role_rank] += float(duration)
            observed_span_roles[name].add(role_rank[0])
            observed_spans.add(name)

    if not step_spans:
        raise ValueError("No valid train.* phase spans with global_step were found")

    # Aggregate ranks within each role, then use the slowest role before
    # combining sequential operations into their phase.
    span_step_seconds: dict[str, list[float]] = defaultdict(list)
    phase_step_seconds: dict[str, list[float]] = defaultdict(list)
    phase_role_counts: dict[str, list[int]] = defaultdict(list)
    worker_counts: list[int] = []
    for spans in step_spans.values():
        workers = set().union(*(by_worker.keys() for by_worker in spans.values()))
        worker_counts.append(len(workers))
    for step in sorted(step_spans):
        phase_role_seconds: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for span, by_worker in step_spans[step].items():
            phase = _SPAN_TO_PHASE[span]
            role_rank_seconds: dict[str, dict[str, float]] = defaultdict(
                lambda: defaultdict(float)
            )
            for (role, rank), duration in by_worker.items():
                role_rank_seconds[role][rank] += duration
            role_seconds = {
                role: max(rank_durations.values())
                for role, rank_durations in role_rank_seconds.items()
            }
            seconds = max(role_seconds.values()) / 1_000_000
            span_step_seconds[span].append(seconds)
            for role, value in role_seconds.items():
                phase_role_seconds[phase][role] += value / 1_000_000
        for phase, role_seconds in phase_role_seconds.items():
            phase_step_seconds[phase].append(max(role_seconds.values()))
            phase_role_counts[phase].append(len(role_seconds))

    observed_seconds = sum(sum(values) for values in phase_step_seconds.values())
    spans_report = {
        span: {
            "phase": _SPAN_TO_PHASE[span],
            "steps": len(values),
            "total_seconds": sum(values),
            "mean_seconds": statistics.mean(values),
            "p95_seconds": _nearest_rank_percentile(values, 0.95),
        }
        for span, values in sorted(span_step_seconds.items())
    }
    phases_report = {}
    for phase, values in sorted(phase_step_seconds.items()):
        total = sum(values)
        phases_report[phase] = {
            "steps": len(values),
            "total_seconds": total,
            "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values),
            "p95_seconds": _nearest_rank_percentile(values, 0.95),
            "observed_roles_min": min(phase_role_counts[phase]),
            "observed_roles_max": max(phase_role_counts[phase]),
            "share_of_observed_phase_time": (
                total / observed_seconds if observed_seconds else 0.0
            ),
        }
    return {
        "trace_files": [str(path) for path in sorted(files)],
        "event_count": event_count,
        "step_count": len(step_spans),
        "step_ids": sorted(step_spans),
        "role_worker_counts": {
            role: len(
                {
                    worker
                    for spans in step_spans.values()
                    for by_worker in spans.values()
                    for worker in by_worker
                    if worker[0] == role and worker[1].isdigit()
                }
            )
            for role in sorted(set().union(*observed_span_roles.values()))
        },
        "observed_worker_count_min": min(worker_counts),
        "observed_worker_count_max": max(worker_counts),
        "observed_spans": sorted(observed_spans),
        "missing_spans": sorted(_EXPECTED_SPANS - observed_spans),
        "missing_spans_by_role": {
            role: sorted(
                span
                for span in _EXPECTED_SPANS
                if role not in observed_span_roles[span]
            )
            for role in sorted(set().union(*observed_span_roles.values()))
        },
        "spans": spans_report,
        "phases": phases_report,
    }


def summarize_config(config: dict[str, Any]) -> dict[str, Any]:
    sections = {
        name: config.get(name) or {} for name in ("actor", "cluster", "rollout")
    }
    for name, section in sections.items():
        if not isinstance(section, dict):
            raise ValueError(f"Expected '{name}' configuration to be a mapping")
    actor = sections["actor"]
    cluster = sections["cluster"]
    rollout = sections["rollout"]
    microbatch = actor.get("mb_spec") or {}
    if not isinstance(microbatch, dict):
        raise ValueError("Expected 'actor.mb_spec' configuration to be a mapping")
    optimizer = actor.get("optimizer") or {}
    if not isinstance(optimizer, dict):
        raise ValueError("Expected 'actor.optimizer' configuration to be a mapping")
    backend = actor.get("backend") or config.get("allocation_mode")
    if not actor.get("backend") and isinstance(backend, str) and "+" in backend:
        try:
            allocation_mode = _AllocationMode.from_str(backend)
            backend = re.sub(r"\([^)]*\)", "", str(allocation_mode["actor"]), count=1)
            backend = re.sub(r":\s*", ":", backend)
        except (KeyError, StopIteration, ValueError, AllocationValidationError):
            backend = None
    return {
        "experiment_name": config.get("experiment_name"),
        "trial_name": config.get("trial_name"),
        "model": actor.get("path"),
        "backend": backend,
        "cluster": {
            "n_nodes": cluster.get("n_nodes"),
            "n_gpus_per_node": cluster.get("n_gpus_per_node"),
        },
        "microbatch": {
            "n_mbs": microbatch.get("n_mbs"),
            "max_tokens_per_mb": microbatch.get("max_tokens_per_mb"),
            "packing_algorithm": microbatch.get("packing_algorithm"),
        },
        "gradient_checkpointing": actor.get("gradient_checkpointing", False),
        "dtype": actor.get("dtype", "bfloat16"),
        "attn_impl": actor.get("attn_impl", "flash_attention_2"),
        "optimizer_dtype": actor.get("optimizer_dtype", "float32"),
        "optimizer_type": optimizer.get("type", "adam"),
        "disable_dropout": actor.get("disable_dropout", False),
        "offload": actor.get("offload", False),
        "use_lora": actor.get("use_lora", False),
        "max_concurrent_rollouts": rollout.get("max_concurrent_rollouts"),
        "queue_size": rollout.get("queue_size"),
    }


def _benchmark_candidates(
    report_path: Path | None, config: dict[str, Any]
) -> dict[str, Any]:
    if report_path is None:
        return {"status": "not_provided", "candidates": []}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid benchmark JSON in {report_path}: {exc}") from exc
    if not isinstance(report, dict) or not isinstance(report.get("plan"), dict):
        raise ValueError(f"Benchmark report {report_path} has no plan object")
    plan = report["plan"]
    backend = config.get("backend")
    reasons = []
    if not isinstance(backend, str) or not backend.startswith("fsdp:"):
        reasons.append("training config actor backend is not FSDP")
    if config["cluster"].get("n_nodes") != 1:
        reasons.append(
            "single-node results are incompatible with the configured topology"
        )
    model = config.get("model")
    if not model or not plan.get("model"):
        reasons.append("model identity could not be verified")
    elif model != plan["model"]:
        reasons.append("benchmark model does not exactly match actor.path")
    try:
        if "+" in backend or "[" in backend:
            allocation = _AllocationMode.from_str(backend)["actor"]
        else:
            from areal.api.alloc_mode import ModelAllocation

            allocation = ModelAllocation.from_str(backend)
        if (
            allocation.parallel.tensor_parallel_size != 1
            or allocation.parallel.context_parallel_size != 1
        ):
            reasons.append("FSDP benchmark only supports data parallelism")
        if allocation.parallel.world_size != plan.get("world_size"):
            reasons.append(
                "FSDP actor allocation size does not match benchmark world size"
            )
        nodes = config["cluster"].get("n_nodes")
        gpus_per_node = config["cluster"].get("n_gpus_per_node")
        if (
            nodes != 1
            or not isinstance(gpus_per_node, int)
            or isinstance(gpus_per_node, bool)
            or gpus_per_node < 1
            or allocation.parallel.world_size > gpus_per_node
        ):
            reasons.append(
                "FSDP allocation does not fit the configured single-node GPU topology"
            )
        for key, benchmark_value in (
            ("dtype", "bfloat16"),
            ("attn_impl", "flash_attention_2"),
            ("optimizer_dtype", "float32"),
            ("optimizer_type", "adam"),
        ):
            if config.get(key) != benchmark_value:
                reasons.append(
                    f"training config {key}={config.get(key)!r} differs from benchmark {benchmark_value!r}"
                )
        if config.get("offload"):
            reasons.append("FSDP benchmark does not measure CPU offload")
        if config.get("use_lora"):
            reasons.append("FSDP benchmark does not measure LoRA training")
        if config["microbatch"].get("n_mbs", 1) != 1:
            reasons.append("FSDP benchmark uses n_mbs=1")
        if not config.get("disable_dropout"):
            reasons.append("FSDP benchmark disables dropout")
    except (TypeError, ValueError, AllocationValidationError) as exc:
        reasons.append(f"could not verify actor allocation: {exc}")
    if reasons:
        return {"status": "incompatible", "reasons": reasons, "candidates": []}

    trials = report.get("trials", [])
    if not isinstance(trials, list):
        raise ValueError(f"Benchmark report {report_path} has invalid trials")
    candidates = []
    for trial in trials:
        if not isinstance(trial, dict) or trial.get("status") != "ok":
            continue
        speed = trial.get("loss_tokens_per_second")
        latency = trial.get("latency_seconds")
        if (
            isinstance(speed, bool)
            or not isinstance(speed, int | float)
            or not math.isfinite(speed)
            or not isinstance(latency, dict)
            or isinstance(latency.get("mean"), bool)
            or not isinstance(latency.get("mean"), int | float)
            or not math.isfinite(latency["mean"])
        ):
            continue
        ranks = trial.get("ranks") or [{}]
        if not isinstance(ranks, list):
            continue
        rank = ranks[0] if ranks and isinstance(ranks[0], dict) else {}
        engine_config = rank.get("effective_engine_config", {})
        if not isinstance(engine_config, dict):
            continue
        candidates.append(
            {
                "max_tokens_per_mb": trial.get("max_tokens_per_mb"),
                "gradient_checkpointing": trial.get("gradient_checkpointing"),
                "loss_tokens_per_second": speed,
                "mean_step_seconds": latency["mean"],
                "peak_allocated_bytes_per_rank": trial.get(
                    "peak_allocated_bytes_per_rank"
                ),
                "model": engine_config.get("path"),
                "world_size": len(trial.get("peak_allocated_bytes_per_rank") or []),
                "dtype": engine_config.get("dtype"),
                "attn_impl": engine_config.get("attn_impl"),
                "optimizer_type": (engine_config.get("optimizer") or {}).get("type"),
                "device": rank.get("device"),
                "provenance": rank.get("provenance"),
                "synthetic_sft_workload": True,
            }
        )
    candidates.sort(
        key=lambda candidate: candidate["loss_tokens_per_second"], reverse=True
    )
    return {
        "status": "compatible_candidates" if candidates else "no_successful_trials",
        "source": str(report_path.resolve()),
        "measurement_type": "synthetic SFT train_batch; candidate only, validate under RL workload",
        "workload": {
            key: plan.get(key)
            for key in (
                "sequence_lengths",
                "global_batch_size",
                "input_tokens_per_step",
                "loss_tokens_per_step",
                "seed",
            )
        },
        "candidates": candidates,
    }


def _recommendations(
    trace_analysis: dict[str, Any],
    config: dict[str, Any],
    benchmark: dict[str, Any],
) -> list[dict[str, str]]:
    phases = trace_analysis["phases"]
    dominant = max(
        phases, key=lambda phase: phases[phase]["share_of_observed_phase_time"]
    )
    share = phases[dominant]["share_of_observed_phase_time"]
    dominant_span = max(
        (
            span
            for span, metrics in trace_analysis["spans"].items()
            if metrics["phase"] == dominant
        ),
        key=lambda span: trace_analysis["spans"][span]["total_seconds"],
    )
    recommendations = [
        {
            "priority": "investigate",
            "finding": (
                f"{dominant} is the largest observed phase at {share:.1%} of "
                f"instrumented phase time across {trace_analysis['step_count']} steps."
            ),
            "action": _PHASE_ACTIONS[dominant],
            "evidence": f"trace_analysis.spans.{dominant_span}",
        },
        {
            "priority": "info",
            "finding": "Phase spans measure elapsed time and do not identify a GPU, CPU, IO, or communication root cause by themselves.",
            "action": "Capture a scheduled Perfetto/PyTorch profile for representative steps before changing runtime settings.",
            "evidence": "PerfTracer train.* complete events with global_step",
        },
    ]
    candidates = benchmark.get("candidates", [])
    if candidates:
        candidate = candidates[0]
        current_tokens = config["microbatch"].get("max_tokens_per_mb")
        current_gc = config.get("gradient_checkpointing")
        if (
            candidate["max_tokens_per_mb"] != current_tokens
            or candidate["gradient_checkpointing"] != current_gc
        ):
            recommendations.append(
                {
                    "priority": "candidate",
                    "finding": "The fastest compatible synthetic SFT benchmark trial differs from the current microbatch settings and can be tested under RL.",
                    "action": (
                        f"Benchmark-only candidate: test actor.mb_spec.max_tokens_per_mb="
                        f"{candidate['max_tokens_per_mb']} and "
                        f"actor.gradient_checkpointing={str(candidate['gradient_checkpointing']).lower()} "
                        "in a separate RL run; compare memory and training stability before adoption."
                    ),
                    "evidence": "benchmark.candidates[0]",
                    "current": {
                        "actor.mb_spec.max_tokens_per_mb": current_tokens,
                        "actor.gradient_checkpointing": current_gc,
                    },
                    "candidate": {
                        "actor.mb_spec.max_tokens_per_mb": candidate[
                            "max_tokens_per_mb"
                        ],
                        "actor.gradient_checkpointing": candidate[
                            "gradient_checkpointing"
                        ],
                    },
                    "benchmark_settings": {
                        "dtype": candidate.get("dtype"),
                        "attn_impl": candidate.get("attn_impl"),
                        "optimizer_type": candidate.get("optimizer_type"),
                        "world_size": candidate.get("world_size"),
                        "device": candidate.get("device"),
                        "provenance": candidate.get("provenance"),
                        "synthetic_sft_workload": candidate.get(
                            "synthetic_sft_workload"
                        ),
                    },
                    "measured_loss_tokens_per_second": candidate[
                        "loss_tokens_per_second"
                    ],
                    "benchmark_workload": benchmark.get("workload"),
                }
            )
    return recommendations


_PHASE_ACTIONS = {
    "rollout": "Inspect per-session traces and rollout queue/concurrency metrics; compare generation latency and queue wait before changing concurrency.",
    "data": "Inspect input batch preparation, remote data access, and data-loader wait times; compare representative loading spans before tuning workers or IO.",
    "training": "Profile representative actor update steps and inspect token throughput, allocated memory, and communication overlap before changing microbatch settings.",
    "weight_update": "Inspect weight-transfer operations, participating ranks, and synchronization waits; compare transfer duration against engine metrics.",
    "checkpoint": "Inspect checkpoint frequency, staging, and storage throughput; compare save and recovery-checkpoint spans separately.",
    "memory_transfer": "Compare role-specific onload/offload and rollout pause/resume spans; inspect placement and memory ownership transitions before changing colocation.",
    "evaluation": "Compare evaluation duration and frequency with training steps; inspect the evaluation workload before changing cadence.",
    "instrumentation": "Inspect cleanup and metric-export operations separately; check whether logging or storage waits delay step completion.",
}


def analyze(
    config_path: Path, trace_paths: Sequence[Path], benchmark_path: Path | None = None
) -> dict[str, Any]:
    config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config_data, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    config = summarize_config(config_data)
    trace_analysis = analyze_traces(trace_paths)
    benchmark = _benchmark_candidates(benchmark_path, config)
    return {
        "schema_version": 1,
        "analysis_type": "offline_rl_autopilot",
        "config_file": str(config_path.resolve()),
        "configuration": config,
        "trace_analysis": trace_analysis,
        "benchmark": benchmark,
        "recommendations": _recommendations(trace_analysis, config, benchmark),
        "limitations": [
            "Phase duration is evidence of elapsed time, not a root-cause diagnosis.",
            "Missing spans may be disabled, uninstrumented, or absent from the input files.",
            "FSDP benchmark measurements use synthetic SFT tokens and require validation with the target RL workload.",
            "No config is changed and no training job is launched.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    config = report["configuration"]
    traces = report["trace_analysis"]
    lines = [
        "# AReaL Autopilot Analysis",
        "",
        f"Experiment: `{config.get('experiment_name')}`; trial: `{config.get('trial_name')}`.",
        f"Model: `{config.get('model')}`; backend: `{config.get('backend')}`; "
        f"steps: {traces['step_count']}; observed workers/step: "
        f"{traces['observed_worker_count_min']}-{traces['observed_worker_count_max']}.",
        "",
        "## Configuration",
        "",
        f"- Topology: {config['cluster'].get('n_nodes')} node(s) x {config['cluster'].get('n_gpus_per_node')} GPU(s)",
        f"- Microbatch: max tokens/MB={config['microbatch'].get('max_tokens_per_mb')}, n_mbs={config['microbatch'].get('n_mbs')}, packing={config['microbatch'].get('packing_algorithm')}",
        f"- Gradient checkpointing: {config.get('gradient_checkpointing')}",
        f"- Dtype/attention/optimizer: {config.get('dtype')} / {config.get('attn_impl')} / {config.get('optimizer_type')} ({config.get('optimizer_dtype')})",
        f"- Max concurrent rollouts: {config.get('max_concurrent_rollouts')}",
        "",
        "## Observed Phases",
        "",
        "Each operation uses the slowest rank within a role; sequential operations are summed per role, then the slowest role is used for the phase.",
        "",
        "| Phase | Steps | Mean (s) | P95 (s) | Share of observed phase time |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for phase, metrics in traces["phases"].items():
        lines.append(
            f"| {phase} | {metrics['steps']} | {metrics['mean_seconds']:.4f} | "
            f"{metrics['p95_seconds']:.4f} | {metrics['share_of_observed_phase_time']:.1%} |"
        )
    lines.extend(
        [
            "",
            "### Span Details",
            "",
            "| Span | Phase | Steps | Mean (s) | P95 (s) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for span, metrics in traces["spans"].items():
        lines.append(
            f"| `{span}` | {metrics['phase']} | {metrics['steps']} | "
            f"{metrics['mean_seconds']:.4f} | {metrics['p95_seconds']:.4f} |"
        )
    lines.extend(["", "Unobserved known spans:", ""])
    lines.extend(f"- `{span}`" for span in traces["missing_spans"])
    if not traces["missing_spans"]:
        lines.append("- None")
    if traces["missing_spans_by_role"]:
        lines.extend(["", "Unobserved known spans by role:", ""])
        for role, spans in traces["missing_spans_by_role"].items():
            lines.append(f"- `{role}`: {', '.join(f'`{span}`' for span in spans)}")
    lines.extend(["", "## Recommendations", ""])
    for item in report["recommendations"]:
        lines.extend(
            [
                f"- **{item['priority']}** {item['finding']}",
                f"  - Next: {item['action']}",
                f"  - Evidence: `{item['evidence']}`",
            ]
        )
    benchmark = report["benchmark"]
    lines.extend(["", "## FSDP Benchmark", "", f"Status: `{benchmark['status']}`."])
    lines.extend(f"- {reason}" for reason in benchmark.get("reasons", []))
    if benchmark.get("candidates"):
        lines.extend(
            [
                "",
                "Synthetic SFT candidates sorted by measured loss tokens/s; validate under the target RL workload.",
                "",
                "| max_tokens_per_mb | checkpointing | loss tokens/s | mean step (s) | peak allocated GiB/rank |",
                "| ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for candidate in benchmark["candidates"]:
            peak_gib = ", ".join(
                f"{value / 2**30:.2f}"
                for value in (candidate["peak_allocated_bytes_per_rank"] or [])
            )
            lines.append(
                f"| {candidate['max_tokens_per_mb']} | {candidate['gradient_checkpointing']} | "
                f"{candidate['loss_tokens_per_second']:.1f} | "
                f"{candidate['mean_step_seconds']:.4f} | {peak_gib} |"
            )
        if benchmark.get("workload"):
            lines.extend(
                [
                    "",
                    "Benchmark workload: "
                    + ", ".join(
                        f"{key}={value}" for key, value in benchmark["workload"].items()
                    ),
                    "",
                    "Candidate engine: "
                    + ", ".join(
                        f"{key}={benchmark['candidates'][0].get(key)}"
                        for key in (
                            "dtype",
                            "attn_impl",
                            "optimizer_type",
                            "world_size",
                        )
                    ),
                ]
            )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--benchmark-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    inputs = [args.config, *args.traces]
    if args.benchmark_report:
        inputs.append(args.benchmark_report)
    for path in inputs:
        if not path.is_file():
            raise SystemExit(f"input file does not exist: {path}")
    try:
        report = analyze(args.config, args.traces, args.benchmark_report)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(str(exc)) from exc
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

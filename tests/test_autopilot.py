# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from areal.api.cli_args import PerfTracerConfig
from areal.tools.autopilot import analyze, analyze_traces, main, render_markdown
from areal.utils import perf_tracer
from areal.utils.perf_tracer import Category


def _write_trace(path: Path, events: list[dict]) -> None:
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def _event(name: str, rank: int, step: int, duration_us: float) -> dict:
    return {
        "name": name,
        "ph": "X",
        "ts": 100,
        "dur": duration_us,
        "cat": "compute",
        "args": {"rank": rank, "role": "actor", "global_step": step},
    }


def test_analyze_traces_uses_slowest_rank_per_phase_and_step(tmp_path: Path):
    rank0 = tmp_path / "traces-r0.jsonl"
    rank1 = tmp_path / "traces-r1.jsonl"
    _write_trace(
        rank0,
        [
            _event("train.rollout", 0, 2, 2_000_000),
            _event("train.rollout", 0, 2, 1_000_000),
            _event("train.ppo_update", 0, 2, 5_000_000),
            {
                **_event("train.update_weights", 0, 2, 1_000_000),
                "args": {"rank": 0, "role": "actor", "global_step": 2},
            },
            {
                **_event("train.update_weights", 0, 2, 2_000_000),
                "args": {"rank": 0, "role": "rollout", "global_step": 2},
            },
            _event("train.rollout", 0, 3, 3_000_000),
            _event("metadata", 0, 3, 50_000_000),
        ],
    )
    _write_trace(
        rank1,
        [
            _event("train.rollout", 1, 2, 4_000_000),
            _event("train.ppo_update", 1, 2, 2_000_000),
            _event("train.rollout", 1, 3, 1_000_000),
        ],
    )

    report = analyze_traces([rank0, rank1])

    assert report["step_ids"] == [2, 3]
    assert report["observed_worker_count_min"] == 2
    assert report["observed_worker_count_max"] == 3
    assert report["phases"]["rollout"]["steps"] == 2
    assert report["phases"]["rollout"]["total_seconds"] == 7
    assert report["phases"]["training"]["total_seconds"] == 5
    assert report["spans"]["train.rollout"]["total_seconds"] == 7
    assert report["spans"]["train.update_weights"]["total_seconds"] == 2
    assert report["role_worker_counts"] == {"actor": 2, "rollout": 1}
    assert report["phases"]["rollout"]["share_of_observed_phase_time"] == pytest.approx(
        7 / 14
    )
    assert "train.sft_step" in report["missing_spans_by_role"]["actor"]


def test_cli_analyzes_perf_tracer_and_available_benchmark_report(tmp_path: Path):
    model = str(
        Path(__file__).resolve().parents[1]
        / ".."
        / ".codex/artifacts/fsdp-benchmark-1823/checkpoints/Qwen2.5-0.5B"
    )
    model = str(Path(model).resolve())
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "cluster: {n_nodes: 1, n_gpus_per_node: 1}\n"
        "actor:\n"
        f"  path: {model}\n"
        "  backend: fsdp:d1\n"
        "  gradient_checkpointing: true\n"
        "  disable_dropout: true\n"
        "  mb_spec: {n_mbs: 1, max_tokens_per_mb: 256}\n"
    )
    trace_root = tmp_path / "traces"
    tracer_config = PerfTracerConfig(
        experiment_name="autopilot-test",
        trial_name="trial",
        fileroot=str(trace_root),
        enabled=True,
    )
    tracer = perf_tracer.configure(tracer_config, rank=0, role="master")
    try:
        for step in range(2):
            with perf_tracer.trace_scope(
                "train.rollout",
                category=Category.COMPUTE,
                args={"global_step": step},
            ):
                pass
            with perf_tracer.trace_scope(
                "train.ppo_update",
                category=Category.COMPUTE,
                args={"global_step": step},
            ):
                pass
        tracer.save(force=True)
    finally:
        perf_tracer.reset()

    trace_path = next(trace_root.rglob("traces-r0.jsonl"))
    benchmark_path = (
        Path(__file__).resolve().parents[1]
        / ".."
        / ".codex/artifacts/fsdp-benchmark-1823/real-qwen025-sweep/report.json"
    ).resolve()
    if not benchmark_path.is_file():
        benchmark_path = tmp_path / "benchmark-report.json"
        benchmark_path.write_text(
            json.dumps(
                {
                    "plan": {
                        "model": model,
                        "world_size": 1,
                        "sequence_lengths": [256] * 4,
                        "global_batch_size": 4,
                        "input_tokens_per_step": 1024,
                        "loss_tokens_per_step": 1020,
                        "seed": 42,
                    },
                    "trials": [
                        {
                            "status": "ok",
                            "max_tokens_per_mb": 512,
                            "gradient_checkpointing": False,
                            "loss_tokens_per_second": 100.0,
                            "latency_seconds": {"mean": 0.2},
                            "peak_allocated_bytes_per_rank": [1024],
                            "ranks": [
                                {
                                    "effective_engine_config": {
                                        "path": model,
                                        "dtype": "bfloat16",
                                        "attn_impl": "flash_attention_2",
                                        "optimizer": {"type": "adam"},
                                    },
                                    "device": {"name": "test-device"},
                                    "provenance": {"git_revision": "test-revision"},
                                }
                            ],
                        }
                    ],
                }
            )
        )
    output = tmp_path / "autopilot-output"

    assert (
        main(
            [
                "--config",
                str(config_path),
                "--traces",
                str(trace_path),
                "--benchmark-report",
                str(benchmark_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads((output / "report.json").read_text())
    assert report["trace_analysis"]["step_count"] == 2
    assert report["benchmark"]["status"] == "compatible_candidates"
    assert report["benchmark"]["candidates"][0]["dtype"] == "bfloat16"
    assert report["recommendations"][-1]["priority"] == "candidate"
    assert "Synthetic SFT candidates" in (output / "report.md").read_text()


def test_analyze_rejects_traces_without_valid_training_spans(tmp_path: Path):
    path = tmp_path / "empty.jsonl"
    _write_trace(path, [_event("train.rollout", 0, -1, 100)])

    with pytest.raises(ValueError, match="No valid train"):
        analyze_traces([path])


def test_analyze_rejects_malformed_jsonl_with_line_number(tmp_path: Path):
    path = tmp_path / "broken-r0.jsonl"
    path.write_text("{}\n{bad json}\n")

    with pytest.raises(ValueError, match=r"broken-r0.jsonl:2"):
        analyze_traces([path])


def test_analyze_includes_only_compatible_fsdp_candidates(tmp_path: Path):
    model = str(tmp_path / "model")
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "experiment_name: exp",
                "trial_name: trial",
                "cluster: {n_nodes: 1, n_gpus_per_node: 1}",
                "actor:",
                f"  path: {model}",
                "  backend: fsdp:d1",
                "  gradient_checkpointing: true",
                "  disable_dropout: true",
                "  dtype: bfloat16",
                "  attn_impl: flash_attention_2",
                "  optimizer_dtype: float32",
                "  optimizer: {type: adam}",
                "  mb_spec: {n_mbs: 1, max_tokens_per_mb: 256}",
                "rollout: {max_concurrent_rollouts: 8}",
            ]
        )
        + "\n"
    )
    trace = tmp_path / "traces-r0.jsonl"
    _write_trace(trace, [_event("train.rollout", 0, 0, 1000)])
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "plan": {"model": model, "world_size": 1},
                "trials": [
                    {
                        "status": "ok",
                        "max_tokens_per_mb": 512,
                        "gradient_checkpointing": False,
                        "loss_tokens_per_second": 100,
                        "latency_seconds": {"mean": 0.2},
                    }
                ],
            }
        )
    )

    report = analyze(config, [trace], benchmark)

    assert report["benchmark"]["status"] == "compatible_candidates"
    assert "synthetic SFT" in report["benchmark"]["measurement_type"]
    assert report["benchmark"]["candidates"][0]["max_tokens_per_mb"] == 512
    assert any(
        recommendation["priority"] == "candidate"
        for recommendation in report["recommendations"]
    )
    assert "No config is changed" in " ".join(report["limitations"])
    assert "Synthetic SFT candidates" in render_markdown(report)


@pytest.mark.parametrize(
    ("backend", "n_nodes", "world_size", "expected"),
    [
        ("megatron:d1", 1, 1, "training config actor backend is not FSDP"),
        ("fsdp:d2", 1, 1, "allocation size does not match benchmark world size"),
        ("fsdp:d1", 2, 1, "incompatible with the configured topology"),
    ],
)
def test_analyze_excludes_incompatible_benchmark(
    tmp_path: Path, backend: str, n_nodes: int, world_size: int, expected: str
):
    model = str(tmp_path / "model")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"cluster: {{n_nodes: {n_nodes}, n_gpus_per_node: 2}}\n"
        f"actor: {{path: {model}, backend: {backend}, dtype: bfloat16, "
        "attn_impl: flash_attention_2, optimizer_dtype: float32, "
        "optimizer: {type: adam}}\n"
    )
    trace = tmp_path / "traces-r0.jsonl"
    _write_trace(trace, [_event("train.rollout", 0, 0, 1000)])
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps({"plan": {"model": model, "world_size": world_size}})
    )

    report = analyze(config, [trace], benchmark)

    assert report["benchmark"]["status"] == "incompatible"
    assert any(expected in reason for reason in report["benchmark"]["reasons"])


def test_actor_allocation_uses_its_role_not_first_allocation(tmp_path: Path):
    model = str(tmp_path / "model")
    config = tmp_path / "config.yaml"
    config.write_text(
        "cluster: {n_nodes: 1, n_gpus_per_node: 1}\n"
        f"allocation_mode: 'sglang:d1+fsdp[actor]:d1'\n"
        f"actor: {{path: {model}, dtype: bfloat16, attn_impl: flash_attention_2, "
        "optimizer_dtype: float32, optimizer: {type: adam}, disable_dropout: true, "
        "mb_spec: {n_mbs: 1}}\n"
    )
    trace = tmp_path / "traces-r0.jsonl"
    _write_trace(trace, [_event("train.rollout", 0, 0, 1000)])
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "plan": {"model": model, "world_size": 1},
                "trials": [
                    {
                        "status": "ok",
                        "max_tokens_per_mb": 512,
                        "gradient_checkpointing": False,
                        "loss_tokens_per_second": 100,
                        "latency_seconds": {"mean": 0.2},
                    }
                ],
            }
        )
    )

    report = analyze(config, [trace], benchmark)

    assert report["configuration"]["backend"] == "fsdp:d1"
    assert report["benchmark"]["status"] == "compatible_candidates"


def test_sft_trace_groups_data_load_separately(tmp_path: Path):
    trace = tmp_path / "traces-r0.jsonl"
    _write_trace(
        trace,
        [
            _event("train.load_bcast", 0, 0, 2_000_000),
            _event("train.sft_step", 0, 0, 8_000_000),
        ],
    )

    report = analyze_traces([trace])

    assert report["phases"]["data"]["total_seconds"] == 2
    assert report["phases"]["training"]["total_seconds"] == 8


def test_cli_writes_json_and_markdown_and_preserves_existing_output(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text("actor: {backend: fsdp:d1}\n")
    trace = tmp_path / "traces-r0.jsonl"
    _write_trace(trace, [_event("train.rollout", 0, 0, 1000)])
    output = tmp_path / "report"

    assert (
        main(
            [
                "--config",
                str(config),
                "--traces",
                str(trace),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads((output / "report.json").read_text())["schema_version"] == 1
    assert (output / "report.md").is_file()
    with pytest.raises(SystemExit, match="already exists"):
        main(
            [
                "--config",
                str(config),
                "--traces",
                str(trace),
                "--output-dir",
                str(output),
            ]
        )

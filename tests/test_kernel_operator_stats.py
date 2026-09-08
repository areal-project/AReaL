# SPDX-License-Identifier: Apache-2.0

import gzip
import json

import pytest

from areal.tools.kernel_operator_stats import attribute_gpu_events, summarize_traces


def event(name, cat, ts, dur, *, pid=1, tid=1, correlation=None):
    return dict(
        name=name,
        cat=cat,
        ph="X",
        ts=ts,
        dur=dur,
        pid=pid,
        tid=tid,
        args={} if correlation is None else {"correlation": correlation},
    )


def test_nested_operators_async_kernels_are_counted_once():
    events = [
        event("parent", "cpu_op", 0, 100),
        event("child", "cpu_op", 10, 20),
        event("wrong thread", "cpu_op", 12, 5, tid=2),
        event("wrong process", "cpu_op", 12, 5, pid=2),
        event("annotation", "user_annotation", 13, 3),
        event("cudaLaunchKernel", "cuda_runtime", 14, 2, correlation=7),
        event("cuLaunchKernel", "cuda_driver", 14.5, 1, correlation=7),
        event("kernel1", "kernel", 200, 11, pid=0, correlation=7),
        event("kernel2", "kernel", 210, 13, pid=0, correlation=7),
        # Child's end is exclusive; this launch belongs to the parent.
        event("cudaLaunchKernel", "cuda_runtime", 30, 2, correlation=8),
        event("kernel3", "kernel", 230, 5, pid=0, correlation=8),
    ]
    rows, quality = attribute_gpu_events(events)
    by_name = {row["cpu_operator"]: row for row in rows}
    assert by_name["child"]["gpu_time_us"] == 24
    assert by_name["child"]["gpu_events"] == 2
    assert by_name["child"]["cpu_calls_with_gpu"] == 1
    assert by_name["parent"]["gpu_time_us"] == 5
    assert quality["aggregated_gpu_events"] == 3
    assert quality["aggregated_gpu_time_us"] == {"kernel": 29}


def test_missing_or_reused_launches_remain_unattributed():
    events = [
        event("parent", "cpu_op", 0, 100),
        event("launch1", "cuda_runtime", 1, 1, correlation=7),
        event("launch2", "cuda_runtime", 4, 1, correlation=7),
        event("ambiguous", "kernel", 20, 3, correlation=7),
        event("missing", "kernel", 22, 4, correlation=8),
        event("launch3", "cuda_runtime", 101, 1, correlation=9),
        event("outside operator", "gpu_memcpy", 102, 5, correlation=9),
    ]
    rows, quality = attribute_gpu_events(events)
    assert all(row["cpu_operator"] == "[unattributed]" for row in rows)
    assert quality["attribution_counts"] == {
        "ambiguous_launch": 1,
        "no_launch": 1,
        "no_cpu_operator": 1,
    }
    assert quality["input_gpu_time_us"] == quality["aggregated_gpu_time_us"]


@pytest.mark.parametrize("compressed", [False, True])
def test_separate_traces_reused_correlations_merge_only_aggregates(
    tmp_path, compressed
):
    paths = []
    for rank in range(2):
        path = tmp_path / (f"rank{rank}.json" + (".gz" if compressed else ""))
        events = [
            event("operator", "cpu_op", 0, 10),
            event("launch", "cuda_runtime", 1, 1, correlation=7),
            event("kernel", "kernel", 2, rank + 1, correlation=7),
        ]
        opener = gzip.open if compressed else open
        with opener(path, "wt") as fout:
            json.dump({"traceEvents": events}, fout)
        paths.append(path)
    quality = summarize_traces(paths + paths, tmp_path / "out")
    assert len(quality) == 2
    import csv

    with (tmp_path / "out/operators-all-traces.csv").open() as fin:
        rows = list(csv.DictReader(fin))
    assert len(rows) == 1
    assert float(rows[0]["gpu_time_us"]) == 3
    assert int(rows[0]["gpu_events"]) == 2
    assert int(rows[0]["cpu_calls_with_gpu"]) == 2


def test_communication_collectives_and_process_groups_remain_separate(tmp_path):
    import csv

    events = []
    cases = [
        (
            "alltoall",
            "EXPERT_MODEL_PARALLEL_GROUP",
            "ep",
            8,
            "[0, 1, 2, 3, 4, 5, 6, 7]",
        ),
        (
            "allreduce",
            "EXPERT_MODEL_PARALLEL_GROUP",
            "ep",
            8,
            "[0, 1, 2, 3, 4, 5, 6, 7]",
        ),
        ("allreduce", "TENSOR_MODEL_PARALLEL_GROUP", "tp0", 2, "[0, 1]"),
        ("allreduce", "TENSOR_MODEL_PARALLEL_GROUP", "tp1", 2, "[2, 3]"),
    ]
    for i, (collective, group, name, size, ranks) in enumerate(cases):
        op = event("record_param_comms", "cpu_op", i * 10, 5)
        op["args"] = {
            "Collective name": collective,
            "Process Group Description": group,
            "Process Group Name": name,
            "Group size": size,
            "Process Group Ranks": ranks,
        }
        events.extend(
            [
                op,
                event("launch", "cuda_runtime", i * 10 + 1, 1, correlation=i),
                event("nccl", "kernel", i * 10 + 2, i + 1, correlation=i),
            ]
        )
    paths = [tmp_path / f"rank{i}.json" for i in range(2)]
    for path in paths:
        path.write_text(json.dumps({"traceEvents": events}))
    summarize_traces(paths, tmp_path / "out")
    with (tmp_path / "out/operators-all-traces.csv").open() as fin:
        rows = list(csv.DictReader(fin))
    assert len(rows) == 4
    assert sum(float(r["gpu_time_us"]) for r in rows) == 20
    for row in rows:
        assert int(row["gpu_events"]) == 2
        assert (
            row["collective"],
            row["process_group"],
            row["group_name"],
            int(row["group_size"]),
            row["group_ranks"],
        ) in cases


def test_components_use_semantic_ancestry_without_inclusive_double_counting(tmp_path):
    from areal.tools.kernel_operator_stats import ComponentRules

    rules = ComponentRules(
        {
            "cpu_scopes": [{"pattern": "Attention", "component": "ATTN"}],
            "gpu_kernels": [{"pattern": "rmsnorm", "component": "NORM"}],
            "communications": [
                {"match": {"Process Group Description": "EP"}, "component": "MOE_COMM"}
            ],
        }
    )
    comm = event("record_param_comms", "cpu_op", 40, 5)
    comm["args"] = {"Process Group Description": "EP"}
    events = [
        event("Attention", "cpu_op", 0, 100),
        event("aten::mm", "cpu_op", 10, 10),
        event("launch", "cuda_runtime", 11, 1, correlation=1),
        event("gemm", "kernel", 200, 7, correlation=1),
        event("rmsnorm", "kernel", 210, 3, correlation=1),
        comm,
        event("launch", "cuda_runtime", 41, 1, correlation=2),
        event("nccl", "kernel", 220, 11, correlation=2),
        event("copy", "gpu_memcpy", 230, 2, correlation=1),
    ]
    rows, quality = attribute_gpu_events(events, rules)
    assert {r["component"]: r["gpu_time_us"] for r in rows} == {
        "ATTN": 7,
        "NORM": 3,
        "MOE_COMM": 11,
        "MEMCPY": 2,
    }
    assert (
        next(r for r in rows if r["component"] == "ATTN")["cpu_operator"] == "aten::mm"
    )
    assert quality["aggregated_gpu_events"] == 4
    assert quality["aggregated_gpu_time_us"] == {"kernel": 21, "gpu_memcpy": 2}

    raw_trace = tmp_path / "trace.json"
    raw_trace.write_text(json.dumps({"traceEvents": events}))
    summarize_traces([raw_trace], tmp_path / "out", rules)
    import csv

    with (tmp_path / "out/component-summary.csv").open() as fin:
        summary = list(csv.DictReader(fin))
    assert sum(int(r["gpu_events"]) for r in summary) == 4
    assert sum(float(r["gpu_time_us"]) for r in summary) == 23
    assert sum(float(r["all_gpu_time_share_pct"]) for r in summary) == pytest.approx(
        100
    )

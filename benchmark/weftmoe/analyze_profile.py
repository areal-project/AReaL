"""Attribute CUDA work to MoE forward scopes and their autograd sequence IDs.

Accept traces from training_profile.py. Report both GPU interval unions
and each layer's first-to-last GPU-work envelope (including launch gaps/waits).
Overlapping streams are not added for a wall-time percentage. The capture is
one warm microbatch, not an entire optimizer step. DDP/optimizer are excluded.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from areal.utils import logging

logger = logging.getLogger("PerfTracer")


def union_us(intervals):
    end = float("-inf")
    total = 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def analyze(path):
    trace = json.loads(path.read_text())
    events = trace["traceEvents"]
    threads = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in ("cpu_op", "user_annotation"):
            threads[(event["pid"], event["tid"])].append(event)
    # Build CPU interval trees. An autograd sequence links the backward node
    # to the corresponding operation inside a forward MoE scope.
    nodes, sequences = [], {}
    main_tid = next(event["tid"] for event in events if event.get("name") == "moe/full")
    backward_nodes = sorted(
        (
            event
            for event in events
            if event.get("cat") == "cpu_op"
            and event["name"].startswith("autograd::engine::evaluate_function:")
        ),
        key=lambda event: event["ts"],
    )
    main_id = next(
        event["args"]["Fwd thread id"]
        for event in backward_nodes
        if event.get("args", {}).get("Fwd thread id", 0) > 0
    )
    recompute_ids = {
        event["args"]["Fwd thread id"]
        for event in backward_nodes
        if event.get("args", {}).get("Fwd thread id", 0) not in (0, main_id)
    }
    assert len(recompute_ids) <= 1, "This analyzer expects PP1 with one autograd worker"
    recompute_id = next(iter(recompute_ids), None)
    layer_index = -1
    for thread in threads.values():
        stack = []
        for event in sorted(thread, key=lambda e: (e["ts"], -e["dur"])):
            while stack and event["ts"] >= stack[-1]["stop"]:
                stack.pop()
            parent = stack[-1] if stack else None
            name = event["name"]
            category = parent["category"] if parent else None
            layer = parent["layer"] if parent else None
            if name == "moe/full":
                category = "routed"
                layer_index += 1
                layer = layer_index
            elif name == "moe/router":
                category = "router"
            elif name == "moe/shared":
                category = "shared"
            elif name.startswith("moe/routed/"):
                category = "routed"
            elif name.startswith("diagnostic/"):
                category = "diagnostic"
            node = dict(
                event=event,
                parent=parent,
                category=category,
                layer=layer,
                stop=event["ts"] + event["dur"],
                phase="fwd",
            )
            nodes.append(node)
            stack.append(node)
            seq = event.get("args", {}).get("Sequence number")
            if event["tid"] == main_tid and seq is not None and seq >= 0:
                # Even non-MoE ops must replace the next sequence's tentative
                # ownership; no-grad ops can carry that same sequence number.
                sequences[(main_id, seq)] = (category, layer)
    external = {}
    for node in nodes:
        event, parent = node["event"], node["parent"]
        name, args = event["name"], event.get("args", {})
        if name.startswith("autograd::engine::evaluate_function:"):
            node["phase"] = "bwd"
            node["category"], node["layer"] = sequences.get(
                (args.get("Fwd thread id"), args.get("Sequence number")), (None, None)
            )
        elif parent and parent["phase"] == "bwd":
            node["phase"] = "bwd"
            if node["category"] is None:
                node["category"] = parent["category"]
                node["layer"] = parent["layer"]
        # Selective checkpointing builds a new autograd graph while backward
        # is running. Its recomputed forward operations need ownership too.
        seq = args.get("Sequence number")
        if (
            event["tid"] != main_tid
            and not args.get("Fwd thread id", 0)
            and seq is not None
            and seq >= 0
        ):
            sequences[(recompute_id, seq)] = (node["category"], node["layer"])
        if "External id" in args:
            external[args["External id"]] = node
    work = defaultdict(list)
    kernels = defaultdict(Counter)
    gpu = []
    unmapped = Counter()
    layer_work = defaultdict(list)
    layer_routed = defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in (
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
        ):
            continue
        interval = (event["ts"], event["ts"] + event["dur"])
        node = external.get(event.get("args", {}).get("External id"))
        if node and node["category"] == "diagnostic":
            continue
        gpu.append(interval)
        if node and node["category"]:
            key = node["phase"] + "/" + node["category"]
        else:
            key = "other"
            unmapped[event["name"]] += event["dur"]
        work[key].append(interval)
        kernels[key][event["name"]] += event["dur"]
        if node and node["layer"] is not None:
            layer_work[(node["phase"], node["layer"])].append(interval)
            if node["category"] == "routed":
                layer_routed[(node["phase"], node["layer"])].append(interval)
    total_span = max(b for a, b in gpu) - min(a for a, b in gpu)
    moe = [
        interval
        for key, intervals in work.items()
        if key != "other"
        for interval in intervals
    ]
    routed = [
        interval
        for key, intervals in work.items()
        if key.endswith("/routed")
        for interval in intervals
    ]

    def spans(groups):
        return [
            (min(a for a, b in values), max(b for a, b in values))
            for values in groups.values()
        ]

    full_spans, routed_spans = spans(layer_work), spans(layer_routed)
    return dict(
        trace=str(path),
        scope="one warm packed microbatch, forward+backward; profiler enabled",
        gpu_span_ms=total_span / 1000,
        gpu_busy_union_ms=union_us(gpu) / 1000,
        full_moe_union_ms=union_us(moe) / 1000,
        routed_moe_union_ms=union_us(routed) / 1000,
        full_moe_share_of_gpu_span=union_us(moe) / total_span,
        routed_moe_share_of_gpu_span=union_us(routed) / total_span,
        full_moe_envelope_ms=union_us(full_spans) / 1000,
        routed_moe_envelope_ms=union_us(routed_spans) / 1000,
        full_moe_envelope_share=union_us(full_spans) / total_span,
        routed_moe_envelope_share=union_us(routed_spans) / total_span,
        layer_envelopes=len(full_spans),
        category_union_ms={key: union_us(value) / 1000 for key, value in work.items()},
        category_kernel_sum_ms={
            key: sum(b - a for a, b in value) / 1000 for key, value in work.items()
        },
        top_kernels_ms={
            key: [(name, us / 1000) for name, us in counts.most_common(30)]
            for key, counts in kernels.items()
        },
        unattributed_top_ms=[
            (name, us / 1000) for name, us in unmapped.most_common(40)
        ],
        forward_sequence_ids=len(sequences),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.traces:
        row = analyze(path)
        rows.append(row)
        logger.info(
            json.dumps(
                {
                    key: value
                    for key, value in row.items()
                    if key not in ("top_kernels_ms", "unattributed_top_ms")
                }
            )
        )
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()

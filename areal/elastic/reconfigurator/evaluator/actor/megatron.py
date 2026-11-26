# SPDX-License-Identifier: Apache-2.0

"""Megatron microbatch execution, pipeline dependencies and PPO stage timing."""

from __future__ import annotations

from dataclasses import replace

from areal.elastic.reconfigurator.evaluator.contracts import (
    Estimate,
    TimedTask,
    critical_path,
)


def pipeline_time(
    forward: list[list[float]], backward: list[list[float]] | None
) -> float:
    """Non-interleaved 1F1B dependencies, or forward-only pipeline; seconds."""
    stages, microbatches = (len(forward), len(forward[0]))
    tasks = []
    for stage in range(stages):
        if backward is None:
            order = [("f", m) for m in range(microbatches)]
        else:
            warmup = min(stages - stage - 1, microbatches)
            order = [("f", m) for m in range(warmup)]
            for m in range(microbatches - warmup):
                order.extend((("f", m + warmup), ("b", m)))
            order.extend(("b", m) for m in range(microbatches - warmup, microbatches))
        previous = None
        for direction, m in order:
            name = f"{direction}:{stage}:{m}"
            deps = [previous] if previous else []
            if direction == "f" and stage:
                deps.append(f"f:{stage - 1}:{m}")
            if direction == "b":
                deps.append(f"f:{stage}:{m}")
                if stage + 1 < stages:
                    deps.append(f"b:{stage + 1}:{m}")
            seconds = forward[stage][m] if direction == "f" else backward[stage][m]
            tasks.append(TimedTask(name, Estimate(seconds, "assumed"), tuple(deps)))
            previous = name
    return max((end for _, _, end in critical_path(tuple(tasks))), default=0.0)


def estimate_logprob(plan, contexts, backend):
    return (
        _transformer_seconds(plan.logprob, contexts, backend, False)
        if plan.logprob
        else 0.0
    )


def estimate_train(plan, contexts, backend):
    train = 0.0
    for ranks in plan.train:
        train += _transformer_seconds(ranks, contexts, backend, True)
        updates = [backend.optimizer_update(ctx) for ctx in contexts]
        train += max(sum(u.values()) for u in updates)
    return train


def _transformer_seconds(ranks, contexts, backend, training):
    forward = [[] for _ in contexts]
    backward = [[] for _ in contexts]
    for s, ctx in enumerate(contexts):
        for index in range(len(ranks[0])):
            parts = []
            for rank in ranks:
                local = replace(ctx, microbatch_index=index, training=training)
                f = backend.forward_cost(local, rank[index])
                ftime = f["compute"] + f["communication_exposed"]
                total = backend.execution_cost(local, rank[index], training=training)
                ttime = total["compute"] + total["communication_exposed"]
                parts.append((ftime, max(0.0, ttime - ftime)))
            forward[s].append(max(v[0] for v in parts))
            backward[s].append(max(v[1] for v in parts))
    return _pipeline_time(forward, backward if training else None)


_pipeline_time = pipeline_time

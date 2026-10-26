# SPDX-License-Identifier: Apache-2.0

"""Backend mesh rules, layer placement and non-interleaved execution dependencies."""

from __future__ import annotations

from ...core import Estimate, TimedTask, critical_path
from .config import ActorConfig, ModelShape


def mesh_is_valid(config: ActorConfig, model: ModelShape) -> bool:
    p, m = config.parallel, model
    invalid = m.experts % p.ep or m.attention_heads % p.tp
    if m.mlp_type == "dense":
        invalid = invalid or p.ep != 1 or m.intermediate_size % p.tp
        if config.backend == "megatron":
            # Initial dense adapter covers the planned TP1/2/4/8, no KV replication.
            invalid = invalid or m.kv_heads % p.tp or p.pp != 1 or p.cp != 1
    if config.backend == "archon":
        invalid = invalid or m.kv_heads % p.tp or (m.attention_heads // p.tp) % p.cp
        kv = m.kv_heads // p.tp
        if not invalid:
            invalid = (
                kv % p.cp
                if kv >= p.cp
                else p.cp % kv or (m.attention_heads // m.kv_heads) % (p.cp // kv)
            )
    return not invalid


def pipeline_partition(config: ActorConfig, model: ModelShape) -> tuple[int, ...]:
    if config.pipeline_layers is not None:
        if sum(config.pipeline_layers) != model.layers:
            raise ValueError("pipeline_layers must sum to model.layers")
        return config.pipeline_layers
    if config.parallel.pp == 1:
        return (model.layers,)
    pp = config.parallel.pp
    if config.backend == "archon":
        first, last = (
            config.pipeline_first_less_layers,
            config.pipeline_last_less_layers,
        )
        count, extra = divmod(model.layers + first + last, pp)
        if count < max(first, last, 1):
            raise ValueError("Archon PP stage overhead exceeds its effective layers")
        return tuple(
            count + (i < extra) - (first if i == 0 else last if i == pp - 1 else 0)
            for i in range(pp)
        )
    per_layer = (
        model.hidden_size
        * model.head_dim
        * (model.attention_heads + 2 * model.kv_heads)
        + model.hidden_size**2
        + 2 * model.hidden_size
        + model.router_parameters_per_layer
        + model.expert_parameters_per_layer / config.parallel.ep
    )
    table = {0: (0.0, ())}
    for stage in range(pp):
        next_table = {}
        for used, (cost, counts) in table.items():
            for count in range(1, model.layers - used - (pp - stage - 1) + 1):
                head = model.head_parameters * ((stage == 0) + (stage == pp - 1))
                candidate = (max(cost, count * per_layer + head), counts + (count,))
                if (
                    used + count not in next_table
                    or candidate < next_table[used + count]
                ):
                    next_table[used + count] = candidate
        table = next_table
    if model.layers not in table:
        raise ValueError("Pipeline stages exceed model layers")
    return table[model.layers][1]


def pipeline_time(
    forward: list[list[float]], backward: list[list[float]] | None
) -> float:
    """Non-interleaved 1F1B dependencies, or forward-only pipeline; seconds."""
    stages, microbatches = len(forward), len(forward[0])
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

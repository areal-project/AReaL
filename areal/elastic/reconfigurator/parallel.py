# SPDX-License-Identifier: Apache-2.0

"""CPU worker entry point; each candidate restores its own private simulation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

_cost_models = None
_identity = None
_search = None


@dataclass(frozen=True)
class ChunkResult:
    indices: tuple[int, ...]
    predictions: tuple
    task_cpu_seconds: float
    pid: int
    input_preparations: int = 0


def evaluate_chunk(job):
    started = time.process_time()
    from areal.elastic.reconfigurator import component_module

    global _cost_models, _identity, _search
    payload, indexed = job
    implementation, payload = payload[0], payload[1:]
    if implementation not in {"basic", "reference"}:
        raise ValueError("Unknown reconfigurator implementation")
    module = component_module(implementation, "evaluator")
    Evaluator = (
        module.Evaluator if implementation == "reference" else module.JointEvaluator
    )
    CalibratedCostModels = component_module(
        implementation, "cost_models"
    ).CalibratedCostModels
    spec, expected, config, handoff, slo, delay, prune = payload[:7]
    if implementation == "basic":
        coalesce, view, forecasts = payload[7:10]
        options = payload[10:]
        legacy = (coalesce,)
    else:
        view, forecasts = payload[7:9]
        options = payload[9:]
        legacy = ()
    identity = implementation, spec, expected
    if identity != _identity:
        if spec[0] == "analytical":
            from areal.elastic.reconfigurator.reference.evaluator.cost_model.loader import (
                AnalyticalCostModels,
                Catalog,
            )

            _, gpu, topology, model, actor, rollout, bandwidth, link, root = spec
            _cost_models = AnalyticalCostModels.from_profiles(
                hardware=gpu,
                topology=topology,
                model=model,
                actor=actor,
                rollout=rollout,
                bandwidth_GBps=bandwidth,
                inter_node_GBps=link,
                catalog=Catalog(Path(root)),
            )
        else:
            bundle, mechanism, publication, bandwidth, *cross_node = spec
            _cost_models = CalibratedCostModels(
                Path(bundle),
                Path(mechanism),
                Path(publication),
                bandwidth,
                inter_node_dp_GBps=cross_node[0] if cross_node else None,
            )
        if _cost_models.source_ids != expected:
            raise ValueError("CPU worker loaded different component models")
        _identity = identity
        _search = None
    context = options[0] if options else None
    reserve = options[1] if len(options) > 1 else 0.0
    scope = options[2] if len(options) > 2 else "all_completed"
    key = (
        identity,
        config,
        handoff,
        slo,
        delay,
        prune,
        legacy,
        context,
        reserve,
        scope,
    )
    if context is None or _search is None or _search[0] != key:
        evaluator = Evaluator(
            _cost_models,
            config,
            handoff,
            slo,
            delay,
            prune_irreversible_slo=prune,
            **(
                dict(actor_memory_reserve_fraction=reserve, actor_memory_scope=scope)
                if implementation == "reference"
                else dict(coalesce_decode=coalesce)
            ),
        )
        if context is not None:
            # Keep the first immutable DTOs for this content identity. Repeated
            # chunk unpickling otherwise defeats the evaluator's identity cache.
            _search = key, evaluator, view, forecasts
    else:
        _, evaluator, view, forecasts = _search
    before = getattr(evaluator, "input_preparations", 0)
    predictions = tuple(
        evaluator.evaluate(view, forecasts, layout, bound=options[3])
        if len(options) > 3
        else evaluator.evaluate(view, forecasts, layout)
        for _, layout in indexed
    )
    return ChunkResult(
        tuple(index for index, _ in indexed),
        predictions,
        time.process_time() - started,
        os.getpid(),
        getattr(evaluator, "input_preparations", 0) - before,
    )

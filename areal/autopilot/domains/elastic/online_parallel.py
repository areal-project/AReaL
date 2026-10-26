# SPDX-License-Identifier: Apache-2.0

"""CPU worker entry point; each candidate restores its own private simulation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

_models = None
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
    from .online_models import CalibratedOnlineModels
    from .online_simulation import OnlineForecastModel

    global _models, _identity, _search
    payload, indexed = job
    spec, expected, config, handoff, slo, delay, prune, coalesce, view, forecasts = (
        payload[:10]
    )
    identity = spec, expected
    if identity != _identity:
        bundle, mechanism, publication, bandwidth, *cross_node = spec
        _models = CalibratedOnlineModels(
            Path(bundle),
            Path(mechanism),
            Path(publication),
            bandwidth,
            inter_node_dp_GBps=cross_node[0] if cross_node else None,
        )
        if _models.source_ids != expected:
            raise ValueError("CPU worker loaded a different frozen calibration")
        _identity = identity
        _search = None
    context = payload[10] if len(payload) > 10 else None
    reserve = payload[11] if len(payload) > 11 else 0.0
    scope = payload[12] if len(payload) > 12 else "all_completed"
    key = (
        identity,
        config,
        handoff,
        slo,
        delay,
        prune,
        coalesce,
        context,
        reserve,
        scope,
    )
    if context is None or _search is None or _search[0] != key:
        evaluator = OnlineForecastModel(
            _models,
            config,
            handoff,
            slo,
            delay,
            prune_irreversible_slo=prune,
            coalesce_decode=coalesce,
            actor_memory_reserve_fraction=reserve,
            actor_memory_scope=scope,
        )
        if context is not None:
            # Keep the first immutable DTOs for this content identity. Repeated
            # chunk unpickling otherwise defeats the evaluator's identity cache.
            _search = key, evaluator, view, forecasts
    else:
        _, evaluator, view, forecasts = _search
    before = getattr(evaluator, "input_preparations", 0)
    predictions = tuple(
        evaluator.evaluate(view, forecasts, layout, bound=payload[13])
        if len(payload) > 13
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

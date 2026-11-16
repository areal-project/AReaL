# SPDX-License-Identifier: Apache-2.0

"""Reference reconfiguration for native and simulated execution."""

from __future__ import annotations

from importlib import import_module

from areal.elastic.reconfigurator.evaluator.contracts import (
    ExecutionConfig,
    Request,
)
from areal.elastic.reconfigurator.evaluator.cost_model.actor.model import (
    ActorEstimate,
    AdvantageModel,
    AdvantageObservation,
    AdvantageRankWork,
    AdvantageTerm,
    AdvantageWorkload,
    AggregateAdvantageModel,
    ExpertActorModel,
    ObservedPacking,
)
from areal.elastic.reconfigurator.evaluator.cost_model.actor.operators import (
    Efficiency,
)
from areal.elastic.reconfigurator.evaluator.cost_model.actor.packing import (
    DispatchConfig,
    Microbatch,
    ObservedPackingWork,
    PackingComparison,
    PackingError,
    PackingMicrobatch,
    PackingPlan,
    PackingRecord,
    PackingSummary,
    RankPacking,
)
from areal.elastic.reconfigurator.evaluator.cost_model.contracts import (
    ActorConfig,
    AllocationChange,
    CalibrationPoint,
    Estimate,
    Hardware,
    InferenceRates,
    ModelShape,
    ParallelLayout,
    PhaseCalibration,
    RolloutConfig,
    ServiceEstimate,
    ServiceSlice,
    ServiceState,
    TransferTask,
    TransitionProfile,
)
from areal.elastic.reconfigurator.evaluator.cost_model.loader import (
    AssembledTrial,
    Catalog,
)
from areal.elastic.reconfigurator.evaluator.cost_model.publication import (
    ExpertPublicationModel,
    ExpertTransitionModel,
    PublicationEstimate,
    PublicationFlow,
    PublicationRank,
    PublicationRates,
    PublicationWork,
    TransitionEstimate,
    publication_binding,
)
from areal.elastic.reconfigurator.evaluator.cost_model.rollout.model import (
    ExpertRolloutModel,
)
from areal.elastic.reconfigurator.reconfigurator import (
    OnlineReconfigurator as OnlineReconfigurator,
)
from areal.elastic.reconfigurator.reconfigurator import (
    OnlineReconfiguratorConfig as OnlineReconfiguratorConfig,
)

IMPLEMENTATIONS = ("reference",)


def validate_reconfigurator(name: str) -> None:
    if name not in IMPLEMENTATIONS:
        raise ValueError("reconfigurator must be reference")


def component_module(name: str, component: str):
    validate_reconfigurator(name)
    if component not in {
        "reconfigurator",
        "optimizer",
        "forecast",
        "history",
        "cost_models",
        "evaluator",
    }:
        raise ValueError("Unknown reconfigurator component")
    module = {
        "history": "forecast",
        "cost_models": "evaluator.cost_model.loader",
    }.get(component, component)
    return import_module(f"{__name__}.{module}")


def build_reconfigurator(
    name: str,
    *,
    candidates,
    settings: dict | None = None,
    replay: dict | None = None,
    forecast: dict | None = None,
    slo=None,
    optimizer=None,
    search_space: dict | None = None,
):
    """Build the Reference algorithm for native or simulated execution."""
    from ..serving import ServingSLO

    module = component_module(name, "reconfigurator")
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("A reconfigurator requires explicit executable candidates")
    if replay is not None and forecast is not None:
        raise ValueError("Choose replay or sampled forecasting")
    if forecast is not None:
        predictor = component_module(name, "forecast")
        forecaster = predictor.HistoricalForecaster(
            predictor.ForecastConfig(**forecast)
        )
    else:
        predictor = component_module(name, "history")
        forecaster = predictor.HistoricalReplay(
            predictor.HistoryReplayConfig(**(replay or {}))
        )
    options = dict(
        check_interval_seconds=10.0,
        slo_window_seconds=120.0,
        cooldown_seconds=0.0,
        decision_delay_seconds=0.0,
        minimum_completed_requests=8,
        max_evaluations=64 if search_space is not None else len(candidates),
    )
    options.update(settings or {})
    if slo is None:
        slo = ServingSLO(5.0, 0.05, 0.8)
    elif isinstance(slo, dict):
        slo = ServingSLO(**slo)
    result = module.OnlineReconfigurator(
        candidates,
        forecaster,
        module.OnlineReconfiguratorConfig(**options),
        slo,
        search_space=search_space,
    )
    if optimizer is not None:
        result.optimizer = optimizer
    return result


__all__ = [
    "AdvantageObservation",
    "AggregateAdvantageModel",
    "AdvantageModel",
    "AdvantageRankWork",
    "AdvantageTerm",
    "AdvantageWorkload",
    "ActorConfig",
    "AssembledTrial",
    "ActorEstimate",
    "AllocationChange",
    "CalibrationPoint",
    "Catalog",
    "DispatchConfig",
    "Efficiency",
    "Estimate",
    "ExecutionConfig",
    "ExpertActorModel",
    "ExpertPublicationModel",
    "ExpertRolloutModel",
    "ExpertTransitionModel",
    "Hardware",
    "InferenceRates",
    "Microbatch",
    "ModelShape",
    "ObservedPacking",
    "ObservedPackingWork",
    "PackingComparison",
    "PackingError",
    "PackingSummary",
    "PackingPlan",
    "PackingMicrobatch",
    "PackingRecord",
    "ParallelLayout",
    "PhaseCalibration",
    "PublicationEstimate",
    "PublicationFlow",
    "PublicationRank",
    "PublicationRates",
    "PublicationWork",
    "RankPacking",
    "Request",
    "RolloutConfig",
    "ServiceEstimate",
    "ServiceSlice",
    "ServiceState",
    "TransferTask",
    "TransitionEstimate",
    "TransitionProfile",
    "publication_binding",
]

# SPDX-License-Identifier: Apache-2.0

"""Versioned complete reconfigurators for both native and simulated execution."""

from __future__ import annotations

from importlib import import_module

IMPLEMENTATIONS = ("basic", "reference")


def validate_reconfigurator(name: str) -> None:
    if name not in IMPLEMENTATIONS:
        raise ValueError("reconfigurator must be basic or reference")


def component_module(name: str, component: str):
    validate_reconfigurator(name)
    if component not in {
        "reconfigurator",
        "optimizer",
        "forecast",
        "history",
        "simulator",
        "cost_models",
        "evaluator",
    }:
        raise ValueError("Unknown reconfigurator component")
    module = {
        ("reference", "history"): "forecast",
        ("reference", "cost_models"): "evaluator.cost_model.loader",
        ("basic", "evaluator"): "prediction",
    }.get((name, component), component)
    return import_module(f"{__name__}.{name}.{module}")


def build_reconfigurator(
    name: str,
    *,
    candidates,
    settings: dict | None = None,
    replay: dict | None = None,
    forecast: dict | None = None,
    slo=None,
    optimizer=None,
):
    """Choose a complete algorithm version, independently of its execution backend."""
    from ..contracts import ExecutionConfig
    from ..serving import ServingSLO

    module = component_module(name, "reconfigurator")
    candidates = tuple(candidates)
    if name == "basic" and any(isinstance(c, ExecutionConfig) for c in candidates):
        raise ValueError(
            "Basic v1 retains the original DP/TP interface; full ExecutionConfig requires Reference"
        )
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
        max_evaluations=len(candidates),
    )
    options.update(settings or {})
    if slo is None:
        slo = ServingSLO(5.0, 0.05, 0.8)
    elif isinstance(slo, dict):
        slo = ServingSLO(**slo)
    result = module.OnlineReconfigurator(
        candidates, forecaster, module.OnlineReconfiguratorConfig(**options), slo
    )
    if optimizer is not None:
        result.optimizer = optimizer
    return result

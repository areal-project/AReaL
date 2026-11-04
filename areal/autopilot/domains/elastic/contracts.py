# SPDX-License-Identifier: Apache-2.0

"""Public Online revision factories and immutable JSON data contracts."""

from __future__ import annotations

import dataclasses
import json
import types
from functools import cache
from typing import Literal, Union, get_args, get_origin, get_type_hints

from areal.elastic.contracts import Forecast, OnlineLayout, OnlineObservation


@cache
def _hints(cls):
    return get_type_hints(cls)


def decode(cls, value):
    """Rebuild the declared DTO only; JSON cannot select arbitrary Python classes."""
    if value is None:
        if cls is type(None) or type(None) in get_args(cls):
            return None
        raise ValueError(f"Missing {cls}")
    origin, args = get_origin(cls), get_args(cls)
    if origin in (Union, types.UnionType):
        for option in args:
            if option is not type(None):
                try:
                    return decode(option, value)
                except (TypeError, ValueError):
                    pass
        raise ValueError(f"Value does not match {cls}")
    if origin is Literal:
        if value not in args:
            raise ValueError("Unknown literal")
        return value
    if origin is tuple or cls is tuple:
        if not isinstance(value, (tuple, list)):
            raise ValueError("A tuple must be a JSON array")
        if not args:
            return tuple(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(decode(args[0], x) for x in value)
        if len(value) != len(args):
            raise ValueError("Tuple arity changed")
        return tuple(decode(t, x) for t, x in zip(args, value, strict=True))
    if dataclasses.is_dataclass(cls):
        hints = _hints(cls)
        return cls(**{k: decode(hints[k], v) for k, v in value.items()})
    if cls is float and type(value) in (int, float):
        return float(value)
    if cls in (str, int, bool) and type(value) is cls:
        return value
    raise ValueError(f"Invalid {cls} value")


@dataclasses.dataclass(frozen=True)
class OnlineProblem:
    observation: OnlineObservation
    forecasts: tuple[Forecast, ...]
    candidates: tuple[OnlineLayout, ...]
    preferred: tuple[OnlineLayout, ...] = ()
    trigger: str = "pressure"

    def __post_init__(self):
        if (
            not self.forecasts
            or not self.candidates
            or self.observation.layout not in self.candidates
        ):
            raise ValueError(
                "An Online problem freezes forecasts and candidates including the current layout"
            )
        if len(set(self.candidates)) != len(self.candidates) or self.trigger not in {
            "pressure",
            "slack",
        }:
            raise ValueError("Invalid Online search problem")


BASELINE_SOURCE = {
    "evaluator": """from areal.autopilot.domains.elastic.evaluation.evaluator import calibrated_cost_models

def build_cost_models(calibration_cases, settings):
    return calibrated_cost_models(settings)
""",
    "optimizer": """from areal.elastic.reconfigurator.reference.optimizer import OnlineOptimizer

def build_optimizer(settings):
    return OnlineOptimizer()
""",
}


def deployment_contract(assembly):
    """Immutable execution/configuration identity, excluding a captured live boundary."""
    value = json.loads(json.dumps(assembly, allow_nan=False))
    handoff = value.get("handoff", {})
    for name in ("native_source", "native_placements", "native_checkpoint_version"):
        handoff.pop(name, None)
    value.get("deployment", {}).pop("native_execution_id", None)
    return value

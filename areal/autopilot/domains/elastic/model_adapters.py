# SPDX-License-Identifier: Apache-2.0

"""The same Online component object supplies historical probes and live predictions."""

from pathlib import Path

from areal.autopilot.domains.elastic.model_evaluator import HistoricalPrediction
from areal.elastic.reconfigurator.reference.evaluator.cost_model.loader import (
    CalibratedOnlineModels,
)


def calibrated_components(settings):
    model = settings["model"]
    return CalibratedOnlineModels(
        Path(model["bundle"]),
        Path(model["mechanism_fits"]),
        Path(model["publication_metadata"]),
        float(model["effective_bandwidth_GBps"]),
        inter_node_dp_GBps=model.get("inter_node_dp_GBps"),
    )


def validate_components(models):
    if not getattr(models, "source_ids", ()) or not all(
        callable(getattr(models, name, None))
        for name in ("rollout_config", "service", "actor", "publication")
    ):
        raise ValueError("A model factory must return complete Online components")
    return (
        models
        if isinstance(models, FrozenOnlineComponents)
        else FrozenOnlineComponents(models)
    )


class FrozenOnlineComponents:
    """Keep factory behavior when evaluating accepted models.

    A baseline bundle worker_spec reloads baseline code, not the revised factory.
    An inherited decode-affine shortcut may likewise bypass a revised service().
    Until those exports have factory-aware reconstruction/probes, use the actual
    component callbacks for both historical and candidate execution.
    """

    def __init__(self, models):
        self._models = models

    def __getattr__(self, name):
        if name in {"worker_spec", "decode_affine"}:
            raise AttributeError(name)
        return getattr(self._models, name)


class OnlineHistoryPredictor:
    def __init__(self, models):
        self.models = validate_components(models)

    def predict(self, inputs):
        try:
            return self._predict(inputs)
        except ValueError as error:
            return HistoricalPrediction(reason=str(error))

    def _predict(self, inputs):
        if inputs.kind == "actor":
            stages = tuple(self.models.actor(inputs.layout, inputs.batch))
            names = {
                "logp": "recompute_logp_seconds",
                "advantage": "compute_advantage_seconds",
                "train": "train_step_seconds",
                "recompute_logp": "recompute_logp_seconds",
                "compute_advantage": "compute_advantage_seconds",
                "train_step": "train_step_seconds",
            }
            values = tuple(
                (
                    names.get(
                        name, name if name.endswith("_seconds") else name + "_seconds"
                    ),
                    seconds,
                )
                for name, seconds in stages
            )
            return HistoricalPrediction(
                values + (("actor_total_seconds", sum(t for _, t in stages)),)
            )
        if inputs.kind == "service":
            return HistoricalPrediction(
                (
                    (
                        "service_seconds",
                        self.models.service(
                            inputs.layout, inputs.kv_tokens, inputs.slices
                        ),
                    ),
                )
            )
        return HistoricalPrediction(
            (
                (
                    "publication_seconds",
                    self.models.publication(inputs.layout, inputs.rollout_layouts),
                ),
            )
        )

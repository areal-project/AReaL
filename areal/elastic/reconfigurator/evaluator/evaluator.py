# SPDX-License-Identifier: Apache-2.0

"""Load profiles and assemble independent Actor and Rollout capacity estimates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from .actor import (
    Efficiency,
    ExpertActorModel,
    actor_config,
    estimate_actor_capacity,
    publication_seconds,
)
from .contracts import (
    ActorConfig,
    Hardware,
    InferenceRates,
    ModelShape,
    ParallelLayout,
    ParameterCalibration,
    RolloutConfig,
    positive,
)
from .packing import DispatchConfig
from .rollout import ExpertRolloutModel, SGLangScheduler, rollout_config, rollout_slo
from .workload import ExecutionConfig, ServingSLO

_KINDS = {"hardware", "topology", "models", "execution/actor", "execution/rollout"}


@dataclass(frozen=True)
class LoadedProfiles:
    actor_model: ExpertActorModel
    actor_template: ActorConfig
    rollout_model: ExpertRolloutModel
    rollout_template: RolloutConfig
    publication_GBps: float


@dataclass(frozen=True)
class CapacityEvaluator:
    models: LoadedProfiles
    template: ExecutionConfig
    slo: ServingSLO
    batch_size: int
    concurrency: int

    def publication_duration(self) -> float:
        actor = actor_config(self.models.actor_template, self.template)
        rollout = rollout_config(
            self.models.rollout_template, self.models.rollout_model, self.template
        )
        model = self.models.actor_model
        return publication_seconds(
            model.model,
            model.hardware,
            actor.parallel,
            (rollout,),
            weight_bytes=model.model.total_parameters * 2,
            hbm_efficiency=model.efficiency.hbm,
            bandwidth=self.models.publication_GBps,
        )

    def actor_capacity(self, role, pattern, publication):
        return estimate_actor_capacity(
            self.models, self.template, role, pattern, publication
        )

    def rollout_slo(self, role, pattern, rate):
        execution = replace(
            self.template, rollout_backend=role.backend, **dict(role.options)
        )
        config = rollout_config(
            self.models.rollout_template, self.models.rollout_model, execution
        )
        return rollout_slo(
            pattern, rate, config, self.models.rollout_model, self.slo, self.concurrency
        )


def _read_json(path: Path) -> dict:
    def reject_nonfinite(value: str):
        raise ValueError(f"Nonfinite archive number: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite)


def _hardware(gpu: dict, topology: dict, hardware_name: str) -> Hardware:
    if topology["hardware_profile"] != hardware_name:
        raise ValueError("Topology belongs to another GPU profile")
    if gpu.keys() & topology["hardware"].keys():
        raise ValueError("GPU and topology hardware fields must be disjoint")
    return Hardware(**gpu, **topology["hardware"])


def _model(value: dict) -> ModelShape:
    config = value["hf_config"]
    architecture = value["architecture"]
    if (
        architecture not in {"qwen3_moe", "qwen3"}
        or config["model_type"] != architecture
    ):
        raise ValueError("This model adapter covers Qwen3 dense and MoE")
    dense = architecture == "qwen3"
    if (
        config.get("decoder_sparse_step", 1) != 1
        or config.get("mlp_only_layers")
        or config.get("use_sliding_window")
        or config.get("attention_bias")
        or config.get("hidden_act", "silu") != "silu"
    ):
        raise ValueError("Unsupported mixed/alternate Qwen3 attention or MLP rules")
    shape = ModelShape(
        config["hidden_size"],
        config["intermediate_size"] if dense else config["moe_intermediate_size"],
        config["num_hidden_layers"],
        config["num_attention_heads"],
        config["num_key_value_heads"],
        config["head_dim"],
        1 if dense else config["num_experts"],
        1 if dense else config["num_experts_per_tok"],
        config["vocab_size"],
        config["tie_word_embeddings"],
        "dense" if dense else "moe",
    )
    if shape != ModelShape(**value["shape"]):
        raise ValueError("Model dimensions disagree with the archived HF configuration")
    return shape


@dataclass(frozen=True)
class Catalog:
    """External catalog root containing profiles/<kind>/<id>.json."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    def profile(self, kind: str, name: str) -> dict:
        if kind not in _KINDS:
            raise ValueError("Unknown profile kind")
        if not isinstance(name, str) or not re.fullmatch(
            "[a-zA-Z0-9][a-zA-Z0-9_.-]*", name
        ):
            raise ValueError("Catalog names must be simple identifiers")
        path = (self.root / f"profiles/{kind}/{name}.json").resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Profile path must remain inside catalog root")
        value = _read_json(path)
        if (value.get("schema_version"), value.get("kind"), value.get("id")) != (
            1,
            kind,
            name,
        ):
            raise ValueError("Profile schema, kind or identity mismatch")
        if not value.get("version") or not isinstance(value.get("values"), dict):
            raise ValueError("Profiles require versioned values")
        return value


def load_profiles(
    *,
    hardware: str,
    topology: str,
    model: str,
    actor: str,
    rollout: str,
    bandwidth_GBps: float,
    inter_node_GBps: float | None = None,
    catalog: Catalog,
    calibration: Path | None = None,
    actor_runtime: ActorConfig | None = None,
) -> LoadedProfiles:
    """Load profiles and optional fitted coefficients from JSON."""
    positive("bandwidth_GBps", bandwidth_GBps)
    profiles = {
        role: catalog.profile(kind, name)["values"]
        for role, kind, name in (
            ("hardware", "hardware", hardware),
            ("topology", "topology", topology),
            ("model", "models", model),
            ("actor", "execution/actor", actor),
            ("rollout", "execution/rollout", rollout),
        )
    }
    for role in ("actor", "rollout"):
        if profiles[role].get("model_profile", model) != model:
            raise ValueError(f"{role} profile belongs to another model")
    hw = _hardware(profiles["hardware"], profiles["topology"], hardware)
    if inter_node_GBps is not None:
        positive("inter_node_GBps", inter_node_GBps)
        hw = replace(hw, inter_node_gbps=inter_node_GBps)
    shape = _model(profiles["model"])
    actor_prof, rollout_prof = profiles["actor"], profiles["rollout"]
    if actor_prof.get("advantage"):
        raise ValueError("Legacy aggregate advantage fits require the archived model")
    if actor_prof["dispatch"]["mode"] != "controller":
        raise ValueError("Reference requires a single-controller Actor profile")

    actor_model = ExpertActorModel(
        shape,
        hw,
        actor_prof["runtime_revision"],
        efficiency=Efficiency(
            **{k: v for k, v in actor_prof["efficiency"].items() if k != "fsdp_overlap"}
        ),
        group_gbps=tuple(profiles["topology"]["group_gbps"]),
        dispatch=DispatchConfig(group_size=actor_prof["dispatch"]["group_size"]),
    )
    rollout_model = ExpertRolloutModel(
        shape,
        hw,
        InferenceRates(**rollout_prof["rates"]),
        rollout_prof["runtime_revision"],
    )
    if calibration is not None:
        data = _read_json(Path(calibration))
        if data.pop("hardware", None) != hardware or data.pop("model", None) != model:
            raise ValueError("Calibration hardware or model differs from profiles")
        fit_topology = data.pop("topology", None)
        if fit_topology is not None and fit_topology != topology:
            raise ValueError("Calibration belongs to another topology")
        if not data or not data.keys() <= {"actor", "rollout"}:
            raise ValueError("Calibration requires actor or rollout parameters")
        if fit_topology is None and any(
            "network" in fit["parameters"] for fit in data.values()
        ):
            raise ValueError("Calibrated communication efficiency requires a topology")
        if "actor" in data:
            fit = ParameterCalibration(**data["actor"])
            actor_model = replace(
                actor_model,
                efficiency=fit.apply(
                    actor_model.efficiency, actor_model.runtime_revision
                ),
            )
        if "rollout" in data:
            fit = ParameterCalibration(**data["rollout"])
            rollout_model = replace(
                rollout_model,
                rates=fit.apply(rollout_model.rates, rollout_model.runtime_revision),
            )

    actor_template = (
        actor_runtime
        if actor_runtime is not None
        else ActorConfig(**{**actor_prof["config"], "parallel": ParallelLayout()})
    )
    if rollout_prof["policy"] != SGLangScheduler.version:
        raise ValueError("Unknown rollout scheduler")
    rollout_template = RolloutConfig(
        **{
            **rollout_prof["config"],
            "mem_fraction_static": rollout_prof["config"].get(
                "mem_fraction_static", 0.9
            ),
        },
        tp=1,
        replicas=1,
        kv_capacity_tokens=1,
    )
    return LoadedProfiles(
        actor_model,
        actor_template,
        rollout_model,
        rollout_template,
        bandwidth_GBps,
    )

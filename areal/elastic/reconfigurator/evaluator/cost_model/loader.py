# SPDX-License-Identifier: Apache-2.0

"""Assemble analytical models from profiles and optional fitted coefficients."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from areal.elastic.reconfigurator.evaluator.cost_model import fingerprint
from areal.elastic.reconfigurator.evaluator.cost_model.actor.megatron import (
    Efficiency,
)
from areal.elastic.reconfigurator.evaluator.cost_model.actor.model import (
    ExpertActorModel,
)
from areal.elastic.reconfigurator.evaluator.cost_model.actor.packing import (
    DispatchConfig,
)
from areal.elastic.reconfigurator.evaluator.cost_model.contracts import (
    ActorConfig,
    Hardware,
    InferenceRates,
    ModelShape,
    ParallelLayout,
    ParameterCalibration,
    RolloutConfig,
    nonnegative,
)

from .rollout.model import ExpertRolloutModel
from .rollout.sglang import scheduler_for

_KINDS = {"hardware", "topology", "models", "execution/actor", "execution/rollout"}


def numerical_environment() -> dict[str, str]:
    """FFD ties depend on NumPy argsort; migration must preserve this binding."""
    return {"numpy": np.__version__}


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
        or (config.get("hidden_act", "silu") != "silu")
    ):
        raise ValueError(
            "Mixed dense/MoE layers or alternate attention/MLP rules require another model adapter"
        )
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


def _actor(value: dict, parallel: ParallelLayout) -> ActorConfig:
    config = value["config"].copy()
    config["parallel"] = parallel
    return ActorConfig(**config)


def _rollout(
    value: dict, *, tp: int, replicas: int, kv_capacity_tokens: int
) -> RolloutConfig:
    config = RolloutConfig(
        **value["config"],
        tp=tp,
        replicas=replicas,
        kv_capacity_tokens=kv_capacity_tokens,
    )
    if scheduler_for(config.backend).version != value["policy"]:
        raise ValueError("Unknown rollout scheduler")
    return config


def _name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        "[a-zA-Z0-9][a-zA-Z0-9_.-]*", value
    ):
        raise ValueError("Catalog names must be simple identifiers")
    return value


def _finite(value: str):
    raise ValueError(f"Nonfinite archive number: {value}")


def _read(path: Path, *, expected_sha256: str | None = None):
    payload = path.read_bytes()
    if (
        expected_sha256 is not None
        and hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise ValueError(f"Archive content hash mismatch: {path.name}")
    if path.suffix == ".gz":
        payload = gzip.decompress(payload)
    text = payload.decode("utf-8")
    if ".jsonl" in path.suffixes:
        return [
            json.loads(line, parse_constant=_finite)
            for line in text.splitlines()
            if line.strip()
        ]
    return json.loads(text, parse_constant=_finite)


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("Archive reference must remain inside its catalog/trial")
    return path


def _components(
    profiles: dict, hardware_name: str
) -> tuple[ExpertActorModel, ExpertRolloutModel]:
    hardware = _hardware(profiles["hardware"], profiles["topology"], hardware_name)
    model = _model(profiles["model"])
    actor, rollout = (profiles["actor"], profiles["rollout"])
    if actor.get("advantage"):
        raise ValueError("Legacy aggregate advantage fits require the archived model")
    if actor["dispatch"]["mode"] != "controller":
        raise ValueError("Reference requires a single-controller Actor profile")
    raw = ExpertActorModel(
        model,
        hardware,
        actor["runtime_revision"],
        efficiency=Efficiency(
            **{k: v for k, v in actor["efficiency"].items() if k != "fsdp_overlap"}
        ),
        group_gbps=tuple(profiles["topology"]["group_gbps"]),
        dispatch=DispatchConfig(group_size=actor["dispatch"]["group_size"]),
        execution_identity=fingerprint(profiles["model"], actor["execution"])
        if actor.get("execution")
        else "",
    )
    service = ExpertRolloutModel(
        model,
        hardware,
        InferenceRates(**rollout["rates"]),
        rollout["runtime_revision"],
        execution_identity=fingerprint(profiles["model"], rollout["execution"])
        if rollout.get("execution")
        else "",
        **rollout.get("model_options", {}),
    )
    return (raw, service)


@dataclass(frozen=True)
class Catalog:
    """Explicit external catalog root containing profiles/<kind>/<id>.json."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    def names(self, kind: str) -> tuple[str, ...]:
        if kind not in _KINDS:
            raise ValueError("Unknown catalog kind")
        directory = self.root / f"profiles/{kind}"
        return tuple(sorted(p.stem for p in directory.glob("*.json")))

    def profile(self, kind: str, name: str, *, sha256: str | None = None) -> dict:
        if kind not in _KINDS:
            raise ValueError("Unknown profile kind")
        path = _inside(self.root, f"profiles/{kind}/{_name(name)}.json")
        value = _read(path, expected_sha256=sha256)
        if (value.get("schema_version"), value.get("kind"), value.get("id")) != (
            1,
            kind,
            name,
        ):
            raise ValueError("Profile schema, kind or identity mismatch")
        if (
            not value.get("version")
            or not isinstance(value.get("values"), dict)
            or (not value.get("provenance"))
        ):
            raise ValueError("Profiles require versioned values and provenance")
        return value

    def hardware(self, hardware: str, topology: str) -> Hardware:
        gpu = self.profile("hardware", hardware)["values"]
        host = self.profile("topology", topology)["values"]
        return _hardware(gpu, host, hardware)

    def model(self, name: str) -> ModelShape:
        return _model(self.profile("models", name)["values"])

    def actor(self, name: str, *, parallel: ParallelLayout) -> ActorConfig:
        return _actor(self.profile("execution/actor", name)["values"], parallel)

    def rollout(
        self, name: str, *, tp: int, replicas: int, kv_capacity_tokens: int
    ) -> RolloutConfig:
        return _rollout(
            self.profile("execution/rollout", name)["values"],
            tp=tp,
            replicas=replicas,
            kv_capacity_tokens=kv_capacity_tokens,
        )


def _load_profile_components(
    catalog,
    hardware,
    topology,
    model,
    actor,
    rollout,
    inter_node_GBps,
    calibration=None,
):
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
    a, r = _components(profiles, hardware)
    if calibration is not None:
        data = _read(Path(calibration))
        fit_id = fingerprint(data)
        if data.pop("hardware", None) != hardware or data.pop("model", None) != model:
            raise ValueError(
                "Calibration hardware or model differs from the selected profiles"
            )
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
            a = replace(
                a,
                efficiency=fit.apply(a.efficiency, a.runtime_revision),
                parameter_sources=(fit_id, *fit.source_ids),
            )
        if "rollout" in data:
            fit = ParameterCalibration(**data["rollout"])
            r = replace(
                r,
                rates=fit.apply(r.rates, r.runtime_revision),
                parameter_sources=(fit_id, *fit.source_ids),
            )
    if inter_node_GBps is not None:
        nonnegative("inter_node_GBps", inter_node_GBps, positive=True)
        a = replace(a, hardware=replace(a.hardware, inter_node_gbps=inter_node_GBps))
        r = replace(r, hardware=replace(r.hardware, inter_node_gbps=inter_node_GBps))
    actor_template = _actor(profiles["actor"], ParallelLayout())
    rollout_template = _rollout(
        profiles["rollout"], tp=1, replicas=1, kv_capacity_tokens=1
    )
    rollout_template = replace(
        rollout_template,
        mem_fraction_static=profiles["rollout"]["config"].get(
            "mem_fraction_static", 0.9
        ),
    )
    return a, actor_template, r, rollout_template

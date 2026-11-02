# SPDX-License-Identifier: Apache-2.0

"""Read profiles and frozen calibration parameters; assemble component models."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from areal.elastic.reconfigurator.reference.evaluator.contracts import ExecutionConfig
from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.fsdp import (
    FSDPHostRates,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.model import (
    AggregateAdvantageModel,
    ExpertActorModel,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.operators import (
    Efficiency,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
    DispatchConfig,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
    ActorConfig,
    Hardware,
    InferenceRates,
    ModelShape,
    ParallelLayout,
    RolloutConfig,
    nonnegative,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.rollout.model import (
    ExpertRolloutModel,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.rollout.scheduling import (
    scheduler_for,
)
from areal.elastic.reconfigurator.reference.identity import fingerprint

from .actor.model import CalibratedActorModel
from .publication import CalibratedPublicationModel
from .rollout.model import CalibratedRolloutModel

_KINDS = {"hardware", "topology", "models", "execution/actor", "execution/rollout"}
_ROLES = {
    "hardware": "hardware",
    "topology": "topology",
    "model": "models",
    "actor": "execution/actor",
    "rollout": "execution/rollout",
}


def numerical_environment() -> dict[str, str]:
    """FFD ties depend on NumPy argsort; migration must preserve this binding."""
    return {"numpy": np.__version__}


def _hardware(gpu: dict, topology: dict, hardware_name: str) -> Hardware:
    if topology["hardware_profile"] != hardware_name:
        raise ValueError("Topology belongs to another GPU profile")
    if gpu.keys() & topology["hardware"].keys():
        raise ValueError("GPU and topology hardware fields must be disjoint")
    return Hardware(**gpu, **topology["hardware"])


def _validate_placement(
    config: ExecutionConfig, topology: dict, hardware_name: str, scopes: dict
) -> None:
    if topology["hardware_profile"] != hardware_name:
        raise ValueError("Topology belongs to another GPU profile")
    locations = {
        device: node
        for node, devices in topology["nodes"].items()
        for device in devices
    }
    for role in ("actor", "rollout"):
        devices = getattr(config, f"{role}_devices")
        if not set(devices) <= locations.keys():
            raise ValueError("Execution devices are outside the declared topology")
        if scopes[role].get("node_local"):
            if len({locations[device] for device in devices}) != 1:
                raise ValueError(
                    "Component calibration assembly requires node-local placement"
                )


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


@dataclass(frozen=True)
class AssembledTrial:
    """Models retain their strict coverage; diagnostic fallback is an explicit opt-in."""

    config: ExecutionConfig
    raw_actor: ExpertActorModel
    actor: ExpertActorModel
    rollout: ExpertRolloutModel
    binding: str


def _components(
    profiles: dict, hardware_name: str
) -> tuple[ExpertActorModel, ExpertRolloutModel]:
    hardware = _hardware(profiles["hardware"], profiles["topology"], hardware_name)
    model = _model(profiles["model"])
    actor, rollout = (profiles["actor"], profiles["rollout"])
    raw = ExpertActorModel(
        model,
        hardware,
        actor["runtime_revision"],
        efficiency=Efficiency(**actor["efficiency"]),
        group_gbps=tuple(profiles["topology"]["group_gbps"]),
        advantage_model=AggregateAdvantageModel(**actor["advantage"])
        if actor.get("advantage")
        else None,
        dispatch=DispatchConfig(**actor["dispatch"]),
        fsdp_host_rates=FSDPHostRates(**profiles["topology"]["host_rates"])
        if profiles["topology"].get("host_rates")
        else None,
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
    root: Path = Path(__file__).resolve().parent

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

    def assemble(
        self,
        *,
        hardware: str,
        topology: str,
        model: str,
        actor: str,
        rollout: str,
        actor_parallel: ParallelLayout,
        rollout_tp: int,
        rollout_replicas: int,
        kv_capacity_tokens: int,
        resource_pools: Mapping[str, Sequence[str]],
    ) -> AssembledTrial:
        """Assemble uncalibrated components; require explicit target KV capacity.

        Resource pools and parallel layouts are explicit assembly inputs.
        Execution profiles describe calibration scope, not live-run quotas.
        """
        names = dict(
            hardware=hardware,
            topology=topology,
            model=model,
            actor=actor,
            rollout=rollout,
        )
        profiles = {
            role: self.profile(_ROLES[role], name) for role, name in names.items()
        }
        values = {role: profile["values"] for role, profile in profiles.items()}
        if any(
            values[role].get("model_profile", model) != model
            for role in ("actor", "rollout")
        ):
            raise ValueError("Execution recipe belongs to another model profile")
        raw, service = _components(values, hardware)
        actor_config = _actor(values["actor"], actor_parallel)
        allowed = values["actor"].get("parallel_dimensions")
        if allowed and any(
            getattr(actor_parallel, name) != 1
            for name in ("dp", "tp", "pp", "cp", "ep", "etp")
            if name not in allowed
        ):
            raise ValueError("Parallel dimensions exceed this Actor profile scope")
        rollout_config = _rollout(
            values["rollout"],
            tp=rollout_tp,
            replicas=rollout_replicas,
            kv_capacity_tokens=kv_capacity_tokens,
        )
        if set(resource_pools) != {"actor", "rollout"}:
            raise ValueError("Supply explicit Actor and Rollout resource pools")
        devices = []
        for role, size in (
            ("actor", actor_parallel.world_size),
            ("rollout", rollout_config.world_size),
        ):
            pool = tuple(resource_pools[role])
            if size > len(pool):
                raise ValueError("Execution exceeds the supplied role device pool")
            chosen = pool[:size]
            devices.append(chosen)
        config = ExecutionConfig(actor_config, rollout_config, *devices)
        _validate_placement(
            config,
            values["topology"],
            hardware,
            {
                role: values[role].get("calibration_scope", {})
                for role in ("actor", "rollout")
            },
        )
        binding = fingerprint(profiles, config, dict(resource_pools))
        return AssembledTrial(config, raw, raw, service, binding)


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CalibratedOnlineModels:
    def __init__(
        self,
        bundle: Path,
        mechanism_fits: Path,
        publication_metadata: Path,
        effective_bandwidth_GBps: float,
        *,
        inter_node_dp_GBps: float | None = None,
    ):
        self.worker_spec = (
            str(bundle.resolve()),
            str(mechanism_fits.resolve()),
            str(publication_metadata.resolve()),
            effective_bandwidth_GBps,
        )
        self.inter_node_dp_GBps = inter_node_dp_GBps
        if inter_node_dp_GBps is not None:
            nonnegative("inter_node_dp_GBps", inter_node_dp_GBps, positive=True)
            self.worker_spec += (inter_node_dp_GBps,)
        self.scenario_notes = (
            (
                f"Actor DP beyond one8GPU node extrapolates frozen fits with assumed effective{inter_node_dp_GBps}GB/s collective bandwidth; TP groups stay node-local",
                "Cross-node Rollout replicas reuse single-replica fits; placement interference and extra-node startup/publication/state paths are assumptions",
            )
            if inter_node_dp_GBps is not None
            else ()
        )
        name = "_areal_online_frozen_" + digest(bundle / "models.py")[:16]
        spec = importlib.util.spec_from_file_location(name, bundle / "models.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.fits = module
        _, parts, self.cal_actor, self.cal_rollout = module.load_bundle(
            bundle, extrapolate=True
        )
        self.manifest = json.loads((bundle / "manifest.json").read_text())
        self.actor_template, self.rollout_template = (
            parts.config.actor,
            parts.config.rollout,
        )
        self.mechanism = json.loads(mechanism_fits.read_text())
        if self.mechanism["frozen_bundle_id"] != self.manifest["bundle_id"]:
            raise ValueError(
                "Mechanism fits belong to a different frozen component bundle"
            )
        self.capacities = dict(self.cal_rollout.capacities)
        self.bandwidth = effective_bandwidth_GBps
        raw = json.loads(publication_metadata.read_text())
        self.parameters = tuple(
            (
                p["data"]["name"],
                tuple(p["data"]["global_shape"]),
                p["data"]["shards"][0]["data"]["sharding_dim"],
                p["data"]["shards"][0]["data"]["num_shards"] > 1,
            )
            for p in raw["parameters"]
        )
        self.weight_bytes = sum(
            (math.prod(shape) * 2 for _, shape, _, _ in self.parameters)
        )
        self.source_ids = (
            self.manifest["bundle_id"],
            digest(mechanism_fits),
            digest(publication_metadata),
        )
        if inter_node_dp_GBps is not None:
            self.source_ids += (
                fingerprint("assumed-cross-node-DP-v1", inter_node_dp_GBps),
            )
        self.actor_model = CalibratedActorModel(
            self.fits,
            self.cal_actor,
            self.actor_template,
            self.mechanism,
            self.inter_node_dp_GBps,
        )
        self.rollout_model = CalibratedRolloutModel(
            self.fits, self.cal_rollout, self.rollout_template
        )
        self.publication_model = CalibratedPublicationModel(
            self.parameters, self.weight_bytes, self.bandwidth, self.mechanism
        )

    def rollout_config(self, layout):
        return self.rollout_model.rollout_config(layout)

    def service(self, layout, kv, slices):
        return self.rollout_model.service(layout, kv, slices)

    def decode_affine(self, layout, kv, slices):
        return self.rollout_model.decode_affine(layout, kv, slices)

    def actor(self, layout, batch):
        return self.actor_model.actor(layout, batch)

    def actor_memory(self, layout, batch):
        return self.actor_model.actor_memory(layout, batch)

    def publication(self, layout, recipients):
        return self.publication_model.publication(layout, recipients)

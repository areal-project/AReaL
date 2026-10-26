# SPDX-License-Identifier: Apache-2.0

"""Versioned local profiles and content-bound trial archives; no fitting or I/O jobs."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .core import PhaseCalibration, code_revision, fingerprint
from .cost_model._actor.components import Efficiency
from .cost_model._actor.config import ActorConfig, Hardware, ModelShape, ParallelLayout
from .cost_model._actor.packing import DispatchConfig
from .cost_model._rollout.scheduling import policy_for
from .cost_model.actor import ExpertActorModel
from .cost_model.aggregate_advantage import AggregateAdvantageModel
from .cost_model.fsdp import FSDPHostRates
from .cost_model.rollout import ExpertRolloutModel, InferenceRates, RolloutConfig
from .cost_model.rpc import RPCTimingModel
from .scenario import ExecutionConfig

_KINDS = {"hardware", "systems", "models", "actor", "rollout"}
_ROLES = {
    "hardware": "hardware",
    "system": "systems",
    "model": "models",
    "actor": "actor",
    "rollout": "rollout",
}


def numerical_environment() -> dict[str, str]:
    """FFD ties depend on NumPy argsort; migration must preserve this binding."""
    return {"numpy": np.__version__}


def _hardware(gpu: dict, system: dict, hardware_name: str) -> Hardware:
    if system["hardware_profile"] != hardware_name:
        raise ValueError("System belongs to another GPU profile")
    if gpu.keys() & system["hardware"].keys():
        raise ValueError("GPU and system hardware fields must be disjoint")
    return Hardware(**gpu, **system["hardware"])


def _validate_placement(
    config: ExecutionConfig, host: dict, hardware_name: str
) -> None:
    if host["hardware_profile"] != hardware_name:
        raise ValueError("System belongs to another GPU profile")
    maximum = host["hardware"]["gpus_per_node"] * host["max_nodes"]
    if len(config.actor_devices + config.rollout_devices) > maximum:
        raise ValueError("Execution exceeds the system topology/measurement scope")
    topology = host.get("topology", {})
    if topology.get("role_node_local"):
        for role in ("actor", "rollout"):
            devices = getattr(config, f"{role}_devices")
            if not set(devices) <= set(topology[f"{role}_devices"]):
                raise ValueError("Execution devices exceed the system role pool")
            nodes = {topology["device_nodes"][device] for device in devices}
            if len(nodes) != 1 or len(devices) > host["hardware"]["gpus_per_node"]:
                raise ValueError("Each role must remain within one node")


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


def _actor(value: dict) -> ActorConfig:
    config = value["config"].copy()
    config["parallel"] = ParallelLayout(**config["parallel"])
    return ActorConfig(**config)


def _rollout(value: dict) -> RolloutConfig:
    config = RolloutConfig(**value["config"])
    if policy_for(config.backend).version != value["policy"]:
        raise ValueError("Unknown rollout scheduling policy")
    return config


def _name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", value
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
    rpc_timing: RPCTimingModel | None
    binding: str


def _components(
    profiles: dict, hardware_name: str
) -> tuple[ExpertActorModel, ExpertRolloutModel]:
    hardware = _hardware(profiles["hardware"], profiles["system"], hardware_name)
    model = _model(profiles["model"])
    actor, rollout = profiles["actor"], profiles["rollout"]
    raw = ExpertActorModel(
        model,
        hardware,
        actor["runtime_revision"],
        efficiency=Efficiency(**actor["efficiency"]),
        group_gbps=tuple(profiles["system"]["group_gbps"]),
        advantage_model=AggregateAdvantageModel(**actor["advantage"])
        if actor.get("advantage")
        else None,
        dispatch=DispatchConfig(**actor["dispatch"]),
        fsdp_host_rates=FSDPHostRates(**profiles["system"]["host_rates"])
        if profiles["system"].get("host_rates")
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
    return raw, service


@dataclass(frozen=True)
class Catalog:
    root: Path = Path(__file__).resolve().parent

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())

    def names(self, kind: str) -> tuple[str, ...]:
        if kind not in _KINDS | {"trials"}:
            raise ValueError("Unknown catalog kind")
        directory = self.root / ("trials" if kind == "trials" else f"profiles/{kind}")
        if kind == "trials":
            return tuple(
                sorted(p.parent.name for p in directory.glob("*/manifest.json"))
            )
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
            or not value.get("provenance")
        ):
            raise ValueError("Profiles require versioned values and provenance")
        return value

    def hardware(self, hardware: str, system: str) -> Hardware:
        gpu = self.profile("hardware", hardware)["values"]
        host = self.profile("systems", system)["values"]
        return _hardware(gpu, host, hardware)

    def model(self, name: str) -> ModelShape:
        return _model(self.profile("models", name)["values"])

    def actor(self, name: str) -> ActorConfig:
        return _actor(self.profile("actor", name)["values"])

    def rollout(self, name: str) -> RolloutConfig:
        return _rollout(self.profile("rollout", name)["values"])

    def assemble(
        self,
        *,
        hardware: str,
        system: str,
        model: str,
        actor: str,
        rollout: str,
        actor_parallel: ParallelLayout,
        rollout_tp: int,
        rollout_replicas: int,
        kv_capacity_tokens: int,
    ) -> AssembledTrial:
        """Assemble uncalibrated components; require explicit target KV capacity.

        Device pools and node-local limits come from the system profile. This
        does not fit, inherit trial coefficients, or invent transition costs.
        """
        names = dict(
            hardware=hardware, system=system, model=model, actor=actor, rollout=rollout
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
        actor_config = replace(_actor(values["actor"]), parallel=actor_parallel)
        allowed = values["actor"].get("parallel_dimensions")
        if allowed and any(
            getattr(actor_parallel, name) != 1
            for name in ("dp", "tp", "pp", "cp", "ep", "etp")
            if name not in allowed
        ):
            raise ValueError("Parallel dimensions exceed this Actor profile scope")
        rollout_config = replace(
            _rollout(values["rollout"]),
            tp=rollout_tp,
            replicas=rollout_replicas,
            kv_capacity_tokens=kv_capacity_tokens,
        )
        host = values["system"]
        topology = host["topology"]
        devices = []
        for role, size in (
            ("actor", actor_parallel.world_size),
            ("rollout", rollout_config.world_size),
        ):
            pool = tuple(topology[f"{role}_devices"])
            if size > len(pool):
                raise ValueError("Execution exceeds the system role device pool")
            chosen = pool[:size]
            devices.append(chosen)
        config = ExecutionConfig(actor_config, rollout_config, *devices)
        _validate_placement(config, host, hardware)
        binding = fingerprint(profiles, config)
        return AssembledTrial(config, raw, raw, service, None, binding)

    def trial(self, name: str) -> TrialArchive:
        trial = TrialArchive(self, _name(name))
        trial.manifest()
        return trial


@dataclass(frozen=True)
class TrialArchive:
    catalog: Catalog
    name: str

    @property
    def root(self) -> Path:
        return _inside(self.catalog.root, f"trials/{_name(self.name)}")

    def manifest(self) -> dict:
        value = _read(self.root / "manifest.json")
        if (value.get("schema_version"), value.get("id")) != (1, self.name):
            raise ValueError("Trial schema or identity mismatch")
        if set(value.get("profiles", {})) != set(_ROLES):
            raise ValueError(
                "Trial requires hardware, system, model, actor and rollout references"
            )
        return value

    def read(self, key: str):
        return self._read_data(self.manifest(), key)

    def _read_data(self, manifest: dict, key: str):
        reference = manifest["data"][key]
        if not re.fullmatch(r"[0-9a-f]{64}", reference.get("sha256", "")):
            raise ValueError("Archive data requires a SHA256 digest")
        return _read(
            _inside(self.root, reference["path"]), expected_sha256=reference["sha256"]
        )

    def verify(self) -> dict:
        manifest = self.manifest()
        profiles = self._profiles(manifest)
        for key in manifest["data"]:
            self._read_data(manifest, key)
        config = self._configuration(manifest, profiles)
        return {
            "trial": self.name,
            "profiles": len(profiles),
            "data_files": len(manifest["data"]),
            "configuration": config.key,
        }

    def _profiles(self, manifest: dict) -> dict:
        result = {}
        for role, kind in _ROLES.items():
            reference = manifest["profiles"][role]
            if not re.fullmatch(r"[0-9a-f]{64}", reference.get("sha256", "")):
                raise ValueError("Trial profile references require SHA256 digests")
            result[role] = self.catalog.profile(
                kind, reference["name"], sha256=reference["sha256"]
            )["values"]
        return result

    def configuration(self) -> ExecutionConfig:
        manifest = self.manifest()
        profiles = self._profiles(manifest)
        return self._configuration(manifest, profiles)

    def _configuration(self, manifest: dict, profiles: dict) -> ExecutionConfig:
        execution = ExecutionConfig(
            _actor(profiles["actor"]),
            _rollout(profiles["rollout"]),
            **manifest["execution"],
        )
        host = profiles["system"]
        _validate_placement(execution, host, manifest["profiles"]["hardware"]["name"])
        return execution

    def models(self, *, calibrated: bool = True) -> AssembledTrial:
        """Rebind frozen coefficients only for explicitly reviewed implementations.

        The complete profile contents, execution and formula version are checked.
        No coefficients, feature bounds or sample IDs are fitted while loading.
        """
        return self._models(self.manifest(), calibrated=calibrated)

    def _models(self, manifest: dict, *, calibrated: bool = True) -> AssembledTrial:
        p = self._profiles(manifest)
        config = self._configuration(manifest, p)
        binding = fingerprint(manifest["profiles"], manifest["execution"])
        raw, rollout = _components(p, manifest["profiles"]["hardware"]["name"])
        if not calibrated:
            return AssembledTrial(config, raw, raw, rollout, None, binding)
        calibration = self._read_data(manifest, "calibration")
        if calibration["binding"] != binding:
            raise ValueError("Calibration belongs to another resolved trial profile")
        implementation = {
            "code_revision": code_revision(),
            "numerical_environment": numerical_environment(),
        }
        if implementation not in manifest["compatible_implementations"]:
            raise ValueError(
                "Formula implementation requires an explicit calibration compatibility audit"
            )
        profile = raw.profile_key(config.actor)

        def rebind(value: dict) -> PhaseCalibration:
            if value["profile"] != calibration["archived_actor_profile"]:
                raise ValueError("Archived phase calibration profile mismatch")
            return replace(PhaseCalibration(**value), profile=profile)

        advantage_data = calibration["actor"]["advantage"].copy()
        advantage_data["calibration"] = rebind(advantage_data["calibration"])
        actor = replace(
            raw,
            advantage_model=AggregateAdvantageModel(**advantage_data),
            calibration=tuple(rebind(v) for v in calibration["actor"]["phase_scales"]),
        )
        rpc = {
            k: v
            for k, v in calibration["rpc"].items()
            if k not in {"evidence", "split"}
        }
        timing = RPCTimingModel(
            profile=RPCTimingModel.profile_for(rollout, config.rollout), **rpc
        )
        return AssembledTrial(config, raw, actor, rollout, timing, binding)

    def simulation(
        self, *, cleanup_seconds: float | None = None, diagnostic_fallback: bool = False
    ):
        from .trial import simulation

        return simulation(
            self,
            cleanup_seconds=cleanup_seconds,
            diagnostic_fallback=diagnostic_fallback,
        )

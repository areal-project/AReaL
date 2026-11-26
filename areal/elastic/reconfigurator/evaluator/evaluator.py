# SPDX-License-Identifier: Apache-2.0

"""Load evaluation profiles, prepare common boundaries and evaluate candidate layouts."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from functools import lru_cache
from math import isfinite
from pathlib import Path

import numpy as np

from areal.elastic.reconfigurator.evaluator import fingerprint
from areal.elastic.reconfigurator.evaluator.actor.cost_model import (
    Advantages,
    Efficiency,
    ExpertActorModel,
)
from areal.elastic.reconfigurator.evaluator.areal.controller import (
    DispatchConfig,
    Observation,
    simulate,
)
from areal.elastic.reconfigurator.evaluator.areal.cost_model import (
    Cutover,
    ReconfigurationModel,
)
from areal.elastic.reconfigurator.evaluator.contracts import (
    ActorConfig,
    Hardware,
    InferenceRates,
    ModelShape,
    ParallelLayout,
    ParameterCalibration,
    RolloutConfig,
    nonnegative,
)
from areal.elastic.reconfigurator.evaluator.rollout.cost_model import (
    ExpertRolloutModel,
)
from areal.elastic.reconfigurator.evaluator.workload import (
    Forecast,
    ServingSLO,
    WorkloadPattern,
)
from areal.elastic.reconfigurator.optimizer import Layout, Prediction

from .actor.capacity import actor_capacity
from .actor.cost_model import actor_config
from .areal.cost_model import publication_seconds
from .rollout.capacity import probe_capacity
from .rollout.cost_model import rollout_config
from .rollout.sglang import scheduler_for


@dataclass(frozen=True)
class CandidateExecution:
    actor: ActorConfig
    rollout: RolloutConfig
    rollout_concurrency: int | None


@dataclass(frozen=True)
class _CandidateJob:
    index: int
    model_spec: tuple
    source_ids: tuple[str, ...]
    context: tuple[Observation, Forecast]
    config: CandidateExecution
    layout: Layout
    cut: Cutover
    publication: float
    slo: ServingSLO
    settings: dict


@lru_cache(maxsize=1)
def _worker_models(spec, expected_ids):
    (
        kind,
        hardware,
        topology,
        model,
        actor,
        rollout,
        bandwidth,
        link,
        root,
        calibration,
        actor_runtime,
    ) = spec
    if kind != "analytical":
        raise ValueError(
            "Parallel evaluation requires an analytical model specification"
        )
    models = load_profiles(
        hardware=hardware,
        topology=topology,
        model=model,
        actor=actor,
        rollout=rollout,
        bandwidth_GBps=bandwidth,
        inter_node_GBps=link,
        catalog=Catalog(Path(root)),
        calibration=Path(calibration) if calibration else None,
        actor_runtime=actor_runtime,
    )
    if models.source_ids != expected_ids:
        raise ValueError("CPU worker loaded different cost models or source revisions")
    return models


def _evaluate_candidate(job: _CandidateJob) -> tuple[int, Prediction]:
    models = _worker_models(job.model_spec, job.source_ids)
    evaluator = ReplayEvaluator(models, None, None, job.slo, **job.settings)
    try:
        prediction = evaluator._predict(
            *job.context, job.config, job.layout, job.cut, job.publication, None
        )
    except ValueError as exc:
        prediction = Prediction(
            job.layout, None, None, None, supported=False, reason=str(exc)
        )
    return job.index, prediction


class ReplayEvaluator:
    """Predict candidates, optionally using a caller-owned CPU process pool."""

    def __init__(
        self,
        models,
        forecaster,
        template,
        slo,
        *,
        batch_size,
        concurrency,
        staleness,
        cutovers=None,
        reconfiguration: ReconfigurationModel | None = None,
        memory_reserve_fraction=0.0,
        executor: Executor | None = None,
    ):
        from areal.elastic.reconfigurator.evaluator.workload import Request

        self.models, self.forecaster, self.template, self.slo = (
            models,
            forecaster,
            template,
            slo,
        )
        self.batch_size, self.concurrency, self.staleness = (
            batch_size,
            concurrency,
            staleness,
        )
        self.cutovers = {} if cutovers is None else cutovers
        self.reconfiguration = reconfiguration
        self.reserve = memory_reserve_fraction
        self.executor = executor
        self.request_type, self.scheduler = Request, scheduler_for("sglang")

    def prepare(self, observation, *, pattern: WorkloadPattern | None = None):
        if pattern is not None:
            forecast = pattern.forecast(
                observation.view.now, self.forecaster.horizon_seconds
            )
        else:
            forecasts = self.forecaster.predict(observation.view, 0)
            if len(forecasts) > 1:
                raise ValueError("Expected deterministic HistoricalReplay")
            forecast = next(iter(forecasts), None)
        context = observation, forecast
        if observation.search_ready_at is None or not forecast or not forecast.episodes:
            return context
        cfg, cut, pub = self._prepare_candidate(context, observation.current)
        # Run the common prefix once. Only time and work after publication
        # can differ in cost across candidates.
        boundary = self._predict(
            *context,
            cfg,
            observation.current,
            cut,
            pub,
            None,
            stop_at_publication=observation.search_ready_at,
        )
        if isinstance(boundary, Prediction):
            return observation, None  # no usable publication in the forecast horizon
        return boundary

    def evaluate(self, context, layout, *, on_batch=None):
        try:
            cfg, cut, pub = self._prepare_candidate(context, layout)
            return self._predict(*context, cfg, layout, cut, pub, on_batch)
        except ValueError as exc:
            return Prediction(
                layout, None, None, None, supported=False, reason=str(exc)
            )

    def estimate_rollout_capacity(self, context, layout, pattern, rate, target):
        obs, forecast = context
        execution = replace(
            self.template,
            rollout_backend=layout.rollout.backend,
            **dict(layout.rollout.options),
        )
        rollout = rollout_config(
            self.models.rollout_template, self.models.rollout_model, execution
        )
        cfg = CandidateExecution(
            self.models.actor_template,
            rollout,
            execution.rollout_max_concurrent_rollouts,
        )
        start = obs.view.now
        sample_seconds = min(60.0, forecast.input_closes_at - start)
        clean = replace(obs, backlog=None)

        def check(probe_rate):
            period = len(pattern.arrivals) / probe_rate
            scaled = replace(
                pattern,
                period_seconds=period,
                arrivals=tuple(
                    (at * period / pattern.period_seconds, family)
                    for at, family in pattern.arrivals
                ),
            )
            sample = scaled.forecast(start, sample_seconds)
            if sample is None or not sample.episodes:
                return None
            # Drain the finite offer window so the last calls can report TPOT.
            probe = replace(sample, input_closes_at=sample.input_closes_at + 75.0)
            row = simulate(
                clean,
                probe,
                cfg,
                layout,
                Cutover(0, 0),
                0.0,
                None,
                batch_size=self.batch_size,
                concurrency=self.concurrency,
                actor_model=self.models.actor_model,
                rollout_model=self.models.rollout_model,
                request_type=self.request_type,
                reserve=self.reserve,
                scheduler=self.scheduler,
                slo=self.slo,
                staleness=self.staleness,
                rollout_only=True,
            )
            if row.unfinished_episodes:
                return min(
                    row.request_fraction or 0.0,
                    1 - row.unfinished_episodes / len(probe.episodes),
                )
            return row.request_fraction

        return probe_capacity(check, rate, target)

    def estimate_actor_capacity(self, context, layout):
        # A representative batch suffices to rank parallel alternatives in one split.
        try:
            cfg, _, pub = self._prepare_candidate(context, layout)
            templates = [episode.turns for episode in context[1].episodes][
                : self.batch_size
            ]
            model = self.models.actor_model

            def duration(batch):
                estimate = model.estimate_batch(cfg.actor, batch)
                if estimate.peak_gib > model.hardware.memory_gb * 1e9 / 2**30 * (
                    1 - self.reserve
                ):
                    raise ValueError("Actor batch exceeds available GPU memory")
                return estimate.duration_seconds()

            return actor_capacity(templates, self.batch_size, duration, pub)
        except ValueError:
            return None

    def evaluate_many(self, context, layouts):
        """Prepare cuts in the parent and return predictions in submitted order."""
        layouts = tuple(layouts)
        if self.executor is None or len(layouts) <= 1:
            return tuple(self.evaluate(context, layout) for layout in layouts)
        spec = getattr(self.models, "worker_spec", None)
        if spec is None:
            raise ValueError("Parallel evaluation requires profile-backed cost models")
        predictions, jobs = [None] * len(layouts), []
        settings = dict(
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            staleness=self.staleness,
            memory_reserve_fraction=self.reserve,
        )
        for index, layout in enumerate(layouts):
            try:
                cfg, cut, pub = self._prepare_candidate(context, layout)
            except ValueError as exc:
                predictions[index] = Prediction(
                    layout, None, None, None, supported=False, reason=str(exc)
                )
            else:
                jobs.append(
                    _CandidateJob(
                        index,
                        spec,
                        self.models.source_ids,
                        context,
                        cfg,
                        layout,
                        cut,
                        pub,
                        self.slo,
                        settings,
                    )
                )
        for index, prediction in self.executor.map(_evaluate_candidate, jobs):
            predictions[index] = prediction
        return tuple(predictions)

    def _prepare_candidate(self, context, layout):
        obs, forecast = context
        if forecast is None:
            raise ValueError("No completed workload history")
        cfg = replace(
            self.template,
            actor_backend=layout.actor.backend,
            rollout_backend=layout.rollout.backend,
            **dict(layout.actor.options),
            **dict(layout.rollout.options),
        )
        if (layout.actor.gpus, layout.rollout.gpus) != (
            cfg.actor_gpus,
            cfg.rollout_gpus,
        ):
            raise ValueError("Backend dimensions disagree with allocated GPU counts")
        actor = actor_config(self.models.actor_template, cfg)
        rollout = rollout_config(
            self.models.rollout_template, self.models.rollout_model, cfg
        )
        execution = CandidateExecution(
            actor, rollout, cfg.rollout_max_concurrent_rollouts
        )
        model = self.models.actor_model
        pub = publication_seconds(
            model.model,
            model.hardware,
            actor.parallel,
            (rollout,),
            weight_bytes=model.model.total_parameters * 2,
            hbm_efficiency=model.efficiency.hbm,
            bandwidth=self.models.publication_GBps,
        )
        if not isfinite(pub) or pub < 0:
            raise ValueError("Unknown publication duration")
        cut = (
            Cutover(0, 0)
            if layout == obs.current
            else self.cutovers.get((obs.current, layout))
        )
        if cut is None:
            if self.reconfiguration is None:
                raise ValueError("Unknown directed cutover cost")
            cut = self.reconfiguration.predict(obs.current, layout, pub)
        if cut.seconds > forecast.input_closes_at - obs.view.now:
            raise ValueError(
                "Cutover extends beyond the forecast; partial residency unknown"
            )
        return execution, cut, pub

    def _predict(
        self,
        obs,
        forecast,
        cfg,
        layout,
        cut,
        pub,
        on_batch,
        *,
        stop_at_publication=None,
    ):
        return simulate(
            obs,
            forecast,
            cfg,
            layout,
            cut,
            pub,
            on_batch,
            stop_at_publication=stop_at_publication,
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            actor_model=self.models.actor_model,
            rollout_model=self.models.rollout_model,
            request_type=self.request_type,
            reserve=self.reserve,
            scheduler=self.scheduler,
            slo=self.slo,
            staleness=self.staleness,
        )


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


@dataclass(frozen=True)
class LoadedProfiles:
    actor_model: ExpertActorModel
    actor_template: ActorConfig
    rollout_model: ExpertRolloutModel
    rollout_template: RolloutConfig
    publication_GBps: float
    source_ids: tuple
    worker_spec: tuple


def load_profiles(
    *,
    hardware,
    topology,
    model,
    actor,
    rollout,
    bandwidth_GBps,
    inter_node_GBps=None,
    catalog: Catalog,
    calibration: Path | None = None,
    actor_runtime: ActorConfig | None = None,
):
    """Load profiles and optional fitted coefficients from JSON.

    A calibration names its hardware and model profiles. Each fitted role
    (actor, rollout) supplies runtime_revision, source_ids and a sparse
    parameters object (e.g. rollout's hbm efficiency). Include the
    topology profile when fitting network efficiency. For a live Actor pass
    ActorConfig.from_areal(resolved_actor) to use its execution settings.
    """
    actor_model, actor_template, rollout_model, rollout_template = (
        _load_profile_components(
            catalog,
            hardware,
            topology,
            model,
            actor,
            rollout,
            inter_node_GBps,
            calibration,
        )
    )
    worker_spec = (
        "analytical",
        hardware,
        topology,
        model,
        actor,
        rollout,
        bandwidth_GBps,
        inter_node_GBps,
        str(catalog.root),
        str(Path(calibration).resolve()) if calibration is not None else None,
        actor_runtime,
    )
    nonnegative("bandwidth_GBps", bandwidth_GBps, positive=True)
    actor_template = actor_runtime if actor_runtime is not None else actor_template
    source_ids = (
        fingerprint(
            "megatron-sglang-analytical-v1",
            actor_model.profile_key(actor_template),
            rollout_model.profile_key(rollout_template),
            bandwidth_GBps,
        ),
    ) + tuple(
        dict.fromkeys(actor_model.parameter_sources + rollout_model.parameter_sources)
    )
    return LoadedProfiles(
        replace(
            actor_model, advantage_model=actor_model.advantage_model or Advantages()
        ),
        actor_template,
        rollout_model,
        rollout_template,
        bandwidth_GBps,
        source_ids,
        worker_spec,
    )

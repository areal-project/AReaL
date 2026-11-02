# SPDX-License-Identifier: Apache-2.0

"""Actor phase estimates: logprob, advantage and training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from functools import lru_cache

from areal.elastic.contracts import (
    ActorMemoryRejected,
    OnlineLayout,
    Turn,
    WorkloadStats,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.fsdp import (
    FSDPHostRates,
    fsdp_work,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.operators import (
    Context,
    Efficiency,
    Operators,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
    DispatchConfig,
    ObservedPackingWork,
    PackingPlan,
    PackingRecord,
    _pad,
    dispatch_workload,
    ffd,
    observed_plan,
    plan_workload,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
    ActorConfig,
    Estimate,
    Features,
    Hardware,
    ModelShape,
    ParallelLayout,
    PhaseCalibration,
    TimedTask,
    apply_calibration,
    critical_path,
    evidence_of,
    integer,
    nonnegative,
)
from areal.elastic.reconfigurator.reference.identity import code_revision, fingerprint

from . import backend_for


@dataclass(frozen=True)
class Advantages:
    """Bandwidth + elementwise launch prior; not a fitted public GAE measurement."""

    def estimate(self, config: ActorConfig, tokens: float, op: Operators) -> float:
        return (
            op.memory(tokens / config.parallel.dp * 4 * 32)
            + 40 * op.efficiency.kernel_launch_us * 1e-06
        )


def pipeline_time(
    forward: list[list[float]], backward: list[list[float]] | None
) -> float:
    """Non-interleaved 1F1B dependencies, or forward-only pipeline; seconds."""
    stages, microbatches = (len(forward), len(forward[0]))
    tasks = []
    for stage in range(stages):
        if backward is None:
            order = [("f", m) for m in range(microbatches)]
        else:
            warmup = min(stages - stage - 1, microbatches)
            order = [("f", m) for m in range(warmup)]
            for m in range(microbatches - warmup):
                order.extend((("f", m + warmup), ("b", m)))
            order.extend(("b", m) for m in range(microbatches - warmup, microbatches))
        previous = None
        for direction, m in order:
            name = f"{direction}:{stage}:{m}"
            deps = [previous] if previous else []
            if direction == "f" and stage:
                deps.append(f"f:{stage - 1}:{m}")
            if direction == "b":
                deps.append(f"f:{stage}:{m}")
                if stage + 1 < stages:
                    deps.append(f"b:{stage + 1}:{m}")
            seconds = forward[stage][m] if direction == "f" else backward[stage][m]
            tasks.append(TimedTask(name, Estimate(seconds, "assumed"), tuple(deps)))
            previous = name
    return max((end for _, _, end in critical_path(tuple(tasks))), default=0.0)


ADVANTAGE_BOUNDARY = "trainer_host_wall_with_stats"
COMPONENTS = ("batching", "recurrence", "normalization", "transport", "diagnostics")
RANK_FEATURES = {
    "constant",
    "padded_slots",
    "gae_iterations",
    "recurrence_elements",
    "trajectories",
    "normalization_groups",
    "collectives",
}
GLOBAL_FEATURES = {"constant", "input_bytes", "output_bytes"}


@dataclass(frozen=True)
class AdvantageRankWork:
    dp_rank: int
    n_seqs: int
    padded_length: int
    trajectories: int | None = None
    turn_iterations: int | None = None
    normalization_groups: int | None = None
    collectives: int | None = None

    def __post_init__(self) -> None:
        integer("dp_rank", self.dp_rank)
        integer("n_seqs", self.n_seqs, minimum=1)
        integer("padded_length", self.padded_length, minimum=1)
        for name in (
            "trajectories",
            "turn_iterations",
            "normalization_groups",
            "collectives",
        ):
            if getattr(self, name) is not None:
                integer(name, getattr(self, name))
        if self.trajectories is not None and (
            not 1 <= self.trajectories <= self.n_seqs
        ):
            raise ValueError("Trajectories must partition rank sequences")
        if (
            self.turn_iterations is not None
            and self.turn_iterations > self.padded_length
        ):
            raise ValueError("Turn iterations exceed the padded width")

    def features(self, device: str, gae_mode: str) -> dict[str, float | None]:
        steps = (
            self.padded_length - 1
            if gae_mode == "token"
            else self.padded_length
            if device == "cuda"
            else self.turn_iterations
        )
        return {
            "constant": 1.0,
            "padded_slots": float(self.n_seqs * self.padded_length),
            "gae_iterations": steps,
            "recurrence_elements": None if steps is None else steps * self.n_seqs,
            "trajectories": self.trajectories,
            "normalization_groups": self.normalization_groups,
            "collectives": self.collectives,
        }


@dataclass(frozen=True)
class AdvantageWorkload:
    """Actual pre-GAE shapes, bound to a profile/workload; never training packing.

    input/output_bytes are total boundary transfer bytes from traces, including
    replication when applicable, not inferred from logical attended tokens.
    """

    profile: str
    workload_id: str
    ranks: tuple[AdvantageRankWork, ...]
    source_ids: tuple[str, ...]
    input_bytes: int | None = None
    output_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", tuple(self.ranks))
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        if (
            not self.profile
            or not self.workload_id
            or (not self.ranks)
            or (not self.source_ids)
        ):
            raise ValueError(
                "Observed advantage work needs profile, workload and sources"
            )
        if len({r.dp_rank for r in self.ranks}) != len(self.ranks):
            raise ValueError("Duplicate advantage DP rank")
        for name in ("input_bytes", "output_bytes"):
            if getattr(self, name) is not None:
                integer(name, getattr(self, name))

    @classmethod
    def for_workload(
        cls,
        profile: str,
        workload: WorkloadStats,
        ranks: tuple[AdvantageRankWork, ...],
        source_ids: tuple[str, ...],
        *,
        input_bytes: int | None = None,
        output_bytes: int | None = None,
    ):
        return cls(
            profile, fingerprint(workload), ranks, source_ids, input_bytes, output_bytes
        )


@dataclass(frozen=True)
class AdvantageTerm:
    """Nonnegative seconds per named work unit for an exclusive component.

    None preserves an unknown coefficient. Explicit zero requires a declared
    source/assumption too; phase-total labels must not also be added as terms.
    """

    name: str
    coefficients: tuple[tuple[str, float | None], ...]
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "coefficients", tuple(tuple(x) for x in self.coefficients)
        )
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        allowed = (
            GLOBAL_FEATURES
            if self.name in {"transport", "diagnostics"}
            else RANK_FEATURES
        )
        if (
            self.name not in COMPONENTS
            or not self.coefficients
            or (not self.source_ids)
        ):
            raise ValueError(
                "Advantage term needs a known component, coefficients and provenance"
            )
        if len(dict(self.coefficients)) != len(self.coefficients):
            raise ValueError("Duplicate advantage coefficient")
        for name, value in self.coefficients:
            if name not in allowed:
                raise ValueError(f"Unsupported {self.name} feature: {name}")
            if value is not None:
                nonnegative(name, value)

    def seconds(self, features: dict[str, float | None]) -> float | None:
        if any(
            (
                c is None or (c != 0 and features.get(k) is None)
                for k, c in self.coefficients
            )
        ):
            return None
        return sum((c * (features.get(k) or 0) for k, c in self.coefficients))


@dataclass(frozen=True)
class AdvantageEstimate:
    total: Estimate
    components: tuple[tuple[str, Estimate], ...]
    features: Features


@dataclass(frozen=True)
class AdvantageModel:
    """Scoped prior for trainer wall time: transport + slowest worker + diagnostics.

    execution_id must fingerprint normalization, GAE/lambda/KL options, tensor
    fields/dtypes and transfer/cache protocol. Mean-reduced multi-controller
    summaries and worker-only spans are not this boundary's calibration labels.
    """

    execution_id: str
    device: str
    controller_mode: str
    gae_mode: str
    terms: tuple[AdvantageTerm, ...] = ()
    timing_boundary: str = ADVANTAGE_BOUNDARY

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if (
            not self.execution_id
            or self.device not in {"cpu", "cuda"}
            or self.controller_mode not in {"single", "multi"}
            or (self.gae_mode not in {"token", "turn"})
        ):
            raise ValueError(
                "Advantage execution identity, device, controller and GAE mode are required"
            )
        if self.timing_boundary != ADVANTAGE_BOUNDARY:
            raise ValueError(
                "Actor simulation requires trainer host wall time with device stats"
            )
        if len({t.name for t in self.terms}) != len(self.terms):
            raise ValueError("Duplicate advantage component")

    def estimate(
        self,
        config: ActorConfig,
        workload: WorkloadStats,
        profile: str,
        observed: AdvantageWorkload | None = None,
        *,
        dispatch: DispatchConfig = DispatchConfig(),
    ) -> AdvantageEstimate:
        notes = [
            "Advantage trainer wall boundary includes batching, recurrence, normalization, transport and diagnostics",
            "Component coefficients are scoped priors, not GPU efficiency or identified physical constants",
            "Worker critical path uses the slowest DP; rank-mean summary labels require separate aggregation",
        ]
        sources = tuple(dict.fromkeys(s for t in self.terms for s in t.source_ids))
        if observed is not None:
            if observed.profile != profile or observed.workload_id != fingerprint(
                workload
            ):
                raise ValueError(
                    "Advantage work belongs to another profile or workload"
                )
            ranks = observed.ranks
            if {r.dp_rank for r in ranks} != set(range(config.parallel.dp)) or sum(
                r.n_seqs for r in ranks
            ) != workload.n_seqs:
                raise ValueError(
                    "Advantage observations must cover all DP sequences exactly once"
                )
            if (
                sum(r.n_seqs * r.padded_length for r in ranks)
                < workload.sequence_tokens
            ):
                raise ValueError(
                    "Advantage padded shape cannot hold the attended tokens"
                )
            by_dp = {r.dp_rank: r for r in ranks}
            if workload.per_dp_sequence_lengths is not None:
                if len(workload.per_dp_sequence_lengths) != config.parallel.dp or any(
                    (
                        by_dp[i].n_seqs != len(row) or by_dp[i].padded_length < max(row)
                        for i, row in enumerate(workload.per_dp_sequence_lengths)
                    )
                ):
                    raise ValueError(
                        "Advantage shapes disagree with observed per-DP lengths"
                    )
            elif workload.sequence_lengths is not None:
                capacities = sorted(
                    r.padded_length for r in ranks for _ in range(r.n_seqs)
                )
                if any(
                    (
                        n > width
                        for n, width in zip(
                            sorted(workload.sequence_lengths), capacities, strict=True
                        )
                    )
                ):
                    raise ValueError(
                        "Advantage shapes cannot contain the observed sequence lengths"
                    )
            elif (
                workload.max_seq_len is not None
                and max(r.padded_length for r in ranks) < workload.max_seq_len
            ):
                raise ValueError(
                    "Advantage shapes cannot contain the observed maximum length"
                )
            sources += observed.source_ids
        else:
            lengths = dispatch_workload(
                workload, workload.lengths(), config.parallel.dp, dispatch
            )
            ranks = tuple(
                (
                    AdvantageRankWork(i, len(row), max(row))
                    for i, row in enumerate(lengths)
                )
            )
            notes.append(
                "Advantage rank shapes inferred from attended lengths; actual tensor padding/dispatch may differ"
            )
        rank_features = [r.features(self.device, self.gae_mode) for r in ranks]
        global_features = {
            "constant": 1.0,
            "input_bytes": observed.input_bytes if observed else None,
            "output_bytes": observed.output_bytes if observed else None,
        }
        features = [("advantage_observed_shapes", float(observed is not None))]
        for key in sorted(RANK_FEATURES - {"constant"}):
            values = [r[key] for r in rank_features]
            if all(v is not None for v in values):
                features.extend(
                    (
                        (f"advantage_{key}_sum", float(sum(values))),
                        (f"advantage_{key}_max", float(max(values))),
                    )
                )
        for key in ("input_bytes", "output_bytes"):
            if global_features[key] is not None:
                features.append((f"advantage_{key}", float(global_features[key])))
        terms = {t.name: t for t in self.terms}
        missing = [name for name in COMPONENTS if name not in terms]
        durations = {}
        for name, term in terms.items():
            values = (
                [global_features]
                if name in {"transport", "diagnostics"}
                else rank_features
            )
            predicted = [term.seconds(row) for row in values]
            if any(v is None for v in predicted):
                missing.append(name + " coefficients or input features")
            else:
                durations[name] = predicted
        if missing:
            unknown = Estimate(
                None,
                "unsupported",
                notes=tuple(notes + ["Missing advantage " + ", ".join(missing)]),
                source_ids=sources,
            )
            return AdvantageEstimate(unknown, (), tuple(features))
        workers = [
            sum(durations[name][i] for name in COMPONENTS[:3])
            for i in range(len(ranks))
        ]
        critical = max(range(len(ranks)), key=workers.__getitem__)
        components = tuple(
            (
                name,
                Estimate(
                    durations[name][0 if name in COMPONENTS[3:] else critical],
                    "assumed",
                    notes=(
                        "Uncalibrated component prior on the selected slowest-worker path; excludes phase-level calibration",
                    ),
                    source_ids=terms[name].source_ids,
                ),
            )
            for name in COMPONENTS
        )
        total = Estimate(
            sum((e.seconds for _, e in components)),
            "assumed",
            notes=tuple(notes),
            source_ids=tuple(dict.fromkeys(sources)),
        )
        return AdvantageEstimate(total, components, tuple(features))


def _features(workload: WorkloadStats) -> Features:
    values = [
        ("advantage_sequence_tokens", workload.sequence_tokens),
        ("advantage_n_seqs", float(workload.n_seqs)),
        ("advantage_mean_seq_len", workload.sequence_tokens / workload.n_seqs),
    ]
    maximum = workload.max_seq_len
    if maximum is None and workload.sequence_lengths is not None:
        maximum = max(workload.sequence_lengths)
    if maximum is None and workload.per_dp_sequence_lengths is not None:
        maximum = max(max(row) for row in workload.per_dp_sequence_lengths)
    if maximum is not None:
        values.append(("advantage_max_seq_len", float(maximum)))
    return tuple(values)


@dataclass(frozen=True)
class AdvantageObservation:
    """One calibration-only phase label with an explicitly audited timing scope.

    sample_id identifies the source run/step. Workload uses global attention
    tokens once, excluding TP/CP/PP copies and repeated PPO uses.
    """

    sample_id: str
    profile: str
    workload: WorkloadStats
    measured_seconds: float
    timing_boundary: str

    def __post_init__(self) -> None:
        if not self.sample_id or not self.profile or (not self.timing_boundary):
            raise ValueError(
                "Advantage labels require source, profile and timing boundary"
            )
        nonnegative("measured_seconds", self.measured_seconds, positive=True)


@dataclass(frozen=True)
class AggregateAdvantageModel:
    """Explicit alternative to the component prior, scoped to one fitted profile.

    execution_id identifies controller/device/GAE/normalization options and label
    aggregation. A run-scoped identity is appropriate when these cannot yet be
    separated. Parameters describe the whole phase, never identified components.
    calibration has scale=1 and only enforces scope/bounds and records sources;
    do not fit another phase scale on top of this already calibrated predictor.
    """

    execution_id: str
    form: str = "affine_tokens"
    timing_boundary: str = ADVANTAGE_BOUNDARY
    calibration: PhaseCalibration | None = None
    intercept_seconds: float | None = None
    seconds_per_token: float | None = None

    def __post_init__(self) -> None:
        if not self.execution_id or self.form not in {"affine_tokens", "constant"}:
            raise ValueError(
                "Aggregate advantage requires execution identity and known form"
            )
        if self.timing_boundary != ADVANTAGE_BOUNDARY:
            raise ValueError(
                "Aggregate advantage requires trainer host wall time with stats"
            )
        fitted = self.calibration is not None
        if any(
            (value is not None) != fitted
            for value in (self.intercept_seconds, self.seconds_per_token)
        ):
            raise ValueError(
                "Aggregate coefficients and calibration must be supplied together"
            )
        if not fitted:
            return
        nonnegative("intercept_seconds", self.intercept_seconds)
        nonnegative("seconds_per_token", self.seconds_per_token)
        if self.intercept_seconds == self.seconds_per_token == 0:
            raise ValueError("A positive phase label cannot fit a zero predictor")
        if self.form == "constant" and self.seconds_per_token != 0:
            raise ValueError("Constant model cannot have a token slope")
        scope = self.calibration
        if (
            scope.phase != "compute_advantage"
            or scope.scale != 1
            or scope.timing_boundary != self.timing_boundary
        ):
            raise ValueError(
                "Aggregate calibration must only guard the matching phase boundary"
            )
        required = {
            "advantage_sequence_tokens",
            "advantage_n_seqs",
            "advantage_mean_seq_len",
        }
        if not required <= {name for name, _, _ in scope.bounds}:
            raise ValueError(
                "Aggregate calibration is missing required workload bounds"
            )

    @property
    def scope_spec(self) -> tuple[str, ...]:
        """Stable before/after fitting; coefficients remain in simulator snapshots."""
        return (
            "aggregate-advantage-v1",
            self.execution_id,
            self.form,
            self.timing_boundary,
        )

    def fit(
        self, version: str, observations: tuple[AdvantageObservation, ...]
    ) -> AggregateAdvantageModel:
        """Nonnegative least squares for one preselected form, using calibration only."""
        if len(observations) < 2:
            raise ValueError("At least two calibration observations are required")
        first = observations[0]
        features = [_features(point.workload) for point in observations]
        names = tuple((name for name, _ in features[0]))
        if any(
            point.profile != first.profile
            or point.timing_boundary != self.timing_boundary
            for point in observations
        ):
            raise ValueError(
                "Aggregate labels must share the full profile and trainer wall boundary"
            )
        if any(tuple((name for name, _ in row)) != names for row in features):
            raise ValueError("Aggregate labels must share available summary features")
        guard = PhaseCalibration(
            version,
            first.profile,
            "compute_advantage",
            1.0,
            tuple(
                (
                    name,
                    min(dict(row)[name] for row in features),
                    max(dict(row)[name] for row in features),
                )
                for name in names
            ),
            tuple(point.sample_id for point in observations),
            self.timing_boundary,
        )
        x = [point.workload.sequence_tokens for point in observations]
        y = [point.measured_seconds for point in observations]
        xscale, yscale = (max(x), max(y))
        x, y = ([v / xscale for v in x], [v / yscale for v in y])
        mean_x, mean_y = (math.fsum(x) / len(x), math.fsum(y) / len(y))
        candidates = [(mean_y, 0.0)]
        if self.form == "affine_tokens":
            variance = math.fsum((v - mean_x) ** 2 for v in x)
            if variance <= 1e-24 * len(x):
                raise ValueError(
                    "Affine fit needs distinct token counts; select constant explicitly"
                )
            slope = (
                math.fsum(
                    ((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
                )
                / variance
            )
            intercept = mean_y - slope * mean_x
            if slope >= 0 and intercept >= 0:
                candidates.append((intercept, slope))
            candidates.append(
                (
                    0.0,
                    math.fsum((a * b for a, b in zip(x, y, strict=True)))
                    / math.fsum(a * a for a in x),
                )
            )
        intercept, slope = min(
            candidates,
            key=lambda pair: math.fsum(
                ((pair[0] + pair[1] * a - b) ** 2 for a, b in zip(x, y, strict=True))
            ),
        )
        return replace(
            self,
            calibration=guard,
            intercept_seconds=intercept * yscale,
            seconds_per_token=slope * yscale / xscale,
        )

    def estimate(self, workload: WorkloadStats, profile: str) -> AdvantageEstimate:
        features = _features(workload)
        notes = (
            "Aggregate empirical advantage phase; no identified component decomposition",
            "Same-profile empirical fit within observed feature bounds; no cross-configuration accuracy claim",
        )
        if self.calibration is None:
            result = Estimate(
                None, "unsupported", notes=notes + ("Missing aggregate advantage fit",)
            )
        else:
            checked = self.calibration.apply(
                1.0,
                profile,
                "compute_advantage",
                features,
                timing_boundary=self.timing_boundary,
            )
            if checked.seconds is not None:
                checked = replace(
                    checked,
                    seconds=self.intercept_seconds
                    + self.seconds_per_token * workload.sequence_tokens,
                )
            result = replace(checked, notes=checked.notes + notes)
        return AdvantageEstimate(result, (), features)


STAGES = ("recompute_logp", "compute_advantage", "train_step")


@dataclass(frozen=True)
class ObservedPacking:
    """A plan bound to one profile and workload, after rank de-duplication."""

    profile: str
    plan: PackingPlan
    source_ids: tuple[str, ...]
    workload_id: str

    def __post_init__(self) -> None:
        if not self.profile or not self.source_ids or (not self.workload_id):
            raise ValueError("Observed packing requires profile and provenance")
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        object.__setattr__(self, "plan", replace(self.plan, count_source="observed"))

    @classmethod
    def for_workload(
        cls,
        profile: str,
        workload: WorkloadStats,
        plan: PackingPlan,
        source_ids: tuple[str, ...],
    ) -> ObservedPacking:
        return cls(profile, plan, source_ids, fingerprint(workload))

    @classmethod
    def from_records(
        cls,
        profile: str,
        config: ActorConfig,
        workload: WorkloadStats,
        records: tuple[PackingRecord, ...],
        *,
        runtime_revision: str,
    ) -> ObservedPacking:
        plan, source_ids = observed_plan(
            config,
            workload,
            tuple(records),
            profile=profile,
            runtime_revision=runtime_revision,
        )
        return cls.for_workload(profile, workload, plan, source_ids)


@dataclass(frozen=True)
class ActorEstimate:
    """Phase results; advantage_components expose uncalibrated priors only.

    A phase-total scale does not independently identify its internal components.
    Use stages for calibrated wall time, not the sum of these prior diagnostics.
    """

    profile: str
    stages: tuple[tuple[str, Estimate], ...]
    features: Features
    packing: PackingPlan | None
    static_memory_gb: tuple[float, ...] = ()
    timing_boundary: str = ADVANTAGE_BOUNDARY
    advantage_components: tuple[tuple[str, Estimate], ...] = ()
    advantage_features: Features = ()
    fsdp_components: tuple[tuple[str, Estimate], ...] = ()
    fsdp_work: tuple[tuple[str, Features], ...] = ()
    host_memory_gb: tuple[float, ...] = ()
    gpu_working_set_lower_bounds_gb: tuple[tuple[str, float], ...] = ()

    def features_for(self, phase: str) -> Features:
        return self.features + (
            self.advantage_features if phase == "compute_advantage" else ()
        )

    @property
    def total(self) -> Estimate:
        values = [value for _, value in self.stages]
        feasibility = (
            "infeasible"
            if any(v.feasibility == "infeasible" for v in values)
            else "unknown"
        )
        if any(v.seconds is None for v in values):
            return Estimate(
                None,
                "unsupported",
                feasibility,
                tuple(note for v in values for note in v.notes),
            )
        active = tuple(v.evidence for v in values if v.seconds)
        return Estimate(
            sum(v.seconds for v in values),
            evidence_of(active),
            feasibility,
            tuple(dict.fromkeys(note for v in values for note in v.notes)),
            tuple(dict.fromkeys(s for v in values for s in v.source_ids)),
        )


@dataclass(frozen=True)
class ExpertActorModel:
    model: ModelShape
    hardware: Hardware
    runtime_revision: str
    efficiency: Efficiency = field(default_factory=Efficiency)
    group_gbps: tuple[tuple[str, float], ...] = ()
    calibration: tuple[PhaseCalibration, ...] = ()
    version: str = "expert-actor-v2"
    advantage_model: AdvantageModel | AggregateAdvantageModel | None = None
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)
    fsdp_host_rates: FSDPHostRates | None = None
    execution_identity: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_gbps", tuple(tuple(x) for x in self.group_gbps))
        object.__setattr__(self, "calibration", tuple(self.calibration))
        if not isinstance(self.dispatch, DispatchConfig):
            raise ValueError("Expert dispatch requires a DispatchConfig")
        if (
            isinstance(self.advantage_model, AdvantageModel)
            and self.dispatch.mode != "proxy"
        ):
            expected = "single" if self.dispatch.mode == "controller" else "multi"
            if self.advantage_model.controller_mode != expected:
                raise ValueError(
                    "Advantage controller mode disagrees with Actor dispatch"
                )
        if (
            not self.version
            or not self.runtime_revision
            or len(dict(self.group_gbps)) != len(self.group_gbps)
        ):
            raise ValueError("Version and unique bandwidth groups are required")
        Operators(self.hardware, self.efficiency, dict(self.group_gbps))
        if isinstance(self.advantage_model, AggregateAdvantageModel) and any(
            c.phase == "compute_advantage" for c in self.calibration
        ):
            raise ValueError(
                "Aggregate advantage is already fitted; do not add phase-scale calibration"
            )

    def profile_key(self, config: ActorConfig) -> str:
        import numpy as np

        identity = (
            code_revision(),
            ("numpy-argsort", np.__version__),
            self.version,
            self.runtime_revision,
            self.dispatch,
            config,
            self.model,
            self.hardware,
            self.efficiency,
            self.group_gbps,
            self.fsdp_host_rates,
            self.advantage_model.scope_spec
            if isinstance(self.advantage_model, AggregateAdvantageModel)
            else self.advantage_model,
        )
        if self.execution_identity:
            identity += (self.execution_identity,)
        return fingerprint(*identity)

    def observe_packing(
        self,
        config: ActorConfig,
        workload: WorkloadStats,
        records: tuple[PackingRecord, ...],
    ) -> ObservedPacking:
        """Validate attributed measurements against this exact model profile."""
        return ObservedPacking.from_records(
            self.profile_key(config),
            config,
            workload,
            records,
            runtime_revision=self.runtime_revision,
        )

    def observe_packing_work(
        self,
        config: ActorConfig,
        workload: WorkloadStats,
        records: tuple[PackingRecord, ...],
    ) -> ObservedPackingWork:
        """Keep attributable partial work for diagnostics, without invented lengths."""
        return ObservedPackingWork(
            config, workload, self.profile_key(config), self.runtime_revision, records
        )

    def estimate(
        self,
        config: ActorConfig,
        workload: WorkloadStats,
        *,
        source_profile: str | None = None,
        observed_packing: ObservedPacking | None = None,
        auxiliary_stages: tuple[tuple[str, Estimate], ...] = (),
        advantage_workload: AdvantageWorkload | None = None,
    ) -> ActorEstimate:
        profile = self.profile_key(config)
        features = (
            ("sequence_tokens", workload.sequence_tokens),
            ("n_seqs", float(workload.n_seqs)),
        )
        if (
            workload.per_dp_sequence_lengths is not None
            or workload.train_microbatches is not None
        ) and source_profile != profile:
            raise ValueError(
                "Rank observations require their exact source_profile; re-pack new layouts"
            )
        if observed_packing is not None and observed_packing.profile != profile:
            raise ValueError("Observed packing belongs to another profile")
        if observed_packing is not None and observed_packing.workload_id != fingerprint(
            workload
        ):
            raise ValueError("Observed packing belongs to another workload")
        p, m = (config.parallel, self.model)
        backend = backend_for(config.backend)
        if not backend.mesh_is_valid(config, m):
            return ActorEstimate(
                profile,
                (
                    (
                        "train_step",
                        Estimate(
                            None,
                            "unsupported",
                            "infeasible",
                            ("Model dimensions do not divide mesh",),
                        ),
                    ),
                ),
                features,
                None,
            )
        try:
            plan = (
                observed_packing.plan
                if observed_packing
                else plan_workload(
                    config, workload, m.hidden_size, dispatch=self.dispatch
                )
            )
            partition = backend.pipeline_partition(config, m)
        except ValueError as exc:
            if observed_packing:
                raise
            return ActorEstimate(
                profile,
                (
                    (
                        "train_step",
                        Estimate(None, "unsupported", "infeasible", (str(exc),)),
                    ),
                ),
                features,
                None,
            )
        self._check_plan(config, workload, plan)
        features += (
            ("train_compute_tokens", float(plan.train_compute_tokens)),
            ("train_microbatches", float(sum(len(update[0]) for update in plan.train))),
            ("exact_lengths", float(plan.exact_lengths)),
            ("exact_dispatch", float(plan.exact_dispatch)),
            ("observed_packing", float(observed_packing is not None)),
        )
        if workload.sequence_lengths is not None:
            features += (("max_seq_len", float(max(workload.sequence_lengths))),)
            features += (
                (
                    "sequence_length_squared",
                    float(sum(n * n for n in workload.sequence_lengths)),
                ),
            )
        elif workload.max_seq_len is not None:
            features += (("max_seq_len", workload.max_seq_len),)
        op = Operators(
            self.hardware,
            self.efficiency,
            dict(self.group_gbps),
            p.world_size // p.pp,
            p.tp * p.cp,
        )
        contexts = tuple(
            (
                Context(config, m, op, count, i == 0, i == p.pp - 1, backend=backend)
                for i, count in enumerate(partition)
            )
        )
        raw = {
            "recompute_logp": self.estimate_logprob(plan, contexts, backend),
            "train_step": self.estimate_train(plan, contexts, backend),
        }
        host_components = ()
        host_work = ()
        host_memory = ()
        if config.backend == "fsdp":
            host_work = fsdp_work(config, m, plan)
            rates = self.fsdp_host_rates or FSDPHostRates()
            host_components = tuple(
                ((phase, rates.estimate(work)) for phase, work in host_work)
            )
            host_memory = (
                (m.total_parameters / p.dp * 10 / 1000000000.0,)
                if config.offload_params
                else (0.0,)
            )
        advantage = self.estimate_advantage(
            config, workload, profile, advantage_workload
        )
        advantage_features = advantage.features if advantage else ()
        memory = []
        for ctx in contexts:
            dense, expert = ctx.owned_parameters
            sharded = dense / (p.dp * p.cp) + expert / p.expert_dp
            stored = (
                (0 if config.offload_params else sharded * 10)
                if config.backend == "fsdp"
                else sharded * 16
                if config.backend == "archon"
                else (dense + expert) * 2 + sharded * 16
                if config.distributed_optimizer
                else (dense + expert) * 18
            )
            if config.precision_aware_optimizer:
                owned = sharded if config.distributed_optimizer else dense + expert
                stored = (dense + expert) * (2 + config.grad_bytes) + owned * (
                    8 + (config.optimizer_master_storage_bytes or 0)
                )
            memory.append(stored / 1000000000.0)
        impossible = max(memory) > self.hardware.memory_gb
        gpu_working_set = ()
        if config.backend == "fsdp":
            largest_layer = (
                m.attention_parameters_per_layer
                + m.expert_parameters_per_layer
                + m.router_parameters_per_layer
                + 2 * m.hidden_size
                + 2 * m.head_dim
            )
            largest_group = 2 * max(largest_layer, m.head_parameters + m.hidden_size)
            largest_logits = (
                2
                * m.vocab_size
                * max(
                    mb.compute_tokens
                    for call in plan.train
                    for rank in call
                    for mb in rank
                )
            )
            gpu_working_set = (
                ("largest_parameter_group", largest_group / 1000000000.0),
                ("largest_bf16_logits", largest_logits / 1000000000.0),
            )
            impossible = (
                impossible
                or max(largest_group, largest_logits) / 1000000000.0
                > self.hardware.memory_gb
            )
        notes = plan.assumptions + (
            "Analytical kernels, routing and communication overlap; target measurements required",
            "Non-interleaved 1F1B dependency schedule; no interleaved/virtual pipeline",
            "Static memory is a lower bound excluding activations and temporary buffers",
        )
        if config.overlap_grad_reduce:
            notes += (
                "overlap_grad_reduce is a serial communication upper prior; its benefit requires per-profile calibration",
            )
        if m.mlp_type == "dense" and p.tp > 1:
            notes += (
                "Dense TP uses two all-reduce-equivalent volumes per layer; sequence-parallel AG/RS latency and overlap remain priors",
            )
        if (
            config.precision_aware_optimizer
            and config.optimizer_master_storage_bytes is None
        ):
            notes += (
                "Precision-aware Adam master/remainder storage is unknown and excluded from the static memory lower bound; optimizer traffic uses a full-master prior",
            )
        stages = []
        for name in STAGES:
            if name == "compute_advantage":
                result = (
                    advantage.total
                    if advantage
                    else Estimate(
                        None,
                        "unsupported",
                        notes=(
                            "Missing advantage execution model and component coefficients; HBM-only timing is not trainer wall time",
                        ),
                    )
                )
                if result.seconds is not None and (
                    not isinstance(self.advantage_model, AggregateAdvantageModel)
                ):
                    calibrated = apply_calibration(
                        result.seconds,
                        profile,
                        name,
                        features + advantage_features,
                        self.calibration,
                        timing_boundary=ADVANTAGE_BOUNDARY,
                    )
                    result = replace(
                        calibrated,
                        notes=calibrated.notes + result.notes,
                        source_ids=tuple(
                            dict.fromkeys(calibrated.source_ids + result.source_ids)
                        ),
                    )
            else:
                seconds = raw[name]
                host = dict(host_components).get(name)
                if host is not None and host.seconds is None:
                    stages.append((name, replace(host, notes=host.notes + notes)))
                    continue
                if host is not None:
                    seconds += host.require_seconds()
                result = (
                    apply_calibration(
                        seconds,
                        profile,
                        name,
                        features,
                        self.calibration,
                        timing_boundary=ADVANTAGE_BOUNDARY,
                    )
                    if seconds
                    else Estimate(
                        0.0, "assumed", notes=("Stage disabled by configuration",)
                    )
                )
                if host is not None:
                    result = replace(
                        result,
                        notes=result.notes + host.notes,
                        source_ids=tuple(
                            dict.fromkeys(result.source_ids + host.source_ids)
                        ),
                    )
            provenance = observed_packing.source_ids if observed_packing else ()
            stages.append(
                (
                    name,
                    replace(
                        result,
                        feasibility="infeasible" if impossible else "unknown",
                        notes=result.notes + notes,
                        source_ids=tuple(dict.fromkeys(result.source_ids + provenance)),
                    ),
                )
            )
        if len({name for name, _ in auxiliary_stages}) != len(auxiliary_stages) or any(
            (name in STAGES for name, _ in auxiliary_stages)
        ):
            raise ValueError("Auxiliary stages must have distinct names")
        before = tuple(
            pair
            for pair in auxiliary_stages
            if pair[0] in {"critic_values", "ref_logp", "teacher_logp"}
        )
        after = tuple(
            pair
            for pair in auxiliary_stages
            if pair[0] not in {"critic_values", "ref_logp", "teacher_logp"}
        )
        return ActorEstimate(
            profile,
            before + tuple(stages) + after,
            features,
            plan,
            tuple(memory),
            ADVANTAGE_BOUNDARY,
            advantage.components if advantage else (),
            advantage_features,
            host_components,
            host_work,
            host_memory,
            gpu_working_set,
        )

    @staticmethod
    def _check_plan(
        config: ActorConfig, workload: WorkloadStats, plan: PackingPlan
    ) -> None:
        if (
            len(plan.train) != config.ppo_n_minibatches
            or bool(plan.logprob) != config.compute_prox_logp
        ):
            raise ValueError("Packing plan does not match PPO/logprob configuration")
        for ranks in (*plan.train, *((plan.logprob,) if plan.logprob else ())):
            if (
                len(ranks) != config.parallel.dp
                or not ranks[0]
                or any(len(r) != len(ranks[0]) for r in ranks)
            ):
                raise ValueError(
                    "Packing plan must contain synchronized nonempty DP ranks"
                )
            for rank in ranks:
                for mb in rank:
                    if (
                        mb.real_tokens <= 0
                        or mb.compute_tokens < mb.real_tokens
                        or mb.attention_length_squared <= 0
                    ):
                        raise ValueError("Invalid observed microbatch work")
        total = sum(
            mb.real_tokens for update in plan.train for rank in update for mb in rank
        )
        if abs(total - workload.sequence_tokens) > 1:
            raise ValueError("Packing plan changes logical training token count")
        if (
            plan.logprob
            and abs(
                sum(mb.real_tokens for rank in plan.logprob for mb in rank)
                - workload.sequence_tokens
            )
            > 1
        ):
            raise ValueError("Packing plan changes logical logprob token count")

    def estimate_logprob(self, plan, contexts, backend):
        return (
            self._transformer_seconds(plan.logprob, contexts, backend, False)
            if plan.logprob
            else 0.0
        )

    def estimate_advantage(self, config, workload, profile, advantage_workload):
        advantage = None
        if isinstance(self.advantage_model, AggregateAdvantageModel):
            if advantage_workload is not None:
                raise ValueError(
                    "Aggregate advantage uses summary workload, not component observations"
                )
            advantage = self.advantage_model.estimate(workload, profile)
        elif self.advantage_model is not None:
            advantage = self.advantage_model.estimate(
                config, workload, profile, advantage_workload, dispatch=self.dispatch
            )
        elif advantage_workload is not None:
            raise ValueError(
                "Advantage observations require an explicit execution model"
            )
        return advantage

    def estimate_train(self, plan, contexts, backend):
        train = 0.0
        for ranks in plan.train:
            train += self._transformer_seconds(ranks, contexts, backend, True)
            updates = [backend.optimizer_update(ctx) for ctx in contexts]
            train += max(sum(u.values()) for u in updates)
        return train

    def _transformer_seconds(self, ranks, contexts, backend, training):
        forward = [[] for _ in contexts]
        backward = [[] for _ in contexts]
        for s, ctx in enumerate(contexts):
            for index in range(len(ranks[0])):
                parts = []
                for rank in ranks:
                    local = replace(ctx, microbatch_index=index, training=training)
                    f = backend.forward_cost(local, rank[index])
                    ftime = f["compute"] + f["communication_exposed"]
                    total = backend.execution_cost(
                        local, rank[index], training=training
                    )
                    ttime = total["compute"] + total["communication_exposed"]
                    parts.append((ftime, max(0.0, ttime - ftime)))
                forward[s].append(max(v[0] for v in parts))
                backward[s].append(max(v[1] for v in parts))
        return _pipeline_time(forward, backward if training else None)


_pipeline_time = pipeline_time


class CalibratedActorModel:
    def __init__(self, fits, cal_actor, actor_template, mechanism, inter_node_dp_GBps):
        self.fits = fits
        self.cal_actor = cal_actor
        self.actor_template = actor_template
        self.mechanism = mechanism
        self.inter_node_dp_GBps = inter_node_dp_GBps

    def actor(self, layout: OnlineLayout, batch: tuple[tuple[Turn, ...], ...]):
        return self._actor(layout.actor_dp, layout.actor_tp, batch)

    def actor_memory(self, layout, batch):
        """Expose the unchanged physical estimate for reconfigurator-only headroom checks."""
        _, peak, limit, measured_oom = self._actor_estimate(
            layout.actor_dp, layout.actor_tp, batch
        )
        return (peak, limit, measured_oom)

    @lru_cache(maxsize=512)
    def _actor(self, dp: int, tp: int, batch: tuple[tuple[Turn, ...], ...]):
        stages, peak, limit, measured_oom = self._actor_estimate(dp, tp, batch)
        if peak > limit or measured_oom:
            raise ActorMemoryRejected(
                f"Conditional Actor memory rejection: {peak:.3f}GiB / {limit:.3f}GiB; measured_shape_oom={measured_oom}"
            )
        return stages

    @lru_cache(maxsize=4096)
    def _actor_estimate(self, dp: int, tp: int, batch: tuple[tuple[Turn, ...], ...]):
        base = self.cal_actor.base
        if dp * tp > 8:
            if self.inter_node_dp_GBps is None:
                raise ValueError(
                    "Cross-node Actor needs an explicit assumed DP bandwidth"
                )
            groups = dict(base.group_gbps)
            groups.update(
                dense_dp=self.inter_node_dp_GBps, expert_dp=self.inter_node_dp_GBps
            )
            base = replace(base, group_gbps=tuple(sorted(groups.items())))
        config = replace(self.actor_template, parallel=ParallelLayout(dp=dp, tp=tp))
        lengths = tuple(
            t.input_tokens + t.output_tokens for episode in batch for t in episode
        )
        work = WorkloadStats(
            len(lengths),
            sum(lengths),
            loss_tokens=sum(t.output_tokens for e in batch for t in e),
            sequence_lengths=lengths,
            trajectory_group_sizes=tuple(len(e) for e in batch),
            trajectory_dispatch_weights=tuple(
                len(e) * max(t.input_tokens + t.output_tokens for t in e) for e in batch
            ),
        )
        raw, features = self.fits.actor_features(base, config, work)
        if not features:
            raise ValueError("Actor workload packing is unsupported")
        memory = self._memory_features(config, work, raw)
        peak = sum(
            (
                a * b
                for a, b in zip(
                    memory, self.mechanism["memory"]["coefficients"], strict=True
                )
            )
        )
        limit = self.cal_actor.base.hardware.memory_gb * 1000000000.0 / 2**30
        rows = dispatch_workload(
            work, list(work.lengths()), config.parallel.dp, self.cal_actor.base.dispatch
        )
        measured_oom = any(
            (dp, tp, tuple(sorted(row))) in self.cal_actor.oom_shapes for row in rows
        )
        result = []
        for phase, values in features.items():
            fit = next(
                (
                    f
                    for tp, name, f in self.cal_actor.fits
                    if tp == config.parallel.tp and name == phase
                )
            )
            seconds, _ = fit.evaluate(values, extrapolate=True)
            result.append((phase, seconds))
        return (tuple(result), peak, limit, measured_oom)

    def _memory_features(self, config, work, raw):
        base = self.cal_actor.base
        rows = dispatch_workload(
            work, list(work.lengths()), config.parallel.dp, base.dispatch
        )
        dense, expert = Context(
            config,
            base.model,
            Operators(base.hardware, base.efficiency),
            base.model.layers,
            True,
            True,
        ).owned_parameters
        persistent = (
            4 * (dense + expert) + 12 * (dense + expert) / config.parallel.dp
        ) / 2**30
        maximum = max(
            m.compute_tokens
            for update in raw.packing.train
            for rank in update
            for m in rank
        )
        parts = [ffd(row, sum(row), config.ppo_n_minibatches) for row in rows]
        repack = 0.0
        for update, ranks in enumerate(raw.packing.train):
            minimum = max(config.minimum_microbatches, work.train_microbatches or 1)
            divisor = (
                config.parallel.pp
                if config.backend == "megatron" or config.parallel.pp > 1
                else config.n_mbs_divisor
            )
            while True:
                grouped = [
                    ffd(parts[rank][update], config.max_tokens_per_mb, minimum, divisor)
                    for rank in range(len(rows))
                ]
                count = max(map(len, grouped))
                if all(len(bins) == count for bins in grouped):
                    break
                minimum = count
            for rank, batches in enumerate(ranks):
                bins = grouped[rank]
                if (
                    tuple(_pad(b, config, base.model.hidden_size) for b in bins)
                    != batches
                ):
                    raise ValueError(
                        "Memory feature replay disagrees with frozen Actor packing"
                    )
                repack = max(repack, *(sum(b) * (1 - 1 / len(b)) for b in bins))
        return (
            1.0,
            persistent,
            maximum / config.parallel.tp / 32768,
            repack / config.parallel.tp / 32768,
        )

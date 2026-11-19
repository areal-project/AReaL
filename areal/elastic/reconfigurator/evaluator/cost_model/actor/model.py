# SPDX-License-Identifier: Apache-2.0

"""Actor phase estimates: logprob, advantage and training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Protocol

from areal.elastic.reconfigurator.evaluator.cost_model import code_revision, fingerprint
from areal.elastic.reconfigurator.evaluator.cost_model.actor.megatron import (
    Context,
    Efficiency,
    Operators,
)
from areal.elastic.reconfigurator.evaluator.cost_model.actor.packing import (
    DispatchConfig,
    PackingPlan,
    dispatch_workload,
    plan_workload,
)
from areal.elastic.reconfigurator.evaluator.cost_model.contracts import (
    ActorConfig,
    Estimate,
    Features,
    Hardware,
    ModelShape,
    TimedTask,
    critical_path,
    evidence_of,
    integer,
    nonnegative,
)
from areal.elastic.reconfigurator.evaluator.workload import (
    WorkloadStats,
)

from .megatron import backend_for


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
    fields/dtypes and transfer/cache protocol for the single-controller boundary.
    """

    execution_id: str
    device: str
    gae_mode: str
    terms: tuple[AdvantageTerm, ...] = ()
    timing_boundary: str = ADVANTAGE_BOUNDARY

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if (
            not self.execution_id
            or self.device not in {"cpu", "cuda"}
            or (self.gae_mode not in {"token", "turn"})
        ):
            raise ValueError(
                "Advantage execution identity, device and GAE mode are required"
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


STAGES = ("recompute_logp", "compute_advantage", "train_step")


class ObservedPacking(Protocol):
    """A packing plan already attributed by the caller."""

    profile: str
    plan: PackingPlan
    source_ids: tuple[str, ...]
    workload_id: str


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
    gpu_working_set_lower_bounds_gb: tuple[tuple[str, float], ...] = ()
    # Analytical per-stage peak: persistent state + live activations/workspaces.
    peak_memory_gb: tuple[float, ...] = ()

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
    parameter_sources: tuple[str, ...] = ()
    version: str = "expert-actor-v2"
    advantage_model: AdvantageModel | Advantages | None = None
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)
    execution_identity: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_gbps", tuple(tuple(x) for x in self.group_gbps))
        if not isinstance(self.dispatch, DispatchConfig):
            raise ValueError("Expert dispatch requires a DispatchConfig")
        if (
            not self.version
            or not self.runtime_revision
            or len(dict(self.group_gbps)) != len(self.group_gbps)
        ):
            raise ValueError("Version and unique bandwidth groups are required")
        Operators(self.hardware, self.efficiency, dict(self.group_gbps))

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
            self.advantage_model,
        )
        if self.execution_identity:
            identity += (self.execution_identity,)
        return fingerprint(*identity)

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
            p.ep * p.etp,
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
        advantage = self.estimate_advantage(
            config, workload, profile, advantage_workload
        )
        advantage_features = advantage.features if advantage else ()
        memory, peaks = [], []
        maximum_tokens = max(
            mb.compute_tokens for update in plan.train for rank in update for mb in rank
        )
        maximum_sequence = workload.max_seq_len or max(
            workload.sequence_lengths
            or (math.ceil(workload.sequence_tokens / workload.n_seqs),)
        )
        for stage, ctx in enumerate(contexts):
            dense, expert = ctx.owned_parameters
            sharded = dense / (p.dp * p.cp) + expert / p.expert_dp
            # BF16 weights and a full FP32 gradient buffer; Adam/master states
            # are partitioned across the corresponding dense/expert DP group.
            stored = (
                (dense + expert) * (2 + config.grad_bytes) + sharded * 12
                if config.distributed_optimizer
                else (dense + expert) * 18
            )
            if config.precision_aware_optimizer:
                owned = sharded if config.distributed_optimizer else dense + expert
                stored = (dense + expert) * (2 + config.grad_bytes) + owned * (
                    8 + (config.optimizer_master_storage_bytes or 0)
                )
            memory.append(stored / 1000000000.0)
            live = min(len(plan.train[0][0]), p.pp - stage)
            # SP partitions saved layer inputs; checkpointing saves only inputs.
            local_tokens = maximum_tokens / (p.tp * p.cp)
            saved = 2 * local_tokens * m.hidden_size * ctx.layers
            if not config.gradient_checkpointing:
                saved += (
                    local_tokens
                    * ctx.layers
                    * (
                        8 * m.hidden_size
                        + 6 * m.intermediate_size
                        + 2 * m.attention_heads * maximum_sequence
                    )
                )
            workspace = local_tokens * (
                8 * m.hidden_size + 4 * m.intermediate_size * m.top_k
            )
            logits_tokens = maximum_tokens / p.cp
            if config.enable_chunked_logits and config.lm_head_loss_chunk_size:
                logits_tokens = min(logits_tokens, config.lm_head_loss_chunk_size)
            logits = 4 * logits_tokens * m.vocab_size / p.tp if ctx.last else 0
            # Unknown precision-aware master/remainder storage uses an explicit
            # full-master capacity prior rather than silently treating it as zero.
            unknown_master = (
                4 * (sharded if config.distributed_optimizer else dense + expert)
                if config.precision_aware_optimizer
                and config.optimizer_master_storage_bytes is None
                else 0
            )
            peaks.append(
                (stored + unknown_master + live * saved + workspace + logits) / 1e9
            )
        impossible = max(peaks) > self.hardware.memory_gb
        gpu_working_set = ()
        notes = plan.assumptions + (
            "Analytical kernels, routing and communication overlap; target measurements required",
            "Non-interleaved 1F1B dependency schedule; no interleaved/virtual pipeline",
            "Peak memory uses analytical saved-activation, routing/workspace and logits priors; no allocator trace calibration",
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
            else:
                seconds = raw[name]
                result = (
                    Estimate(seconds, "assumed")
                    if seconds
                    else Estimate(
                        0.0, "assumed", notes=("Stage disabled by configuration",)
                    )
                )
            provenance = self.parameter_sources + (
                observed_packing.source_ids if observed_packing else ()
            )
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
            gpu_working_set,
            tuple(peaks),
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
        if isinstance(self.advantage_model, Advantages):
            seconds = self.advantage_model.estimate(
                config,
                workload.sequence_tokens,
                Operators(self.hardware, self.efficiency),
            )
            advantage = AdvantageEstimate(
                Estimate(
                    seconds,
                    "assumed",
                    notes=(
                        "Explicit bandwidth/launch advantage prior; trainer transport and host overhead are not calibrated",
                    ),
                ),
                (),
                (),
            )
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

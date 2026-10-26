# SPDX-License-Identifier: Apache-2.0

"""Independent actor physics, pipeline dependencies and scoped phase calibration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ....contracts import WorkloadStats
from ..core import (
    Estimate,
    Features,
    PhaseCalibration,
    apply_calibration,
    code_revision,
    evidence_of,
    fingerprint,
)
from ._actor.components import (
    AdamUpdate,
    Context,
    Efficiency,
    Operators,
    Transformer,
)
from ._actor.config import ActorConfig, Hardware, ModelShape
from ._actor.execution import mesh_is_valid
from ._actor.execution import pipeline_partition as _pipeline_partition
from ._actor.execution import pipeline_time as _pipeline_time
from ._actor.packing import DispatchConfig, PackingPlan, plan_workload
from .advantage import ADVANTAGE_BOUNDARY, AdvantageModel, AdvantageWorkload
from .aggregate_advantage import AggregateAdvantageModel
from .fsdp import FSDPHostRates, fsdp_work
from .packing_observations import ObservedPackingWork, PackingRecord, observed_plan

STAGES = ("recompute_logp", "compute_advantage", "train_step")


@dataclass(frozen=True)
class ObservedPacking:
    """A plan bound to one profile and workload, after rank de-duplication."""

    profile: str
    plan: PackingPlan
    source_ids: tuple[str, ...]
    workload_id: str

    def __post_init__(self) -> None:
        if not self.profile or not self.source_ids or not self.workload_id:
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
        p, m = config.parallel, self.model
        if not mesh_is_valid(config, m):
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
            partition = _pipeline_partition(config, m)
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
            Context(config, m, op, count, i == 0, i == p.pp - 1)
            for i, count in enumerate(partition)
        )
        transformer = Transformer()

        def run(ranks, training):
            forward = [[] for _ in contexts]
            backward = [[] for _ in contexts]
            for s, ctx in enumerate(contexts):
                for index in range(len(ranks[0])):
                    parts = []
                    for rank in ranks:
                        local = replace(ctx, microbatch_index=index, training=training)
                        f = transformer.forward(local, rank[index])
                        ftime = f["compute"] + f["communication_exposed"]
                        total = transformer.estimate(
                            local, rank[index], training=training
                        )
                        ttime = total["compute"] + total["communication_exposed"]
                        parts.append((ftime, max(0.0, ttime - ftime)))
                    forward[s].append(max(v[0] for v in parts))
                    backward[s].append(max(v[1] for v in parts))
            return _pipeline_time(forward, backward if training else None)

        train = 0.0
        for ranks in plan.train:
            train += run(ranks, True)
            updates = [AdamUpdate().estimate(ctx) for ctx in contexts]
            # One Adam update per PPO minibatch; no extra replication of workload.
            train += max(sum(u.values()) for u in updates)
        raw = {
            "recompute_logp": run(plan.logprob, False) if plan.logprob else 0.0,
            "train_step": train,
        }
        host_components = ()
        host_work = ()
        host_memory = ()
        if config.backend == "fsdp":
            host_work = fsdp_work(config, m, plan)
            rates = self.fsdp_host_rates or FSDPHostRates()
            host_components = tuple(
                (phase, rates.estimate(work)) for phase, work in host_work
            )
            host_memory = (
                ((m.total_parameters / p.dp * 10 / 1e9),)
                if config.offload_params
                else (0.0,)
            )
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
                else (
                    (dense + expert) * 2 + sharded * 16
                    if config.distributed_optimizer
                    else (dense + expert) * 18
                )
            )
            if config.precision_aware_optimizer:
                # BF16 parameters and reduction buffers stay local; Adam m/v
                # are DP-sharded. Master/remainder storage must be observed.
                owned = sharded if config.distributed_optimizer else dense + expert
                stored = (dense + expert) * (2 + config.grad_bytes) + owned * (
                    8 + (config.optimizer_master_storage_bytes or 0)
                )
            memory.append(stored / 1e9)
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
                ("largest_parameter_group", largest_group / 1e9),
                ("largest_bf16_logits", largest_logits / 1e9),
            )
            impossible = (
                impossible
                or max(largest_group, largest_logits) / 1e9 > self.hardware.memory_gb
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
                if result.seconds is not None and not isinstance(
                    self.advantage_model, AggregateAdvantageModel
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
            name in STAGES for name, _ in auxiliary_stages
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

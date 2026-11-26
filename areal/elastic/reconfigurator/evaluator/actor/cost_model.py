# SPDX-License-Identifier: Apache-2.0

"""Transformer training work, operator costs, communication and memory estimates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from typing import Protocol

from areal.elastic.reconfigurator.evaluator import (
    code_revision,
    fingerprint,
)
from areal.elastic.reconfigurator.evaluator.areal.controller import (
    DispatchConfig,
    Microbatch,
    PackingPlan,
    dispatch_workload,
    plan_workload,
)
from areal.elastic.reconfigurator.evaluator.contracts import (
    ActorConfig,
    Estimate,
    Features,
    Hardware,
    ModelShape,
    evidence_of,
    integer,
    nonnegative,
)
from areal.elastic.reconfigurator.evaluator.workload import WorkloadStats

from ..contracts import ParallelLayout, positive
from ..workload import ActorMemoryRejected
from ..workload import ExecutionConfig as CandidateConfig
from . import megatron


@dataclass(frozen=True)
class Efficiency:
    """Analytical priors, not fitted stage-level measurements.

    network and HBM rates are fractions of the explicit hardware peak. imbalance
    covers unknown MoE routing. Aggregate public calibration is separate.
    """

    dense: float = 0.45
    moe: float = 0.28
    attention: float = 0.3
    hbm: float = 0.65
    network: float = 0.65
    moe_imbalance: float = 1.15
    kernel_launch_us: float = 5.0
    ep_overlap: float = 0.25

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name.endswith("overlap"):
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"{item.name} must be between zero and one")
            else:
                positive(item.name, value)
                if (
                    item.name in {"dense", "moe", "attention", "hbm", "network"}
                    and value > 1
                ):
                    raise ValueError(f"{item.name} efficiency exceeds peak")


@dataclass(frozen=True)
class Operators:
    hardware: Hardware
    efficiency: Efficiency = field(default_factory=Efficiency)
    group_gbps: dict[str, float] = field(default_factory=dict)
    stage_world_size: int | None = None
    cp_rank_span: int | None = None
    expert_rank_span: int | None = None

    def __post_init__(self) -> None:
        for key, value in self.group_gbps.items():
            if key not in {"tp", "cp", "ep", "etp", "dense_dp", "expert_dp", "pp"}:
                raise ValueError(f"Unknown communication group {key}")
            positive(key, value)

    def roofline(self, flops: float, moved_bytes: float, kind: str) -> float:
        fraction = getattr(self.efficiency, kind)
        return max(
            flops / (self.hardware.bf16_tflops * 1e12 * fraction),
            self.memory(moved_bytes),
        )

    def memory(self, moved_bytes: float) -> float:
        return moved_bytes / (self.hardware.hbm_gbps * 1e9 * self.efficiency.hbm)

    def collective(
        self,
        payload_bytes: float,
        size: int,
        group: str,
        world: int,
        *,
        allreduce: bool = False,
    ) -> float:
        if size <= 1:
            return 0.0
        span = size
        if group.endswith("dp"):
            span = self.stage_world_size or world
        elif group == "pp":
            span = world
        elif group == "cp":
            span = self.cp_rank_span or size
        elif group == "ep":
            span = self.expert_rank_span or size
        remote = span > self.hardware.gpus_per_node
        peak = (
            self.hardware.inter_node_gbps if remote else self.hardware.intra_node_gbps
        )
        bandwidth = self.group_gbps.get(group, peak * self.efficiency.network)
        latency = (
            self.hardware.inter_latency_us if remote else self.hardware.intra_latency_us
        )
        factor = (size - 1) / size * (2 if allreduce else 1)
        return payload_bytes * factor / (bandwidth * 1000000000.0) + latency * 1e-06 * (
            size - 1
        )


def training_dense_work(model, layers, tokens, tp, cp, *, last_stage):
    """Forward linear FLOPs and local BF16 weight bytes for a training rank."""
    dense = layers * (
        model.attention_parameters_per_layer + model.router_parameters_per_layer
    )
    if model.mlp_type == "dense":
        dense += layers * model.expert_parameters_per_layer
    if last_stage:
        dense += model.head_parameters
    kv_replication = layers * (
        model.attention_parameters_per_rank(tp)
        - model.attention_parameters_per_layer / tp
    )
    local_dense = dense / tp + kv_replication
    if model.mlp_type == "moe":
        local_dense += layers * model.router_parameters_per_layer * (1 - 1 / tp)
    return 2 * (
        tokens / (tp * cp)
    ) * dense + 2 * tokens / cp * kv_replication, 2 * local_dense


@dataclass(frozen=True)
class Context:
    config: ActorConfig
    model: ModelShape
    operators: Operators
    layers: int
    first: bool
    last: bool
    training: bool = False
    microbatch_index: int = 0

    @property
    def final_dense_parameters(self) -> float:
        return (
            self.model.head_parameters / self.config.parallel.tp if self.last else 0.0
        )

    @property
    def owned_parameters(self) -> tuple[float, float]:
        backend = self.backend or backend_for(self.config.backend)
        return backend.owned_parameters(self)

    backend: object | None = None


@dataclass(frozen=True)
class Communications:
    """Megatron tensor, context, pipeline and expert communication volumes."""

    def pipeline_transfer(self, ctx: Context, mb: Microbatch) -> float:
        p = ctx.config.parallel
        return ctx.operators.collective(
            mb.compute_tokens / (p.tp * p.cp) * ctx.model.hidden_size * 4,
            2 if p.pp > 1 else 1,
            "pp",
            p.world_size,
        )

    def expert_exchange(self, ctx: Context, tokens: float) -> float:
        """Hierarchical EP with uniform routing and token deduplication per node.

        DeepEP/HybridEP can send one token per remote node, then fan out through
        NVLink. Charging top_k full copies over IB greatly overstates traffic.
        This topology/routing assumption is replaceable, not measured telemetry.
        """
        p, m, op = (ctx.config.parallel, ctx.model, ctx.operators)
        if p.ep == 1:
            return 0.0
        payload = tokens * m.hidden_size * 2
        if ctx.config.expert_dispatcher == "alltoall" or "ep" in op.group_gbps:
            exchange = op.collective(payload * m.top_k, p.ep, "ep", p.world_size)
        else:
            local = min(p.ep, max(1, op.hardware.gpus_per_node // p.etp))
            nodes = math.ceil(p.ep * p.etp / op.hardware.gpus_per_node)
            intra_copies = (local - 1) * (1 - (1 - 1 / p.ep) ** m.top_k)
            inter_copies = (nodes - 1) * (1 - (1 - 1 / nodes) ** m.top_k)
            intra = (
                payload
                * intra_copies
                / (op.hardware.intra_node_gbps * 1000000000.0 * op.efficiency.network)
            )
            inter = (
                payload
                * inter_copies
                / (op.hardware.inter_node_gbps * 1000000000.0 * op.efficiency.network)
            )
            exchange = intra + inter
            exchange += op.hardware.intra_latency_us * 1e-06 * (local - 1)
            exchange += op.hardware.inter_latency_us * 1e-06 * (nodes - 1)
        return 2 * ctx.layers * exchange

    def forward_cost(self, context, microbatch):
        return Transformer(self).forward(context, microbatch)

    def execution_cost(self, context, microbatch, *, training):
        return Transformer(self).estimate(context, microbatch, training=training)


@dataclass(frozen=True)
class Transformer:
    communications: Communications | None = None

    def forward(self, ctx: Context, mb: Microbatch) -> dict[str, float]:
        communications = (
            self.communications or ctx.backend or backend_for(ctx.config.backend)
        )
        m, p, op = (ctx.model, ctx.config.parallel, ctx.operators)
        t = mb.compute_tokens / (p.tp * p.cp)
        active_experts = m.active_expert_parameters_per_layer
        dense_flops, dense_bytes = training_dense_work(
            m, ctx.layers, mb.compute_tokens, p.tp, p.cp, last_stage=ctx.last
        )
        dense = op.roofline(dense_flops, dense_bytes, "dense")
        moe_params = ctx.layers * m.expert_parameters_per_layer / (p.ep * p.etp)
        moe = (
            0.0
            if m.mlp_type == "dense"
            else op.roofline(
                2 * t * ctx.layers * active_experts * op.efficiency.moe_imbalance,
                2 * moe_params,
                "moe",
            )
        )
        pair_flops = m.attention_flops_per_pair(ctx.layers)
        # Preserve the squared-length proxy for twice the causal pair count.
        attention = op.roofline(
            (pair_flops // 2) * mb.attention_length_squared / (p.tp * p.cp),
            8 * t * ctx.layers * m.attention_heads * m.head_dim,
            "attention",
        )
        pointwise = op.memory(t * ctx.layers * m.hidden_size * 2 * 24)
        if ctx.last:
            pointwise += op.memory(t * m.vocab_size * 4)
        launch = (
            ctx.layers
            * (6 if ctx.config.compile else 12)
            * op.efficiency.kernel_launch_us
            * 1e-06
        )
        compute = dense + moe + attention + pointwise + launch
        comm = communications.forward(ctx, mb)
        exposed = comm["tp"] + comm["cp"] + comm["etp"] + comm["pipeline_transfer"]
        ep_overlap = (
            op.efficiency.ep_overlap
            if ctx.config.expert_dispatcher == "hierarchical"
            else 0
        )
        exposed += (1 - ep_overlap) * comm["ep"]
        return {
            "compute": compute,
            "communication_exposed": exposed,
            "dense": dense,
            "moe": moe,
            "attention": attention,
            "pointwise_and_launch": pointwise + launch,
            **{f"raw_{k}": v for k, v in comm.items()},
        }

    def estimate(
        self, ctx: Context, mb: Microbatch, *, training: bool
    ) -> dict[str, float]:
        communications = (
            self.communications or ctx.backend or backend_for(ctx.config.backend)
        )
        ctx = replace(ctx, training=training)
        forward = self.forward(ctx, mb)
        if not training:
            return forward
        result = {k: v * (2 if k.startswith("raw_") else 3) for k, v in forward.items()}
        result["raw_pipeline_transfer"] = forward["raw_pipeline_transfer"] + (
            communications.pipeline_transfer(ctx, mb) if not ctx.first else 0.0
        )
        if ctx.config.gradient_checkpointing:
            layer_ctx = replace(ctx, first=False, last=False)
            recompute = self.forward(layer_ctx, mb)
            recompute["raw_pipeline_transfer"] = 0.0
            for key, value in recompute.items():
                result[key] += value * ctx.config.checkpoint_recompute_fraction
        result["raw_gradient_sync"] = (
            communications.gradient_sync(ctx)
            if communications.sync_each_microbatch
            else 0.0
        )
        efficiency = ctx.operators.efficiency
        ep_overlap = (
            efficiency.ep_overlap
            if ctx.config.expert_dispatcher == "hierarchical"
            else 0
        )
        result["communication_exposed"] = (
            result["raw_tp"]
            + result["raw_cp"]
            + result["raw_etp"]
            + result["raw_pipeline_transfer"]
        )
        result["communication_exposed"] += (1 - ep_overlap) * result["raw_ep"]
        return result


def partition_layers(config: ActorConfig, model: ModelShape) -> tuple[int, ...]:
    if config.pipeline_layers is not None:
        if sum(config.pipeline_layers) != model.layers:
            raise ValueError("pipeline_layers must sum to model.layers")
        return config.pipeline_layers
    if config.parallel.pp == 1:
        return (model.layers,)
    pp = config.parallel.pp
    # Mirror AReaL's non-interleaved stage partition: parameter-balanced
    # decoder layers, with embedding/loss storage on the first/last stage.
    per_layer = (
        model.attention_parameters_per_layer
        + 2 * model.hidden_size
        + model.router_parameters_per_layer
        + model.expert_parameters_per_layer / config.parallel.ep
    )
    layers = model.layers
    costs = [[math.inf] * (layers + 1) for _ in range(pp + 1)]
    splits = [[-1] * (layers + 1) for _ in range(pp + 1)]
    costs[0][0] = 0.0
    for stage in range(1, pp + 1):
        head = model.head_parameters if stage in (1, pp) else 0
        for end in range(layers + 1):
            for start in range(end + 1):
                if not math.isfinite(costs[stage - 1][start]):
                    continue
                if stage not in (1, pp) and start == end:
                    continue
                candidate = max(
                    costs[stage - 1][start], head + (end - start) * per_layer
                )
                if candidate < costs[stage][end]:
                    costs[stage][end], splits[stage][end] = candidate, start
    if not math.isfinite(costs[pp][layers]):
        raise ValueError("Pipeline stages exceed the model's layer/head partition")
    counts, end = [], layers
    for stage in range(pp, 0, -1):
        start = splits[stage][end]
        counts.append(end - start)
        end = start
    return tuple(reversed(counts))


class MegatronCosts(Communications):
    gather_backward = False
    sync_each_microbatch = False

    def forward(self, ctx: Context, mb: Microbatch) -> dict[str, float]:
        p, m, op = (ctx.config.parallel, ctx.model, ctx.operators)
        t = mb.compute_tokens
        tp = (
            (1 if m.mlp_type == "moe" else 2)
            * ctx.layers
            * op.collective(
                t / p.cp * m.hidden_size * 2, p.tp, "tp", p.world_size, allreduce=True
            )
        )
        ep = self.expert_exchange(ctx, t / (p.tp * p.cp))
        cp_payload = t / (p.cp * p.tp) * m.kv_heads * m.head_dim * 4
        cp = ctx.layers * op.collective(cp_payload * p.cp, p.cp, "cp", p.world_size)
        pp = self.pipeline_transfer(ctx, mb) if not ctx.last else 0.0
        etp = (
            2
            * ctx.layers
            * op.collective(
                t / (p.tp * p.cp) * m.top_k * p.etp * m.hidden_size * 2,
                p.etp,
                "etp",
                p.world_size,
            )
            if m.mlp_type == "moe"
            else 0.0
        )
        return {"tp": tp, "cp": cp, "ep": ep, "etp": etp, "pipeline_transfer": pp}

    def gradient_sync(self, ctx: Context) -> float:
        dense, expert = ctx.owned_parameters
        p, op, cfg = (ctx.config.parallel, ctx.operators, ctx.config)
        sharded = cfg.distributed_optimizer
        dense_sync = op.collective(
            dense * cfg.grad_bytes,
            p.dp * p.cp,
            "dense_dp",
            p.world_size,
            allreduce=not sharded,
        )
        if not expert:
            return dense_sync
        return dense_sync + op.collective(
            expert * cfg.grad_bytes,
            p.expert_dp,
            "expert_dp",
            p.world_size,
            allreduce=not sharded,
        )

    def parameter_gather(self, ctx: Context, *, backward: bool = False) -> float:
        dense, expert = ctx.owned_parameters
        head = ctx.final_dense_parameters
        p, op, cfg = (ctx.config.parallel, ctx.operators, ctx.config)
        if backward:
            body_needed = cfg.reshard_after_forward
            head_needed = cfg.reshard_head_after_forward
        else:
            body_needed = (
                ctx.training or cfg.reshard_after_forward or ctx.microbatch_index == 0
            )
            head_needed = (
                ctx.training
                or cfg.reshard_head_after_forward
                or ctx.microbatch_index == 0
            )
        result = 0.0
        if body_needed:
            result += op.collective(
                (dense - head) * 2, p.dp * p.cp, "dense_dp", p.world_size
            )
            result += op.collective(expert * 2, p.expert_dp, "expert_dp", p.world_size)
        if head_needed and head > 0:
            result += op.collective(head * 2, p.dp * p.cp, "dense_dp", p.world_size)
        return result

    def optimizer_update(self, ctx: Context) -> dict[str, float]:
        dense, expert = ctx.owned_parameters
        p, op, cfg = (ctx.config.parallel, ctx.operators, ctx.config)
        sharded = cfg.distributed_optimizer
        dense_dp, expert_dp = (p.dp * p.cp, p.expert_dp)
        owned = dense / (dense_dp if sharded else 1) + expert / (
            expert_dp if sharded else 1
        )
        traffic = 30
        if cfg.precision_aware_optimizer:
            master = cfg.optimizer_master_storage_bytes or 4
            traffic = 16 + 4 + cfg.main_grad_bytes + 2 * master
        optimizer = op.memory(owned * traffic)
        sync = self.gradient_sync(ctx)
        gather = 0.0
        if sharded:
            gather += op.collective(dense * 2, dense_dp, "dense_dp", p.world_size)
            if expert:
                gather += op.collective(
                    expert * 2, expert_dp, "expert_dp", p.world_size
                )
        return {
            "optimizer": optimizer,
            "gradient_sync_raw": sync,
            "parameter_gather": gather,
        }

    def owned_parameters(self, ctx) -> tuple[float, float]:
        """Full local dense/expert tensors, BEFORE DP sharding, AFTER TP/EP/PP."""
        m, p = (ctx.model, ctx.config.parallel)
        if m.mlp_type == "dense":
            dense = ctx.layers * (
                m.attention_parameters_per_rank(p.tp)
                + m.expert_parameters_per_layer / p.tp
                + 2 * m.hidden_size
                + 2 * m.head_dim
            )
            heads = (
                ctx.first + ctx.last - int(m.tied_embeddings and ctx.first and ctx.last)
            )
            dense += heads * m.head_parameters / p.tp
            dense += m.hidden_size if ctx.last else 0
            return (dense, 0.0)
        dense = ctx.layers * (
            m.attention_parameters_per_rank(p.tp)
            + m.router_parameters_per_layer
            + 2 * m.hidden_size
            + 2 * m.head_dim
        )
        dense += (ctx.first + ctx.last) * m.head_parameters / p.tp
        if m.tied_embeddings and ctx.first and ctx.last:
            dense -= m.head_parameters / p.tp
        dense += m.hidden_size if ctx.last else 0
        expert = ctx.layers * m.expert_parameters_per_layer / (p.ep * p.etp)
        return (dense, expert)

    @staticmethod
    def mesh_is_valid(config: ActorConfig, model: ModelShape) -> bool:
        p, m = (config.parallel, model)
        invalid = m.experts % p.ep or m.attention_heads % p.tp
        invalid = invalid or m.intermediate_size % (
            p.tp if m.mlp_type == "dense" else p.etp
        )
        invalid = invalid or m.vocab_size % p.tp
        if m.mlp_type == "dense":
            invalid = invalid or p.ep != 1 or p.etp != 1
        invalid = invalid or (m.kv_heads % p.tp != 0 and p.tp % m.kv_heads != 0)
        return not invalid

    pipeline_partition = staticmethod(partition_layers)


def backend_for(name):
    if name != "megatron":
        raise ValueError("Reference Actor cost model supports Megatron")

    return MegatronCosts()


def _loss_peak_bytes(config: ActorConfig, packed_tokens: int, vocab_size: int) -> int:
    """Capacity prior for the pipeline last stage's FP32 vocabulary tensors."""
    rows = math.ceil(packed_tokens / config.parallel.cp)
    row_bytes = 4 * math.ceil(vocab_size / config.parallel.tp)
    if config.enable_chunked_logits and config.lm_head_loss_chunk_size:
        # The fused LM head projects and backpropagates one token chunk at a time.
        return min(rows, config.lm_head_loss_chunk_size) * row_bytes
    logits = rows * row_bytes
    if config.enable_chunked_logits and not config.entropy_requires_grad:
        # Native Megatron's contiguous FP32 head output supports in-place reuse.
        return logits

    # Regular loss retains per-chunk normalized values for backward. Budget the
    # head output as live too; compiled autograd may shorten its lifetime.
    chunk_rows = min(rows, config.logprobs_chunk_size)
    chunk = chunk_rows * row_bytes
    forward = logits + 2 * chunk  # normalized chunks plus chunk-local temporaries
    last_chunk = ((rows - 1) % chunk_rows + 1) * row_bytes
    # At slice backward the processed chunk's graph releases its saved values.
    # The first full-size gradient becomes the accumulator. From the second
    # slice on, the accumulator overlaps another full-size incoming gradient.
    backward = logits + (logits - last_chunk)
    if rows > chunk_rows:
        backward = max(backward, 2 * logits + (logits - last_chunk - chunk))
    return logits + max(forward, backward)


@dataclass(frozen=True)
class Advantages:
    """Bandwidth + elementwise launch prior; not a fitted public GAE measurement."""

    def estimate(self, config: ActorConfig, tokens: float, op: Operators) -> float:
        return (
            op.memory(tokens / config.parallel.dp * 4 * 32)
            + 40 * op.efficiency.kernel_launch_us * 1e-06
        )


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


def stage_memory(config, model, workload, plan, contexts):
    """Estimate stored and peak GB per pipeline stage and the loss workspace."""
    p, m = config.parallel, model
    features = ()
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
        logits = (
            _loss_peak_bytes(config, maximum_tokens, m.vocab_size) if ctx.last else 0
        )
        if ctx.last:
            features += (
                ("largest_packed_microbatch_tokens", float(maximum_tokens)),
                ("loss_peak_bytes", float(logits)),
            )
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
    return memory, peaks, features


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

    @property
    def peak_gib(self):
        if not self.peak_memory_gb:
            raise ValueError("Megatron peak memory unavailable for this layout/batch")
        return max(self.peak_memory_gb) * 1e9 / 2**30

    def duration_seconds(self):
        if self.total.feasibility == "infeasible":
            raise ActorMemoryRejected(
                "Megatron candidate exceeds model/packing/memory constraints: "
                + "; ".join(self.total.notes)
            )
        return sum(estimate.require_seconds() for _, estimate in self.stages)

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

    @lru_cache(maxsize=512)
    def estimate_batch(self, config, batch):
        return self.estimate(config, WorkloadStats.from_batch(batch))

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
        memory, peaks, memory_features = stage_memory(
            config, m, workload, plan, contexts
        )
        features += memory_features
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

    estimate_logprob = staticmethod(megatron.estimate_logprob)
    estimate_train = staticmethod(megatron.estimate_train)

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


def actor_config(template, candidate):
    parallel = (
        ParallelLayout(*candidate.actor_parallel)
        if isinstance(candidate, CandidateConfig)
        else ParallelLayout(dp=candidate.actor_dp, tp=candidate.actor_tp)
    )
    values = dict(parallel=parallel)
    if isinstance(candidate, CandidateConfig):
        values.update(
            n_mbs=candidate.actor_n_mbs or template.n_mbs,
            max_tokens_per_mb=candidate.actor_max_tokens_per_mb
            or template.max_tokens_per_mb,
            pad_to_maximum=candidate.actor_pad_to_maximum,
        )
    return replace(template, **values)

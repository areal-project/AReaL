# SPDX-License-Identifier: Apache-2.0

"""Megatron Actor training cost, memory, capacity and weight publication models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace

from .contracts import ActorConfig, Hardware, ModelShape, ParallelLayout, positive
from .packing import DispatchConfig, Microbatch, plan_workload
from .rollout import memory_per_rank
from .workload import ActorMemoryRejected, ExecutionConfig, WorkloadStats


@dataclass(frozen=True)
class Efficiency:
    """Analytical priors (fractions of hardware peak); calibration is separate."""

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
        return payload_bytes * factor / (bandwidth * 1e9) + latency * 1e-6 * (size - 1)


def _training_dense_work(model, layers, tokens, tp, cp, *, last_stage):
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
    return (
        2 * (tokens / (tp * cp)) * dense + 2 * tokens / cp * kv_replication,
        2 * local_dense,
    )


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
    backend: MegatronCosts | None = None

    @property
    def owned_parameters(self) -> tuple[float, float]:
        return (self.backend or _MEGATRON).owned_parameters(self)


def partition_layers(config: ActorConfig, model: ModelShape) -> tuple[int, ...]:
    if config.pipeline_layers is not None:
        if sum(config.pipeline_layers) != model.layers:
            raise ValueError("pipeline_layers must sum to model.layers")
        return config.pipeline_layers
    if config.parallel.pp == 1:
        return (model.layers,)
    pp = config.parallel.pp
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


class MegatronCosts:
    """Megatron forward/backward compute, collective communication and optimizer costs."""

    def pipeline_transfer(self, ctx: Context, mb: Microbatch) -> float:
        p = ctx.config.parallel
        return ctx.operators.collective(
            mb.compute_tokens / (p.tp * p.cp) * ctx.model.hidden_size * 4,
            2 if p.pp > 1 else 1,
            "pp",
            p.world_size,
        )

    def expert_exchange(self, ctx: Context, tokens: float) -> float:
        p, m, op = ctx.config.parallel, ctx.model, ctx.operators
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
                / (op.hardware.intra_node_gbps * 1e9 * op.efficiency.network)
            )
            inter = (
                payload
                * inter_copies
                / (op.hardware.inter_node_gbps * 1e9 * op.efficiency.network)
            )
            exchange = (
                intra
                + inter
                + op.hardware.intra_latency_us * 1e-6 * (local - 1)
                + op.hardware.inter_latency_us * 1e-6 * (nodes - 1)
            )
        return 2 * ctx.layers * exchange

    def forward_comm(self, ctx: Context, mb: Microbatch) -> dict[str, float]:
        p, m, op = ctx.config.parallel, ctx.model, ctx.operators
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

    def forward_cost(self, ctx: Context, mb: Microbatch) -> dict[str, float]:
        m, p, op = ctx.model, ctx.config.parallel, ctx.operators
        t = mb.compute_tokens / (p.tp * p.cp)
        dense_flops, dense_bytes = _training_dense_work(
            m, ctx.layers, mb.compute_tokens, p.tp, p.cp, last_stage=ctx.last
        )
        dense = op.roofline(dense_flops, dense_bytes, "dense")
        moe_params = ctx.layers * m.expert_parameters_per_layer / (p.ep * p.etp)
        moe = (
            0.0
            if m.mlp_type == "dense"
            else op.roofline(
                2
                * t
                * ctx.layers
                * m.active_expert_parameters_per_layer
                * op.efficiency.moe_imbalance,
                2 * moe_params,
                "moe",
            )
        )
        pair_flops = m.attention_flops_per_pair(ctx.layers)
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
            * 1e-6
        )
        compute = dense + moe + attention + pointwise + launch
        comm = self.forward_comm(ctx, mb)
        ep_overlap = (
            op.efficiency.ep_overlap
            if ctx.config.expert_dispatcher == "hierarchical"
            else 0
        )
        exposed = (
            comm["tp"]
            + comm["cp"]
            + comm["etp"]
            + comm["pipeline_transfer"]
            + (1 - ep_overlap) * comm["ep"]
        )
        return {
            "compute": compute,
            "communication_exposed": exposed,
            **{f"raw_{k}": v for k, v in comm.items()},
        }

    def execution_cost(
        self, ctx: Context, mb: Microbatch, *, training: bool
    ) -> dict[str, float]:
        ctx = replace(ctx, training=training)
        forward = self.forward_cost(ctx, mb)
        if not training:
            return forward
        result = {k: v * (2 if k.startswith("raw_") else 3) for k, v in forward.items()}
        result["raw_pipeline_transfer"] = forward["raw_pipeline_transfer"] + (
            self.pipeline_transfer(ctx, mb) if not ctx.first else 0.0
        )
        if ctx.config.gradient_checkpointing:
            recompute = self.forward_cost(replace(ctx, first=False, last=False), mb)
            recompute["raw_pipeline_transfer"] = 0.0
            for key, value in recompute.items():
                result[key] += value * ctx.config.checkpoint_recompute_fraction
        ep_overlap = (
            ctx.operators.efficiency.ep_overlap
            if ctx.config.expert_dispatcher == "hierarchical"
            else 0
        )
        result["communication_exposed"] = (
            result["raw_tp"]
            + result["raw_cp"]
            + result["raw_etp"]
            + result["raw_pipeline_transfer"]
            + (1 - ep_overlap) * result["raw_ep"]
        )
        return result

    def gradient_sync(self, ctx: Context) -> float:
        dense, expert = ctx.owned_parameters
        p, op, cfg = ctx.config.parallel, ctx.operators, ctx.config
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

    def optimizer_update(self, ctx: Context) -> float:
        dense, expert = ctx.owned_parameters
        p, op, cfg = ctx.config.parallel, ctx.operators, ctx.config
        sharded = cfg.distributed_optimizer
        dense_dp, expert_dp = p.dp * p.cp, p.expert_dp
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
        return optimizer + sync + gather

    def owned_parameters(self, ctx: Context) -> tuple[float, float]:
        """Full local dense/expert tensors, before DP sharding, after TP/EP/PP."""
        m, p = ctx.model, ctx.config.parallel
        heads = ctx.first + ctx.last - int(m.tied_embeddings and ctx.first and ctx.last)
        extra_layer = (
            m.expert_parameters_per_layer / p.tp
            if m.mlp_type == "dense"
            else m.router_parameters_per_layer
        )
        dense = (
            ctx.layers
            * (
                m.attention_parameters_per_rank(p.tp)
                + extra_layer
                + 2 * m.hidden_size
                + 2 * m.head_dim
            )
            + heads * m.head_parameters / p.tp
            + (m.hidden_size if ctx.last else 0)
        )
        expert = (
            0.0
            if m.mlp_type == "dense"
            else ctx.layers * m.expert_parameters_per_layer / (p.ep * p.etp)
        )
        return dense, expert

    @staticmethod
    def mesh_is_valid(config: ActorConfig, model: ModelShape) -> bool:
        p, m = config.parallel, model
        invalid = (
            m.experts % p.ep
            or m.attention_heads % p.tp
            or m.intermediate_size % (p.tp if m.mlp_type == "dense" else p.etp)
            or m.vocab_size % p.tp
            or (m.mlp_type == "dense" and (p.ep != 1 or p.etp != 1))
            or (m.kv_heads % p.tp != 0 and p.tp % m.kv_heads != 0)
        )
        return not invalid


_MEGATRON = MegatronCosts()


def pipeline_time(
    forward: list[list[float]], backward: list[list[float]] | None
) -> float:
    """Non-interleaved 1F1B dependencies, or forward-only pipeline; seconds."""
    stages, microbatches = len(forward), len(forward[0])
    pending: dict[str, tuple[float, tuple[str, ...]]] = {}
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
            pending[name] = (seconds, tuple(deps))
            previous = name
    end: dict[str, float] = {}
    while pending:
        ready = [
            name for name, (_, deps) in pending.items() if all(d in end for d in deps)
        ]
        for name in ready:
            seconds, deps = pending.pop(name)
            end[name] = max((end[d] for d in deps), default=0.0) + seconds
    return max(end.values(), default=0.0)


def _transformer_seconds(
    ranks, contexts, backend: MegatronCosts, training: bool
) -> float:
    forward = [[] for _ in contexts]
    backward = [[] for _ in contexts]
    for s, ctx in enumerate(contexts):
        for index in range(len(ranks[0])):
            parts = []
            for rank in ranks:
                local = replace(ctx, microbatch_index=index, training=training)
                f = backend.forward_cost(local, rank[index])
                ftime = f["compute"] + f["communication_exposed"]
                total = backend.execution_cost(local, rank[index], training=training)
                ttime = total["compute"] + total["communication_exposed"]
                parts.append((ftime, max(0.0, ttime - ftime)))
            forward[s].append(max(v[0] for v in parts))
            backward[s].append(max(v[1] for v in parts))
    return pipeline_time(forward, backward if training else None)


def _loss_peak_bytes(config: ActorConfig, packed_tokens: int, vocab_size: int) -> int:
    rows = math.ceil(packed_tokens / config.parallel.cp)
    row_bytes = 4 * math.ceil(vocab_size / config.parallel.tp)
    if config.enable_chunked_logits and config.lm_head_loss_chunk_size:
        return min(rows, config.lm_head_loss_chunk_size) * row_bytes
    logits = rows * row_bytes
    if config.enable_chunked_logits and not config.entropy_requires_grad:
        return logits
    chunk_rows = min(rows, config.logprobs_chunk_size)
    chunk = chunk_rows * row_bytes
    forward = logits + 2 * chunk
    last_chunk = ((rows - 1) % chunk_rows + 1) * row_bytes
    backward = logits + (logits - last_chunk)
    if rows > chunk_rows:
        backward = max(backward, 2 * logits + (logits - last_chunk - chunk))
    return logits + max(forward, backward)


def _stage_peak_gb(config, model, workload, plan, contexts) -> tuple[float, ...]:
    p, m = config.parallel, model
    maximum_tokens = max(
        mb.compute_tokens for update in plan.train for rank in update for mb in rank
    )
    maximum_sequence = max(
        workload.sequence_lengths
        or (math.ceil(workload.sequence_tokens / workload.n_seqs),)
    )
    peaks = []
    for stage, ctx in enumerate(contexts):
        dense, expert = ctx.owned_parameters
        sharded = dense / (p.dp * p.cp) + expert / p.expert_dp
        if config.precision_aware_optimizer:
            owned = sharded if config.distributed_optimizer else dense + expert
            stored = (dense + expert) * (2 + config.grad_bytes) + owned * (
                8 + (config.optimizer_master_storage_bytes or 0)
            )
        else:
            stored = (
                (dense + expert) * (2 + config.grad_bytes) + sharded * 12
                if config.distributed_optimizer
                else (dense + expert) * 18
            )
        live = min(len(plan.train[0][0]), p.pp - stage)
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
        unknown_master = (
            4 * (sharded if config.distributed_optimizer else dense + expert)
            if config.precision_aware_optimizer
            and config.optimizer_master_storage_bytes is None
            else 0
        )
        peaks.append(
            (stored + unknown_master + live * saved + workspace + logits) / 1e9
        )
    return tuple(peaks)


@dataclass(frozen=True)
class ActorEstimate:
    duration: float | None
    peak_memory_gb: tuple[float, ...] = ()
    infeasible: bool = False
    reason: str = ""

    @property
    def peak_gib(self) -> float:
        if not self.peak_memory_gb:
            raise ValueError("Megatron peak memory unavailable for this layout/batch")
        return max(self.peak_memory_gb) * 1e9 / 2**30

    def duration_seconds(self) -> float:
        if self.infeasible or self.duration is None:
            raise ActorMemoryRejected(
                "Megatron candidate exceeds model/packing/memory constraints: "
                + self.reason
            )
        return self.duration


@dataclass(frozen=True)
class ExpertActorModel:
    model: ModelShape
    hardware: Hardware
    runtime_revision: str
    efficiency: Efficiency = field(default_factory=Efficiency)
    group_gbps: tuple[tuple[str, float], ...] = ()
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_gbps", tuple(tuple(x) for x in self.group_gbps))
        if not self.runtime_revision or len(dict(self.group_gbps)) != len(
            self.group_gbps
        ):
            raise ValueError(
                "runtime_revision and unique bandwidth groups are required"
            )
        Operators(self.hardware, self.efficiency, dict(self.group_gbps))

    def estimate_batch(self, config: ActorConfig, batch) -> ActorEstimate:
        return self.estimate(config, WorkloadStats.from_batch(batch))

    def estimate(self, config: ActorConfig, workload: WorkloadStats) -> ActorEstimate:
        p, m = config.parallel, self.model
        if not _MEGATRON.mesh_is_valid(config, m):
            return ActorEstimate(
                None, infeasible=True, reason="Model dimensions do not divide mesh"
            )
        try:
            plan = plan_workload(config, workload, dispatch=self.dispatch)
            partition = partition_layers(config, m)
        except ValueError as exc:
            return ActorEstimate(None, infeasible=True, reason=str(exc))
        op = Operators(
            self.hardware,
            self.efficiency,
            dict(self.group_gbps),
            p.world_size // p.pp,
            p.tp * p.cp,
            p.ep * p.etp,
        )
        contexts = tuple(
            Context(config, m, op, count, i == 0, i == p.pp - 1, backend=_MEGATRON)
            for i, count in enumerate(partition)
        )
        logprob = (
            _transformer_seconds(plan.logprob, contexts, _MEGATRON, False)
            if plan.logprob
            else 0.0
        )
        train = sum(
            _transformer_seconds(ranks, contexts, _MEGATRON, True)
            + max(_MEGATRON.optimizer_update(ctx) for ctx in contexts)
            for ranks in plan.train
        )
        advantage = (
            op.memory(workload.sequence_tokens / p.dp * 4 * 32)
            + 40 * op.efficiency.kernel_launch_us * 1e-6
        )
        peaks = _stage_peak_gb(config, m, workload, plan, contexts)
        impossible = max(peaks) > self.hardware.memory_gb
        return ActorEstimate(
            logprob + advantage + train,
            peaks,
            infeasible=impossible,
            reason="Peak memory exceeds GPU HBM" if impossible else "",
        )


def actor_config(template: ActorConfig, candidate: ExecutionConfig) -> ActorConfig:
    return replace(
        template,
        parallel=ParallelLayout(*candidate.actor_parallel),
        n_mbs=candidate.actor_n_mbs or template.n_mbs,
        max_tokens_per_mb=candidate.actor_max_tokens_per_mb
        or template.max_tokens_per_mb,
        pad_to_maximum=candidate.actor_pad_to_maximum,
    )


def estimate_actor_capacity(models, template, role, pattern, publication):
    """Return the capacity of a legal representative Actor batch in episodes/s."""
    execution = replace(template, actor_backend=role.backend)
    config = actor_config(models.actor_template, execution)
    model = models.actor_model
    batch = tuple(episode.completed_turns for episode in pattern.templates)
    try:
        estimate = model.estimate_batch(config, batch)
        duration = estimate.duration_seconds()
        if (
            not math.isfinite(duration)
            or duration <= 0
            or estimate.peak_gib > model.hardware.memory_gb * 1e9 / 2**30
        ):
            return None
        return len(batch) / (duration + publication)
    except ValueError:
        return None


def publication_seconds(
    model, hardware, parallel, replicas, *, weight_bytes, hbm_efficiency, bandwidth
) -> float:
    """Estimate the cost of publishing Actor weights to Rollout."""
    source = weight_bytes / parallel.pp * (1 - 1 / parallel.tp)
    if model.mlp_type == "moe":
        source += (
            model.expert_parameters_per_layer
            * model.layers
            * 2
            / parallel.pp
            * (1 - 1 / (parallel.ep * parallel.etp))
        )
    sent = max_endpoint = 0.0
    for config in replicas:
        weights, _ = memory_per_rank(model, config)
        sent += weights * config.world_size
        max_endpoint = max(max_endpoint, weights)
    copy_seconds = (source + max_endpoint) / (hardware.hbm_gbps * 1e9 * hbm_efficiency)
    return copy_seconds + sent / (bandwidth * 1e9)

# SPDX-License-Identifier: Apache-2.0

"""Shared hardware operators, tensor compute and communication costs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace

from areal.elastic.reconfigurator.reference.evaluator.cost_model.actor.packing import (
    Microbatch,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
    ActorConfig,
    Hardware,
    ModelShape,
    positive,
)

from . import backend_for


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
            flops / (self.hardware.bf16_tflops * 1000000000000.0 * fraction),
            moved_bytes / (self.hardware.hbm_gbps * 1000000000.0 * self.efficiency.hbm),
        )

    def memory(self, moved_bytes: float) -> float:
        return moved_bytes / (
            self.hardware.hbm_gbps * 1000000000.0 * self.efficiency.hbm
        )

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
        active_experts = m.expert_parameters_per_layer * m.top_k / m.experts
        dense_params = ctx.layers * (
            m.attention_parameters_per_layer + m.router_parameters_per_layer
        )
        if m.mlp_type == "dense":
            dense_params += ctx.layers * m.expert_parameters_per_layer
        if ctx.last:
            dense_params += m.head_parameters
        kv_replication = ctx.layers * (
            m.attention_parameters_per_rank(p.tp)
            - m.attention_parameters_per_layer / p.tp
        )
        local_dense = dense_params / p.tp + kv_replication
        if m.mlp_type == "moe":
            local_dense += ctx.layers * m.router_parameters_per_layer * (1 - 1 / p.tp)
        dense = op.roofline(
            2 * t * dense_params + 2 * mb.compute_tokens / p.cp * kv_replication,
            2 * local_dense,
            "dense",
        )
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
        attention = op.roofline(
            2
            * ctx.layers
            * m.attention_heads
            * m.head_dim
            * mb.attention_length_squared
            / (p.tp * p.cp),
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

# SPDX-License-Identifier: Apache-2.0
# Initial analytical primitives frozen from cfdbb716; owned independently here.

"""Replaceable hardware operators and backend communication components.

All rates are explicit modeling assumptions unless a calibration report says
otherwise. No component consumes expert routing, kernel traces or overlap logs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace

from .config import ActorConfig, Hardware, ModelShape, positive
from .packing import Microbatch


@dataclass(frozen=True)
class Efficiency:
    """Analytical priors, not fitted stage-level measurements.

    network and HBM rates are fractions of the explicit hardware peak. imbalance
    covers unknown MoE routing. Aggregate public calibration is separate.
    """

    dense: float = 0.45
    moe: float = 0.28
    attention: float = 0.30
    hbm: float = 0.65
    network: float = 0.65
    moe_imbalance: float = 1.15
    kernel_launch_us: float = 5.0
    fsdp_overlap: float = 0.50
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
    # Override a specific group's effective bandwidth when actual placement is known.
    # Keys: tp, cp, ep, dense_dp, expert_dp, pp. Values already include efficiency.
    group_gbps: dict[str, float] = field(default_factory=dict)
    stage_world_size: int | None = None
    cp_rank_span: int | None = None

    def __post_init__(self) -> None:
        for key, value in self.group_gbps.items():
            if key not in {"tp", "cp", "ep", "dense_dp", "expert_dp", "pp"}:
                raise ValueError(f"Unknown communication group {key}")
            positive(key, value)

    def roofline(self, flops: float, moved_bytes: float, kind: str) -> float:
        fraction = getattr(self.efficiency, kind)
        return max(
            flops / (self.hardware.bf16_tflops * 1e12 * fraction),
            moved_bytes / (self.hardware.hbm_gbps * 1e9 * self.efficiency.hbm),
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
        # Default backend meshes and contiguous node ranks. DP stays inside its
        # PP-stage block; e.g. PP4 on 4x8 GPUs has node-local dense/expert DP.
        span = size
        if group.endswith("dp"):
            span = self.stage_world_size or world
        elif group == "pp":
            span = world
        elif group == "cp":
            span = self.cp_rank_span or size
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
        """Full local dense/expert tensors, BEFORE DP sharding, AFTER TP/EP/PP."""
        m, p = self.model, self.config.parallel
        if m.mlp_type == "dense":
            # Linear weights are TP-sharded; Q/K and residual norms are replicated.
            dense = self.layers * (
                (m.attention_parameters_per_layer + m.expert_parameters_per_layer)
                / p.tp
                + 2 * m.hidden_size
                + 2 * m.head_dim
            )
            heads = (
                self.first
                + self.last
                - int(m.tied_embeddings and self.first and self.last)
            )
            dense += heads * m.head_parameters / p.tp
            dense += m.hidden_size if self.last else 0
            return dense, 0.0
        dense = (
            self.layers
            * (
                m.attention_parameters_per_layer
                + m.router_parameters_per_layer
                + 2 * m.hidden_size
                + 2 * m.head_dim
            )
            / p.tp
        )
        dense += ((self.first + self.last) * m.head_parameters) / p.tp
        if m.tied_embeddings and self.first and self.last:
            dense -= m.head_parameters / p.tp
        if self.config.backend == "fsdp" and self.last:
            dense += m.hidden_size  # Final root norm, outside decoder wrappers.
        expert = self.layers * m.expert_parameters_per_layer / p.ep
        return dense, expert


@dataclass(frozen=True)
class Communications:
    """Volume model: Megatron TP/CP/EP, or Archon Ulysses + FSDP2."""

    def forward(self, ctx: Context, mb: Microbatch) -> dict[str, float]:
        p, m, op = ctx.config.parallel, ctx.model, ctx.operators
        t = mb.compute_tokens
        tp = (
            (1 if ctx.config.backend == "megatron" and m.mlp_type == "moe" else 2)
            * ctx.layers
            * op.collective(
                t / p.cp * m.hidden_size * 2, p.tp, "tp", p.world_size, allreduce=True
            )
        )
        ep = self.expert_exchange(ctx, t / (p.tp * p.cp))
        if ctx.config.backend == "archon":
            # Ulysses Q/K/V redistribution and inverse output. KV heads may repeat.
            kv = max(m.kv_heads, p.tp * p.cp)
            cp_payload = (
                t / (p.cp * p.tp) * m.head_dim * (2 * m.attention_heads + 2 * kv) * 2
            )
            cp = ctx.layers * op.collective(cp_payload, p.cp, "cp", p.world_size)
        else:
            cp_payload = t / (p.cp * p.tp) * m.kv_heads * m.head_dim * 4
            cp = ctx.layers * op.collective(cp_payload * p.cp, p.cp, "cp", p.world_size)
        fsdp = 0.0
        if ctx.config.backend in {"archon", "fsdp"}:
            fsdp = self.parameter_gather(ctx)
        pp = self.pipeline_transfer(ctx, mb) if not ctx.last else 0.0
        return {"tp": tp, "cp": cp, "ep": ep, "fsdp": fsdp, "pipeline_transfer": pp}

    def pipeline_transfer(self, ctx: Context, mb: Microbatch) -> float:
        p = ctx.config.parallel
        # Size-2 collective volume is one BF16 activation/gradient send.
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
        p, m, op = ctx.config.parallel, ctx.model, ctx.operators
        if p.ep == 1:
            return 0.0
        payload = tokens * m.hidden_size * 2
        if ctx.config.expert_dispatcher == "alltoall" or "ep" in op.group_gbps:
            exchange = op.collective(payload * m.top_k, p.ep, "ep", p.world_size)
        else:
            local = min(p.ep, op.hardware.gpus_per_node)
            nodes = math.ceil(p.ep / op.hardware.gpus_per_node)
            # Expected distinct destinations under uniform independent top-k draws.
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
            exchange = intra + inter
            exchange += op.hardware.intra_latency_us * 1e-6 * (local - 1)
            exchange += op.hardware.inter_latency_us * 1e-6 * (nodes - 1)
        return 2 * ctx.layers * exchange  # Dispatch + combine.

    def gradient_sync(self, ctx: Context) -> float:
        dense, expert = ctx.owned_parameters
        p, op, cfg = ctx.config.parallel, ctx.operators, ctx.config
        if cfg.backend == "fsdp":
            groups = (
                ctx.layers
                + int(ctx.first and not ctx.model.tied_embeddings)
                + int(ctx.last)
            )
            return (
                op.collective(
                    (dense + expert) * cfg.grad_bytes, p.dp, "dense_dp", p.world_size
                )
                + max(groups - 1, 0) * op.collective(0, p.dp, "dense_dp", p.world_size)
                + op.memory(2 * (dense + expert) * cfg.grad_bytes)
            )
        sharded = cfg.distributed_optimizer or cfg.backend in {"archon", "fsdp"}
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
        p, op, cfg = ctx.config.parallel, ctx.operators, ctx.config
        if cfg.backend == "fsdp":
            if not cfg.reshard_after_forward or not cfg.fsdp_root_reshard_after_forward:
                raise ValueError(
                    "FSDP profile requires body/root reshard after forward"
                )
            groups = (
                ctx.layers
                + int(ctx.first and not ctx.model.tied_embeddings)
                + int(ctx.last)
            )
            return (
                op.collective((dense + expert) * 2, p.dp, "dense_dp", p.world_size)
                + max(groups - 1, 0) * op.collective(0, p.dp, "dense_dp", p.world_size)
                + op.memory(
                    4
                    * (dense + expert)
                    * (1 + (1 + int(not cfg.offload_params and p.dp > 1)) / p.dp)
                )
            )
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


@dataclass(frozen=True)
class Transformer:
    communications: Communications = field(default_factory=Communications)

    def forward(self, ctx: Context, mb: Microbatch) -> dict[str, float]:
        m, p, op = ctx.model, ctx.config.parallel, ctx.operators
        t = mb.compute_tokens / (p.tp * p.cp)
        active_experts = m.expert_parameters_per_layer * m.top_k / m.experts
        dense_params = ctx.layers * (
            m.attention_parameters_per_layer + m.router_parameters_per_layer
        )
        if m.mlp_type == "dense":
            dense_params += ctx.layers * m.expert_parameters_per_layer
        if ctx.last:
            dense_params += m.head_parameters
        dense = op.roofline(2 * t * dense_params, 2 * dense_params / p.tp, "dense")
        moe_params = ctx.layers * m.expert_parameters_per_layer / p.ep
        # Full expert weights may be touched even though FLOPs use top_k experts.
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
            # Logits/loss may be fused; 4 bytes is a conservative materialization prior.
            pointwise += op.memory(t * m.vocab_size * 4)
        launch = (
            ctx.layers
            * (6 if ctx.config.compile else 12)
            * op.efficiency.kernel_launch_us
            * 1e-6
        )
        compute = dense + moe + attention + pointwise + launch
        comm = self.communications.forward(ctx, mb)
        # Only EP and FSDP have modeled overlap. Do not subtract from raw work twice.
        exposed = comm["tp"] + comm["cp"] + comm["pipeline_transfer"]
        ep_overlap = (
            op.efficiency.ep_overlap
            if ctx.config.expert_dispatcher == "hierarchical"
            else 0
        )
        exposed += (1 - ep_overlap) * comm["ep"]
        exposed += max(comm["fsdp"] - op.efficiency.fsdp_overlap * compute, 0)
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
        ctx = replace(ctx, training=training)
        forward = self.forward(ctx, mb)
        if not training:
            return forward
        # Backward GEMMs ~2x forward, but backward TP/EP exchanges are ~1x
        # forward exchanges. Weight-gradient GEMMs do not dispatch tokens again.
        result = {k: v * (2 if k.startswith("raw_") else 3) for k, v in forward.items()}
        result["raw_pipeline_transfer"] = forward["raw_pipeline_transfer"] + (
            self.communications.pipeline_transfer(ctx, mb) if not ctx.first else 0.0
        )
        if ctx.config.gradient_checkpointing:
            layer_ctx = replace(ctx, first=False, last=False)
            recompute = self.forward(layer_ctx, mb)
            recompute["raw_pipeline_transfer"] = 0.0
            for key, value in recompute.items():
                result[key] += value * ctx.config.checkpoint_recompute_fraction
        # Checkpoint recompute reuses the gather required by backward.
        result["raw_fsdp"] = forward["raw_fsdp"] + (
            self.communications.parameter_gather(ctx, backward=True)
            if ctx.config.backend in {"archon", "fsdp"}
            else 0
        )
        # Current Archon SequentialRunner calls backward on EVERY microbatch
        # without disabling FSDP gradient sync. Megatron defers sync to the update.
        result["raw_gradient_sync"] = (
            self.communications.gradient_sync(ctx)
            if ctx.config.backend in {"archon", "fsdp"}
            else 0.0
        )
        efficiency = ctx.operators.efficiency
        ep_overlap = (
            efficiency.ep_overlap
            if ctx.config.expert_dispatcher == "hierarchical"
            else 0
        )
        result["communication_exposed"] = (
            result["raw_tp"] + result["raw_cp"] + result["raw_pipeline_transfer"]
        )
        result["communication_exposed"] += (1 - ep_overlap) * result["raw_ep"]
        result["communication_exposed"] += max(
            result["raw_fsdp"]
            + result["raw_gradient_sync"]
            - efficiency.fsdp_overlap * result["compute"],
            0,
        )
        return result


@dataclass(frozen=True)
class AdamUpdate:
    """Training update traffic, not a configuration optimizer."""

    def estimate(self, ctx: Context) -> dict[str, float]:
        dense, expert = ctx.owned_parameters
        p, op, cfg = ctx.config.parallel, ctx.operators, ctx.config
        if cfg.backend == "fsdp":
            from ..fsdp import optimizer_bytes_per_element

            # FSDP already reduces every microbatch. CPU Adam/clip belongs to
            # the host component; GPU Adam/Kahan uses its own eager traffic.
            owned = (dense + expert) / p.dp
            return {
                "optimizer": 0.0
                if cfg.offload_params
                else op.memory(owned * (optimizer_bytes_per_element(cfg) + 10)),
                "gradient_norm_scalar": op.collective(
                    4, p.dp, "dense_dp", p.world_size, allreduce=True
                ),
            }
        sharded = cfg.distributed_optimizer or cfg.backend == "archon"
        dense_dp, expert_dp = p.dp * p.cp, p.expert_dp
        owned = dense / (dense_dp if sharded else 1) + expert / (
            expert_dp if sharded else 1
        )
        traffic = 30
        if cfg.precision_aware_optimizer:
            # FP32 m/v read+write, BF16 parameter read+write, main grad read.
            # Unknown remainder storage uses a full-master traffic prior only.
            master = cfg.optimizer_master_storage_bytes or 4
            traffic = 16 + 4 + cfg.main_grad_bytes + 2 * master
        optimizer = op.memory(owned * traffic)
        # Gradient reduce-scatter + updated parameter gather for distributed Adam;
        # ordinary DDP instead performs a full gradient allreduce.
        sync = Communications().gradient_sync(ctx) if cfg.backend == "megatron" else 0.0
        gather = 0.0
        if sharded and cfg.backend == "megatron":
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


@dataclass(frozen=True)
class Advantages:
    """Bandwidth + elementwise launch prior; not a fitted public GAE measurement."""

    def estimate(self, config: ActorConfig, tokens: float, op: Operators) -> float:
        return (
            op.memory(tokens / config.parallel.dp * 4 * 32)
            + 40 * op.efficiency.kernel_launch_us * 1e-6
        )

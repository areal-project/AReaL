# SPDX-License-Identifier: Apache-2.0

"""Archon execution cost mechanisms."""

from __future__ import annotations

from ..contracts import ActorConfig, ModelShape
from .operators import Communications, Context
from .packing import Microbatch


class ArchonCosts(Communications):
    gather_backward = True
    sync_each_microbatch = True

    def forward(self, ctx: Context, mb: Microbatch) -> dict[str, float]:
        p, m, op = (ctx.config.parallel, ctx.model, ctx.operators)
        t = mb.compute_tokens
        tp = (
            2
            * ctx.layers
            * op.collective(
                t / p.cp * m.hidden_size * 2, p.tp, "tp", p.world_size, allreduce=True
            )
        )
        ep = self.expert_exchange(ctx, t / (p.tp * p.cp))
        kv = max(m.kv_heads, p.tp * p.cp)
        cp_payload = (
            t / (p.cp * p.tp) * m.head_dim * (2 * m.attention_heads + 2 * kv) * 2
        )
        cp = ctx.layers * op.collective(cp_payload, p.cp, "cp", p.world_size)
        fsdp = 0.0
        fsdp = self.parameter_gather(ctx)
        pp = self.pipeline_transfer(ctx, mb) if not ctx.last else 0.0
        return {"tp": tp, "cp": cp, "ep": ep, "fsdp": fsdp, "pipeline_transfer": pp}

    def gradient_sync(self, ctx: Context) -> float:
        dense, expert = ctx.owned_parameters
        p, op, cfg = (ctx.config.parallel, ctx.operators, ctx.config)
        sharded = True
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
        sharded = True
        dense_dp, expert_dp = (p.dp * p.cp, p.expert_dp)
        owned = dense / (dense_dp if sharded else 1) + expert / (
            expert_dp if sharded else 1
        )
        traffic = 30
        if cfg.precision_aware_optimizer:
            master = cfg.optimizer_master_storage_bytes or 4
            traffic = 16 + 4 + cfg.main_grad_bytes + 2 * master
        optimizer = op.memory(owned * traffic)
        sync = 0.0
        gather = 0.0
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
                (m.attention_parameters_per_layer + m.expert_parameters_per_layer)
                / p.tp
                + 2 * m.hidden_size
                + 2 * m.head_dim
            )
            heads = (
                ctx.first + ctx.last - int(m.tied_embeddings and ctx.first and ctx.last)
            )
            dense += heads * m.head_parameters / p.tp
            dense += m.hidden_size if ctx.last else 0
            return (dense, 0.0)
        dense = (
            ctx.layers
            * (
                m.attention_parameters_per_layer
                + m.router_parameters_per_layer
                + 2 * m.hidden_size
                + 2 * m.head_dim
            )
            / p.tp
        )
        dense += (ctx.first + ctx.last) * m.head_parameters / p.tp
        if m.tied_embeddings and ctx.first and ctx.last:
            dense -= m.head_parameters / p.tp
        expert = ctx.layers * m.expert_parameters_per_layer / p.ep
        return (dense, expert)

    @staticmethod
    def mesh_is_valid(config: ActorConfig, model: ModelShape) -> bool:
        p, m = (config.parallel, model)
        invalid = m.experts % p.ep or m.attention_heads % p.tp
        if m.mlp_type == "dense":
            invalid = invalid or p.ep != 1 or m.intermediate_size % p.tp
        invalid = invalid or m.kv_heads % p.tp or m.attention_heads // p.tp % p.cp
        kv = m.kv_heads // p.tp
        if not invalid:
            invalid = (
                kv % p.cp
                if kv >= p.cp
                else p.cp % kv or m.attention_heads // m.kv_heads % (p.cp // kv)
            )
        return not invalid

    @staticmethod
    def pipeline_partition(config: ActorConfig, model: ModelShape) -> tuple[int, ...]:
        if config.pipeline_layers is not None:
            if sum(config.pipeline_layers) != model.layers:
                raise ValueError("pipeline_layers must sum to model.layers")
            return config.pipeline_layers
        if config.parallel.pp == 1:
            return (model.layers,)
        pp = config.parallel.pp
        first, last = (
            config.pipeline_first_less_layers,
            config.pipeline_last_less_layers,
        )
        count, extra = divmod(model.layers + first + last, pp)
        if count < max(first, last, 1):
            raise ValueError("Archon PP stage overhead exceeds its effective layers")
        return tuple(
            count + (i < extra) - (first if i == 0 else last if i == pp - 1 else 0)
            for i in range(pp)
        )

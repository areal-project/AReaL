# SPDX-License-Identifier: Apache-2.0

"""Transformer inference work, iteration latency, KV memory and communications."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal

from areal.elastic.reconfigurator.evaluator import (
    code_revision,
    fingerprint,
)
from areal.elastic.reconfigurator.evaluator.contracts import (
    Estimate,
    Hardware,
    InferenceRates,
    ModelShape,
    RolloutConfig,
    ServiceEstimate,
    ServiceSlice,
    ServiceState,
)

from ..workload import ExecutionConfig as CandidateConfig
from .sglang import SGLangScheduler


def compute_seconds(hardware, flops, efficiency):
    return flops / (hardware.bf16_tflops * 1e12 * efficiency)


def memory_seconds(hardware, moved_bytes, efficiency):
    return moved_bytes / (hardware.hbm_gbps * 1e9 * efficiency)


def inference_linear_flops(model, tokens, emitted, tp):
    dense = model.attention_parameters_per_layer + model.router_parameters_per_layer
    return (
        2 * tokens * model.layers * (dense + model.active_expert_parameters_per_layer)
        + 2 * emitted * model.head_parameters
    ) / tp


def kv_cache_bytes_per_token(
    model, layers: int, local_kv_heads: float, element_bytes: int
) -> float:
    """Count logical bytes for both key and value vectors of one token."""
    return 2 * layers * local_kv_heads * model.head_dim * element_bytes


def iteration_cost(
    model: ModelShape,
    hardware: Hardware,
    rates: InferenceRates,
    config: RolloutConfig,
    slices: tuple[ServiceSlice, ...],
    kv_after: int,
    *,
    _decode_lines: list[tuple[float, float]] | None = None,
) -> tuple[float, float]:
    if (
        config.pp != 1
        or config.dp_size != 1
        or config.ep_size != 1
        or config.tp > model.kv_heads
    ):
        return _distributed_iteration_cost(
            model, hardware, rates, config, slices, kv_after, _decode_lines
        )
    m, h, r, tp = (model, hardware, rates, config.tp)
    tokens = sum(s.query_tokens for s in slices)
    pairs = sum(s.attention_pairs for s in slices)
    pair_flops = m.attention_flops_per_pair(m.layers)
    dense = m.attention_parameters_per_layer + m.router_parameters_per_layer
    emitted = sum(s.emit_token for s in slices)
    linear_flops = inference_linear_flops(m, tokens, emitted, tp)
    attention_flops = pair_flops * pairs / tp
    compute = compute_seconds(h, linear_flops, r.dense) + compute_seconds(
        h, attention_flops, r.attention
    )
    touched = 1 - (1 - m.top_k / m.experts) ** tokens
    weights = (
        (
            m.layers * (dense + touched * m.expert_parameters_per_layer)
            + m.head_parameters
        )
        * config.weight_bytes
        / tp
    )
    local_kv_heads = max(1.0, m.kv_heads / tp)
    kv_width = kv_cache_bytes_per_token(m, m.layers, local_kv_heads, config.kv_bytes)
    kv_traffic = (sum(s.context_tokens for s in slices) + tokens) * kv_width
    activations = tokens * m.layers * m.hidden_size * config.weight_bytes * 24 / tp
    memory = memory_seconds(h, weights + kv_traffic + activations, r.hbm)
    network = 0.0
    if tp > 1:
        remote = tp > h.gpus_per_node
        bandwidth = (
            (h.inter_node_gbps if remote else h.intra_node_gbps)
            * 1000000000.0
            * r.network
        )
        latency = (h.inter_latency_us if remote else h.intra_latency_us) * 1e-06
        payload = tokens * m.hidden_size * config.weight_bytes
        network = (
            m.layers
            * r.collectives_per_layer
            * (2 * (tp - 1) / tp * payload / bandwidth + (tp - 1) * latency)
        )
    seconds = max(compute, memory) + network + m.layers * r.launch_seconds
    if _decode_lines is not None:
        fixed = network + m.layers * r.launch_seconds
        _decode_lines.extend(
            (
                (
                    compute + fixed,
                    pair_flops
                    * len(slices)
                    / tp
                    / (h.bf16_tflops * 1e12 * r.attention),
                ),
                (
                    memory + fixed,
                    len(slices) * kv_width / (h.hbm_gbps * 1e9 * r.hbm),
                ),
            )
        )
    lower_bound = m.total_parameters * config.weight_bytes / tp + kv_after * kv_width
    return (seconds, lower_bound)


def supports_layout(model: ModelShape, config: RolloutConfig) -> bool:
    """KV heads may be sharded across attention ranks or replicated onto them."""
    tp = config.attention_tp
    return (
        model.attention_heads % tp == 0
        and (model.kv_heads % tp == 0 or tp % model.kv_heads == 0)
        and model.experts % config.ep_size == 0
        and config.pp <= model.layers
        and (model.mlp_type != "dense" or config.ep_size == 1)
    )


def memory_per_rank(model, config):
    """Analytical maximum-stage weight bytes and KV bytes per token per DP group."""
    layers = math.ceil(model.layers / config.pp)
    attention = model.attention_parameters_per_rank(config.attention_tp)
    expert = model.expert_parameters_per_layer / config.tp
    replicated = (
        model.router_parameters_per_layer + 2 * model.hidden_size + 2 * model.head_dim
    )
    heads = 1 if config.pp > 1 or model.tied_embeddings else 2
    weights = (
        layers * (attention + expert + replicated)
        + heads * model.head_parameters / config.tp
    ) * config.weight_bytes
    kv = kv_cache_bytes_per_token(
        model,
        layers,
        max(1, model.kv_heads / config.attention_tp),
        config.kv_bytes,
    )
    return weights, kv


def _distributed_iteration_cost(
    model, hardware, rates, config, slices, kv_after, decode_lines=None
):
    """SGLang PP/DP-Attention/EP roofline prior with explicit grouping costs."""

    m, h, r, c = model, hardware, rates, config
    groups = [[] for _ in range(c.dp_size)]
    for item in slices:
        groups[SGLangScheduler.attention_group(item.request_id, c.dp_size)].append(item)
    group_tokens = [sum(s.query_tokens for s in row) for row in groups]
    actual_tokens = sum(group_tokens)
    # Prefill uses the sum-length DP gather; decode chooses padded/summed
    # communication. Use the larger uniform-work prior for dense execution.
    tokens = (
        actual_tokens
        if any(s.kind == "prefill" for s in slices)
        else max(group_tokens) * c.dp_size
    )
    pairs = max(sum(s.attention_pairs for s in row) for row in groups)
    emitted = sum(s.emit_token for s in slices)
    active = m.active_expert_parameters_per_layer
    linear = (
        2
        * m.layers
        * (
            tokens * (active + m.router_parameters_per_layer) / c.tp
            + max(group_tokens) * m.attention_parameters_per_rank(c.attention_tp)
        )
        + 2 * emitted * m.head_parameters / c.tp
    )
    pair_flops = m.attention_flops_per_pair(m.layers)
    attention = pair_flops * pairs / c.attention_tp
    compute = compute_seconds(h, linear, r.dense) + compute_seconds(
        h, attention, r.attention
    )
    weights, kv_width = memory_per_rank(m, c)
    microbatches = min(
        c.pp,
        max(
            1,
            math.ceil(len(slices) / max(1, c.max_batch_sequences // c.dp_size // c.pp)),
        ),
    )
    memory = (
        weights * microbatches * c.pp
        + (sum(s.context_tokens for s in slices) + actual_tokens)
        * kv_width
        * c.pp
        / c.dp_size
        + tokens * m.layers * m.hidden_size * c.weight_bytes * 24 / c.tp
    )
    memory = memory_seconds(h, memory, r.hbm)

    def collective(payload, size, span, *, allreduce=True):
        if size == 1:
            return 0.0
        remote = span > h.gpus_per_node
        bandwidth = (
            (h.inter_node_gbps if remote else h.intra_node_gbps) * 1e9 * r.network
        )
        latency = (h.inter_latency_us if remote else h.intra_latency_us) * 1e-6
        return (2 if allreduce else 1) * (size - 1) / size * payload / bandwidth + (
            size - 1
        ) * latency

    payload = tokens * m.hidden_size * c.weight_bytes
    network = m.layers * collective(payload / c.dp_size, c.attention_tp, c.attention_tp)
    if m.mlp_type == "dense":
        network += m.layers * collective(payload, c.tp, c.tp)
    else:
        # EP dispatch/combine; remaining TP ranks shard each expert's matrix.
        network += (
            2
            * m.layers
            * collective(payload * m.top_k / c.tp, c.ep_size, c.tp, allreduce=False)
        )
        network += m.layers * collective(
            payload * m.top_k / c.ep_size, c.tp // c.ep_size, c.tp
        )
    if c.dp_size > 1:
        network += 2 * m.layers * collective(payload, c.dp_size, c.tp, allreduce=False)
    pipeline = 0.0
    if c.pp > 1:
        remote = c.tp * c.pp > h.gpus_per_node
        bandwidth = (
            (h.inter_node_gbps if remote else h.intra_node_gbps) * 1e9 * r.network
        )
        latency = (h.inter_latency_us if remote else h.intra_latency_us) * 1e-6
        pipeline = (c.pp - 1) * (payload / c.tp / bandwidth + microbatches * latency)
    # Forward pipeline wave: fill/drain plus the occupied microbatch slots.
    bubble = (c.pp + microbatches - 1) / (c.pp * microbatches)
    seconds = (
        max(compute, memory) + network + m.layers * r.launch_seconds * microbatches
    ) * bubble + pipeline
    if decode_lines is not None:
        fixed = (
            network + m.layers * r.launch_seconds * microbatches
        ) * bubble + pipeline
        for row in groups:
            group_pairs = sum(s.context_tokens + 1 for s in row)
            group_attention = pair_flops * group_pairs / c.attention_tp
            group_compute = compute_seconds(h, linear, r.dense) + compute_seconds(
                h, group_attention, r.attention
            )
            decode_lines.append(
                (
                    group_compute * bubble + fixed,
                    pair_flops
                    * len(row)
                    / c.attention_tp
                    / (h.bf16_tflops * 1e12 * r.attention)
                    * bubble,
                )
            )
        decode_lines.append(
            (
                memory * bubble + fixed,
                len(slices)
                * kv_width
                * c.pp
                / c.dp_size
                / (h.hbm_gbps * 1e9 * r.hbm)
                * bubble,
            )
        )
    return seconds, weights + kv_after / c.dp_size * kv_width


@dataclass(frozen=True)
class ExpertRolloutModel:
    model: ModelShape
    hardware: Hardware
    rates: InferenceRates
    runtime_revision: str
    parameter_sources: tuple[str, ...] = ()
    version: str = "expert-rollout-v1"
    calibration_scope: Literal["deployment", "replica"] = "deployment"
    separate_phases: bool = False
    timing_boundary: str = ""
    execution_identity: str = ""

    def __post_init__(self) -> None:
        if not self.version or not self.runtime_revision:
            raise ValueError("Model version is required")
        if (
            self.calibration_scope not in {"deployment", "replica"}
            or type(self.separate_phases) is not bool
        ):
            raise ValueError("Unknown service calibration identity")

    @lru_cache(maxsize=128)
    def profile_key(self, config: RolloutConfig) -> str:
        if self.calibration_scope == "replica":
            config = replace(config, replicas=1)
        identity = (
            code_revision(),
            self.version,
            self.runtime_revision,
            self.model,
            self.hardware,
            self.rates,
            config,
        )
        if self.calibration_scope != "deployment" or self.separate_phases:
            identity += (
                (self.calibration_scope, self.separate_phases, self.timing_boundary),
            )
        if self.execution_identity:
            identity += (self.execution_identity,)
        return fingerprint(*identity)

    def decode_interval(self, config, kv_tokens, slices, max_steps):
        """First step duration, per-step growth and length of one stable decode interval."""
        first = self.estimate(
            config, ServiceState(kv_tokens), slices
        ).duration.require_seconds()
        if max_steps == 1:
            return first, 0.0, 1

        # Pure decode with unchanged members: each compute/memory branch is
        # affine until a different branch wins or the next step exceeds capacity.
        width = len(slices)
        kv_after = kv_tokens + width
        lines = []
        args = self.model, self.hardware, self.rates, config, slices
        _, lower = iteration_cost(*args, kv_after, _decode_lines=lines)
        _, next_lower = iteration_cost(*args, kv_after + width)
        room = self.hardware.memory_gb * 1e9 * config.mem_fraction_static - lower
        steps = min(max_steps, (config.kv_capacity_tokens - kv_tokens) // width)
        if next_lower > lower:
            steps = min(steps, 1 + math.floor(max(0.0, room) / (next_lower - lower)))

        base, slope = max(lines)
        for other_base, other_slope in lines:
            if other_slope > slope:
                steps = min(
                    steps,
                    max(1, math.ceil((base - other_base) / (other_slope - slope))),
                )
        return first, slope, steps

    def estimate(
        self,
        config: RolloutConfig,
        state: ServiceState,
        ready_requests: tuple[ServiceSlice, ...],
    ) -> ServiceEstimate:
        slices = tuple(ready_requests)
        if not slices or len({s.request_id for s in slices}) != len(slices):
            raise ValueError("A service iteration needs distinct requests")
        profile = self.profile_key(config)
        tokens = sum(s.query_tokens for s in slices)
        prefill = sum(s.query_tokens for s in slices if s.kind == "prefill")
        decode = sum(s.kind == "decode" for s in slices)
        pairs = sum(s.attention_pairs for s in slices)
        kv_after = state.kv_tokens + tokens
        features = (
            ("prefill_tokens", float(prefill)),
            ("decode_sequences", float(decode)),
            ("attention_pairs", pairs),
            ("kv_tokens", float(state.kv_tokens)),
        )
        if self.separate_phases:
            features += (
                ("batch_sequences", float(len(slices))),
                ("query_tokens", float(tokens)),
                ("max_context_tokens", float(max(s.context_tokens for s in slices))),
                ("max_query_tokens", float(max(s.query_tokens for s in slices))),
                (
                    "kv_bytes_per_token_per_rank",
                    float(
                        kv_cache_bytes_per_token(
                            self.model,
                            self.model.layers,
                            max(1, self.model.kv_heads // config.tp),
                            config.kv_bytes,
                        )
                    ),
                ),
            )
        token_overflow = tokens > config.max_batch_tokens
        if config.prefill_chunk_tokens is not None:
            per_group = [0] * config.dp_size
            for item in slices:
                if item.kind == "prefill":
                    group = SGLangScheduler.attention_group(
                        item.request_id, config.dp_size
                    )
                    per_group[group] += item.query_tokens
            token_overflow = max(per_group) > min(
                config.max_batch_tokens, config.prefill_chunk_tokens // config.dp_size
            )
        if (
            len(slices) > config.max_batch_sequences
            or token_overflow
            or kv_after > config.kv_capacity_tokens
        ):
            return ServiceEstimate(
                profile,
                Estimate(
                    None,
                    "unsupported",
                    "infeasible",
                    ("Service batch exceeds sequence/token/KV limit",),
                ),
                features,
                kv_after,
            )
        if not supports_layout(self.model, config):
            return ServiceEstimate(
                profile,
                Estimate(
                    None,
                    "unsupported",
                    "infeasible",
                    ("Model dimensions do not divide the SGLang layout",),
                ),
                features,
                kv_after,
            )
        seconds, lower_bound = iteration_cost(
            self.model, self.hardware, self.rates, config, slices, kv_after
        )
        estimate = Estimate(
            seconds,
            "assumed",
            source_ids=self.parameter_sources,
            feasibility="infeasible"
            if lower_bound
            > self.hardware.memory_gb * 1000000000.0 * config.mem_fraction_static
            else "unknown",
            notes=(
                "Dense mixed-batch roofline prior; target service traces required"
                if self.model.mlp_type == "dense"
                else "Uniform MoE routing and mixed-batch roofline prior; target service traces required",
                "Queue, external tools and admission latency are excluded",
                "Weight plus KV memory is a lower bound; temporary buffers are not certified",
            ),
        )
        return ServiceEstimate(profile, estimate, features, kv_after)


def rollout_config(template, rollout_model, candidate):
    values = dict(replicas=candidate.rollout_replicas, tp=candidate.rollout_tp)
    if isinstance(candidate, CandidateConfig):
        values.update(
            pp=candidate.rollout_pp,
            dp_size=candidate.sglang_dp_size,
            ep_size=1,  # Expert-parallel search awaits reconfiguration validation.
            max_batch_sequences=candidate.sglang_max_running_requests
            or template.max_batch_sequences,
            max_batch_tokens=candidate.sglang_max_prefill_tokens,
            mem_fraction_static=candidate.sglang_mem_fraction_static
            if candidate.sglang_mem_fraction_static is not None
            else template.mem_fraction_static,
        )
    config = replace(template, **values)
    model, hardware = rollout_model.model, rollout_model.hardware
    if not supports_layout(model, config):
        raise ValueError("Model dimensions do not divide the SGLang layout")
    weights, kv_bytes = memory_per_rank(model, config)
    available = hardware.memory_gb * 1e9 * config.mem_fraction_static - weights
    capacity = math.floor(available / kv_bytes) * config.dp_size
    if capacity <= 0:
        raise ValueError("SGLang weights exhaust the configured static memory budget")
    return replace(config, kv_capacity_tokens=capacity)

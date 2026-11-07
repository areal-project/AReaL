# SPDX-License-Identifier: Apache-2.0

"""Mixed prefill/decode roofline prior, separate from request scheduling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._actor.config import Hardware, ModelShape
from .types import ServiceSlice

if TYPE_CHECKING:
    from ..rollout import InferenceRates, RolloutConfig


def iteration_cost(
    model: ModelShape,
    hardware: Hardware,
    rates: InferenceRates,
    config: RolloutConfig,
    slices: tuple[ServiceSlice, ...],
    kv_after: int,
) -> tuple[float, float]:
    m, h, r, tp = model, hardware, rates, config.tp
    tokens = sum(s.query_tokens for s in slices)
    pairs = sum(
        s.query_tokens * s.context_tokens + s.query_tokens * (s.query_tokens + 1) / 2
        for s in slices
    )
    active_expert = m.expert_parameters_per_layer * m.top_k / m.experts
    dense = m.attention_parameters_per_layer + m.router_parameters_per_layer
    emitted = sum(s.emit_token for s in slices)
    linear_flops = (
        2 * tokens * m.layers * (dense + active_expert)
        + 2 * emitted * m.head_parameters
    ) / tp
    attention_flops = 4 * m.layers * m.attention_heads * m.head_dim * pairs / tp
    compute = linear_flops / (h.bf16_tflops * 1e12 * r.dense) + attention_flops / (
        h.bf16_tflops * 1e12 * r.attention
    )
    # Expected touched experts under uniform routing, not routing telemetry.
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
    kv_width = 2 * m.layers * local_kv_heads * m.head_dim * config.kv_bytes
    kv_traffic = (sum(s.context_tokens for s in slices) + tokens) * kv_width
    activations = tokens * m.layers * m.hidden_size * config.weight_bytes * 24 / tp
    memory = (weights + kv_traffic + activations) / (h.hbm_gbps * 1e9 * r.hbm)
    network = 0.0
    if tp > 1:
        remote = tp > h.gpus_per_node
        bandwidth = (
            (h.inter_node_gbps if remote else h.intra_node_gbps) * 1e9 * r.network
        )
        latency = (h.inter_latency_us if remote else h.intra_latency_us) * 1e-6
        payload = tokens * m.hidden_size * config.weight_bytes
        network = (
            m.layers
            * r.collectives_per_layer
            * (2 * (tp - 1) / tp * payload / bandwidth + (tp - 1) * latency)
        )
    seconds = max(compute, memory) + network + m.layers * r.launch_seconds
    lower_bound = m.total_parameters * config.weight_bytes / tp + kv_after * kv_width
    return seconds, lower_bound

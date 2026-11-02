# SPDX-License-Identifier: Apache-2.0

"""Replica-level prefill/decode costs for a selected inference batch."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

from areal.elastic.contracts import OnlineLayout
from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
    Estimate,
    Hardware,
    ModelShape,
    PhaseCalibration,
    ServiceSlice,
    ServiceState,
    apply_calibration,
)
from areal.elastic.reconfigurator.reference.identity import code_revision, fingerprint

from ..contracts import ServiceEstimate

if TYPE_CHECKING:
    from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import (
        InferenceRates,
        RolloutConfig,
    )


def iteration_cost(
    model: ModelShape,
    hardware: Hardware,
    rates: InferenceRates,
    config: RolloutConfig,
    slices: tuple[ServiceSlice, ...],
    kv_after: int,
) -> tuple[float, float]:
    m, h, r, tp = (model, hardware, rates, config.tp)
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
    compute = linear_flops / (
        h.bf16_tflops * 1000000000000.0 * r.dense
    ) + attention_flops / (h.bf16_tflops * 1000000000000.0 * r.attention)
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
    memory = (weights + kv_traffic + activations) / (h.hbm_gbps * 1000000000.0 * r.hbm)
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
    lower_bound = m.total_parameters * config.weight_bytes / tp + kv_after * kv_width
    return (seconds, lower_bound)


@dataclass(frozen=True)
class ExpertRolloutModel:
    model: ModelShape
    hardware: Hardware
    rates: InferenceRates
    runtime_revision: str
    calibration: tuple[PhaseCalibration, ...] = ()
    version: str = "expert-rollout-v1"
    calibration_scope: Literal["deployment", "replica"] = "deployment"
    separate_phases: bool = False
    timing_boundary: str = ""
    execution_identity: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "calibration", tuple(self.calibration))
        if not self.version or not self.runtime_revision:
            raise ValueError("Model version is required")
        if (
            self.calibration_scope not in {"deployment", "replica"}
            or type(self.separate_phases) is not bool
        ):
            raise ValueError("Unknown service calibration identity")

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
        pairs = sum(
            s.query_tokens * s.context_tokens
            + s.query_tokens * (s.query_tokens + 1) / 2
            for s in slices
        )
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
                        2
                        * self.model.layers
                        * max(1, self.model.kv_heads // config.tp)
                        * self.model.head_dim
                        * config.kv_bytes
                    ),
                ),
            )
        if (
            len(slices) > config.max_batch_sequences
            or tokens > config.max_batch_tokens
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
        m, tp = (self.model, config.tp)
        if m.attention_heads % tp or (m.mlp_type == "dense" and m.kv_heads % tp):
            return ServiceEstimate(
                profile,
                Estimate(
                    None,
                    "unsupported",
                    "infeasible",
                    (
                        "TP must divide query heads; dense adapter also requires KV heads divisible by TP",
                    ),
                ),
                features,
                kv_after,
            )
        seconds, lower_bound = iteration_cost(
            self.model, self.hardware, self.rates, config, slices, kv_after
        )
        phase = "mixed" if prefill and decode else "prefill" if prefill else "decode"
        estimate = apply_calibration(
            seconds,
            profile,
            phase if self.separate_phases else "service",
            features,
            self.calibration,
            timing_boundary=self.timing_boundary,
        )
        estimate = replace(
            estimate,
            feasibility="infeasible"
            if lower_bound > self.hardware.memory_gb * 1000000000.0
            else "unknown",
            notes=estimate.notes
            + (
                "Dense mixed-batch roofline prior; target service traces required"
                if m.mlp_type == "dense"
                else "Uniform MoE routing and mixed-batch roofline prior; target service traces required",
                "Queue, external tools and admission latency are excluded",
                "Weight plus KV memory is a lower bound; temporary buffers are not certified",
            ),
        )
        return ServiceEstimate(profile, estimate, features, kv_after)


class CalibratedRolloutModel:
    def __init__(self, fits, cal_rollout, rollout_template):
        self.fits = fits
        self.cal_rollout = cal_rollout
        self.rollout_template = rollout_template
        self.capacities = dict(cal_rollout.capacities)

    @lru_cache(maxsize=128)
    def rollout_config(self, layout: OnlineLayout):
        return replace(
            self.rollout_template,
            replicas=layout.rollout_replicas,
            tp=layout.rollout_tp,
            kv_capacity_tokens=self.capacities[layout.rollout_tp],
        )

    def service(self, layout, kv, slices):
        cfg = self.rollout_config(layout)
        query = sum(s.query_tokens for s in slices)
        if (
            len(slices) > cfg.max_batch_sequences
            or query > cfg.max_batch_tokens
            or kv + query > cfg.kv_capacity_tokens
        ):
            raise ValueError(
                "GPU batch exceeds the measured sequence/token/KV capacity"
            )
        durations, intercepts = ([], [])
        for phase in ("prefill", "decode"):
            subset = tuple(s for s in slices if s.kind == phase)
            if not subset:
                continue
            _, features = self.fits.rollout_features(subset)
            fit = next(
                (
                    fit
                    for tp, name, fit in self.cal_rollout.fits
                    if tp == cfg.tp and name == phase
                )
            )
            seconds, _ = fit.evaluate(features, extrapolate=True)
            durations.append(seconds)
            intercepts.append(fit.coefficients[0])
        return sum(durations) - sum(intercepts) + max(intercepts)

    def decode_affine(self, layout, kv, slices):
        """Exact per-step growth for this frozen linear, pure-decode fit."""
        if not slices or any(s.kind != "decode" for s in slices):
            raise ValueError("Affine decode requires a pure decode batch")
        first = self.service(layout, kv, slices)
        fit = next(
            (
                fit
                for tp, phase, fit in self.cal_rollout.fits
                if tp == layout.rollout_tp and phase == "decode"
            )
        )
        return (first, fit.coefficients[2] * len(slices) / (64 * 16384))

# SPDX-License-Identifier: Apache-2.0

"""Predict rollout service, Actor work and publication costs for candidate layouts."""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from ..workload import ActorMemoryRejected, WorkloadStats
from ..workload import ExecutionConfig as CandidateConfig
from . import fingerprint
from .actor.model import Advantages
from .contracts import ActorConfig, ParallelLayout, ServiceState, nonnegative
from .loader import (
    Catalog,
    _load_profile_components,
)
from .publication import publication_seconds
from .rollout.model import memory_per_rank, supports_layout


class AnalyticalCostModels:
    """Analytical Megatron/SGLang costs with optionally fitted operator parameters.

    Profiles fix hardware/software and operator efficiencies. The same object
    serves the experiment environment and the independent candidate evaluator.
    """

    def __init__(
        self, actor, actor_template, rollout, rollout_template, bandwidth_GBps
    ):
        nonnegative("bandwidth_GBps", bandwidth_GBps, positive=True)
        self.actor_model = replace(
            actor, advantage_model=actor.advantage_model or Advantages()
        )
        self.actor_template, self.rollout_model, self.rollout_template = (
            actor_template,
            rollout,
            rollout_template,
        )
        self.bandwidth = bandwidth_GBps
        self.weight_bytes = actor.model.total_parameters * 2
        self.source_ids = (
            fingerprint(
                "megatron-sglang-analytical-v1",
                actor.profile_key(actor_template),
                rollout.profile_key(rollout_template),
                bandwidth_GBps,
            ),
        ) + tuple(dict.fromkeys(actor.parameter_sources + rollout.parameter_sources))
        self.scenario_notes = (
            "Megatron/SGLang full configurations use analytical compute, communication, memory and publication priors",
            "Candidate layouts share operator coefficients; prediction accuracy depends on measurement coverage",
            "Actor advantage uses an explicit bandwidth/launch prior when no measured model is supplied",
        )

    @classmethod
    def from_profiles(
        cls,
        *,
        hardware,
        topology,
        model,
        actor,
        rollout,
        bandwidth_GBps,
        inter_node_GBps=None,
        catalog: Catalog,
        calibration: Path | None = None,
        actor_runtime: ActorConfig | None = None,
    ):
        """Load profiles and optional fitted coefficients from JSON.

        A calibration names its hardware and model profiles. Each fitted role
        (actor, rollout) supplies runtime_revision, source_ids and a sparse
        parameters object (e.g. rollout's hbm efficiency). Include the
        topology profile when fitting network efficiency. For a live Actor pass
        ActorConfig.from_areal(resolved_actor) to use its execution settings.
        """
        actor_model, actor_template, rollout_model, rollout_template = (
            _load_profile_components(
                catalog,
                hardware,
                topology,
                model,
                actor,
                rollout,
                inter_node_GBps,
                calibration,
            )
        )
        result = cls(
            actor_model,
            actor_runtime if actor_runtime is not None else actor_template,
            rollout_model,
            rollout_template,
            bandwidth_GBps,
        )
        result.worker_spec = (
            "analytical",
            hardware,
            topology,
            model,
            actor,
            rollout,
            bandwidth_GBps,
            inter_node_GBps,
            str(catalog.root),
            str(Path(calibration).resolve()) if calibration is not None else None,
            actor_runtime,
        )
        return result

    def validate_layout(self, candidate):
        """Check model dimensions and weight fit; batch packing needs observations."""
        from .actor.megatron import MegatronCosts

        if not MegatronCosts.mesh_is_valid(
            self.actor_config(candidate), self.actor_model.model
        ):
            raise ValueError("Model dimensions do not divide the Megatron layout")
        self.rollout_config(candidate)

    def actor_config(self, candidate):
        parallel = (
            ParallelLayout(*candidate.actor_parallel)
            if isinstance(candidate, CandidateConfig)
            else ParallelLayout(dp=candidate.actor_dp, tp=candidate.actor_tp)
        )
        values = dict(parallel=parallel)
        if isinstance(candidate, CandidateConfig):
            values.update(
                n_mbs=candidate.actor_n_mbs or self.actor_template.n_mbs,
                max_tokens_per_mb=candidate.actor_max_tokens_per_mb
                or self.actor_template.max_tokens_per_mb,
                pad_to_maximum=candidate.actor_pad_to_maximum,
            )
        return replace(self.actor_template, **values)

    def resolve(self, candidate):
        """Bind inherited settings before candidate identity, caching or handoff."""
        if not isinstance(candidate, CandidateConfig):
            return candidate
        return replace(
            candidate,
            actor_n_mbs=candidate.actor_n_mbs or self.actor_template.n_mbs,
            actor_max_tokens_per_mb=candidate.actor_max_tokens_per_mb
            or self.actor_template.max_tokens_per_mb,
            rollout_max_concurrent_rollouts=candidate.rollout_max_concurrent_rollouts,
            sglang_max_running_requests=candidate.sglang_max_running_requests
            or self.rollout_template.max_batch_sequences,
            sglang_mem_fraction_static=candidate.sglang_mem_fraction_static
            if candidate.sglang_mem_fraction_static is not None
            else self.rollout_template.mem_fraction_static,
            sglang_enable_dp_attention=candidate.sglang_dp_size > 1,
        )

    @lru_cache(maxsize=512)
    def rollout_config(self, candidate):
        values = dict(replicas=candidate.rollout_replicas, tp=candidate.rollout_tp)
        if isinstance(candidate, CandidateConfig):
            values.update(
                pp=candidate.rollout_pp,
                dp_size=candidate.sglang_dp_size,
                ep_size=1,  # Expert-parallel search awaits reconfiguration validation.
                max_batch_sequences=candidate.sglang_max_running_requests
                or self.rollout_template.max_batch_sequences,
                max_batch_tokens=candidate.sglang_max_prefill_tokens,
                mem_fraction_static=candidate.sglang_mem_fraction_static
                if candidate.sglang_mem_fraction_static is not None
                else self.rollout_template.mem_fraction_static,
            )
        config = replace(self.rollout_template, **values)
        model, hardware = self.rollout_model.model, self.rollout_model.hardware
        if not supports_layout(model, config):
            raise ValueError("Model dimensions do not divide the SGLang layout")
        weights, kv_bytes = memory_per_rank(model, config)
        available = hardware.memory_gb * 1e9 * config.mem_fraction_static - weights
        capacity = math.floor(available / kv_bytes) * config.dp_size
        if capacity <= 0:
            raise ValueError(
                "SGLang weights exhaust the configured static memory budget"
            )
        return replace(config, kv_capacity_tokens=capacity)

    def service(self, layout, kv, slices):
        return self.rollout_model.estimate(
            self.rollout_config(layout), ServiceState(kv), tuple(slices)
        ).duration.require_seconds()

    def decode_interval(self, layout, kv, slices, max_steps):
        return self.rollout_model.decode_interval(
            self.rollout_config(layout), kv, slices, max_steps
        )

    @lru_cache(maxsize=512)
    def actor_estimate(self, layout, batch):
        config = self.actor_config(layout)
        work = WorkloadStats.from_batch(batch)
        return self.actor_model.estimate(config, work)

    def actor(self, layout, batch):
        result = self.actor_estimate(layout, batch)
        if result.total.feasibility == "infeasible":
            raise ActorMemoryRejected(
                "Megatron candidate exceeds model/packing/memory constraints: "
                + "; ".join(result.total.notes)
            )
        return tuple(
            (name, estimate.require_seconds()) for name, estimate in result.stages
        )

    def actor_memory(self, layout, batch):
        result = self.actor_estimate(layout, batch)
        if not result.peak_memory_gb:
            raise ValueError("Megatron peak memory unavailable for this layout/batch")
        return (
            max(result.peak_memory_gb) * 1e9 / 2**30,
            self.actor_model.hardware.memory_gb * 1e9 / 2**30,
            False,
        )

    @lru_cache(maxsize=512)
    def publication(self, layout, recipients):
        return publication_seconds(
            self.actor_model.model,
            self.actor_model.hardware,
            self.actor_config(layout).parallel,
            (self.rollout_config(recipient) for recipient in recipients),
            weight_bytes=self.weight_bytes,
            hbm_efficiency=self.actor_model.efficiency.hbm,
            bandwidth=self.bandwidth,
        )

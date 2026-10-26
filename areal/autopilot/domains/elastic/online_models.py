# SPDX-License-Identifier: Apache-2.0

"""Read-only composition of a frozen component bundle and scoped mechanism fits."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from areal.elastic.contracts import WorkloadStats
from areal.elastic.policy.basic.forecast import OnlineLayout, Turn
from areal.elastic.policy.reference import ParallelLayout
from areal.elastic.policy.reference.core import fingerprint, nonnegative
from areal.elastic.policy.reference.cost_model._actor.components import (
    Context,
    Operators,
)
from areal.elastic.policy.reference.cost_model._actor.packing import (
    _pad,
    dispatch_workload,
    ffd,
)

from .online_transitions import ActorMemoryRejected


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CalibratedOnlineModels:
    def __init__(
        self,
        bundle: Path,
        mechanism_fits: Path,
        publication_metadata: Path,
        effective_bandwidth_GBps: float,
        *,
        inter_node_dp_GBps: float | None = None,
    ):
        self.worker_spec = (
            str(bundle.resolve()),
            str(mechanism_fits.resolve()),
            str(publication_metadata.resolve()),
            effective_bandwidth_GBps,
        )
        self.inter_node_dp_GBps = inter_node_dp_GBps
        if inter_node_dp_GBps is not None:
            nonnegative("inter_node_dp_GBps", inter_node_dp_GBps, positive=True)
            self.worker_spec += (inter_node_dp_GBps,)
        self.scenario_notes = (
            (
                f"Actor DP beyond one8GPU node extrapolates frozen fits with assumed effective{inter_node_dp_GBps}GB/s collective bandwidth; TP groups stay node-local",
                "Cross-node Rollout replicas reuse single-replica fits; placement interference and extra-node startup/publication/state paths are assumptions",
            )
            if inter_node_dp_GBps is not None
            else ()
        )
        name = "_areal_online_frozen_" + digest(bundle / "models.py")[:16]
        spec = importlib.util.spec_from_file_location(name, bundle / "models.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.fits = module
        _, parts, self.cal_actor, self.cal_rollout = module.load_bundle(
            bundle, extrapolate=True
        )
        self.manifest = json.loads((bundle / "manifest.json").read_text())
        self.actor_template, self.rollout_template = (
            parts.config.actor,
            parts.config.rollout,
        )
        self.mechanism = json.loads(mechanism_fits.read_text())
        if self.mechanism["frozen_bundle_id"] != self.manifest["bundle_id"]:
            raise ValueError(
                "Mechanism fits belong to a different frozen component bundle"
            )
        self.capacities = dict(self.cal_rollout.capacities)
        self.bandwidth = effective_bandwidth_GBps
        raw = json.loads(publication_metadata.read_text())
        self.parameters = tuple(
            (
                p["data"]["name"],
                tuple(p["data"]["global_shape"]),
                p["data"]["shards"][0]["data"]["sharding_dim"],
                p["data"]["shards"][0]["data"]["num_shards"] > 1,
            )
            for p in raw["parameters"]
        )
        self.weight_bytes = sum(
            math.prod(shape) * 2 for _, shape, _, _ in self.parameters
        )
        self.source_ids = (
            self.manifest["bundle_id"],
            digest(mechanism_fits),
            digest(publication_metadata),
        )
        if inter_node_dp_GBps is not None:
            self.source_ids += (
                fingerprint("assumed-cross-node-DP-v1", inter_node_dp_GBps),
            )

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
        durations, intercepts = [], []
        for phase in ("prefill", "decode"):
            subset = tuple(s for s in slices if s.kind == phase)
            if not subset:
                continue
            _, features = self.fits.rollout_features(subset)
            fit = next(
                fit
                for tp, name, fit in self.cal_rollout.fits
                if tp == cfg.tp and name == phase
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
            fit
            for tp, phase, fit in self.cal_rollout.fits
            if tp == layout.rollout_tp and phase == "decode"
        )
        return first, fit.coefficients[2] * len(slices) / (64 * 16384)

    def actor(self, layout: OnlineLayout, batch: tuple[tuple[Turn, ...], ...]):
        return self._actor(layout.actor_dp, layout.actor_tp, batch)

    @lru_cache(maxsize=512)
    def _actor(self, dp: int, tp: int, batch: tuple[tuple[Turn, ...], ...]):
        base = self.cal_actor.base
        if dp * tp > 8:
            if self.inter_node_dp_GBps is None:
                raise ValueError(
                    "Cross-node Actor needs an explicit assumed DP bandwidth"
                )
            groups = dict(base.group_gbps)
            groups.update(
                dense_dp=self.inter_node_dp_GBps, expert_dp=self.inter_node_dp_GBps
            )
            base = replace(base, group_gbps=tuple(sorted(groups.items())))
        config = replace(
            self.actor_template,
            parallel=ParallelLayout(dp=dp, tp=tp),
        )
        lengths = tuple(
            t.input_tokens + t.output_tokens for episode in batch for t in episode
        )
        work = WorkloadStats(
            len(lengths),
            sum(lengths),
            loss_tokens=sum(t.output_tokens for e in batch for t in e),
            sequence_lengths=lengths,
            trajectory_group_sizes=tuple(len(e) for e in batch),
            trajectory_dispatch_weights=tuple(
                len(e) * max(t.input_tokens + t.output_tokens for t in e) for e in batch
            ),
        )
        raw, features = self.fits.actor_features(base, config, work)
        if not features:
            raise ValueError("Actor workload packing is unsupported")
        memory = self._memory_features(config, work, raw)
        peak = sum(
            a * b
            for a, b in zip(
                memory, self.mechanism["memory"]["coefficients"], strict=True
            )
        )
        limit = self.cal_actor.base.hardware.memory_gb * 1e9 / 2**30
        rows = dispatch_workload(
            work, list(work.lengths()), config.parallel.dp, self.cal_actor.base.dispatch
        )
        measured_oom = any(
            (dp, tp, tuple(sorted(row))) in self.cal_actor.oom_shapes for row in rows
        )
        if peak > limit or measured_oom:
            raise ActorMemoryRejected(
                f"Conditional Actor memory rejection: {peak:.3f}GiB / {limit:.3f}GiB; measured_shape_oom={measured_oom}"
            )
        result = []
        for phase, values in features.items():
            fit = next(
                f
                for tp, name, f in self.cal_actor.fits
                if tp == config.parallel.tp and name == phase
            )
            seconds, _ = fit.evaluate(values, extrapolate=True)
            result.append((phase, seconds))
        return tuple(result)

    def _memory_features(self, config, work, raw):
        base = self.cal_actor.base
        rows = dispatch_workload(
            work, list(work.lengths()), config.parallel.dp, base.dispatch
        )
        dense, expert = Context(
            config,
            base.model,
            Operators(base.hardware, base.efficiency),
            base.model.layers,
            True,
            True,
        ).owned_parameters
        persistent = (
            4 * (dense + expert) + 12 * (dense + expert) / config.parallel.dp
        ) / 2**30
        maximum = max(
            m.compute_tokens
            for update in raw.packing.train
            for rank in update
            for m in rank
        )
        parts = [ffd(row, sum(row), config.ppo_n_minibatches) for row in rows]
        repack = 0.0
        for update, ranks in enumerate(raw.packing.train):
            minimum = max(config.minimum_microbatches, work.train_microbatches or 1)
            divisor = (
                config.parallel.pp
                if config.backend == "megatron" or config.parallel.pp > 1
                else config.n_mbs_divisor
            )
            while True:
                grouped = [
                    ffd(parts[rank][update], config.max_tokens_per_mb, minimum, divisor)
                    for rank in range(len(rows))
                ]
                count = max(map(len, grouped))
                if all(len(bins) == count for bins in grouped):
                    break
                minimum = count
            for rank, batches in enumerate(ranks):
                bins = grouped[rank]
                if (
                    tuple(_pad(b, config, base.model.hidden_size) for b in bins)
                    != batches
                ):
                    raise ValueError(
                        "Memory feature replay disagrees with frozen Actor packing"
                    )
                repack = max(repack, *(sum(b) * (1 - 1 / len(b)) for b in bins))
        return (
            1.0,
            persistent,
            maximum / config.parallel.tp / 32768,
            repack / config.parallel.tp / 32768,
        )

    @lru_cache(maxsize=256)
    def publication(self, layout: OnlineLayout, recipients: tuple[OnlineLayout, ...]):
        actors = layout.actor_gpus
        sent, ops, copies = [0] * actors, [0] * actors, [{} for _ in range(actors)]
        received = [0] * sum(r.rollout_gpus for r in recipients)
        for name, shape, dimension, sharded in self.parameters:
            endpoint = engine = 0
            for target in recipients:
                for _ in range(target.rollout_replicas):
                    for rank in range(target.rollout_tp):
                        local = list(shape)
                        if sharded:
                            local[dimension] //= target.rollout_tp
                        source = (engine if sharded else endpoint) % actors
                        size = math.prod(local) * 2
                        sent[source] += size
                        received[endpoint] += size
                        ops[source] += 1
                        if sharded and target.rollout_tp > 1 and dimension > 0:
                            copies[source][name, rank, target.rollout_tp] = size
                        endpoint += 1
                    engine += 1
        fit = self.mechanism["publication"]
        coefficients = fit["prep_fit"]["coefficients"]
        gather = self.weight_bytes * (1 - 1 / layout.actor_tp) / 1e9
        prep = max(
            sum(
                a * b
                for a, b in zip(
                    coefficients,
                    (1.0, gather, sum(copy.values()) / 1e9, float(count)),
                    strict=True,
                )
            )
            for copy, count in zip(copies, ops, strict=True)
        )
        endpoint = fit["endpoint_fit"]["coefficients"][0] * max(*sent, *received) / 1e9
        wire = max(endpoint, sum(sent) / (self.bandwidth * 1e9))
        return fit["control_seconds"] + prep + wire

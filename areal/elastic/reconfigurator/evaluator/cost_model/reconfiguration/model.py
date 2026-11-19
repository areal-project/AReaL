# SPDX-License-Identifier: Apache-2.0

"""Estimate v2 migration stages and compose them along the GPU handoff path."""

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence, Set
from dataclasses import dataclass
from math import gcd, isfinite, log2
from operator import attrgetter

from ....optimizer import Layout, RoleLayout
from ...workload import ExecutionConfig, parse_actor_parallel
from ..contracts import nonnegative


@dataclass(frozen=True)
class Cutover:
    seconds: float
    gpu_seconds: float  # integral of GPUs occupied during the pause


@dataclass(frozen=True)
class RoleTiming:
    """Actor release includes save, rebuild includes load. Rollout rebuild includes rebind."""

    release: float
    rebuild: float

    def __post_init__(self):
        if not all(isfinite(t) and t >= 0 for t in (self.release, self.rebuild)):
            raise ValueError("Invalid release or rebuild duration")


@dataclass(frozen=True)
class IOBandwidth:
    """Effective decimal GB/s per rank, per host and across shared storage."""

    rank: float
    node: float
    shared: float

    def __post_init__(self):
        for key in ("rank", "node", "shared"):
            nonnegative(key, getattr(self, key), positive=True)

    def seconds(self, peak_rank_gb, host_gb):
        return max(
            peak_rank_gb / self.rank,
            max(host_gb.values()) / self.node,
            sum(host_gb.values()) / self.shared,
        )


@dataclass(frozen=True)
class ActorStages:
    """Raw snapshot, distributed optimizer, Megatron TP-fastest rank order."""

    model_gb: float
    optimizer_gb: float  # includes master weights and both Adam moments
    write: IOBandwidth
    read: IOBandwidth
    release_s: float
    engine_s: float
    cold_engine_s: float = 0  # extra when a target host is new to the Actor
    extra_source_s: float = 0  # assembly cost per additional source partition
    expert_fraction: float = 0  # fraction of weights using the expert mesh
    warm_nodes: Set[Hashable] = frozenset()

    def __post_init__(self):
        for key in (
            "model_gb",
            "optimizer_gb",
            "release_s",
            "engine_s",
            "cold_engine_s",
            "extra_source_s",
            "expert_fraction",
        ):
            nonnegative(key, getattr(self, key))
        if self.expert_fraction > 1:
            raise ValueError("expert_fraction must be at most one")

    def _io(self, parallel, nodes, *, writing):
        dp, tp, pp, cp, ep, etp = parallel
        if len(nodes) != dp * tp * pp * cp:
            raise ValueError("Megatron placement must include all ranks")
        stage_size = len(nodes) // pp
        per_rank = self.optimizer_gb / len(nodes)
        rank_gb = [per_rank] * len(nodes)
        host_gb = defaultdict(float)
        for host in nodes:
            host_gb[host] += per_rank
        for width, fraction in (
            (tp, 1 - self.expert_fraction),
            (ep * etp, self.expert_fraction),
        ):
            shard_gb = self.model_gb * fraction / (pp * width)
            seen = set()
            for rank, host in enumerate(nodes):
                stage, local = divmod(rank, stage_size)
                if writing and local >= width:  # only the main replica writes
                    continue
                rank_gb[rank] += shard_gb
                # DP readers on a host copy separately but share file cache pages.
                shard = (host, stage, local % width)
                if shard not in seen:
                    host_gb[host] += shard_gb
                    seen.add(shard)
        bandwidth = self.write if writing else self.read
        return bandwidth.seconds(max(rank_gb), host_gb)

    def predict(self, source, target, old_nodes, new_nodes):
        old = parse_actor_parallel(source.actor.backend)
        new = parse_actor_parallel(target.actor.backend)
        save = self._io(old, old_nodes, writing=True)
        load = self._io(new, new_nodes, writing=False)

        # Mean extra source intervals overlapping one target interval, for uniform
        # partitions. Account for TP model shards and distributed optimizer shards.
        def extra(src, dst):
            return (src - gcd(src, dst)) / dst

        model_extra = (1 - self.expert_fraction) * extra(old[1], new[1])
        model_extra += self.expert_fraction * extra(old[4] * old[5], new[4] * new[5])
        assembly = self.extra_source_s * (
            model_extra + extra(len(old_nodes), len(new_nodes))
        )
        cold = set(new_nodes).difference(old_nodes, self.warm_nodes)
        engine = self.engine_s + (self.cold_engine_s if cold else 0)
        return RoleTiming(save + self.release_s, engine + load + assembly)


@dataclass(frozen=True)
class RolloutStages:
    """SGLang replicas launch concurrently; the slowest participating host gates ready."""

    release_s: float
    warm_start_s: float
    extra_replica_s: float  # host launch contention per additional replica
    group_doubling_s: float  # TP/PP process-group setup
    cold_jit_s: float
    warm_nodes: Set[Hashable] = frozenset()  # hosts with reusable kernels

    def __post_init__(self):
        for key in (
            "release_s",
            "warm_start_s",
            "extra_replica_s",
            "group_doubling_s",
            "cold_jit_s",
        ):
            nonnegative(key, getattr(self, key))

    def predict(self, source, target, old_nodes, new_nodes):
        cfg = ExecutionConfig(target.actor.backend, target.rollout.backend)
        replicas, tp, pp = cfg.rollout_parallel
        width = tp * pp
        if len(new_nodes) != replicas * width:
            raise ValueError("SGLang placement must include all replica groups")
        groups = Counter(
            node
            for first in range(0, len(new_nodes), width)
            for node in set(new_nodes[first : first + width])
        )
        warm = set(old_nodes).union(self.warm_nodes)
        host_overhead = max(
            (replicas - 1) * self.extra_replica_s
            + (0 if node in warm else self.cold_jit_s)
            for node, replicas in groups.items()
        )
        return RoleTiming(
            self.release_s,
            self.warm_start_s + log2(width) * self.group_doubling_s + host_overhead,
        )


# Explicit per-layout spans are also accepted for reconstruction of measured runs.
StageSource = (
    Mapping[RoleLayout, RoleTiming]
    | Callable[[Layout, Layout, Sequence[Hashable], Sequence[Hashable]], RoleTiming]
)


@dataclass(frozen=True)
class ReconfigurationModel:
    """Stage seconds for one hardware/model/storage profile, with actual device placement."""

    actor: StageSource
    rollout: StageSource
    placement: Callable[[Layout], tuple[Sequence[Hashable], Sequence[Hashable]]]
    pause_disconnect: float
    reconnect_resume: float  # excludes the target layout's weight publication
    node_of: Callable[[Hashable], Hashable] = attrgetter("node")

    def predict(self, source: Layout, target: Layout, publication: float) -> Cutover:
        if source == target:
            return Cutover(0, 0)
        if not all(
            isfinite(t) and t >= 0
            for t in (self.pause_disconnect, self.reconnect_resume, publication)
        ):
            raise ValueError("Invalid pause, reconnection or publication duration")

        def devices(layout):
            try:
                actor, rollout = self.placement(layout)
            except KeyError as exc:
                raise ValueError(f"Missing device placement for {layout}") from exc
            if (len(actor), len(rollout)) != (
                layout.actor.gpus,
                layout.rollout.gpus,
            ) or len(set(actor).union(rollout)) != layout.gpus:
                raise ValueError("Placement must match both disjoint role layouts")
            return tuple(actor), tuple(rollout)

        def stages(profile, old, new, before, after):
            if old == new and before == after:
                return 0.0, 0.0
            if callable(profile):
                result = profile(
                    source,
                    target,
                    tuple(map(self.node_of, before)),
                    tuple(map(self.node_of, after)),
                )
                return result.release, result.rebuild
            try:
                return profile[old].release, profile[new].rebuild
            except KeyError as exc:
                raise ValueError(f"Missing reconfiguration timings for {exc}") from exc

        old_a, old_r = devices(source)
        new_a, new_r = devices(target)
        a_release, a_rebuild = stages(
            self.actor, source.actor, target.actor, old_a, new_a
        )
        r_release, r_rebuild = stages(
            self.rollout, source.rollout, target.rollout, old_r, new_r
        )
        # Both releases start together. A rebuild may also need GPUs from the other role.
        a_start = max(a_release, r_release if set(new_a).intersection(old_r) else 0.0)
        r_start = max(r_release, a_release if set(new_r).intersection(old_a) else 0.0)
        migration = max(a_start + a_rebuild, r_start + r_rebuild)
        finish = migration + self.reconnect_resume + publication
        # Charge old devices until release and target devices from the start of rebuild.
        gpu_seconds = (
            source.gpus * self.pause_disconnect
            + len(old_a) * a_release
            + len(new_a) * (finish - a_start)
            + len(old_r) * r_release
            + len(new_r) * (finish - r_start)
        )
        return Cutover(self.pause_disconnect + finish, gpu_seconds)

# SPDX-License-Identifier: Apache-2.0

"""Steady-state weight publication from explicit rank work and peer traffic.

No training-engine or AWEX dependency. Traffic comes from a validated peer plan,
including replica fanout; this model never substitutes model size times replicas.
Preparation is a bulk-synchronous approximation. Network rounds use concurrent
resource bottlenecks, not an exact NCCL stream scheduler. All rates are priors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace

from ..core import (
    Estimate,
    Features,
    PhaseCalibration,
    code_revision,
    fingerprint,
    integer,
    nonnegative,
)
from ._actor.config import Hardware
from .transition import TransferTask, TransitionProfile


def publication_binding(model, actor, rollout) -> str:
    """Bind geometry to the resolved source, destination, runtime and hardware."""
    return fingerprint(
        model.model, model.hardware, model.runtime_revision, actor, rollout
    )


@dataclass(frozen=True)
class PublicationRank:
    name: str
    role: str
    node: str
    # Logical traffic, including read+write for materialized GPU copies.
    h2d_bytes: int | None = None
    h2d_tensors: int | None = None
    gather_bytes: int | None = None
    gather_calls: int | None = None
    convert_bytes: int | None = None
    convert_tensors: int | None = None
    send_copy_bytes: int | None = None
    send_copy_tensors: int | None = None
    receive_prepare_bytes: int | None = None
    receive_prepare_tensors: int | None = None
    receive_copy_bytes: int | None = None
    receive_copy_tensors: int | None = None
    expected_receive_bytes: int | None = None

    def __post_init__(self):
        if not self.name or not self.node or self.role not in {"actor", "rollout"}:
            raise ValueError("Publication ranks require identity, role and node")
        for field in fields(self)[3:]:
            value = getattr(self, field.name)
            if value is not None:
                integer(field.name, value)
        for prefix in (
            "h2d",
            "gather",
            "convert",
            "send_copy",
            "receive_prepare",
            "receive_copy",
        ):
            size, count = (
                getattr(self, prefix + "_bytes"),
                getattr(
                    self, prefix + ("_calls" if prefix == "gather" else "_tensors")
                ),
            )
            if size and count == 0:
                raise ValueError("Nonzero publication traffic needs an operation count")


@dataclass(frozen=True)
class PublicationFlow:
    source: str
    target: str
    payload_bytes: int
    messages: int
    round: int = 0

    def __post_init__(self):
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("Publication flow needs distinct peers")
        integer("payload_bytes", self.payload_bytes, minimum=1)
        integer("messages", self.messages, minimum=1)
        integer("round", self.round)


@dataclass(frozen=True)
class PublicationRates:
    h2d_gbps: float | None = None  # one concurrently staging rank
    host_h2d_gbps: float | None = None  # shared capacity per node
    h2d_launch_seconds: float | None = None
    gather_gbps: float | None = None  # effective concurrent-rank collective rate
    gather_launch_seconds: float | None = None
    copy_gbps: float | None = None  # logical read+write GB/s per GPU
    copy_launch_seconds: float | None = None
    network_efficiency: float | None = None
    node_network_gbps: float | None = None  # shared NIC capacity, one direction
    node_fabric_gbps: float | None = None  # shared local GPU fabric, one direction
    message_launch_seconds: float | None = None
    bucket_ack_seconds: float | None = None  # legacy XCCL RPC completion per round
    pause_seconds: float | None = None
    cleanup_seconds: float | None = None
    barrier_seconds: float | None = None
    resume_seconds: float | None = None  # includes version confirmation

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                nonnegative(field.name, value, positive=field.name.endswith("gbps"))
        if self.network_efficiency is not None and not 0 < self.network_efficiency <= 1:
            raise ValueError("network_efficiency must be in (0, 1]")


@dataclass(frozen=True)
class PublicationWork:
    binding: str
    backend: str
    transport: str
    source_id: str
    ranks: tuple[PublicationRank, ...]
    flows: tuple[PublicationFlow, ...]
    schedule: str = "bottleneck_rounds"
    # True only for a complete exported or explicitly constructed transfer plan.
    complete: bool = False

    def __post_init__(self):
        object.__setattr__(self, "ranks", tuple(self.ranks))
        object.__setattr__(self, "flows", tuple(self.flows))
        if not self.binding or not self.source_id:
            raise ValueError("Publication work requires binding and provenance")
        if self.backend not in {"fsdp", "megatron", "archon"}:
            raise ValueError("Unknown publication backend")
        if type(self.complete) is not bool:
            raise ValueError("complete must be a boolean")
        ranks = {r.name: r for r in self.ranks}
        if not ranks or len(ranks) != len(self.ranks):
            raise ValueError("Publication rank identities must be unique")
        if any(f.source not in ranks or f.target not in ranks for f in self.flows):
            raise ValueError("Unknown publication peer")
        if len({(f.source, f.target, f.round) for f in self.flows}) != len(self.flows):
            raise ValueError("Aggregate each peer pair once per network round")
        if self.transport in {"awex_dense", "xccl_broadcast"} and any(
            ranks[f.source].role != "actor" or ranks[f.target].role != "rollout"
            for f in self.flows
        ):
            raise ValueError("Publication requires Actor-to-Rollout peer traffic")


@dataclass(frozen=True)
class PublicationEstimate:
    duration: Estimate
    stages: tuple[tuple[str, Estimate], ...]
    ranks: tuple[dict, ...]
    rounds: tuple[dict, ...]
    profile: str
    features: Features = ()

    def calibrated(self, calibration: PhaseCalibration) -> PublicationEstimate:
        if self.duration.seconds is None:
            return self
        return replace(
            self,
            duration=calibration.apply(
                self.duration.seconds,
                self.profile,
                "update_weights",
                self.features,
                timing_boundary="trainer.pause-through-version-confirmation",
            ),
        )

    def transition(self, source: str, target: str) -> TransitionProfile:
        """Plug into ExpertSimulator's existing publication/version event path."""
        tasks, previous = [], ()
        # Whole-stage calibration replaces all raw components, never adds to them.
        stages = self.stages if self.duration.evidence == "assumed" else ()
        for name, estimate in stages or (("publication", self.duration),):
            tasks.append(TransferTask(name, previous, measured=estimate))
            previous = (name,)
        return TransitionProfile(
            "publish",
            source,
            target,
            tuple(tasks),
            blocks=("actor", "generation", "admission"),
        )


@dataclass(frozen=True)
class ExpertPublicationModel:
    hardware: Hardware
    rates: PublicationRates
    version: str = "expert-publication-v1"

    def __post_init__(self):
        if not self.version:
            raise ValueError("Publication formula version is required")

    def estimate(
        self, actor_model, actor, rollout, work: PublicationWork
    ) -> PublicationEstimate:
        profile = fingerprint(
            code_revision(), self.version, self.hardware, self.rates, work
        )
        unsupported = []
        if work.binding != publication_binding(actor_model, actor, rollout):
            unsupported.append(
                "Publication geometry binding mismatch; export or rebuild the target plan"
            )
        if self.hardware != actor_model.hardware or work.backend != actor.backend:
            unsupported.append("Publication hardware/backend mismatch")
        if work.transport not in {"awex_dense", "xccl_broadcast"} or (
            work.transport == "awex_dense" and actor.backend == "archon"
        ):
            unsupported.append(
                "Unsupported publication transport/backend; Archon v2 AWEX has no runtime adapter"
            )
        if work.schedule != "bottleneck_rounds":
            unsupported.append("Unknown publication scheduling approximation")
        if not work.complete or not work.flows:
            unsupported.append("Incomplete publication peer plan")
        for node in {r.node for r in work.ranks}:
            if sum(r.node == node for r in work.ranks) > self.hardware.gpus_per_node:
                unsupported.append(
                    "Publication node exceeds GPU capacity; colocate is unsupported"
                )
        if (
            sum(r.role == "actor" for r in work.ranks) != actor.parallel.world_size
            or sum(r.role == "rollout" for r in work.ranks) != rollout.world_size
        ):
            unsupported.append("Publication ranks do not match the resolved layouts")
        for rank in work.ranks:
            if any(getattr(rank, f.name) is None for f in fields(rank)[3:]):
                unsupported.append(f"Missing publication work counters: {rank.name}")
            received = sum(f.payload_bytes for f in work.flows if f.target == rank.name)
            if received != rank.expected_receive_bytes:
                unsupported.append(f"Incomplete receive coverage: {rank.name}")
        if unsupported:
            return PublicationEstimate(
                Estimate(None, "unsupported", notes=tuple(unsupported)),
                (),
                (),
                (),
                profile,
            )

        def value(seconds):
            return Estimate(seconds, "assumed", source_ids=(work.source_id,))

        missing = []

        def term(amount, rate_name, unit=1):
            rate = getattr(self.rates, rate_name)
            if not amount:
                return 0.0
            if rate is None:
                missing.append(rate_name)
                return 0.0
            return amount / (rate * unit)

        def launch(count, name):
            rate = getattr(self.rates, name)
            if rate is None:
                missing.append(name)
                return 0.0
            return count * rate

        rank_details, preparation, writeback = [], [], []
        staging_ranks = {}
        for r in work.ranks:
            if r.h2d_bytes:
                staging_ranks[r.node] = staging_ranks.get(r.node, 0) + 1
        for r in work.ranks:
            h2d = 0.0
            if r.h2d_bytes:
                if self.rates.h2d_gbps is None or self.rates.host_h2d_gbps is None:
                    missing.append("h2d_gbps/host_h2d_gbps")
                else:
                    h2d = r.h2d_bytes / (
                        1e9
                        * min(
                            self.rates.h2d_gbps,
                            self.rates.host_h2d_gbps / staging_ranks[r.node],
                        )
                    )
            components = {
                "staging": h2d
                + (launch(r.h2d_tensors, "h2d_launch_seconds") if r.h2d_tensors else 0),
                "gather": term(r.gather_bytes, "gather_gbps", 1e9)
                + (
                    launch(r.gather_calls, "gather_launch_seconds")
                    if r.gather_calls
                    else 0
                ),
                "conversion": term(r.convert_bytes, "copy_gbps", 1e9)
                + (
                    launch(r.convert_tensors, "copy_launch_seconds")
                    if r.convert_tensors
                    else 0
                ),
                "send_copy": term(r.send_copy_bytes, "copy_gbps", 1e9)
                + (
                    launch(r.send_copy_tensors, "copy_launch_seconds")
                    if r.send_copy_tensors
                    else 0
                ),
                "receive_prepare": term(r.receive_prepare_bytes, "copy_gbps", 1e9)
                + (
                    launch(r.receive_prepare_tensors, "copy_launch_seconds")
                    if r.receive_prepare_tensors
                    else 0
                ),
                "receive_copy": term(r.receive_copy_bytes, "copy_gbps", 1e9)
                + (
                    launch(r.receive_copy_tensors, "copy_launch_seconds")
                    if r.receive_copy_tensors
                    else 0
                ),
            }
            preparation.append(
                sum(v for k, v in components.items() if k != "receive_copy")
            )
            writeback.append(components["receive_copy"])
            rank_details.append({"rank": r.name, "components": components})
        # All local preparations precede network in this conservative bulk model.
        # Host capacity is partitioned equally among staging ranks, without
        # reclaiming an early finisher's share. Rank-local components stay serial.
        prep = max(preparation)
        peers = {r.name: r for r in work.ranks}
        rounds = []
        for index in sorted({f.round for f in work.flows}):
            loads, messages, capacities = {}, {}, {}
            for flow in (f for f in work.flows if f.round == index):
                src, dst = peers[flow.source], peers[flow.target]
                cross = src.node != dst.node
                resources = {
                    f"send:{src.name}": self.hardware.inter_node_gbps
                    if cross
                    else self.hardware.intra_node_gbps,
                    f"receive:{dst.name}": self.hardware.inter_node_gbps
                    if cross
                    else self.hardware.intra_node_gbps,
                }
                if cross:
                    resources.update(
                        {
                            f"nic-out:{src.node}": self.rates.node_network_gbps,
                            f"nic-in:{dst.node}": self.rates.node_network_gbps,
                        }
                    )
                else:
                    resources[f"fabric:{src.node}"] = self.rates.node_fabric_gbps
                for resource, gbps in resources.items():
                    loads[resource] = loads.get(resource, 0) + flow.payload_bytes
                    # If mixed local/remote traffic shares an endpoint, use the
                    # slower capacity; this is explicit conservative contention.
                    old = capacities.get(resource, gbps)
                    capacities[resource] = (
                        min(old, gbps) if old is not None and gbps is not None else None
                    )
                latency = (
                    self.hardware.inter_latency_us
                    if cross
                    else self.hardware.intra_latency_us
                )
                for resource in (f"send:{src.name}", f"receive:{dst.name}"):
                    messages[resource] = (
                        messages.get(resource, 0.0) + flow.messages * latency * 1e-6
                    )
            times = {}
            for resource, size in loads.items():
                capacity = capacities[resource]
                efficiency = self.rates.network_efficiency
                if capacity is None or efficiency is None:
                    missing.append(
                        (
                            "node_fabric_gbps"
                            if resource.startswith("fabric:")
                            else "node_network_gbps"
                        )
                        if capacity is None
                        else "network_efficiency"
                    )
                    times[resource] = 0.0
                else:
                    times[resource] = size / (
                        capacity * 1e9 * efficiency
                    ) + messages.get(resource, 0.0)
            counts = {}
            for flow in (f for f in work.flows if f.round == index):
                for rank in (flow.source, flow.target):
                    counts[rank] = counts.get(rank, 0) + flow.messages
            posting = max(
                launch(count, "message_launch_seconds") for count in counts.values()
            )
            ack = (
                launch(1, "bucket_ack_seconds")
                if work.transport == "xccl_broadcast"
                else 0.0
            )
            rounds.append(
                {
                    "round": index,
                    "resource_bytes": loads,
                    "resource_seconds": times,
                    "posting_seconds": posting,
                    "bucket_ack_seconds": ack,
                    "seconds": posting + max(times.values()) + ack,
                }
            )
        stages = (
            ("pause", launch(1, "pause_seconds")),
            ("cleanup", launch(1, "cleanup_seconds")),
            ("prepare", prep),
            ("network", sum(r["seconds"] for r in rounds)),
            ("writeback", max(writeback)),
            ("barrier", launch(1, "barrier_seconds")),
            ("resume", launch(1, "resume_seconds")),
        )
        if missing:
            # Provisional zeros used to collect missing rates are not predictions.
            rank_details = [
                {"rank": r["rank"], "components": {k: None for k in r["components"]}}
                for r in rank_details
            ]
            rounds = [
                {
                    **r,
                    "seconds": None,
                    "posting_seconds": None,
                    "bucket_ack_seconds": None,
                    "resource_seconds": {k: None for k in r["resource_seconds"]},
                }
                for r in rounds
            ]
            return PublicationEstimate(
                Estimate(
                    None,
                    "unsupported",
                    notes=(
                        "Missing publication rates: " + ", ".join(sorted(set(missing))),
                    ),
                ),
                (),
                tuple(rank_details),
                tuple(rounds),
                profile,
            )
        return PublicationEstimate(
            Estimate(
                sum(t for _, t in stages),
                "assumed",
                source_ids=(work.source_id,),
                notes=(
                    "Bulk-synchronous preparation/network/writeback prior; network rounds use shared-resource bottlenecks, not exact NCCL scheduling",
                    "Steady-state pause through version confirmation; connection, metadata registration and process-group setup excluded",
                    "No component rates fitted from a whole-stage label; all replica traffic retained and source copies deduplicated upstream per version",
                    "XCCL uses explicit star-expanded logical bucket traffic with one RPC ack per round; not an inferred NCCL broadcast tree or Archon v2 runtime support",
                ),
            ),
            tuple((name, value(t)) for name, t in stages),
            tuple(rank_details),
            tuple(rounds),
            profile,
            (
                ("wire_bytes", sum(f.payload_bytes for f in work.flows)),
                ("messages", sum(f.messages for f in work.flows)),
            ),
        )

    @classmethod
    def from_dict(cls, document: dict, hardware):
        if document.get("schema_version") != 1 or document.keys() - {
            "schema_version",
            "work",
            "rates",
            "provenance",
        }:
            raise ValueError("Unknown publication profile schema/fields")
        data = document["work"].copy()
        data["ranks"] = tuple(PublicationRank(**r) for r in data["ranks"])
        data["flows"] = tuple(PublicationFlow(**f) for f in data["flows"])
        return cls(hardware, PublicationRates(**document["rates"])), PublicationWork(
            **data
        )

    @staticmethod
    def report(result: PublicationEstimate) -> dict:
        return asdict(result)

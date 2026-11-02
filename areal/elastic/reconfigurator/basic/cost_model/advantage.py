# SPDX-License-Identifier: Apache-2.0

"""Explicit advantage phase boundary and operation counts; no fitted defaults.

Coefficients describe end-to-end components, not GPU efficiency. They must be
supplied from matching measurements or declared assumptions by the data owner.
"""

from __future__ import annotations

from dataclasses import dataclass

from ....contracts import WorkloadStats
from ..core import Estimate, Features, fingerprint, integer, nonnegative
from ._actor.config import ActorConfig
from ._actor.packing import DispatchConfig, dispatch_workload

ADVANTAGE_BOUNDARY = "trainer_host_wall_with_stats"
COMPONENTS = ("batching", "recurrence", "normalization", "transport", "diagnostics")
RANK_FEATURES = {
    "constant",
    "padded_slots",
    "gae_iterations",
    "recurrence_elements",
    "trajectories",
    "normalization_groups",
    "collectives",
}
GLOBAL_FEATURES = {"constant", "input_bytes", "output_bytes"}


@dataclass(frozen=True)
class AdvantageRankWork:
    dp_rank: int
    n_seqs: int
    padded_length: int
    trajectories: int | None = None
    turn_iterations: int | None = None
    normalization_groups: int | None = None
    collectives: int | None = None

    def __post_init__(self) -> None:
        integer("dp_rank", self.dp_rank)
        integer("n_seqs", self.n_seqs, minimum=1)
        integer("padded_length", self.padded_length, minimum=1)
        for name in (
            "trajectories",
            "turn_iterations",
            "normalization_groups",
            "collectives",
        ):
            if getattr(self, name) is not None:
                integer(name, getattr(self, name))
        if self.trajectories is not None and not 1 <= self.trajectories <= self.n_seqs:
            raise ValueError("Trajectories must partition rank sequences")
        if (
            self.turn_iterations is not None
            and self.turn_iterations > self.padded_length
        ):
            raise ValueError("Turn iterations exceed the padded width")

    def features(self, device: str, gae_mode: str) -> dict[str, float | None]:
        steps = (
            self.padded_length - 1
            if gae_mode == "token"
            else (self.padded_length if device == "cuda" else self.turn_iterations)
        )
        return {
            "constant": 1.0,
            "padded_slots": float(self.n_seqs * self.padded_length),
            "gae_iterations": steps,
            "recurrence_elements": None if steps is None else steps * self.n_seqs,
            "trajectories": self.trajectories,
            "normalization_groups": self.normalization_groups,
            "collectives": self.collectives,
        }


@dataclass(frozen=True)
class AdvantageWorkload:
    """Actual pre-GAE shapes, bound to a profile/workload; never training packing.

    input/output_bytes are total boundary transfer bytes from traces, including
    replication when applicable, not inferred from logical attended tokens.
    """

    profile: str
    workload_id: str
    ranks: tuple[AdvantageRankWork, ...]
    source_ids: tuple[str, ...]
    input_bytes: int | None = None
    output_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", tuple(self.ranks))
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        if (
            not self.profile
            or not self.workload_id
            or not self.ranks
            or not self.source_ids
        ):
            raise ValueError(
                "Observed advantage work needs profile, workload and sources"
            )
        if len({r.dp_rank for r in self.ranks}) != len(self.ranks):
            raise ValueError("Duplicate advantage DP rank")
        for name in ("input_bytes", "output_bytes"):
            if getattr(self, name) is not None:
                integer(name, getattr(self, name))

    @classmethod
    def for_workload(
        cls,
        profile: str,
        workload: WorkloadStats,
        ranks: tuple[AdvantageRankWork, ...],
        source_ids: tuple[str, ...],
        *,
        input_bytes: int | None = None,
        output_bytes: int | None = None,
    ):
        return cls(
            profile, fingerprint(workload), ranks, source_ids, input_bytes, output_bytes
        )


@dataclass(frozen=True)
class AdvantageTerm:
    """Nonnegative seconds per named work unit for an exclusive component.

    None preserves an unknown coefficient. Explicit zero requires a declared
    source/assumption too; phase-total labels must not also be added as terms.
    """

    name: str
    coefficients: tuple[tuple[str, float | None], ...]
    source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "coefficients", tuple(tuple(x) for x in self.coefficients)
        )
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        allowed = (
            GLOBAL_FEATURES
            if self.name in {"transport", "diagnostics"}
            else RANK_FEATURES
        )
        if self.name not in COMPONENTS or not self.coefficients or not self.source_ids:
            raise ValueError(
                "Advantage term needs a known component, coefficients and provenance"
            )
        if len(dict(self.coefficients)) != len(self.coefficients):
            raise ValueError("Duplicate advantage coefficient")
        for name, value in self.coefficients:
            if name not in allowed:
                raise ValueError(f"Unsupported {self.name} feature: {name}")
            if value is not None:
                nonnegative(name, value)

    def seconds(self, features: dict[str, float | None]) -> float | None:
        if any(
            c is None or (c != 0 and features.get(k) is None)
            for k, c in self.coefficients
        ):
            return None
        return sum(c * (features.get(k) or 0) for k, c in self.coefficients)


@dataclass(frozen=True)
class AdvantageEstimate:
    total: Estimate
    components: tuple[tuple[str, Estimate], ...]
    features: Features


@dataclass(frozen=True)
class AdvantageModel:
    """Scoped prior for trainer wall time: transport + slowest worker + diagnostics.

    execution_id must fingerprint normalization, GAE/lambda/KL options, tensor
    fields/dtypes and transfer/cache protocol. Mean-reduced multi-controller
    summaries and worker-only spans are not this boundary's calibration labels.
    """

    execution_id: str
    device: str
    controller_mode: str
    gae_mode: str
    terms: tuple[AdvantageTerm, ...] = ()
    timing_boundary: str = ADVANTAGE_BOUNDARY

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        if (
            not self.execution_id
            or self.device not in {"cpu", "cuda"}
            or self.controller_mode not in {"single", "multi"}
            or self.gae_mode not in {"token", "turn"}
        ):
            raise ValueError(
                "Advantage execution identity, device, controller and GAE mode are required"
            )
        if self.timing_boundary != ADVANTAGE_BOUNDARY:
            raise ValueError(
                "Actor simulation requires trainer host wall time with device stats"
            )
        if len({t.name for t in self.terms}) != len(self.terms):
            raise ValueError("Duplicate advantage component")

    def estimate(
        self,
        config: ActorConfig,
        workload: WorkloadStats,
        profile: str,
        observed: AdvantageWorkload | None = None,
        *,
        dispatch: DispatchConfig = DispatchConfig(),
    ) -> AdvantageEstimate:
        notes = [
            "Advantage trainer wall boundary includes batching, recurrence, normalization, transport and diagnostics",
            "Component coefficients are scoped priors, not GPU efficiency or identified physical constants",
            "Worker critical path uses the slowest DP; rank-mean summary labels require separate aggregation",
        ]
        sources = tuple(dict.fromkeys(s for t in self.terms for s in t.source_ids))
        if observed is not None:
            if observed.profile != profile or observed.workload_id != fingerprint(
                workload
            ):
                raise ValueError(
                    "Advantage work belongs to another profile or workload"
                )
            ranks = observed.ranks
            if {r.dp_rank for r in ranks} != set(range(config.parallel.dp)) or sum(
                r.n_seqs for r in ranks
            ) != workload.n_seqs:
                raise ValueError(
                    "Advantage observations must cover all DP sequences exactly once"
                )
            if (
                sum(r.n_seqs * r.padded_length for r in ranks)
                < workload.sequence_tokens
            ):
                raise ValueError(
                    "Advantage padded shape cannot hold the attended tokens"
                )
            by_dp = {r.dp_rank: r for r in ranks}
            if workload.per_dp_sequence_lengths is not None:
                if len(workload.per_dp_sequence_lengths) != config.parallel.dp or any(
                    by_dp[i].n_seqs != len(row) or by_dp[i].padded_length < max(row)
                    for i, row in enumerate(workload.per_dp_sequence_lengths)
                ):
                    raise ValueError(
                        "Advantage shapes disagree with observed per-DP lengths"
                    )
            elif workload.sequence_lengths is not None:
                capacities = sorted(
                    r.padded_length for r in ranks for _ in range(r.n_seqs)
                )
                if any(
                    n > width
                    for n, width in zip(
                        sorted(workload.sequence_lengths), capacities, strict=True
                    )
                ):
                    raise ValueError(
                        "Advantage shapes cannot contain the observed sequence lengths"
                    )
            elif (
                workload.max_seq_len is not None
                and max(r.padded_length for r in ranks) < workload.max_seq_len
            ):
                raise ValueError(
                    "Advantage shapes cannot contain the observed maximum length"
                )
            sources += observed.source_ids
        else:
            lengths = dispatch_workload(
                workload, workload.lengths(), config.parallel.dp, dispatch
            )
            ranks = tuple(
                AdvantageRankWork(i, len(row), max(row))
                for i, row in enumerate(lengths)
            )
            notes.append(
                "Advantage rank shapes inferred from attended lengths; actual tensor padding/dispatch may differ"
            )
        rank_features = [r.features(self.device, self.gae_mode) for r in ranks]
        global_features = {
            "constant": 1.0,
            "input_bytes": observed.input_bytes if observed else None,
            "output_bytes": observed.output_bytes if observed else None,
        }
        features = [("advantage_observed_shapes", float(observed is not None))]
        for key in sorted(RANK_FEATURES - {"constant"}):
            values = [r[key] for r in rank_features]
            if all(v is not None for v in values):
                features.extend(
                    (
                        (f"advantage_{key}_sum", float(sum(values))),
                        (f"advantage_{key}_max", float(max(values))),
                    )
                )
        for key in ("input_bytes", "output_bytes"):
            if global_features[key] is not None:
                features.append((f"advantage_{key}", float(global_features[key])))
        terms = {t.name: t for t in self.terms}
        missing = [name for name in COMPONENTS if name not in terms]
        durations = {}
        for name, term in terms.items():
            values = (
                [global_features]
                if name in {"transport", "diagnostics"}
                else rank_features
            )
            predicted = [term.seconds(row) for row in values]
            if any(v is None for v in predicted):
                missing.append(name + " coefficients or input features")
            else:
                durations[name] = predicted
        if missing:
            unknown = Estimate(
                None,
                "unsupported",
                notes=tuple(notes + ["Missing advantage " + ", ".join(missing)]),
                source_ids=sources,
            )
            return AdvantageEstimate(unknown, (), tuple(features))
        workers = [
            sum(durations[name][i] for name in COMPONENTS[:3])
            for i in range(len(ranks))
        ]
        critical = max(range(len(ranks)), key=workers.__getitem__)
        components = tuple(
            (
                name,
                Estimate(
                    durations[name][0 if name in COMPONENTS[3:] else critical],
                    "assumed",
                    notes=(
                        "Uncalibrated component prior on the selected slowest-worker path; excludes phase-level calibration",
                    ),
                    source_ids=terms[name].source_ids,
                ),
            )
            for name in COMPONENTS
        )
        total = Estimate(
            sum(e.seconds for _, e in components),
            "assumed",
            notes=tuple(notes),
            source_ids=tuple(dict.fromkeys(sources)),
        )
        return AdvantageEstimate(total, components, tuple(features))

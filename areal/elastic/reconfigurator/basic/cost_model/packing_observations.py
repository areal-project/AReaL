# SPDX-License-Identifier: Apache-2.0

"""Attributed packing observations: retain partial work without inventing lengths."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from ....contracts import WorkloadStats
from ..core import UnsupportedEstimate, integer
from ._actor.config import ActorConfig
from ._actor.packing import Microbatch, PackingPlan


@dataclass(frozen=True)
class PackingSummary:
    """One DP/call's pre-shard totals, never a sum over TP/CP/PP copies.

    padding_tokens means the appended THD tail, excluding sequence alignment.
    None means unmeasured; padded_to_tokens is a request, not actual compute.
    """

    microbatch_count: int
    real_tokens: int | None = None
    compute_tokens: int | None = None
    aligned_tokens: int | None = None
    padding_tokens: int | None = None
    attention_length_squared: int | None = None
    padded_to_tokens: int | None = None
    local_microbatch_count: int | None = None

    def __post_init__(self) -> None:
        integer("microbatch_count", self.microbatch_count, minimum=1)
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                integer(f.name, value)
        if (
            self.local_microbatch_count is not None
            and not 1 <= self.local_microbatch_count <= self.microbatch_count
        ):
            raise ValueError(
                "Local microbatch count must be positive and no larger than synchronized count"
            )
        c, a, p = self.compute_tokens, self.aligned_tokens, self.padding_tokens
        if a is not None and p is not None:
            if c is not None and c != a + p:
                raise ValueError(
                    "Actual compute tokens disagree with aligned + tail padding"
                )
            c = a + p
        if c is not None and p is not None:
            a = c - p
        if c is not None and a is not None:
            p = c - a
        for name, value in (
            ("compute_tokens", c),
            ("aligned_tokens", a),
            ("padding_tokens", p),
        ):
            if value is not None:
                integer(name, value)
                object.__setattr__(self, name, value)
        real = self.real_tokens
        for name in ("compute_tokens", "aligned_tokens", "attention_length_squared"):
            value = getattr(self, name)
            if value is not None and value < self.microbatch_count:
                raise ValueError(
                    f"{name} cannot be smaller than nonempty microbatch count"
                )
        if real is not None and real < self.microbatch_count:
            raise ValueError("Every microbatch must contain real tokens")
        if real is not None and any(v is not None and v < real for v in (c, a)):
            raise ValueError(
                "Compute/aligned tokens cannot be smaller than real tokens"
            )
        square = self.attention_length_squared
        if square is not None and c is not None:
            if square < c or square > c * c or square % 2 != c % 2:
                raise ValueError(
                    "Attention square sum cannot describe these compute tokens"
                )
            if self.microbatch_count == 1 and p is not None:
                if not p * p + a <= square <= p * p + a * a:
                    raise ValueError("Attention square sum disagrees with tail padding")


def _merge(left: PackingSummary, right: PackingSummary) -> PackingSummary:
    values = {}
    for f in fields(left):
        a, b = getattr(left, f.name), getattr(right, f.name)
        if a is not None and b is not None and a != b:
            raise ValueError(f"Conflicting packing copies/summary: {f.name}")
        values[f.name] = a if a is not None else b
    return PackingSummary(**values)


def _sum(rows: tuple[PackingSummary, ...]) -> PackingSummary:
    return PackingSummary(
        **{
            f.name: None
            if any(getattr(row, f.name) is None for row in rows)
            else sum(getattr(row, f.name) for row in rows)
            for f in fields(PackingSummary)
        }
    )


@dataclass(frozen=True)
class PackingMicrobatch:
    """Final sequence lengths include the tail segment; unknown lengths stay None."""

    real_tokens: int
    compute_tokens: int | None = None
    aligned_tokens: int | None = None
    padding_tokens: int | None = None
    attention_length_squared: int | None = None
    final_sequence_lengths: tuple[int, ...] | None = None
    padded_to_tokens: int | None = None

    def __post_init__(self) -> None:
        integer("real_tokens", self.real_tokens, minimum=1)
        for name in (
            "compute_tokens",
            "aligned_tokens",
            "padding_tokens",
            "attention_length_squared",
            "padded_to_tokens",
        ):
            if getattr(self, name) is not None:
                integer(name, getattr(self, name))
        if self.final_sequence_lengths is not None:
            object.__setattr__(
                self, "final_sequence_lengths", tuple(self.final_sequence_lengths)
            )
            if not self.final_sequence_lengths:
                raise ValueError("Final attention segments cannot be empty")
            for length in self.final_sequence_lengths:
                integer("attention segment length", length, minimum=1)

    def summary(self) -> PackingSummary:
        known = PackingSummary(
            1,
            self.real_tokens,
            self.compute_tokens,
            self.aligned_tokens,
            self.padding_tokens,
            self.attention_length_squared,
            self.padded_to_tokens,
        )
        compute, attention = known.compute_tokens, known.attention_length_squared
        if self.final_sequence_lengths is not None:
            actual = sum(self.final_sequence_lengths)
            squared = sum(n * n for n in self.final_sequence_lengths)
            if (
                self.padding_tokens
                and self.final_sequence_lengths[-1] != self.padding_tokens
            ):
                raise ValueError("Tail padding must match the final attention segment")
            if compute is not None and compute != actual:
                raise ValueError(
                    "Final attention segments disagree with compute tokens"
                )
            if attention is not None and attention != squared:
                raise ValueError("Attention square sum disagrees with final segments")
            compute, attention = actual, squared
        return PackingSummary(
            1,
            self.real_tokens,
            compute,
            self.aligned_tokens,
            self.padding_tokens,
            attention,
            self.padded_to_tokens,
        )

    def resolve(self) -> Microbatch:
        work = self.summary()
        missing = [
            name
            for name in ("compute_tokens", "padding_tokens", "attention_length_squared")
            if getattr(work, name) is None
        ]
        if missing:
            raise UnsupportedEstimate(
                "Missing packing "
                + ", ".join(missing)
                + "; padded_to is not actual compute work"
            )
        return Microbatch(
            work.real_tokens,
            work.compute_tokens,
            work.attention_length_squared,
            work.padding_tokens,
        )


@dataclass(frozen=True)
class PackingRecord:
    """One attributed engine call on a physical rank.

    call_index is the PPO call for train_step, zero for recompute_logp.
    mesh=(DP,TP,CP,PP); EP overlaps that mesh. For summary-only logs pass
    microbatches=() and summary=PackingSummary(...), never fabricated MB arrays.
    """

    run_id: str
    source_revision: str
    global_step: int
    phase: str
    call_index: int
    global_rank: int
    mesh: tuple[int, int, int, int]
    microbatches: tuple[PackingMicrobatch, ...]
    source_id: str
    profile: str
    token_scope: str = "pre_shard"
    packing_mode: str = "thd"
    summary: PackingSummary | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mesh", tuple(self.mesh))
        object.__setattr__(self, "microbatches", tuple(self.microbatches))
        if not all((self.run_id, self.source_revision, self.source_id, self.profile)):
            raise ValueError(
                "Packing record requires run, source revision, source ID and profile"
            )
        for name in ("global_step", "call_index", "global_rank"):
            integer(name, getattr(self, name))
        if self.phase not in {"recompute_logp", "train_step"} or len(self.mesh) != 4:
            raise ValueError("Packing needs a known phase and full mesh")
        if not self.microbatches and self.summary is None:
            raise ValueError("Packing needs ordered microbatches or a measured summary")
        for value in self.mesh:
            integer("mesh coordinate", value)
        if self.token_scope != "pre_shard" or self.packing_mode != "thd":
            raise ValueError(
                "Observed converter supports explicitly pre-shard THD work only"
            )

    def known_work(self) -> PackingSummary:
        if not self.microbatches:
            return self.summary
        work = _sum(tuple(mb.summary() for mb in self.microbatches))
        return _merge(work, self.summary) if self.summary is not None else work


def _expected(config: ActorConfig) -> set[tuple[str, int, int]]:
    return {
        (phase, call, dp)
        for phase, count in (
            ("train_step", config.ppo_n_minibatches),
            ("recompute_logp", int(config.compute_prox_logp)),
        )
        for call in range(count)
        for dp in range(config.parallel.dp)
    }


def _attribute(config, records, profile, runtime_revision):
    if not records:
        raise UnsupportedEstimate("Missing attributed packing records")
    if len({(r.run_id, r.source_revision, r.global_step) for r in records}) != 1:
        raise ValueError("Packing records mix run, revision or step")
    if any(
        r.profile != profile or r.source_revision != runtime_revision for r in records
    ):
        raise ValueError(
            "Packing records belong to another profile or runtime revision"
        )
    limits = (
        config.parallel.dp,
        config.parallel.tp,
        config.parallel.cp,
        config.parallel.pp,
    )
    rank_mesh, mesh_rank, seen, calls = {}, {}, set(), {}
    expected = _expected(config)
    for record in records:
        if (
            any(v >= n for v, n in zip(record.mesh, limits, strict=True))
            or record.global_rank >= config.parallel.world_size
        ):
            raise ValueError("Packing rank/mesh lies outside the configured world")
        if (
            rank_mesh.setdefault(record.global_rank, record.mesh) != record.mesh
            or mesh_rank.setdefault(record.mesh, record.global_rank)
            != record.global_rank
        ):
            raise ValueError("Inconsistent global rank to mesh mapping")
        key = (record.phase, record.call_index, record.global_rank)
        if key in seen:
            raise ValueError(
                "Duplicate physical rank/call record; attribute repeated invocations first"
            )
        seen.add(key)
        key = (record.phase, record.call_index, record.mesh[0])
        if key not in expected:
            raise ValueError(f"Unexpected packing phase/call/DP record: {key}")
        calls.setdefault(key, []).append(record)
    return calls


@dataclass(frozen=True)
class PackingError:
    phase: str
    call_index: int
    dp_rank: int
    metric: str
    predicted: int | None
    observed: int | None

    @property
    def signed_error(self) -> int | None:
        return (
            None
            if self.observed is None or self.predicted is None
            else self.predicted - self.observed
        )


@dataclass(frozen=True)
class PackingComparison:
    """Rows retain missing observations; metric WAPE uses only known paired work."""

    rows: tuple[PackingError, ...]

    def metric(self, name: str) -> tuple[float | None, int, int]:
        """Return (WAPE, covered cells, expected cells); zero denominator -> None."""
        rows = [row for row in self.rows if row.metric == name]
        known = [
            row
            for row in rows
            if row.observed is not None and row.predicted is not None
        ]
        total = sum(row.observed for row in known)
        return (
            sum(abs(row.signed_error) for row in known) / total if total else None,
            len(known),
            len(rows),
        )


def _plan_calls(plan: PackingPlan):
    result = {
        ("train_step", call, dp): rank
        for call, ranks in enumerate(plan.train)
        for dp, rank in enumerate(ranks)
    }
    result.update(
        {("recompute_logp", 0, dp): rank for dp, rank in enumerate(plan.logprob)}
    )
    return result


@dataclass(frozen=True)
class ObservedPackingWork:
    """Partial evidence for one workload/step, separate from an exact PackingPlan."""

    config: ActorConfig
    workload: WorkloadStats
    profile: str
    runtime_revision: str
    records: tuple[PackingRecord, ...]
    calls: tuple[tuple[tuple[str, int, int], PackingSummary], ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        attributed = _attribute(
            self.config, self.records, self.profile, self.runtime_revision
        )
        calls = {}
        for key, copies in attributed.items():
            merged = copies[0].known_work()
            per_mb = None
            for record in copies:
                merged = _merge(merged, record.known_work())
                if record.microbatches:
                    work = tuple(mb.summary() for mb in record.microbatches)
                    if per_mb is not None:
                        if len(per_mb) != len(work):
                            raise ValueError("Conflicting TP/CP/PP microbatch counts")
                        work = tuple(
                            _merge(a, b) for a, b in zip(per_mb, work, strict=True)
                        )
                    per_mb = work
            # Complementary replica fields can establish additional known totals.
            calls[key] = _merge(merged, _sum(per_mb)) if per_mb else merged
        expected = _expected(self.config)
        if any(
            row.real_tokens is not None
            and row.real_tokens
            > (row.local_microbatch_count or row.microbatch_count)
            * self.config.max_tokens_per_mb
            for row in calls.values()
        ):
            raise ValueError(
                "Observed rank tokens exceed synchronized microbatch capacity"
            )
        for phase in {key[0] for key in expected}:
            keys = {key for key in expected if key[0] == phase}
            known = [calls[k].real_tokens for k in keys & calls.keys()]
            measured = sum(n for n in known if n is not None)
            if measured > self.workload.sequence_tokens + 1:
                raise ValueError("Observed packing real tokens exceed logical workload")
            if keys <= calls.keys() and all(n is not None for n in known):
                if abs(measured - self.workload.sequence_tokens) > 1:
                    raise ValueError(
                        "Observed packing real tokens do not conserve logical workload"
                    )
            for call in {k[1] for k in keys}:
                counts = {
                    v.microbatch_count
                    for k, v in calls.items()
                    if k[:2] == (phase, call)
                }
                if len(counts) > 1:
                    raise ValueError(
                        "Observed counts must be post-synchronization across DP ranks"
                    )
        object.__setattr__(self, "calls", tuple(sorted(calls.items())))

    @property
    def missing_calls(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(sorted(_expected(self.config) - dict(self.calls).keys()))

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(r.source_id for r in self.records))

    def phase_microbatches(self, phase: str) -> int:
        keys = {k for k in _expected(self.config) if k[0] == phase}
        calls = dict(self.calls)
        if not keys or not keys <= calls.keys():
            raise UnsupportedEstimate(
                f"Missing packing phase/call/DP coverage: {phase}"
            )
        # All DP counts are synchronized; neither DP nor model-parallel copies add.
        return sum(
            calls[(phase, call, 0)].microbatch_count for call in {k[1] for k in keys}
        )

    def compare(self, predicted: PackingPlan) -> PackingComparison:
        """Caller supplies a prediction for this same config/workload."""
        from .actor import ExpertActorModel

        if (
            self.workload.train_microbatches is not None
            or predicted.count_source != "predicted"
        ):
            raise ValueError(
                "Packing comparison requires independently predicted counts; remove train_microbatches from workload"
            )
        ExpertActorModel._check_plan(self.config, self.workload, predicted)
        observed = dict(self.calls)
        traces = {(r.phase, r.call_index, r.dp_rank): r for r in predicted.rank_work}
        rows = []
        for key, rank in sorted(_plan_calls(predicted).items()):
            values = PackingSummary(
                len(rank),
                sum(m.real_tokens for m in rank),
                sum(m.compute_tokens for m in rank),
                sum(m.compute_tokens - m.padding_tokens for m in rank),
                sum(m.padding_tokens for m in rank),
                sum(m.attention_length_squared for m in rank),
            )
            for name in (
                "microbatch_count",
                "real_tokens",
                "compute_tokens",
                "aligned_tokens",
                "padding_tokens",
                "attention_length_squared",
            ):
                rows.append(
                    PackingError(
                        *key,
                        name,
                        getattr(values, name),
                        getattr(observed[key], name) if key in observed else None,
                    )
                )
            measured = observed.get(key)
            local = traces.get(key)
            rows.extend(
                (
                    PackingError(
                        *key,
                        "capacity_microbatch_lower_bound",
                        (
                            sum(m.real_tokens for m in rank)
                            + self.config.max_tokens_per_mb
                            - 1
                        )
                        // self.config.max_tokens_per_mb,
                        None
                        if measured is None or measured.real_tokens is None
                        else (measured.real_tokens + self.config.max_tokens_per_mb - 1)
                        // self.config.max_tokens_per_mb,
                    ),
                    PackingError(
                        *key,
                        "local_microbatch_count",
                        local.local_microbatches if local else None,
                        measured.local_microbatch_count if measured else None,
                    ),
                )
            )
        return PackingComparison(tuple(rows))

    def to_exact_plan(self) -> PackingPlan:
        return observed_plan(
            self.config,
            self.workload,
            self.records,
            profile=self.profile,
            runtime_revision=self.runtime_revision,
        )[0]


def observed_plan(
    config: ActorConfig,
    workload: WorkloadStats,
    records: tuple[PackingRecord, ...],
    *,
    profile: str,
    runtime_revision: str,
) -> tuple[PackingPlan, tuple[str, ...]]:
    work = ObservedPackingWork(config, workload, profile, runtime_revision, records)
    if work.missing_calls:
        raise UnsupportedEstimate(
            f"Missing packing phase/call/DP coverage: {work.missing_calls}"
        )
    calls = {}
    for key, copies in _attribute(config, records, profile, runtime_revision).items():
        exact = [
            tuple(mb.resolve() for mb in r.microbatches)
            for r in copies
            if r.microbatches
        ]
        if not exact:
            raise UnsupportedEstimate("Missing per-microbatch work for exact packing")
        if any(row != exact[0] for row in exact):
            raise ValueError("Conflicting TP/CP/PP copies of the same DP packing call")
        calls[key] = exact[0]
    train = tuple(
        tuple(calls[("train_step", call, dp)] for dp in range(config.parallel.dp))
        for call in range(config.ppo_n_minibatches)
    )
    logprob = (
        tuple(calls[("recompute_logp", 0, dp)] for dp in range(config.parallel.dp))
        if config.compute_prox_logp
        else ()
    )
    plan = PackingPlan(
        logprob,
        train,
        True,
        True,
        (
            "Observed pre-shard microbatch work; TP/CP/PP copies de-duplicated, DP work retained",
        ),
        "observed",
    )
    from .actor import ExpertActorModel

    ExpertActorModel._check_plan(config, workload, plan)
    return plan, work.source_ids

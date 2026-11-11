# SPDX-License-Identifier: Apache-2.0

"""Proposal requests, revision reports and the Revision Benchmark contract."""

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from areal.utils.atomic import json_digest as digest

Target = str


def source_digest(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


@dataclass(frozen=True)
class Proposal:
    source: str
    rationale: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Candidate source is empty")
        if len(self.source.encode()) > 128_000:
            raise ValueError("Candidate source exceeds 128 KB")


@dataclass(frozen=True)
class ProposalRequest:
    target: Target
    round: int
    incumbent_source: str
    incumbent_sha256: str
    calibration: dict
    settings: dict
    feedback: tuple[dict, ...]
    previous_source: str | None = None
    context: dict = field(default_factory=dict)
    revision_history: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return {
            **{
                k: v
                for k, v in self.context.items()
                if k not in {"reference_sources", "instructions"}
            },
            **asdict(self),
            "response_schema": {
                "source": "Python module string",
                "rationale": "short explanation string",
            },
        }


@dataclass(frozen=True)
class ReflectionRequest:
    """One completed attempt and its public, program-owned evaluation feedback."""

    observation: ProposalRequest
    proposal: Proposal
    baseline: dict
    outcome: dict
    remaining_rounds: int | None = None

    def to_dict(self) -> dict:
        observation = self.observation.to_dict()
        observation.pop("response_schema")
        return {
            "task": "reflect",
            "observation": observation,
            "proposal": {
                "source": self.proposal.source,
                "rationale": self.proposal.rationale,
            },
            "baseline": self.baseline,
            "outcome": self.outcome,
            "remaining_rounds": self.remaining_rounds,
        }


@dataclass(frozen=True)
class RevisionInsight:
    """A short Agent-authored lesson; objective metrics remain in the reports."""

    hypothesis: str
    change: str
    finding: str
    evidence: tuple[str, ...]
    scope: str
    unresolved: str = ""
    metadata: dict = field(default_factory=dict)
    next_action: Literal["continue", "finish"] = "continue"
    next_action_reason: str = ""

    def __post_init__(self) -> None:
        for name in (
            "hypothesis",
            "change",
            "finding",
            "scope",
            "unresolved",
            "next_action_reason",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) > 800:
                raise ValueError(
                    f"Insight {name} must be a string of at most 800 characters"
                )
            if name not in {"unresolved", "next_action_reason"} and not value.strip():
                raise ValueError(f"Insight {name} must be nonempty")
        if self.next_action not in {"continue", "finish"}:
            raise ValueError("Insight next_action must be continue or finish")
        if self.next_action == "finish" and not self.next_action_reason.strip():
            raise ValueError("Finishing revision requires a short reason")
        if (
            not isinstance(self.evidence, list | tuple)
            or not 1 <= len(self.evidence) <= 4
        ):
            raise ValueError("Insight needs one to four short evidence statements")
        if any(
            not isinstance(item, str) or not item.strip() or len(item) > 300
            for item in self.evidence
        ):
            raise ValueError(
                "Insight evidence must be nonempty strings of at most 300 characters"
            )
        object.__setattr__(self, "evidence", tuple(self.evidence))


@dataclass(frozen=True)
class RevisionReport:
    target: Target
    candidate_sha256: str
    benchmark_sha256: str
    valid: bool
    score: float | None
    metrics: dict
    errors: tuple[str, ...] = ()


__all__ = [
    "digest",
    "source_digest",
    "Proposal",
    "ProposalRequest",
    "ReflectionRequest",
    "RevisionInsight",
    "RevisionReport",
    "RevisionBenchmark",
    "Target",
]


class RevisionBenchmark(Protocol):
    target: str
    benchmark_sha256: str
    initial_source: str | None
    owns_acceptance: bool
    replay_fingerprints: tuple[str, ...]
    criteria: Any
    context: dict
    settings: dict
    calibration: dict

    def evaluate(self, source: str, directory: Path) -> RevisionReport: ...
    def compare(
        self, candidate: RevisionReport, incumbent: RevisionReport
    ) -> tuple[bool, str]: ...
    def feedback(self, report: RevisionReport) -> dict: ...

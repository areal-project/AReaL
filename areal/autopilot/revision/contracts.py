# SPDX-License-Identifier: Apache-2.0

"""Proposal requests, revision reports and the Revision Benchmark contract."""

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

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
    proposal_context: dict

    def evaluate(self, source: str, directory: Path) -> RevisionReport: ...
    def compare(
        self, candidate: RevisionReport, incumbent: RevisionReport
    ) -> tuple[bool, str]: ...
    def feedback(self, report: RevisionReport) -> dict: ...

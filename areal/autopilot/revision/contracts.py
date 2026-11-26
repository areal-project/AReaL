# SPDX-License-Identifier: Apache-2.0

"""Proposal and evaluation contracts for offline revision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Proposal:
    settings: dict
    rationale: str = ""

    def __post_init__(self):
        if not isinstance(self.settings, dict) or not isinstance(self.rationale, str):
            raise ValueError("Expected a settings object and a rationale string")
        if len(self.rationale) > 2000:
            raise ValueError("Proposal rationale exceeds 2000 characters")


@dataclass(frozen=True)
class ProposalRequest:
    round: int
    incumbent: dict
    feedback: tuple[dict, ...]
    settings_schema: dict


@dataclass(frozen=True)
class Reward:
    accepted: bool
    reason: str
    metrics: dict[str, float | int]


class RevisionEvaluator(Protocol):
    context_sha256: str
    settings_schema: dict

    def normalize(self, settings: dict) -> dict: ...

    def evaluate(self, settings: dict) -> dict: ...

    def reward(self, candidate: dict, incumbent: dict) -> Reward: ...


class RevisionAgent(Protocol):
    def propose(self, request: ProposalRequest) -> Proposal: ...

# SPDX-License-Identifier: Apache-2.0

"""Domain-independent proposal, evaluation orchestration and revision storage."""

from areal.autopilot.revision.agent import (
    CommandRevisionAgent,
    LLMRevisionAgent,
    RevisionAgent,
)
from areal.autopilot.revision.contracts import (
    Proposal,
    ProposalRequest,
    ReflectionRequest,
    RevisionBenchmark,
    RevisionInsight,
    RevisionReport,
)
from areal.autopilot.revision.execution import CandidateRunner
from areal.autopilot.revision.loop import RevisionLoop, RevisionStore

__all__ = [
    "CommandRevisionAgent",
    "RevisionAgent",
    "Proposal",
    "ProposalRequest",
    "ReflectionRequest",
    "CandidateRunner",
    "RevisionBenchmark",
    "RevisionInsight",
    "RevisionReport",
    "LLMRevisionAgent",
    "RevisionLoop",
    "RevisionStore",
]

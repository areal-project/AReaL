# SPDX-License-Identifier: Apache-2.0

"""Elastic implementation of the generic evaluator and subprocess ports."""

from dataclasses import replace

from areal.autopilot.revision.execution import CandidateRunner

from .contracts import BASELINE_SOURCE, proposal_context


def candidate_runner(runner=None):
    runner = runner or CandidateRunner()
    if (
        not isinstance(runner, CandidateRunner)
        or runner.command
        or runner.worker_module is not None
    ):
        return runner
    return replace(runner, worker_module="areal.autopilot.domains.elastic.worker")


class ElasticEvaluator:
    owns_acceptance = True

    @property
    def initial_source(self):
        return BASELINE_SOURCE.get(self.target)

    @property
    def proposal_context(self):
        return proposal_context(self.target)

    @property
    def replay_fingerprints(self):
        return (
            ("prediction_sha256",) if self.target == "model" else ("selection_sha256",)
        )

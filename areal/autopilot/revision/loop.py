# SPDX-License-Identifier: Apache-2.0

"""Propose code revisions, run the Revision Benchmark and store accepted versions."""

import fcntl
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path

from areal.autopilot.revision.agent import RevisionAgent
from areal.autopilot.revision.contracts import (
    ProposalRequest,
    RevisionBenchmark,
    digest,
    source_digest,
)
from areal.utils.atomic import atomic_json


class RevisionStore:
    """One target and one fixed evaluation context per store; atomic active pointer."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def active(self) -> tuple[Path, dict] | None:
        pointer = self.root / "active.json"
        if not pointer.exists():
            return None
        value = json.loads(pointer.read_text())
        path = (self.root / value["source"]).resolve()
        if (
            not path.is_relative_to(self.root)
            or not path.is_file()
            or source_digest(path.read_text()) != value["source_sha256"]
        ):
            raise ValueError("Accepted revision source is missing or changed")
        report = json.loads((path.parent / "evaluation.json").read_text())
        acceptance = json.loads((path.parent / "acceptance.json").read_text())
        context = json.loads((self.root / "context.json").read_text())
        if (
            acceptance.get("accepted") is not True
            or acceptance.get("candidate_sha256") != value["source_sha256"]
            or acceptance.get("benchmark_sha256") != report["benchmark_sha256"]
            or report["target"] != context["target"]
            or not report["valid"]
            or report["candidate_sha256"] != value["source_sha256"]
            or report["benchmark_sha256"] != context["benchmark_sha256"]
            or value["benchmark_sha256"] != context["benchmark_sha256"]
        ):
            raise ValueError("Accepted revision does not match its evaluation")
        return path, value

    def promote(self, path: Path, report: dict) -> None:
        if (
            not report["valid"]
            or source_digest(path.read_text()) != report["candidate_sha256"]
        ):
            raise ValueError("Only a verified candidate can be promoted")
        context = json.loads((self.root / "context.json").read_text())
        acceptance = json.loads((path.parent / "acceptance.json").read_text())
        stored_report = json.loads((path.parent / "evaluation.json").read_text())
        active = self.active()
        parent_sha = (
            active[1]["source_sha256"]
            if active
            else json.loads((self.root / "baseline.json").read_text())[
                "candidate_sha256"
            ]
        )
        if (
            acceptance.get("accepted") is not True
            or acceptance.get("parent_sha256") != parent_sha
            or acceptance.get("candidate_sha256") != report["candidate_sha256"]
            or acceptance.get("benchmark_sha256") != context["benchmark_sha256"]
            or digest(stored_report) != digest(report)
        ):
            raise ValueError("Candidate has no matching acceptance receipt")
        # evaluation.json is durable before active points at this generation.
        descriptor = os.open(path.parent.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        atomic_json(
            self.root / "active.json",
            {
                "source": str(path.relative_to(self.root)),
                "source_sha256": report["candidate_sha256"],
                "target": report["target"],
                "benchmark_sha256": report["benchmark_sha256"],
                "score": report["score"],
            },
        )


class RevisionLoop:
    def __init__(
        self,
        agent: RevisionAgent,
        benchmark: RevisionBenchmark,
        root: Path,
        *,
        criteria=None,
        initial_source: str | None = None,
    ):
        self.agent, self.benchmark = agent, benchmark
        self.store = RevisionStore(root)
        self.initial_source = (
            initial_source if initial_source is not None else benchmark.initial_source
        )
        if not isinstance(self.initial_source, str) or not self.initial_source.strip():
            raise ValueError(
                "Component revisions require an explicit initial_source factory"
            )
        self._benchmark_owns_acceptance = benchmark.owns_acceptance
        if self._benchmark_owns_acceptance:
            if criteria is not None and criteria != benchmark.criteria:
                raise ValueError(
                    "Acceptance criteria are frozen by the Revision Benchmark"
                )
            self.criteria = benchmark.criteria
        else:
            self.criteria = criteria or benchmark.criteria

    def _public_feedback(self, outcome: dict) -> dict:
        if not self._benchmark_owns_acceptance:
            return outcome
        return {
            **outcome.get("feedback", {}),
            **{
                k: outcome[k]
                for k in (
                    "round",
                    "status",
                    "reason",
                    "parent_sha256",
                    "candidate_sha256",
                )
                if k in outcome
            },
        }

    def run(self, *, max_rounds: int = 3, stop_after_accept: bool = True) -> dict:
        if type(max_rounds) is not int or max_rounds < 1:
            raise ValueError("Revision rounds must be positive")
        with (self.store.root / "revision.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._run(max_rounds, stop_after_accept)

    @property
    def context(self) -> dict:
        context = {
            "format_version": 1,
            "target": self.benchmark.target,
            "benchmark_sha256": self.benchmark.benchmark_sha256,
            "deployment_context": self.benchmark.context,
            "criteria": asdict(self.criteria),
        }
        if self._benchmark_owns_acceptance:
            context["initial_sha256"] = source_digest(
                self.initial_source
                if self.initial_source is not None
                else self.benchmark.initial_source
            )
        return context

    def _run(self, max_rounds: int, stop_after_accept: bool) -> dict:
        root, context = self.store.root, self.context
        if (root / "context.json").exists():
            if json.loads((root / "context.json").read_text()) != context:
                raise ValueError(
                    "Revision store belongs to another fixed evaluation context"
                )
        else:
            atomic_json(root / "context.json", context)
        attempts = root / "attempts"
        attempts.mkdir(exist_ok=True)
        active = self.store.active()
        source = (
            active[0].read_text()
            if active
            else (
                self.initial_source
                if self.initial_source is not None
                else self.benchmark.initial_source
            )
        )
        initial_sha256 = source_digest(source)
        baseline = self.benchmark.evaluate(
            source, root / f"baseline-{uuid.uuid4().hex}"
        )
        previous_report = (
            active[0].parent / "evaluation.json"
            if active is not None
            else root / "baseline.json"
        )
        if self._benchmark_owns_acceptance and previous_report.exists():
            accepted_report = json.loads(previous_report.read_text())
            fingerprints = self.benchmark.replay_fingerprints
            if accepted_report["valid"] and (
                not baseline.valid
                or baseline.candidate_sha256 != accepted_report["candidate_sha256"]
                or baseline.benchmark_sha256 != accepted_report["benchmark_sha256"]
                or any(
                    baseline.metrics.get(key) != accepted_report["metrics"].get(key)
                    for key in fingerprints
                )
            ):
                raise ValueError(
                    "Accepted component predictions/selections changed since acceptance"
                    if active is not None
                    else "Incumbent predictions/selections changed since baseline evaluation"
                )
        atomic_json(root / "baseline.json", asdict(baseline))
        if not baseline.valid:
            raise ValueError(f"Incumbent failed evaluation: {baseline.errors}")
        previous = []
        previous_source = None
        for directory in sorted(attempts.glob("round-*")):
            if (directory / "outcome.json").exists():
                previous.append(json.loads((directory / "outcome.json").read_text()))
                if (directory / "proposal.json").exists():
                    previous_source = json.loads(
                        (directory / "proposal.json").read_text()
                    )["source"]
        seen = {initial_sha256, *(item.get("candidate_sha256") for item in previous)}
        baseline_feedback = (
            self.benchmark.feedback(baseline)
            if self._benchmark_owns_acceptance
            else {"score": baseline.score, "metrics": baseline.metrics}
        )
        feedback = [
            {"status": "incumbent", **baseline_feedback},
            *(self._public_feedback(o) for o in previous[-3:]),
        ]
        outcomes = []
        for _ in range(max_rounds):
            round_number = len(list(attempts.glob("round-*"))) + 1
            directory = attempts / f"round-{round_number:04d}-{uuid.uuid4().hex[:8]}"
            directory.mkdir()
            request = ProposalRequest(
                self.benchmark.target,
                round_number,
                source,
                source_digest(source),
                self.benchmark.calibration,
                json.loads(json.dumps(self.benchmark.settings)),
                tuple(feedback[-4:]),
                previous_source=previous_source if previous_source != source else None,
                context=self.benchmark.proposal_context,
            )
            atomic_json(directory / "proposal-request.json", request.to_dict())
            outcome = {
                "round": round_number,
                "status": "rejected",
                "parent_sha256": source_digest(source),
            }
            try:
                proposal = self.agent.propose(request)
            except Exception as error:
                outcome.update(reason=f"{type(error).__name__}: {error}")
            else:
                try:
                    previous_source = proposal.source
                    candidate_sha = source_digest(proposal.source)
                    outcome["candidate_sha256"] = candidate_sha
                    atomic_json(directory / "proposal.json", asdict(proposal))
                    if candidate_sha in seen:
                        outcome["reason"] = "Duplicate candidate"
                    else:
                        seen.add(candidate_sha)
                        report = self.benchmark.evaluate(proposal.source, directory)
                        atomic_json(directory / "evaluation.json", asdict(report))
                        accepted, reason = (
                            self.benchmark.compare(report, baseline)
                            if self._benchmark_owns_acceptance
                            else self.criteria.compare(report, baseline)
                        )
                        outcome.update(
                            reason=reason,
                            score=report.score,
                            metrics=report.metrics,
                            errors=list(report.errors),
                        )
                        if self._benchmark_owns_acceptance:
                            outcome["feedback"] = self.benchmark.feedback(report)
                        if accepted:
                            path = directory / "candidate.py"
                            # Runner materializes the exact source that was evaluated.
                            if source_digest(path.read_text()) != candidate_sha:
                                raise ValueError(
                                    "Candidate source changed during evaluation"
                                )
                            with path.open("rb") as stream:
                                os.fsync(stream.fileno())
                            atomic_json(
                                directory / "acceptance.json",
                                {
                                    "accepted": True,
                                    "parent_sha256": outcome["parent_sha256"],
                                    "candidate_sha256": candidate_sha,
                                    "benchmark_sha256": report.benchmark_sha256,
                                },
                            )
                            self.store.promote(path, asdict(report))
                            source, baseline = proposal.source, report
                            outcome["status"] = "accepted"
                except OSError:
                    raise
                except Exception as error:
                    outcome.update(reason=f"{type(error).__name__}: {error}")
            atomic_json(directory / "outcome.json", outcome)
            feedback.append(self._public_feedback(outcome))
            outcomes.append(outcome)
            if outcome["status"] == "accepted" and stop_after_accept:
                break
        active = self.store.active()
        result = {
            "target": self.benchmark.target,
            "benchmark_sha256": self.benchmark.benchmark_sha256,
            "initial_sha256": initial_sha256,
            "incumbent_sha256": source_digest(source),
            "active_source": None if active is None else str(active[0]),
            "accepted": sum(x["status"] == "accepted" for x in outcomes),
            "rounds": outcomes,
            "final_score": baseline.score,
            "final_metrics": baseline.metrics,
        }
        atomic_json(root / "run.json", result)
        return result

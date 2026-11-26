# SPDX-License-Identifier: Apache-2.0

"""Run bounded offline revisions and persist one accepted version and its evidence."""

from __future__ import annotations

import fcntl
import json
import os
from dataclasses import asdict
from pathlib import Path

from areal.utils.fs import atomic_write_json

from .contracts import ProposalRequest, RevisionAgent, RevisionEvaluator, digest


class RevisionStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = root / "revision.json"

    def load(self) -> dict | None:
        return json.loads(self.path.read_text()) if self.path.exists() else None

    def save(self, state: dict) -> None:
        # Keep the active pointer and its evaluation in the same atomic write.
        atomic_write_json(self.path, state, default=None, allow_nan=False)
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @staticmethod
    def incumbent(state: dict) -> dict:
        active = state["accepted_round"]
        entry = state["reference"] if active is None else state["history"][active - 1]
        if digest(entry["settings"]) != entry["settings_sha256"] or (
            active is not None
            and (entry["round"] != active or not entry["reward"]["accepted"])
        ):
            raise ValueError("Active revision does not match its acceptance record")
        return entry


class RevisionLoop:
    def __init__(
        self,
        agent: RevisionAgent,
        evaluator: RevisionEvaluator,
        root: Path,
        *,
        reference: dict,
    ):
        self.agent, self.evaluator = agent, evaluator
        self.store = RevisionStore(root)
        self.reference = evaluator.normalize(reference)
        self.context = digest(
            {"evaluation": evaluator.context_sha256, "reference": self.reference}
        )

    def run(self, *, max_rounds: int) -> dict:
        """Apply a total budget across invocations, including rejected proposals."""
        if type(max_rounds) is not int or max_rounds < 1:
            raise ValueError("Revision rounds must be a positive integer")
        with (self.store.root / "revision.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._run(max_rounds)

    def _run(self, max_rounds: int) -> dict:
        state = self.store.load()
        if state is None:
            reference = {
                "settings": self.reference,
                "settings_sha256": digest(self.reference),
                "evaluation": self.evaluator.evaluate(self.reference),
            }
            state = {
                "schema_version": 1,
                "context_sha256": self.context,
                "reference": reference,
                "accepted_round": None,
                "pending": None,
                "history": [],
            }
            self.store.save(state)
        elif state["schema_version"] != 1 or state["context_sha256"] != self.context:
            raise ValueError("Saved revision belongs to a different evaluation context")
        else:
            incumbent = self.store.incumbent(state)
            if (
                self.evaluator.evaluate(incumbent["settings"])
                != incumbent["evaluation"]
            ):
                raise ValueError("Active policy did not reproduce its saved evaluation")

        history = state["history"]
        if len(history) > max_rounds or (
            len(history) == max_rounds and state["pending"]
        ):
            raise ValueError("Saved attempts exceed the requested total budget")
        while len(history) < max_rounds:
            incumbent = self.store.incumbent(state)
            if state["pending"] is None:
                state["pending"] = self._propose(len(history) + 1, incumbent, history)
                self.store.save(state)
            attempt = self._evaluate(state["pending"], incumbent, state)
            history.append(attempt)
            if attempt["status"] == "accepted":
                state["accepted_round"] = attempt["round"]
            state["pending"] = None
            self.store.save(state)
        return state

    def _propose(self, round_number: int, incumbent: dict, history: list) -> dict:
        attempt = {"round": round_number, "parent_sha256": incumbent["settings_sha256"]}
        request = ProposalRequest(
            round_number,
            incumbent["settings"],
            tuple(
                {
                    "round": row["round"],
                    "status": row["status"],
                    "settings": row.get("settings"),
                    "feedback": row.get("reward", row.get("reason")),
                }
                for row in history[-4:]
            ),
            self.evaluator.settings_schema,
        )
        try:
            proposal = self.agent.propose(request)
            attempt["rationale"] = proposal.rationale
            settings = self.evaluator.normalize(proposal.settings)
            attempt.update(settings=settings, settings_sha256=digest(settings))
        except (TypeError, ValueError, RuntimeError, OSError) as error:
            attempt["reason"] = f"{type(error).__name__}: {error}"
        return attempt

    def _evaluate(self, attempt: dict, incumbent: dict, state: dict) -> dict:
        attempt = dict(attempt, status="rejected")
        if "settings" not in attempt:
            return attempt
        identities = {row.get("settings_sha256") for row in state["history"]}
        identities.add(state["reference"]["settings_sha256"])
        if attempt["settings_sha256"] in identities:
            return dict(attempt, reason="Duplicate settings")
        try:
            evaluated = self.evaluator.evaluate(attempt["settings"])
            attempt["evaluation"] = evaluated
            reward = self.evaluator.reward(evaluated, incumbent["evaluation"])
        except ValueError as error:
            return dict(attempt, reason=str(error))
        attempt["reward"] = asdict(reward)
        if reward.accepted:
            attempt["status"] = "accepted"
        return attempt

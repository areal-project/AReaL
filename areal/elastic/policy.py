# SPDX-License-Identifier: Apache-2.0

"""Small event-driven lifecycle policy; no configuration search or simulator."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from typing import Any

from areal.utils import logging
from areal.utils.atomic import atomic_json

logger = logging.getLogger("RLTrainer")


def validate_policy(policy: Any) -> None:
    if policy is not None and (
        not isinstance(policy, LifecyclePolicy) or policy.owner is not None
    ):
        raise ValueError("Pass a fresh, unbound LifecyclePolicy")


class LifecyclePolicy:
    """Cheap polling discovers events; only decision events call the LLM.

    ``manual`` waits for the local control CLI. ``idle`` is an explicit normal-path
    test/baseline policy. ``agent`` requires a real decision callback and never
    silently substitutes a rule on API failure.
    """

    def __init__(
        self,
        *,
        mode: str = "manual",
        decide: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        observe: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        if mode not in {"manual", "agent", "idle", "resident"}:
            raise ValueError("Unknown lifecycle policy")
        if mode == "agent" and decide is None:
            raise ValueError("Agent mode requires an LLM decision callback")
        self.mode, self.decision_fn = mode, decide
        self.observer = observe
        self.owner = None
        self._last_event = None
        self._last_progress = None
        self._last_decision_at = float("-inf")
        self._prestarted_version = None
        self._wake_at = 0.0
        self._sequence = 0

    def bind(self, owner: Any) -> None:
        if self.owner is not None:
            raise ValueError("A policy cannot belong to two coordinators")
        self.owner = owner
        (owner.root / "control/requests").mkdir(parents=True, exist_ok=True)
        (owner.root / "control/responses").mkdir(parents=True, exist_ok=True)

    def _record(self, record: dict[str, Any]) -> None:
        with (self.owner.root / "decisions.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def observation(self, *, now: float | None = None) -> dict[str, Any]:
        status = self.owner.get_status(now=now)
        if self.observer is not None:
            status["workload"] = self.observer(status)
        if self.mode == "agent":
            status["allowed_actions"] = ["wait_for_event"]
            if not status.get("training_active", False):
                status["allowed_actions"].append(
                    "start_actor" if self.owner.actor is None else "stop_actor"
                )
        return status

    @staticmethod
    def _decision_error(decision: Any, observation: dict[str, Any]) -> str | None:
        if not isinstance(decision, dict):
            return "Return a tool action with a short reason"
        if decision.get("action") not in observation["allowed_actions"]:
            return (
                "Action is unavailable; choose from allowed_actions in current status"
            )
        reason = decision.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return "Give a short decision reason"
        wait = decision.get("wake_after_seconds", 30.0)
        if (
            isinstance(wait, bool)
            or not isinstance(wait, (int, float))
            or not math.isfinite(wait)
            or not 5 <= wait <= 300
        ):
            return "Wake delay must be between 5 and 300 seconds"
        return None

    def decide(self, batch: Any, *, now: float | None = None) -> str:
        owner = self.owner
        now = time.monotonic() if now is None else now
        if batch is not None and owner.actor is not None:
            return "train"
        if self.mode == "resident":
            return "train" if batch is not None else "keep"
        if self.mode == "manual":
            return "keep"
        status = self.observation(now=now)
        if self.mode == "idle":
            if batch is not None:
                return "train"
            return (
                "release"
                if owner.actor is not None
                and status["idle_seconds"] >= owner.options.idle_timeout
                and status["residence_seconds"] >= owner.options.min_ready_seconds
                else "keep"
            )

        # A speculative start commits this generation until the next update.
        # Do not immediately undo it when initialization changes the event key.
        if (
            owner.actor is not None
            and self._prestarted_version == status["applied_version"]
        ):
            return "keep"

        signature = (
            owner.actor is not None,
            status["generation"],
            status["applied_version"],
            getattr(batch, "batch_id", None),
        )
        progress = (
            status["counts"].get("ready", 0),
            status["counts"].get("admitted", 0),
            status.get("workload", {}).get("event_signature"),
        )
        if signature != self._last_event:
            if batch is not None:
                event = "batch_ready"
            elif owner.actor is not None:
                event = "update_completed"
            else:
                event = (
                    "collection_started"
                    if status["generation"] == 0
                    else "actor_released"
                )
        elif progress != self._last_progress and now - self._last_decision_at >= 5:
            event = "progress_changed"
        elif now >= self._wake_at:
            event = "scheduled_wake"
        else:
            return "keep"
        observation = {
            **status,
            "event": event,
            "batch_ready": batch is not None,
            "batch_id": getattr(batch, "batch_id", None),
        }
        for attempt in range(1, 3):
            started = time.monotonic()
            # API/transport and coordinator failures still propagate. Only invalid
            # action proposals are rejected, logged and offered one correction.
            decision = self.decision_fn(observation)
            error = self._decision_error(decision, observation)
            self._sequence += 1
            self._record(
                {
                    "sequence": self._sequence,
                    "wall_time": time.time(),
                    "monotonic": time.monotonic(),
                    "mode": self.mode,
                    "observation": observation,
                    "decision": decision,
                    "decision_seconds": time.monotonic() - started,
                    "attempt": attempt,
                    "outcome": "rejected" if error else "accepted",
                    "error": error,
                    "retry_after_seconds": 5 if error and attempt == 2 else None,
                }
            )
            if error is None:
                break
            logger.warning("Elastic agent: rejected attempt=%s: %s", attempt, error)
            if attempt == 1:
                observation = {
                    **observation,
                    **self.observation(),
                    "event": "action_rejected",
                    "rejected_decision": decision,
                    "rejection_reason": error,
                }
        # Repeated invalid proposals preserve the current Actor state. Yield to
        # the coordinator so ready batches can train, then retry on an event or
        # after five seconds; this is not an LLM-selected lifecycle action.
        wait = 5.0 if error else decision.get("wake_after_seconds", 30.0)
        self._last_event = signature
        self._last_progress = progress
        self._last_decision_at = time.monotonic()
        self._wake_at = self._last_decision_at + wait
        if error:
            return "keep"
        action, reason = decision["action"], decision["reason"]
        if action == "start_actor" and batch is None:
            self._prestarted_version = status["applied_version"]
        logger.info(
            "Elastic agent: event=%s action=%s reason=%s", event, action, reason
        )
        return {
            "start_actor": "train" if batch is not None else "start",
            "stop_actor": "release",
        }.get(action, "keep")

    def poll_commands(self) -> None:
        """Execute local commands serially on the training coordinator thread."""
        owner = self.owner
        if owner is None:
            return
        for path in sorted((owner.root / "control/requests").glob("*.json")):
            result_path = owner.root / "control/responses" / path.name
            if result_path.exists():
                continue
            request = json.loads(path.read_text())
            try:
                if self.mode != "manual":
                    raise ValueError("Manual actions require policy=manual")
                if request["generation"] != owner.generation:
                    raise ValueError("Stale Actor generation; query status and retry")
                if request["action"] == "start_actor":
                    status = owner.start_actor()
                elif request["action"] == "stop_actor":
                    status = owner.stop_actor()
                else:
                    raise ValueError("Unknown manual action")
                result = {"status": "completed", "observation": status}
            except Exception as error:
                result = {"status": "failed", "error": str(error)}
            atomic_json(result_path, {"request_id": path.stem, **result})

    def close(self) -> None:
        self.owner = None

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
    ) -> None:
        if mode not in {"manual", "agent", "idle", "resident"}:
            raise ValueError("Unknown lifecycle policy")
        if mode == "agent" and decide is None:
            raise ValueError("Agent mode requires an LLM decision callback")
        self.mode, self.decision_fn = mode, decide
        self.owner = None
        self._last_event = None
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

    def decide(self, batch: Any, *, now: float | None = None) -> str:
        owner = self.owner
        now = time.monotonic() if now is None else now
        if batch is not None and owner.actor is not None:
            return "train"
        if self.mode == "resident":
            return "train" if batch is not None else "keep"
        if self.mode == "manual":
            return "keep"
        status = owner.get_status(now=now)
        event = None
        if batch is not None:
            event = "batch_ready"
        elif (
            owner.actor is not None
            and status["idle_seconds"] >= owner.options.idle_timeout
            and status["residence_seconds"] >= owner.options.min_ready_seconds
        ):
            event = "idle_timeout"
        if event is None:
            return "keep"
        if self.mode == "idle":
            return "train" if batch is not None else "release"

        signature = (
            event,
            status["generation"],
            status["applied_version"],
            getattr(batch, "batch_id", None),
            status["counts"].get("ready", 0),
            status["counts"].get("admitted", 0),
        )
        if signature == self._last_event and now < self._wake_at:
            return "keep"
        observation = {
            **status,
            "event": event,
            "batch_ready": batch is not None,
            "batch_id": getattr(batch, "batch_id", None),
            "idle_timeout": owner.options.idle_timeout,
            "min_ready_seconds": owner.options.min_ready_seconds,
        }
        started = time.monotonic()
        decision = self.decision_fn(observation)
        action = decision.get("action")
        if action not in {"start_actor", "stop_actor", "wait_for_event"}:
            raise ValueError("Agent returned an unsupported action")
        reason = decision.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Agent must give a short decision reason")
        if action == "start_actor" and batch is None:
            raise ValueError("No complete batch is available to start Actor")
        if action == "stop_actor" and batch is not None:
            raise ValueError("Agent cannot stop Actor with a complete batch ready")
        wait = decision.get("wake_after_seconds", 30.0)
        if (
            not isinstance(wait, (int, float))
            or not math.isfinite(wait)
            or not 5 <= wait <= 300
        ):
            raise ValueError("Wake delay must be between 5 and 300 seconds")
        self._sequence += 1
        self._last_event = signature
        self._wake_at = time.monotonic() + wait
        self._record(
            {
                "sequence": self._sequence,
                "wall_time": time.time(),
                "monotonic": time.monotonic(),
                "mode": self.mode,
                "observation": observation,
                "decision": decision,
                "decision_seconds": time.monotonic() - started,
            }
        )
        logger.info(
            "Elastic agent: event=%s action=%s reason=%s", event, action, reason
        )
        return {"start_actor": "train", "stop_actor": "release"}.get(action, "keep")

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

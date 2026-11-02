# SPDX-License-Identifier: Apache-2.0

"""Basic aggregate Joint Simulator, restored from elastic-policy-v1.

The original batch-level evaluate(scenario, config) kernel is retained. Online
state/window adaptation lives in joint_online.py and uses the same aggregate
service and sequential residency assumptions, rather than ExpertSimulator.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from typing import TYPE_CHECKING

from .cost_model.aggregate_rollout import RolloutCostModel
from .cost_model.joint_transition import JointTransitionModel
from .joint import JointConfig, JointEvent, JointPrediction, JointScenario

if TYPE_CHECKING:
    from .joint_online import ActorCatalog


class JointSimulator:
    """Fixed-config finite windows with one shared aggregate inference channel.

    Admission/staleness are deliberately in whole logical batches. This is a
    coarse Basic model, not a reconstruction of group-level controller capacity.
    Initial state must be committed and drained of in-flight work; ready batches
    and cumulative counters are explicit. All candidates finish with cleanup.
    """

    def __init__(
        self,
        actors: ActorCatalog,
        rollout: RolloutCostModel,
        transitions: JointTransitionModel,
    ):
        self.actors, self.rollout, self.transitions = actors, rollout, transitions

    @classmethod
    def from_forecast(cls, view, forecast, models, config, handoff, slo, **kwargs):
        from .joint_online import JointWindow

        return JointWindow(view, forecast, models, config, handoff, slo, **kwargs)

    def evaluate(self, scenario: JointScenario, config: JointConfig) -> JointPrediction:
        return _JointRun(self, scenario, config).run()


class _JointRun:
    def __init__(
        self, model: JointSimulator, scenario: JointScenario, config: JointConfig
    ):
        self.model, self.scenario, self.config = model, scenario, config
        self.now = self.actor_cost = self.rollout_cost = self.queue_wait = 0.0
        self.provider, self.accepted = (
            scenario.provider_version,
            scenario.accepted_batches,
        )
        self.completed = self.sequence = 0
        self.ready = dict.fromkeys(range(scenario.ready_batches), 0.0)
        self.next_input = scenario.ready_batches
        self.running = set()
        self.gpu_queue = deque()
        self.heap = []
        self.events = []
        self.actor_times, self.rollout_times, self.profiles = [], {}, {}
        self.sources = []
        self.notes = [
            "Fixed logical workload and trajectory outcomes; no reward prediction",
            "Batch-level FIFO and staleness proxy; max_inflight_batches is not group-level K",
            "One shared aggregate inference channel; external waits release that channel",
            "Replica speedup is capped by stage parallel requests and assumes balanced token work; no TP scaling",
            "Publication waits for the current aggregate GPU stage; external waits continue",
            "Initial state is committed with no in-flight work; supplied ready batches retain their cost boundary",
            "Actor memory and rollout KV checks do not certify complete resource feasibility",
        ]
        self.active_transition = None
        self.actor_busy = self.gpu_busy = self.pending_publish = self.done = False
        self.allocation = (config.actor.world_size, config.rollout.world_size)

    def emit(self, kind: str, batch: int = -1) -> None:
        self.events.append(
            JointEvent(self.now, kind, batch, self.provider, self.accepted)
        )

    def schedule(self, seconds: float, kind: str, data=None) -> None:
        self.sequence += 1
        heapq.heappush(self.heap, (self.now + seconds, self.sequence, kind, data))

    def within_limits(self, actor: int, rollout: int) -> bool:
        s = self.scenario
        return (
            actor <= s.max_actor_gpus
            and rollout <= s.max_rollout_gpus
            and actor + rollout <= s.max_total_gpus
        )

    def preflight(self) -> tuple[str, str] | None:
        s, c = self.scenario, self.config
        if not self.within_limits(c.actor.world_size, c.rollout.world_size):
            return "infeasible", "Joint configuration exceeds GPU pool limits"
        if s.current and (
            s.current.rollout.model_id,
            s.current.rollout.hardware_id,
            s.current.rollout.sampling_id,
        ) != (c.rollout.model_id, c.rollout.hardware_id, c.rollout.sampling_id):
            return (
                "infeasible",
                "Joint comparison changes model, hardware cost basis or sampling semantics",
            )
        try:
            if s.current is not None:
                self.model.actors.actor_config(s.current.actor)
            self.model.actors.actor_config(c.actor)
            for index, batch in enumerate(s.batches):
                prediction = self.model.actors.predict(
                    c.actor, batch.stats_for(c.actor)
                )
                if prediction.static_memory_exceeds_capacity:
                    return (
                        "infeasible",
                        "Actor static memory lower bound exceeds capacity",
                    )
                if not math.isfinite(prediction.seconds) or prediction.seconds < 0:
                    raise ValueError("Actor returned invalid service time")
                self.actor_times.append(prediction.seconds)
                self.notes.extend(prediction.assumptions)
                if batch.inferred_lengths:
                    self.notes.append(
                        "Actor lengths are a fixed shared surrogate inferred before candidate search"
                    )
                if index >= s.ready_batches:
                    estimate = self.model.rollout.predict(c.rollout, batch.rollout)
                    if not estimate.supported:
                        return "unsupported", estimate.reason
                    self.rollout_times[index] = estimate.stages
                    self.sources.append(estimate.source_id)
                    self.notes.append(
                        f"Rollout rate evidence: {estimate.evidence} ({estimate.source_id})"
                    )
        except ValueError as exc:
            return "unsupported", str(exc)
        actions = [("publish", c, c), ("shutdown", c, None)]
        if s.checkpoint_every and any(
            (s.provider_version + i + 1) % s.checkpoint_every == 0
            for i in range(len(s.batches))
        ):
            actions.append(("checkpoint", c, c))
        if s.current != c:
            actions.append(
                ("start" if s.current is None else "reconfigure", s.current, c)
            )
        for action, source, target in actions:
            profile = self.model.transitions.lookup(action, source, target)
            if profile is None:
                return "unsupported", f"Missing directed {action} transition profile"
            if any(
                not self.within_limits(p.actor_gpus, p.rollout_gpus)
                for p in profile.phases
            ):
                return "infeasible", f"{action} temporary allocation exceeds GPU limits"
            self.profiles[action] = profile
            self.sources.append(profile.source_id)
        return None

    def transition(self, action: str) -> None:
        profile = self.profiles[action]
        self.active_transition = action
        self.emit(action + "_start")
        self.allocation = (profile.phases[0].actor_gpus, profile.phases[0].rollout_gpus)
        elapsed = 0.0
        for index, phase in enumerate(profile.phases):
            if index:
                self.schedule(
                    elapsed, "allocation", (phase.actor_gpus, phase.rollout_gpus)
                )
            elapsed += phase.seconds
        self.schedule(elapsed, "transition_end", action)

    def pump(self) -> None:
        if self.active_transition or self.done:
            return
        if self.pending_publish:
            if not self.gpu_busy:
                self.transition("publish")
            return
        s = self.scenario
        if self.completed == len(s.batches):
            self.transition("shutdown")
            return
        if not self.actor_busy and self.completed in self.ready:
            self.queue_wait += self.now - self.ready.pop(self.completed)
            self.actor_busy = True
            self.emit("actor_start", self.completed)
            self.schedule(self.actor_times[self.completed], "actor_end", self.completed)
        # accepted is cumulative; dequeuing a batch never grants fresh capacity.
        while (
            self.next_input < len(s.batches)
            and min(
                self.config.rollout.max_inflight_batches - len(self.running),
                s.staleness + self.provider + 1 - self.accepted - len(self.running),
            )
            > 0
        ):
            index = self.next_input
            self.next_input += 1
            self.running.add(index)
            self.gpu_queue.append((index, 0))
            self.emit("rollout_admitted", index)
        if not self.gpu_busy and self.gpu_queue:
            index, stage = self.gpu_queue.popleft()
            self.gpu_busy = True
            self.emit("rollout_service_start", index)
            self.schedule(
                self.rollout_times[index][stage][0], "gpu_end", (index, stage)
            )

    def dispatch(self, kind: str, data) -> None:
        if kind == "allocation":
            self.allocation = data
        elif kind == "gpu_end":
            self.gpu_busy = False
            index, stage = data
            self.emit("rollout_service_end", index)
            self.schedule(self.rollout_times[index][stage][1], "external_end", data)
        elif kind == "external_end":
            index, stage = data
            if stage + 1 < len(self.rollout_times[index]):
                self.gpu_queue.append((index, stage + 1))
            else:
                self.running.remove(index)
                self.ready[index] = self.now
                self.accepted += 1
                self.emit("batch_ready", index)
        elif kind == "actor_end":
            self.pending_publish = True
            self.emit("actor_end", data)
        elif kind == "transition_end":
            self.active_transition = None
            self.allocation = (
                self.config.actor.world_size,
                self.config.rollout.world_size,
            )
            self.emit(data + "_end")
            if data == "publish":
                self.provider += 1
                self.completed += 1
                self.actor_busy = self.pending_publish = False
                self.emit("provider_version", self.completed - 1)
                if (
                    self.scenario.checkpoint_every
                    and self.provider % self.scenario.checkpoint_every == 0
                ):
                    self.transition("checkpoint")
            elif data == "shutdown":
                self.allocation = (0, 0)
                self.done = True

    def result(self, status: str, reason: str) -> JointPrediction:
        consumed = self.scenario.batches[: self.completed]
        return JointPrediction(
            status,
            self.config,
            self.now,
            self.actor_cost,
            self.rollout_cost,
            self.completed,
            sum(b.actor.samples for b in consumed),
            sum(b.actor.effective_tokens for b in consumed),
            sum(b.actor.sequence_tokens for b in consumed),
            self.queue_wait,
            reason,
            tuple(self.events),
            tuple(dict.fromkeys(self.notes)),
            tuple(dict.fromkeys(self.sources)),
        )

    def run(self) -> JointPrediction:
        error = self.preflight()
        if error:
            return self.result(*error)
        if self.scenario.current != self.config:
            self.transition("start" if self.scenario.current is None else "reconfigure")
        self.pump()
        while not self.done:
            if not self.heap:
                return self.result(
                    "incomplete",
                    "No progress: initial counters or batch admission capacity block the fixed window",
                )
            next_time = self.heap[0][0]
            delta = next_time - self.now
            self.actor_cost += self.allocation[0] * delta
            self.rollout_cost += self.allocation[1] * delta
            self.now = next_time
            # Complete all same-time work before allowing a new service/admission.
            while self.heap and self.heap[0][0] == self.now:
                _, _, kind, data = heapq.heappop(self.heap)
                self.dispatch(kind, data)
            self.pump()
        return self.result(
            "complete",
            "Fixed joint configuration completed the same window and terminal cleanup",
        )

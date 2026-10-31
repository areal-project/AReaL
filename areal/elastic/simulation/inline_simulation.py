# SPDX-License-Identifier: Apache-2.0

"""Task-pool-driven training and equal-work candidate branches under a hard cap."""

import time
from dataclasses import asdict, replace

from areal.elastic.policy.basic.inline import InlinePrediction
from areal.elastic.policy.basic.inline_trigger import InlineWorkloadFeatures
from areal.elastic.policy.reference.core import fingerprint

from .inline_inputs import TaskPool
from .inline_transitions import InlineTransition
from .online_inputs import WorkInputs
from .simulation_core import SimulationCore


class InlineSimulation(SimulationCore):
    def __init__(
        self,
        episodes,
        initial,
        models,
        config,
        handoff,
        slo=None,
        *,
        policy=None,
        gpu_budget=None,
        measured_search_time=True,
        search_clock=None,
        search_callback=None,
        input_closes_at=0.0,
        _work=None,
        **kwargs,
    ):
        if config.actor_lifecycle_enabled:
            raise ValueError("Inline first example uses a resident Actor")
        if not episodes or (_work is None and len(episodes) % config.batch_size):
            raise ValueError("Inline task pool must contain complete training batches")
        self.gpu_budget = sum(handoff.node_gpus) if gpu_budget is None else gpu_budget
        if self.gpu_budget != sum(handoff.node_gpus) or self.gpu_budget > 16:
            raise ValueError(
                "Inline pool topology must equal the whole-run GPU hard cap (at most 16)"
            )
        if initial.actor_gpus + initial.rollout_gpus > self.gpu_budget:
            raise ValueError("Initial inline configuration exceeds the hard GPU cap")
        self.task_pool = TaskPool(episodes)
        self.total_updates = len(episodes) // config.batch_size
        self.transition = None
        self.pending_search = None
        self.measured_search_time = measured_search_time
        self.search_clock = search_clock or time.perf_counter
        self.search_callback = search_callback
        self.search_seconds = self.verification_seconds = 0.0
        self.search_check_seconds = 0.0
        self.trigger_checks = []
        work = _work or WorkInputs.prepare(
            episodes, max(e.available_at for e in episodes), 0.0
        )
        work = replace(
            work,
            heap=tuple(e for e in work.heap if e[3] not in {"arrival", "input_closed"}),
        )
        super().__init__(
            episodes,
            initial,
            models,
            config,
            handoff,
            slo,
            input_closes_at=input_closes_at,
            policy=policy,
            _work=work,
            **kwargs,
        )
        self._truth = dict(self._truth)

    def admit(self):
        if self.transition is not None:
            return
        super().admit()
        while (
            self.task_pool.cursor < len(self.task_pool)
            and self._active_episodes < self.config.client_concurrency
            and self.admission_credits() > 0
        ):
            task = self.task_pool.take()
            self._truth[task.episode_id] = replace(task, available_at=self.now)
            self.arrive(task.episode_id)
            super().admit()
        if self.task_pool.cursor == len(self.task_pool) and not self.input_closed:
            self.input_closed = True
            self.emit("input_closed")

    def advance(self, when):
        if (
            self.task_pool.cursor < len(self.task_pool)
            and self._admission_cursor == len(self.arrival_order)
            and self.admission_credits() <= 0
        ):
            self.admission_credit_blocked_seconds += when - self.now
        super().advance(when)

    def tick(self, local_services=None):
        if self.transition is not None and self.transition.phase in {
            "saving",
            "loading",
        }:
            return
        # Full ticks also refill the task pool after credit/concurrency changes.
        super().tick()
        if sum(len(g) for _, g in self.pool.leases.values()) > self.gpu_budget:
            raise ValueError("Inline resource hard cap exceeded")

    def before_work(self):
        if self.transition is not None:
            self.transition.tick()

    def start_actor(self):
        if self.transition is None:
            super().start_actor()

    def after_work(self):
        if self.transition is not None:
            self.transition.tick()
            return
        if self.consumed == len(self.task_pool) and self.publication == "idle":
            self.pending_search = None
            self.decision_pending = False
            return
        if (
            self.policy is not None
            and self.pending_search is None
            and self.provider_version >= self.policy.next_check_update
            and self.publication == "idle"
            and not self.decision_pending
            and self.reconfiguration_allowed()
            and self._completed_episodes
            >= self.policy.config.minimum_completed_episodes
        ):
            trigger = "periodic"
            if self.policy.config.workload_trigger_enabled:
                started = time.perf_counter()
                end = self.provider_version * self.config.batch_size
                width = (
                    self.policy.config.feature_window_updates * self.config.batch_size
                )
                if end < width:
                    self.policy.search_trigger.last_probe_update = self.provider_version
                    return
                # Published batches provide complete, equally weighted episodes;
                # fast-finished future episodes cannot bias this shape window.
                episodes = tuple(
                    tuple(self.episodes[identity].past)
                    for identity in self.arrival_order[end - width : end]
                )
                check = self.policy.search_trigger.check(
                    self.provider_version, InlineWorkloadFeatures.from_turns(episodes)
                )
                self.search_check_seconds += time.perf_counter() - started
                record = asdict(check)
                self.trigger_checks.append(record)
                self.emit("inline_search_check", **record)
                if not check.search:
                    return
                trigger = check.reason
            view = self.observe()
            if self.policy.due(view):
                self.begin_search(view, trigger=trigger)

    def request_handoff(self, target):
        if self.transition is not None or target == self.layout:
            return False
        if not InlineTransition.feasible(self, target):
            self.emit(
                "handoff_rejected",
                target=asdict(target),
                reason="No legal hard-budget placement",
            )
            return False
        self.transition = InlineTransition(self, target)
        self.pending_handoff = {"target": target}
        self.emit("handoff_start", target=asdict(target), source=asdict(self.layout))
        return True

    def handle_transition_event(self, kind, data):
        if kind == "decision_apply":
            self.decision_pending = False
            self.request_handoff(data)
        elif kind == "inline_saved":
            self.transition.saved()
        elif kind == "inline_process_started":
            self.transition.process_started(data)
        elif kind == "inline_role_ready":
            self.transition.role_ready(data)
        elif kind == "inline_search_complete":
            self.verify_search()
        elif kind == "inline_verify_complete":
            self.apply_verified()
        else:
            super().handle_transition_event(kind, data)

    def begin_search(self, view, *, trigger="periodic"):
        started = self.search_clock()
        model = InlineCandidateModel(
            self.models,
            self.config,
            self.handoff,
            self.gpu_budget,
            delay=self.policy.config.decision_delay_seconds,
        )
        decision = self.policy.decide(
            view, self.task_pool.pending(), model, self.total_updates
        )
        elapsed = (
            max(0.0, self.search_clock() - started)
            if self.measured_search_time
            else self.policy.config.decision_delay_seconds
        )
        self.search_seconds += elapsed
        self.policy.last_checked_update = view.provider_version
        record = dict(
            stage="search",
            observation_id=fingerprint(view),
            **asdict(decision),
            search_seconds=elapsed,
            trigger=trigger,
        )
        self.decisions.append(record)
        self.emit("decision", **record)
        if self.search_callback is not None:
            self.search_callback(record)
        self.pending_search = decision
        self.decision_pending = True
        self.schedule_at(self.now + elapsed, "inline_search_complete", priority=9)

    def verify_search(self):
        decision = self.pending_search
        if decision is None:
            return
        self.decision_pending = False
        view = self.observe()
        if view.busy or decision.target == view.layout:
            self.pending_search = None
            self.emit(
                "search_retained",
                reason="No switch or publication/deployment boundary moved",
            )
            return
        started = self.search_clock()
        model = InlineCandidateModel(
            self.models,
            self.config,
            self.handoff,
            self.gpu_budget,
            delay=self.policy.config.decision_delay_seconds,
        )
        verified = self.policy.decide(
            view,
            self.task_pool.pending(),
            model,
            self.total_updates,
            targets=(decision.target,),
        )
        elapsed = (
            max(0.0, self.search_clock() - started)
            if self.measured_search_time
            else self.policy.config.decision_delay_seconds
        )
        self.verification_seconds += elapsed
        self.pending_search = verified
        self.decision_pending = True
        record = dict(
            stage="verification",
            observation_id=fingerprint(view),
            **asdict(verified),
            verification_seconds=elapsed,
        )
        self.decisions.append(record)
        self.emit("decision", **record)
        self.schedule_at(self.now + elapsed, "inline_verify_complete", priority=9)

    def apply_verified(self):
        decision = self.pending_search
        self.pending_search = None
        self.decision_pending = False
        if decision is None:
            return
        view = self.observe()
        if (
            view.busy
            or view.provider_version != decision.observed_version
            or view.training_version != view.provider_version
        ):
            self.emit(
                "search_discarded",
                reason="Training/publication boundary moved after verification",
            )
            return
        if decision.target != view.layout:
            self.request_handoff(decision.target)

    @classmethod
    def from_forecast(cls, view, forecast, models, config, handoff, gpu_budget):
        branch = cls.restore(
            view,
            forecast,
            models,
            config,
            handoff,
            gpu_budget=gpu_budget,
            coalesce_decode=True,
        )
        branch.task_pool.cursor = len(view.episodes)
        return branch

    def result(self):
        result = super().result()
        result.update(
            mode="inline",
            input_tasks=len(self.task_pool),
            submitted_tasks=self.task_pool.cursor,
            gpu_budget=self.gpu_budget,
            search_seconds=self.search_seconds,
            verification_seconds=self.verification_seconds,
            search_check_seconds=self.search_check_seconds,
            search_checks=len(self.trigger_checks),
            full_searches=sum(d["stage"] == "search" for d in self.decisions),
            assumptions=tuple(result["assumptions"])
            + (
                "Finite ordered task pool; concurrency and staleness limit active supply",
                "Stop-before-start reconfiguration; all changing deployments release before target allocation",
                "Conditional timing simulation; GPU state/quality preservation requires native validation",
            ),
        )
        return result


class InlineCandidateModel:
    def __init__(self, models, config, handoff, gpu_budget, *, delay=0.0):
        self.models, self.config, self.handoff = models, config, handoff
        self.pipeline_episodes = (config.staleness + 1) * config.batch_size
        self.gpu_budget, self.delay, self.batch_size = (
            gpu_budget,
            delay,
            config.batch_size,
        )

    def evaluate(self, view, forecast, layout, target_update):
        branch = None
        try:
            branch = InlineSimulation.from_forecast(
                view, forecast, self.models, self.config, self.handoff, self.gpu_budget
            )
            branch.training_limit = target_update
            if self.delay:
                branch.run(until=view.now + self.delay, until_update=target_update)
            if layout != view.layout and not branch.request_handoff(layout):
                raise ValueError("Candidate has no legal hard-budget transition")
            result = branch.run(until_update=target_update)
            supported = (
                branch.provider_version >= target_update
                and result["status"] in {"complete", "running"}
                and branch.pending_handoff is None
                and result["peak_gpus"] <= self.gpu_budget
            )
            return InlinePrediction(
                layout,
                supported,
                branch.now - view.now if supported else None,
                target_update,
                branch.provider_version,
                result["total_gpu_seconds"],
                result["peak_gpus"],
                result["reason"],
            )
        except (ValueError, ArithmeticError) as exc:
            return InlinePrediction(
                layout,
                False,
                None,
                target_update,
                branch.provider_version if branch else view.provider_version,
                0.0,
                0,
                str(exc),
            )

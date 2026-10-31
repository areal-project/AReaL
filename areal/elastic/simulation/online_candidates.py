# SPDX-License-Identifier: Apache-2.0

"""SLO-window candidate evaluation from causal observations and forecasts."""

from __future__ import annotations

import math
from concurrent.futures import Executor
from dataclasses import replace

from areal.elastic.policy.basic.observations import (
    Forecast,
    OnlineLayout,
    OnlineObservation,
)
from areal.elastic.policy.basic.online import (
    ActorMemorySafety,
    ActorSupply,
    CandidatePrediction,
    rolling_slo,
)
from areal.elastic.policy.basic.search import (
    SearchBound,
    candidate_order,
    select_prediction,
)
from areal.elastic.policy.basic.serving import ServingSLO
from areal.elastic.policy.reference.core import fingerprint
from areal.elastic.policy.reference.cost_model._rollout.types import (
    ServiceSlice,
)

from .online_inputs import ForecastInputs
from .online_memory import EvaluationMemoryModels
from .online_simulation import OnlineSimulation, _CandidateGuard, _ForecastSLOGuard
from .online_transitions import (
    HandoffConfig,
)
from .simulation_core import SimulationConfig, SimulationModels


class OnlineForecastModel:
    """A candidate evaluator whose only workload input is the sampled forecast."""

    def __init__(
        self,
        models: SimulationModels,
        config: SimulationConfig,
        handoff: HandoffConfig,
        slo: ServingSLO,
        decision_delay_seconds: float = 0.0,
        *,
        prune_irreversible_slo: bool = True,
        coalesce_decode: bool = True,
        executor: Executor | None = None,
        actor_memory_reserve_fraction: float = 0.0,
        actor_memory_scope: str = "all_completed",
    ):
        self.models, self.config, self.handoff, self.slo = models, config, handoff, slo
        self.delay = decision_delay_seconds
        self.prune_irreversible_slo = prune_irreversible_slo
        self.coalesce_decode = coalesce_decode
        self.executor = executor
        if not 0 <= actor_memory_reserve_fraction < 1:
            raise ValueError("Actor memory reserve must be in [0, 1)")
        self.actor_memory_reserve_fraction = actor_memory_reserve_fraction
        if actor_memory_scope not in ("all_completed", "evaluation"):
            raise ValueError("Actor memory scope must be all_completed or evaluation")
        self.actor_memory_scope = actor_memory_scope
        self._view = None
        self._forecasts = None
        self._work_ids = ()
        self._prepared: dict[int, ForecastInputs] = {}
        self.input_preparations = 0
        self.execution_stats = {"mode": "serial"}

    @property
    def source_ids(self):
        return self.models.source_ids

    def prepare_context(self, view, forecasts):
        if self._view is view and self._forecasts is forecasts:
            return
        self._view, self._forecasts = view, forecasts
        self._work_ids = tuple(fingerprint(f) for f in forecasts)
        self._context_id = (
            fingerprint(view),
            self._work_ids,
            fingerprint(
                self.config,
                self.handoff,
                self.slo,
                self.delay,
                self.source_ids,
                self.actor_memory_reserve_fraction,
                self.actor_memory_scope,
            ),
        )
        self._prepared = {}
        self._historical = any(f.replay is not None for f in forecasts)
        self._past = frozenset(
            r.request_id for r in view.requests if r.completed_at is not None
        )
        self._remaining = tuple(
            sum(len(e.turns) for e in f.episodes) - len(self._past) for f in forecasts
        )
        self._known_batch = ()
        self._memory_checks = {}
        self._memory_batches = {}
        if self._historical:
            completed = {
                e.episode_id: e.completed_turns
                for e in view.episodes
                if e.completed_at is not None
            }
            identities = (
                view.actor.waiting.batch
                if view.actor.waiting is not None
                else view.actor.batch
            )
            if identities:
                self._known_batch = tuple(completed[e] for e in identities)
            elif len(completed) >= self.config.batch_size:
                self._known_batch = tuple(
                    list(completed.values())[-self.config.batch_size :]
                )
        if (
            self.actor_memory_reserve_fraction
            and self.actor_memory_scope == "all_completed"
        ):
            # Preserve the real arrival-order batch boundaries, including holes
            # caused by unfinished episodes. Never fill them with future turns.
            ordered = sorted(
                view.episodes, key=lambda e: (e.available_at, e.episode_id)
            )
            size = self.config.batch_size
            for start in range(0, len(ordered) - size + 1, size):
                rows = ordered[start : start + size]
                if all(
                    e.completed_at is not None and e.completed_at <= view.now
                    for e in rows
                ):
                    self._memory_batches[tuple(e.episode_id for e in rows)] = tuple(
                        e.completed_turns for e in rows
                    )
            if self._known_batch:
                identities = identities or tuple(list(completed)[-size:])
                self._memory_batches[tuple(identities)] = self._known_batch

    def actor_memory_safety(self, view, layout):
        reserve = self.actor_memory_reserve_fraction
        if not reserve:
            return None
        key = layout.actor_dp, layout.actor_tp
        if key in self._memory_checks:
            return self._memory_checks[key]
        peak = capacity = allowed = None
        worst, count, admissible = (), 0, True
        reason = "All completed historical batches retain the configured memory reserve"
        estimate = getattr(self.models, "actor_memory", None)
        if estimate is None or not self._memory_batches:
            admissible = False
            reason = "Actor memory safety unknown: need memory estimates and a completed observed batch"
        else:
            try:
                for ids, batch in self._memory_batches.items():
                    used, limit, measured_oom = estimate(layout, batch)
                    if (
                        not math.isfinite(used)
                        or used < 0
                        or not math.isfinite(limit)
                        or limit <= 0
                    ):
                        raise ValueError("Invalid Actor memory estimate")
                    count += 1
                    if peak is None or used > peak:
                        peak, capacity, allowed, worst = (
                            used,
                            limit,
                            limit * (1 - reserve),
                            ids,
                        )
                    if measured_oom or used > limit * (1 - reserve):
                        admissible = False
                        reason = "Historical Actor shape exceeds the memory safety budget or has a measured OOM"
            except ValueError as exc:
                admissible = False
                reason = f"Actor memory safety unavailable: {exc}"
        result = ActorMemorySafety(
            admissible, reserve, count, peak, capacity, allowed, worst, view.now, reason
        )
        self._memory_checks[key] = result
        return result

    def actor_supply(self, view, forecast, layout, *, models=None):
        """Estimate ongoing demand from offered arrivals and causal turn templates.

        Incomplete final batches wrap to the start of the same historical sequence
        and carry fractional batch weight. This estimates a repeated workload's
        capacity, not extra training in the serving window or an exact queue bound.
        """
        models = models or self.models
        replay = forecast.replay
        recent = sorted(
            (
                e
                for e in view.episodes
                if replay.template_started_at <= e.available_at < view.now
            ),
            key=lambda e: (e.available_at, e.episode_id),
        )
        rate = (
            len(recent) / replay.template_seconds
            if replay.template_seconds > 0
            else 0.0
        )
        if view.input_closed:
            rate = 0.0
        templates = {e.episode_id: e.turns for e in forecast.episodes}
        rows = tuple(templates[e.episode_id] for e in recent) or self._known_batch
        batch_size = self.config.batch_size
        actor_seconds = publication_seconds = capacity = balance = None
        sustainable = True if rate == 0 else None
        batches = padded = 0
        reason = (
            "No ongoing offered arrivals"
            if rate == 0
            else "No historical Actor workload"
        )
        if rows:
            weighted_seconds = weight = 0.0
            try:
                for offset in range(0, len(rows), batch_size):
                    count = min(batch_size, len(rows) - offset)
                    batch = tuple(
                        rows[(offset + i) % len(rows)] for i in range(batch_size)
                    )
                    stages = models.actor(layout, batch)
                    if not stages:
                        raise ValueError("Actor model supplied no capacity stages")
                    seconds = sum(seconds for _, seconds in stages)
                    if not math.isfinite(seconds) or seconds <= 0:
                        raise ValueError(
                            "Actor capacity requires a positive finite cycle"
                        )
                    weighted_seconds += seconds * count / batch_size
                    weight += count / batch_size
                    batches += 1
                    padded += batch_size - count
                actor_seconds = weighted_seconds / weight
                publication_seconds = models.publication(layout, (layout,))
                if not math.isfinite(publication_seconds) or publication_seconds < 0:
                    raise ValueError("Publication capacity requires a finite duration")
                capacity = batch_size / (actor_seconds + publication_seconds)
                balance = capacity - rate
                sustainable = balance >= -1e-12
                reason = "Target-only historical Actor stages plus steady-layout publication; excludes input waiting, cutover and old Actor work"
            except ValueError as exc:
                reason = f"Actor supply estimate unavailable: {exc}"
        return ActorSupply(
            len(recent),
            replay.template_seconds,
            rate,
            actor_seconds,
            publication_seconds,
            capacity,
            balance,
            sustainable,
            batches,
            padded,
            sum(e.completed_at is None for e in recent),
            view.input_closed,
            reason,
        )

    def search(self, view, forecasts, layouts, *, preferred=(), trigger="pressure"):
        self.prepare_context(view, forecasts)
        if not self._historical:
            return self.evaluate_many(view, forecasts, layouts)
        from .online_parallel import evaluate_chunk

        actor_scores, rollout_scores = {}, {}
        turns = tuple(t for e in self._known_batch for t in e)[-16:]

        def score(layout):
            actor_key = layout.actor_dp, layout.actor_tp
            rollout_key = layout.rollout_replicas, layout.rollout_tp
            try:
                if actor_key not in actor_scores:
                    actor_scores[actor_key] = (
                        sum(s for _, s in self.models.actor(layout, self._known_batch))
                        if self._known_batch
                        else 0.0
                    )
                if rollout_key not in rollout_scores:
                    cfg = self.models.rollout_config(layout)
                    seconds = 0.0
                    for t in turns:
                        q = min(t.input_tokens, cfg.max_batch_tokens)
                        prefill = self.models.service(
                            layout,
                            0,
                            (
                                ServiceSlice(
                                    "rank", 0, q, "prefill", q == t.input_tokens
                                ),
                            ),
                        )
                        decode = self.models.service(
                            layout,
                            t.input_tokens,
                            (ServiceSlice("rank", t.input_tokens, 1, "decode", True),),
                        )
                        seconds += (
                            math.ceil(t.input_tokens / q) * prefill
                            + t.output_tokens * decode
                        )
                    rollout_scores[rollout_key] = (
                        seconds
                        / max(1, len(turns))
                        * self.config.batch_size
                        / layout.rollout_replicas
                    )
                return max(actor_scores[actor_key], rollout_scores[rollout_key])
            except ValueError:
                return math.inf  # Priority only; the real evaluation records support.

        order = candidate_order(layouts, view.layout, preferred, score)
        ordered = [None] * len(layouts)
        best = None
        work, cpus, preparations, pids = [], 0.0, 0, set()
        parallel = (
            self.executor is not None
            and getattr(self.models, "worker_spec", None) is not None
        )
        payload = (
            getattr(self.models, "worker_spec", None),
            self.source_ids,
            self.config,
            self.handoff,
            self.slo,
            self.delay,
            self.prune_irreversible_slo,
            self.coalesce_decode,
            view,
            forecasts,
            self._context_id,
            self.actor_memory_reserve_fraction,
            self.actor_memory_scope,
        )
        # Score current first for the below-target fallback, not as a pressure
        # cost incumbent: any feasible replacement takes precedence over it.
        # Subsequent small waves propagate only already completed incumbents.
        waves = [order[:1]] + [order[i : i + 8] for i in range(1, len(order), 8)]
        for wave in waves:
            bound = (
                SearchBound(
                    self._context_id,
                    self.slo.minimum_request_fraction,
                    best,
                    fraction_pruning=False,
                )
                if best and not (trigger == "pressure" and best.layout == view.layout)
                else None
            )
            if parallel and len(wave) > 1:
                chunks = tuple(
                    self.executor.map(
                        evaluate_chunk,
                        (
                            (payload + (bound,), wave[i : i + 2])
                            for i in range(0, len(wave), 2)
                        ),
                    )
                )
                for chunk in chunks:
                    pids.add(chunk.pid)
                    cpus += chunk.task_cpu_seconds
                    preparations += chunk.input_preparations
                    for index, prediction in zip(
                        chunk.indices, chunk.predictions, strict=True
                    ):
                        ordered[index] = prediction
                        work.append(prediction)
            else:
                for index, layout in wave:
                    prediction = self.evaluate(view, forecasts, layout, bound=bound)
                    ordered[index] = prediction
                    work.append(prediction)
            # Original order keeps the current configuration as the work-ID anchor.
            best = select_prediction(
                tuple(p for p in ordered if p is not None),
                view.layout,
                self.slo.minimum_request_fraction,
                trigger=trigger,
            )
        self.execution_stats = dict(
            mode="bounded_process_pool" if parallel else "bounded_serial",
            considered=len(ordered),
            completed=sum(p.supported for p in ordered),
            pruned=sum(p.pruned for p in ordered),
            unsupported=sum(not p.supported and not p.pruned for p in ordered),
            evaluation_order=tuple(p.layout.key for p in work),
            workers_used=len(pids),
            worker_task_cpu_seconds=cpus,
            input_preparations=preparations,
        )
        return tuple(ordered)

    def evaluate_many(self, view, forecasts, layouts):
        spec = getattr(self.models, "worker_spec", None)
        if self.executor is None or spec is None:
            return tuple(self.evaluate(view, forecasts, layout) for layout in layouts)
        from .online_parallel import evaluate_chunk

        self.prepare_context(view, forecasts)
        payload = (
            spec,
            self.models.source_ids,
            self.config,
            self.handoff,
            self.slo,
            self.delay,
            self.prune_irreversible_slo,
            self.coalesce_decode,
            view,
            forecasts,
            self._context_id,
            self.actor_memory_reserve_fraction,
            self.actor_memory_scope,
        )
        indexed = tuple(enumerate(layouts))
        jobs = ((payload, indexed[i : i + 8]) for i in range(0, len(indexed), 8))
        # Small jobs balance unequal worker speeds; explicit indices retain
        # candidate order, including repeated layout entries.
        results = tuple(self.executor.map(evaluate_chunk, jobs))
        self.execution_stats = dict(
            mode="process_pool",
            workers_used=len({chunk.pid for chunk in results}),
            chunks=len(results),
            worker_task_cpu_seconds=sum(chunk.task_cpu_seconds for chunk in results),
            input_preparations=sum(chunk.input_preparations for chunk in results),
        )
        ordered = [None] * len(layouts)
        for chunk in results:
            for index, prediction in zip(chunk.indices, chunk.predictions, strict=True):
                ordered[index] = prediction
        return tuple(ordered)

    def evaluate(
        self,
        view: OnlineObservation,
        forecasts: tuple[Forecast, ...],
        layout: OnlineLayout,
        *,
        bound: SearchBound | None = None,
    ) -> CandidatePrediction:
        self.prepare_context(view, forecasts)
        if (
            self.actor_memory_reserve_fraction
            and self.actor_memory_scope == "evaluation"
        ):
            models = EvaluationMemoryModels(
                self.models, layout, self.actor_memory_reserve_fraction, view.now
            )
            prediction = self._evaluate(
                view, forecasts, layout, bound=bound, models=models
            )
            memory = models.report
            if not memory.admissible:
                prediction = replace(
                    prediction,
                    supported=False,
                    worst_request_fraction=None,
                    mean_gpu_seconds=None,
                    reason=memory.reason,
                    actor_coverage="evaluation_memory_rejected",
                    pruned=False,
                )
            return replace(prediction, actor_memory=memory)
        memory = self.actor_memory_safety(view, layout)
        if memory is not None and not memory.admissible:
            return CandidatePrediction(
                layout,
                False,
                None,
                None,
                self._work_ids,
                memory.reason,
                self.source_ids,
                actor_coverage="historical_memory_rejected",
                actor_memory=memory,
            )
        prediction = self._evaluate(view, forecasts, layout, bound=bound)
        return (
            replace(prediction, actor_memory=memory)
            if memory is not None
            else prediction
        )

    def _evaluate(self, view, forecasts, layout, *, bound=None, models=None):
        models = models or self.models
        self.prepare_context(view, forecasts)
        work_ids = self._work_ids
        if (
            view.actor.lifecycle is not None
            and view.actor.lifecycle.state == "stopped"
            and (layout.actor_dp, layout.actor_tp)
            != (view.layout.actor_dp, view.layout.actor_tp)
        ):
            return CandidatePrediction(
                layout,
                False,
                None,
                None,
                work_ids,
                "Stopped Actor retains its recovery DP/TP; only Rollout may change",
                self.source_ids,
            )
        if bound is not None and bound.context_id != self._context_id:
            bound = None
        if self.config.actor_lifecycle_enabled:
            # Constant target-residency lower bounds do not hold during sleep.
            bound = None
        costs, fractions, window_costs = [], [], []
        running_costs = []
        switching_costs, horizons, window_stats, window_results = [], [], [], []
        historical = self._historical
        coverage = "legacy_sampled_execution"
        known_batch = self._known_batch
        supply = (
            tuple(self.actor_supply(view, f, layout, models=models) for f in forecasts)
            if historical
            else ()
        )
        if historical:
            if known_batch:
                try:
                    if not models.actor(layout, known_batch):
                        raise ValueError("Actor model supplied no update stages")
                except ValueError as exc:
                    return CandidatePrediction(
                        layout,
                        False,
                        None,
                        None,
                        work_ids,
                        str(exc),
                        self.models.source_ids,
                        actor_coverage="known_batch_rejected",
                        actor_supply=supply,
                    )
                coverage = "known_complete_batch_capacity_checked"
        past = self._past
        if self.prune_irreversible_slo and not historical:
            failed = sum(
                r.completed_at is None
                and (
                    (r.first_token_at if r.first_token_at is not None else view.now)
                    - r.ready_at
                    > self.slo.ttft_seconds + 1e-12
                    or max(
                        r.maximum_tbt_seconds,
                        view.now - r.last_token_at
                        if r.last_token_at is not None
                        else 0.0,
                    )
                    > self.slo.tbt_seconds + 1e-12
                )
                for r in view.requests
            )
            for remaining in self._remaining:
                upper = 1 - failed / remaining if remaining else 1.0
                if upper + 1e-12 < self.slo.minimum_request_fraction:
                    return CandidatePrediction(
                        layout,
                        False,
                        None,
                        None,
                        work_ids,
                        f"Observed irreversible SLO failures: {failed}; sampled future scope: {remaining}; joint_fraction_upper_bound={upper:.12g}; no cost simulation performed",
                        self.models.source_ids,
                    )
        for index, forecast in enumerate(forecasts):
            horizon = forecast.input_closes_at - view.now
            running_cost = horizon * (layout.actor_gpus + layout.rollout_gpus)
            try:
                prepared = self._prepared.get(index)
                if prepared is None:
                    prepared = ForecastInputs.prepare(
                        view,
                        forecast,
                        self.config.max_new_tokens,
                        self.config.max_context_tokens,
                    )
                    self._prepared[index] = prepared
                    self.input_preparations += 1
                simulation = OnlineSimulation.from_forecast(
                    view,
                    forecast,
                    models,
                    self.config,
                    self.handoff,
                    self.slo,
                    coalesce_decode=self.coalesce_decode,
                    _prepared=prepared,
                )
                simulation._forecast_target = layout
                if historical:
                    # Keep residency through H even when the observed input has
                    # already closed. The synthetic close event still ends at H.
                    simulation.input_closed = False
                if bound is not None:
                    simulation._candidate_guard = _CandidateGuard(
                        bound,
                        past,
                        self._remaining[index],
                        sum(costs),
                        min((f for f in fractions if f is not None), default=1.0),
                        len(forecasts),
                        fixed_running_cost=running_cost if historical else None,
                    )
                if self.prune_irreversible_slo and not historical:
                    simulation._forecast_slo_guard = _ForecastSLOGuard(
                        past, self._remaining[index]
                    )
                if layout != view.layout:
                    simulation.decision_pending = True
                    simulation.schedule_at(
                        view.now + self.delay, "decision_apply", layout, priority=9
                    )
                if historical:
                    result = simulation.run(until=forecast.input_closes_at)
                    window_costs.append(result["total_gpu_seconds"])
                    if self.config.actor_lifecycle_enabled:
                        running_cost = (
                            result["total_gpu_seconds"]
                            - simulation.switching_gpu_seconds()
                        )
                    window_results.append(result)
                    window_stats.append(
                        rolling_slo(
                            replace(
                                view,
                                now=simulation.now,
                                requests=tuple(
                                    r.view.snapshot()
                                    for key, r in simulation.requests.items()
                                    if key not in past
                                ),
                            ),
                            self.slo,
                            horizon,
                        )
                    )
                    # Only estimate an unfinished transition beyond H. Do not
                    # drain target work or let tail outcomes alter window SLO.
                    if result["status"] == "running" and (
                        simulation.pending_handoff or simulation.decision_pending
                    ):
                        result = simulation.run(stop_after_handoff=True)
                else:
                    result = simulation.run()
            except ValueError as exc:
                return CandidatePrediction(
                    layout,
                    False,
                    None,
                    None,
                    work_ids,
                    str(exc),
                    self.models.source_ids,
                    actor_supply=supply,
                )
            if result["status"] == "candidate_pruned":
                guard = simulation._candidate_guard
                return CandidatePrediction(
                    layout,
                    False,
                    None,
                    None,
                    work_ids,
                    result["reason"],
                    self.source_ids,
                    actor_coverage=coverage,
                    pruned=True,
                    fraction_upper_bound=guard.upper,
                    mean_cost_lower_bound=guard.lower,
                    pruning_incumbent=(
                        bound.best.layout.key,
                        bound.best.worst_request_fraction,
                        bound.best.mean_gpu_seconds,
                    ),
                    actor_supply=supply,
                )
            if (
                result["status"]
                not in (("complete", "running") if historical else ("complete",))
                or simulation.counters["handoff_deferred"]
                or (
                    historical
                    and (simulation.pending_handoff or simulation.decision_pending)
                )
            ):
                return CandidatePrediction(
                    layout,
                    False,
                    None,
                    None,
                    work_ids,
                    result["reason"]
                    if result["status"] not in ("complete", "running")
                    else "No spare GPUs for TP-local overlap",
                    self.models.source_ids,
                    actor_supply=supply,
                )
            if historical and not known_batch:
                changes_actor = (layout.actor_dp, layout.actor_tp) != (
                    view.layout.actor_dp,
                    view.layout.actor_tp,
                )
                owners = {
                    e["target"]
                    for e in simulation.events
                    if e["kind"] == "actor_cutover"
                }
                exercised = any(
                    e["kind"] == "actor_start"
                    and e["deployment"] in owners
                    and any(
                        row["episodes"] == e["episodes"]
                        and row["committed_at"] is not None
                        and row["committed_at"] <= forecast.input_closes_at
                        for row in simulation.training_records
                    )
                    for e in simulation.events
                )
                if changes_actor and not exercised:
                    return CandidatePrediction(
                        layout,
                        False,
                        None,
                        None,
                        work_ids,
                        "Actor capacity unknown: no observed full batch or post-cutover complete update",
                        self.models.source_ids,
                        actor_coverage="unknown_no_complete_batch",
                        actor_supply=supply,
                    )
                coverage = (
                    "post_cutover_complete_update"
                    if exercised
                    else "current_actor_unchanged"
                )
            if historical:
                cost = simulation.switching_gpu_seconds()
                switching_costs.append(cost)
                horizons.append(horizon)
                costs.append(cost + running_cost)
                running_costs.append(running_cost)
                fractions.append(window_stats[-1].fraction)
                continue
            future = [
                r.view for key, r in simulation.requests.items() if key not in past
            ]
            passed = sum(
                r.completed_at is not None
                and r.first_token_at - r.ready_at <= self.slo.ttft_seconds + 1e-12
                and r.maximum_tbt_seconds <= self.slo.tbt_seconds + 1e-12
                for r in future
            )
            fractions.append(passed / len(future) if future else 1.0)
            costs.append(result["total_gpu_seconds"])
        return CandidatePrediction(
            layout,
            True,
            min(fractions) if all(f is not None for f in fractions) else None,
            sum(costs) / len(costs),
            work_ids,
            "Fixed serving window; score = switching GPU seconds + actual target residency integral; lifecycle enabled"
            if historical and self.config.actor_lifecycle_enabled
            else "Fixed serving window; score = switching GPU seconds + window seconds * target GPU count; target tail not scored"
            if historical
            else "Sampled arrivals close at horizon; all sampled serving and full training batches drain afterwards",
            self.models.source_ids,
            sum(window_costs) / len(window_costs) if historical else None,
            0.0 if historical else None,
            coverage,
            switching_gpu_seconds=sum(switching_costs) / len(costs)
            if historical
            else None,
            running_gpu_seconds_per_second=sum(running_costs) / sum(horizons)
            if historical
            else None,
            evaluation_seconds=sum(horizons) / len(costs) if historical else None,
            running_gpu_seconds=sum(running_costs) / len(costs) if historical else None,
            score_gpu_seconds=sum(costs) / len(costs) if historical else None,
            completed_requests=sum(s.completed for s in window_stats)
            if historical
            else None,
            pending_requests=sum(s.pending for s in window_stats)
            if historical
            else None,
            failed_pending_requests=sum(s.decided - s.completed for s in window_stats)
            if historical
            else None,
            completed_updates=sum(
                r["completed_updates"] - view.training_version for r in window_results
            )
            if historical
            else None,
            retained_episodes=sum(r["retained_episodes"] for r in window_results)
            if historical
            else None,
            actor_supply=supply,
            admission_credits_start=(
                (self.config.staleness + view.provider_version + 1)
                * self.config.batch_size
                - sum(e.admitted for e in view.episodes)
            )
            if historical
            else None,
            admission_credits_end=sum(r["admission_credits"] for r in window_results)
            / len(window_results)
            if historical
            else None,
            admission_credit_blocked_seconds=sum(
                r["admission_credit_blocked_seconds"] for r in window_results
            )
            / len(window_results)
            if historical
            else None,
        )

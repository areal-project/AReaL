# SPDX-License-Identifier: Apache-2.0

"""CPU prediction at a published update boundary, using frozen Reference components."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Executor
from dataclasses import dataclass, field, replace
from functools import cache, lru_cache
from heapq import heapify, heappop, heappush
from itertools import count
from math import isfinite
from pathlib import Path

from ..optimizer import Layout, Prediction
from .cost_model.reconfiguration.model import Cutover, ReconfigurationModel
from .workload import (
    Backlog,
    ExecutionConfig,
    Forecast,
    InflightCall,
    OnlineObservation,
    ServingSLO,
    WorkloadPattern,
)


@dataclass(frozen=True)
class Observation:
    view: OnlineObservation
    current: Layout
    backlog: Backlog | None = None
    search_ready_at: float | None = None  # earliest time a decision can be applied


@dataclass(frozen=True)
class _CandidateJob:
    index: int
    model_spec: tuple
    source_ids: tuple[str, ...]
    context: tuple[Observation, Forecast]
    config: ExecutionConfig
    layout: Layout
    cut: Cutover
    publication: float
    slo: ServingSLO
    settings: dict


@lru_cache(maxsize=1)
def _worker_models(spec, expected_ids):
    from .cost_model.loader import Catalog
    from .cost_model.models import AnalyticalCostModels

    (
        kind,
        hardware,
        topology,
        model,
        actor,
        rollout,
        bandwidth,
        link,
        root,
        calibration,
        actor_runtime,
    ) = spec
    if kind != "analytical":
        raise ValueError(
            "Parallel evaluation requires an analytical model specification"
        )
    models = AnalyticalCostModels.from_profiles(
        hardware=hardware,
        topology=topology,
        model=model,
        actor=actor,
        rollout=rollout,
        bandwidth_GBps=bandwidth,
        inter_node_GBps=link,
        catalog=Catalog(Path(root)),
        calibration=Path(calibration) if calibration else None,
        actor_runtime=actor_runtime,
    )
    if models.source_ids != expected_ids:
        raise ValueError("CPU worker loaded different cost models or source revisions")
    return models


def _evaluate_candidate(job: _CandidateJob) -> tuple[int, Prediction]:
    models = _worker_models(job.model_spec, job.source_ids)
    evaluator = ReplayEvaluator(models, None, job.config, job.slo, **job.settings)
    try:
        prediction = evaluator._predict(
            *job.context, job.config, job.layout, job.cut, job.publication, None
        )
    except ValueError as exc:
        prediction = Prediction(
            job.layout, None, None, None, supported=False, reason=str(exc)
        )
    return job.index, prediction


@dataclass
class Call:
    group_id: str
    index: int
    order: int
    ready_at: float
    request: object
    prefilled: int = 0
    generated: int = 0
    retained: int = 0
    first: float | None = None

    @property
    def kv_tokens(self):
        return self.prefilled + max(0, self.generated - 1)

    def resume(self):
        """Re-prefill acknowledged output as context on a fresh backend."""
        self.retained += self.generated
        self.request = replace(
            self.request,
            input_tokens=self.request.input_tokens + self.generated,
            output_tokens=self.request.output_tokens - self.generated,
        )
        self.prefilled = self.generated = 0


@dataclass
class DecodeSpan:
    first_end: float
    first_seconds: float
    increment: float
    limit: int
    # Routing can add residents while the current decode batch is in flight.
    resident_count: int = 0

    def end(self, steps):
        return (
            self.first_end
            + (steps - 1) * self.first_seconds
            + self.increment * steps * (steps - 1) / 2
        )

    def steps_before(self, boundary):
        # Leave an ordinary iteration in flight and handle ties on the event heap.
        low, high = 0, self.limit - 1
        while low < high:
            middle = (low + high + 1) // 2
            if self.end(middle) < boundary - 1e-9:
                low = middle
            else:
                high = middle - 1
        return low


@dataclass
class Replica:
    rows: dict = field(default_factory=dict)
    busy: bool = False
    decode_span: DecodeSpan | None = None

    @property
    def kv_tokens(self):
        return sum(r.kv_tokens for r in self.rows.values())

    @property
    def reserved_tokens(self):
        return sum(
            r.request.input_tokens + r.request.output_tokens - 1
            for r in self.rows.values()
        )


class ReplayEvaluator:
    """Predict candidates, optionally using a caller-owned CPU process pool."""

    def __init__(
        self,
        models,
        forecaster,
        template,
        slo,
        *,
        batch_size,
        concurrency,
        staleness,
        cutovers=None,
        reconfiguration: ReconfigurationModel | None = None,
        memory_reserve_fraction=0.0,
        executor: Executor | None = None,
    ):
        from .cost_model.rollout.sglang import scheduler_for
        from .workload import Request

        self.models, self.forecaster, self.template, self.slo = (
            models,
            forecaster,
            template,
            slo,
        )
        self.batch_size, self.concurrency, self.staleness = (
            batch_size,
            concurrency,
            staleness,
        )
        self.cutovers = {} if cutovers is None else cutovers
        self.reconfiguration = reconfiguration
        self.reserve = memory_reserve_fraction
        self.executor = executor
        self.request_type, self.scheduler = Request, scheduler_for("sglang")

    def prepare(self, observation, *, pattern: WorkloadPattern | None = None):
        if pattern is not None:
            forecast = pattern.forecast(
                observation.view.now, self.forecaster.horizon_seconds
            )
        else:
            forecasts = self.forecaster.predict(observation.view, 0)
            if len(forecasts) > 1:
                raise ValueError("Expected deterministic HistoricalReplay")
            forecast = next(iter(forecasts), None)
        context = observation, forecast
        if observation.search_ready_at is None or not forecast or not forecast.episodes:
            return context
        cfg, cut, pub = self._prepare_candidate(context, observation.current)
        # Run the common prefix once. Only time and work after publication
        # can differ in cost across candidates.
        boundary = self._predict(
            *context,
            cfg,
            observation.current,
            cut,
            pub,
            None,
            stop_at_publication=observation.search_ready_at,
        )
        if isinstance(boundary, Prediction):
            return observation, None  # no usable publication in the forecast horizon
        return boundary

    def evaluate(self, context, layout, *, on_batch=None):
        try:
            cfg, cut, pub = self._prepare_candidate(context, layout)
            return self._predict(*context, cfg, layout, cut, pub, on_batch)
        except ValueError as exc:
            return Prediction(
                layout, None, None, None, supported=False, reason=str(exc)
            )

    def evaluate_many(self, context, layouts):
        """Prepare cuts in the parent and return predictions in submitted order."""
        layouts = tuple(layouts)
        if self.executor is None or len(layouts) <= 1:
            return tuple(self.evaluate(context, layout) for layout in layouts)
        spec = getattr(self.models, "worker_spec", None)
        if spec is None:
            raise ValueError("Parallel evaluation requires profile-backed cost models")
        predictions, jobs = [None] * len(layouts), []
        settings = dict(
            batch_size=self.batch_size,
            concurrency=self.concurrency,
            staleness=self.staleness,
            memory_reserve_fraction=self.reserve,
        )
        for index, layout in enumerate(layouts):
            try:
                cfg, cut, pub = self._prepare_candidate(context, layout)
            except ValueError as exc:
                predictions[index] = Prediction(
                    layout, None, None, None, supported=False, reason=str(exc)
                )
            else:
                jobs.append(
                    _CandidateJob(
                        index,
                        spec,
                        self.models.source_ids,
                        context,
                        cfg,
                        layout,
                        cut,
                        pub,
                        self.slo,
                        settings,
                    )
                )
        for index, prediction in self.executor.map(_evaluate_candidate, jobs):
            predictions[index] = prediction
        return tuple(predictions)

    def _prepare_candidate(self, context, layout):
        obs, forecast = context
        if forecast is None:
            raise ValueError("No completed workload history")
        cfg = replace(
            self.template,
            actor_backend=layout.actor.backend,
            rollout_backend=layout.rollout.backend,
            **dict(layout.actor.options),
            **dict(layout.rollout.options),
        )
        if (layout.actor.gpus, layout.rollout.gpus) != (
            cfg.actor_gpus,
            cfg.rollout_gpus,
        ):
            raise ValueError("Backend dimensions disagree with allocated GPU counts")
        pub = self.models.publication(cfg, (cfg,))
        if not isfinite(pub) or pub < 0:
            raise ValueError("Unknown publication duration")
        cut = (
            Cutover(0, 0)
            if layout == obs.current
            else self.cutovers.get((obs.current, layout))
        )
        if cut is None:
            if self.reconfiguration is None:
                raise ValueError("Unknown directed cutover cost")
            cut = self.reconfiguration.predict(obs.current, layout, pub)
        if cut.seconds > forecast.input_closes_at - obs.view.now:
            raise ValueError(
                "Cutover extends beyond the forecast; partial residency unknown"
            )
        return cfg, cut, pub

    def _predict(
        self,
        obs,
        forecast,
        cfg,
        layout,
        cut,
        pub,
        on_batch,
        *,
        stop_at_publication=None,
    ):
        view, size = obs.view, self.batch_size
        if view.busy or view.provider_version != view.training_version:
            raise ValueError("Expected an idle Actor at the published update boundary")
        end, start = forecast.input_closes_at, view.now
        rc = self.models.rollout_config(cfg)
        shapes = {e.episode_id: e.turns for e in forecast.episodes}

        @cache
        def actor_time(batch):
            peak, cap, oom = self.models.actor_memory(cfg, batch)
            if oom or peak > cap * (1 - self.reserve):
                raise ValueError(
                    f"Batch exceeds Actor memory allowance (peak={peak:.2f} GiB, "
                    f"allowance={cap * (1 - self.reserve):.2f} GiB, oom={oom})"
                )
            stages = self.models.actor(cfg, batch)
            duration = sum(t for _, t in stages)
            if not stages or not isfinite(duration) or duration <= 0:
                raise ValueError("Unknown Actor duration")
            return duration

        templates = [episode.turns for episode in forecast.episodes]
        deficit = 0.0
        if templates:
            rate = len(templates) / (end - start)
            weighted = 0.0
            for offset in range(0, len(templates), size):
                batch = tuple(
                    templates[(offset + i) % len(templates)] for i in range(size)
                )
                weighted += min(size, len(templates) - offset) * actor_time(batch)
            deficit = rate - size / (weighted / len(templates) + pub)

        # All state below belongs to this candidate. Simultaneous completions are
        # applied before the next batch is chosen or new inference work is scheduled.
        ready = deque()
        arrivals, waiting = deque(), {}
        replicas = {i: Replica() for i in range(rc.replicas)}
        events, serial = [], count()
        active = set()
        accepted = 0
        passed = completed = updates = 0
        unknown_timing = actor_busy = False
        paused = cut.seconds > 0
        generation = 0
        now = start
        interval = getattr(self.models, "decode_interval", None)
        # Full-output reservations stay fixed until admission, completion or pause.
        routing_dirty = True

        def push(at, kind, payload=None):
            heappush(events, (at, next(serial), kind, payload))

        def add_call(key, i, at):
            nonlocal routing_dirty
            routing_dirty = True
            t = shapes[key][i]
            req = self.request_type(f"turn-{i}", t.input_tokens, t.output_tokens)
            call = Call(key, i, next(serial), at, req)
            waiting[key] = call
            return call

        # Only the next arrival bounds an interval. Preserve the original event
        # serials for equal timestamps even when input is supplied out of order.
        offers = iter(
            sorted(
                (
                    e.available_at,
                    next(serial),
                    "arrival",
                    (e.episode_id, e.available_at),
                )
                for e in forecast.episodes
            )
        )

        def enqueue_offer():
            event = next(offers, None)
            if event is not None:
                heappush(events, event)

        backlog = obs.backlog
        if backlog is not None:
            shapes.update(backlog.shapes)
            arrivals.extend(backlog.queued)
            ready.extend(backlog.ready)
            accepted = len(ready)
            passed, completed = backlog.passed_calls, backlog.completed_calls
            unknown_timing = backlog.unknown_timing
            for progress in backlog.running:
                key = progress.episode_id
                call = add_call(key, progress.turn, progress.ready_at)
                call.first = progress.first_token_at
                call.prefilled, call.generated = progress.prefilled, progress.generated
                call.retained = progress.retained
                call.request = replace(
                    call.request,
                    input_tokens=call.request.input_tokens + call.retained,
                    output_tokens=call.request.output_tokens - call.retained,
                )
                active.add(key)
                if cut.seconds:
                    call.resume()
                elif progress.replica is not None:
                    replicas[progress.replica].rows[key] = waiting.pop(key)
            for key, turn, at in backlog.tools:
                active.add(key)
                push(at, "tool", (key, turn))
            if backlog.actor_event is not None:
                kind, remaining = backlog.actor_event
                accepted += size
                actor_busy = True
                paused = paused or kind == "published"
                push(start + cut.seconds + remaining, kind)

        enqueue_offer()
        if paused:
            push(start + cut.seconds, "resume")
        push(end, "end")

        def simulate_step():
            nonlocal routing_dirty
            limit = min(
                self.concurrency,
                cfg.rollout_max_concurrent_rollouts or size,
            )
            while (
                arrivals
                and len(active) < limit
                and accepted + len(active) < (updates + 1 + self.staleness) * size
            ):
                key, at = arrivals.popleft()
                active.add(key)
                add_call(key, 0, at)
            if routing_dirty:
                for key, call in tuple(waiting.items()):
                    rid = self.scheduler.choose_replica(rc, call, replicas)
                    if rid is not None:
                        replicas[rid].rows[key] = waiting.pop(key)
                routing_dirty = False
            for rid, replica in replicas.items():
                if replica.rows and not replica.busy:
                    slices, keys = self.scheduler.batch(rc, list(replica.rows.values()))
                    if not slices:
                        raise ValueError(
                            "Serving profile cannot schedule the pending tokens"
                        )
                    replica.decode_span = None
                    if interval is not None and all(s.kind == "decode" for s in slices):
                        remaining = min(
                            replica.rows[key].request.output_tokens
                            - replica.rows[key].generated
                            for key, _ in keys
                        )
                        duration, slope, steps = interval(
                            cfg, replica.kv_tokens, slices, remaining
                        )
                        if (
                            steps > 1
                            and isfinite(slope)
                            and duration + (steps - 1) * slope > 0
                        ):
                            replica.decode_span = DecodeSpan(
                                now + duration,
                                duration,
                                slope,
                                steps,
                                resident_count=len(replica.rows),
                            )
                    else:
                        duration = self.models.service(cfg, replica.kv_tokens, slices)
                    if not isfinite(duration) or duration <= 0:
                        raise ValueError("Unknown inference duration")
                    replica.busy = True
                    push(
                        now + duration,
                        "tokens",
                        (generation, rid, tuple(zip(keys, slices))),
                    )

        def advance_decode():
            # Stop at the earliest event that could change routing, batch membership,
            # policy version, or the cost model's segment on any replica.
            boundary = end
            spans = []
            for index, (when, _, kind, payload) in enumerate(events):
                if kind == "tokens":
                    gen, rid, items = payload
                    replica = replicas.get(rid)
                    span = (
                        replica.decode_span if gen == generation and replica else None
                    )
                    if span is not None and span.resident_count == len(replica.rows):
                        spans.append((index, rid, span, items))
                        boundary = min(boundary, span.end(span.limit))
                        continue
                boundary = min(boundary, when)

            moved = []
            for index, rid, span, items in spans:
                steps = span.steps_before(boundary)
                if steps:
                    for (key, _), _ in items:
                        replicas[rid].rows[key].generated += steps
                    moved.append((span.end(steps), rid, index))
                    span.first_end = span.end(steps + 1)
                    span.first_seconds += steps * span.increment
                    span.limit -= steps
            if moved:
                # The next token would have been scheduled after the last skipped
                # completion. Replica order breaks ties as in simulate_step().
                for _, rid, index in sorted(moved):
                    payload = events[index][3]
                    events[index] = (
                        replicas[rid].decode_span.first_end,
                        next(serial),
                        "tokens",
                        payload,
                    )
                heapify(events)

        while True:
            # Process all events at now, including newly scheduled zero-length tools.
            while events and events[0][0] <= now:
                _, _, kind, payload = heappop(events)
                if kind == "arrival":
                    arrivals.append(payload)
                    enqueue_offer()
                elif kind == "tool":
                    add_call(*payload, now)
                elif kind == "actor":
                    routing_dirty = True
                    paused = True
                    generation += 1
                    for replica in replicas.values():
                        for key, call in replica.rows.items():
                            call.resume()
                            waiting[key] = call
                        replica.rows.clear()
                        replica.busy = False
                    push(now + pub, "published")
                elif kind == "published":
                    paused = actor_busy = False
                    updates += 1
                    if stop_at_publication is not None and now >= stop_at_publication:
                        if now >= end:
                            return obs, None
                        # Publishing flushed replica KV. Calls waiting for the
                        # new weights still own their acknowledged output.
                        running = tuple(
                            InflightCall(
                                c.group_id,
                                c.index,
                                c.ready_at,
                                c.first,
                                c.prefilled,
                                c.generated,
                                c.retained,
                            )
                            for c in waiting.values()
                        )
                        tools = tuple(
                            (*item, at)
                            for at, _, event, item in events
                            if event == "tool"
                        )
                        keys = set(active) | set(ready) | {k for k, _ in arrivals}
                        carry = Backlog(
                            {k: shapes[k] for k in keys},
                            tuple(arrivals),
                            running,
                            tools,
                            tuple(ready),
                            passed_calls=passed,
                            completed_calls=completed,
                            unknown_timing=unknown_timing,
                        )
                        unseen = {
                            item[0]
                            for _, _, event, item in events
                            if event == "arrival"
                        }
                        unseen.update(item[0] for _, _, _, item in offers)
                        rest = tuple(
                            e for e in forecast.episodes if e.episode_id in unseen
                        )
                        version = view.training_version + updates
                        next_view = replace(
                            view,
                            now=now,
                            training_version=version,
                            provider_version=version,
                        )
                        return (
                            Observation(next_view, layout, carry),
                            replace(forecast, episodes=rest),
                        )
                elif kind == "resume":
                    paused = any(event[2] == "published" for event in events)
                elif kind == "tokens":
                    gen, rid, items = payload
                    if gen != generation:
                        continue
                    replica = replicas[rid]
                    replica.busy = False
                    for (key, _), item in items:
                        call = replica.rows[key]
                        if item.kind == "prefill":
                            call.prefilled += item.query_tokens
                        if not item.emit_token:
                            continue
                        if call.generated + call.retained == 0:
                            call.first = now
                        call.generated += 1
                        if call.generated != call.request.output_tokens:
                            continue
                        del replica.rows[key]
                        routing_dirty = True
                        completed += 1
                        total = call.generated + call.retained
                        if call.first is None:
                            unknown_timing = True
                        else:
                            passed += (
                                call.first - call.ready_at <= self.slo.ttft_seconds
                                and (
                                    total == 1
                                    or (now - call.first) / (total - 1)
                                    <= self.slo.tpot_seconds
                                )
                            )
                        t = shapes[key][call.index]
                        if t.tool_seconds is None:
                            ready.append(key)
                            active.remove(key)
                            accepted += 1
                        else:
                            push(now + t.tool_seconds, "tool", (key, call.index + 1))
            if now >= end:
                break
            if not paused:
                if len(ready) >= size and not actor_busy:
                    ids = tuple(ready.popleft() for _ in range(size))
                    duration = actor_time(tuple(shapes[key] for key in ids))
                    if on_batch is not None:
                        on_batch(now, ids)
                    push(now + duration, "actor")
                    actor_busy = True
                simulate_step()
                if interval is not None:
                    advance_decode()
            now = events[0][0]

        pending_calls = [
            *waiting.values(),
            *(c for r in replicas.values() for c in r.rows.values()),
        ]
        # Offered first calls held by permits still contribute to prospective TTFT.
        failures = sum(end - at > self.slo.ttft_seconds for _, at in arrivals)
        failures += sum(
            (end if c.first is None else c.first) - c.ready_at > self.slo.ttft_seconds
            for c in pending_calls
            if c.first is not None or not (c.generated + c.retained)
        )
        decided = completed + failures
        return Prediction(
            layout,
            passed / decided if decided and not unknown_timing else None,
            max(
                (end - start) * layout.gpus,
                cut.gpu_seconds + (end - start - cut.seconds) * layout.gpus,
            ),
            deficit <= 1e-12,
            max(0.0, deficit),
            completed_updates=updates,
            pending_requests=len(pending_calls) + len(arrivals),
            completed_requests=completed,
        )

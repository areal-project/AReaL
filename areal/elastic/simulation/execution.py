# SPDX-License-Identifier: Apache-2.0

"""Execute request, tool, training and publication events in one experiment."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from areal.elastic.contracts import (
    ActorMemoryRejected,
    ActorWaitView,
    RequestView,
    ServiceView,
    ToolView,
)
from areal.elastic.reconfigurator.reference.evaluator.cost_model.rollout.scheduling import (
    scheduler_for,
)
from areal.elastic.serving import request_tpot_seconds

from .state import _Episode, _Request


class SimulationExecution:
    def __init__(self, simulation):
        self.sim = simulation

    def arrive(self, identity: str):
        sim = self.sim
        truth = sim._truth[identity]
        sim.episodes[identity] = _Episode(identity)
        sim._episode_order[identity] = len(sim.arrival_order)
        sim.arrival_order.append(identity)
        sim.execution.new_request(sim.episodes[identity], truth.available_at)
        sim.emit("episode_arrival", episode=identity, family=truth.family)

    def new_request(self, episode: _Episode, ready_at: float):
        sim = self.sim
        index = len(episode.past)
        turn = sim._truth[episode.episode_id].turns[index]
        if (
            turn.output_tokens > sim.config.max_new_tokens
            or turn.input_tokens + turn.output_tokens > sim.config.max_context_tokens
        ):
            raise ValueError(
                "Workload exceeds the explicitly configured generation/context budget"
            )
        identity = f"{episode.episode_id}/request-{index}"
        view = RequestView(
            identity, episode.episode_id, index, turn.input_tokens, ready_at
        )
        reservation = min(
            sim.config.max_new_tokens,
            sim.config.max_context_tokens - turn.input_tokens,
        )
        sim.requests[identity] = _Request(view, reservation, len(sim.requests))
        sim._live_requests[identity] = sim.requests[identity]
        sim._unassigned[identity] = sim.requests[identity]
        episode.current = identity
        sim.emit(
            "request_ready",
            request=identity,
            input_tokens=turn.input_tokens,
            ready_at=ready_at,
        )

    def tick(self, local_services: set[tuple[int, int]] | None = None):
        sim = self.sim
        if local_services is not None:
            sim.local_service_ticks += 1
            sim.reconfiguration.service(local_services)
            return
        if sim.publication == "pending" and (
            not any(d.busy for d in sim.deployments.values())
        ):
            sim.execution.start_publication()
        sim.reconfiguration.before_work()
        if sim.publication == "idle":
            sim.reconfiguration.admit()
            sim.execution.start_tools()
            sim.reconfiguration.service()
            sim.reconfiguration.start_actor()
        else:
            sim.execution.start_tools()
        sim.reconfiguration.after_work()
        if (
            sim.input_closed
            and len(sim.episodes) == len(sim._truth)
            and (sim._completed_episodes == len(sim.episodes))
            and (not sim.actor_batch)
            and (sim.publication == "idle")
            and (sim.pending_handoff is None)
            and (not sim.decision_pending)
            and (sim._restoring_actor is None)
            and (len(sim.arrival_order) - sim.consumed < sim.config.batch_size)
        ):
            if sim.control is not None:
                sim.control.cancel()
            for identity in tuple(sim.deployments):
                sim.execution.release_deployment(identity)
            sim.done = True

    def admit(self):
        sim = self.sim
        active = sim._active_episodes
        credits = sim.admission_credits()
        while sim._admission_cursor < len(sim.arrival_order):
            identity = sim.arrival_order[sim._admission_cursor]
            episode = sim.episodes[identity]
            if active >= sim.config.client_concurrency or credits <= 0:
                break
            sim._admission_cursor += 1
            episode.admitted = True
            active += 1
            credits -= 1
            sim.emit("episode_admitted", episode=identity)
        sim._active_episodes = active

    def service(self, local_services: set[tuple[int, int]] | None = None):
        sim = self.sim
        target = sim.deployments[sim.rollout_id]
        cfg = sim.cost_models.rollout_config(target.layout)
        unassigned = sim._unassigned.values() if local_services is None else ()
        for request in sorted(unassigned, key=lambda r: r.order):
            view = request.view
            if not sim.episodes[view.episode_id].admitted:
                continue
            replica = scheduler_for(cfg.backend).choose_replica(
                cfg, request, target.replicas
            )
            if replica is not None:
                view.deployment, view.replica = (target.identity, replica)
                target.replicas[replica].add(request)
                del sim._unassigned[view.request_id]
                sim.emit(
                    "request_assigned",
                    request=view.request_id,
                    deployment=target.identity,
                    replica=replica,
                )
        for deployment in tuple(sim.deployments.values()):
            if deployment.role != "rollout" or deployment.status not in {
                "active",
                "draining",
            }:
                continue
            cfg = sim.cost_models.rollout_config(deployment.layout)
            for replica in range(cfg.replicas):
                if (
                    local_services is not None
                    and (deployment.identity, replica) not in local_services
                ):
                    continue
                if replica in deployment.busy:
                    continue
                group = deployment.replicas[replica]
                if not group.rows:
                    continue
                plan = group.decode_plan
                continuation = (
                    sim.coalesce_decode
                    and sim.sink is None
                    and (not sim.prediction_guard_active())
                    and (plan is not None)
                    and plan.continuation
                    and (plan.revision == group.revision)
                    and (plan.version == deployment.version)
                    and (plan.limit > 0)
                    and (plan.first_seconds > 0)
                    and (
                        group.kv_tokens + len(plan.batch.slices)
                        <= cfg.kv_capacity_tokens
                    )
                )
                if continuation:
                    duration = plan.first_seconds
                    rows = tuple(
                        (
                            (key, context + 1, query, phase, emits)
                            for key, context, query, phase, emits in plan.batch.slices
                        )
                    )
                    sim.decode_batch_reuses += 1
                else:
                    group.flush_decode()
                    slices, _ = scheduler_for(cfg.backend).batch(
                        cfg, list(group.rows.values())
                    )
                    duration = sim.cost_models.service(
                        deployment.layout, group.kv_tokens, slices
                    )
                    rows = tuple(
                        (
                            s.request_id,
                            s.context_tokens,
                            s.query_tokens,
                            s.kind,
                            s.emit_token,
                        )
                        for s in slices
                    )
                batch = ServiceView(replica, sim.now, rows, deployment.version)
                if continuation:
                    plan.batch = batch
                    plan.continuation = False
                deployment.busy[replica] = batch
                sim.schedule_at(
                    sim.now + duration,
                    "service_complete",
                    (deployment.identity, replica),
                )

    def finish_service(self, identity: int, replica: int):
        sim = self.sim
        deployment = sim.deployments[identity]
        batch = deployment.busy.pop(replica)
        group = deployment.replicas[replica]
        plan = group.decode_plan
        defer = (
            sim.coalesce_decode
            and sim.sink is None
            and (not sim.prediction_guard_active())
            and (plan is not None)
            and (plan.batch is batch)
            and (plan.revision == group.revision)
            and (plan.version == deployment.version)
            and (plan.limit > 1)
        )
        if defer:
            group.defer_decode(plan, 1, sim.now, sim.now, 0.0)
            group.kv_tokens += len(batch.slices)
            sim.counters["token_generated"] += len(batch.slices)
        else:
            group.flush_decode()
            for key, _, query, phase, emit_token in batch.slices:
                request = sim.requests[key]
                view = request.view
                kv_delta = query if phase == "prefill" else 0
                if phase == "prefill":
                    view.prefilled += query
                    group.revision += 1
                if emit_token:
                    if view.generated:
                        kv_delta += 1
                    view.add_tokens(
                        1,
                        batch.version,
                        sim.now,
                        sim.now - view.last_token_at
                        if view.last_token_at is not None
                        else 0.0,
                    )
                    sim.counters["token_generated"] += 1
                group.kv_tokens += kv_delta
                truth = sim._truth[view.episode_id].turns[view.turn]
                if view.generated == truth.output_tokens:
                    sim.execution.finish_request(request)
        if (
            sim.coalesce_decode
            and sim.sink is None
            and (not sim.prediction_guard_active())
            and (plan is not None)
            and (plan.batch is batch)
            and (plan.revision == group.revision)
            and (plan.version == deployment.version)
            and (plan.limit > 0)
        ):
            plan.first_seconds += plan.increment
            plan.limit -= 1
            plan.continuation = plan.limit > 0
        sim.emit(
            "service_batch",
            deployment=identity,
            replica=replica,
            version=batch.version,
            start=batch.started_at,
            end=sim.now,
            slices=batch.slices,
        )

    def finish_request(self, request: _Request):
        sim = self.sim
        view = request.view
        view.completed_at = sim.now
        sim.deployments[view.deployment].replicas[view.replica].remove(request)
        del sim._live_requests[view.request_id]
        episode = sim.episodes[view.episode_id]
        truth = sim._truth[view.episode_id].turns[view.turn]
        sim.emit(
            "request_complete",
            request=view.request_id,
            deployment=view.deployment,
            generated=view.generated,
            ttft_seconds=view.first_token_at - view.ready_at,
            mean_tpot_seconds=request_tpot_seconds(view),
        )
        if truth.tool_seconds is None:
            episode.past.append(truth)
            episode.current = None
            episode.completed_at = sim.now
            sim._active_episodes -= 1
            sim._completed_episodes += 1
            sim.emit("episode_complete", episode=episode.episode_id)
        else:
            episode.tool = ToolView(truth.tool_pool, sim.now, None)
            sim._tool_episodes[episode.episode_id] = episode

    def start_tools(self):
        sim = self.sim
        waiting = sorted(
            sim._tool_episodes.values(),
            key=lambda e: sim._episode_order[e.episode_id],
        )
        for pool, capacity in sim.config.tool_capacities:
            active = sum(
                e.tool is not None
                and e.tool.pool == pool
                and (e.tool.started_at is not None)
                for e in waiting
            )
            for episode in waiting:
                tool = episode.tool
                if active >= capacity:
                    break
                if tool is None or tool.pool != pool or tool.started_at is not None:
                    continue
                episode.tool = replace(tool, started_at=sim.now)
                turn = sim._truth[episode.episode_id].turns[len(episode.past)]
                sim.schedule_at(
                    sim.now + float(turn.tool_seconds),
                    "tool_complete",
                    episode.episode_id,
                )
                sim.emit("tool_start", episode=episode.episode_id, pool=pool)
                active += 1

    def finish_tool(self, identity: str):
        sim = self.sim
        episode = sim.episodes[identity]
        truth = sim._truth[identity].turns[len(episode.past)]
        sim.emit(
            "tool_complete",
            episode=identity,
            pool=episode.tool.pool,
            seconds=sim.now - episode.tool.started_at,
            output_tokens=truth.tool_output_tokens,
        )
        episode.past.append(truth)
        episode.tool = None
        del sim._tool_episodes[identity]
        sim.execution.new_request(episode, sim.now)

    def start_actor(self):
        sim = self.sim
        if sim._restoring_actor is not None:
            return
        if (
            sim.training_limit is not None
            and sim.training_version >= sim.training_limit
        ):
            return
        if sim.actor_batch or sim.publication != "idle":
            return
        pending = sim.pending_handoff
        if pending and (pending.get("actor_ready") or pending.get("rollout_fence")):
            return
        batch = tuple(
            sim.arrival_order[sim.consumed : sim.consumed + sim.config.batch_size]
        )
        if len(batch) != sim.config.batch_size or any(
            sim.episodes[e].completed_at is None for e in batch
        ):
            return
        if sim.actor_id not in sim.deployments:
            identity = sim.next_deployment
            if (
                sim.pool.acquire(
                    str(identity),
                    "actor",
                    sim.layout.actor_gpus,
                    tp=sim.layout.actor_tp,
                )
                is None
            ):
                raise ValueError("Initial Actor does not fit the declared GPU pool")
            sim.next_deployment += 1
            sim._restoring_actor = identity
            sim.reconfiguration.add_restoring_actor(identity)
            sim.emit("actor_restore_start", deployment=identity)
            sim.schedule_at(
                sim.now + sim.handoff.actor_startup_seconds,
                "process_started",
                identity,
            )
            return
        wait = sim.actor_wait
        if wait and wait.deployment == sim.actor_id and (wait.batch == batch):
            sim.reconfiguration.reject_unrepairable_forecast_wait()
            return
        try:
            stages = sim.cost_models.actor(
                sim.layout, tuple(tuple(sim.episodes[e].past) for e in batch)
            )
        except ActorMemoryRejected as exc:
            sim.actor_wait = ActorWaitView(
                batch, sim.actor_id, wait.since if wait else sim.now, str(exc)
            )
            sim.emit(
                "actor_wait",
                deployment=sim.actor_id,
                episodes=batch,
                cursor=sim.consumed,
                version=sim.training_version,
                since=sim.actor_wait.since,
                reason=str(exc),
            )
            sim.reconfiguration.reject_unrepairable_forecast_wait()
            return
        if not stages:
            raise ValueError("Actor model did not supply update stages")
        if wait:
            seconds = sim.now - wait.since
            sim.actor_wait_seconds += seconds
            sim.emit(
                "actor_wait_complete",
                deployment=sim.actor_id,
                previous_deployment=wait.deployment,
                episodes=batch,
                cursor=sim.consumed,
                version=sim.training_version,
                since=wait.since,
                seconds=seconds,
            )
            sim.actor_wait = None
        sim.actor_batch, sim.actor_stages = (batch, stages)
        versions = Counter()
        for request in sim.requests.values():
            if request.view.episode_id in batch:
                versions.update(dict(request.view.version_counts))
        sim.training_records.append(
            dict(
                step=sim.training_version,
                started_at=sim.now,
                preupdate_version=sim.training_version,
                episodes=batch,
                behavior_versions=dict(versions),
                maximum_version_lag=sim.training_version - min(versions),
                committed_at=None,
            )
        )
        sim.emit(
            "actor_start",
            deployment=sim.actor_id,
            episodes=batch,
            version=sim.training_version,
            cursor=sim.consumed,
        )
        sim.actor_phase = 0
        sim.execution.start_actor_phase()

    def start_actor_phase(self):
        sim = self.sim
        name, seconds = sim.actor_stages[sim.actor_phase]
        sim.actor_phase_started = sim.now
        sim.emit("actor_phase_start", deployment=sim.actor_id, phase=name)
        sim.schedule_at(sim.now + seconds, "actor_phase_complete")

    def finish_actor_phase(self):
        sim = self.sim
        name, _ = sim.actor_stages[sim.actor_phase]
        sim.emit("actor_phase_complete", deployment=sim.actor_id, phase=name)
        sim.actor_phase += 1
        if sim.actor_phase < len(sim.actor_stages):
            sim.execution.start_actor_phase()
            return
        sim.training_version += 1
        sim.deployments[sim.actor_id].version = sim.training_version
        sim.consumed += len(sim.actor_batch)
        sim.training_records[-1]["committed_at"] = sim.now
        sim.actor_batch = ()
        sim.publication = "pending"
        sim.emit("actor_complete", version=sim.training_version, cursor=sim.consumed)

    def start_publication(self):
        sim = self.sim
        recipients = tuple(
            d
            for d in sim.deployments.values()
            if d.role == "rollout" and d.status in {"active", "draining"}
        )
        seconds = sim.cost_models.publication(
            sim.layout, tuple(d.layout for d in recipients)
        )
        start, end = sim.link.reserve(sim.now, fitted_seconds=seconds)
        for deployment in recipients:
            deployment.transfers += 1
        sim.publication = "running"
        sim.emit(
            "publication_start",
            version=sim.training_version,
            transfer_start=start,
            recipients=tuple(d.identity for d in recipients),
            ends_at=end,
        )
        sim.schedule_at(
            end,
            "publication_complete",
            (sim.training_version, tuple(d.identity for d in recipients)),
        )

    def finish_publication(self, data):
        sim = self.sim
        version, identities = data
        for identity in identities:
            deployment = sim.deployments[identity]
            for group in deployment.replicas.values():
                group.flush_decode()
            deployment.version = version
            deployment.transfers -= 1
        sim.provider_version = version
        sim.publication = "idle"
        sim.emit("weights_published", version=version, recipients=identities)
        if sim.control is not None:
            sim.control.publication_complete()
        sim.reconfiguration.publication_complete()

    def release_deployment(self, identity: int):
        sim = self.sim
        deployment = sim.deployments[identity]
        if deployment.busy or deployment.transfers:
            raise ValueError(
                "Cannot release a deployment with live GPU work or transfers"
            )
        for group in deployment.replicas.values():
            group.flush_decode()
        sim.pool.release(str(identity))
        del sim.deployments[identity]
        sim.emit(
            "deployment_release",
            deployment=identity,
            role=deployment.role,
            leases=dict(sim.pool.leases),
        )
        sim.reconfiguration.deployment_released(identity, deployment)

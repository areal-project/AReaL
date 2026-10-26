# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, dataclass, fields, replace

import pytest

from areal.elastic.contracts import WorkloadStats
from areal.elastic.policy import reference as expert
from areal.elastic.policy.basic import ElasticObservation, IdleTimeoutPolicy


def observation(**changes):
    return replace(
        ElasticObservation(
            state="ACTIVE",
            model_version=2,
            serving_version=2,
            training_active=True,
            batch_ready=False,
            idle_duration=5,
            gate_available=True,
            pending_publication=False,
            committed=True,
            observed_at=10,
        ),
        **changes,
    )


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, "release"),
        ({"idle_duration": 4.99}, "keep"),
        ({"training_active": False}, "keep"),
        ({"gate_available": False}, "release"),
        ({"batch_ready": True}, "train"),
        ({"batch_ready": True, "gate_available": False}, "keep"),
        ({"pending_publication": True}, "keep"),
        ({"pending_publication": True, "batch_ready": True}, "keep"),
        ({"committed": False}, "keep"),
        ({"committed": False, "batch_ready": True}, "keep"),
    ],
)
def test_idle_policy_decides_without_execution(changes, expected):
    assert IdleTimeoutPolicy(5).decide(observation(**changes)) == expected


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_idle_policy_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        IdleTimeoutPolicy(timeout)


def test_idle_policy_zero_timeout_releases_at_idle_boundary():
    """Zero disables waiting without altering publication or batch checks."""
    policy = IdleTimeoutPolicy(0)
    assert policy.decide(observation(idle_duration=0)) == "release"
    assert policy.decide(observation(idle_duration=0, batch_ready=True)) == "train"
    assert policy.decide(observation(idle_duration=0, committed=False)) == "keep"


def test_policy_selection_keeps_reference_explicitly_unimplemented():
    """Selecting Reference cannot silently execute Basic instead."""
    from areal.elastic.policy import build_policy

    assert build_policy("basic", idle_timeout=60) == IdleTimeoutPolicy(60)
    with pytest.raises(NotImplementedError, match="reference.*not implemented"):
        build_policy("reference", idle_timeout=60)
    with pytest.raises(ValueError, match="basic or reference"):
        build_policy("unknown", idle_timeout=60)


def test_policy_observation_is_immutable():
    sample = observation()
    with pytest.raises(FrozenInstanceError):
        sample.model_version = 3


# Expert reference simulation checks use explicit deterministic service durations.
# These are accounting oracles, not target-hardware calibration samples.
@dataclass(frozen=True)
class _FixedActor:
    seconds: float = 2.0

    def estimate(self, config, workload, *, auxiliary_stages=()):
        return expert.ActorEstimate(
            "fixed-actor",
            tuple(auxiliary_stages)
            + (("train_step", expert.Estimate(self.seconds, "assumed")),),
            (("sequence_tokens", workload.sequence_tokens),),
            None,
        )


@dataclass(frozen=True)
class _FixedRollout:
    seconds: float = 1.0

    def estimate(self, config, state, requests):
        return expert.ServiceEstimate(
            "fixed-rollout",
            expert.Estimate(self.seconds, "assumed"),
            (),
            state.kv_tokens + sum(request.query_tokens for request in requests),
        )


def _expert_config(actor_dp=1, replicas=1):
    return expert.ExecutionConfig(
        expert.ActorConfig(
            "archon", expert.ParallelLayout(dp=actor_dp), max_tokens_per_mb=128
        ),
        expert.RolloutConfig(replicas, 1, 8, 64, 4096),
        tuple(f"a{i}" for i in range(actor_dp)),
        tuple(f"r{i}" for i in range(replicas)),
    )


def _expert_group(name, *, accepted=True, external=0, output=1):
    nodes = (expert.Request("request", 8, output),)
    if external:
        nodes = (
            expert.ExternalCall("tool", "tools", expert.Estimate(external, "assumed")),
            expert.Request("request", 8, output, ("tool",)),
        )
    return expert.Group(
        name,
        nodes,
        WorkloadStats(8, 64, sequence_lengths=(8,) * 8),
        1,
        accepted=accepted,
    )


def _expert_transitions(config, *, extra=()):
    def task(seconds):
        return (
            expert.TransferTask("work", measured=expert.Estimate(seconds, "assumed")),
        )

    def change(role, action, when="end"):
        devices = config.actor_devices if role == "actor" else config.rollout_devices
        return expert.AllocationChange("work", when, action, role, devices)

    return expert.ExpertTransitionModel(
        (
            expert.TransitionProfile(
                "publish",
                config.key,
                config.key,
                task(3),
                blocks=("actor", "generation", "admission"),
            ),
            expert.TransitionProfile("checkpoint", config.key, config.key, task(2)),
            expert.TransitionProfile(
                "release", config.key, None, task(1), (change("actor", "release"),)
            ),
            expert.TransitionProfile(
                "start",
                None,
                config.key,
                task(2),
                (change("actor", "acquire", "start"),),
            ),
            expert.TransitionProfile(
                "restore",
                config.key,
                config.key,
                task(2),
                (change("actor", "acquire", "start"),),
            ),
            expert.TransitionProfile("cancel", config.key, config.key, task(0.5)),
            expert.TransitionProfile(
                "shutdown",
                config.key,
                None,
                task(1),
                (change("actor", "release"), change("rollout", "release")),
            ),
            expert.TransitionProfile(
                "shutdown_suspended",
                config.key,
                None,
                task(1),
                (change("rollout", "release"),),
            ),
        )
        + tuple(extra)
    )


def _expert_scenario(config, groups, **changes):
    initial = changes.pop("initial", expert.InitialState(config))
    values = dict(
        groups=tuple(groups),
        initial=initial,
        groups_per_batch=1,
        consumer_batch_size=1,
        samples_per_group=1,
        max_concurrent_groups=2,
        runner_concurrency=2,
        pending_limit=4,
        runner_queue_limit=4,
        staleness=1,
        external_pools=(("tools", 4),),
    )
    values.update(changes)
    return expert.Scenario(**values)


def _simulate(scenario, steps, transitions=None):
    models = expert.ExpertModels(
        _FixedActor(),
        _FixedRollout(),
        transitions or _expert_transitions(scenario.initial.config),
    )
    return expert.ExpertSimulator(models).simulate(scenario, tuple(steps))


@pytest.mark.parametrize("dp,replicas", [(1, 1), (8, 8)])
def test_expert_qwen4b_megatron_request_tool_chain_completes_with_explicit_priors(
    dp, replicas
):
    """Both roles remain node-local; fixture overheads are not target calibration."""
    bundle = expert.Catalog().assemble(
        hardware="h20",
        system="h20-actor8-rollout8",
        model="qwen3-4b-instruct-2507",
        actor="megatron-qwen3-4b-ppo",
        rollout="sglang-qwen3-4b",
        actor_parallel=expert.ParallelLayout(dp=dp),
        rollout_tp=1,
        rollout_replicas=replicas,
        kv_capacity_tokens=65536,
    )
    cfg = bundle.config
    assert len(cfg.actor_devices) <= 8 and len(cfg.rollout_devices) <= 8
    assert {d.split("/")[0] for d in cfg.actor_devices} == {"actor-node"}
    assert {d.split("/")[0] for d in cfg.rollout_devices} == {"rollout-node"}
    advantage = expert.AdvantageModel(
        "unit-only-gpu-advantage",
        "cuda",
        "single",
        "token",
        tuple(
            expert.AdvantageTerm(name, ((feature, coefficient),), ("unit-only",))
            for name, feature, coefficient in (
                ("batching", "padded_slots", 1e-8),
                ("recurrence", "gae_iterations", 1e-6),
                ("normalization", "constant", 1e-5),
                ("transport", "constant", 1e-4),
                ("diagnostics", "constant", 1e-4),
            )
        ),
    )
    groups = tuple(
        expert.Group(
            f"q{i}",
            (
                expert.Request("first", 24, 8),
                expert.ExternalCall(
                    "tool", "tools", expert.Estimate(0.01, "assumed"), ("first",)
                ),
                expert.Request("second", 48, 16, ("tool",)),
            ),
            WorkloadStats(2, 96, loss_tokens=24, sequence_lengths=(32, 64)),
            1,
        )
        for i in range(8)
    )
    scenario = _expert_scenario(
        cfg,
        groups,
        groups_per_batch=8,
        consumer_batch_size=8,
        max_concurrent_groups=8,
        runner_concurrency=8,
        pending_limit=8,
        runner_queue_limit=8,
    )
    models = expert.ExpertModels(
        replace(bundle.actor, advantage_model=advantage),
        bundle.rollout,
        _expert_transitions(cfg),
    )
    simulator = expert.ExpertSimulator(models)
    result = simulator.simulate(scenario, (expert.PlanStep(cfg),))
    assert result == simulator.simulate(scenario, (expert.PlanStep(cfg),))
    assert result.status == "complete" and result.completed_updates == 1
    assert result.accepted_total == 8 and not result.remaining_devices
    assert result.actor_gpu_seconds == pytest.approx(dp * result.seconds)
    assert result.rollout_gpu_seconds == pytest.approx(replicas * result.seconds)
    assert result.evidence == "assumed"


@pytest.mark.parametrize("workload_kind", ["handcrafted", "poisson"])
def test_basic_joint_optimizer_uses_frozen_expert_request_simulator(
    tmp_path, workload_kind
):
    """Real Expert physics/scheduling and Basic search; no GPU or fitted new labels."""
    from dataclasses import asdict

    from areal.autopilot.domains.elastic.expert_planning import (
        ExpertJointModel,
        ExpertJointWindow,
        expert_groups_from_workload,
    )
    from areal.elastic.contracts import TrainingConfig
    from areal.elastic.policy.basic import JointConfig, JointOptimizer, RolloutConfig
    from areal.elastic.policy.reference.core import fingerprint
    from areal.utils.atomic import atomic_json

    catalog = expert.Catalog()
    shape = catalog.model("qwen3-30b-a3b")
    hardware = catalog.hardware("h20", "h20-8gpu-r10")
    actor_values = catalog.profile("actor", "fsdp-dp4-offload")["values"]
    system = catalog.profile("systems", "h20-8gpu-r10")["values"]
    rollout_values = catalog.profile("rollout", "sglang-dp2-tp2")["values"]
    advantage = expert.AdvantageModel(
        "synthetic-advantage-prior-v1",
        "cpu",
        "single",
        "token",
        tuple(
            expert.AdvantageTerm(
                name, ((feature, coefficient),), ("explicit-unmeasured-prior",)
            )
            for name, feature, coefficient in (
                ("batching", "padded_slots", 1e-8),
                ("recurrence", "gae_iterations", 1e-6),
                ("normalization", "constant", 1e-5),
                ("transport", "constant", 1e-4),
                ("diagnostics", "constant", 1e-4),
            )
        ),
    )
    actor = expert.ExpertActorModel(
        shape,
        hardware,
        actor_values["runtime_revision"],
        efficiency=expert.Efficiency(**actor_values["efficiency"]),
        dispatch=expert.DispatchConfig(**actor_values["dispatch"]),
        fsdp_host_rates=expert.FSDPHostRates(**system["host_rates"]),
        advantage_model=advantage,
    )
    rollout = expert.ExpertRolloutModel(
        shape,
        hardware,
        expert.InferenceRates(**rollout_values["rates"]),
        rollout_values["runtime_revision"],
    )
    executions = tuple(
        expert.ExecutionConfig(
            replace(
                catalog.actor("fsdp-dp4-offload"), parallel=expert.ParallelLayout(dp=n)
            ),
            replace(catalog.rollout("sglang-dp2-tp2"), replicas=8 - n, tp=1),
            tuple(f"gpu{i}" for i in range(n)),
            tuple(f"gpu{i}" for i in range(n, 8)),
        )
        for n in (4, 2)
    )
    profiles = [p for c in executions for p in _expert_transitions(c).profiles]
    for source, target in (
        (executions[0], executions[1]),
        (executions[1], executions[0]),
    ):
        profiles.append(
            expert.TransitionProfile(
                "reconfigure",
                source.key,
                target.key,
                (
                    expert.TransferTask(
                        "rebuild",
                        measured=expert.Estimate(
                            2,
                            "assumed",
                            source_ids=("synthetic-rebuild-prior",),
                        ),
                    ),
                ),
                tuple(
                    expert.AllocationChange("rebuild", "end", action, role, devices)
                    for action, config in (("release", source), ("acquire", target))
                    for role, devices in (
                        ("actor", config.actor_devices),
                        ("rollout", config.rollout_devices),
                    )
                ),
                blocks=("actor", "generation", "admission"),
            )
        )
    simulator = expert.ExpertSimulator(
        expert.ExpertModels(
            actor,
            rollout,
            expert.ExpertTransitionModel(tuple(profiles)),
        )
    )
    bindings = tuple(
        (
            JointConfig(
                TrainingConfig(c.actor.parallel.world_size, actor.profile_key(c.actor)),
                RolloutConfig(
                    fingerprint(shape),
                    fingerprint(hardware),
                    rollout.runtime_revision,
                    "synthetic-fixed-sampling",
                    c.rollout.replicas,
                    c.rollout.tp,
                    c.rollout.max_batch_sequences,
                    c.rollout.max_batch_tokens,
                    c.rollout.kv_capacity_tokens,
                    3,
                ),
            ),
            c,
        )
        for c in executions
    )
    groups = []
    for i in range(12):
        prompt, output = 128 + 32 * (i % 4), 16 + 8 * (i % 3)
        lengths = (prompt + output, prompt + 2 * output + 16)
        groups.append(
            expert.Group(
                f"episode-{i}",
                (
                    expert.Request("first", prompt, output),
                    expert.ExternalCall(
                        "tool",
                        "tools",
                        expert.Estimate(0.25 + 0.1 * (i % 5), "assumed"),
                        ("first",),
                    ),
                    expert.Request("second", prompt + output + 16, output, ("tool",)),
                ),
                WorkloadStats(
                    2, sum(lengths), loss_tokens=2 * output, sequence_lengths=lengths
                ),
                1,
                available_at=(i // 4) * 0.03 + (i % 4) * 0.005,
                task_id=i,
            )
        )
    manifest, selection = None, None
    if workload_kind == "poisson":
        from areal.elastic.workload import (
            ArrivalPhase,
            CompositionPhase,
            LogNormal,
            generate,
            write_workload,
        )

        spec = _synthetic_workload_spec()
        # Small normal-path physics fixture; approved 30-minute traces stay intact.
        spec = replace(
            spec,
            duration_s=30,
            arrivals=(ArrivalPhase(30, 0.4),),
            composition=(CompositionPhase(30, spec.composition[0].weights),),
            families=tuple(
                replace(
                    family,
                    prompt=LogNormal(128, 0.2, 64, 256),
                    output=LogNormal(16, 0.2, 8, 32),
                    tool_output=LogNormal(32, 0.2, 16, 64),
                )
                for family in spec.families
            ),
        )
        episodes = generate(spec)
        manifest = write_workload(tmp_path / "workload", spec, episodes)
        count = len(episodes) // 4 * 4
        assert count >= 4
        # Selection is an explicit experiment boundary; never edit/pad the trace.
        selection = {
            "policy": "complete-batch-prefix-only",
            "selected_ids": [e["episode_id"] for e in episodes[:count]],
            "tail_ids": [e["episode_id"] for e in episodes[count:]],
            "full_stream_evaluated": count == len(episodes),
        }
        groups = expert_groups_from_workload(
            episodes[:count],
            tool_seconds_per_unit={
                "synthetic-python-work-v1": 1.0,
                "synthetic-terminal-work-v1": 1.0,
            },
        )
    updates = len(groups) // 4
    scenario = _expert_scenario(
        executions[0],
        groups,
        initial=expert.InitialState(
            executions[0],
            provider_version=1,
            training_version=1,
            serving_version=1,
            accepted_total=4,
        ),
        groups_per_batch=4,
        consumer_batch_size=4,
        max_concurrent_groups=8,
        runner_concurrency=8,
        pending_limit=16,
        runner_queue_limit=16,
        staleness=2,
        selection="task_id",
        workflow="v2_inline",
        external_pools=tuple(
            (pool, 3)
            for pool in sorted(
                {
                    n.pool
                    for g in groups
                    for n in g.nodes
                    if isinstance(n, expert.ExternalCall)
                }
            )
        ),
    )
    window = ExpertJointWindow(
        scenario,
        bindings[0][0],
        (expert.PlanStep(executions[0], checkpoint=True, inflight="drain"),) * updates,
        tuple(f"gpu{i}" for i in range(8)),
        "handcrafted-request-dag-v1" if manifest is None else manifest["trace_sha256"],
    )
    model = ExpertJointModel(simulator, bindings)
    frozen = simulator.models.snapshot_id
    search = JointOptimizer(tuple(c for c, _ in bindings), switch_margin=0).optimize(
        window, model
    )
    assert search.config is not None and search.exhaustive
    assert all(c.eligible for c in search.candidates)
    assert search.prediction.total_gpu_seconds == min(
        c.prediction.total_gpu_seconds for c in search.candidates
    )
    assert simulator.models.snapshot_id == frozen
    results = []
    for candidate in search.candidates:
        result = model.results[model.query_id(window, candidate.config)]
        assert result.completed_updates == updates and result.accepted_total == 4 + len(
            groups
        )
        assert candidate.prediction.samples == len(groups)
        assert candidate.prediction.sequence_tokens == sum(
            g.workload.sequence_tokens for g in groups
        )
        assert not result.remaining_devices
        kinds = {e.kind for e in result.events}
        assert {
            "request_complete",
            "external_start",
            "external_complete",
            "service_start",
            "actor_start",
            "weights_published",
        } <= kinds
        assert result.evidence == "assumed"
        results.append({"config": asdict(candidate.config), "expert": asdict(result)})
    atomic_json(
        tmp_path / "expert-joint-trace.json",
        {
            "model_snapshot": model.snapshot_id,
            "models": asdict(simulator.models),
            "numerical_environment": model.numerical_environment,
            "profiles": [
                {
                    "kind": kind,
                    "id": name,
                    "snapshot_sha256": fingerprint(catalog.profile(kind, name)),
                    "profile": catalog.profile(kind, name),
                }
                for kind, name in (
                    ("models", "qwen3-30b-a3b"),
                    ("hardware", "h20"),
                    ("systems", "h20-8gpu-r10"),
                    ("actor", "fsdp-dp4-offload"),
                    ("rollout", "sglang-dp2-tp2"),
                )
            ],
            "window": asdict(window),
            "bindings": [(asdict(c), asdict(e)) for c, e in bindings],
            "search": asdict(search),
            "expert_results": results,
            "scope": "Offline Qwen3-30B-A3B/H20 analytical model, synthetic requests and unmeasured timing priors; no GPU execution or calibration refit",
            "workload_manifest": manifest,
            "selection": selection,
            "conversion_assumptions": {
                "trajectories": "individual inference calls; all generated tokens are loss tokens",
                "samples_per_episode": 1,
                "accepted": "all; no reward prediction",
                "cache": "disabled; no observed prefix reuse",
                "tool_seconds_per_reference_unit": {
                    "synthetic-python-work-v1": 1.0,
                    "synthetic-terminal-work-v1": 1.0,
                }
                if manifest
                else None,
                "external_pool_capacities": list(scenario.external_pools),
            },
        },
    )


def _synthetic_workload_spec(profile="single-peak"):
    import json
    from pathlib import Path

    from areal.elastic.workload import ArrivalPhase, WorkloadSpec

    path = Path(__file__).parents[1] / "examples/autopilot/elastic/workload-spec.json"
    spec = WorkloadSpec.decode(json.loads(path.read_text()))
    if profile != "single-peak":
        phases = json.loads(
            (path.parent / "workload-arrivals" / f"{profile}.json").read_text()
        )
        spec = replace(spec, arrivals=tuple(ArrivalPhase(**p) for p in phases))
    return spec


@pytest.mark.parametrize("profile", ["single-peak", "periodic", "short-bursts"])
def test_synthetic_workload_preserves_seed_dag_and_roundtrip(tmp_path, profile):
    """Frozen business inputs stay causal and contain no execution placement."""
    from areal.elastic.workload import (
        WorkloadSpec,
        arrival_times,
        generate,
        read_workload,
        write_workload,
    )

    spec = _synthetic_workload_spec(profile)
    episodes = generate(spec)
    assert episodes == generate(WorkloadSpec.decode(spec.encode()))
    first = write_workload(tmp_path / "first", spec, episodes)
    second = write_workload(tmp_path / "second", spec, generate(spec))
    assert first["trace_sha256"] == second["trace_sha256"]
    assert (tmp_path / "first/workload.jsonl").read_bytes() == (
        tmp_path / "second/workload.jsonl"
    ).read_bytes()
    manifest, loaded = read_workload(tmp_path / "first")
    assert manifest == first and loaded == episodes
    assert len(episodes) == len(arrival_times(spec)) == manifest["episodes"]
    assert manifest["tool_calls"] == manifest["inference_requests"] - len(episodes)
    families = {f.name: f for f in spec.families}
    for index, episode in enumerate(episodes):
        assert set(episode) == {
            "episode_id",
            "task_id",
            "available_at",
            "family",
            "nodes",
        }
        assert episode["task_id"] == index
        assert 0 <= episode["available_at"] < spec.duration_s
        turns = sum(n["kind"] == "inference" for n in episode["nodes"])
        assert [n["kind"] for n in episode["nodes"]] == ["inference", "tool"] * (
            turns - 1
        ) + ["inference"]
        known, previous, prompt = set(), [], None
        family = families[episode["family"]]
        for node in episode["nodes"]:
            assert node["predecessors"] == previous
            assert node["id"] not in known
            if node["kind"] == "inference":
                if prompt is not None:
                    assert node["input_tokens"] == prompt
                assert (
                    family.output.minimum
                    <= node["output_tokens"]
                    <= family.output.maximum
                )
                prompt = (
                    node["input_tokens"]
                    + node["output_tokens"]
                    + family.message_overhead_tokens
                )
            else:
                assert node["kind"] == "tool" and node["work_units"] > 0
                assert (
                    family.tool_output.minimum
                    <= node["output_tokens"]
                    <= family.tool_output.maximum
                )
                prompt += node["output_tokens"]
            known.add(node["id"])
            previous = [node["id"]]


def test_synthetic_workload_arrival_and_composition_axes_are_independent():
    """Composition cannot perturb the arrival stream or scheduling-independent draws."""
    from areal.elastic.workload import ArrivalPhase, CompositionPhase, generate

    spec = _synthetic_workload_spec()
    constant_mix = replace(
        spec, composition=(CompositionPhase(spec.duration_s, (("short", 1.0),)),)
    )
    varied = generate(spec)
    fixed = generate(constant_mix)
    assert [e["available_at"] for e in varied] == [e["available_at"] for e in fixed]
    faster = generate(
        replace(
            constant_mix,
            arrivals=tuple(
                ArrivalPhase(p.until_s, p.rate_per_s * 2) for p in spec.arrivals
            ),
        )
    )
    for first, second in zip(fixed, faster):
        assert first["episode_id"] == second["episode_id"]
        assert first["nodes"] == second["nodes"]
    assert len(faster) > len(fixed)
    start, intensity = 0, 0
    for phase in constant_mix.arrivals:
        intensity += (phase.until_s - start) * phase.rate_per_s
        start = phase.until_s
    uniform = generate(
        replace(
            constant_mix,
            arrivals=(ArrivalPhase(spec.duration_s, intensity / spec.duration_s),),
        )
    )
    assert len(uniform) == len(fixed)
    assert [e["nodes"] for e in uniform] == [e["nodes"] for e in fixed]
    # The approved profiles use the same finite-window intensity and task content.
    baseline = generate(spec)
    for profile in ("periodic", "short-bursts"):
        other = generate(_synthetic_workload_spec(profile))
        assert len(other) == len(baseline)
        assert [e["available_at"] for e in other] != [
            e["available_at"] for e in baseline
        ]
        assert [{k: v for k, v in e.items() if k != "available_at"} for e in other] == [
            {k: v for k, v in e.items() if k != "available_at"} for e in baseline
        ]


def test_synthetic_workload_poisson_counts_match_mean_and_variance():
    """Independent fixed-duration windows exercise count dispersion, not exact totals."""
    from statistics import mean, pvariance

    from areal.elastic.workload import ArrivalPhase, CompositionPhase, arrival_times

    spec = replace(
        _synthetic_workload_spec(),
        duration_s=10,
        arrivals=(ArrivalPhase(10, 4),),
        composition=(CompositionPhase(10, (("short", 1.0),)),),
    )
    counts = [len(arrival_times(replace(spec, seed=seed))) for seed in range(1024)]
    # E[N]=Var[N]=lambda*T=40. Broad, deterministic Monte Carlo checks.
    assert abs(mean(counts) - 40) < 1
    assert abs(pvariance(counts) - 40) < 8


def test_expert_overlap_counts_residency_once_and_cleans_terminal_resources():
    """Rollout prefetch overlaps actor compute; cost follows ownership, not phase sum."""
    config = _expert_config()
    scenario = _expert_scenario(config, [_expert_group(str(i)) for i in range(3)])
    result = _simulate(scenario, [expert.PlanStep(config), expert.PlanStep(config)])
    assert result.status == "complete"
    assert result.seconds == pytest.approx(12)
    assert result.actor_gpu_seconds == result.rollout_gpu_seconds == result.seconds
    assert result.completed_updates == result.attempted_updates == 2
    assert result.accepted_total == 3 and result.queued_groups == 1
    assert not result.remaining_devices and not result.running_groups
    assert result.evidence == "assumed"


def test_expert_group_affinity_queues_at_original_replica_with_spare_peer():
    """A group's samples share a replica even when the other replica is idle."""
    config = _expert_config(replicas=2)
    config = replace(config, rollout=replace(config.rollout, max_batch_sequences=1))
    group = expert.Group(
        "g",
        (expert.Request("a", 8, 1), expert.Request("b", 8, 1)),
        WorkloadStats(2, 16, sequence_lengths=(8, 8)),
        2,
    )
    times = {}
    for routing in ("request_load", "group_affinity"):
        result = _simulate(
            _expert_scenario(
                config, [group], samples_per_group=2, request_routing=routing
            ),
            [expert.PlanStep(config)],
        )
        assert result.status == "complete"
        replicas = {
            dict(event.details)["replica"]
            for event in result.events
            if event.kind == "request_assigned"
        }
        assert replicas == ({0, 1} if routing == "request_load" else {0})
        times[routing] = next(
            event.time for event in result.events if event.kind == "group_result_ready"
        )
    assert times == {"request_load": 1, "group_affinity": 2}


def test_expert_group_affinity_balances_live_groups_through_external_waits():
    """External waits keep affinity; newly freed replicas receive new groups."""
    config = _expert_config(replicas=2)
    groups = [
        _expert_group("long", external=10),
        _expert_group("short", external=1),
        _expert_group("next", external=1),
    ]
    result = _simulate(
        _expert_scenario(
            config, groups, request_routing="group_affinity", staleness=10
        ),
        [expert.PlanStep(config), expert.PlanStep(config)],
    )
    mapping = {
        event.subject: dict(event.details)["replica"]
        for event in result.events
        if event.kind == "group_started"
    }
    assert mapping == {"long": 0, "short": 1, "next": 1}
    for event in result.events:
        if event.kind == "request_assigned":
            assert (
                dict(event.details)["replica"] == mapping[event.subject.split("/")[0]]
            )


@pytest.mark.parametrize("replica", [None, 0])
def test_expert_group_affinity_restores_known_replica_for_following_turns(replica):
    """A live request or explicit group mapping preserves subsequent affinity."""
    config = _expert_config(replicas=2)
    group = replace(
        _expert_group("g"),
        nodes=(
            expert.Request("first", 8, 1),
            expert.ExternalCall(
                "tool", "tools", expert.Estimate(2, "assumed"), ("first",)
            ),
            expert.Request("next", 8, 1, ("tool",)),
        ),
    )
    initial = expert.InitialState(
        config,
        running=(
            expert.RunningGroup(
                "g",
                0,
                requests=(expert.RequestProgress("first", 0, 0, 0),),
                replica=replica,
            ),
        ),
    )
    result = _simulate(
        _expert_scenario(
            config, [group], initial=initial, request_routing="group_affinity"
        ),
        [expert.PlanStep(config)],
    )
    assert result.status == "complete"
    assigned = [e for e in result.events if e.kind == "request_assigned"]
    assert len(assigned) == 1 and dict(assigned[0].details)["replica"] == 0


@pytest.mark.parametrize("replica", [None, 2])
def test_expert_group_affinity_rejects_unknown_or_outside_restored_replica(replica):
    """A tool-only restored group cannot silently invent its original replica."""
    config = _expert_config(replicas=2)
    group = _expert_group("g", external=2)
    initial = expert.InitialState(
        config,
        running=(
            expert.RunningGroup(
                "g", 0, external_remaining=(("tool", 2),), replica=replica
            ),
        ),
    )
    with pytest.raises(ValueError, match="replica"):
        _simulate(
            _expert_scenario(
                config, [group], initial=initial, request_routing="group_affinity"
            ),
            [expert.PlanStep(config)],
        )


def test_expert_parallel_sample_cleanup_is_critical_path_and_releases_gpu_slot():
    """Parallel cleanups add their maximum, with group export charged only once."""
    config = _expert_config()
    config = replace(config, rollout=replace(config.rollout, max_batch_sequences=1))
    group = expert.Group(
        "cleanup",
        (
            expert.Request("a", 8, 1),
            expert.ExternalCall(
                "clean-a", "tools", expert.Estimate(30, "assumed"), ("a",)
            ),
            expert.Request("b", 8, 1),
            expert.ExternalCall(
                "clean-b", "tools", expert.Estimate(30, "assumed"), ("b",)
            ),
        ),
        WorkloadStats(2, 16, sequence_lengths=(8, 8)),
        2,
        postprocess=expert.Estimate(0.5, "assumed"),
    )
    peer = replace(group, group_id="peer", nodes=(expert.Request("x", 8, 1),))
    result = _simulate(
        _expert_scenario(config, [group, peer], samples_per_group=2, staleness=10),
        [expert.PlanStep(config), expert.PlanStep(config)],
    )
    ready = {
        event.subject: event.time
        for event in result.events
        if event.kind == "group_result_ready"
    }
    assert result.status == "complete"
    assert ready["cleanup"] == 32.5
    assert ready["peer"] < 31


def test_expert_group_affinity_rejects_restored_requests_on_different_replicas():
    """Resuming cannot silently merge a group that already spans replicas."""
    config = _expert_config(replicas=2)
    group = replace(
        _expert_group("g"),
        nodes=(expert.Request("a", 8, 1), expert.Request("b", 8, 1)),
    )
    initial = expert.InitialState(
        config,
        running=(
            expert.RunningGroup(
                "g",
                0,
                requests=(
                    expert.RequestProgress("a", 0, 0, 0),
                    expert.RequestProgress("b", 1, 0, 0),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="one known replica"):
        _simulate(
            _expert_scenario(
                config, [group], initial=initial, request_routing="group_affinity"
            ),
            [expert.PlanStep(config)],
        )


def test_expert_unknown_request_routing_is_rejected():
    """Routing must remain an explicit supported scenario assumption."""
    config = _expert_config()
    with pytest.raises(ValueError, match="routing"):
        _expert_scenario(config, [_expert_group("g")], request_routing="actual_router")


def test_expert_v2_inline_producer_batches_then_polls_and_stops_during_train():
    config = _expert_config()
    groups = [_expert_group(str(i), external=10) for i in range(6)]
    scenario = _expert_scenario(config, groups, workflow="v2_inline", terminal="cancel")
    result = _simulate(scenario, [expert.PlanStep(config)])
    assert result.status == "complete"
    enqueues = [e.time for e in result.events if e.kind == "group_enqueued"]
    assert enqueues[:3] == [0, 1, 2]
    start = next(e.time for e in result.events if e.kind == "actor_start")
    end = next(e.time for e in result.events if e.kind == "actor_complete")
    assert not any(start < time < end for time in enqueues)
    assert any(e.kind == "admission_resume" for e in result.events)


def test_expert_v2_inline_pause_defers_delivery_and_final_resume_starts_prefetch():
    config = _expert_config()
    groups = [
        _expert_group("ready"),
        _expert_group("late", external=3.5),
        _expert_group("tail", external=20),
    ]
    initial = expert.InitialState(
        config, accepted_total=1, ready_groups=(("ready", 0, 0),)
    )
    scenario = _expert_scenario(
        config,
        groups,
        initial=initial,
        workflow="v2_inline",
        terminal="cancel",
        staleness=0,
        max_concurrent_groups=1,
        runner_concurrency=1,
    )
    result = _simulate(scenario, [expert.PlanStep(config), expert.PlanStep(config)])
    assert result.status == "complete" and result.completed_updates == 2
    resume = [e.time for e in result.events if e.kind == "admission_resume"]
    tail = next(
        e for e in result.events if e.kind == "group_started" and e.subject == "tail"
    )
    assert tail.time >= resume[-1]
    assert result.cancelled_groups == 1 and not result.remaining_devices


def test_expert_v2_inline_completed_coroutine_is_collected_after_resume():
    config = _expert_config()
    late = replace(
        _expert_group("late"),
        nodes=(expert.ExternalCall("tool", "tools", expert.Estimate(4, "assumed")),),
    )
    groups = [_expert_group("first"), late]
    initial = expert.InitialState(
        config,
        running=(expert.RunningGroup("late", 0, external_remaining=(("tool", 4),)),),
    )
    result = _simulate(
        _expert_scenario(config, groups, initial=initial, workflow="v2_inline"),
        [expert.PlanStep(config), expert.PlanStep(config)],
    )
    accepted = next(
        e.time
        for e in result.events
        if e.kind == "group_accepted" and e.subject == "late"
    )
    ready = next(
        e.time
        for e in result.events
        if e.kind == "group_result_ready" and e.subject == "late"
    )
    assert accepted < ready
    assert ready == next(e.time for e in result.events if e.kind == "admission_resume")


def test_expert_v2_inline_finite_input_shortfall_does_not_poll_until_event_budget():
    config = _expert_config()
    result = _simulate(
        _expert_scenario(
            config,
            [_expert_group("only")],
            workflow="v2_inline",
            groups_per_batch=2,
            consumer_batch_size=2,
            max_events=100,
        ),
        [expert.PlanStep(config)],
    )
    assert result.status == "incomplete" and result.attempted_updates == 0
    assert result.seconds == 1
    assert "Event budget" not in result.reason


def test_expert_staleness_capacity_waits_for_publication_and_keeps_cumulative_accepts():
    """At staleness zero, consuming a group cannot grant capacity before publication."""
    config = _expert_config()
    scenario = _expert_scenario(
        config, [_expert_group("g0"), _expert_group("g1")], staleness=0
    )
    result = _simulate(scenario, [expert.PlanStep(config), expert.PlanStep(config)])
    assert result.status == "complete"
    submitted = [event for event in result.events if event.kind == "group_submitted"]
    published = [
        event.time for event in result.events if event.kind == "weights_published"
    ]
    actor_end = [
        event.time for event in result.events if event.kind == "actor_complete"
    ]
    assert actor_end[0] == 3 and published[0] == 6
    assert submitted[1].time >= published[0]
    assert dict(submitted[1].details)["accepted_total"] == 1
    assert result.accepted_total == 2 and result.queued_groups == 0


def test_expert_rejected_groups_spend_time_and_release_capacity_without_training_credit():
    """Rejected rollout work remains billed and does not become accepted data."""
    config = _expert_config()
    groups = [_expert_group("bad", accepted=False), _expert_group("good")]
    scenario = _expert_scenario(config, groups, staleness=0)
    result = _simulate(scenario, [expert.PlanStep(config)])
    assert result.status == "complete" and result.seconds == 8
    assert (
        result.rejected_total == result.accepted_total == result.completed_updates == 1
    )
    starts = [e.time for e in result.events if e.kind == "actor_start"]
    assert starts == [2]


def test_expert_recovery_uses_explicit_cumulative_offset():
    """A restored provider version and counters admit the next logical batch only."""
    config = _expert_config()
    initial = expert.InitialState(
        config,
        provider_version=4,
        training_version=4,
        serving_version=4,
        accepted_total=8,
    )
    scenario = _expert_scenario(
        config,
        [_expert_group("g0"), _expert_group("g1")],
        initial=initial,
        groups_per_batch=2,
        consumer_batch_size=2,
        staleness=0,
    )
    result = _simulate(scenario, [expert.PlanStep(config)])
    assert result.status == "complete" and result.accepted_total == 10
    versions = [
        dict(e.details)["version"]
        for e in result.events
        if e.kind == "provider_version"
    ]
    assert versions == [5]


def test_expert_ready_selection_uses_start_time_not_completion_order():
    """Queue membership follows task start ordering at the batch selection boundary."""
    config = _expert_config()
    initial = expert.InitialState(
        config, time=10, accepted_total=2, ready_groups=(("slow", 0, 9), ("fast", 1, 2))
    )
    scenario = _expert_scenario(
        config, [_expert_group("slow"), _expert_group("fast")], initial=initial
    )
    result = _simulate(scenario, [expert.PlanStep(config)])
    chosen = next(
        dict(e.details)["groups"] for e in result.events if e.kind == "actor_start"
    )
    assert chosen == ("slow",)
    assert result.queue_wait_seconds == 1


def test_expert_suspend_restore_pays_transitions_and_preserves_rollout_counters():
    """Actor suspension does not reset the live rollout controller's accepted total."""
    config = _expert_config()
    scenario = _expert_scenario(
        config, [_expert_group("g0"), _expert_group("g1")], staleness=0
    )
    result = _simulate(
        scenario,
        [
            expert.PlanStep(config, checkpoint=True, release_after=True),
            expert.PlanStep(config),
        ],
    )
    assert result.status == "complete" and result.accepted_total == 2
    actions = [e.subject for e in result.events if e.kind == "transition_start"]
    assert actions == [
        "publish",
        "checkpoint",
        "release",
        "restore",
        "publish",
        "shutdown",
    ]
    assert result.actor_gpu_seconds <= result.rollout_gpu_seconds
    assert result.completed_updates == 2


def test_expert_cancel_and_drain_keep_distinct_cost_and_terminal_counts():
    """Cancellation pays its cleanup duration; draining continues external work."""
    config = _expert_config()
    groups = [_expert_group("train"), _expert_group("slow", external=20)]
    scenario = _expert_scenario(config, groups)
    drained = _simulate(scenario, [expert.PlanStep(config)])
    cancelled = _simulate(
        replace(scenario, terminal="cancel"), [expert.PlanStep(config)]
    )
    assert drained.status == cancelled.status == "complete"
    assert drained.seconds == 22 and cancelled.seconds == 7.5
    assert drained.cancelled_groups == 0 and cancelled.cancelled_groups == 1
    assert cancelled.rejected_total == 0 and cancelled.accepted_total == 1
    assert not cancelled.remaining_devices


def test_expert_missing_transition_has_unknown_cost_and_keeps_live_resources():
    """No missing publication or restore is silently treated as free."""
    config = _expert_config()
    result = _simulate(
        _expert_scenario(config, [_expert_group("g")]),
        [expert.PlanStep(config)],
        expert.ExpertTransitionModel(),
    )
    assert result.status == "unsupported" and "publish" in result.reason
    assert result.seconds == 3 and result.remaining_devices == ("a0", "r0")
    assert result.evidence == "unsupported"


def test_expert_unknown_initial_state_and_insufficient_capacity_do_not_claim_completion():
    """Missing initial state and a mismatched consumer batch are different failures."""
    config = _expert_config()
    scenario = _expert_scenario(config, [_expert_group("g0"), _expert_group("g1")])
    unknown = _simulate(
        replace(scenario, initial=replace(scenario.initial, complete=False)),
        [expert.PlanStep(config)],
    )
    stalled = _simulate(
        replace(scenario, groups_per_batch=2, consumer_batch_size=1, staleness=0),
        [expert.PlanStep(config)],
    )
    assert unknown.status == "unsupported" and unknown.attempted_updates == 0
    assert stalled.status == "incomplete" and stalled.queued_groups == 1


def test_expert_failed_updates_and_failed_publication_preserve_cost_without_false_success():
    """Numerical update success and weight publication are independent scenario facts."""
    config = _expert_config()
    scenario = _expert_scenario(config, [_expert_group("g")])
    skipped = _simulate(scenario, [expert.PlanStep(config, update_succeeds=False)])
    failed = _simulate(scenario, [expert.PlanStep(config, publish_succeeds=False)])
    assert (
        skipped.status == "complete"
        and skipped.attempted_updates == 1
        and skipped.completed_updates == 0
    )
    assert failed.status == "unsupported" and failed.seconds == 6
    assert not any(e.kind == "provider_version" for e in failed.events)


def test_expert_replay_preserves_unknown_membership_and_observation_window_costs():
    """Aggregate group timestamps suffice for accounting, never episode reconstruction."""
    initial = expert.ReplayInitial(0, ("a",), ("r",))
    records = (
        expert.ReplayRecord(0, "group_submit", count=2),
        expert.ReplayRecord(2, "group_accept", count=2),
        expert.ReplayRecord(3, "actor_start", count=2),
        expert.ReplayRecord(5, "actor_complete", version=1),
        expert.ReplayRecord(6, "publish", version=1),
        expert.ReplayRecord(6, "provider_version", version=1),
        expert.ReplayRecord(7, "release", devices=("a",)),
    )
    result = expert.replay(
        initial, records, stop_at=10, source_id="unit-observed-window"
    )
    assert result.actor_gpu_seconds == 7 and result.rollout_gpu_seconds == 10
    assert result.accepted_total == 2 and result.queued_groups == 0
    assert result.queue_wait_seconds is None and result.remaining_devices == ("r",)
    assert any("Aggregate counts" in note for note in result.notes)


def test_expert_replay_checks_membership_and_rejects_inconsistent_events():
    """Task-attributed records must conserve submission, acceptance and consumption."""
    initial = expert.ReplayInitial(0, ("a",), ("r",))
    rows = (
        expert.ReplayRecord(0, "group_submit", group_ids=("g",)),
        expert.ReplayRecord(2, "group_accept", group_ids=("g",)),
        expert.ReplayRecord(4, "actor_start", group_ids=("g",)),
        expert.ReplayRecord(6, "actor_complete", version=1),
    )
    result = expert.replay(initial, rows, stop_at=6, source_id="unit-attributed-window")
    assert result.queue_wait_seconds == 2
    with pytest.raises(ValueError, match="completion without"):
        expert.replay(initial, (rows[1],), stop_at=3, source_id="invalid")
    with pytest.raises(ValueError, match="ahead"):
        expert.replay(
            initial,
            (expert.ReplayRecord(1, "provider_version", version=1),),
            stop_at=2,
            source_id="invalid",
        )


def test_expert_pause_stops_submitted_backlog_but_terminal_drain_finishes_it():
    cfg = _expert_config()
    groups = [
        _expert_group("g0"),
        replace(
            _expert_group("g1"),
            nodes=(
                expert.ExternalCall("tool", "tools", expert.Estimate(3, "assumed")),
            ),
        ),
        _expert_group("g2"),
    ]
    scenario = _expert_scenario(
        cfg, groups, staleness=2, max_concurrent_groups=3, runner_concurrency=1
    )
    result = _simulate(scenario, [expert.PlanStep(cfg)])
    assert result.status == "complete" and not result.running_groups
    publish = next(e.time for e in result.events if e.kind == "weights_published")
    assert (
        next(
            e.time
            for e in result.events
            if e.kind == "group_started" and e.subject == "g2"
        )
        >= publish
    )


def test_expert_dynamic_batch_counts_ordered_attempts_and_deterministic_task_ids():
    cfg = _expert_config()
    groups = [
        replace(_expert_group(name, accepted=name != "bad"), task_id=i)
        for i, name in ((1, "bad"), (2, "2"), (10, "10"))
    ]
    scenario = _expert_scenario(
        cfg,
        groups,
        groups_per_batch=2,
        consumer_batch_size=2,
        max_concurrent_groups=3,
        runner_concurrency=3,
        selection="task_id",
        dynamic_batch=True,
    )
    result = _simulate(scenario, [expert.PlanStep(cfg)])
    start = next(e for e in result.events if e.kind == "actor_start")
    assert dict(start.details)["groups"] == ("2",)
    assert result.queued_groups == 1 and result.rejected_total == 1
    deterministic = replace(
        scenario,
        groups=tuple(replace(g, accepted=True) for g in groups),
        groups_per_batch=3,
        dynamic_batch=False,
    )
    ordered = _simulate(deterministic, [expert.PlanStep(cfg)])
    assert dict(next(e for e in ordered.events if e.kind == "actor_start").details)[
        "groups"
    ] == ("bad", "2", "10")


def test_expert_fixed_batch_retains_partial_collection_while_replacing_rejections():
    cfg = _expert_config()
    groups = [
        _expert_group("bad", accepted=False),
        _expert_group("g1"),
        _expert_group("g2", external=3),
    ]
    scenario = _expert_scenario(
        cfg,
        groups,
        groups_per_batch=2,
        consumer_batch_size=2,
        max_concurrent_groups=3,
        runner_concurrency=3,
        selection="task_id",
    )
    result = _simulate(scenario, [expert.PlanStep(cfg)])
    start = next(e for e in result.events if e.kind == "actor_start")
    assert start.time == 4 and dict(start.details)["groups"] == ("g1", "g2")


def _expert_reconfigure(source, target, seconds=5):
    return expert.TransitionProfile(
        "reconfigure",
        source.key,
        target.key,
        (expert.TransferTask("change", measured=expert.Estimate(seconds, "assumed")),),
        (
            expert.AllocationChange(
                "change", "start", "acquire", "actor", target.actor_devices
            ),
            expert.AllocationChange(
                "change", "end", "release", "actor", source.actor_devices
            ),
        ),
        blocks=("actor",),
    )


def test_expert_actor_only_reconfigure_retains_live_rollout_and_unique_allocation_cost():
    source, target = _expert_config(), _expert_config(actor_dp=2)
    group = _expert_group("g", output=4)
    initial = expert.InitialState(
        source,
        running=(
            expert.RunningGroup(
                "g", 0, requests=(expert.RequestProgress("request", 0, 8, 1),)
            ),
        ),
    )
    scenario = _expert_scenario(source, [group], initial=initial)
    model = expert.ExpertTransitionModel(
        _expert_transitions(source).profiles
        + _expert_transitions(target).profiles
        + (_expert_reconfigure(source, target),)
    )
    result = _simulate(scenario, [expert.PlanStep(target)], model)
    assert result.status == "complete"
    assert next(e.time for e in result.events if e.kind == "request_complete") == 3
    assert next(e.time for e in result.events if e.kind == "actor_start") == 5
    assert (
        result.seconds == 11
        and result.actor_gpu_seconds == 22
        and result.rollout_gpu_seconds == 11
    )
    assert not result.remaining_devices


def test_expert_zero_duration_transition_keeps_allocation_references_balanced():
    source, target = _expert_config(), _expert_config(actor_dp=2)
    profiles = (
        _expert_transitions(source).profiles
        + _expert_transitions(target).profiles
        + (_expert_reconfigure(source, target, 0),)
    )
    scenario = _expert_scenario(source, [_expert_group("g")])
    result = _simulate(
        scenario, [expert.PlanStep(target)], expert.ExpertTransitionModel(profiles)
    )
    assert result.status == "complete" and not result.remaining_devices
    assert result.actor_gpu_seconds == 2 * result.rollout_gpu_seconds


def test_expert_resumed_request_keeps_progress_and_can_span_weight_versions():
    cfg = _expert_config()
    groups = [_expert_group("ready"), _expert_group("live", output=6)]
    initial = expert.InitialState(
        cfg,
        accepted_total=1,
        ready_groups=(("ready", 0, 0),),
        running=(
            expert.RunningGroup(
                "live", 0, requests=(expert.RequestProgress("request", 0, 8, 1),)
            ),
        ),
    )
    result = _simulate(
        _expert_scenario(cfg, groups, initial=initial), [expert.PlanStep(cfg)]
    )
    tokens = [dict(e.details) for e in result.events if e.kind == "token_generated"]
    assert [t["output_index"] for t in tokens] == [2, 3, 4, 5, 6]
    assert {t["version"] for t in tokens} == {0, 1}
    assert result.status == "complete" and result.accepted_total == 2


def test_expert_replay_rejects_releasing_resources_needed_by_active_update():
    initial = expert.ReplayInitial(
        0, ("a",), ("r",), accepted_total=1, queued_groups=1, ready_ids=("g",)
    )
    records = (
        expert.ReplayRecord(1, "actor_start", group_ids=("g",)),
        expert.ReplayRecord(2, "release", devices=("a",)),
    )
    with pytest.raises(ValueError, match="active actor resources"):
        expert.replay(initial, records, stop_at=10, source_id="invalid-release")


@pytest.mark.parametrize("inflight", ["drain", "cancel"])
def test_expert_rollout_layout_change_preserves_declared_inflight_policy(inflight):
    source, target = _expert_config(), _expert_config(replicas=2)
    groups = [_expert_group("live", output=4), _expert_group("new")]
    initial = expert.InitialState(
        source,
        running=(
            expert.RunningGroup(
                "live", 0, requests=(expert.RequestProgress("request", 0, 8, 1),)
            ),
        ),
    )
    change = expert.TransitionProfile(
        "reconfigure",
        source.key,
        target.key,
        (expert.TransferTask("change", measured=expert.Estimate(5, "assumed")),),
        (
            expert.AllocationChange(
                "change", "start", "acquire", "rollout", target.rollout_devices
            ),
            expert.AllocationChange(
                "change", "end", "release", "rollout", source.rollout_devices
            ),
        ),
        blocks=("actor", "generation", "admission"),
    )
    transitions = expert.ExpertTransitionModel(
        _expert_transitions(source).profiles
        + _expert_transitions(target).profiles
        + (change,)
    )
    scenario = _expert_scenario(source, groups, initial=initial)
    result = _simulate(
        scenario, [expert.PlanStep(target, inflight=inflight)], transitions
    )
    assert result.status == "complete" and not result.remaining_devices
    assert result.cancelled_groups == (1 if inflight == "cancel" else 0)
    assert any(
        e.kind == "request_complete" and e.subject == "live/request"
        for e in result.events
    ) == (inflight == "drain")
    unsupported = _simulate(
        scenario, [expert.PlanStep(target, inflight="retain")], transitions
    )
    assert unsupported.status == "unsupported" and unsupported.seconds == 0


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("fsdp", [False, True])
def test_expert_analytical_models_integrate_with_event_simulation_without_training_stack(
    aggregate,
    fsdp,
):
    from tests.test_autopilot_actor_cost import unit_advantage_model

    cfg = _expert_config()
    if fsdp:
        cfg = replace(
            cfg,
            actor=expert.ActorConfig(
                "fsdp",
                expert.ParallelLayout(),
                max_tokens_per_mb=128,
                optimizer_type="adam_bf16",
                optimizer_dtype="bfloat16",
                grad_bytes=2,
                offload_params=True,
            ),
        )
    shape = expert.ModelShape(128, 64, 4, 4, 2, 32, 4, 2, 1024)
    hardware = expert.Hardware("unit-only", 100, 1000, 100, 20, 8, 80)
    models = expert.ExpertModels(
        expert.ExpertActorModel(
            shape, hardware, "unit-runtime", advantage_model=unit_advantage_model()
        ),
        expert.ExpertRolloutModel(
            shape,
            hardware,
            expert.InferenceRates(0.5, 0.5, 0.5, 0.5, 1e-6),
            "unit-runtime",
        ),
        _expert_transitions(cfg),
    )
    if fsdp:
        models = replace(
            models,
            actor=replace(
                models.actor, fsdp_host_rates=expert.FSDPHostRates(10, 10, 20, 0, 0)
            ),
        )
    if aggregate:
        actor = replace(
            models.actor,
            advantage_model=expert.AggregateAdvantageModel(
                "unit-options", form="constant"
            ),
        )
        observations = tuple(
            expert.AdvantageObservation(
                f"unit-calibration-{i}",
                actor.profile_key(cfg.actor),
                WorkloadStats(8, 64, max_seq_len=8),
                0.02,
                "trainer_host_wall_with_stats",
            )
            for i in range(2)
        )
        actor = replace(
            actor, advantage_model=actor.advantage_model.fit("unit-fit", observations)
        )
        models = replace(models, actor=actor)
    scenario = _expert_scenario(cfg, [_expert_group(str(i)) for i in range(3)])
    result = expert.ExpertSimulator(models).simulate(
        scenario, (expert.PlanStep(cfg), expert.PlanStep(cfg))
    )
    assert result.status == "complete" and result.attempted_updates == 2
    assert (
        result.seconds > 7
    )  # Explicit publication/cleanup plus positive modeled work.
    assert result.actor_gpu_seconds == result.rollout_gpu_seconds == result.seconds
    assert result.evidence == "assumed" and not result.remaining_devices
    assert {e.subject for e in result.events if e.kind == "actor_phase_start"} == {
        "recompute_logp",
        "compute_advantage",
        "train_step",
    }
    if aggregate:
        starts = [
            e.time
            for e in result.events
            if e.kind == "actor_phase_start" and e.subject == "compute_advantage"
        ]
        ends = [
            e.time
            for e in result.events
            if e.kind == "actor_phase_complete" and e.subject == "compute_advantage"
        ]
        assert len(starts) == len(ends) == 2
        assert [
            end - start for start, end in zip(starts, ends, strict=True)
        ] == pytest.approx([0.02, 0.02])


def _rpc_timing(config=None, **changes):
    config = config or _expert_config()
    values = dict(
        version="unit-rpc-fit",
        profile=expert.RPCTimingModel.profile_for(_FixedRollout(), config.rollout),
        service_scale=1.0,
        response_seconds=3.0,
        sample_ids=("calibration-only",),
        request_bounds=(("input_tokens", 1, 64), ("output_tokens", 1, 64)),
    )
    values.update(changes)
    return expert.RPCTimingModel(**values)


def test_expert_rpc_timing_keeps_scope_bounds_and_component_uncertainty():
    """RPC fits remain conditional assumptions, including within length bounds."""
    model = _rpc_timing(profile="p", service_scale=2)
    original = expert.Estimate(4, "calibrated", source_ids=("gpu-iterations",))
    service = model.service_duration("p", original)
    assert service.seconds == 8 and service.evidence == "assumed"
    assert original.seconds == 4 and original.evidence == "calibrated"
    assert service.source_ids == ("gpu-iterations", "unit-rpc-fit")
    assert model.response_delay("p", 8, 2).seconds == 3
    assert model.response_delay("different-profile", 8, 2).seconds is None
    assert model.service_duration("different-profile", original).seconds is None
    assert model.response_delay("p", 65, 2).seconds is None
    outside = replace(model, allow_request_extrapolation=True).response_delay(
        "p", 65, 2
    )
    assert outside.seconds == 3 and outside.evidence == "assumed"
    assert any("Outside RPC" in n for n in outside.notes)
    infeasible = expert.Estimate(4, "assumed", "infeasible")
    assert model.service_duration("p", infeasible).feasibility == "infeasible"
    assert expert.RPCTimingModel.profile_for(
        _FixedRollout(1), _expert_config().rollout
    ) != expert.RPCTimingModel.profile_for(_FixedRollout(2), _expert_config().rollout)


@pytest.mark.parametrize(
    "changes",
    [
        {"service_scale": 0},
        {"response_seconds": -1},
        {"sample_ids": ("duplicate", "duplicate")},
        {"request_bounds": (("input_tokens", 1, 64),)},
        {"request_bounds": (("input_tokens", 10, 1), ("output_tokens", 1, 2))},
        {"allow_request_extrapolation": 1},
    ],
)
def test_expert_rpc_timing_rejects_invalid_parameters(changes):
    """No zero service scale, negative delay or implicit coverage/provenance."""
    with pytest.raises(ValueError):
        _rpc_timing(**changes)


def test_expert_rpc_tail_frees_gpu_slot_and_preserves_request_dependencies():
    """Response tails overlap new GPU work; dependent requests await delivery."""
    config = _expert_config()
    config = replace(config, rollout=replace(config.rollout, max_batch_sequences=1))
    a = replace(
        _expert_group("a"),
        nodes=(
            expert.Request("first", 8, 2),
            expert.Request("second", 8, 1, ("first",)),
        ),
    )
    scenario = _expert_scenario(config, [a, _expert_group("b")])
    models = expert.ExpertModels(
        _FixedActor(), _FixedRollout(), _expert_transitions(config), _rpc_timing(config)
    )
    result = expert.ExpertSimulator(models).simulate(
        scenario, (expert.PlanStep(config),)
    )
    assert result.status == "complete"
    events = {(e.kind, e.subject): e.time for e in result.events}
    assert events["response_wait", "a/first"] == 2
    assert events["request_assigned", "b/request"] == 2
    assert events["request_complete", "a/first"] == 5
    assert events["request_queued", "a/second"] == 5
    assert events["request_complete", "a/second"] == 9
    assert result.rollout_gpu_seconds == result.seconds


def test_expert_rpc_tail_survives_publication_without_changing_token_version():
    """Publication waits on GPU iterations, not asynchronous response delivery."""
    config = _expert_config()
    groups = [_expert_group("ready"), _expert_group("live")]
    initial = expert.InitialState(
        config, accepted_total=1, ready_groups=(("ready", 0, 0),)
    )
    models = expert.ExpertModels(
        _FixedActor(),
        _FixedRollout(),
        _expert_transitions(config),
        _rpc_timing(response_seconds=10),
    )
    result = expert.ExpertSimulator(models).simulate(
        _expert_scenario(config, groups, initial=initial), (expert.PlanStep(config),)
    )
    assert result.status == "complete"
    delivery = next(e for e in result.events if e.kind == "request_complete")
    publication = next(e for e in result.events if e.kind == "weights_published")
    assert publication.time < delivery.time
    assert dict(delivery.details)["version"] == 0
    assert dict(publication.details)["version"] == 1


@pytest.mark.parametrize("terminal", ["cancel", "drain"])
def test_expert_rpc_initial_tail_resumes_or_cancels_without_new_gpu_work(terminal):
    """A residual response is restored once; canceled callbacks cannot revive it."""
    config = _expert_config()
    groups = [_expert_group("ready"), _expert_group("live")]
    initial = expert.InitialState(
        config,
        accepted_total=1,
        ready_groups=(("ready", 0, 0),),
        running=(
            expert.RunningGroup("live", 0, response_remaining=(("request", 20),)),
        ),
    )
    result = _simulate(
        _expert_scenario(config, groups, initial=initial, terminal=terminal),
        (expert.PlanStep(config),),
    )
    assert result.status == "complete"
    assert not any(e.kind == "service_start" for e in result.events)
    deliveries = [e for e in result.events if e.kind == "request_complete"]
    assert len(deliveries) == (terminal == "drain")
    assert result.cancelled_groups == (terminal == "cancel")
    if deliveries:
        assert deliveries[0].time == 20
        assert dict(deliveries[0].details)["version"] is None


def test_expert_rpc_initial_response_rejects_duplicate_gpu_progress():
    """A request cannot simultaneously hold a GPU slot and await delivery."""
    config = _expert_config()
    group = _expert_group("live", output=2)
    initial = expert.InitialState(
        config,
        running=(
            expert.RunningGroup(
                "live",
                0,
                requests=(expert.RequestProgress("request", 0, 8, 1),),
                response_remaining=(("request", 1),),
            ),
        ),
    )
    with pytest.raises(ValueError, match="initial response"):
        _simulate(
            _expert_scenario(config, [group], initial=initial),
            (expert.PlanStep(config),),
        )


def test_expert_rpc_zero_tail_matches_existing_service_events_and_time():
    """An optional identity timing model preserves the default scheduling path."""
    config = _expert_config()
    scenario = _expert_scenario(config, [_expert_group(str(i)) for i in range(3)])
    steps = (expert.PlanStep(config),)
    default = _simulate(scenario, steps)
    timing = _rpc_timing(response_seconds=0)
    models = expert.ExpertModels(
        _FixedActor(), _FixedRollout(), _expert_transitions(config), timing
    )
    explicit = expert.ExpertSimulator(models).simulate(scenario, steps)
    assert explicit.events == default.events
    assert explicit.seconds == default.seconds
    assert explicit.actor_gpu_seconds == default.actor_gpu_seconds
    assert explicit.rollout_gpu_seconds == default.rollout_gpu_seconds
    assert explicit.snapshot_id != default.snapshot_id
    changed = expert.ExpertSimulator(
        replace(models, rpc_timing=replace(timing, sample_ids=("other-source",)))
    ).simulate(scenario, steps)
    assert changed.snapshot_id != explicit.snapshot_id


def test_expert_rpc_cancelled_tail_callback_is_harmless_after_next_update():
    """A layout change cancels a pending response even if its deadline fires later."""
    source, target = _expert_config(), _expert_config(actor_dp=2)
    ready = _expert_group("ready")
    live = _expert_group("live")
    delayed = replace(_expert_group("later", external=20), available_at=1)
    initial = expert.InitialState(
        source,
        accepted_total=1,
        ready_groups=(("ready", 0, 0),),
        running=(expert.RunningGroup("live", 0, response_remaining=(("request", 5),)),),
    )
    transitions = expert.ExpertTransitionModel(
        _expert_transitions(source).profiles
        + _expert_transitions(target).profiles
        + (_expert_reconfigure(source, target, 0),)
    )
    scenario = _expert_scenario(source, [ready, live, delayed], initial=initial)
    result = _simulate(
        scenario, (expert.PlanStep(target, inflight="cancel"),), transitions
    )
    assert result.status == "complete" and result.seconds > 5
    assert result.cancelled_groups == 1
    assert not any(
        e.kind == "request_complete" and e.subject == "live/request"
        for e in result.events
    )
    assert result.accepted_total == 2


@pytest.mark.parametrize("remaining", [-1, float("nan"), float("inf")])
def test_expert_rpc_initial_tail_requires_finite_nonnegative_remaining(remaining):
    """Unmeasured residual delivery duration cannot silently become zero."""
    with pytest.raises(ValueError, match="response_remaining"):
        expert.RunningGroup("g", 0, response_remaining=(("rpc", remaining),))


def test_expert_rpc_profile_binds_gpu_calibration_and_full_layout():
    """Changing the underlying calibration invalidates the aggregate RPC fit."""

    @dataclass(frozen=True)
    class CalibratedService:
        calibration: tuple = ()

    config = _expert_config().rollout
    first = expert.RPCTimingModel.profile_for(CalibratedService(("fit-one",)), config)
    assert first != expert.RPCTimingModel.profile_for(
        CalibratedService(("fit-two",)), config
    )
    assert first != expert.RPCTimingModel.profile_for(
        CalibratedService(("fit-one",)), replace(config, replicas=2)
    )


def test_expert_rpc_scale_is_per_iteration_and_tail_once_after_chunked_prefill():
    """Multiple prefill/decode iterations share a scale, not repeated RPC tails."""
    config = _expert_config()
    config = replace(config, rollout=replace(config.rollout, max_batch_tokens=4))
    scenario = _expert_scenario(
        config, [_expert_group("a", output=3), _expert_group("b", output=3)]
    )
    models = expert.ExpertModels(
        _FixedActor(),
        _FixedRollout(),
        _expert_transitions(config),
        _rpc_timing(config, service_scale=2, response_seconds=5),
    )
    result = expert.ExpertSimulator(models).simulate(
        scenario, (expert.PlanStep(config),)
    )
    assert result.status == "complete"
    waits = [e for e in result.events if e.kind == "response_wait"]
    assert len(waits) == 2
    for wait in waits:
        delivered = next(
            e
            for e in result.events
            if e.kind == "request_complete" and e.subject == wait.subject
        )
        assert delivered.time - wait.time == 5
    starts = [e for e in result.events if e.kind == "service_start"]
    completions = [e for e in result.events if e.kind == "service_complete"]
    assert len(starts) == len(completions) > 3
    assert all(
        end.time - start.time == 2
        for start, end in zip(starts, completions, strict=True)
    )


def test_expert_rpc_v2_response_delivery_waits_for_collector_resume():
    """Response delivery can finish during a paused collector without early batch use."""
    config = _expert_config()
    groups = [_expert_group("ready"), _expert_group("live")]
    initial = expert.InitialState(
        config,
        accepted_total=1,
        ready_groups=(("ready", 0, 0),),
        running=(expert.RunningGroup("live", 0, response_remaining=(("request", 3),)),),
    )
    result = _simulate(
        _expert_scenario(config, groups, initial=initial, workflow="v2_inline"),
        (expert.PlanStep(config), expert.PlanStep(config)),
    )
    assert result.status == "complete"
    delivery = next(e.time for e in result.events if e.kind == "request_complete")
    ready = next(
        e.time
        for e in result.events
        if e.kind == "group_result_ready" and e.subject == "live"
    )
    assert delivery == 3 and ready == 5
    starts = [e.time for e in result.events if e.kind == "actor_start"]
    assert starts == [0, 5]


def test_structural_publication_drives_simulator_versions_time_and_residency():
    """Publication components run through the same version/ownership event path."""
    config = _expert_config()
    raw = expert.ExpertActorModel(
        expert.ModelShape(128, 64, 4, 4, 2, 32, 4, 2, 1024),
        expert.Hardware("unit-only", 100, 1000, 100, 20, 8, 80),
        "unit-runtime",
    )
    hardware = replace(raw.hardware, intra_node_gbps=1, intra_latency_us=1)
    raw = replace(raw, hardware=hardware)
    zero = {f.name: 0 for f in fields(expert.PublicationRank)[3:]}
    work = expert.PublicationWork(
        expert.publication_binding(raw, config.actor, config.rollout),
        "archon",
        "xccl_broadcast",
        "unit-broadcast-oracle",
        (
            expert.PublicationRank("a0", "actor", "node0", **zero),
            expert.PublicationRank(
                "r0",
                "rollout",
                "node0",
                **(zero | {"expected_receive_bytes": 1_000_000_000}),
            ),
        ),
        (expert.PublicationFlow("a0", "r0", 1_000_000_000, 1),),
        complete=True,
    )
    model = expert.ExpertPublicationModel(
        hardware,
        expert.PublicationRates(
            network_efficiency=1,
            node_fabric_gbps=1000,
            message_launch_seconds=0,
            bucket_ack_seconds=0,
            pause_seconds=0,
            cleanup_seconds=0,
            barrier_seconds=0,
            resume_seconds=0,
        ),
    )
    publication = model.estimate(raw, config.actor, config.rollout, work)
    transitions = _expert_transitions(config)
    transitions = replace(
        transitions,
        profiles=tuple(
            publication.transition(config.key, config.key)
            if p.action == "publish"
            else p
            for p in transitions.profiles
        ),
    )
    scenario = _expert_scenario(config, [_expert_group("one")])
    plan = [expert.PlanStep(config)]
    baseline = _simulate(scenario, plan)
    result = _simulate(scenario, plan, transitions)
    assert result.status == "complete" and result.completed_updates == 1
    assert result.seconds - baseline.seconds == pytest.approx(
        publication.duration.seconds - 3
    )
    assert result.actor_gpu_seconds == result.rollout_gpu_seconds == result.seconds
    assert sum(e.kind == "weights_published" for e in result.events) == 1
    assert not result.remaining_devices

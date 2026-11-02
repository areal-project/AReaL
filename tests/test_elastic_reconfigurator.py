# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, dataclass, replace

import pytest

from areal.elastic.contracts import ElasticObservation
from areal.elastic.reconfigurator import reference as expert


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


def test_reconfigurator_observation_is_immutable():
    sample = observation()
    with pytest.raises(FrozenInstanceError):
        sample.model_version = 3


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


def _synthetic_workload_spec(profile="single-peak"):
    import json
    from pathlib import Path

    from areal.elastic.simulation.workload import ArrivalPhase, WorkloadSpec

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
    from areal.elastic.simulation.workload import (
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
        known, previous, prompt = (set(), [], None)
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
    from areal.elastic.simulation.workload import (
        ArrivalPhase,
        CompositionPhase,
        generate,
    )

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
    start, intensity = (0, 0)
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

    from areal.elastic.simulation.workload import (
        ArrivalPhase,
        CompositionPhase,
        arrival_times,
    )

    spec = replace(
        _synthetic_workload_spec(),
        duration_s=10,
        arrivals=(ArrivalPhase(10, 4),),
        composition=(CompositionPhase(10, (("short", 1.0),)),),
    )
    counts = [len(arrival_times(replace(spec, seed=seed))) for seed in range(1024)]
    assert abs(mean(counts) - 40) < 1
    assert abs(pvariance(counts) - 40) < 8


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

# SPDX-License-Identifier: Apache-2.0

"""Normal-path serving constraints and exact event-level timing oracles."""

from dataclasses import replace
from types import SimpleNamespace

from tests.test_elastic_policy import (
    _expert_config,
    _expert_transitions,
    _FixedActor,
    _FixedRollout,
)

from areal.autopilot.domains.elastic.serving_observer import ServingObserver
from areal.elastic.contracts import BatchWorkload, TrainingConfig, WorkloadStats
from areal.elastic.policy.basic import (
    JointConfig,
    JointOptimizer,
    JointPrediction,
    ServingMetrics,
    ServingSLO,
)
from areal.elastic.policy.basic.cost_model.rollout import RolloutConfig
from areal.elastic.policy.reference import (
    Estimate,
    ExpertModels,
    ExpertSimulator,
    ExternalCall,
    Group,
    InitialState,
    PlanStep,
    Request,
    Scenario,
)


def test_joint_search_filters_slo_before_comparing_gpu_cost():
    """A cheaper completed job cannot displace a serving-feasible deployment."""
    limits = ServingSLO(5, 0.1, 0.95)
    rollout = RolloutConfig(
        "model", "gpu", "runtime", "sampling", 1, 1, 32, 8192, 100000, 3
    )
    configs = tuple(JointConfig(TrainingConfig(n), rollout) for n in (1, 2, 4))
    metrics = ServingMetrics(
        limits, 100, 100, 95, 95, 90, 6, 0.2, "fixture-client-clock"
    )
    predictions = {
        configs[0]: JointPrediction(
            "complete", configs[0], 10, 10, 10, 1, 40, 100, 200, serving=metrics
        ),
        configs[1]: JointPrediction(
            "complete",
            configs[1],
            10,
            20,
            10,
            1,
            40,
            100,
            200,
            serving=replace(
                metrics,
                ttft_met_requests=100,
                tbt_met_requests=96,
                joint_met_requests=96,
            ),
        ),
        configs[2]: JointPrediction(
            "complete",
            configs[2],
            10,
            40,
            10,
            1,
            40,
            100,
            200,
            serving=replace(
                metrics,
                ttft_met_requests=100,
                tbt_met_requests=100,
                joint_met_requests=100,
            ),
        ),
    }
    window = SimpleNamespace(
        current=None, batches=(SimpleNamespace(actor=BatchWorkload(40, 100, 200)),)
    )
    model = SimpleNamespace(evaluate=lambda _, config: predictions[config])
    result = JointOptimizer(
        configs, serving_slo=limits, expected_serving_requests=100
    ).optimize(window, model)
    assert result.config == configs[1]
    assert not result.candidates[0].eligible
    assert metrics.joint_fraction == 0.90  # Marginal 95% pass rates are insufficient.
    assert JointOptimizer(configs).optimize(window, model).config == configs[0]


def observed(groups, *, staleness=2):
    config = _expert_config()
    scenario = Scenario(
        tuple(groups),
        InitialState(config),
        1,
        1,
        1,
        2,
        2,
        8,
        4,
        staleness,
        external_pools=(("tools", 1),),
        selection="task_id",
    )
    simulator = ExpertSimulator(
        ExpertModels(_FixedActor(), _FixedRollout(), _expert_transitions(config))
    )
    result = simulator.simulate(scenario, tuple(PlanStep(config) for _ in groups))
    assert result.status == "complete"
    observer = ServingObserver(scenario.groups, ServingSLO(5, 1.1, 1))
    version = 0
    for event in result.events:
        details = dict(event.details)
        observer.observe(
            event.kind, event.subject, event.time, details, actor_version=version
        )
        if event.kind == "weights_published":
            version = details["version"]
    return observer, result


def test_tool_time_is_not_a_token_gap_and_followup_ttft_starts_when_ready():
    """A ten-second tool call does not become a ten-second TBT violation."""
    group = Group(
        "episode",
        (
            Request("first", 8, 2),
            ExternalCall("tool", "tools", Estimate(10, "assumed"), ("first",)),
            Request("second", 8, 2, ("tool",)),
        ),
        WorkloadStats(2, 20, loss_tokens=4, sequence_lengths=(10, 10)),
        1,
    )
    observer, result = observed((group,))
    first, second = observer.requests.values()
    assert first.ttft_seconds == second.ttft_seconds == 1
    assert first.maximum_tbt_seconds == second.maximum_tbt_seconds == 1
    assert second.ready_at - first.last_token_at == 10
    assert observer.metrics().joint_fraction == 1
    assert observer.training_checks[0]["maximum_version_lag"] == 0


def test_ttft_includes_wait_before_episode_admission():
    """Completion of one rollout does not release the training admission credit."""
    groups = tuple(
        Group(
            str(i),
            (Request("request", 8, 1),),
            WorkloadStats(1, 9, loss_tokens=1, sequence_lengths=(9,)),
            1,
        )
        for i in range(2)
    )
    observer, _ = observed(groups, staleness=0)
    first, second = observer.requests.values()
    assert first.ready_at == second.ready_at == 0
    assert first.ttft_seconds == 1
    assert second.ttft_seconds == 7  # First token + Actor2 + publish3 + next prefill1.
    assert second.maximum_tbt_seconds == 0  # A one-token answer has no token gap.
    assert observer.metrics().joint_fraction == 0.5


def test_publication_stall_enters_tbt_and_training_uses_preupdate_version():
    """Generation remains one request across publication; its gap is not reset."""
    groups = tuple(
        Group(
            str(i),
            (Request("request", 8, n),),
            WorkloadStats(1, 8 + n, loss_tokens=n, sequence_lengths=(8 + n,)),
            1,
        )
        for i, n in enumerate((1, 5))
    )
    observer, _ = observed(groups)
    assert observer.requests["1/request"].maximum_tbt_seconds == 4
    assert observer.metrics().ttft_met_requests == 2
    assert observer.metrics().joint_met_requests == 1
    second = observer.training_checks[1]
    assert second["preupdate_actor_version"] == 1
    assert second["maximum_version_lag"] == 1
    assert second["generated_tokens"] == 5
    assert set(second["behavior_version_counts"]) == {0, 1}

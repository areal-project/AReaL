# SPDX-License-Identifier: Apache-2.0

"""CPU coordinator acceptance; fake Actors do not establish tensor resharding."""

import json

import pytest

from tests.test_elastic_checkpoint import (
    _complete_native_update,
    _ready_native_episodes,
    native_lifecycle,  # noqa: F401 -- shared protocol fixture
)

from areal.elastic.control.actor_reconfiguration import poll_request, submit_request

pytestmark = pytest.mark.usefixtures("native_lifecycle")


def update(manager, first):
    _ready_native_episodes(manager, first, manager.applied_version)
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, manager.applied_version + 1)


@pytest.mark.parametrize("native_lifecycle", [(2, 4)], indirect=True)
def test_actor_dp_roundtrip_preserves_cursor_and_continues_updates(request):
    """2→4→2 changes GPU generation only, retaining the same run and inbox."""
    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    identity, compatibility = manager.identity, manager.compatibility.copy()
    update(manager, 0)
    cursor = manager.consumer.cursor.copy()
    serving = fixture.rollout
    old_measurement = manager.policy_adapter.execution_id
    first = manager.reconfigure_actor(
        world_size=4,
        request_id="grow",
        expected_generation=0,
        expected_weights_version=1,
    )
    assert first["status"] == "applied" and first["generation"] == 1
    assert manager.consumer.cursor == cursor
    assert (
        manager.actor.updates
        == manager.applied_version
        == manager.published_version
        == 1
    )
    assert manager.actor.config.backend == "fsdp:d4"
    assert manager.config.actor.backend == "fsdp:d2"
    assert manager.allocation.parallel.world_size == 4
    assert manager.policy_adapter.profile["native_actor_world_size"] == 4
    assert manager.policy_adapter.execution_id != old_measurement
    assert manager.rollout is serving
    assert fixture.events.index("release") < fixture.events.index("load-1")
    assert (
        manager.reconfigure_actor(
            world_size=4,
            request_id="grow",
            expected_generation=0,
            expected_weights_version=1,
        )
        == first
    )
    update(manager, 4)
    second = manager.reconfigure_actor(
        world_size=2,
        request_id="shrink",
        expected_generation=1,
        expected_weights_version=2,
    )
    assert second["status"] == "applied" and second["generation"] == 2
    assert manager.actor.restore_plan["source_world_size"] == 4
    assert manager.actor.updates == 2
    update(manager, 8)
    assert manager.identity == identity and manager.compatibility == compatibility
    assert (
        manager.applied_version == 3 and manager.consumer.cursor["applied_batches"] == 3
    )
    assert manager.data_progress == {"samples": 12, "effective_tokens": 24}
    manager.close()
    restored = fixture.create()
    try:
        assert restored.actor_world_size == 2
        restored.ensure_actor()
        assert restored.actor.updates == 3 and restored.actor.restore_plan is None
    finally:
        restored.close()


@pytest.mark.parametrize("native_lifecycle", [(2, 4)], indirect=True)
def test_committed_target_restarts_while_checkpoint_still_has_source_layout(
    request,
):
    """A clean restart can retain the source checkpoint until the next target save."""
    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    update(manager, 0)
    manager.reconfigure_actor(
        world_size=4,
        request_id="grow",
        expected_generation=0,
        expected_weights_version=1,
    )
    assert manager.latest_manifest["world_size"] == 2
    # Release without creating a target checkpoint: the durable config receipt
    # and source DCP together must be sufficient for the next coordinator.
    manager._release()
    manager._checkpoint_dirty = False
    manager.close()
    restored = fixture.create()
    try:
        assert restored.actor_world_size == 4
        restored.ensure_actor()
        assert restored.actor.restore_plan["source_world_size"] == 2
        assert restored.actor.updates == 1
        update(restored, 4)
        assert restored.latest_manifest["world_size"] == 4
        assert (
            json.loads((restored.latest_path / "actor-layout.json").read_text())[
                "backend"
            ]
            == "fsdp:d4"
        )
        restored.close()
        again = fixture.create()
        try:
            assert again.actor_world_size == 4
            again.ensure_actor()
            assert again.actor.restore_plan is None and again.actor.updates == 2
        finally:
            again.close()
    finally:
        restored.close()


@pytest.mark.parametrize("native_lifecycle", [(2, 4)], indirect=True)
def test_manual_mailbox_changes_dp_at_completed_update(request):
    """A request file yields a durable receipt and leaves serving usable."""
    manager = request.getfixturevalue("native_lifecycle").manager
    update(manager, 0)
    path = submit_request(
        manager.root,
        world_size=4,
        request_id="manual-grow",
        expected_generation=0,
        expected_weights_version=1,
    )
    poll_request(manager)
    assert not path.exists()
    response = json.loads(path.with_name("response.json").read_text())
    assert response["receipt"]["status"] == "applied"
    assert manager.actor_world_size == 4 and manager.control_snapshot().boundary_ready
    update(manager, 4)
    assert manager.applied_version == 2


@pytest.mark.parametrize("native_lifecycle", [(2, 4)], indirect=True)
def test_resource_layout_roundtrip_preserves_cpu_service_and_progress(request):
    from unittest.mock import MagicMock

    from areal.elastic.control.resource_reconfiguration import (
        ResourceLayout,
        poll_request,
        submit_request,
    )

    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    events, rollout = fixture.events, fixture.rollout
    rollout.resource_reconfiguration_enabled = True
    manager._actor_pool_setter = setter = MagicMock(
        side_effect=lambda devices: events.append(f"pool-{devices}")
    )
    rollout.arm_reconfiguration_observation = MagicMock()
    rollout.reconfiguration_metrics = MagicMock(
        return_value=[{"proxy": "fixture", "marker": None, "first_call": None}]
    )
    rollout.drain_model_calls = MagicMock(side_effect=lambda: events.append("drain"))
    rollout.open_model_calls = MagicMock(side_effect=lambda: events.append("open"))
    rollout.stop_inference_replicas = MagicMock(
        side_effect=lambda: events.append("stop-rollout")
    )
    rollout.rebuild_inference_replicas = MagicMock(
        side_effect=lambda devices, **kwargs: events.append(f"rollout-{devices}")
    )
    update(manager, 0)
    original_identity = manager.identity
    source = manager.resource_layout
    cursor = manager.consumer.cursor.copy()
    # Legacy Actor receipt becomes read-only when complete layouts take authority.
    manager.reconfigure_actor(
        world_size=2,
        request_id="legacy",
        expected_generation=0,
        expected_weights_version=1,
    )
    legacy_bytes = (manager.root / "actor-reconfiguration/state.json").read_bytes()
    events.clear()
    target = ResourceLayout(1, (0,), (1, 2, 3, 4))
    submit_request(
        manager.root,
        target=target,
        request_id="more-rollout",
        expected_generation=0,
        expected_weights_version=1,
    )
    poll_request(manager)
    first = json.loads(
        (manager.root / "resource-reconfiguration/response.json").read_text()
    )["receipt"]
    assert first["status"] == "applied" and first["generation"] == 1
    assert manager.resource_layout == target
    assert manager.actor_world_size == 1 and manager.actor.updates == 1
    assert manager.consumer.cursor == cursor and manager.rollout is rollout
    assert (
        events.index("release")
        < events.index("stop-rollout")
        < events.index("pool-[0]")
    )
    assert (
        events.index("rollout-(1, 2, 3, 4)")
        < events.index("publish-1")
        < events.index("open")
    )
    assert (
        manager.reconfigure_resources(
            target=target,
            request_id="more-rollout",
            expected_generation=0,
            expected_weights_version=1,
        )
        == first
    )
    update(manager, 4)
    timing = [
        json.loads(line)
        for line in (manager.root / "reconfiguration-timing.jsonl")
        .read_text()
        .splitlines()
    ]
    phases = {row["phase"] for row in timing if row["event"] == "phase_end"}
    assert {
        "checkpoint",
        "drain",
        "actor_release",
        "rollout_stop",
        "placement",
        "actor_start",
        "actor_restore",
        "awex_connect",
        "publish",
        "application_and_commit",
        "admission_open",
    } <= phases
    assert all(row["seconds"] >= 0 for row in timing if row["event"] == "phase_end")
    assert any(row["event"] == "first_inference_snapshot" for row in timing)
    assert rollout.arm_reconfiguration_observation.call_count == 1
    placement = json.loads((manager.latest_path / "actor-layout.json").read_text())
    assert placement["format_version"] == 2
    assert placement["reservation"] == [0, 1, 2, 3, 4]
    assert placement["gpu_pool"] == [0] and placement["rollout_devices"] == [1, 2, 3, 4]
    assert manager.policy_adapter.profile["inference_gpus"] == 4
    assert (
        manager.root / "actor-reconfiguration/state.json"
    ).read_bytes() == legacy_bytes
    # Restart selects the authoritative target lease BEFORE the first Actor.
    manager.close()
    restored = fixture.create(actor_pool_setter=setter)
    try:
        restored.ensure_actor()
        assert restored.resource_layout == target and restored.actor.updates == 2
        restored.reconfigure_resources(
            target=source,
            request_id="more-actor",
            expected_generation=1,
            expected_weights_version=2,
        )
        assert restored.resource_layout == source
        assert restored.actor.restore_plan["target_world_size"] == 2
        assert restored.identity == original_identity
        assert restored.consumer.cursor["applied_batches"] == 2
        update(restored, 8)
        assert restored.actor.updates == 3
    finally:
        restored.close()


@pytest.mark.parametrize("native_lifecycle", [(2, 4)], indirect=True)
def test_resource_device_movement_same_dp_uses_rank_state(request):
    from unittest.mock import MagicMock

    from areal.elastic.control.resource_reconfiguration import ResourceLayout

    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    fixture.rollout.resource_reconfiguration_enabled = True
    manager._actor_pool_setter = MagicMock()
    for method in (
        "drain_model_calls",
        "open_model_calls",
        "stop_inference_replicas",
        "rebuild_inference_replicas",
    ):
        setattr(fixture.rollout, method, MagicMock())
    update(manager, 0)
    manager.reconfigure_resources(
        target=ResourceLayout(2, (3, 4), (0, 1, 2)),
        request_id="move",
        expected_generation=0,
        expected_weights_version=1,
    )
    assert manager.actor.restore_plan is None
    assert manager.actor.updates == manager.applied_version == 1
    assert manager.gpu_uuids == ["GPU-3", "GPU-4"]


@pytest.mark.parametrize("native_lifecycle", [(4, 4)], indirect=True)
def test_native_joint_planner_synthetic_trace_roundtrip(request, tmp_path):
    """Real optimizer/SQLite/native transactions; synthetic costs and GPU workers."""
    from dataclasses import asdict, replace
    from unittest.mock import MagicMock

    from tests.test_elastic_lifecycle_optimizer import joint_simulator

    from areal.elastic.contracts import BatchWorkload, TrainingConfig, WorkloadStats
    from areal.elastic.control.resource_reconfiguration import ResourceLayout
    from areal.elastic.policy.basic import (
        JointBatch,
        JointConfig,
        JointOptimizer,
        JointPolicy,
        JointScenario,
        RolloutConfig,
        RolloutStage,
        RolloutWorkload,
    )
    from areal.elastic.policy.basic.actor_cost import ActorPrediction
    from areal.elastic.policy.native_joint import JointPlacement, NativeJointPlanner
    from areal.utils.atomic import atomic_json

    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    manager._actor_pool_setter = MagicMock()
    fixture.rollout.resource_reconfiguration_enabled = True
    for method in (
        "drain_model_calls",
        "open_model_calls",
        "stop_inference_replicas",
        "rebuild_inference_replicas",
    ):
        setattr(fixture.rollout, method, MagicMock())

    placements = tuple(
        JointPlacement(
            JointConfig(
                TrainingConfig(actor),
                RolloutConfig(
                    "synthetic-model",
                    "synthetic-gpu",
                    "synthetic-runtime",
                    "fixed-sampling",
                    5 - actor,
                    1,
                    8,
                    32768,
                    32768,
                    3,
                ),
            ),
            ResourceLayout(actor, tuple(range(actor)), tuple(range(actor, 5))),
        )
        for actor in (4, 2, 3)
    )
    configs = tuple(p.config for p in placements)
    simulator = joint_simulator(configs, switch=2, publish=0.1, checkpoint=0.2)
    simulator.rollout = replace(
        simulator.rollout,
        profiles=tuple(
            replace(
                p,
                prefill_tokens_per_second=100000,
                max_tokens=100000,
                max_request_tokens=32768,
            )
            for p in simulator.rollout.profiles
        ),
    )

    class SyntheticActor:
        def actor_config(self, config):
            return config

        def predict(self, config, stats):
            return ActorPrediction(
                (("train_step", stats.sequence_tokens / (100 * config.world_size)),),
                False,
                ("Synthetic linear Actor throughput; not measured",),
            )

    simulator.actors = SyntheticActor()
    trace = []

    def forecast(owner):
        # Each of four episodes has two sequential requests separated by a tool.
        # These future lengths/outcomes are synthetic inputs, not online facts.
        heavy = owner.applied_version < 3
        prefill, decode, tool = (16, 600, 5) if heavy else (4000, 1, 20)
        prompts = (prefill, prefill + decode + 16)
        lengths = tuple(p + decode for _ in range(4) for p in prompts)
        stats = WorkloadStats(
            8,
            sum(lengths),
            loss_tokens=8 * decode,
            sequence_lengths=lengths,
            trajectory_group_sizes=(2,) * 4,
        )
        batch = JointBatch(
            BatchWorkload(4, 8 * decode, sum(lengths), actor_stats=stats),
            RolloutWorkload(
                tuple(
                    RolloutStage(4 * p, 4 * decode, tool if i == 0 else 0, 4)
                    for i, p in enumerate(prompts)
                ),
                8,
                max(lengths),
            ),
            4,
        )
        current = next(
            p.config for p in placements if p.layout == owner.resource_layout
        )
        scenario = JointScenario(
            (batch,),
            current,
            1,
            owner.inbox.max_staleness,
            4,
            4,
            5,
            provider_version=owner.applied_version,
            accepted_batches=owner.applied_version,
            checkpoint_every=1,
        )
        trace.append(
            {
                "observed_version": owner.applied_version,
                "episodes": [
                    {
                        "episode_id": f"synthetic-{owner.applied_version}-{i}",
                        "requests": [
                            {
                                "prefill_tokens": p,
                                "decode_tokens": decode,
                                "tool_after_seconds": tool if j == 0 else 0,
                            }
                            for j, p in enumerate(prompts)
                        ],
                    }
                    for i in range(4)
                ],
                "scenario": asdict(scenario),
            }
        )
        return scenario

    planner = NativeJointPlanner(
        JointPolicy(simulator, JointOptimizer(configs)),
        placements,
        model_id="synthetic-linear-v1",
        forecast_source="handcrafted-two-turn-trace-v1",
        forecast=forecast,
    )
    manager.resource_planner = planner
    # A real native protocol update is needed before any topology change.
    update(manager, 0)
    decisions = []
    identity = manager.identity
    for index, (status, actor, rollout) in enumerate(
        (("applied", 2, 3), ("unchanged", 2, 3), ("applied", 4, 1))
    ):
        cursor = manager.consumer.cursor.copy()
        progress = manager.data_progress.copy()
        assert manager._poll_batch() is None
        decision = planner.last_result
        decisions.append(decision)
        assert decision["status"] == status
        assert manager.resource_layout.world_size == actor
        assert len(manager.resource_layout.rollout_devices) == rollout
        assert manager.consumer.cursor == cursor and manager.data_progress == progress
        assert manager.inbox.status()["admission_open"]
        assert len(decision["excluded"]) == 1  # DP3 cannot divide batch4.
        assert decision["search"]["unevaluated"] == 0
        if status == "applied":
            receipt = decision["receipt"]
            assert receipt["request"]["policy_bundle_id"].startswith("native-joint-")
            assert receipt["actual"]["weights_version"] == index + 1
        # Repeated polling at the same version must not rerun the provider/search.
        assert manager._poll_batch() is None and len(trace) == index + 1
        update(manager, (index + 1) * 4)
        assert manager.applied_version == index + 2
    assert manager.identity == identity
    assert manager.consumer.cursor["applied_batches"] == 4
    assert manager.data_progress == {"samples": 16, "effective_tokens": 32}
    assert manager.resource_config_application.state()["generation"] == 2
    fixture.rollout.stop_inference_replicas.assert_called()
    assert fixture.rollout.stop_inference_replicas.call_count == 2
    atomic_json(
        tmp_path / "synthetic-trace.json",
        {
            "evidence": "synthetic forecasts and GPU workers; real CPU control protocol",
            "trace": trace,
            "decisions": decisions,
            "final": {
                "updates": manager.applied_version,
                "cursor": manager.consumer.cursor,
                "data_progress": manager.data_progress,
                "config_generation": 2,
            },
        },
    )


@pytest.mark.parametrize("native_lifecycle", [(4, 4)], indirect=True)
def test_native_joint_boundary_allows_arriving_training_work(request):
    """Healthy incoming episodes finish admission before the planning fence."""
    from unittest.mock import MagicMock

    from areal.elastic.policy.native_joint import NativeJointPlanner

    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    manager._actor_pool_setter = MagicMock()
    fixture.rollout.resource_reconfiguration_enabled = True
    update(manager, 0)

    def arriving_calls():
        assert manager.inbox.status()["admission_open"]
        _ready_native_episodes(manager, 4, manager.applied_version)

    fixture.rollout.drain_model_calls = MagicMock(side_effect=arriving_calls)
    fixture.rollout.open_model_calls = MagicMock()
    with manager.control_operation():
        with NativeJointPlanner._quiet_boundary(manager) as quiet:
            assert not quiet
            assert manager.inbox.status()["admission_open"]
            assert manager.inbox.status()["active_credits"] == 4
    fixture.rollout.open_model_calls.assert_called_once()
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, 2)
    assert manager.consumer.cursor["applied_batches"] == 2
    assert manager.data_progress["samples"] == 8

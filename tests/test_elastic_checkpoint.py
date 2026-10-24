# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from areal.elastic.state.checkpoint_store import (
    CheckpointStore,
    atomic_json,
)


@pytest.mark.parametrize("backend", ["fsdp", "megatron"])
@pytest.mark.parametrize("legacy_rng", [False, True])
def test_native_full_checkpoint_restores_optimizer_scheduler_and_rng(
    tmp_path, monkeypatch, backend, legacy_rng
):
    import random
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import numpy as np
    import torch

    from areal.v2.training_service import checkpoint as native

    class Engine:
        rank, world_size, device = 0, 1, torch.device("cpu")

        def __init__(self):
            self.model = torch.nn.Linear(2, 1)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1, gamma=0.5
            )
            self.version = 1
            self.checkpointer = SimpleNamespace(wait_async_saves=MagicMock())
            self.mcore_config = SimpleNamespace(use_checkpoint_opt_param_scheduler=True)

        def get_version(self):
            return self.version

        def set_version(self, version):
            self.version = version

        def save(self, meta):
            root = Path(meta.path)
            root.mkdir(exist_ok=True)
            (root / ".metadata").write_text("test native tensor payload")
            if backend == "megatron":
                (root / "metadata.json").write_text(
                    json.dumps(
                        {
                            "sharded_backend": "torch_dist",
                            "sharded_backend_version": 1,
                            "common_backend": "torch",
                            "common_backend_version": 1,
                        }
                    )
                )
                (root / "common.pt").write_text("test mcore common state")
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                },
                root / "state.pt",
            )
            torch.rand(1)

        def load(self, meta):
            saved = torch.load(Path(meta.path) / "state.pt", weights_only=True)
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            if backend == "megatron":
                self.lr_scheduler.load_state_dict(saved["lr_scheduler"])

    source = Engine()
    source.model(torch.ones(1, 2)).sum().backward()
    source.optimizer.step()
    source.lr_scheduler.step()
    tracker = SimpleNamespace(
        get_states=lambda: {"model": torch.tensor([1, 2, 3], dtype=torch.uint8)},
        set_states=MagicMock(),
    )
    monkeypatch.setattr(native, "_tracker", lambda: tracker)
    identity = "a" * 64
    metadata = dict(
        path=str(tmp_path), backend=backend, identity_sha256=identity, version=1
    )
    saved = native.save_full_checkpoint(source, **metadata)
    if legacy_rng:
        path = tmp_path / "runtime-0.json"
        state = json.loads(path.read_text())
        state.pop("seed_state")
        state.pop("dtensor_rng")
        atomic_json(path, state)
        saved = native._report(state)
    expected = (random.random(), np.random.rand(), torch.rand(4))
    restored = Engine()
    restored.version = 0
    if backend == "megatron":
        scheduler = restored.lr_scheduler

        def restore_additive_scheduler(state):
            # MCore restores by incrementing its existing num_steps. An
            # idempotent PyTorch scheduler would conceal a duplicate load.
            scheduler.load_state_dict(
                {**state, "last_epoch": scheduler.last_epoch + state["last_epoch"]}
            )

        restored.lr_scheduler = SimpleNamespace(
            state_dict=scheduler.state_dict,
            load_state_dict=MagicMock(side_effect=restore_additive_scheduler),
        )
    assert native.inspect_full_checkpoint(restored, loading=True, **metadata) == saved
    assert native.load_full_checkpoint(restored, **metadata) == saved
    actual = (random.random(), np.random.rand(), torch.rand(4))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    assert restored.get_version() == 1
    assert restored.optimizer.param_groups[0]["lr"] == 0.005
    for source_param, restored_param in zip(
        source.model.parameters(), restored.model.parameters()
    ):
        torch.testing.assert_close(source_param, restored_param, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                source.optimizer.state[source_param][key],
                restored.optimizer.state[restored_param][key],
                rtol=0,
                atol=0,
            )
    if backend == "megatron":
        restored.lr_scheduler.load_state_dict.assert_called_once()
        source.checkpointer.wait_async_saves.assert_called_once()
        tracker.set_states.assert_called()
    with pytest.raises(ValueError, match="identity"):
        native.inspect_full_checkpoint(
            restored, **{**metadata, "identity_sha256": "b" * 64}, loading=True
        )


def write_generation(path, *, step, ranks=2):
    (path / "dcp").mkdir()
    (path / "dcp" / ".metadata").write_bytes(b"test metadata")
    for rank in range(ranks):
        atomic_json(
            path / f"runtime-{rank}.json",
            {"rank": rank, "world_size": 2, "policy_version": step},
        )


def commit(store, step):
    return store.commit(
        step=step,
        world_size=2,
        compatibility={"backend": "d2"},
        estimated_bytes=0,
        write=lambda path: write_generation(path, step=step),
    )


def test_checkpoint_incomplete_write_preserves_previous_generation(tmp_path):
    """A missing rank must never replace the last recoverable pointer."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    previous = commit(store, 1)

    with pytest.raises(FileNotFoundError):
        store.commit(
            step=2,
            world_size=2,
            compatibility={"backend": "d2"},
            estimated_bytes=0,
            write=lambda path: write_generation(path, step=2, ranks=1),
        )

    path, manifest = store.latest({"backend": "d2"})
    assert path == previous
    assert manifest["global_step"] == 1
    assert previous.is_dir()


def test_checkpoint_retention_prunes_only_after_successful_commit(tmp_path):
    """Bound storage while retaining the latest two completed generations."""
    store = CheckpointStore(tmp_path, keep=2, free_space_floor=0)
    first = commit(store, 1)
    second = commit(store, 2)
    third = commit(store, 3)

    assert not first.exists()
    assert second.is_dir() and third.is_dir()
    assert store.latest({"backend": "d2"})[0] == third


def test_checkpoint_incompatible_config_rejected(tmp_path):
    """Changing the parallel layout requires an explicit migration."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    commit(store, 1)

    with pytest.raises(ValueError, match="incompatible"):
        store.latest({"backend": "d4"})


def test_checkpoint_corrupted_progress_rejected(tmp_path):
    """A changed manifest must not skip batches while loading older rank states."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    path = commit(store, 1)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest.update(global_step=5, next_batch_id=5)
    atomic_json(path / "manifest.json", manifest)

    with pytest.raises(ValueError, match="rank state"):
        store.latest({"backend": "d2"})


def test_checkpoint_missing_payload_rejected(tmp_path):
    """A manifest alone does not make a checkpoint recoverable."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    path = commit(store, 1)
    (path / "dcp" / ".metadata").unlink()

    with pytest.raises(ValueError, match="payload"):
        store.latest({"backend": "d2"})


def test_checkpoint_pointer_cannot_escape_store(tmp_path):
    """Reject a malformed pointer before reading a different directory."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    (tmp_path / "latest.json").write_text(
        json.dumps({"generation": "../step-00000001"})
    )

    with pytest.raises(ValueError, match="generation"):
        store.latest({"backend": "d2"})


def test_checkpoint_low_space_keeps_existing_checkpoint(tmp_path, monkeypatch):
    """Do not start a large write that would consume the free-space floor."""
    from types import SimpleNamespace

    store = CheckpointStore(tmp_path, free_space_floor=0)
    previous = commit(store, 1)
    monkeypatch.setattr(
        "areal.elastic.state.checkpoint_store.shutil.disk_usage",
        lambda path: SimpleNamespace(free=0),
    )

    with pytest.raises(OSError, match="space"):
        store.commit(
            step=2,
            world_size=2,
            compatibility={},
            estimated_bytes=1,
            write=lambda path: pytest.fail("writer must not run"),
        )

    assert previous.is_dir()
    assert store.latest({"backend": "d2"})[0] == previous


def test_checkpoint_directory_published_without_pointer_can_be_replayed(
    tmp_path, monkeypatch
):
    """An orphan at the same step must not block recovery and a fresh commit."""
    store = CheckpointStore(tmp_path, keep=1, free_space_floor=0)
    previous = commit(store, 1)
    original = atomic_json

    def fail_pointer(path, value):
        if path.name == "latest.json":
            raise OSError("injected pointer failure")
        return original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr("areal.elastic.state.checkpoint_store.atomic_json", fail_pointer)
        with pytest.raises(OSError, match="pointer failure"):
            commit(store, 2)
    assert store.latest({"backend": "d2"})[0] == previous
    replacement = commit(store, 2)
    assert store.latest({"backend": "d2"})[0] == replacement
    assert list(tmp_path.glob("step-*")) == [replacement]


def test_checkpoint_queue_cursor_is_bound_to_optimizer_progress(tmp_path):
    """A queue manifest uses an explicit cursor rather than step as batch ID."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    cursor = {
        "format_version": 1,
        "queue_id": "queue",
        "applied_batches": 2,
        "through_sequence": 3,
        "chain_sha256": "a" * 64,
    }
    store.commit(
        step=2,
        world_size=2,
        compatibility={},
        estimated_bytes=0,
        data_cursor=cursor,
        write=lambda path: write_generation(path, step=2),
    )
    path, manifest = store.latest({})
    assert manifest["data_cursor"] == cursor
    assert "next_batch_id" not in manifest
    manifest["data_cursor"]["applied_batches"] = 3
    atomic_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="data cursor"):
        store.latest({})


def test_checkpoint_retention_keeps_previous_commit_instead_of_newer_orphan(
    tmp_path, monkeypatch
):
    """Only generations that reached latest publication count toward retention."""
    store = CheckpointStore(tmp_path, keep=2, free_space_floor=0)
    previous = commit(store, 1)
    original = atomic_json

    def fail_pointer(path, value):
        if path.name == "latest.json":
            raise OSError("pointer failure")
        original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr("areal.elastic.state.checkpoint_store.atomic_json", fail_pointer)
        with pytest.raises(OSError):
            commit(store, 2)
    current = commit(store, 2)
    assert set(tmp_path.glob("step-*")) == {previous, current}


def test_checkpoint_cannot_publish_older_step(tmp_path):
    """A stale coordinator must not replace latest with earlier training state."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    current = commit(store, 3)
    with pytest.raises(ValueError, match="backwards"):
        commit(store, 2)
    assert store.latest({"backend": "d2"})[0] == current


@pytest.fixture
def native_lifecycle(tmp_path, monkeypatch, request):
    """Protocol fixture: real inbox/checkpoint transactions, no GPU evidence."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from areal.api import FinetuneSpec, WeightUpdateMeta
    from areal.api.cli_args import ActorLifecycleConfig, PPOConfig
    from areal.elastic import runtime as on_demand

    config = PPOConfig()
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"protocol fixture")
    config.actor.path = str(base)
    initial_dp, pool_size = getattr(request, "param", (1, 1))
    config.actor.backend = f"fsdp:d{initial_dp}"
    config.train_dataset.batch_size = config.rollout.consumer_batch_size = 4
    config.rollout.max_head_offpolicyness = 2
    config.recover.mode = "on"
    config.recover.freq_steps = 1
    config.actor_lifecycle = ActorLifecycleConfig(
        mode="on_demand",
        state_root=str(tmp_path / "state"),
        actor_devices=tuple(range(pool_size)),
        rollout_devices=(pool_size,),
        checkpoint_estimated_bytes=1,
        checkpoint_free_space_floor=0,
    )
    monkeypatch.setattr(
        on_demand,
        "resolve_gpu_uuids",
        lambda devices: [f"GPU-{device}" for device in devices],
    )
    monkeypatch.setattr(on_demand, "gpu_processes", lambda: {})
    monkeypatch.setattr(on_demand, "owned_processes", lambda owner: [])
    stopped = []
    events = []
    monkeypatch.setattr(on_demand, "stop_owned_processes", stopped.append)
    rollout = SimpleNamespace(
        version=0,
        _ensure_initialized=lambda: None,
        pause=MagicMock(side_effect=lambda: events.append("pause")),
        resume=MagicMock(side_effect=lambda: events.append("resume")),
        pause_generation=MagicMock(
            side_effect=lambda: events.append("pause-generation")
        ),
        continue_generation=MagicMock(
            side_effect=lambda: events.append("continue-generation")
        ),
    )
    rollout.serving_versions = lambda: [rollout.version]
    actors = []

    class Actor:
        def __init__(self, cfg):
            from areal.api.alloc_mode import ModelAllocation

            self.config = cfg
            self.world_size = ModelAllocation.from_str(cfg.backend).parallel.world_size
            self.version = self.updates = 0
            self.create_process_group = MagicMock()
            self.initialize = MagicMock()
            self.config_perf_tracer = MagicMock()
            self.save_perf_tracer = MagicMock()
            self.quiesce_initialization = MagicMock()
            self.connect_engine = MagicMock()
            self.destroy = MagicMock(
                side_effect=lambda **kwargs: events.append("release")
            )
            self.fail_publication = False
            actors.append(self)

        def set_version(self, version):
            self.version = version

        def full_checkpoint(
            self,
            operation,
            *,
            path,
            backend,
            identity_sha256,
            version,
            restore_plan=None,
        ):
            root = Path(path)
            events.append(f"{operation}-{version}")
            if operation == "save":
                (root / "dcp").mkdir()
                (root / "dcp/.metadata").write_text("protocol")
                atomic_json(root / "dcp/state.json", {"updates": self.updates})
                for rank in range(self.world_size):
                    atomic_json(
                        root / f"runtime-{rank}.json",
                        {
                            "rank": rank,
                            "world_size": self.world_size,
                            "policy_version": version,
                        },
                    )
            else:
                source_size = json.loads((root / "manifest.json").read_text())[
                    "world_size"
                ]
                if source_size != self.world_size:
                    assert restore_plan["source_world_size"] == source_size
                    assert restore_plan["target_world_size"] == self.world_size
                else:
                    assert restore_plan is None
                self.restore_plan = restore_plan
                self.updates = json.loads((root / "dcp/state.json").read_text())[
                    "updates"
                ]
                self.version = version

        def update_weights(self, meta, *, resume_generation=True):
            events.append(f"publish-{meta.version}")
            assert not resume_generation
            rollout.version = meta.version
            if self.fail_publication:
                raise TimeoutError("lost publication response")

    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=48, train_batch_size=4)
    coordinator = {"step": 0}

    def create(policy_adapter=None):
        manager = on_demand.OnDemandActor(
            config, ft_spec, Actor, policy_adapter=policy_adapter
        )
        try:
            manager.bind(
                rollout,
                WeightUpdateMeta.from_awex(),
                capture_cpu=lambda: dict(coordinator),
                restore_cpu=lambda state: coordinator.update(state),
            )
        except BaseException:
            manager.close()
            raise
        return manager

    manager = create()
    yield SimpleNamespace(
        manager=manager,
        create=create,
        actors=actors,
        rollout=rollout,
        coordinator=coordinator,
        stopped=stopped,
        config=config,
        events=events,
    )
    manager.close()


def _ready_native_episodes(manager, first, version):
    from tests.test_elastic_batch_queue import _receive_episode

    for number in range(first, first + 4):
        _receive_episode(manager.inbox, number, version=version)


def _complete_native_update(manager, version):
    manager.publish_update(version)
    manager.maybe_checkpoint(epoch_step=(version - 1) % manager.ft_spec.steps_per_epoch)
    manager.finish_update()


def test_native_on_demand_uses_basic_policy_for_batch_and_release(
    native_lifecycle, monkeypatch
):
    """Default on-demand execution asks Basic before training and idle release."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from areal.elastic.policy.basic import IdleTimeoutPolicy

    manager = native_lifecycle.manager
    assert manager.policy_adapter.policy == IdleTimeoutPolicy(
        manager.options.idle_timeout
    )
    assert manager.policy_hooks is None
    assert (manager.root / "native-profile.json").is_file()
    assert manager.monitor.runtime_history.path.is_file()
    assert not (manager.root / "policy-application").exists()
    original = manager.policy_adapter.policy
    decide = MagicMock(return_value="keep")
    monkeypatch.setattr(
        manager.policy_adapter, "policy", SimpleNamespace(decide=decide)
    )
    _ready_native_episodes(manager, 0, 0)
    assert manager._poll_batch() is None
    assert manager.actor is None and native_lifecycle.actors == []
    assert decide.call_args.args[0].batch_ready
    monkeypatch.setattr(manager.policy_adapter, "policy", original)
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, 1)
    monkeypatch.setattr(
        manager.policy_adapter, "policy", SimpleNamespace(decide=decide)
    )
    now = max(manager.ready_since + 120, manager.idle_since + 60)
    assert not manager.maybe_suspend(now=now)
    assert not decide.call_args.args[0].batch_ready
    assert manager.actor is not None
    monkeypatch.setattr(manager.policy_adapter, "policy", original)
    assert manager.maybe_suspend(now=now)
    assert manager.actor is None


def test_native_measured_basic_without_autopilot_uses_shared_monitor(
    native_lifecycle, tmp_path, monkeypatch
):
    """Ordinary measured Basic uses the default decision boundary without AutoPilot."""
    import builtins
    from dataclasses import asdict

    from areal.elastic.contracts import Measurement
    from areal.elastic.policy.adapter import PolicyAdapter
    from areal.elastic.policy.basic.policy import BasicPolicy

    fixture = native_lifecycle
    profile = {
        **fixture.manager.policy_adapter.profile,
        "measurements": [
            asdict(Measurement(phase, 1, 1, 16))
            for phase in (
                "start",
                "restore",
                "train",
                "checkpoint",
                "publish",
                "release",
            )
        ],
    }
    fixture.manager.close()
    fixture.config.actor_lifecycle.state_root = str(tmp_path / "measured")
    original_import = builtins.__import__

    def without_autopilot(name, *args, **kwargs):
        if name == "areal.autopilot" or name.startswith("areal.autopilot."):
            raise AssertionError("Ordinary Basic must not import AutoPilot")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_autopilot)
    adapter = PolicyAdapter(profile=profile)
    manager = fixture.create(adapter)
    try:
        assert isinstance(adapter.policy, BasicPolicy)
        assert manager.policy_adapter is adapter and manager.policy_hooks is None
        _ready_native_episodes(manager, 0, 0)
        manager.next_batch()
        assert adapter.policy.last_search is not None
        manager.actor.updates += 1
        _complete_native_update(manager, 1)
        assert manager.monitor.runtime_history.path.is_file()
        assert not (manager.root / "policy-application").exists()
        assert not (manager.latest_path / "policy-state.json").exists()
        assert (manager.latest_path / "monitor-state.json").is_file()
        assert manager.policy_adapter.policy.history is manager.monitor.history
        assert manager.monitor.completed_tokens == 8
        manager.close()
        resumed = fixture.create(PolicyAdapter(profile=profile))
        try:
            assert resumed.applied_version == resumed.checkpoint_version == 1
            assert isinstance(resumed.policy_adapter.policy, BasicPolicy)
            assert resumed.monitor.completed_tokens == 8
        finally:
            resumed.close()
    finally:
        manager.close()


def test_native_reference_selection_fails_before_creating_actor(native_lifecycle):
    """A reserved Reference selector cannot start a training generation."""
    fixture = native_lifecycle
    fixture.manager.close()
    fixture.config.actor_lifecycle.policy = "reference"
    from areal.elastic.runtime import validate_on_demand

    with pytest.raises(NotImplementedError, match="reference.*not implemented"):
        validate_on_demand(fixture.config)
    with pytest.raises(NotImplementedError, match="reference.*not implemented"):
        fixture.create()
    assert fixture.actors == []


def test_native_lifecycle_three_cycles_restore_cursor_and_actor_state(native_lifecycle):
    fixture = native_lifecycle
    manager = fixture.manager
    assert manager.actor is None and fixture.actors == []
    rollout = manager.rollout
    for version in range(3):
        _ready_native_episodes(manager, version * 4, version)
        trajectories = manager.next_batch()
        assert len(trajectories) == 4
        assert manager.actor.updates == version
        manager.actor.initialize.assert_called_once_with(
            role="actor",
            addr=None,
            ft_spec=manager.ft_spec,
            wait=True,
            worker_seed=fixture.config.seed,
        )
        manager.actor.updates += 1
        fixture.coordinator["step"] = version + 1
        _complete_native_update(manager, version + 1)
        assert manager.inbox.status()["active_credits"] == 0
        assert not manager.maybe_suspend(now=manager.ready_since + 59)
        assert manager.maybe_suspend(now=manager.ready_since + 121)
        assert manager.actor is None and manager.owner is None
        assert manager.rollout is rollout
    manager.close()
    recovered = fixture.create()
    try:
        assert recovered.actor is None and recovered.applied_version == 3
        assert recovered.consumer.cursor["applied_batches"] == 3
        _ready_native_episodes(recovered, 12, 3)
        recovered.next_batch()
        assert recovered.actor.updates == 3
    finally:
        recovered.close()


def test_native_lost_publication_ack_rolls_back_unsaved_update(native_lifecycle):
    manager = native_lifecycle.manager
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    old = manager.actor
    old.updates = 1
    old.fail_publication = True
    with pytest.raises(TimeoutError):
        manager.publish_update(1)
    assert manager.latest_manifest is None
    assert manager.inbox.status()["published_version"] == 0
    state = manager.control_snapshot()
    assert state.applied_version == 1
    assert state.published_version == state.checkpoint_version == 0
    manager.recover_uncertain(TimeoutError("lost response"))
    assert manager.applied_version == manager.actor.updates == 0
    assert manager.actor is not old
    assert manager.active is None and manager.rollout.version == 0
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, 1)
    assert manager.inbox.status()["active_credits"] == 0


def test_native_uncertain_uncommitted_update_replays_original_batch(native_lifecycle):
    manager = native_lifecycle.manager
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    original_id = manager.active.batch_id
    manager.actor.updates = 1
    manager.recover_uncertain(RuntimeError("worker died during update"))
    assert manager.applied_version == manager.actor.updates == 0
    manager.next_batch()
    assert manager.active.batch_id == original_id
    assert manager.actor.updates == 0


@pytest.mark.parametrize("lost_restore_ack", [False, True])
def test_native_partial_publication_with_unchanged_version_requires_full_restore(
    native_lifecycle, monkeypatch, lost_restore_ack
):
    """Version labels cannot certify tensors after a partially failed transfer."""
    fixture, manager = native_lifecycle, native_lifecycle.manager
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    manager.actor.updates = 1
    _complete_native_update(manager, 1)
    _ready_native_episodes(manager, 4, 1)
    manager.next_batch()
    batch_id = manager.active.batch_id
    actor = manager.actor
    actor.updates = 2
    fixture.rollout.weights = [1, 1]

    def partially_transfer(*args, **kwargs):
        fixture.rollout.weights[0] = 2
        raise TimeoutError("partial transfer before version acknowledgment")

    actor.update_weights = partially_transfer
    with pytest.raises(TimeoutError, match="partial transfer"):
        manager.publish_update(2)
    assert fixture.rollout.version == manager.checkpoint_version == 1
    assert manager.inbox.status()["published_version"] == 1
    resumes = fixture.rollout.resume.call_count
    original = type(actor).update_weights

    def restore_weights(restored, meta, **kwargs):
        assert not manager.inbox.status()["admission_open"]
        assert fixture.rollout.resume.call_count == resumes
        original(restored, meta, **kwargs)
        fixture.rollout.weights[:] = [restored.updates] * 2
        if lost_restore_ack:
            raise TimeoutError("restored publication acknowledgment lost")

    monkeypatch.setattr(type(actor), "update_weights", restore_weights)
    if lost_restore_ack:
        with pytest.raises(TimeoutError, match="restored publication"):
            manager.recover_uncertain(TimeoutError("AWEX failed"))
        assert not manager.inbox.status()["admission_open"]
        assert not manager.control_snapshot().boundary_ready
        assert fixture.rollout.resume.call_count == resumes
        assert not manager.maybe_suspend(now=manager.ready_since + 1000)
        manager._checkpoint_dirty = True
        with pytest.raises(RuntimeError, match="fully published"):
            manager.save_checkpoint(reason="uncertain")
        manager._checkpoint_dirty = False
        restored = manager.actor
        manager.close()
        restored.destroy.assert_called_with(training_stopped=True)
    else:
        manager.recover_uncertain(TimeoutError("AWEX failed"))
        assert fixture.rollout.weights == [1, 1]
        assert manager.inbox.status()["admission_open"]
        assert fixture.rollout.resume.call_count == resumes + 1
        manager.next_batch()
        assert manager.active.batch_id == batch_id
        assert manager.actor.updates == 1


def test_native_publication_precedes_periodic_save_and_release_waits_for_finish(
    native_lifecycle,
):
    """No control or release boundary opens between publication and CPU hooks."""
    fixture = native_lifecycle
    manager = fixture.manager
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    manager.actor.updates += 1
    manager.publish_update(1)
    assert manager.latest_manifest is None
    assert manager.inbox.status()["checkpoint_version"] == 0
    assert manager.inbox.status()["published_version"] == 1
    assert not manager.control_snapshot().boundary_ready
    assert not manager.maybe_suspend(now=manager.ready_since + 1000)
    assert manager.maybe_checkpoint(epoch_step=0)
    assert fixture.events[-2:] == ["publish-1", "save-1"]
    manager.finish_update()
    assert fixture.events[-2:] == ["continue-generation", "resume"]
    assert manager.control_snapshot().boundary_ready


def test_native_multiple_updates_only_save_at_configured_period(native_lifecycle):
    """Published batches free admission without committing their checkpoint cursor."""
    fixture = native_lifecycle
    manager = fixture.manager
    manager.checkpoint_frequency.step_ctl.frequency_steps = 3
    for version in range(3):
        _ready_native_episodes(manager, version * 4, version)
        manager.next_batch()
        manager.actor.updates += 1
        _complete_native_update(manager, version + 1)
        assert manager.inbox.status()["active_credits"] == 0
        assert manager.inbox.status()["published_version"] == version + 1
        checkpoint_version = 3 if version == 2 else 0
        assert manager.inbox.status()["checkpoint_version"] == checkpoint_version
        state = manager.control_snapshot()
        assert state.applied_version == state.published_version == version + 1
        assert state.checkpoint_version == checkpoint_version
        assert state.boundary_ready
        assert manager.policy_adapter.applied_version == state.applied_version
        assert manager.policy_adapter.published_version == state.published_version
        assert manager.policy_adapter.checkpoint_version == checkpoint_version
    assert [event for event in fixture.events if event.startswith("save-")] == [
        "save-3"
    ]
    assert manager.latest_manifest["data_cursor"] == manager.consumer.cursor


def test_native_release_refreshes_cpu_state_after_periodic_checkpoint(native_lifecycle):
    """A same-version periodic snapshot predates end-of-update CPU mutations."""
    fixture = native_lifecycle
    manager = fixture.manager
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    manager.actor.updates += 1
    manager.publish_update(1)
    assert manager.maybe_checkpoint(epoch_step=0)
    periodic = manager.latest_path
    fixture.coordinator["step"] = 1
    manager.finish_update()
    assert manager.maybe_suspend(now=manager.ready_since + 121)
    assert manager.latest_path != periodic
    assert json.loads((periodic / "coordinator.json").read_text()) == {"step": 0}
    assert json.loads((manager.latest_path / "coordinator.json").read_text()) == {
        "step": 1
    }
    manager.close()
    fixture.coordinator["step"] = -1
    recovered = fixture.create()
    try:
        assert fixture.coordinator["step"] == 1 and recovered.applied_version == 1
    finally:
        recovered.close()


def test_native_idle_release_saves_latest_state_and_does_not_duplicate_save(
    native_lifecycle,
):
    """A release snapshot covers updates newer than the periodic checkpoint."""
    fixture = native_lifecycle
    manager = fixture.manager
    manager.config.recover.mode = "off"
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, 1)
    assert manager.latest_manifest is None
    assert manager.maybe_suspend(now=manager.ready_since + 121)
    assert fixture.events[-2:] == ["save-1", "release"]
    manager.close()
    recovered = fixture.create()
    try:
        _ready_native_episodes(recovered, 4, 1)
        recovered.next_batch()
        assert recovered.actor.updates == 1
    finally:
        recovered.close()
    assert fixture.events.count("save-1") == 1


def test_native_failed_release_snapshot_retains_actor_and_consumer(
    native_lifecycle,
    monkeypatch,
):
    """A failed planned snapshot cannot turn into an unsaved release."""
    manager = native_lifecycle.manager
    manager.config.recover.mode = "off"
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, 1)
    actor = manager.actor
    with monkeypatch.context() as patch:
        patch.setattr(
            manager.store,
            "commit",
            lambda **kwargs: (_ for _ in ()).throw(OSError("disk full")),
        )
        with pytest.raises(OSError, match="disk full"):
            manager.maybe_suspend(now=manager.ready_since + 121)
        with pytest.raises(ExceptionGroup, match="Checkpoint incomplete"):
            manager.close()
        actor.destroy.assert_not_called()
        with pytest.raises(BlockingIOError):
            manager.inbox.queue.consumer()
    assert manager.maybe_suspend(now=manager.ready_since + 121)


def test_native_batch_arriving_during_release_snapshot_keeps_actor_ready(
    native_lifecycle,
    monkeypatch,
):
    """Recheck the inbox after a slow save before destroying the workers."""
    manager = native_lifecycle.manager
    manager.config.recover.mode = "off"
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, 1)
    save = manager.actor.full_checkpoint

    def receive_while_saving(*args, **kwargs):
        result = save(*args, **kwargs)
        _ready_native_episodes(manager, 4, 1)
        return result

    monkeypatch.setattr(manager.actor, "full_checkpoint", receive_while_saving)
    assert not manager.maybe_suspend(now=manager.ready_since + 121)
    manager.actor.destroy.assert_not_called()
    assert manager.consumer.peek().policy_version == 1


def test_native_fault_replays_unsaved_prefix_and_rejects_abandoned_episodes(
    native_lifecycle,
):
    """Restore a periodic checkpoint, reconcile serving, then replay lost PPO work."""
    from tests.test_elastic_batch_queue import _online_payload, _receive_episode

    manager = native_lifecycle.manager
    for version in range(3):
        _ready_native_episodes(manager, version * 4, version)
        manager.next_batch()
        manager.actor.updates += 1
        _complete_native_update(manager, version + 1)
        manager.config.recover.mode = "off"
    ready = _receive_episode(manager.inbox, 12, version=3)
    admitted = manager.inbox.admit("source", "inflight", 0)
    assert manager.latest_manifest["global_step"] == 1
    manager.recover_uncertain(RuntimeError("Actor lost after v3 publication"))
    assert (
        manager.applied_version == manager.rollout.version == manager.actor.updates == 1
    )
    for episode in (ready, admitted):
        assert (
            manager.inbox.episode(episode["episode_id"])["rejection"]
            == "recovery_rollback"
        )
    digest = manager.inbox.receive(
        admitted["episode_id"], admitted["lease"], _online_payload(3)
    )
    assert digest
    manager.inbox.acknowledge_source(admitted["episode_id"], admitted["lease"])
    assert manager.inbox.episode(admitted["episode_id"])["status"] == "rejected"
    for version in (1, 2):
        manager.next_batch()
        assert manager.active.policy_version == version
        manager.actor.updates += 1
        _complete_native_update(manager, version + 1)
    assert manager.actor.updates == 3
    assert manager.inbox.status()["active_credits"] == 0


def test_native_checkpoint_seconds_schedule_works_in_cpu_coordinator(native_lifecycle):
    """Wall-clock saving must not require a distributed coordinator process group."""
    manager = native_lifecycle.manager
    manager.checkpoint_frequency.step_ctl.frequency_steps = None
    manager.checkpoint_frequency.time_ctl.frequency_seconds = 0
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, 1)
    assert manager.latest_manifest["global_step"] == 1


@pytest.mark.parametrize("native_lifecycle", [(1, 1), (2, 4)], indirect=True)
def test_native_trainer_defers_actor_initialization_and_legacy_recover(
    native_lifecycle, monkeypatch
):
    from unittest.mock import MagicMock

    from areal.trainer import rl_trainer

    fixture = native_lifecycle
    fixture.manager.close()
    config = fixture.config
    config.actor._version = config.rollout._version = "v2"
    config.rollout.backend = "sglang:d1"
    config.sglang.model_path = config.actor.path
    config.actor.scheduling_spec[0].gpu = config.rollout.scheduling_spec[0].gpu = 1
    config.total_train_epochs, config.total_train_steps = 1, 12
    config.actor.adv_norm = config.actor.reward_norm = None
    config.gconfig.n_samples = 1
    config.gconfig.reward_normalization = False
    config.rollout.agent.mode = "online"
    config.scheduler.type = "local"
    monkeypatch.setenv("AREAL_SPMD_MODE", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(rl_trainer, "LocalScheduler", MagicMock())
    monkeypatch.setattr(
        rl_trainer, "load_hf_processor_and_tokenizer", lambda path: (None, object())
    )
    monkeypatch.setattr(rl_trainer.logging, "setup_file_logging", MagicMock())
    for name in ("Saver", "Evaluator", "StatsLogger"):
        instance = MagicMock()
        instance.state_dict.return_value = {}
        monkeypatch.setattr(rl_trainer, name, MagicMock(return_value=instance))
    legacy_recover = MagicMock()
    monkeypatch.setattr(rl_trainer, "RecoverHandler", legacy_recover)
    serving = MagicMock()
    serving.serving_versions.return_value = [0]
    monkeypatch.setattr(
        rl_trainer.PPOTrainer, "_init_rollout", MagicMock(return_value=serving)
    )
    factory = MagicMock()
    monkeypatch.setattr(rl_trainer.PPOTrainer, "_create_train_engine", factory)
    trainer = rl_trainer.PPOTrainer(config)
    try:
        assert trainer.actor is None and fixture.actors == []
        factory.assert_not_called()
        legacy_recover.return_value.load.assert_not_called()
        assert trainer.scheduler is not None and trainer.rollout_scheduler is not None
        assert trainer._on_demand.inbox.status()["active_credits"] == 0
    finally:
        trainer.close()


def test_native_failed_teardown_preserves_fence_and_retries_same_generation(
    native_lifecycle, monkeypatch
):
    from areal.elastic import runtime as on_demand

    manager = native_lifecycle.manager
    manager.ensure_actor()
    actor, owner = manager.actor, manager.owner
    actor.destroy.side_effect = RuntimeError("inference teardown unknown")
    actor.save_perf_tracer.side_effect = RuntimeError("worker unreachable")
    monkeypatch.setattr(
        manager,
        "_event",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(RuntimeError, match="teardown unknown"):
        manager._release()
    assert manager.actor is actor and manager.owner == owner
    assert native_lifecycle.stopped == [owner]
    assert actor.destroy.call_args.kwargs == {"training_stopped": True}
    with pytest.raises(BlockingIOError):
        manager.inbox.queue.consumer()
    actor.destroy.side_effect = None
    monkeypatch.setattr(on_demand, "gpu_processes", lambda: {"GPU-0": [123]})
    with pytest.raises(RuntimeError, match="empty"):
        manager._release()
    assert manager.owner == owner
    monkeypatch.setattr(on_demand, "gpu_processes", lambda: {})
    manager._release()
    assert manager.actor is None and manager.owner is None


def test_native_failed_update_skips_blocked_worker_rpcs_and_retains_fence(
    native_lifecycle,
):
    """A failed collective is stopped before any surviving inference teardown."""
    fixture, manager = native_lifecycle, native_lifecycle.manager
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    actor, owner = manager.actor, manager.owner
    fixture.events.clear()

    def disconnect(*, training_stopped=False):
        assert training_stopped
        assert owner in fixture.stopped
        assert fixture.events == ["pause"]
        raise RuntimeError("inference teardown unknown")

    actor.destroy.side_effect = disconnect
    with pytest.raises(RuntimeError, match="inference teardown unknown"):
        manager.recover_uncertain(TimeoutError("AWEX collective timed out"))
    actor.save_perf_tracer.assert_not_called()
    assert manager.actor is actor and manager.owner == owner
    assert not manager.inbox.status()["admission_open"]
    with pytest.raises(BlockingIOError):
        manager.inbox.queue.consumer()
    actor.destroy.side_effect = None
    manager.close()
    actor.destroy.assert_called_with(training_stopped=True)

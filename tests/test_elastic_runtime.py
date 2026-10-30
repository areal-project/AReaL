# SPDX-License-Identifier: Apache-2.0

"""CPU protocol tests: no GPU performance or distributed tensor-restore claim."""

import json
from pathlib import Path

import pytest

from areal.utils.atomic import atomic_json


def _inline_payload(version=0, rows=1):
    import torch

    return {
        "input_ids": torch.tensor([[1, 2, 3, 4]] * rows),
        "attention_mask": torch.ones((rows, 4), dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 0, 1, 1]] * rows),
        "logprobs": torch.tensor([[0.0, 0.0, -0.3, -0.4]] * rows),
        "versions": torch.tensor([[-1, -1, version, version]] * rows),
        "rewards": torch.ones(rows),
    }


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
    initial_dp, pool_size = 1, 1
    config.actor.backend = f"fsdp:d{initial_dp}"
    config.rollout.backend = "sglang:d1"
    config.gconfig.n_samples = getattr(request, "param", 1)
    config.train_dataset.batch_size = config.rollout.consumer_batch_size = (
        2 if config.gconfig.n_samples > 1 else 4
    )
    config.rollout.max_head_offpolicyness = 2
    config.recover.mode = "on"
    config.recover.freq_steps = 1
    config.actor_lifecycle = ActorLifecycleConfig(
        mode="on_demand",
        policy="idle",
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

    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=48,
        train_batch_size=config.train_dataset.batch_size,
    )
    coordinator = {"step": 0}

    def create(policy_adapter=None):
        manager = on_demand.OnDemandActor(
            config,
            ft_spec,
            Actor,
            policy_adapter=policy_adapter,
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


def _ready_native_episodes(manager, first, version, *, token_version=None):
    from areal.infra.rpc.serialization import serialize_value

    manager.inbox.queue.enqueue(
        f"inline-{version}",
        {
            "episode_ids": [
                str(i) for i in range(first, first + manager.inbox.batch_size)
            ],
            "trajectories": [
                serialize_value(
                    _inline_payload(
                        version if token_version is None else token_version,
                        rows=manager.inbox.group_size,
                    )
                )
                for _ in range(manager.inbox.batch_size)
            ],
        },
        policy_version=version,
    )


def _complete_native_update(manager, version):
    manager.publish_update(version)
    manager.maybe_checkpoint(epoch_step=(version - 1) % manager.ft_spec.steps_per_epoch)
    manager.finish_update()


@pytest.mark.parametrize("mode", ["resident", "idle"])
def test_resident_starts_with_collection_before_a_batch_exists(
    native_lifecycle, monkeypatch, mode
):
    """Only the resident policy acquires its initial Actor without ready data."""
    import importlib.util
    import sys
    from types import ModuleType, SimpleNamespace

    # This CPU lifecycle check does not initialize experiment dashboards.
    for name in ("trackio", "wandb"):
        if importlib.util.find_spec(name) is None:
            monkeypatch.setitem(sys.modules, name, ModuleType(name))

    from areal.trainer.rl_trainer import PPOTrainer

    fixture = native_lifecycle
    manager = fixture.manager
    manager.policy_adapter.mode = mode
    order = []
    monkeypatch.setattr(
        manager.inbox, "start", lambda *args, **kwargs: order.append("collection")
    )
    original_factory = manager.actor_factory

    def actor_factory(config):
        assert order == ["collection"]
        assert manager.inbox.queue.status()["counts"]["pending"] == 0
        order.append("actor")
        return original_factory(config)

    monkeypatch.setattr(manager, "actor_factory", actor_factory)

    def train_step(**kwargs):
        assert kwargs["global_step"] == 0
        assert (manager.actor is not None) == (mode == "resident")
        order.append("wait-for-batch")

    trainer = SimpleNamespace(
        config=fixture.config,
        recover_info=None,
        train_dataloader=[{}],
        _on_demand=manager,
        rollout=fixture.rollout,
        _requires_proxy_workflow=lambda workflow: False,
        _train_step=train_step,
    )
    PPOTrainer.train(trainer, workflow=object(), total_epochs=1)
    assert order == (
        ["collection", "actor", "wait-for-batch"]
        if mode == "resident"
        else ["collection", "wait-for-batch"]
    )


@pytest.mark.parametrize("native_lifecycle", [4], indirect=True)
def test_grouped_update_preserves_eight_samples_and_checkpoint_progress(
    native_lifecycle,
):
    from areal.utils.data import concat_batch

    manager = native_lifecycle.manager
    for version in range(2):
        _ready_native_episodes(manager, version * 2, version)
        groups = manager.next_batch()
        assert concat_batch(groups)[1].traj_group_sizes == [4, 4]
        manager.actor.updates += 1
        _complete_native_update(manager, version + 1)
        assert manager.data_progress == {
            "task_groups": (version + 1) * 2,
            "samples": (version + 1) * 8,
            "effective_tokens": (version + 1) * 16,
        }
        assert manager.compatibility["group_size"] == 4
        assert manager.maybe_suspend(now=manager.ready_since + 121)
        manager.data_progress = {}  # Restore must come from durable checkpoint state.
        manager._restore_data_progress()
        manager.start_actor()
        assert manager.data_progress["samples"] == (version + 1) * 8
        assert manager.actor.updates == version + 1


def test_single_sample_multiturn_progress_restores_actual_rows(native_lifecycle):
    from areal.infra.rpc.serialization import serialize_value

    manager = native_lifecycle.manager
    manager.inbox.queue.enqueue(
        "inline-0",
        {
            "episode_ids": [str(i) for i in range(4)],
            "trajectories": [
                serialize_value(_inline_payload(rows=2)) for _ in range(4)
            ],
        },
        policy_version=0,
    )
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, 1)
    manager.data_progress = {}
    manager._restore_data_progress()
    assert manager.data_progress == {
        "task_groups": 4,
        "samples": 8,
        "effective_tokens": 16,
    }


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
        assert manager.inbox.status()["counts"]["published"] == (version + 1) * 4
        assert not manager.maybe_suspend(now=manager.ready_since + 59)
        assert manager.maybe_suspend(now=manager.ready_since + 121)
        assert manager.actor is None and manager.owner is None
        assert manager.rollout is rollout
    assert manager.consumer.cursor["applied_batches"] == 3


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
    assert manager.checkpoint_version == 0
    assert manager.inbox.status()["published_version"] == 1
    assert manager.active is not None
    assert not manager.maybe_suspend(now=manager.ready_since + 1000)
    assert manager.maybe_checkpoint(epoch_step=0)
    assert fixture.events[-2:] == ["publish-1", "save-1"]
    manager.finish_update()
    assert fixture.events[-2:] == ["continue-generation", "resume"]
    assert manager.active is None


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
    assert manager.applied_version == 1


def test_native_batch_arriving_during_decision_keeps_actor_without_save(
    native_lifecycle, monkeypatch
):
    fixture = native_lifecycle
    fixture.config.recover.mode = "disabled"
    manager = fixture.manager
    _ready_native_episodes(manager, 0, 0)
    manager.next_batch()
    manager.actor.updates += 1
    _complete_native_update(manager, 1)
    actor = manager.actor

    def decide(batch, **kwargs):
        assert batch is None
        _ready_native_episodes(manager, 4, 1)
        return "release"

    with monkeypatch.context() as patch:
        patch.setattr(manager.policy_adapter, "decide", decide)
        assert not manager.maybe_suspend(now=manager.ready_since + 121)
    assert manager.actor is actor
    assert "save-1" not in fixture.events
    assert len(manager.next_batch()) == 4
    manager.actor.updates += 1
    _complete_native_update(manager, 2)


def test_native_long_tail_token_versions_are_preserved_and_measured(native_lifecycle):
    fixture = native_lifecycle
    fixture.config.recover.mode = "disabled"
    manager = fixture.manager
    for version in range(5):
        _ready_native_episodes(manager, version * 4, version, token_version=0)
        rows = manager.next_batch()
        assert all(
            (row["versions"][row["loss_mask"].bool()] == 0).all() for row in rows
        )
        manager.actor.updates += 1
        _complete_native_update(manager, version + 1)
    events = [
        json.loads(line)
        for line in (manager.root / "events.jsonl").read_text().splitlines()
    ]
    training = [event for event in events if event["state"] == "TRAINING"][-1]
    assert training["token_age_max"] == training["token_age_mean"] == 4
    assert (
        training["tokens_older_than_head_window"] == training["effective_tokens"] == 8
    )


@pytest.mark.parametrize(
    "backend,restore", [("megatron", False), ("megatron", True), ("fsdp", True)]
)
def test_fast_bootstrap_only_used_for_megatron_full_restore(
    native_lifecycle, backend, restore
):
    """First boot and FSDP retain normal initialization; restored Megatron opts in."""
    from unittest.mock import MagicMock

    from areal.api.alloc_mode import ModelAllocation

    manager = native_lifecycle.manager
    manager.allocation = ModelAllocation.from_str(f"{backend}:d1")
    manager.config.actor.dtype = "bfloat16"
    if restore:
        manager.latest_path = manager.root / "checkpoint"
    factory = manager.actor_factory

    def create(config):
        actor = factory(config)
        actor.full_checkpoint = MagicMock(return_value=[])
        return actor

    manager.actor_factory = create
    actor = manager.ensure_actor()
    assert actor.initialize.call_args.kwargs.get(
        "restore_from_full_checkpoint", False
    ) == (restore and backend == "megatron")
    assert manager.ready_since is not None
    manager._release()
    actor.destroy.assert_called_once_with(parallel_child_cleanup=True)


def test_failed_restore_cannot_be_reused_or_saved_on_close(native_lifecycle):
    """An incomplete generation is fenced and cleaned without replacing its checkpoint."""
    from unittest.mock import MagicMock

    from areal.api.alloc_mode import ModelAllocation

    manager = native_lifecycle.manager
    manager.allocation = ModelAllocation.from_str("megatron:d1")
    manager.config.actor.dtype = "bfloat16"
    manager.latest_path = manager.root / "checkpoint"
    factory = manager.actor_factory

    def create(config):
        actor = factory(config)
        actor.full_checkpoint = MagicMock(side_effect=ValueError("invalid checkpoint"))
        return actor

    manager.actor_factory = create
    with pytest.raises(ValueError, match="invalid checkpoint"):
        manager.ensure_actor()
    actor = manager.actor
    actor.connect_engine.assert_not_called()
    assert manager.ready_since is None
    with pytest.raises(RuntimeError, match="restore is incomplete"):
        manager.ensure_actor()
    manager.close()
    assert actor.full_checkpoint.call_count == 1
    actor.destroy.assert_called_once_with(training_stopped=True)
    assert manager.actor is None

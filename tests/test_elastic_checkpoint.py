# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from areal.elastic.infra.state.checkpoint_store import CheckpointStore, atomic_json


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
    """The manifest version must match the saved rank states."""
    store = CheckpointStore(tmp_path, free_space_floor=0)
    path = commit(store, 1)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest.update(global_step=5)
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
        "areal.elastic.infra.state.checkpoint_store.shutil.disk_usage",
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
        patch.setattr(
            "areal.elastic.infra.state.checkpoint_store.atomic_json", fail_pointer
        )
        with pytest.raises(OSError, match="pointer failure"):
            commit(store, 2)
    assert store.latest({"backend": "d2"})[0] == previous
    replacement = commit(store, 2)
    assert store.latest({"backend": "d2"})[0] == replacement
    assert list(tmp_path.glob("step-*")) == [replacement]


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
        patch.setattr(
            "areal.elastic.infra.state.checkpoint_store.atomic_json", fail_pointer
        )
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


@pytest.mark.parametrize("initial_dp,pool_size", [(1, 1), (2, 4)])
@pytest.mark.parametrize("enable_control", [False, True])
def test_native_trainer_owns_actor_initialization_and_standard_recover(
    tmp_path, monkeypatch, enable_control, initial_dp, pool_size
):
    from unittest.mock import MagicMock

    from areal.api.cli_args import ElasticConfig, PPOConfig
    from areal.trainer import rl_trainer

    config = PPOConfig()
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "model.safetensors").write_bytes(b"CPU protocol model placeholder")
    config.actor.path = str(base)
    config.actor.backend = f"fsdp:d{initial_dp}"
    config.train_dataset.batch_size = config.rollout.consumer_batch_size = 4
    config.rollout.max_head_offpolicyness = 2
    config.elastic = ElasticConfig(
        enabled=True,
        state_root=str(tmp_path / "trainer"),
        actor_devices=tuple(range(pool_size)),
        rollout_devices=(pool_size,),
        checkpoint_estimated_bytes=1,
        checkpoint_free_space_floor=0,
    )
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
    native_recover = MagicMock()
    native_recover.return_value.load.return_value = None
    monkeypatch.setattr(rl_trainer, "RecoverHandler", native_recover)
    serving = MagicMock()
    serving.resource_reconfiguration_enabled = False
    serving.serving_versions.return_value = [0]
    monkeypatch.setattr(
        rl_trainer, "RolloutControllerV2", MagicMock(return_value=serving)
    )
    monkeypatch.setattr(
        rl_trainer.SGLangConfig, "build_args", MagicMock(return_value={})
    )
    monkeypatch.setattr(
        "areal.elastic.infra.reconfiguration.resolve_gpu_uuids", lambda devices: []
    )
    monkeypatch.setattr("areal.elastic.infra.reconfiguration.gpu_processes", lambda: {})
    factory = MagicMock()
    factory.return_value.config = config.actor
    factory.return_value.get_version.return_value = 0
    serving.get_version.return_value = 0
    monkeypatch.setattr(rl_trainer.PPOTrainer, "_create_train_engine", factory)
    from areal.elastic.infra.adapter import ReconfiguratorAdapter

    adapter = ReconfiguratorAdapter() if enable_control else None
    trainer = rl_trainer.PPOTrainer(config, elastic_reconfigurator=adapter)
    try:
        assert trainer.actor is factory.return_value
        factory.assert_called_once()
        trainer.actor.initialize.assert_called_once()
        trainer.actor.connect_engine.assert_called_once_with(
            serving, trainer.weight_update_meta
        )
        native_recover.return_value.load.assert_called_once()
        assert native_recover.return_value.load.call_args.args[0] is trainer.actor
        assert trainer.scheduler is not None and trainer.rollout_scheduler is not None
        assert trainer._reconfiguration.trainer is trainer
        assert not (Path(config.elastic.state_root) / "inbox").exists()
        if enable_control:
            assert adapter.owner is trainer._reconfiguration
            assert serving.configure_control_callbacks.call_count == 1
    finally:
        trainer.close()

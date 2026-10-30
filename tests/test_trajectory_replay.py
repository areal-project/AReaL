# SPDX-License-Identifier: Apache-2.0

"""Focused snapshot and trainer lifecycle coverage; no model/GPU is needed."""

from unittest.mock import MagicMock

import pytest
import torch

from areal.api.cli_args import (
    GRPOConfig,
    InferenceEngineConfig,
    PPOActorConfig,
    TrainDatasetConfig,
    TrajectoryDebugConfig,
)
from areal.infra.rpc import rtensor
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.trainer import rl_trainer
from areal.trainer.rl_trainer import PPOTrainer
from areal.utils.data import RolloutGroup
from areal.utils.trajectory import TrajectoryStore


@pytest.fixture
def batch():
    image = {"pixel_values": torch.ones(2, 3, dtype=torch.bfloat16)}
    return [
        {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 4]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 1, 1], [0, 1, 1]]),
            "logprobs": torch.full((2, 3), -0.5),
            "versions": torch.zeros(2, 3, dtype=torch.int32),
            "rewards": torch.tensor([0.0, 1.0]),
            "rollout_group": RolloutGroup((1, 1), (0.0, 1.0)),
            "multi_modal_input": [image, image],
        }
    ]


def test_snapshot_round_trip_preserves_training_fields_and_image_aliases(
    tmp_path, batch
):
    """Every replay needs values, group metadata and nested multimodal payloads."""
    store = TrajectoryStore(str(tmp_path), scope="rollout", identity={})
    store.save(batch, 0)

    loaded = store.load(0)[0]
    for key in (
        "input_ids",
        "attention_mask",
        "loss_mask",
        "logprobs",
        "versions",
        "rewards",
    ):
        torch.testing.assert_close(loaded[key], batch[0][key], rtol=0, atol=0)
    assert loaded["rollout_group"] == batch[0]["rollout_group"]
    assert loaded["multi_modal_input"][0] is loaded["multi_modal_input"][1]
    torch.testing.assert_close(
        loaded["multi_modal_input"][0]["pixel_values"],
        batch[0]["multi_modal_input"][0]["pixel_values"],
        rtol=0,
        atol=0,
    )


def test_snapshot_remote_references_remain_live_and_fetch_buffer_drains(
    tmp_path, monkeypatch, batch
):
    """Controller capture must not replace the live training transport graph."""
    original = batch[0]["logprobs"]
    remote = RTensor(
        TensorShardInfo(shard_id="logprobs", node_addr="test.invalid:1234"),
        original.to("meta"),
    )
    batch[0]["logprobs"] = remote
    backend = MagicMock()
    backend.fetch.return_value = [original]
    monkeypatch.setattr(rtensor, "get_backend", lambda: backend)
    buffer = {"unrelated": torch.tensor(1)}
    monkeypatch.setattr(rtensor, "_fetch_buffer", buffer)
    store = TrajectoryStore(str(tmp_path), scope="rollout", identity={})

    store.save(batch, 0)

    assert batch[0]["logprobs"] is remote
    assert remote.data.is_meta
    backend.fetch.assert_called_once()
    assert set(buffer) == {"unrelated"}
    torch.testing.assert_close(store.load(0)[0]["logprobs"], original, rtol=0, atol=0)


def test_snapshot_published_step_cannot_be_overwritten(tmp_path, batch):
    """Reusing a capture directory must preserve the original debugging input."""
    store = TrajectoryStore(str(tmp_path), scope="rollout", identity={})
    store.save(batch, 0)
    with pytest.raises(FileExistsError):
        store.save(batch, 0)
    assert [p.name for p in tmp_path.iterdir()] == ["step_000000.pt"]


@pytest.mark.parametrize("mismatch", ["scope", "identity", "batch_size"])
def test_replay_manifest_rejects_incompatible_capture(tmp_path, mismatch):
    """A different topology or capture boundary must not silently train."""
    writer = TrajectoryStore(str(tmp_path), scope="full", identity={"actor": "d2"})
    writer.initialize_dump(steps_per_epoch=2, batch_size=2)
    reader = TrajectoryStore(
        str(tmp_path),
        scope="rollout" if mismatch == "scope" else "full",
        identity={"actor": "d1" if mismatch == "identity" else "d2"},
    )
    with pytest.raises(ValueError, match=mismatch):
        reader.replay_manifest(batch_size=1 if mismatch == "batch_size" else 2)


def test_replay_missing_step_fails_before_training(tmp_path):
    """Selected capture steps must not be silently reused or cycled."""
    store = TrajectoryStore(str(tmp_path), scope="rollout", identity={})
    with pytest.raises(FileNotFoundError, match="step 0"):
        store.validate_steps(0, 1)


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("scope", ["rollout", "full"])
def test_replay_trainer_updates_without_creating_inference_services(
    tmp_path, monkeypatch, batch, version, scope
):
    """Drive the real trainer constructor and complete step through training-only seams."""
    config = GRPOConfig(
        experiment_name="trajectory",
        trial_name="test",
        actor=PPOActorConfig(
            backend="megatron:d1", path="test-model", _version=version
        ),
        rollout=InferenceEngineConfig(backend="sglang:d1", _version=version),
        train_dataset=TrainDatasetConfig(batch_size=1),
        trajectory_debug=TrajectoryDebugConfig(
            mode="replay", path=str(tmp_path), scope=scope
        ),
        total_train_steps=1,
    )
    config.gconfig.reward_normalization = False
    config.cluster.fileroot = str(tmp_path / "output")
    if config.actor.should_compute_prox_logp():
        batch[0]["prox_logp"] = torch.zeros_like(batch[0]["logprobs"])
    skeleton = PPOTrainer.__new__(PPOTrainer)
    skeleton.config = config
    # Create the dump with the same public identity contract, then exercise the
    # constructor separately. No inference-controller interface is faked.
    config.trajectory_debug.mode = "dump"
    store = skeleton._create_trajectory_store()
    store.initialize_dump(steps_per_epoch=1, batch_size=1)
    store.save(batch, 0)
    config.trajectory_debug.mode = "replay"

    actor = MagicMock()
    actor.parallel_strategy.dp_size = 1
    actor.compute_logp.return_value = [torch.zeros_like(batch[0]["logprobs"])]
    actor.compute_advantages.side_effect = lambda data, **kwargs: data
    actor.export_stats.return_value = {}
    monkeypatch.setattr(PPOTrainer, "_create_train_engine", lambda *args: actor)
    monkeypatch.setattr(PPOTrainer, "_init_scheduler", lambda *args: MagicMock())
    for name in (
        "_apply_dte_config_envvars",
        "_amend_xccl_weight_update_envvar",
        "_amend_deterministic_envvar",
    ):
        monkeypatch.setattr(PPOTrainer, name, lambda *args: None)
    monkeypatch.setattr(
        rl_trainer, "load_hf_processor_and_tokenizer", lambda *args: (None, MagicMock())
    )
    monkeypatch.setattr(rl_trainer.logging, "setup_file_logging", lambda *args: None)
    monkeypatch.setattr(rl_trainer, "StatsLogger", MagicMock())
    saver = MagicMock()
    saver.is_async = False
    monkeypatch.setattr(rl_trainer, "Saver", MagicMock(return_value=saver))
    recovery = MagicMock()
    recovery.load.return_value = None
    monkeypatch.setattr(rl_trainer, "RecoverHandler", MagicMock(return_value=recovery))
    forbidden = MagicMock(
        side_effect=AssertionError("Replay attempted inference lifecycle")
    )
    monkeypatch.setattr(PPOTrainer, "_initialize_rollout_services", forbidden)
    monkeypatch.setattr(PPOTrainer, "_ensure_proxy_started", forbidden)

    with PPOTrainer(config) as trainer:
        trainer.train()

    forbidden.assert_not_called()
    actor.connect_engine.assert_not_called()
    actor.prepare_batch.assert_not_called()
    actor.update_weights.assert_not_called()
    actor.ppo_update.assert_called_once()
    actor.set_version.assert_called_once_with(1)
    assert actor.compute_logp.call_count == int(
        scope == "rollout" and config.actor.should_compute_prox_logp()
    )
    assert recovery.load.call_args.kwargs["inference_engine"] is None


def test_trajectory_disabled_does_not_construct_store(monkeypatch):
    """All existing callers keep the default path and perform no snapshot IO."""
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = MagicMock(trajectory_debug=TrajectoryDebugConfig())
    store = MagicMock(side_effect=AssertionError("Disabled mode attempted snapshot IO"))
    monkeypatch.setattr(rl_trainer, "TrajectoryStore", store)
    assert trainer._create_trajectory_store() is None
    store.assert_not_called()

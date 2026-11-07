from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from areal.infra.data_service.controller.config import DataServiceConfig
from areal.utils.data import cycle_dataloader


class _SamplerWithSetEpoch:
    def __init__(self):
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class _FiniteDataloader:
    def __init__(self, batches_per_epoch: int = 2, with_sampler: bool = True):
        self.batches_per_epoch = batches_per_epoch
        if with_sampler:
            self.sampler = _SamplerWithSetEpoch()

    def __iter__(self):
        for i in range(self.batches_per_epoch):
            yield {"batch": i}


class _DataloaderWithSamplerNoSetEpoch:
    class _Sampler:
        pass

    def __init__(self):
        self.sampler = self._Sampler()

    def __iter__(self):
        yield {"batch": 0}


class TestCycleDataloader:
    def test_cycle_yields_correct_batches(self):
        dl = _FiniteDataloader(batches_per_epoch=3)
        gen = cycle_dataloader(dl, num_cycles=2)
        all_batches = list(gen)
        assert len(all_batches) == 6

    def test_cycle_calls_set_epoch(self):
        dl = _FiniteDataloader(batches_per_epoch=2)
        gen = cycle_dataloader(dl, num_cycles=3)
        list(gen)
        assert dl.sampler.epochs == [0, 1, 2]

    def test_cycle_no_sampler_does_not_crash(self):
        dl = _FiniteDataloader(batches_per_epoch=2, with_sampler=False)
        gen = cycle_dataloader(dl, num_cycles=1)
        all_batches = list(gen)
        assert len(all_batches) == 2

    def test_cycle_sampler_without_set_epoch(self):
        dl = _DataloaderWithSamplerNoSetEpoch()
        gen = cycle_dataloader(dl, num_cycles=1)
        all_batches = list(gen)
        assert len(all_batches) == 1

    def test_infinite_cycle_generates_beyond_one_epoch(self):
        dl = _FiniteDataloader(batches_per_epoch=2)
        gen = cycle_dataloader(dl)
        collected = []
        for i, batch in enumerate(gen):
            collected.append(batch)
            if i >= 5:
                break
        assert len(collected) == 6
        assert dl.sampler.epochs == [0, 1, 2]


class TestDataServiceConfig:
    def test_from_dataset_config_defaults(self):
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(path="dummy", type="sft")
        svc_cfg = DataServiceConfig.from_dataset_config(ds_cfg)

        assert svc_cfg.num_workers >= 1
        assert svc_cfg.num_workers == 1
        assert svc_cfg.dataloader_num_workers == 0
        assert svc_cfg.scheduling_strategy.type == "separation"

    def test_from_dataset_config_custom_workers(self):
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(
            path="dummy",
            type="sft",
            num_workers=4,
            num_dataset_workers=3,
        )
        svc_cfg = DataServiceConfig.from_dataset_config(ds_cfg)
        assert svc_cfg.num_workers == 3
        assert svc_cfg.dataloader_num_workers == 4

    def test_from_dataset_config_zero_workers_defaults_to_one(self):
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(path="dummy", type="sft", num_workers=0)
        svc_cfg = DataServiceConfig.from_dataset_config(ds_cfg)
        assert svc_cfg.num_workers == 1
        assert svc_cfg.dataloader_num_workers == 0

    def test_from_dataset_config_allows_disabling_data_service(self):
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(path="dummy", type="sft", scheduling_spec=None)
        svc_cfg = DataServiceConfig.from_dataset_config(ds_cfg)
        assert svc_cfg.scheduling_spec is None

    def test_dataset_config_has_scheduling_spec(self):
        from areal.api.cli_args import TrainDatasetConfig

        ds_cfg = TrainDatasetConfig(path="dummy", type="sft")
        assert ds_cfg.scheduling_spec is not None


class TestTrainerDataServicePath:
    def test_data_controller_importable(self):
        from areal.infra.data_service import DataController, RDataset

        assert DataController is not None
        assert RDataset is not None

    def test_data_controller_config_importable(self):
        assert DataServiceConfig is not None

    def test_rdataset_has_required_protocol(self):
        from areal.infra.data_service import RDataset

        dataset = RDataset(path="dummy", type="rl", split="train")

        assert hasattr(dataset, "connect")
        assert hasattr(dataset, "close")
        assert hasattr(dataset, "__len__")
        assert hasattr(dataset, "__getitem__")
        assert dataset.connected is False

    def test_get_custom_dataset_respects_scheduling_spec_none(self, monkeypatch):
        from areal.api.cli_args import TrainDatasetConfig
        from areal.dataset import get_custom_dataset

        sentinel = object()

        def _fake_custom_dataset(**_kwargs):
            return sentinel

        monkeypatch.setenv("AREAL_SPMD_MODE", "0")
        monkeypatch.setattr("areal.dataset._get_custom_dataset", _fake_custom_dataset)

        cfg = TrainDatasetConfig(path="dummy", type="sft", scheduling_spec=None)
        dataset = get_custom_dataset(split="train", dataset_config=cfg)

        assert dataset is sentinel


class TestGenericDatasetFallback:
    def test_none_split_uses_first_available_split(self, tmp_path: Path):
        from datasets import Dataset, DatasetDict

        from areal.dataset import _get_custom_dataset

        dataset_path = tmp_path / "dict-ds-none-split"
        dataset = DatasetDict({"train": Dataset.from_dict({"x": [1, 2]})})
        dataset.save_to_disk(str(dataset_path))

        loaded = _get_custom_dataset(
            path=str(dataset_path),
            type="sft",
            split=None,
        )
        assert len(loaded) == 2

    def test_explicit_missing_split_raises_error(self, tmp_path: Path):
        from datasets import Dataset, DatasetDict

        from areal.dataset import _get_custom_dataset

        dataset_path = tmp_path / "dict-ds"
        dataset = DatasetDict({"train": Dataset.from_dict({"x": [1, 2]})})
        dataset.save_to_disk(str(dataset_path))

        with pytest.raises(ValueError, match="Requested split 'test' not found"):
            _get_custom_dataset(
                path=str(dataset_path),
                type="sft",
                split="test",
            )


class TestEmptyDataLoaderCompat:
    def test_empty_dataloader_still_works(self):
        from areal.trainer.rl_trainer import _EmptyDataLoader

        dataloader = _EmptyDataLoader(batch_size=2, steps_per_epoch=3)

        assert len(dataloader) == 3
        assert dataloader.batch_size == 2

        batches = []
        for batch in dataloader:
            batches.append(batch)
            if len(batches) >= 3:
                break
        assert len(batches) == 3

    def test_empty_dataloader_state_dict(self):
        from areal.trainer.rl_trainer import _EmptyDataLoader

        dataloader = _EmptyDataLoader()
        assert dataloader.state_dict() == {}
        dataloader.load_state_dict({"some": "state"})


class TestTrainerSignature:
    def test_elastic_config_roundtrips_through_structured_yaml(self, tmp_path):
        from omegaconf import OmegaConf

        from areal.api.cli_args import ElasticConfig, PPOConfig

        default = OmegaConf.to_object(OmegaConf.structured(PPOConfig).elastic)
        assert default.enabled is False and default.reconfigurator == "reference"
        config = ElasticConfig(
            enabled=True,
            state_root=str(tmp_path),
            actor_devices=(0, 1),
            rollout_devices=(2,),
            checkpoint_estimated_bytes=1024,
        )
        parsed = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(ElasticConfig),
                OmegaConf.create(OmegaConf.to_yaml(config)),
            )
        )
        assert OmegaConf.structured(parsed) == OmegaConf.structured(config)
        selected = OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(ElasticConfig),
                {"reconfigurator": "basic"},
            )
        )
        assert selected.reconfigurator == "basic"

    def test_cpu_coordinator_assigns_exact_worker_cuda_masks(
        self, tmp_path, monkeypatch
    ):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from areal.api import Job
        from areal.api.cli_args import SchedulingSpec
        from areal.infra.scheduler import local

        monkeypatch.setattr(
            local, "current_platform", SimpleNamespace(device_control_env_var="")
        )
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        monkeypatch.setattr(local, "gethostip", lambda: "127.0.0.1")
        monkeypatch.setattr(
            local,
            "find_free_ports",
            lambda count, exclude_ports: [
                port for port in range(19000, 19100) if port not in exclude_ports
            ][:count],
        )
        monkeypatch.setattr(local.time, "sleep", lambda value: None)
        process = MagicMock(pid=99999999)
        process.poll.return_value = None
        spawn = MagicMock(return_value=process)
        monkeypatch.setattr(local, "run_with_streaming_logs", spawn)
        scheduler = local.LocalScheduler(
            gpu_devices=[3, 4],
            log_dir=str(tmp_path),
            nfs_record_root=str(tmp_path),
            worker_device_control_env_var="CUDA_VISIBLE_DEVICES",
        )
        try:
            scheduler.create_workers(
                Job(
                    role="actor",
                    replicas=2,
                    tasks=[
                        SchedulingSpec(
                            gpu=1,
                            cmd="python -m protocol_guard",
                            env_vars={"CUDA_VISIBLE_DEVICES": "incorrect"},
                        ),
                    ],
                )
            )
            scheduler.create_workers(
                Job(
                    role="cpu",
                    replicas=1,
                    tasks=[
                        SchedulingSpec(gpu=0, cmd="python -m protocol_guard"),
                    ],
                )
            )
            assert [
                call.kwargs["env_vars_in_cmd"]["CUDA_VISIBLE_DEVICES"]
                for call in spawn.call_args_list
            ] == ["3", "4", ""]
            assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

            # A failure after role registration must not block the next Actor.
            scheduler.exp_config = object()
            configure = MagicMock(side_effect=RuntimeError("configure failed"))
            monkeypatch.setattr(scheduler, "_configure_worker", configure)
            cleanup = MagicMock()
            monkeypatch.setattr(scheduler, "_cleanup_workers", cleanup)
            job = Job(
                role="next-actor",
                replicas=1,
                tasks=[
                    SchedulingSpec(gpu=1, cmd="python -m protocol_guard"),
                ],
            )
            with pytest.raises(local.WorkerCreationError, match="configure failed"):
                scheduler.create_workers(job)
            assert "next-actor" not in scheduler._workers
            cleanup.assert_called_once()
            configure.side_effect = None
            assert scheduler.create_workers(job) == ["next-actor/0"]
        finally:
            # No actual process or port was created by this protocol test.
            scheduler._workers.clear()

    @pytest.mark.parametrize("exited", [False, True])
    def test_guard_readiness_is_bounded_and_detects_process_exit(
        self, tmp_path, monkeypatch, exited
    ):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from areal.infra.scheduler import local

        scheduler = local.LocalScheduler(
            gpu_devices=[0], log_dir=str(tmp_path), nfs_record_root=str(tmp_path)
        )
        worker = SimpleNamespace(
            worker=SimpleNamespace(id="actor/0"),
            process=MagicMock(returncode=23),
            log_file="unused",
        )
        worker.process.poll.return_value = 23 if exited else None
        monkeypatch.setattr(scheduler, "_is_worker_ready", lambda value: False)
        monkeypatch.setattr(scheduler, "_read_log_tail", lambda path: "failed guard")
        clock = iter((0, 31))
        monkeypatch.setattr(local.time, "monotonic", lambda: next(clock))
        expected = local.WorkerFailedError if exited else local.WorkerConfigurationError
        with pytest.raises(expected):
            scheduler._configure_worker(worker, 0)

    def test_native_ppo_publishes_before_saving_and_finishes_after_cpu_hooks(
        self, monkeypatch
    ):
        from unittest.mock import MagicMock

        from areal.api import WeightUpdateMeta
        from areal.api.cli_args import PPOConfig
        from areal.trainer.rl_trainer import PPOTrainer

        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.config = PPOConfig()
        trainer._should_offload_rollout = trainer._should_offload_actor = False
        trainer.critic = trainer.ref = trainer.teacher = trainer.eval_rollout = None
        trainer.data_controller = None
        trainer.train_dataloader = object()
        trainer.weight_update_meta = WeightUpdateMeta.from_awex()
        events = []
        actor = trainer.actor = MagicMock()
        rollout = trainer.rollout = MagicMock()
        batch, adv = [{"tokens": "durable"}], [{"advantages": "native"}]
        actor.prepare_batch.return_value = batch
        actor.compute_advantages.side_effect = (
            lambda value: events.append("advantage") or adv
        )
        actor.ppo_update.side_effect = lambda value: events.append("ppo")
        actor.step_lr_scheduler.side_effect = lambda: events.append("lr")
        actor.update_weights.side_effect = lambda value: events.append("publish")
        trainer.config.actor.recompute_logprob = False
        trainer.config.actor.use_decoupled_loss = False
        trainer.saver = MagicMock()
        # Legacy batch/publication coordination must not be used by Trainer.
        trainer._save_hf = MagicMock(side_effect=lambda **kwargs: events.append("hf"))
        trainer._save_training_state = MagicMock(
            side_effect=lambda **kwargs: events.append("resident-save")
        )
        trainer._evaluate = MagicMock(
            side_effect=lambda **kwargs: events.append("eval")
        )
        trainer._export_and_commit_stats = MagicMock(
            side_effect=lambda **kwargs: events.append("stats")
        )
        trainer._save_perf_tracer = MagicMock(
            side_effect=lambda **kwargs: events.append("trace")
        )
        monkeypatch.setenv("AREAL_SPMD_MODE", "0")
        trainer._train_step(
            global_step=0,
            steps_per_epoch=2,
            workflow=None,
            workflow_kwargs=None,
            dynamic_filter_fn=None,
            eval_workflow=None,
            eval_workflow_kwargs=None,
        )
        actor.compute_advantages.assert_called_once_with(batch)
        actor.ppo_update.assert_called_once_with(adv)
        assert events[:3] == ["advantage", "ppo", "lr"]
        assert events[3:] == ["publish", "resident-save", "eval", "stats", "trace"]
        rollout.pause.assert_called_once()
        rollout.resume.assert_called_once()

    def test_partial_initialization_cleanup_survives_missing_cpu_helpers(
        self, monkeypatch
    ):
        from unittest.mock import MagicMock

        from areal.trainer.rl_trainer import PPOTrainer

        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.actor = MagicMock()
        trainer.rollout = MagicMock()
        monkeypatch.setattr("areal.trainer.rl_trainer.perf_tracer.save", MagicMock())
        trainer.close()
        trainer.actor.destroy.assert_called_once()
        trainer.rollout.destroy.assert_called_once()

    def test_sft_trainer_accepts_dataset_params(self):
        from areal.trainer.sft_trainer import SFTTrainer

        sig = inspect.signature(SFTTrainer.__init__)
        params = list(sig.parameters.keys())
        assert "train_dataset" in params
        assert "valid_dataset" in params
        assert "config" in params

    def test_rw_trainer_accepts_dataset_params(self):
        from areal.trainer.rw_trainer import RWTrainer

        sig = inspect.signature(RWTrainer.__init__)
        params = list(sig.parameters.keys())
        assert "train_dataset" in params
        assert "valid_dataset" in params
        assert "config" in params

    def test_ppo_trainer_accepts_dataset_params(self):
        from areal.trainer.rl_trainer import PPOTrainer

        sig = inspect.signature(PPOTrainer.__init__)
        params = list(sig.parameters.keys())
        assert "train_dataset" in params
        assert "valid_dataset" in params
        assert "config" in params

    def test_ppo_close_disconnects_actor_before_destroying_rollout(self, monkeypatch):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from areal.trainer.rl_trainer import PPOTrainer

        calls = []
        actor, rollout = MagicMock(), MagicMock()
        actor.destroy.side_effect = lambda: calls.append("actor")

        def destroy_rollout():
            assert calls == ["actor"]
            calls.append("rollout")

        rollout.destroy.side_effect = destroy_rollout
        trainer = SimpleNamespace(
            saver=MagicMock(),
            stats_logger=MagicMock(),
            actor=actor,
            rollout=rollout,
            teacher=None,
            ref=None,
            critic=None,
            eval_rollout=None,
        )
        monkeypatch.setattr("areal.trainer.rl_trainer.perf_tracer.save", MagicMock())
        PPOTrainer.close(trainer)
        assert calls == ["actor", "rollout"]

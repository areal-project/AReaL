from __future__ import annotations

import inspect
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

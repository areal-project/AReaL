from types import SimpleNamespace
from unittest.mock import MagicMock, sentinel

from areal.trainer import rl_trainer
from areal.trainer.rl_trainer import PPOTrainer


def _bare_trainer() -> PPOTrainer:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.config = SimpleNamespace(
        seed=17,
        experiment_name="online-eval",
        trial_name="trial-0",
        tokenizer_path="tokenizer-path",
    )
    trainer.scheduler = sentinel.scheduler
    trainer.data_controller = None
    trainer._train_rdataset = None
    trainer._valid_rdataset = None
    return trainer


class TestOnlineEvalDataController:
    def test_data_controller_is_created_from_valid_config(self, monkeypatch):
        trainer = _bare_trainer()
        valid_config = SimpleNamespace(shuffle=False, drop_last=True)
        service_config = SimpleNamespace(num_workers=3)
        dataset = MagicMock()
        controller = MagicMock()
        controller_factory = MagicMock(return_value=controller)
        from_dataset_config = MagicMock(return_value=service_config)
        monkeypatch.setattr(rl_trainer, "DataController", controller_factory)
        monkeypatch.setattr(
            rl_trainer.DataServiceConfig,
            "from_dataset_config",
            from_dataset_config,
        )

        trainer._connect_rdataset(dataset, valid_config, split="valid")

        from_dataset_config.assert_called_once_with(valid_config, seed=17)
        controller_factory.assert_called_once_with(service_config, sentinel.scheduler)
        controller.initialize.assert_called_once_with(
            role="data", num_dataset_workers=3
        )
        dataset.connect.assert_called_once_with(
            controller,
            dataset_id="online-eval_trial-0_valid",
            tokenizer_or_processor_path="tokenizer-path",
            shuffle=False,
            drop_last=True,
        )
        assert trainer.data_controller is controller
        assert trainer._valid_rdataset is dataset

    def test_data_controller_is_reused_for_train_config(self, monkeypatch):
        trainer = _bare_trainer()
        existing_controller = MagicMock()
        trainer.data_controller = existing_controller
        train_config = SimpleNamespace(shuffle=True, drop_last=False)
        dataset = MagicMock()
        controller_factory = MagicMock()
        from_dataset_config = MagicMock()
        monkeypatch.setattr(rl_trainer, "DataController", controller_factory)
        monkeypatch.setattr(
            rl_trainer.DataServiceConfig,
            "from_dataset_config",
            from_dataset_config,
        )

        trainer._connect_rdataset(dataset, train_config, split="train")

        controller_factory.assert_not_called()
        from_dataset_config.assert_not_called()
        existing_controller.initialize.assert_not_called()
        dataset.connect.assert_called_once_with(
            existing_controller,
            dataset_id="online-eval_trial-0_train",
            tokenizer_or_processor_path="tokenizer-path",
            shuffle=True,
            drop_last=False,
        )
        assert trainer.data_controller is existing_controller
        assert trainer._train_rdataset is dataset

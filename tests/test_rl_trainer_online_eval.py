from types import SimpleNamespace
from unittest.mock import MagicMock, sentinel

import pytest

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


def _eval_trainer(*, online_mode: bool, wait_results: list[object]) -> PPOTrainer:
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer._online_mode = online_mode
    trainer.config = SimpleNamespace(eval_gconfig=SimpleNamespace(n_samples=1))
    trainer.actor = MagicMock()
    trainer.actor.is_data_parallel_head.return_value = True
    trainer.valid_dataloader = [[{"question": "one"}, {"question": "two"}]]
    trainer.eval_rollout = MagicMock()
    trainer.eval_rollout.wait.return_value = wait_results
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


class TestEvalRolloutTopology:
    def test_eval_rollout_topology_online_without_valid_data_is_disabled(self):
        assert not PPOTrainer._should_initialize_eval_rollout(
            online_mode=True,
            has_valid_dataloader=False,
        )

    def test_eval_rollout_topology_online_with_valid_data_is_enabled(self):
        assert PPOTrainer._should_initialize_eval_rollout(
            online_mode=True,
            has_valid_dataloader=True,
        )

    def test_eval_rollout_topology_offline_remains_enabled(self):
        assert PPOTrainer._should_initialize_eval_rollout(
            online_mode=False,
            has_valid_dataloader=False,
        )


class TestOnlineEvalIntegrity:
    def test_online_validation_requires_eval_workflow_before_training_side_effects(
        self,
    ):
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.config = SimpleNamespace(
            total_train_epochs=1,
            total_train_steps=1,
            rollout=SimpleNamespace(agent=SimpleNamespace(mode="online")),
            gconfig=SimpleNamespace(n_samples=1),
            dynamic_bs=False,
        )
        trainer._online_mode = True
        trainer.valid_dataloader = [[{"question": "held-out"}]]
        trainer.train_dataloader = [[{}]]
        trainer.recover_info = None
        trainer._should_offload_rollout = False
        trainer._ensure_proxy_started = MagicMock()
        trainer.actor = MagicMock()
        trainer.actor.prepare_batch.side_effect = AssertionError(
            "training batch must not be consumed before eval preflight"
        )

        with pytest.raises(ValueError, match="eval_workflow"):
            trainer.train(workflow=None, eval_workflow=None)

        trainer._ensure_proxy_started.assert_not_called()
        trainer.actor.prepare_batch.assert_not_called()

    def test_incomplete_eval_reports_rejected_and_expected_counts(self, monkeypatch):
        monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)
        trainer = _eval_trainer(
            online_mode=True,
            wait_results=[sentinel.accepted_trajectory, None],
        )

        with pytest.raises(
            RuntimeError,
            match=r"rejected=1, expected=2",
        ):
            trainer._evaluate_fn(
                eval_workflow=sentinel.workflow, eval_workflow_kwargs={}
            )

    def test_all_valid_online_eval_results_complete_normally(self, monkeypatch):
        monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)
        trainer = _eval_trainer(
            online_mode=True,
            wait_results=[sentinel.trajectory_one, sentinel.trajectory_two],
        )

        result = trainer._evaluate_fn(
            eval_workflow=sentinel.workflow,
            eval_workflow_kwargs={},
        )

        assert result is None

    def test_offline_incomplete_eval_preserves_current_behavior(self, monkeypatch):
        monkeypatch.setattr(rl_trainer, "is_single_controller", lambda: True)
        trainer = _eval_trainer(
            online_mode=False,
            wait_results=[sentinel.accepted_trajectory, None],
        )

        result = trainer._evaluate_fn(
            eval_workflow=sentinel.workflow,
            eval_workflow_kwargs={},
        )

        assert result is None

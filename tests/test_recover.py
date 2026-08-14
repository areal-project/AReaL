"""Tests for the recovery configuration and functionality."""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from areal.api.cli_args import RecoverConfig
from areal.api.io_struct import FinetuneSpec, StepInfo
from areal.utils.recover import (
    RecoverHandler,
    RecoverInfo,
    check_if_auto_recover,
    check_if_recover,
)
from areal.v2.training_service.controller.controller import (
    GatewayTrainController,
)


class TestRecoverConfig:
    """Tests for RecoverConfig dataclass validation."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot="/tmp",
        )
        assert config.mode == "disabled"
        assert config.retries == 3

    @pytest.mark.parametrize("mode", ["on", "off", "auto", "disabled"])
    def test_valid_modes(self, mode):
        """Test that all valid modes are accepted."""
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot="/tmp",
            mode=mode,
        )
        assert config.mode == mode

    @pytest.mark.parametrize("mode", ["fault", "resume", "invalid", "ON", "OFF", ""])
    def test_invalid_modes(self, mode):
        """Test that invalid modes raise ValueError with helpful message."""
        with pytest.raises(ValueError) as exc_info:
            RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot="/tmp",
                mode=mode,
            )
        error_msg = str(exc_info.value)
        assert f"Invalid recover mode '{mode}'" in error_msg
        assert "fault" in error_msg and "resume" in error_msg  # Migration hint


class TestCheckIfRecover:
    """Tests for the check_if_recover function."""

    @pytest.mark.parametrize("mode", ["disabled", "off"])
    def test_disabled_modes_return_false(self, mode):
        """Test that disabled modes always return False."""
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot="/tmp",
            mode=mode,
        )
        # Should return False regardless of run_id
        assert check_if_recover(config, 0) is False
        assert check_if_recover(config, 1) is False
        assert check_if_recover(config, 10) is False

    @pytest.mark.parametrize("mode", ["on", "auto"])
    def test_enabled_modes_check_for_checkpoint(self, mode):
        """Test that enabled modes check for existing checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode=mode,
            )
            # No checkpoint exists, should return False
            assert check_if_recover(config, 0) is False

    @pytest.mark.parametrize("run_id", [0, 1, 5, 100])
    def test_run_id_parameter_unused(self, run_id):
        """Test that run_id parameter doesn't affect the result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with disabled mode
            config_disabled = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="disabled",
            )
            assert check_if_recover(config_disabled, run_id) is False

            # Test with enabled mode (no checkpoint)
            config_enabled = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="on",
            )
            # Result should be the same regardless of run_id
            result = check_if_recover(config_enabled, run_id)
            assert result == check_if_recover(config_enabled, 0)


class TestCheckIfAutoRecover:
    """Tests for the check_if_auto_recover function."""

    def test_no_checkpoint_returns_false(self):
        """Test that missing checkpoint returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="on",
            )
            assert check_if_auto_recover(config) is False

    def test_empty_directory_returns_false(self):
        """Test that empty directory (no checkpoint) returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="on",
            )
            assert check_if_auto_recover(config) is False


class TestModeEquivalence:
    """Tests to verify mode equivalences (on=auto, off=disabled)."""

    def test_on_equals_auto(self):
        """Test that 'on' and 'auto' modes behave identically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_on = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="on",
            )
            config_auto = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="auto",
            )
            # Both should return the same result
            assert check_if_recover(config_on, 0) == check_if_recover(config_auto, 0)

    def test_off_equals_disabled(self):
        """Test that 'off' and 'disabled' modes behave identically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_off = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="off",
            )
            config_disabled = RecoverConfig(
                experiment_name="test_exp",
                trial_name="test_trial",
                fileroot=tmpdir,
                mode="disabled",
            )
            # Both should return False
            assert check_if_recover(config_off, 0) is False
            assert check_if_recover(config_disabled, 0) is False


class TestRecoverHandler:
    @staticmethod
    def _make_handler(tmpdir: str, mode: str) -> RecoverHandler:
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot=tmpdir,
            mode=mode,
        )
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=8,
            train_batch_size=2,
        )
        return RecoverHandler(config, ft_spec)

    @staticmethod
    def _make_gateway_controller() -> GatewayTrainController:
        return GatewayTrainController.__new__(GatewayTrainController)

    @pytest.mark.parametrize("mode", ["on", "auto"])
    def test_load_accepts_gateway_train_controller(self, mode):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, mode)
            handler.freq_ctl = Mock()
            controller = self._make_gateway_controller()
            recover_info = SimpleNamespace(
                last_step_info=SimpleNamespace(global_step=0, next=lambda: "step-1"),
                saver_info={},
                evaluator_info={},
                stats_logger_info={},
                dataloader_info={},
                checkpoint_info={},
            )
            saver = Mock()
            evaluator = Mock()
            stats_logger = Mock()
            dataloader = Mock()
            handler._load_checkpoint = Mock()

            with patch(
                "areal.utils.recover.RecoverInfo.load", return_value=recover_info
            ):
                result = handler.load(
                    controller,
                    saver,
                    evaluator,
                    stats_logger,
                    dataloader,
                )

            assert result is recover_info
            handler._load_checkpoint.assert_called_once_with(controller, name="default")
            saver.load_state_dict.assert_called_once_with({})
            dataloader.load_state_dict.assert_called_once_with({})

    @pytest.mark.parametrize("mode", ["on", "auto"])
    def test_dump_accepts_gateway_train_controller(self, mode):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, mode)
            handler.freq_ctl = Mock()
            handler.freq_ctl.check.return_value = True
            handler.freq_ctl.state_dict.return_value = {}
            controller = self._make_gateway_controller()
            step_info = StepInfo(
                epoch=0,
                epoch_step=0,
                global_step=0,
                steps_per_epoch=handler.ft_spec.steps_per_epoch,
            )
            saver = Mock(state_dict=Mock(return_value={}))
            evaluator = Mock(state_dict=Mock(return_value={}))
            stats_logger = Mock(state_dict=Mock(return_value={}))
            dataloader = Mock(state_dict=Mock(return_value={}))
            handler._save_checkpoint = Mock()

            with patch("areal.utils.recover.RecoverInfo.dump") as dump_info:
                handler.dump(
                    controller,
                    step_info,
                    saver,
                    evaluator,
                    stats_logger,
                    dataloader,
                )

            handler._save_checkpoint.assert_called_once_with(
                controller,
                name="default",
                tokenizer=None,
                processor=None,
                base_model_path=None,
            )
            dump_info.assert_called_once()

    def test_dump_captures_inference_pipeline_state_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, "auto")
            handler.freq_ctl = Mock()
            handler.freq_ctl.check.return_value = True
            handler.freq_ctl.state_dict.return_value = {}
            controller = self._make_gateway_controller()
            step_info = StepInfo(
                epoch=0,
                epoch_step=0,
                global_step=0,
                steps_per_epoch=handler.ft_spec.steps_per_epoch,
            )
            saver = Mock(state_dict=Mock(return_value={}))
            evaluator = Mock(state_dict=Mock(return_value={}))
            stats_logger = Mock(state_dict=Mock(return_value={}))
            dataloader = Mock(state_dict=Mock(return_value={"cursor": 1}))
            inference_engine = Mock()
            events = []
            inference_engine.recover_state_dict.side_effect = lambda: (
                events.append("pipeline"),
                {"task_id_generator": {"next_task_id": 4}},
            )[1]
            handler._save_checkpoint = lambda *args, **kwargs: events.append(
                "checkpoint"
            )

            handler.dump(
                controller,
                step_info,
                saver,
                evaluator,
                stats_logger,
                dataloader,
                inference_engine=inference_engine,
            )

            assert events == ["pipeline", "checkpoint"]
            saved_info = RecoverInfo.load(
                handler.recover_info_path(
                    handler.config.experiment_name,
                    handler.config.trial_name,
                    handler.config.fileroot,
                )
            )
            assert saved_info.pipeline_info == {
                "task_id_generator": {"next_task_id": 4}
            }

    def test_load_restores_inference_pipeline_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, "auto")
            handler.freq_ctl = Mock()
            handler._load_checkpoint = Mock()
            controller = self._make_gateway_controller()
            controller.connect_engine = Mock()
            controller.update_weights = Mock()
            controller.set_version = Mock()
            recover_info = SimpleNamespace(
                last_step_info=SimpleNamespace(global_step=0, next=lambda: "step-1"),
                saver_info={},
                evaluator_info={},
                stats_logger_info={},
                dataloader_info={},
                checkpoint_info={},
                pipeline_info={"task_id_generator": {"next_task_id": 4}},
            )
            saver = Mock()
            evaluator = Mock()
            stats_logger = Mock()
            dataloader = Mock()
            inference_engine = Mock()
            weight_update_meta = Mock(type="disk", colocate=False)
            weight_update_meta.with_version.return_value = Mock(version=1)

            with patch(
                "areal.utils.recover.RecoverInfo.load", return_value=recover_info
            ):
                handler.load(
                    controller,
                    saver,
                    evaluator,
                    stats_logger,
                    dataloader,
                    inference_engine=inference_engine,
                    weight_update_meta=weight_update_meta,
                )

            inference_engine.load_recover_state_dict.assert_called_once_with(
                {"task_id_generator": {"next_task_id": 4}}
            )

    def test_failed_weight_sync_does_not_advance_recovery_versions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, "auto")
            handler.freq_ctl = Mock()
            handler._load_checkpoint = Mock()
            controller = self._make_gateway_controller()
            controller.connect_engine = Mock()
            controller.update_weights = Mock(
                side_effect=RuntimeError("weight transfer failed")
            )
            controller.set_version = Mock()
            recover_info = SimpleNamespace(
                last_step_info=SimpleNamespace(global_step=2, next=lambda: "step-3"),
                saver_info={},
                evaluator_info={},
                stats_logger_info={},
                dataloader_info={},
                checkpoint_info={},
                pipeline_info=None,
            )
            inference_engine = Mock()
            weight_update_meta = Mock(type="awex", colocate=False)
            weight_update_meta.with_version.return_value = Mock(version=3)

            with (
                patch(
                    "areal.utils.recover.RecoverInfo.load", return_value=recover_info
                ),
                pytest.raises(RuntimeError, match="weight transfer failed"),
            ):
                handler.load(
                    controller,
                    Mock(),
                    Mock(),
                    Mock(),
                    Mock(),
                    inference_engine=inference_engine,
                    weight_update_meta=weight_update_meta,
                )

            inference_engine.resume.assert_called_once_with()
            controller.set_version.assert_not_called()
            inference_engine.set_version.assert_not_called()

    def test_resume_failure_does_not_mask_failed_weight_sync(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, "auto")
            handler.freq_ctl = Mock()
            handler._load_checkpoint = Mock()
            controller = self._make_gateway_controller()
            controller.connect_engine = Mock()
            controller.update_weights = Mock(
                side_effect=RuntimeError("weight transfer failed")
            )
            controller.set_version = Mock()
            recover_info = SimpleNamespace(
                last_step_info=SimpleNamespace(global_step=2, next=lambda: "step-3"),
                saver_info={},
                evaluator_info={},
                stats_logger_info={},
                dataloader_info={},
                checkpoint_info={},
                pipeline_info=None,
            )
            inference_engine = Mock()
            inference_engine.resume.side_effect = RuntimeError("resume failed")
            weight_update_meta = Mock(type="awex", colocate=False)
            weight_update_meta.with_version.return_value = Mock(version=3)

            with (
                patch(
                    "areal.utils.recover.RecoverInfo.load", return_value=recover_info
                ),
                pytest.raises(RuntimeError, match="weight transfer failed"),
            ):
                handler.load(
                    controller,
                    Mock(),
                    Mock(),
                    Mock(),
                    Mock(),
                    inference_engine=inference_engine,
                    weight_update_meta=weight_update_meta,
                )

            inference_engine.resume.assert_called_once_with()
            controller.set_version.assert_not_called()
            inference_engine.set_version.assert_not_called()


class TestRecoverInfoPipelineState:
    @staticmethod
    def _make_info() -> RecoverInfo:
        return RecoverInfo(
            last_step_info=StepInfo(
                epoch=0,
                epoch_step=2,
                global_step=2,
                steps_per_epoch=4,
            ),
            saver_info={},
            evaluator_info={},
            stats_logger_info={},
            dataloader_info={"cursor": 3},
            checkpoint_info={},
            pipeline_info={"task_id_generator": {"next_task_id": 12}},
        )

    def test_round_trip_preserves_pipeline_state(self):
        """New recover info preserves the rollout task counter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_info().dump(tmpdir)

            loaded = RecoverInfo.load(tmpdir)

            assert loaded.pipeline_info == {"task_id_generator": {"next_task_id": 12}}

    def test_legacy_checkpoint_without_pipeline_file_loads(self):
        """Old recover directories remain loadable without task ID state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_info().dump(tmpdir)
            os.unlink(os.path.join(tmpdir, "pipeline_info.pkl"))

            loaded = RecoverInfo.load(tmpdir)

            assert loaded.pipeline_info is None


class TestAwexColocateGate:
    """The AWEX pre-transfer sequence must run only for colocated rollouts."""

    @staticmethod
    def _awex_meta():
        return Mock(type="awex")

    def test_awex_transport_without_colocation_is_not_colocate(self):
        assert not RecoverHandler._should_run_awex_colocate_transfer(
            inference_engine=Mock(),
            weight_update_meta=self._awex_meta(),
            colocated_rollout=False,
        )

    def test_awex_transport_with_colocation_is_colocate(self):
        assert RecoverHandler._should_run_awex_colocate_transfer(
            inference_engine=Mock(),
            weight_update_meta=self._awex_meta(),
            colocated_rollout=True,
        )

    @pytest.mark.parametrize("meta_type", ["disk", "xccl"])
    def test_non_awex_transport_is_never_colocate(self, meta_type):
        assert not RecoverHandler._should_run_awex_colocate_transfer(
            inference_engine=Mock(),
            weight_update_meta=Mock(type=meta_type),
            colocated_rollout=True,
        )

    def test_missing_inference_engine_is_not_colocate(self):
        assert not RecoverHandler._should_run_awex_colocate_transfer(
            inference_engine=None,
            weight_update_meta=self._awex_meta(),
            colocated_rollout=True,
        )

    def test_meta_without_type_attribute_is_not_colocate(self):
        assert not RecoverHandler._should_run_awex_colocate_transfer(
            inference_engine=Mock(),
            weight_update_meta=None,
            colocated_rollout=True,
        )


class TestColocateRolloutProtocol:
    """Engines lacking the colocate protocol must fail before any side effect."""

    def test_engine_with_full_protocol_is_accepted(self):
        engine = Mock(spec=["pause_generation_sync", "offload"])
        engine.offload = lambda tags=None: None

        RecoverHandler._require_colocate_rollout_protocol(engine)

    def test_engine_without_pause_generation_sync_is_rejected(self):
        engine = Mock(spec=["offload"])
        engine.offload = lambda tags=None: None

        with pytest.raises(NotImplementedError) as exc_info:
            RecoverHandler._require_colocate_rollout_protocol(engine)

        assert "pause_generation_sync" in str(exc_info.value)

    def test_engine_with_untagged_offload_is_rejected(self):
        engine = Mock(spec=["pause_generation_sync", "offload"])
        engine.offload = lambda: None

        with pytest.raises(NotImplementedError) as exc_info:
            RecoverHandler._require_colocate_rollout_protocol(engine)

        assert "tags" in str(exc_info.value)

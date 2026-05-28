"""Tests for the recovery configuration and functionality."""

import tempfile
from unittest.mock import Mock

import pytest

from areal.api.cli_args import RecoverConfig
from areal.api.io_struct import FinetuneSpec, StepInfo
from areal.experimental.training_service.controller.controller import (
    GatewayTrainController,
)
from areal.utils.recover import RecoverHandler, check_if_auto_recover, check_if_recover


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
    def test_load_rejects_gateway_train_controller(self, mode):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, mode)

            with pytest.raises(NotImplementedError) as exc_info:
                handler.load(
                    self._make_gateway_controller(),
                    Mock(),
                    Mock(),
                    Mock(),
                    Mock(),
                )

            assert "GatewayTrainController" in str(exc_info.value)
            assert '`_version="v2"`' in str(exc_info.value)

    @pytest.mark.parametrize("mode", ["on", "auto"])
    def test_dump_rejects_gateway_train_controller(self, mode):
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = self._make_handler(tmpdir, mode)
            step_info = StepInfo(
                epoch=0,
                epoch_step=0,
                global_step=0,
                steps_per_epoch=handler.ft_spec.steps_per_epoch,
            )

            with pytest.raises(NotImplementedError) as exc_info:
                handler.dump(
                    self._make_gateway_controller(),
                    step_info,
                    Mock(),
                    Mock(),
                    Mock(),
                    Mock(),
                )

            assert "GatewayTrainController" in str(exc_info.value)
            assert "recover.mode" in str(exc_info.value)

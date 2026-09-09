"""Tests for Trackio experiment tracking backend integration."""

from dataclasses import fields
from unittest.mock import MagicMock, patch

import torch

from areal.api.cli_args import (
    StatsLoggerConfig,
    TrackioConfig,
)


class TestTrackioConfig:
    """Tests for TrackioConfig dataclass."""

    def test_default_mode_is_disabled(self):
        """TrackioConfig should default to disabled mode."""
        config = TrackioConfig()
        assert config.mode == "disabled"

    def test_default_optional_fields_are_none(self):
        """Optional fields should default to None."""
        config = TrackioConfig()
        assert config.project is None
        assert config.name is None
        assert config.space_id is None
        assert config.max_rollout_traces_per_step == 0

    def test_custom_values(self):
        """TrackioConfig should accept custom values."""
        config = TrackioConfig(
            mode="online",
            project="my-project",
            name="my-run",
            space_id="user/my-space",
            max_rollout_traces_per_step=8,
        )
        assert config.mode == "online"
        assert config.project == "my-project"
        assert config.name == "my-run"
        assert config.space_id == "user/my-space"
        assert config.max_rollout_traces_per_step == 8

    def test_invalid_mode_raises_error(self):
        """TrackioConfig should reject invalid mode values."""
        import pytest

        with pytest.raises(ValueError, match="Invalid trackio mode"):
            TrackioConfig(mode="invalid")

    def test_all_valid_modes_accepted(self):
        """TrackioConfig should accept all valid mode values."""
        for mode in ("disabled", "online", "local"):
            config = TrackioConfig(mode=mode)
            assert config.mode == mode


class TestStatsLoggerConfigTrackio:
    """Tests for Trackio field in StatsLoggerConfig."""

    def test_trackio_field_exists(self):
        """StatsLoggerConfig should have a trackio field."""
        field_names = [f.name for f in fields(StatsLoggerConfig)]
        assert "trackio" in field_names

    def test_trackio_field_default_is_disabled(self):
        """StatsLoggerConfig.trackio should default to disabled TrackioConfig."""
        config = StatsLoggerConfig(
            experiment_name="test_exp",
            trial_name="trial_0",
            fileroot="/tmp/test",
        )
        assert isinstance(config.trackio, TrackioConfig)
        assert config.trackio.mode == "disabled"


def _make_test_config(trackio_config=None):
    """Create a minimal BaseExperimentConfig for testing StatsLogger."""
    from areal.api.cli_args import BaseExperimentConfig

    config = BaseExperimentConfig(
        experiment_name="test_exp",
        trial_name="trial_0",
        total_train_epochs=1,
    )
    config.stats_logger.experiment_name = "test_exp"
    config.stats_logger.trial_name = "trial_0"
    config.stats_logger.fileroot = "/tmp/test"
    if trackio_config is not None:
        config.stats_logger.trackio = trackio_config
    return config


def _make_ft_spec():
    """Create a mock FinetuneSpec for testing."""
    from areal.api import FinetuneSpec

    ft_spec = MagicMock(spec=FinetuneSpec)
    ft_spec.total_train_epochs = 1
    ft_spec.steps_per_epoch = 10
    ft_spec.total_train_steps = 10
    return ft_spec


class TestStatsLoggerTrackioIntegration:
    """Tests for Trackio integration in StatsLogger (mocked)."""

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_init_called_when_enabled(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        """trackio.init() should be called when mode is not disabled."""
        mock_dist.is_initialized.return_value = False

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config(
            TrackioConfig(mode="online", max_rollout_traces_per_step=32)
        )
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.init.assert_called_once()
        assert logger._trackio_enabled is True

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_not_init_when_disabled(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        """trackio.init() should NOT be called when mode is disabled."""
        mock_dist.is_initialized.return_value = False

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config()  # trackio defaults to disabled
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.init.assert_not_called()
        assert logger._trackio_enabled is False

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_log_called_on_commit(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        """trackio.log() should be called during commit when enabled."""
        mock_dist.is_initialized.return_value = False

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config(
            TrackioConfig(mode="online", max_rollout_traces_per_step=32)
        )
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.log.reset_mock()

        data = {"loss/avg": 0.5, "reward/avg": 1.0}
        logger.commit(epoch=0, step=0, global_step=0, data=data)
        mock_trackio.log.assert_called_once_with(data, step=0)

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_finish_called_on_close(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        """trackio.finish() should be called during close when enabled."""
        mock_dist.is_initialized.return_value = False

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config(
            TrackioConfig(mode="online", max_rollout_traces_per_step=32)
        )
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.finish.reset_mock()

        logger.close()
        mock_trackio.finish.assert_called_once()

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_not_logged_when_disabled(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        """trackio.log() should NOT be called during commit when disabled."""
        mock_dist.is_initialized.return_value = False

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config()  # trackio defaults to disabled
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.log.reset_mock()

        data = {"loss/avg": 0.5}
        logger.commit(epoch=0, step=0, global_step=0, data=data)
        mock_trackio.log.assert_not_called()

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_trace_logging_from_rollout_tensors(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        """Rollout tensors should be decoded and logged as trackio.Trace records."""
        mock_dist.is_initialized.return_value = False
        mock_trackio.Trace.side_effect = lambda messages, metadata: {
            "_type": "trackio.trace",
            "messages": messages,
            "metadata": metadata,
        }

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config(
            TrackioConfig(mode="online", max_rollout_traces_per_step=32)
        )
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.log.reset_mock()

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, skip_special_tokens=False: " ".join(
            str(i) for i in ids
        )
        trajectory = {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 0, 1, 1]], dtype=torch.int64),
            "rewards": torch.tensor([0.75], dtype=torch.float32),
            "versions": torch.tensor([[0, 0, 2, 2]], dtype=torch.int64),
        }

        logger.log_rollout_traces(
            [trajectory],
            split="rollout",
            global_step=3,
            tokenizer=tokenizer,
        )

        mock_trackio.Trace.assert_called_once()
        trace_kwargs = mock_trackio.Trace.call_args.kwargs
        assert trace_kwargs["messages"] == [
            {"role": "user", "content": "1 2"},
            {"role": "assistant", "content": "3 4"},
        ]
        assert trace_kwargs["metadata"] == {
            "split": "rollout",
            "global_step": 3,
            "trajectory_index": 0,
            "sample_index": 0,
            "seqlen": 4,
            "prompt_len": 2,
            "reward": 0.75,
            "head_version": 0,
            "tail_version": 2,
        }
        mock_trackio.log.assert_called_once()
        assert mock_trackio.log.call_args.args[0]["rollout/trajectories"] == [
            {
                "_type": "trackio.trace",
                "messages": trace_kwargs["messages"],
                "metadata": trace_kwargs["metadata"],
            }
        ]
        assert mock_trackio.log.call_args.kwargs == {"step": 3}

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_trace_logging_respects_cap(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        """Trace logging should respect max_rollout_traces_per_step."""
        mock_dist.is_initialized.return_value = False
        mock_trackio.Trace.side_effect = lambda messages, metadata: {
            "messages": messages,
            "metadata": metadata,
        }

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config(
            TrackioConfig(mode="online", max_rollout_traces_per_step=1)
        )
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.log.reset_mock()

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, skip_special_tokens=False: " ".join(
            str(i) for i in ids
        )
        trajectory = {
            "input_ids": torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 1], [0, 1]], dtype=torch.int64),
            "rewards": torch.tensor([1.0, 0.0], dtype=torch.float32),
        }

        logger.log_rollout_traces(
            [trajectory],
            split="rollout",
            global_step=0,
            tokenizer=tokenizer,
        )

        assert mock_trackio.Trace.call_count == 1

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_trace_logging_skips_none_trajectories(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        """Rejected rollout trajectories should be skipped."""
        mock_dist.is_initialized.return_value = False
        mock_trackio.Trace.side_effect = lambda messages, metadata: {
            "messages": messages,
            "metadata": metadata,
        }

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config(
            TrackioConfig(mode="online", max_rollout_traces_per_step=32)
        )
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.log.reset_mock()

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, skip_special_tokens=False: " ".join(
            str(i) for i in ids
        )
        trajectory = {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 1]], dtype=torch.int64),
        }

        logger.log_rollout_traces(
            [None, trajectory],
            split="eval-rollout",
            global_step=4,
            tokenizer=tokenizer,
        )

        mock_trackio.Trace.assert_called_once()
        assert mock_trackio.Trace.call_args.kwargs["metadata"]["trajectory_index"] == 1
        mock_trackio.log.assert_called_once()

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_trace_logging_reconstructs_multiturn_tool_messages(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        """Interleaved loss masks should become multi-message traces."""
        mock_dist.is_initialized.return_value = False

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config(
            TrackioConfig(mode="online", max_rollout_traces_per_step=32)
        )
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.log.reset_mock()

        tokenizer = MagicMock()
        tokenizer.decode.side_effect = lambda ids, skip_special_tokens=False: "|".join(
            str(i) for i in ids
        )
        trajectory = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 0, 1, 1, 0, 0, 1]], dtype=torch.int64),
        }

        logger.log_rollout_traces(
            [trajectory],
            split="rollout",
            global_step=5,
            tokenizer=tokenizer,
        )

        assert mock_trackio.Trace.call_args.kwargs["messages"] == [
            {"role": "user", "content": "1|2"},
            {"role": "assistant", "content": "3|4"},
            {"role": "tool", "content": "5|6"},
            {"role": "assistant", "content": "7"},
        ]

    @patch("areal.utils.stats_logger.trackio")
    @patch("areal.utils.stats_logger.wandb")
    @patch("areal.utils.stats_logger.swanlab")
    @patch("areal.utils.stats_logger.dist")
    def test_trackio_trace_logging_prefers_structured_messages(
        self, mock_dist, mock_swanlab, mock_wandb, mock_trackio
    ):
        """Structured conversation data should be logged directly when present."""
        mock_dist.is_initialized.return_value = False

        from areal.utils.stats_logger import StatsLogger

        config = _make_test_config(
            TrackioConfig(mode="online", max_rollout_traces_per_step=32)
        )
        logger = StatsLogger(config, _make_ft_spec())
        mock_trackio.log.reset_mock()

        tokenizer = MagicMock()
        trajectory = {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 1, 1]], dtype=torch.int64),
            "messages": [
                [
                    {"role": "system", "content": "Use tools when needed."},
                    {"role": "user", "content": "What is 2 + 2?"},
                    {"role": "assistant", "content": "I will calculate it."},
                    {"role": "tool", "content": "calculator: 4"},
                    {"role": "assistant", "content": "4"},
                ]
            ],
        }

        logger.log_rollout_traces(
            [trajectory],
            split="rollout",
            global_step=6,
            tokenizer=tokenizer,
        )

        assert (
            mock_trackio.Trace.call_args.kwargs["messages"] == trajectory["messages"][0]
        )
        tokenizer.decode.assert_not_called()

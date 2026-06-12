# SPDX-License-Identifier: Apache-2.0

"""Tests for the trajectory dump/replay debug feature (issue #1343).

Tests cover:
1. TrajectoryDebugConfig validation logic.
2. Trajectory save/load round-trip correctness.
3. _trajectory_path naming conventions.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch

from areal.api.cli_args import TrajectoryDebugConfig

# ------------------------------------------------------------------ #
#  Phase 1: Config validation                                         #
# ------------------------------------------------------------------ #


class TestTrajectoryDebugConfig:
    """Validate TrajectoryDebugConfig __post_init__ constraints."""

    def test_default_config_valid(self):
        """Default config (all False) should be valid."""
        cfg = TrajectoryDebugConfig()
        assert not cfg.dump_rollout_data
        assert not cfg.replay_rollout_data

    def test_dump_only_valid(self):
        cfg = TrajectoryDebugConfig(dump_rollout_data=True)
        assert cfg.dump_rollout_data
        assert not cfg.replay_rollout_data

    def test_replay_only_valid(self):
        cfg = TrajectoryDebugConfig(replay_rollout_data=True)
        assert cfg.replay_rollout_data
        assert not cfg.dump_rollout_data

    def test_mutual_exclusion_raises(self):
        """dump and replay cannot both be True."""
        with pytest.raises(ValueError, match="[Mm]utual"):
            TrajectoryDebugConfig(dump_rollout_data=True, replay_rollout_data=True)

    def test_negative_dump_steps_raises(self):
        """dump_steps with negative values should raise."""
        with pytest.raises(ValueError, match="non-negative"):
            TrajectoryDebugConfig(dump_rollout_data=True, dump_steps=[1, -5, 10])

    def test_invalid_scope_raises(self):
        """dump_scope must be 'rollout' or 'full'."""
        with pytest.raises(ValueError, match="scope"):
            TrajectoryDebugConfig(dump_rollout_data=True, dump_scope="invalid")

    def test_full_scope_valid(self):
        """dump_scope='full' should be accepted without error."""
        cfg = TrajectoryDebugConfig(dump_rollout_data=True, dump_scope="full")
        assert cfg.dump_scope == "full"

    def test_replay_with_full_scope_valid(self):
        """replay with dump_scope='full' should be accepted."""
        cfg = TrajectoryDebugConfig(replay_rollout_data=True, dump_scope="full")
        assert cfg.dump_scope == "full"

    def test_dump_steps_preserved(self):
        """dump_steps should pass through when valid."""
        cfg = TrajectoryDebugConfig(dump_rollout_data=True, dump_steps=[1, 5, 10])
        assert cfg.dump_steps == [1, 5, 10]

    def test_custom_path(self):
        cfg = TrajectoryDebugConfig(dump_rollout_data=True, path="/tmp/my_traj")
        assert cfg.path == "/tmp/my_traj"

    def test_max_keep_valid(self):
        cfg = TrajectoryDebugConfig(dump_rollout_data=True, max_keep=5)
        assert cfg.max_keep == 5

    def test_max_keep_zero_raises(self):
        """max_keep must be positive."""
        with pytest.raises(ValueError, match="positive"):
            TrajectoryDebugConfig(dump_rollout_data=True, max_keep=0)

    def test_max_keep_negative_raises(self):
        """max_keep must be positive."""
        with pytest.raises(ValueError, match="positive"):
            TrajectoryDebugConfig(dump_rollout_data=True, max_keep=-3)

    def test_pin_steps_valid(self):
        cfg = TrajectoryDebugConfig(
            dump_rollout_data=True, max_keep=5, pin_steps=[100, 200]
        )
        assert cfg.pin_steps == [100, 200]

    def test_pin_steps_negative_raises(self):
        """pin_steps must be non-negative."""
        with pytest.raises(ValueError, match="non-negative"):
            TrajectoryDebugConfig(dump_rollout_data=True, pin_steps=[10, -1])


# ------------------------------------------------------------------ #
#  Phase 2: Trajectory IO helpers                                      #
# ------------------------------------------------------------------ #


class TestTrajectoryIO:
    """Test _trajectory_path, _save_trajectory, _load_trajectory.

    We import and call the methods directly on a minimal mock of PPOTrainer
    to avoid standing up the full trainer.
    """

    @pytest.fixture()
    def fake_trainer(self, tmp_path):
        """Create a minimal object that has the trajectory helper methods."""
        from areal.trainer.rl_trainer import PPOTrainer

        # We'll attach the unbound methods to a simple namespace
        class _FakeTrainer:
            pass

        obj = _FakeTrainer()
        obj._trajectory_dir = str(tmp_path / "trajectories")
        os.makedirs(obj._trajectory_dir, exist_ok=True)

        # Bind the methods from PPOTrainer
        obj._trajectory_path = PPOTrainer._trajectory_path.__get__(obj)
        obj._save_trajectory = PPOTrainer._save_trajectory.__get__(obj)
        obj._load_trajectory = PPOTrainer._load_trajectory.__get__(obj)
        obj._rotate_trajectories = PPOTrainer._rotate_trajectories.__get__(obj)
        obj._max_keep = None
        obj._pin_steps = frozenset()
        return obj

    @pytest.fixture()
    def sample_batch(self):
        """Create a representative rollout batch."""
        return [
            {
                "input_ids": torch.randint(0, 1000, (4, 128)),
                "logprobs": torch.randn(4, 128),
                "rewards": torch.randn(4),
            },
            {
                "input_ids": torch.randint(0, 1000, (4, 64)),
                "logprobs": torch.randn(4, 64),
                "rewards": torch.randn(4),
            },
        ]

    def test_path_single_controller(self, fake_trainer):
        """In single-controller mode, path has no rank suffix."""
        with patch("areal.trainer.rl_trainer.is_single_controller", return_value=True):
            path = fake_trainer._trajectory_path(42)
        assert path.endswith("step_000042.pt")
        assert "rank" not in path

    def test_path_spmd_mode(self, fake_trainer):
        """In SPMD mode, path includes data-parallel rank."""
        # Mock actor.data_parallel_rank for SPMD path
        fake_trainer.actor = type("_FakeActor", (), {"data_parallel_rank": 3})()
        with patch(
            "areal.trainer.rl_trainer.is_single_controller",
            return_value=False,
        ):
            path = fake_trainer._trajectory_path(7)
        assert path.endswith("step_000007_dp_3.pt")

    def test_save_load_roundtrip(self, fake_trainer, sample_batch):
        """Save then load should produce identical tensors."""
        step = 100
        with patch("areal.trainer.rl_trainer.is_single_controller", return_value=True):
            fake_trainer._save_trajectory(sample_batch, step)
            loaded = fake_trainer._load_trajectory(step)

        assert len(loaded) == len(sample_batch)
        for orig, reloaded in zip(sample_batch, loaded):
            for key in orig:
                assert torch.equal(orig[key], reloaded[key]), f"Mismatch on key '{key}'"

    def test_load_missing_raises(self, fake_trainer):
        """Loading a non-existent step should raise FileNotFoundError."""
        with patch("areal.trainer.rl_trainer.is_single_controller", return_value=True):
            with pytest.raises(FileNotFoundError, match="step 999"):
                fake_trainer._load_trajectory(999)

    def test_dump_steps_filtering(self, fake_trainer, sample_batch):
        """Only steps in dump_steps_set should produce files."""
        dump_steps = frozenset([5, 10])

        with patch("areal.trainer.rl_trainer.is_single_controller", return_value=True):
            for step in range(1, 12):
                if dump_steps is None or step in dump_steps:
                    fake_trainer._save_trajectory(sample_batch, step)

        files = os.listdir(fake_trainer._trajectory_dir)
        assert len(files) == 2
        assert "step_000005.pt" in files
        assert "step_000010.pt" in files

    def test_max_keep_rotation(self, fake_trainer, sample_batch):
        """With max_keep=3, only the 3 most recent files should remain."""
        fake_trainer._max_keep = 3

        with patch("areal.trainer.rl_trainer.is_single_controller", return_value=True):
            for step in range(1, 8):
                fake_trainer._save_trajectory(sample_batch, step)

        files = sorted(os.listdir(fake_trainer._trajectory_dir))
        assert len(files) == 3
        assert files == ["step_000005.pt", "step_000006.pt", "step_000007.pt"]

    def test_max_keep_none_keeps_all(self, fake_trainer, sample_batch):
        """With max_keep=None (default), all files are kept."""
        fake_trainer._max_keep = None

        with patch("areal.trainer.rl_trainer.is_single_controller", return_value=True):
            for step in range(1, 6):
                fake_trainer._save_trajectory(sample_batch, step)

        files = os.listdir(fake_trainer._trajectory_dir)
        assert len(files) == 5

    def test_pin_steps_preserved_during_rotation(self, fake_trainer, sample_batch):
        """Pinned steps should never be deleted by max_keep rotation."""
        fake_trainer._max_keep = 2
        fake_trainer._pin_steps = frozenset([3, 5])

        with patch("areal.trainer.rl_trainer.is_single_controller", return_value=True):
            for step in range(1, 8):
                fake_trainer._save_trajectory(sample_batch, step)

        files = sorted(os.listdir(fake_trainer._trajectory_dir))
        # Pinned: step 3, 5 (always kept)
        # Rotatable: steps 1,2,4,6,7 → keep most recent 2 → keep 6,7
        assert "step_000003.pt" in files  # pinned
        assert "step_000005.pt" in files  # pinned
        assert "step_000006.pt" in files  # recent rotatable
        assert "step_000007.pt" in files  # recent rotatable
        assert len(files) == 4
        # Old non-pinned files should be gone
        assert "step_000001.pt" not in files
        assert "step_000002.pt" not in files
        assert "step_000004.pt" not in files

    def test_pin_steps_without_max_keep(self, fake_trainer, sample_batch):
        """pin_steps without max_keep has no effect (all files kept anyway)."""
        fake_trainer._max_keep = None
        fake_trainer._pin_steps = frozenset([2, 4])

        with patch("areal.trainer.rl_trainer.is_single_controller", return_value=True):
            for step in range(1, 6):
                fake_trainer._save_trajectory(sample_batch, step)

        files = os.listdir(fake_trainer._trajectory_dir)
        assert len(files) == 5

    def test_rotation_skips_unparseable_filenames(self, fake_trainer, sample_batch):
        """Malformed step_*.pt files must be skipped, not crash rotation.

        The trajectory directory is user-visible, so a hand-copied or
        half-written file like 'step_backup.pt' can appear. Rotation should
        warn-and-skip such files and still rotate the well-formed ones.
        """
        fake_trainer._max_keep = 2

        # Drop in a malformed file that matches the step_*.pt glob but has no
        # numeric step component.
        bad_path = os.path.join(fake_trainer._trajectory_dir, "step_backup.pt")
        torch.save(sample_batch, bad_path)

        with patch("areal.trainer.rl_trainer.is_single_controller", return_value=True):
            # Saving triggers _rotate_trajectories on each write; the malformed
            # file must never cause an exception.
            for step in range(1, 6):
                fake_trainer._save_trajectory(sample_batch, step)

        files = sorted(os.listdir(fake_trainer._trajectory_dir))
        # Malformed file is exempt from the budget and never deleted.
        assert "step_backup.pt" in files
        # Well-formed files rotate down to the 2 most recent.
        assert "step_000004.pt" in files
        assert "step_000005.pt" in files
        assert "step_000003.pt" not in files

    def test_rotation_tolerates_missing_file(self, fake_trainer, sample_batch):
        """Concurrent rotation (file already gone) must not raise.

        In SPMD mode multiple data-parallel ranks rotate the shared directory
        at once, so a file we plan to delete may already be removed by a peer.
        os.remove is wrapped in try/except FileNotFoundError; simulate the race
        by deleting the target out from under _rotate_trajectories.
        """
        # Write several files with rotation effectively disabled (huge budget),
        # so we have a real backlog to rotate in the patched call below.
        fake_trainer._max_keep = 100
        with patch("areal.trainer.rl_trainer.is_single_controller", return_value=True):
            for step in range(1, 8):
                fake_trainer._save_trajectory(sample_batch, step)

        # Now tighten the budget so the explicit rotation has files to delete.
        fake_trainer._max_keep = 2

        real_remove = os.remove

        def racing_remove(path):
            # Simulate a peer rank deleting the file first, then our own call.
            real_remove(path)
            # A second delete of the same path raises FileNotFoundError, which
            # _rotate_trajectories must swallow.
            real_remove(path)

        with patch("os.remove", side_effect=racing_remove):
            # Should complete without raising despite the double-delete.
            fake_trainer._rotate_trajectories()

        # The 2 most recent files survive; the rest were rotated out.
        files = sorted(os.listdir(fake_trainer._trajectory_dir))
        assert files == ["step_000006.pt", "step_000007.pt"]


# ------------------------------------------------------------------ #
#  _NoOpRollout stub interface                                         #
# ------------------------------------------------------------------ #


class TestNoOpRollout:
    """Verify _NoOpRollout stub satisfies the InferenceEngine interface."""

    def test_noop_rollout_sync_methods(self):
        """All synchronous methods should be callable without error."""
        from areal.trainer.rl_trainer import _NoOpRollout

        noop = _NoOpRollout()
        # These should all be no-ops that don't raise
        noop.pause()
        noop.resume()
        noop.destroy()
        noop.set_version(42)
        noop.offload()
        noop.onload()
        noop.config_perf_tracer(enabled=True)
        noop.save_perf_tracer(path="/tmp/fake")
        assert noop.export_stats() == {}
        assert noop.staleness_manager is None
        assert noop.workflow_executor is None

    @pytest.mark.asyncio
    async def test_noop_rollout_async_methods(self):
        """Async methods should be awaitable without error."""
        from areal.trainer.rl_trainer import _NoOpRollout

        noop = _NoOpRollout()
        await noop.pause_generation()
        await noop.continue_generation()

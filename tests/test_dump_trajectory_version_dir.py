"""Tests for WorkflowExecutor._dump_trajectory version-directory routing."""

import asyncio
import getpass
import json
import logging
import os

import pytest
import torch

from areal.infra.workflow_executor import WorkflowExecutor


class _FakeTokenizer:
    def decode(self, ids: list[int], **kwargs) -> str:
        return f"[{len(ids)} tokens]"


class _FakeConfig:
    def __init__(self, fileroot: str):
        self.fileroot = fileroot
        self.experiment_name = "exp"
        self.trial_name = "trial"
        self.dump_to_file = True
        self.tokenizer_path = None


class _FakeInferenceEngine:
    def __init__(self, version: int):
        self._version = version

    def get_version(self) -> int:
        return self._version


def _make_executor(tmp_path, engine_version: int = 99) -> WorkflowExecutor:
    """Build a WorkflowExecutor with only the fields _dump_trajectory needs."""
    ex = object.__new__(WorkflowExecutor)
    ex.config = _FakeConfig(str(tmp_path))
    ex._tokenizer = _FakeTokenizer()
    ex.inference_engine = _FakeInferenceEngine(engine_version)
    ex.logger = logging.getLogger("test_dump_trajectory_version_dir")
    return ex


def _rollout_dir(tmp_path) -> str:
    return os.path.join(
        str(tmp_path), "logs", getpass.getuser(), "exp", "trial", "rollout"
    )


def _read_dir(rollout_dir: str, version: int) -> list[dict]:
    """Read every record dumped under <rollout_dir>/<version>/."""
    vdir = os.path.join(rollout_dir, str(version))
    if not os.path.isdir(vdir):
        return []
    records = []
    for name in sorted(os.listdir(vdir)):
        with open(os.path.join(vdir, name)) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _traj(versions: list[list[int]], prompt_len: int = 2) -> dict:
    """Build a trajectory group; versions[i] is the per-token version of sample i."""
    batch_size = len(versions)
    seqlen = len(versions[0])
    mask_row = [0] * prompt_len + [1] * (seqlen - prompt_len)
    return {
        "input_ids": torch.arange(batch_size * seqlen).reshape(batch_size, seqlen),
        "rewards": torch.zeros(batch_size),
        "loss_mask": torch.tensor([mask_row] * batch_size),
        "attention_mask": torch.ones(batch_size, seqlen, dtype=torch.long),
        "versions": torch.tensor(versions),
    }


class TestDumpTrajectoryVersionDir:
    def test_samples_of_different_versions_go_to_own_version_dirs(self, tmp_path):
        executor = _make_executor(tmp_path)
        # sample 0 generated entirely under version 0, sample 1 under version 1
        traj = _traj([[0, 0, 0, 0], [1, 1, 1, 1]])

        ok, reason = asyncio.run(
            executor._dump_trajectory(traj, task_id=7, is_eval=False)
        )

        assert ok, reason
        rollout_dir = _rollout_dir(tmp_path)
        v0 = _read_dir(rollout_dir, 0)
        v1 = _read_dir(rollout_dir, 1)
        assert [r["sample_idx"] for r in v0] == [0]
        assert [r["sample_idx"] for r in v1] == [1]
        assert v0[0]["tail_version"] == 0
        assert v1[0]["tail_version"] == 1

    def test_every_record_tail_version_matches_its_directory(self, tmp_path):
        executor = _make_executor(tmp_path)
        traj = _traj([[0, 0, 0, 0], [0, 0, 1, 2], [0, 0, 2, 2]])

        ok, reason = asyncio.run(
            executor._dump_trajectory(traj, task_id=1, is_eval=False)
        )

        assert ok, reason
        rollout_dir = _rollout_dir(tmp_path)
        assert os.path.isdir(rollout_dir)
        for version in sorted(os.listdir(rollout_dir)):
            for record in _read_dir(rollout_dir, int(version)):
                assert record["tail_version"] == int(version)

    def test_uniform_version_group_uses_single_dir(self, tmp_path):
        executor = _make_executor(tmp_path)
        traj = _traj([[3, 3, 3, 3], [3, 3, 3, 3]])

        ok, reason = asyncio.run(
            executor._dump_trajectory(traj, task_id=5, is_eval=False)
        )

        assert ok, reason
        rollout_dir = _rollout_dir(tmp_path)
        assert sorted(os.listdir(rollout_dir)) == ["3"]
        assert [r["sample_idx"] for r in _read_dir(rollout_dir, 3)] == [0, 1]

    def test_missing_versions_field_falls_back_to_engine_version(self, tmp_path):
        executor = _make_executor(tmp_path, engine_version=42)
        traj = _traj([[0, 0, 0, 0]])
        del traj["versions"]

        ok, reason = asyncio.run(
            executor._dump_trajectory(traj, task_id=2, is_eval=False)
        )

        assert ok, reason
        rollout_dir = _rollout_dir(tmp_path)
        assert sorted(os.listdir(rollout_dir)) == ["42"]
        assert _read_dir(rollout_dir, 42)[0]["tail_version"] == 42

    def test_appends_across_calls_without_overwriting(self, tmp_path):
        executor = _make_executor(tmp_path)
        traj = _traj([[1, 1, 1, 1]])

        asyncio.run(executor._dump_trajectory(traj, task_id=9, is_eval=False))
        asyncio.run(executor._dump_trajectory(traj, task_id=9, is_eval=False))

        assert len(_read_dir(_rollout_dir(tmp_path), 1)) == 2

    def test_samples_with_empty_completion_are_skipped(self, tmp_path):
        executor = _make_executor(tmp_path)
        traj = _traj([[0, 0, 0, 0], [1, 1, 1, 1]])
        # sample 1's final token is prompt/context -> empty completion tail
        traj["loss_mask"] = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 0]])

        ok, reason = asyncio.run(
            executor._dump_trajectory(traj, task_id=3, is_eval=False)
        )

        assert ok, reason
        rollout_dir = _rollout_dir(tmp_path)
        assert sorted(os.listdir(rollout_dir)) == ["0"]
        assert [r["sample_idx"] for r in _read_dir(rollout_dir, 0)] == [0]

    def test_eval_dump_uses_eval_rollout_dir(self, tmp_path):
        executor = _make_executor(tmp_path)
        traj = _traj([[2, 2, 2, 2]])

        ok, reason = asyncio.run(
            executor._dump_trajectory(traj, task_id=4, is_eval=True)
        )

        assert ok, reason
        base = os.path.join(str(tmp_path), "logs", getpass.getuser(), "exp", "trial")
        assert sorted(os.listdir(os.path.join(base, "eval-rollout"))) == ["2"]
        assert not os.path.isdir(os.path.join(base, "rollout"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

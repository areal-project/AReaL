"""Tests for WorkflowExecutor trajectory dump helpers."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.infra.workflow_executor import WorkflowExecutor


class _FakeTokenizer:
    """Minimal tokenizer that returns a deterministic string for testing."""

    def decode(self, ids: list[int], **kwargs) -> str:
        return f"[{len(ids)} tokens]"


class TestSplitTrajectoryForDump:
    @pytest.fixture
    def tokenizer(self):
        return _FakeTokenizer()

    def test_single_turn_no_segments(self, tokenizer):
        ids = [1, 2, 3, 4, 5, 6]
        mask = [0, 0, 0, 1, 1, 1]
        result = WorkflowExecutor._split_trajectory_for_dump(ids, mask, tokenizer)
        assert result["prompt_end"] == 3
        assert result["prompt_text"] == "[3 tokens]"
        assert result["completion_text"] == "[3 tokens]"
        assert result["segments"] is None

    def test_multi_turn_has_segments(self, tokenizer):
        ids = list(range(8))
        mask = [0, 0, 1, 1, 0, 0, 1, 1]
        result = WorkflowExecutor._split_trajectory_for_dump(ids, mask, tokenizer)
        assert result["prompt_end"] == 2
        assert result["segments"] is not None
        assert len(result["segments"]) == 4
        roles = [s["role"] for s in result["segments"]]
        assert roles == ["prompt", "gen", "context", "gen"]
        lengths = [s["len"] for s in result["segments"]]
        assert lengths == [2, 2, 2, 2]

    def test_all_zeros_prompt_only(self, tokenizer):
        ids = [1, 2, 3]
        mask = [0, 0, 0]
        result = WorkflowExecutor._split_trajectory_for_dump(ids, mask, tokenizer)
        assert result["prompt_end"] == 3
        assert result["prompt_text"] == "[3 tokens]"
        assert result["completion_text"] == "[0 tokens]"
        assert result["segments"] is None

    def test_all_ones_gen_only(self, tokenizer):
        ids = [10, 20, 30]
        mask = [1, 1, 1]
        result = WorkflowExecutor._split_trajectory_for_dump(ids, mask, tokenizer)
        assert result["prompt_end"] == 0
        assert result["prompt_text"] == "[0 tokens]"
        assert result["completion_text"] == "[3 tokens]"
        assert result["segments"] is None

    def test_three_gen_runs(self, tokenizer):
        ids = list(range(12))
        mask = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        result = WorkflowExecutor._split_trajectory_for_dump(ids, mask, tokenizer)
        assert result["prompt_end"] == 2
        assert result["segments"] is not None
        roles = [s["role"] for s in result["segments"]]
        assert roles == ["prompt", "gen", "context", "gen", "context", "gen"]

    def test_length_mismatch_raises_valueerror(self, tokenizer):
        with pytest.raises(ValueError, match="ids length 2 != mask length 1"):
            WorkflowExecutor._split_trajectory_for_dump([1, 2], [0], tokenizer)

    def test_single_token_gen(self, tokenizer):
        ids = [42]
        mask = [1]
        result = WorkflowExecutor._split_trajectory_for_dump(ids, mask, tokenizer)
        assert result["prompt_end"] == 0
        assert result["segments"] is None


class TestComputeOutputVersions:
    def test_filters_negative_one_placeholders(self):
        versions = [-1, -1, 5, 5, 6]
        mask = [0, 0, 1, 1, 1]
        head, tail, rle = WorkflowExecutor._compute_output_versions(versions, mask)
        assert head == 5
        assert tail == 6
        assert rle == [[5, 2], [6, 1]]

    def test_single_version(self):
        versions = [-1, 3, 3, 3]
        mask = [0, 1, 1, 1]
        head, tail, rle = WorkflowExecutor._compute_output_versions(versions, mask)
        assert head == 3
        assert tail == 3
        assert rle == [[3, 3]]

    def test_multiple_version_transitions(self):
        versions = [-1, 2, 2, 3, 3, 4]
        mask = [0, 1, 1, 1, 1, 1]
        head, tail, rle = WorkflowExecutor._compute_output_versions(versions, mask)
        assert head == 2
        assert tail == 4
        assert rle == [[2, 2], [3, 2], [4, 1]]

    def test_all_masked_out(self):
        versions = [1, 2, 3]
        mask = [0, 0, 0]
        head, tail, rle = WorkflowExecutor._compute_output_versions(versions, mask)
        assert head == -1
        assert tail == -1
        assert rle == []

    def test_interleaved_multi_turn(self):
        versions = [-1, -1, 5, 5, -1, -1, 6, 6]
        mask = [0, 0, 1, 1, 0, 0, 1, 1]
        head, tail, rle = WorkflowExecutor._compute_output_versions(versions, mask)
        assert head == 5
        assert tail == 6
        assert rle == [[5, 2], [6, 2]]


class TestComputeTrajectoryRowMetadata:
    def test_computes_grouped_row_indices(self):
        trajectory_indices, row_indices = (
            WorkflowExecutor._compute_trajectory_row_metadata([1, 0, 0, 1], 4)
        )

        assert trajectory_indices == [0, 0, 0, 1]
        assert row_indices == [0, 1, 2, 0]

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="length 1 != batch size 2"):
            WorkflowExecutor._compute_trajectory_row_metadata([1], 2)

    def test_rejects_missing_first_row_marker(self):
        with pytest.raises(ValueError, match="first row must begin"):
            WorkflowExecutor._compute_trajectory_row_metadata([0, 1], 2)


@pytest.mark.asyncio
async def test_workflow_executor_marks_single_proxy_session_rows():
    interactions = {
        f"response-{index}": InteractionWithTokenLogpReward(
            _cache={
                "input_ids": torch.tensor([[index, index + 1]]),
                "attention_mask": torch.ones((1, 2), dtype=torch.bool),
            }
        )
        for index in range(3)
    }
    workflow = SimpleNamespace(arun_episode=AsyncMock(return_value=interactions))
    executor = object.__new__(WorkflowExecutor)
    executor.inference_engine = MagicMock()
    executor._staleness_manager = MagicMock()
    executor.logger = MagicMock()
    executor.config = SimpleNamespace(
        check_trajectory_format=False,
        dump_to_file=False,
        enable_rollout_tracing=False,
    )
    pending_task = SimpleNamespace(
        task_id=7,
        data={},
        workflow=workflow,
        should_accept_fn=None,
        is_eval=False,
    )

    result = await executor._create_workflow_task(pending_task)()

    assert result is not None
    assert result.trajectory["begin_of_trajectory"].tolist() == [1, 0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_markers", [False, True])
async def test_workflow_executor_tensor_episode_preserves_trajectory_boundaries(
    explicit_markers,
):
    """Non-grouped tensor episodes get a default boundary without overwriting one."""
    trajectory = {
        "input_ids": torch.ones(3, 2, dtype=torch.long),
        "attention_mask": torch.ones(3, 2, dtype=torch.bool),
    }
    expected = torch.tensor(
        [1, 0, 1] if explicit_markers else [1, 0, 0], dtype=torch.int32
    )
    if explicit_markers:
        trajectory["begin_of_trajectory"] = expected.clone()
    executor = object.__new__(WorkflowExecutor)
    executor.inference_engine = MagicMock()
    executor._staleness_manager = MagicMock()
    executor.logger = MagicMock()
    executor.config = SimpleNamespace(
        check_trajectory_format=False,
        dump_to_file=False,
        enable_rollout_tracing=False,
    )
    pending_task = SimpleNamespace(
        task_id=7,
        data={},
        workflow=SimpleNamespace(arun_episode=AsyncMock(return_value=trajectory)),
        should_accept_fn=None,
        is_eval=False,
    )

    result = await executor._create_workflow_task(pending_task)()

    assert result is not None
    torch.testing.assert_close(
        result.trajectory["begin_of_trajectory"], expected, rtol=0, atol=0
    )
    assert ("begin_of_trajectory" in trajectory) == explicit_markers


@pytest.mark.asyncio
async def test_dump_trajectory_writes_grouped_row_indices(tmp_path):
    executor = object.__new__(WorkflowExecutor)
    executor.logger = MagicMock()
    executor._get_dump_dir = lambda _is_eval: str(tmp_path)
    executor._get_tokenizer = lambda: _FakeTokenizer()

    trajectory = {
        "input_ids": torch.tensor(
            [[1, 11], [2, 12], [3, 13], [4, 14]], dtype=torch.long
        ),
        "rewards": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "loss_mask": torch.tensor([[0, 1]] * 4, dtype=torch.int32),
        "attention_mask": torch.ones((4, 2), dtype=torch.bool),
        "versions": torch.tensor([[-1, 3]] * 4, dtype=torch.int32),
        "begin_of_trajectory": torch.tensor([1, 0, 0, 1], dtype=torch.int32),
    }

    success, reason = await executor._dump_trajectory(
        trajectory, task_id=7, is_eval=False
    )

    assert success, reason
    records = [
        json.loads(line)
        for line in (tmp_path / "3" / "7.jsonl").read_text().splitlines()
    ]
    assert [record["reward"] for record in records] == [1.0, 1.0, 1.0, 0.0]
    assert [
        {
            key: record[key]
            for key in (
                "begin_of_trajectory",
                "trajectory_idx",
                "row_idx_in_trajectory",
            )
        }
        for record in records
    ] == [
        {
            "begin_of_trajectory": 1,
            "trajectory_idx": 0,
            "row_idx_in_trajectory": 0,
        },
        {
            "begin_of_trajectory": 0,
            "trajectory_idx": 0,
            "row_idx_in_trajectory": 1,
        },
        {
            "begin_of_trajectory": 0,
            "trajectory_idx": 0,
            "row_idx_in_trajectory": 2,
        },
        {
            "begin_of_trajectory": 1,
            "trajectory_idx": 1,
            "row_idx_in_trajectory": 0,
        },
    ]


@pytest.mark.asyncio
async def test_dump_trajectory_treats_legacy_batch_rows_as_samples(tmp_path):
    executor = object.__new__(WorkflowExecutor)
    executor.logger = MagicMock()
    executor._get_dump_dir = lambda _is_eval: str(tmp_path)
    executor._get_tokenizer = lambda: _FakeTokenizer()

    trajectory = {
        "input_ids": torch.tensor([[1, 11], [2, 12]], dtype=torch.long),
        "rewards": torch.tensor([1.0, 0.0]),
        "loss_mask": torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
        "attention_mask": torch.ones((2, 2), dtype=torch.bool),
        "versions": torch.tensor([[-1, 3], [-1, 3]], dtype=torch.int32),
    }

    success, reason = await executor._dump_trajectory(
        trajectory, task_id=8, is_eval=False
    )

    assert success, reason
    records = [
        json.loads(line)
        for line in (tmp_path / "3" / "8.jsonl").read_text().splitlines()
    ]
    assert [record["begin_of_trajectory"] for record in records] == [1, 1]
    assert [record["trajectory_idx"] for record in records] == [0, 1]
    assert [record["row_idx_in_trajectory"] for record in records] == [0, 0]

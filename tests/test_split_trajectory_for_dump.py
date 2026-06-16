"""Tests for WorkflowExecutor._split_trajectory_for_dump and _compute_output_versions."""

import pytest

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

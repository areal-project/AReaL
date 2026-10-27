# SPDX-License-Identifier: Apache-2.0
"""Retained-row metric counting, metadata alignment, and tensor transport."""

from types import SimpleNamespace

import pytest
import torch

from areal.utils.data import concat_padded_tensors, split_and_unpad_tensor
from areal.utils.rollout_stats import (
    ROLLOUT_HARNESS_OUTCOME,
    ROLLOUT_TURN_COUNT,
    _count_generated_turns,
    record_rollout_turn_stats,
    with_rollout_turn_stats,
)
from areal.utils.stats_tracker import DistributedStatsTracker


@pytest.fixture
def recorded_stats(monkeypatch):
    trackers = {}

    def get_tracker(name=""):
        return trackers.setdefault(name, DistributedStatsTracker(name))

    monkeypatch.setattr("areal.utils.stats_tracker.get", get_tracker)
    return trackers


def _trajectory(reward, turns, metadata=None):
    turn_ids = torch.cat(
        [torch.tensor([-1]), torch.arange(turns).repeat_interleave(2)]
    ).unsqueeze(0)
    return with_rollout_turn_stats(
        {
            "turn_ids": turn_ids,
            "input_ids": torch.ones_like(turn_ids),
            "loss_mask": turn_ids >= 0,
            "attention_mask": torch.ones_like(turn_ids, dtype=torch.bool),
            "rewards": torch.tensor([reward], dtype=torch.float32),
        },
        metadata,
    )


@pytest.mark.parametrize("turn_ids", [[-1, 0, 0], [-1, 0, 1], [-1, 0, 1, 2]])
@pytest.mark.parametrize("empty_last_response", [False, True])
def test_history_count_uses_responses_independently_of_tensor_fallback(
    turn_ids, empty_last_response
):
    parent = SimpleNamespace(model_response=SimpleNamespace(output_len=2), parent=None)
    child = SimpleNamespace(model_response=SimpleNamespace(output_len=2), parent=parent)
    leaf = (
        SimpleNamespace(model_response=SimpleNamespace(output_len=0), parent=child)
        if empty_last_response
        else child
    )
    row = {
        "turn_ids": torch.tensor([turn_ids]),
        "loss_mask": torch.zeros(1, len(turn_ids)),
    }
    torch.testing.assert_close(
        _count_generated_turns(row, leaf), torch.tensor([2.0]), rtol=0, atol=0
    )


def test_selected_batch_records_one_turn_count_per_row(recorded_stats):
    """Turn metrics retain the original average/minimum/maximum/count names."""
    record_rollout_turn_stats(
        [_trajectory(1.0, 2), _trajectory(0.0, 3), _trajectory(1.0, 1)]
    )

    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_count"] == 3
    assert metrics["train/num_turns/avg"] == pytest.approx(2.0)
    assert metrics["train/num_turns/min"] == pytest.approx(1.0)
    assert metrics["train/num_turns/max"] == pytest.approx(3.0)
    assert metrics["train/num_turns_no_harness_err_count"] == 3
    assert metrics["train/num_turns_no_harness_err/avg"] == pytest.approx(2.0)
    assert metrics["train/num_turns_no_harness_err/min"] == pytest.approx(1.0)
    assert metrics["train/num_turns_no_harness_err/max"] == pytest.approx(3.0)


def test_selected_batch_excludes_harness_errors_from_filtered_turns(
    recorded_stats,
):
    """Filtered turn stats omit retained Harness failures only."""
    record_rollout_turn_stats(
        [
            _trajectory(
                0.0,
                4,
                {
                    "arena_status": "HARNESS_FAILED",
                    "harness_outcome_code": "LLM_CONTEXT_WINDOW_EXCEEDED",
                },
            ),
            _trajectory(1.0, 2),
        ]
    )

    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns/avg"] == pytest.approx(3.0)
    assert metrics["train/num_turns/min"] == pytest.approx(2.0)
    assert metrics["train/num_turns/max"] == pytest.approx(4.0)
    assert metrics["train/num_turns_no_harness_err_count"] == 1
    assert metrics["train/num_turns_no_harness_err/avg"] == pytest.approx(2.0)
    assert metrics["train/num_turns_no_harness_err/min"] == pytest.approx(2.0)
    assert metrics["train/num_turns_no_harness_err/max"] == pytest.approx(2.0)
    assert metrics["train/num_turns_harness_err_count"] == 1
    assert metrics["train/num_turns_harness_err/avg"] == pytest.approx(4.0)
    assert metrics["train/num_turns_harness_err/min"] == pytest.approx(4.0)
    assert metrics["train/num_turns_harness_err/max"] == pytest.approx(4.0)
    outcome_metric = "train/num_turns_harness_err/LLM_CONTEXT_WINDOW_EXCEEDED"
    assert metrics[f"{outcome_metric}_count"] == 1
    assert metrics[f"{outcome_metric}/avg"] == pytest.approx(4.0)
    assert metrics[f"{outcome_metric}/min"] == pytest.approx(4.0)
    assert metrics[f"{outcome_metric}/max"] == pytest.approx(4.0)


@pytest.mark.parametrize("code", ["FUTURE_UNKNOWN_CODE", None, ["invalid"]])
def test_selected_batch_bounds_unknown_harness_outcome(recorded_stats, code):
    """Unknown outcome values aggregate under OTHER instead of adding keys."""
    record_rollout_turn_stats(
        [
            _trajectory(
                0.0, 3, {"arena_status": "HARNESS_FAILED", "harness_outcome_code": code}
            )
        ]
    )

    metrics = recorded_stats["train"].export()

    assert "train/num_turns_harness_err/FUTURE_UNKNOWN_CODE/avg" not in metrics
    assert metrics["train/num_turns_harness_err/OTHER/avg"] == pytest.approx(3.0)


def test_turn_metadata_tracks_concat_reorder_and_row_filter(recorded_stats):
    rows = [
        _trajectory(1.0, 2),
        _trajectory(0.0, 7),
        _trajectory(0.0, 4, {"arena_status": "HARNESS_FAILED"}),
    ]
    joined = concat_padded_tensors(rows)
    selected = {key: value[[2, 0]] for key, value in joined.items()}
    split = split_and_unpad_tensor(selected, n_trajs=2, traj_group_sizes=[1, 1])

    record_rollout_turn_stats(split)
    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_count"] == 2
    assert metrics["train/num_turns/avg"] == 3
    assert metrics["train/num_turns_no_harness_err/avg"] == 2
    assert metrics["train/num_turns_harness_err/OTHER/avg"] == 4


def test_count_turns_includes_loss_masked_turns_but_excludes_original_padding():
    """Loss masking does not undo a generation; prompt/padding are not turns."""
    trajectory = {
        "turn_ids": torch.tensor([[-1, 2, 2, -1, 8, 8, 2, 12, 0]]),
        "loss_mask": torch.tensor([[0, 1, 1, 0, 1, 1, 1, 0, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]]),
    }

    row = with_rollout_turn_stats(trajectory, None)

    torch.testing.assert_close(
        row[ROLLOUT_TURN_COUNT], torch.tensor([3.0]), rtol=0, atol=0
    )


@pytest.mark.parametrize("status", [None, "", " ", 123])
def test_required_harness_status_missing_is_neither_success_nor_failure(
    recorded_stats, status
):
    """Unknown V2 metadata counts only toward the overall turn distribution."""
    row = with_rollout_turn_stats(
        _trajectory(0.0, 3), {"arena_status": status}, require_harness_status=True
    )

    record_rollout_turn_stats([row])
    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_count"] == 1
    assert metrics["train/num_turns/avg"] == 3
    assert metrics["train/num_turns_no_harness_err_count"] == 0
    assert metrics["train/num_turns_harness_err_count"] == 0


def test_original_turns_follow_row_reorder_despite_mask_edits(recorded_stats):
    """Case history and Harness labels stay aligned after token/row edits."""
    joined = concat_padded_tensors(
        [
            _trajectory(1.0, 2),
            _trajectory(0.0, 4, {"arena_status": "HARNESS_FAILED"}),
        ]
    )
    edited = {key: value[[1, 0]] for key, value in joined.items()}
    edited["loss_mask"][edited["turn_ids"] >= 1] = 0

    record_rollout_turn_stats([edited])
    metrics = recorded_stats["train"].export()

    torch.testing.assert_close(
        edited[ROLLOUT_TURN_COUNT], torch.tensor([4.0, 2.0]), rtol=0, atol=0
    )
    assert metrics["train/num_turns_no_harness_err_count"] == 1
    assert metrics["train/num_turns_harness_err/OTHER_count"] == 1
    assert metrics["train/num_turns_harness_err/OTHER/avg"] == 4
    assert metrics["train/num_turns_no_harness_err/avg"] == 2


def test_zero_turn_row_is_retained(recorded_stats):
    record_rollout_turn_stats([_trajectory(0.0, 0)])

    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_count"] == 1
    assert metrics["train/num_turns/avg"] == 0


def test_empty_selected_batch_has_zero_count_not_stale_average(recorded_stats):
    record_rollout_turn_stats([_trajectory(1.0, 5)])
    recorded_stats["train"].export()

    record_rollout_turn_stats([])
    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_count"] == 0
    assert "train/num_turns/avg" not in metrics


@pytest.mark.parametrize("classified", [False, True])
def test_turn_stats_fetch_only_compact_columns(recorded_stats, monkeypatch, classified):
    from areal.infra.rpc.rtensor import RTensor

    count_handle, outcome_handle = object(), object()
    large_handle = SimpleNamespace(shape=(1, 128000))
    row = {
        ROLLOUT_TURN_COUNT: count_handle,
        ROLLOUT_HARNESS_OUTCOME: outcome_handle,
        "turn_ids": large_handle,
        "input_ids": large_handle,
    }
    if not classified:
        row.pop(ROLLOUT_HARNESS_OUTCOME)
    fetched = []

    def localize(payload):
        fetched.append(payload)
        data = {ROLLOUT_TURN_COUNT: torch.tensor([3.0])}
        if classified:
            data[ROLLOUT_HARNESS_OUTCOME] = torch.tensor([0], dtype=torch.int32)
        return [data]

    monkeypatch.setattr(RTensor, "localize", localize)
    record_rollout_turn_stats([row])

    expected = {ROLLOUT_TURN_COUNT: count_handle}
    if classified:
        expected[ROLLOUT_HARNESS_OUTCOME] = outcome_handle
    assert fetched == [[expected]]
    stats = recorded_stats["train"].export()
    assert stats["train/num_turns/avg"] == 3
    assert stats["train/num_turns_no_harness_err_count"] == int(classified)


def test_workflow_without_turn_metadata_does_not_emit_turn_metrics(recorded_stats):
    row = {"rewards": torch.tensor([1.0])}

    assert with_rollout_turn_stats(row, None) is row
    record_rollout_turn_stats([row])

    assert not recorded_stats


@pytest.mark.parametrize("code", ["LLM_CONTEXT_WINDOW_EXCEEDED", "OTHER"])
def test_each_known_harness_code_keeps_original_metric_name(recorded_stats, code):
    row = _trajectory(
        0.0, 3, {"arena_status": "HARNESS_FAILED", "harness_outcome_code": code}
    )

    record_rollout_turn_stats([row])
    metrics = recorded_stats["train"].export()

    assert metrics[f"train/num_turns_harness_err/{code}_count"] == 1
    assert metrics[f"train/num_turns_harness_err/{code}/avg"] == 3
    assert metrics["train/num_turns_no_harness_err_count"] == 0


def test_task_failure_without_harness_error_stays_in_filtered_turns(recorded_stats):
    record_rollout_turn_stats(
        [_trajectory(0.0, 4, {"arena_status": "DONE", "harness_outcome_code": "OTHER"})]
    )

    metrics = recorded_stats["train"].export()

    assert metrics["train/num_turns_no_harness_err_count"] == 1
    assert metrics["train/num_turns_harness_err_count"] == 0


@pytest.mark.parametrize(
    "misalignment", ["outcome_shape", "rows", "rows_without_outcome"]
)
def test_misaligned_turn_metadata_is_rejected(recorded_stats, misalignment):
    row = _trajectory(1.0, 2)
    if misalignment == "outcome_shape":
        row[ROLLOUT_HARNESS_OUTCOME] = torch.zeros(2, dtype=torch.int32)
    else:
        row["input_ids"] = torch.ones((2, 5), dtype=torch.int32)
        if misalignment == "rows_without_outcome":
            row.pop(ROLLOUT_HARNESS_OUTCOME)

    with pytest.raises(ValueError, match="rollout turn"):
        record_rollout_turn_stats([row])

    assert not recorded_stats


@pytest.mark.parametrize("mixed", [False, True])
def test_turn_counts_without_harness_classification_are_valid(recorded_stats, mixed):
    row = {ROLLOUT_TURN_COUNT: torch.tensor([2.0, 4.0])}
    batch = [row]
    if mixed:
        batch.append(_trajectory(0.0, 6, {"arena_status": "HARNESS_FAILED"}))
    record_rollout_turn_stats(batch)
    stats = recorded_stats["train"].export()
    assert stats["train/num_turns_count"] == (3 if mixed else 2)
    assert stats["train/num_turns/avg"] == (4 if mixed else 3)
    assert stats["train/num_turns_no_harness_err_count"] == 0
    assert stats["train/num_turns_harness_err_count"] == (1 if mixed else 0)
    assert ROLLOUT_HARNESS_OUTCOME not in row


def test_eval_row_recording_does_not_scan_history(recorded_stats):
    """Per-task eval recording must not inspect earlier tasks in the window."""

    class ObservedHistory(list):
        inspected = 0

        def __iter__(self):
            self.inspected += len(self)
            return super().__iter__()

        def __add__(self, other):
            self.inspected += len(self)
            return super().__add__(other)

    batch = [
        _trajectory(1.0, 2),
        _trajectory(0.0, 4, {"arena_status": "HARNESS_FAILED"}),
    ]
    record_rollout_turn_stats(batch, scope="eval-train")
    tracker = recorded_stats["eval-train"]
    histories = {key: ObservedHistory(values) for key, values in tracker.stats.items()}
    tracker.stats.update(histories)
    for _ in range(64):
        record_rollout_turn_stats(batch, scope="eval-train")
    assert all(history.inspected == 0 for history in histories.values())

    metrics = tracker.export()
    assert metrics["eval-train/num_turns_count"] == 130
    assert metrics["eval-train/num_turns/avg"] == 3
    assert metrics["eval-train/num_turns/min"] == 2
    assert metrics["eval-train/num_turns/max"] == 4
    assert metrics["eval-train/num_turns_harness_err/OTHER_count"] == 65
    assert metrics["eval-train/num_turns_harness_err/OTHER/avg"] == 4
    assert tracker.export() == {}

# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.utils.harness_metrics import HARNESS_OUTCOME_CODES, classify_harness_outcome


@pytest.mark.parametrize(
    "metadata,required,expected",
    [
        (None, False, 0),
        (None, True, -1),
        ({"arena_status": " "}, True, -1),
        ({"arena_status": 123}, True, -1),
        ({"arena_status": "DONE"}, True, 0),
        ({"arena_status": "TASK_FAILED"}, True, 0),
        ({"arena_status": "HARNESS_FAILED"}, True, 1),
        (
            {"arena_status": "HARNESS_FAILED", "harness_outcome_code": ["invalid"]},
            False,
            1,
        ),
        (
            {"arena_status": "HARNESS_FAILED", "harness_outcome_code": "FUTURE_CODE"},
            True,
            1,
        ),
    ],
)
def test_harness_classification_preserves_unknown_and_legacy_defaults(
    metadata, required, expected
):
    assert classify_harness_outcome(metadata, require_status=required) == expected


def test_known_harness_classification_keeps_named_failure():
    outcome = classify_harness_outcome(
        {
            "arena_status": "HARNESS_FAILED",
            "harness_outcome_code": "LLM_CONTEXT_WINDOW_EXCEEDED",
        }
    )
    assert HARNESS_OUTCOME_CODES[outcome - 1] == "LLM_CONTEXT_WINDOW_EXCEEDED"

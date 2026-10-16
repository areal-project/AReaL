# SPDX-License-Identifier: Apache-2.0
"""A parity run cannot pass without complete last-stage observations."""

import pytest
import torch

from examples.mtp.qwen38_flash_next.full_no_mtp_parity import (
    _forward_observation_counts,
    _global_observation_summary,
)


def test_two_empty_last_stage_outputs_cannot_pass():
    with pytest.raises(AssertionError, match="observation count mismatch"):
        _forward_observation_counts([], [], expected_microbatches=4, is_last_stage=True)


def test_empty_nonlast_stage_allowed_but_empty_global_run_rejected():
    counts = _forward_observation_counts(
        [], [], expected_microbatches=4, is_last_stage=False
    )
    assert counts == [0, 0, 0, 0, 0]
    with pytest.raises(AssertionError, match="zero observations"):
        _global_observation_summary(counts)


def test_missing_scheduled_microbatch_cannot_pass():
    outputs = [(torch.ones(2, 3), torch.tensor(1.0))]
    with pytest.raises(AssertionError, match="observation count mismatch"):
        _forward_observation_counts(
            outputs, outputs, expected_microbatches=2, is_last_stage=True
        )


def test_complete_microbatches_report_observed_logit_elements():
    outputs = [(torch.ones(2, 3), torch.tensor(1.0)) for _ in range(2)]
    counts = _forward_observation_counts(
        outputs, outputs, expected_microbatches=2, is_last_stage=True
    )
    result = _global_observation_summary(counts)
    assert result["compared_microbatches_rank_sum"] == 2
    assert result["compared_logit_elements_rank_sum"] == 12


def test_empty_logit_tensor_cannot_count_as_observation():
    outputs = [(torch.empty(0), torch.tensor(1.0))]
    with pytest.raises(AssertionError, match="nonempty logits"):
        _forward_observation_counts(
            outputs, outputs, expected_microbatches=1, is_last_stage=True
        )

# SPDX-License-Identifier: Apache-2.0
"""A parity run cannot pass without complete last-stage observations."""

import pytest
import torch

from examples.mtp.qwen38_flash_next import full_no_mtp_parity as parity
from examples.mtp.qwen38_flash_next.full_no_mtp_parity import (
    _forward_observation_counts,
    _global_observation_summary,
)

from areal.utils.data import collate_samples_to_list, concat_batch


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


def _collated_parity_samples():
    lengths = (520, 516, 511, 505)
    samples = [
        {
            "input_ids": torch.full((length,), row + 1, dtype=torch.long),
            "loss_mask": torch.arange(length) >= 2,
        }
        for row, length in enumerate(lengths)
    ]
    return collate_samples_to_list(samples), lengths


def test_parity_batch_collated_samples_preserve_all_rows_and_masks():
    """Real SFT collation retains four distinct rows through parity truncation."""
    samples, lengths = _collated_parity_samples()

    fixed = parity._prepare_parity_batch(samples)

    assert fixed["input_ids"].shape == torch.Size([4, 512])
    for row, length in enumerate(lengths):
        valid = min(length, 512)
        torch.testing.assert_close(
            fixed["input_ids"][row, :valid],
            torch.full((valid,), row + 1, dtype=torch.long),
            rtol=0,
            atol=0,
        )
        expected_attention = torch.arange(512) < valid
        expected_loss = expected_attention & (torch.arange(512) >= 2)
        torch.testing.assert_close(
            fixed["attention_mask"][row], expected_attention, rtol=0, atol=0
        )
        torch.testing.assert_close(
            fixed["loss_mask"][row], expected_loss, rtol=0, atol=0
        )
        assert samples[row]["input_ids"].shape == torch.Size([1, length])
    outputs = [
        (row.unsqueeze(-1).float(), torch.tensor(0.0)) for row in fixed["input_ids"]
    ]
    counts = parity._forward_observation_counts(
        outputs, outputs, expected_microbatches=4, is_last_stage=True
    )
    summary = parity._global_observation_summary([value * 8 for value in counts])
    assert summary["expected_microbatches_rank_sum"] == 32
    assert summary["compared_microbatches_rank_sum"] == 32
    assert summary["compared_logit_elements_rank_sum"] > 0


def test_parity_batch_dict_matches_collated_list():
    """The already-batched production path and raw SFT list remain equivalent."""
    samples, _ = _collated_parity_samples()
    batched, _ = concat_batch(samples)

    from_list = parity._prepare_parity_batch(samples)
    from_dict = parity._prepare_parity_batch(batched)

    assert from_list.keys() == from_dict.keys()
    for key in from_list:
        torch.testing.assert_close(from_dict[key], from_list[key], rtol=0, atol=0)


def test_parity_entry_consumes_production_collator_list(monkeypatch):
    """Exercise the actual comparison entry before optional model/GPU imports."""
    samples, _ = _collated_parity_samples()
    real_prepare = parity._prepare_parity_batch
    prepared = []

    class PreparedContractChecked(Exception):
        pass

    def prepare_and_stop(batch):
        prepared.append(real_prepare(batch))
        raise PreparedContractChecked

    monkeypatch.setattr(parity, "_prepare_parity_batch", prepare_and_stop)

    with pytest.raises(PreparedContractChecked):
        parity.compare_full_no_mtp(None, samples)

    assert len(prepared) == 1
    assert prepared[0]["input_ids"].shape == torch.Size([4, 512])
    assert prepared[0]["input_ids"][:, 0].tolist() == [1, 2, 3, 4]

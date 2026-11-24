# SPDX-License-Identifier: Apache-2.0

"""Tests for the column contracts applied to datasets loaded from disk."""

from types import SimpleNamespace

import pytest

from areal.dataset.contracts import (
    dataset_columns,
    required_columns,
    validate_dataset_contract,
)


def _dataset(*columns: str) -> SimpleNamespace:
    return SimpleNamespace(column_names=list(columns))


def test_required_columns_documents_supported_types():
    assert required_columns("sft") == ("input_ids", "loss_mask")
    assert required_columns("rl") == ("messages",)
    assert required_columns("rw") == ()
    assert required_columns("dpo") == ()


def test_validate_accepts_text_rl_record_with_messages():
    validate_dataset_contract(_dataset("messages", "answer"), "rl", "/data/rl")


def test_validate_rejects_text_rl_record_without_messages():
    with pytest.raises(ValueError, match="messages"):
        validate_dataset_contract(_dataset("prompt", "answer"), "rl", "/data/rl")


def test_validate_rejects_tokenized_sft_record_missing_loss_mask():
    with pytest.raises(ValueError, match="loss_mask"):
        validate_dataset_contract(
            _dataset("input_ids", "attention_mask"), "sft", "/data/sft"
        )


def test_validate_accepts_tokenized_sft_record():
    validate_dataset_contract(_dataset("input_ids", "loss_mask"), "sft", "/data/sft")


def test_validate_skips_unconstrained_type():
    validate_dataset_contract(_dataset("chosen", "rejected"), "dpo", "/data/dpo")


def test_validate_skips_opaque_objects_without_column_names():
    validate_dataset_contract(object(), "rl", "/data/opaque")


def test_dataset_columns_reads_first_split_of_dataset_dict():
    fake = SimpleNamespace(
        column_names={"train": ["messages"], "validation": ["messages"]}
    )

    assert dataset_columns(fake) == ["messages"]

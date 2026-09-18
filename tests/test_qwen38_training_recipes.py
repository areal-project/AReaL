# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest
import torch

from examples.swe.qwen38_flash_next import patch_sglang_qsa_topk as patch


def test_thinking_defaults_respect_explicit_switch_without_mutation():
    from examples.swe.qwen38_flash_next.template_defaults import with_template_defaults

    assert with_template_defaults()["chat_template_kwargs"] == dict(
        enable_thinking=True, reasoning_effort="medium", thinking_option=None
    )
    body = {"chat_template_kwargs": {"thinking_option": "off"}, "other": 42}
    merged = with_template_defaults(body)
    assert merged["chat_template_kwargs"] == dict(
        thinking_option="off", reasoning_effort="medium"
    )
    assert merged["other"] == 42
    assert body == {"chat_template_kwargs": {"thinking_option": "off"}, "other": 42}


def test_external_task_selection_preserves_order_and_rejects_drift():
    from examples.swe.qwen38_flash_next.train_rl import select_task_indices

    assert select_task_indices(["a", "b", "c"], ["c", "a"]) == [2, 0]
    for selected in ([], ["a", "a"], ["missing"], "a", [None]):
        with pytest.raises(ValueError):
            select_task_indices(["a", "b"], selected)


def test_cache_isolation_preserves_original_request():
    from types import SimpleNamespace

    from examples.swe.qwen38_flash_next.proxy import wrap_isolated_cache

    original = SimpleNamespace(payload={"input_ids": [1, 2]})
    build = wrap_isolated_cache(lambda _: original)
    first, second = build(None), build(None)
    assert first.payload["cache_salt"] != second.payload["cache_salt"]
    assert first.payload["input_ids"] == original.payload["input_ids"]
    assert "cache_salt" not in original.payload


def test_stable_topk_ties_respect_bounds_and_pad():
    logits = torch.tensor([[9.0, 3.0, 3.0, 3.0], [9.0, 8.0, 7.0, 6.0]])
    starts = torch.tensor([1, 2], dtype=torch.int32)
    ends = torch.tensor([4, 3], dtype=torch.int32)

    result = patch.stable_qsa_topk(logits, starts, ends, 2)

    torch.testing.assert_close(
        result, torch.tensor([[0, 1], [0, -1]], dtype=torch.int32), rtol=0, atol=0
    )


def test_stable_topk_empty_prefix_returns_padding():
    result = patch.stable_qsa_topk(
        torch.zeros(1, 2), torch.tensor([0]), torch.tensor([0]), 4
    )

    torch.testing.assert_close(
        result, torch.full((1, 4), -1, dtype=torch.int32), rtol=0, atol=0
    )


@pytest.mark.parametrize("start,end", [(-1, 2), (2, 1), (0, 4)])
def test_stable_topk_invalid_bounds_rejected(start, end):
    with pytest.raises(ValueError, match="Invalid row bounds"):
        patch.stable_qsa_topk(
            torch.zeros(1, 3), torch.tensor([start]), torch.tensor([end]), 1
        )


def test_patch_unknown_source_rejected():
    with pytest.raises(ValueError, match="SHA256"):
        patch.patched_source("def qsa_fast_topk(): pass\n")


def test_patch_wrapper_preserves_native_and_enables_stable_selection(monkeypatch):
    source = (
        "import torch\n"
        "def qsa_fast_topk(logits, row_starts, row_ends, topk):\n"
        "    return torch.tensor([[2, 1]], dtype=torch.int32)\n"
    )
    monkeypatch.setattr(
        patch, "EXPECTED_SHA256", hashlib.sha256(source.encode()).hexdigest()
    )
    monkeypatch.delenv("QWEN_QSA_STABLE_TOPK", raising=False)
    namespace = {}
    exec(patch.patched_source(source), namespace)
    inputs = (torch.ones(1, 3), torch.tensor([0]), torch.tensor([3]), 2)

    torch.testing.assert_close(
        namespace["qsa_fast_topk"](*inputs),
        torch.tensor([[2, 1]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    monkeypatch.setenv("QWEN_QSA_STABLE_TOPK", "1")
    torch.testing.assert_close(
        namespace["qsa_fast_topk"](*inputs),
        torch.tensor([[0, 1]], dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_bounds_patch_rejects_unknown_source():
    from examples.swe.qwen38_flash_next.patch_sglang_qsa_compress_gather import (
        patched_source,
    )

    with pytest.raises(RuntimeError, match="Unknown or already-patched"):
        patched_source(b"# unrelated source\n")

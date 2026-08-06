# SPDX-License-Identifier: Apache-2.0
"""Unit tests for separated-card post-apply weight verification."""

import importlib.util
import logging as stdlib_logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_module(monkeypatch):
    """Load the pure verifier without importing AReaL's optional HTTP stack."""
    logging_mod = types.ModuleType("areal.utils.logging")
    logging_mod.getLogger = stdlib_logging.getLogger
    utils_mod = types.ModuleType("areal.utils")
    utils_mod.logging = logging_mod
    monkeypatch.setitem(sys.modules, "areal", types.ModuleType("areal"))
    monkeypatch.setitem(sys.modules, "areal.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "areal.utils.logging", logging_mod)

    path = (
        Path(__file__).resolve().parent.parent
        / "areal/v2/weight_update/awex/separation_verify.py"
    )
    spec = importlib.util.spec_from_file_location("awex_separation_verify_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operation(*, dtype=torch.bfloat16):
    return SimpleNamespace(
        send_rank=1,
        recv_rank=0,
        send_shard_meta=SimpleNamespace(
            name="train_w", shape=(2, 3), dtype=torch.float32
        ),
        recv_shard_meta=SimpleNamespace(name="infer_w", shape=(2, 2), dtype=dtype),
        send_offset=(0, 0),
        recv_offset=(0, 0),
        overlap_shape=(2, 2),
        train_slices=(slice(None), slice(1, 3)),
        inf_slices=(slice(None), slice(None)),
    )


def _reports(module, train: torch.Tensor, infer: torch.Tensor):
    plan = SimpleNamespace(operations={0: [_operation()]})
    return [
        module._build_local_report({"train_w": train}, plan, role="train"),
        module._build_local_report({"infer_w": infer}, plan, role="infer"),
    ]


def test_post_apply_verify_matches_noncontiguous_dtype_converted_slices(monkeypatch):
    """Equivalent train/infer slices match after receiver dtype conversion."""
    train = torch.tensor([[99.0, 1.0, 2.0], [99.0, 3.0, 4.0]], dtype=torch.float32)
    infer = torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.bfloat16).T
    assert not infer.is_contiguous()

    module = _load_module(monkeypatch)
    assert module._validate_reports(_reports(module, train, infer)) == (1, 4)


def test_post_apply_verify_rejects_single_value_corruption(monkeypatch):
    """One changed receiver value must fail the 128-bit fingerprint gate."""
    train = torch.tensor([[99.0, 1.0, 2.0], [99.0, 3.0, 4.0]], dtype=torch.bfloat16)
    infer = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.bfloat16)

    module = _load_module(monkeypatch)
    with pytest.raises(RuntimeError, match="post-apply weight mismatch"):
        module._validate_reports(_reports(module, train, infer))


def test_post_apply_verify_rejects_empty_global_plan(monkeypatch):
    """A vacuous all-empty comparison cannot qualify weight correctness."""
    module = _load_module(monkeypatch)
    with pytest.raises(RuntimeError, match="non-empty train and infer plans"):
        module._validate_reports(
            [
                {"role": "train", "entries": [], "error": None},
                {"role": "infer", "entries": [], "error": None},
            ]
        )


def test_post_apply_verify_rejects_receiver_coverage_gap(monkeypatch):
    """A plan that omits part of a receiver shard must fail before hashing."""
    module = _load_module(monkeypatch)
    op = _operation()
    op.recv_shard_meta.shape = (3, 2)
    op.inf_slices = (slice(0, 2), slice(None))
    plan = SimpleNamespace(operations={0: [op]})

    report = module._build_local_report(
        {"infer_w": torch.zeros(3, 2, dtype=torch.bfloat16)},
        plan,
        role="infer",
    )

    assert "coverage gap" in report["error"]


def test_post_apply_verify_allows_identical_replicated_receiver_regions(monkeypatch):
    """Replicated senders may target one identical complete receiver region."""
    module = _load_module(monkeypatch)
    first = _operation()
    second = _operation()
    second.send_rank = 2
    plan = SimpleNamespace(operations={0: [first], 1: [second]})

    report = module._build_local_report(
        {"infer_w": torch.zeros(2, 2, dtype=torch.bfloat16)},
        plan,
        role="infer",
    )

    assert report["error"] is None
    assert len(report["entries"]) == 2


def test_post_apply_verify_rejects_divergent_replicated_sender(monkeypatch):
    """Every replicated sender must match the final receiver fingerprint."""
    module = _load_module(monkeypatch)
    first = _operation()
    second = _operation()
    second.send_rank = 2
    infer_plan = SimpleNamespace(operations={1: [first], 2: [second]})
    train = torch.tensor([[99.0, 1.0, 2.0], [99.0, 3.0, 4.0]])
    divergent = torch.tensor([[99.0, 1.0, 2.0], [99.0, 3.0, 5.0]])
    infer = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    reports = [
        module._build_local_report(
            {"train_w": train},
            SimpleNamespace(operations={0: [first]}),
            role="train",
        ),
        module._build_local_report(
            {"train_w": divergent},
            SimpleNamespace(operations={0: [second]}),
            role="train",
        ),
        module._build_local_report({"infer_w": infer}, infer_plan, role="infer"),
    ]

    with pytest.raises(RuntimeError, match="post-apply weight mismatch"):
        module._validate_reports(reports)


def test_post_apply_verify_accepts_matching_replicated_senders(monkeypatch):
    """Matching replicated senders and their shared receiver must qualify."""
    module = _load_module(monkeypatch)
    first = _operation()
    second = _operation()
    second.send_rank = 2
    infer_plan = SimpleNamespace(operations={1: [first], 2: [second]})
    train = torch.tensor([[99.0, 1.0, 2.0], [99.0, 3.0, 4.0]])
    infer = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    reports = [
        module._build_local_report(
            {"train_w": train},
            SimpleNamespace(operations={0: [first]}),
            role="train",
        ),
        module._build_local_report(
            {"train_w": train},
            SimpleNamespace(operations={0: [second]}),
            role="train",
        ),
        module._build_local_report({"infer_w": infer}, infer_plan, role="infer"),
    ]

    assert module._validate_reports(reports) == (2, 8)


def test_post_apply_verify_rejects_partial_receiver_overlap(monkeypatch):
    """Distinct receiver regions must remain disjoint."""
    module = _load_module(monkeypatch)
    first = _operation()
    first.inf_slices = (slice(None), slice(0, 2))
    second = _operation()
    second.send_rank = 2
    second.inf_slices = (slice(None), slice(1, 3))
    plan = SimpleNamespace(operations={0: [first], 1: [second]})

    report = module._build_local_report(
        {"infer_w": torch.zeros(2, 3, dtype=torch.bfloat16)},
        plan,
        role="infer",
    )

    assert "overlap" in report["error"]


def test_post_apply_verify_rejects_unplanned_parameter(monkeypatch):
    """A parameter absent from the transfer plan cannot be silently ignored."""
    module = _load_module(monkeypatch)
    train = torch.tensor([[99.0, 1.0, 2.0], [99.0, 3.0, 4.0]])
    infer = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    reports = _reports(module, train, infer)
    reports[1]["param_names"].append("unplanned.weight")

    with pytest.raises(RuntimeError, match="parameter coverage mismatch"):
        module._validate_reports(reports)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_post_apply_fingerprint_cuda_matches_cpu(monkeypatch):
    """GPU byte-lane reductions must produce the same digest as CPU."""
    module = _load_module(monkeypatch)
    plan = SimpleNamespace(operations={0: [_operation()]})
    tensor = torch.tensor([[99.0, 1.0, 2.0], [99.0, 3.0, 4.0]], dtype=torch.float32)

    cpu = module._build_local_report({"train_w": tensor}, plan, role="train")
    gpu = module._build_local_report({"train_w": tensor.cuda()}, plan, role="train")

    assert gpu == cpu

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch

from areal.v2.weight_update.awex.weight_digest import (
    _ROLLING_SNAPSHOTS,
    _filter_digest_items,
    _phase_enabled,
    build_tensor_digest,
    build_tensor_digest_report,
    log_tensor_digest,
)


def test_build_tensor_digest_supports_bfloat16() -> None:
    first = {
        "bf16": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        "fp32": torch.tensor([[3.0], [4.0]], dtype=torch.float32),
    }
    second = {
        "fp32": torch.tensor([[3.0], [4.0]], dtype=torch.float32),
        "bf16": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
    }

    left = build_tensor_digest(first.items())
    right = build_tensor_digest(second.items())

    assert left["digest"] == right["digest"]
    assert left["tensors"] == 2
    assert left["missing"] == 0
    assert left["elements"] == 4


def test_build_tensor_digest_changes_with_tensor_value() -> None:
    left = build_tensor_digest(
        [("value", torch.tensor([1.0, 2.0], dtype=torch.bfloat16))]
    )
    right = build_tensor_digest(
        [("value", torch.tensor([1.0, 3.0], dtype=torch.bfloat16))]
    )

    assert left["digest"] != right["digest"]


def test_build_tensor_digest_report_records_per_tensor_metadata() -> None:
    report = build_tensor_digest_report(
        [("value", torch.tensor([[1.0, 2.0]], dtype=torch.float32))]
    )

    assert report["tensors"] == 1
    assert report["tensor_records"] == [
        {
            "param": "value",
            "dtype": "torch.float32",
            "shape": [1, 2],
            "numel": 2,
            "nbytes": 8,
            "digest_bytes": 8,
            "digest": report["tensor_records"][0]["digest"],
        }
    ]


def test_build_tensor_digest_can_sample_large_tensors(monkeypatch) -> None:
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST_SAMPLE_ELEMENTS", "2")

    left = build_tensor_digest(
        [("value", torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))]
    )
    right = build_tensor_digest(
        [("value", torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32))]
    )

    assert left["digest"] != right["digest"]
    assert left["bytes"] == 8


def test_filter_digest_items_respects_regex_and_limits(monkeypatch) -> None:
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST_NAME_REGEX", "router")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST_MAX_TENSORS", "1")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST_MAX_BYTES", "8")

    selected, meta = _filter_digest_items(
        [
            ("layers.0.mlp.dense.weight", torch.ones(2, dtype=torch.float32)),
            ("layers.0.mlp.router.weight", torch.ones(2, dtype=torch.float32)),
            ("layers.1.mlp.router.weight", torch.ones(2, dtype=torch.float32)),
        ]
    )

    assert [name for name, _ in selected] == ["layers.0.mlp.router.weight"]
    assert meta["considered"] == 3
    assert meta["selected"] == 1
    assert meta["skipped_name"] == 1
    assert meta["skipped_limit"] == 1
    assert meta["selected_names"] == ["layers.0.mlp.router.weight"]
    assert meta["skipped_limit_names"] == ["layers.1.mlp.router.weight"]


def test_phase_enabled_accepts_comma_and_semicolon(monkeypatch) -> None:
    monkeypatch.setenv(
        "AREAL_DTE_WEIGHT_DIGEST_PHASES",
        "post_optimizer_param; pre_send,post_apply",
    )

    assert _phase_enabled("post_optimizer_param")
    assert _phase_enabled("pre_send")
    assert _phase_enabled("post_apply")
    assert not _phase_enabled("pre_optimizer_grad")


def test_log_tensor_digest_writes_manifest_and_index(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST", "1")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_ROOT", str(tmp_path))
    monkeypatch.setenv("AREAL_DTE_WEIGHT_COMPARE_GROUP", "group")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_RUN", "left")
    monkeypatch.setenv("EXP_NAME", "exp")
    monkeypatch.setenv("TRIAL_NAME", "trial")

    log_tensor_digest(
        [("param", torch.tensor([1.0], dtype=torch.float32))],
        role="train",
        phase="post_optimizer_param",
        step=1,
    )

    manifest = tmp_path / "group" / "left" / "manifest.jsonl"
    index = (
        tmp_path
        / "group"
        / "index"
        / "role=train"
        / "phase=post_optimizer_param"
        / "step=1"
        / "version=none"
        / "rank=-1"
        / "left.json"
    )
    assert manifest.exists()
    assert index.exists()
    assert '"record_type": "aggregate"' in manifest.read_text()
    assert '"record_type": "tensor"' in manifest.read_text()


def test_log_tensor_digest_peer_mismatch_captures_current_and_previous(
    tmp_path,
    monkeypatch,
) -> None:
    _ROLLING_SNAPSHOTS.clear()
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST", "1")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_ROOT", str(tmp_path))
    monkeypatch.setenv("AREAL_DTE_WEIGHT_COMPARE_GROUP", "group")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_ROLLING", "1")
    monkeypatch.setenv(
        "AREAL_DTE_WEIGHT_CAPTURE_ROLLING_PHASES",
        "post_optimizer_param",
    )
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST_FAIL_FAST", "0")

    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_RUN", "left")
    log_tensor_digest(
        [("param", torch.tensor([1.0], dtype=torch.float32))],
        role="train",
        phase="post_optimizer_param",
        step=1,
    )

    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_RUN", "right")
    log_tensor_digest(
        [("param", torch.tensor([2.0], dtype=torch.float32))],
        role="train",
        phase="post_optimizer_param",
        step=1,
    )

    flag = tmp_path / "group" / "mismatch_flag.json"
    summary = tmp_path / "group" / "right" / "mismatch_summary.json"
    assert flag.exists()
    assert summary.exists()
    text = (tmp_path / "group" / "right" / "manifest.jsonl").read_text()
    assert '"label": "current"' in text
    assert '"label": "previous"' in text
    assert list((tmp_path / "group" / "right" / "captures").rglob("*.pt"))
    _ROLLING_SNAPSHOTS.clear()


def test_log_tensor_digest_expected_peer_timeout_writes_flag(
    tmp_path,
    monkeypatch,
) -> None:
    _ROLLING_SNAPSHOTS.clear()
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST", "1")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_ROOT", str(tmp_path))
    monkeypatch.setenv("AREAL_DTE_WEIGHT_COMPARE_GROUP", "group")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_RUN", "left")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_EXPECTED_RUNS", "left,right")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_COMPARE_WAIT_SECONDS", "0")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST_MAX_STEP", "-1")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST_FAIL_FAST", "0")

    log_tensor_digest(
        [("param", torch.tensor([1.0], dtype=torch.float32))],
        role="train",
        phase="post_optimizer_param",
        step=1,
    )

    flag = tmp_path / "group" / "mismatch_flag.json"
    assert flag.exists()
    assert "expected_peer_timeout" in flag.read_text()


def test_log_tensor_digest_can_force_test_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    _ROLLING_SNAPSHOTS.clear()
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST", "1")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_ROOT", str(tmp_path))
    monkeypatch.setenv("AREAL_DTE_WEIGHT_COMPARE_GROUP", "group")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST_MAX_STEP", "-1")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST_FAIL_FAST", "0")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_ROLLING", "1")
    monkeypatch.setenv(
        "AREAL_DTE_WEIGHT_CAPTURE_ROLLING_PHASES",
        "post_optimizer_param",
    )
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST_FORCE_MISMATCH_RUNS", "full")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST_FORCE_MISMATCH_ROLE", "train")
    monkeypatch.setenv(
        "AREAL_DTE_WEIGHT_DIGEST_FORCE_MISMATCH_PHASE",
        "post_optimizer_param",
    )
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST_FORCE_MISMATCH_STEP", "3")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_DIGEST_FORCE_MISMATCH_LABEL", "step3_test")

    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_RUN", "adamw")
    log_tensor_digest(
        [("param", torch.tensor([1.0], dtype=torch.float32))],
        role="train",
        phase="post_optimizer_param",
        step=3,
    )

    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_RUN", "full")
    log_tensor_digest(
        [("param", torch.tensor([1.0], dtype=torch.float32))],
        role="train",
        phase="post_optimizer_param",
        step=3,
    )

    flag = tmp_path / "group" / "mismatch_flag.json"
    assert flag.exists()
    summary = (tmp_path / "group" / "full" / "mismatch_summary.json").read_text()
    assert "__areal_dte_weight_digest_forced_mismatch__.step3_test" in summary
    manifest = (tmp_path / "group" / "full" / "manifest.jsonl").read_text()
    assert '"forced_mismatch"' in manifest
    _ROLLING_SNAPSHOTS.clear()

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from types import SimpleNamespace


def test_rollout_fingerprint_disabled_does_not_write(tmp_path, monkeypatch):
    from areal.utils.rollout_fingerprint import log_event

    monkeypatch.delenv("AREAL_ROLLOUT_FINGERPRINT", raising=False)
    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_ROOT", str(tmp_path))
    logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    log_event(logger, "rollout_batch_selected", task_ids=[1])

    assert not list(tmp_path.rglob("*.jsonl"))


def test_rollout_fingerprint_writes_batch_manifest(tmp_path, monkeypatch):
    from areal.utils.rollout_fingerprint import log_event

    monkeypatch.setenv("AREAL_ROLLOUT_FINGERPRINT", "1")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_ROOT", str(tmp_path))
    monkeypatch.setenv("AREAL_DTE_WEIGHT_COMPARE_GROUP", "group")
    monkeypatch.setenv("AREAL_DTE_WEIGHT_CAPTURE_RUN", "run")
    logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    log_event(
        logger,
        "rollout_batch_selected",
        task_ids=[1, 2],
        accepted=2,
        returned=2,
    )

    manifest = tmp_path / "group" / "run" / "rollout_batch_manifest.jsonl"
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["event"] == "rollout_batch_selected"
    assert record["task_ids"] == [1, 2]

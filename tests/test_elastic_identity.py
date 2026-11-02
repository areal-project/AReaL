# SPDX-License-Identifier: Apache-2.0

"""Recovery compatibility, measurement applicability and bundle proof stay separate."""

from pathlib import Path

import pytest

from tests.test_elastic_checkpoint import (
    native_lifecycle,  # noqa: F401 -- shared native controller fixture
)
from tests.test_elastic_monitor import _update

from areal.elastic.infra.monitor.identity import measurement_identity, semantic_digest


def test_measurement_digest_ignores_comments_formatting_and_docstrings():
    """Text-only edits keep measurements applicable, real statements do not."""
    original = '"""old docs"""\ndef value():\n    """old docs"""\n    return 1\n'
    cosmetic = (
        '"""new docs"""\n# comment\ndef value():\n    """new docs"""\n    return (1)\n'
    )
    assert semantic_digest(original) == semantic_digest(cosmetic)
    assert semantic_digest(original) != semantic_digest(
        original.replace("return 1", "return 2")
    )


def test_offline_trace_and_comment_changes_keep_measurement_identity(
    request, monkeypatch
):
    """Only timing-relevant implementations contribute semantic source digests."""
    manager = request.getfixturevalue("native_lifecycle").manager
    original = Path.read_text
    identity = measurement_identity(manager)
    read_paths = []

    def cosmetic(path, *args, **kwargs):
        read_paths.append(path.as_posix())
        return original(path, *args, **kwargs) + "\n# harmless comment\n"

    monkeypatch.setattr(Path, "read_text", cosmetic)
    assert measurement_identity(manager) == identity
    assert not any(path.endswith("monitor/trace.py") for path in read_paths)


def test_changed_timing_identity_restores_model_and_starts_fresh_history(
    request, monkeypatch
):
    """A changed phase implementation does not reject compatible full state."""
    fixture = request.getfixturevalue("native_lifecycle")
    manager = fixture.manager
    _update(manager)
    manager.close()
    identity = manager.identity
    prior_history = manager.monitor.runtime_history.path
    prior_bytes = prior_history.read_bytes()
    original = Path.read_text

    def changed_timing(path, *args, **kwargs):
        source = original(path, *args, **kwargs)
        if path.as_posix().endswith("elastic/infra/runtime.py"):
            source += "\nMEASUREMENT_IMPLEMENTATION_CHANGED = True\n"
        return source

    monkeypatch.setattr(Path, "read_text", changed_timing)
    restored = fixture.create()
    try:
        assert restored.identity == identity
        assert (
            restored.reconfigurator_adapter.execution_id
            != manager.reconfigurator_adapter.execution_id
        )
        assert restored.applied_version == restored.checkpoint_version == 1
        assert restored.data_progress == {"samples": 4, "effective_tokens": 8}
        assert restored.monitor.runtime_history.path != prior_history
        assert prior_history.read_bytes() == prior_bytes
        assert restored.monitor.runtime_history.snapshot(1)["updates"] == []
        restored.ensure_actor()
        assert restored.actor.updates == 1
        _update(restored)
        assert (
            restored.monitor.runtime_history.snapshot(2)["updates"][0][
                "committed_tokens"
            ]
            == 16
        )
    finally:
        restored.close()


@pytest.mark.parametrize(
    "progress",
    [
        [],
        {"samples": -1, "effective_tokens": 8},
        {"samples": 4, "effective_tokens": True},
    ],
)
def test_invalid_logical_progress_rejected_before_writing_checkpoint(
    tmp_path, progress
):
    """Optional metrics never weaken validation of critical manifest counters."""
    from areal.elastic.infra.state.checkpoint_store import CheckpointStore

    store = CheckpointStore(tmp_path, free_space_floor=0)
    with pytest.raises(ValueError, match="logical data progress"):
        store.commit(
            step=1,
            world_size=1,
            compatibility={},
            estimated_bytes=0,
            data_progress=progress,
            write=lambda path: pytest.fail("writer must not run"),
        )
    assert not (tmp_path / "latest.json").exists()

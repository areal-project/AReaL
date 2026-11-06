# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from dataclasses import replace

import pytest

from areal.v2.data_engineering import (
    CalibrationConfig,
    DataJobConfig,
    DatasetSnapshot,
    InsufficientDataError,
    export_hf_dataset,
    load_snapshot,
    run_data_job,
)
from areal.v2.data_engineering.snapshot import read_jsonl


def config(**kwargs):
    return DataJobConfig(
        recipe="tests.data_engineering_fixtures:create_recipe", target_count=2, **kwargs
    )


def calibration(mode="normal"):
    return CalibrationConfig(
        "tests.data_engineering_fixtures:create_calibrator",
        params={"mode": mode},
    )


@pytest.mark.asyncio
async def test_custom_recipe_cleans_excludes_deduplicates_and_publishes(tmp_path):
    """Domain cleaning composes with generic ID/input exclusion and publication."""
    excluded = tmp_path / "excluded.jsonl"
    excluded.write_text('{"task_id":"excluded"}\n')
    snapshot = await run_data_job(config(exclude=(str(excluded),)), tmp_path / "job")
    rows = snapshot.training_rows()
    assert rows == [
        {"task_id": "a", "prompt": "1", "answer": "1"},
        {"task_id": "b", "prompt": "3", "answer": "3"},
    ]
    assert all("reference" not in row for row in read_jsonl(snapshot.public_path))
    report = json.loads((snapshot.root / "report.json").read_text())
    assert report["duplicates"] == 2
    assert report["excluded"] == 1
    assert report["rejected_by_recipe"] == 1
    assert load_snapshot(snapshot.root).dataset_id == snapshot.dataset_id
    assert json.loads((tmp_path / "job/job.json").read_text())["status"] == "succeeded"


@pytest.mark.asyncio
async def test_repeated_jobs_have_identical_snapshot_ids_and_protect_existing_output(
    tmp_path,
):
    """Destination and wall-clock state do not change content identity."""
    first = await run_data_job(config(), tmp_path / "first")
    second = await run_data_job(config(), tmp_path / "second")
    assert first.dataset_id == second.dataset_id
    with pytest.raises(FileExistsError):
        await run_data_job(config(), tmp_path / "first")
    assert first.training_rows() == second.training_rows()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["normal", "mutate", "failure"])
async def test_public_only_calibration_isolates_backend_mutation_and_execution_failure(
    tmp_path, mode
):
    """A failing execution is not graded as an incorrect model answer."""
    snapshot = await run_data_job(
        config(calibration=calibration(mode)), tmp_path / "job"
    )
    public = read_jsonl(snapshot.public_path)
    assert all(row["input"]["value"] != "mutated backend input" for row in public)
    if mode == "failure":
        assert "a" not in {row["task_id"] for row in public}
        probes = read_jsonl(snapshot.root / "probes.jsonl")
        assert all(row["score"] is None for row in probes if row["task_id"] == "a")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["missing", "duplicate", "unknown", "bad_repeat", "bad_provenance"]
)
async def test_bad_calibration_identity_fails_without_publishing(tmp_path, mode):
    """Calibration alignment errors cannot silently alter a dataset."""
    output = tmp_path / "job"
    with pytest.raises(ValueError):
        await run_data_job(config(calibration=calibration(mode)), output)
    assert not (output / "dataset").exists()
    assert json.loads((output / "job.json").read_text())["status"] == "failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "conflict",
        "id_conflict",
        "alias_conflict",
        "nan",
        "bad_selection",
        "bad_row",
        "exception",
    ],
)
async def test_invalid_recipe_output_does_not_publish(tmp_path, mode):
    """Conflicting labels, bad scores, and invalid projections are rejected."""
    output = tmp_path / "job"
    with pytest.raises((ValueError, TypeError, RuntimeError)):
        await run_data_job(
            config(params={"mode": mode}, calibration=calibration()), output
        )
    assert not (output / "dataset").exists()
    assert not list(output.glob(".dataset-*"))


@pytest.mark.asyncio
async def test_training_projection_does_not_mutate_public_tasks(tmp_path):
    """A projection can format its row without rewriting the probed task."""
    snapshot = await run_data_job(
        config(params={"mode": "mutating_projection"}, calibration=calibration()),
        tmp_path / "job",
    )
    assert read_jsonl(snapshot.public_path)[0]["input"]["value"] == "1"
    assert snapshot.training_rows()[0]["prompt"] == "formatted training input"


@pytest.mark.asyncio
async def test_cleaning_can_reject_non_json_raw_references(tmp_path):
    """Dirty input can be cleaned before the retained-data serialization gate."""
    snapshot = await run_data_job(
        config(params={"mode": "dirty_reference"}), tmp_path / "job"
    )
    assert len(snapshot.training_rows()) == 2
    assert (
        json.loads((snapshot.root / "report.json").read_text())["rejected_by_recipe"]
        == 1
    )


@pytest.mark.asyncio
async def test_partial_snapshot_write_failure_cleans_staging_and_marks_job_failed(
    tmp_path, monkeypatch
):
    """An I/O failure after earlier files were written must not publish a snapshot."""
    from areal.v2.data_engineering import snapshot as snapshot_module

    original = snapshot_module.write_file

    def fail_training_file(path, value):
        if path.name == "train.jsonl":
            raise OSError("simulated storage failure")
        original(path, value)

    monkeypatch.setattr(snapshot_module, "write_file", fail_training_file)
    output = tmp_path / "job"
    with pytest.raises(OSError, match="storage failure"):
        await run_data_job(config(), output)
    assert not (output / "dataset").exists()
    assert not list(output.glob(".dataset-*"))
    assert json.loads((output / "job.json").read_text())["status"] == "failed"


@pytest.mark.asyncio
async def test_multiple_batches_are_bounded_and_keep_progress(tmp_path):
    """Only declared shortage retries generation; the pool accumulates by batch."""
    snapshot = await run_data_job(
        config(params={"mode": "shortage"}, max_batches=2), tmp_path / "success"
    )
    assert [row["task_id"] for row in read_jsonl(snapshot.public_path)] == [
        "batch-0",
        "batch-1",
    ]
    with pytest.raises(InsufficientDataError):
        await run_data_job(config(params={"mode": "shortage"}), tmp_path / "shortage")
    with pytest.raises(InsufficientDataError, match="Candidate budget"):
        await run_data_job(config(max_candidates=2), tmp_path / "budget")
    assert not (tmp_path / "budget/dataset").exists()


@pytest.mark.asyncio
async def test_source_changes_and_cancellation_leave_explicit_terminal_state(tmp_path):
    """A data reference never claims unchanged inputs when a recipe rewrites them."""
    source = tmp_path / "source.txt"
    source.write_text("original")
    with pytest.raises(ValueError, match="sources changed"):
        await run_data_job(
            config(params={"mode": "change_source"}, inputs={"source": str(source)}),
            tmp_path / "changed",
        )
    with pytest.raises(asyncio.CancelledError):
        await run_data_job(
            config(calibration=calibration("cancel")), tmp_path / "cancelled"
        )
    assert (
        json.loads((tmp_path / "cancelled/job.json").read_text())["status"]
        == "cancelled"
    )


@pytest.mark.asyncio
async def test_snapshot_tampering_and_wrong_reference_are_rejected(tmp_path):
    """Both file content and the caller's expected dataset identity are checked."""
    snapshot = await run_data_job(config(), tmp_path / "job")
    with pytest.raises(ValueError, match="requested dataset identity"):
        DatasetSnapshot(snapshot.root, "sha256:wrong").training_rows()
    snapshot.training_path.write_text("{}\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_snapshot(snapshot.root)


@pytest.mark.asyncio
async def test_training_view_exports_to_existing_dataset_loader(tmp_path):
    """A data artifact can be consumed by AReaL's current HF fallback loader."""
    from areal.dataset import get_custom_dataset

    snapshot = await run_data_job(config(), tmp_path / "job")
    path = export_hf_dataset(snapshot, tmp_path / "hf-data")
    loaded = get_custom_dataset(path=str(path), type="rl")
    assert loaded.to_list() == snapshot.training_rows()
    with pytest.raises(FileExistsError):
        export_hf_dataset(snapshot, path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"max_batches": 0},
        {"target_count": 0},
        {"purpose": "unknown"},
        {"exclude": "path"},
    ],
)
def test_invalid_job_config_is_rejected(kwargs):
    """Malformed generic settings fail before recipe execution."""
    with pytest.raises(ValueError):
        replace(config(), **kwargs)

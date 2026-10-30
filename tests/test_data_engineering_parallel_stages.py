# SPDX-License-Identifier: Apache-2.0

import json
import os
from collections import Counter
from dataclasses import replace

import pytest

from tests.test_data_engineering_execution import calls

from areal.v2.data_engineering import (
    CalibrationConfig,
    DataJobConfig,
    ExecutionConfig,
    GenerationShard,
    InsufficientDataError,
    RankedCandidate,
    SelectionPolicy,
    run_data_job,
    select_ranked_candidates,
)
from areal.v2.data_engineering.snapshot import read_jsonl


def config(directory, workers=1, backend="thread", **params):
    return DataJobConfig(
        "tests.test_data_engineering_execution:create_sharded_recipe",
        8,
        params={"directory": str(directory), "count": 16, **params},
        calibration=CalibrationConfig(
            "tests.test_data_engineering_execution:create_calibrator",
            params={"directory": str(directory), "fail_task": "absent"},
        ),
        execution=ExecutionConfig(max_workers=workers, batch_size=1, backend=backend),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["thread", "process"])
async def test_generation_scoring_and_ranking_use_bounded_parallel_workers(
    tmp_path, backend
):
    """Actual workers overlap in every added stage; quotas merge globally."""
    cfg = config(tmp_path, workers=2, backend=backend, overlap_workers=2)
    snapshot = await run_data_job(cfg, tmp_path / "job")
    assert [row["task_id"] for row in snapshot.training_rows()] == [
        str(i) for i in range(15, 7, -1)
    ]
    for phase in ("generation", "score", "rank"):
        active = peak = 0
        pids = set()
        for call in calls(tmp_path):
            if call["phase"] == phase + ".start":
                active += 1
                peak = max(peak, active)
                pids.add(call["pid"])
            elif call["phase"] == phase + ".end":
                active -= 1
        assert peak == 2, (phase, peak)
        if backend == "process":
            assert len(pids) == 2 and os.getpid() not in pids


@pytest.mark.asyncio
async def test_worker_count_does_not_change_generated_scored_or_selected_content(
    tmp_path,
):
    """Scheduling cannot alter candidate identities, scores or the global decision."""
    cfg = config(tmp_path)
    serial = await run_data_job(cfg, tmp_path / "serial")
    parallel = await run_data_job(
        replace(cfg, execution=replace(cfg.execution, max_workers=4)),
        tmp_path / "parallel",
    )
    for name in (
        "public.jsonl",
        "references.jsonl",
        "probes.jsonl",
        "train.jsonl",
        "report.json",
    ):
        assert (serial.root / name).read_bytes() == (parallel.root / name).read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["generation", "score", "rank"])
async def test_failed_parallel_stage_resumes_without_repeating_committed_work(
    tmp_path, phase
):
    """Generation files, raw model responses and ranks survive independent failures."""
    cfg = config(tmp_path, fail_phase=phase)
    (tmp_path / "fail").touch()
    with pytest.raises(OSError, match="Injected"):
        await run_data_job(cfg, tmp_path / "job")
    before = calls(tmp_path)
    (tmp_path / "fail").unlink()
    snapshot = await run_data_job(cfg, tmp_path / "job", resume=True)
    counts = Counter((call["phase"], call["task_id"]) for call in calls(tmp_path))
    assert counts[phase + ".start", "0"] == (2 if phase == "score" else 1)
    assert counts["generation.start", "0"] == 1
    if phase == "score":
        assert counts["calibration.start", "3"] == 1
    if phase == "rank":
        assert [c for c in calls(tmp_path) if c["phase"] == "calibration.start"] == [
            c for c in before if c["phase"] == "calibration.start"
        ]
    fresh = await run_data_job(cfg, tmp_path / "fresh")
    assert fresh.dataset_id == snapshot.dataset_id


@pytest.mark.asyncio
async def test_generation_file_survives_database_commit_failure(tmp_path, monkeypatch):
    """A complete shard can be reused even if its first DB commit never completed."""
    from areal.v2.data_engineering.storage import WorkStore

    original = WorkStore.retain

    def fail(*args, **kwargs):
        raise OSError("Checkpoint storage failed")

    cfg = config(tmp_path)
    monkeypatch.setattr(WorkStore, "retain", fail)
    with pytest.raises(OSError):
        await run_data_job(cfg, tmp_path / "job")
    assert (tmp_path / "job/work/generation-0-0.json").exists()
    monkeypatch.setattr(WorkStore, "retain", original)
    await run_data_job(cfg, tmp_path / "job", resume=True)
    counts = Counter((call["phase"], call["task_id"]) for call in calls(tmp_path))
    assert counts["generation.start", "0"] == 1


@pytest.mark.asyncio
async def test_global_tir_selection_matches_direct_recipe_for_all_worker_counts(
    tmp_path,
):
    """The TIR demo exercises sharded generation and preserves its full policy."""
    from examples.data_engineering.tir.data_recipe import Family, TirDataRecipe

    from areal.v2.data_engineering import RecipeContext

    recipe = TirDataRecipe(candidate_count=256, maximum_per_recipe=8)
    context = RecipeContext(42, 64, "joint", {})
    samples = recipe.generate(context)
    expected = recipe.select(samples, None, context)
    cfg = DataJobConfig(
        "examples.data_engineering.tir.data_recipe:create_recipe",
        64,
        seed=42,
        params={"candidate_count": 256, "maximum_per_recipe": 8},
        execution=ExecutionConfig(max_workers=4, batch_size=8, backend="process"),
    )
    snapshot = await run_data_job(cfg, tmp_path / "job")
    assert [row["task_id"] for row in read_jsonl(snapshot.public_path)] == list(
        expected.task_ids
    )
    stats = json.loads((snapshot.root / "report.json").read_text())["selection"]
    assert set(stats["families"]) == {family.value for family in Family}
    assert min(stats["families"].values()) >= 4
    assert max(stats["recipes"].values()) <= 8


def test_global_merge_rechecks_quotas_and_caps_after_local_ranking():
    """Global fill cannot reuse a candidate or bypass bucket caps."""
    values = [
        RankedCandidate(str(i), i, str(i), "a" if i % 2 == 0 else "b", str(i // 2))
        for i in range(12)
    ]

    def ordered(group):
        return (c for c in values if group is None or c.group == group)

    selected = select_ranked_candidates(
        SelectionPolicy({"a": 2, "b": 2}, {str(i): 1 for i in range(6)}), 4, ordered
    )
    assert len({c.task_id for c in selected}) == 4
    assert Counter(c.group for c in selected) == {"a": 2, "b": 2}
    assert max(Counter(c.bucket for c in selected).values()) == 1
    with pytest.raises(InsufficientDataError):
        select_ranked_candidates(SelectionPolicy({"missing": 1}), 4, ordered)


@pytest.mark.parametrize("start, stop", [(-1, 2), (True, 2), (2, 2), (2, 1)])
def test_invalid_generation_ranges_fail_before_execution(start, stop):
    """A shard plan cannot reserve an empty, negative or boolean range."""
    with pytest.raises(ValueError):
        GenerationShard(start, stop)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["gap", "budget", "overflow", "rank_id"])
async def test_invalid_parallel_outputs_never_publish(tmp_path, monkeypatch, mode):
    """Shard budgets, partitioning and ranked identities remain executor invariants."""
    from tests.test_data_engineering_execution import ShardedRankRecipe

    cfg = config(tmp_path)
    if mode == "gap":
        monkeypatch.setattr(
            ShardedRankRecipe,
            "generation_shards",
            lambda *args: [GenerationShard(1, 2)],
        )
    elif mode == "budget":
        cfg = replace(cfg, max_candidates=8)
    elif mode == "overflow":
        original = ShardedRankRecipe.generate_shard

        def overflow(*args):
            yield from original(*args)
            yield from original(*args)

        monkeypatch.setattr(ShardedRankRecipe, "generate_shard", overflow)
    else:
        monkeypatch.setattr(
            ShardedRankRecipe, "rank", lambda *args: RankedCandidate("unknown", 0, "")
        )
    with pytest.raises((ValueError, InsufficientDataError)):
        await run_data_job(cfg, tmp_path / "job")
    assert not (tmp_path / "job/dataset").exists()


@pytest.mark.asyncio
async def test_corrupt_generated_file_is_not_silently_regenerated(
    tmp_path, monkeypatch
):
    """Resume checks complete generation files before accepting their contents."""
    from areal.v2.data_engineering.storage import WorkStore

    original = WorkStore.retain

    def fail(*args, **kwargs):
        raise OSError("Checkpoint storage failed")

    cfg = config(tmp_path)
    monkeypatch.setattr(WorkStore, "retain", fail)
    with pytest.raises(OSError):
        await run_data_job(cfg, tmp_path / "job")
    monkeypatch.setattr(WorkStore, "retain", original)
    path = tmp_path / "job/work/generation-0-0.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checkpoint identity mismatch"):
        await run_data_job(cfg, tmp_path / "job", resume=True)
    assert not (tmp_path / "job/dataset").exists()

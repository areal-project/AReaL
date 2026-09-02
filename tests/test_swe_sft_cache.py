import importlib.util
import json
import logging
import os
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from datasets import Dataset


def _load_swe_sft_module():
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "areal" or name.startswith("areal.")
    }
    for name in list(sys.modules):
        if name == "areal" or name.startswith("areal."):
            del sys.modules[name]

    areal_module = types.ModuleType("areal")
    dataset_module = types.ModuleType("areal.dataset")
    dataset_module.__path__ = []
    swe_package = types.ModuleType("areal.dataset.swe_sft")
    swe_package.__path__ = []
    utils_module = types.ModuleType("areal.utils")
    utils_module.logging = logging
    areal_module.dataset = dataset_module
    areal_module.utils = utils_module
    sys.modules["areal"] = areal_module
    sys.modules["areal.dataset"] = dataset_module
    sys.modules["areal.dataset.swe_sft"] = swe_package
    sys.modules["areal.utils"] = utils_module

    package_path = Path(__file__).parents[1] / "areal" / "dataset" / "swe_sft"
    try:
        for name in ("messages", "tokenization", "pipeline"):
            full_name = f"areal.dataset.swe_sft.{name}"
            spec = importlib.util.spec_from_file_location(
                full_name, package_path / f"{name}.py"
            )
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
    finally:
        for name in list(sys.modules):
            if name == "areal" or name.startswith("areal."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
    return module


swe_sft = _load_swe_sft_module()


@pytest.fixture(autouse=True)
def _clear_cache_run_environment(monkeypatch):
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)
    monkeypatch.delenv("TORCHELASTIC_RESTART_COUNT", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_STEP_ID", raising=False)
    monkeypatch.delenv("SLURM_RESTART_COUNT", raising=False)


class _FakeTokenizer:
    name_or_path = "fake/tokenizer"
    vocab_size = 3
    special_tokens_map = {"eos_token": "</s>"}

    def __init__(self, chat_template="template-v1", backend_state=None):
        self.chat_template = chat_template
        if backend_state is not None:
            self.backend_tokenizer = types.SimpleNamespace(to_str=lambda: backend_state)

    def get_vocab(self):
        return {"a": 0, "b": 1, "</s>": 2}


def _cache_process_kwargs(max_length=2):
    return {
        "max_length": max_length,
        "num_proc": None,
        "pre_split": False,
        "filter_errors": True,
        "strip_all_thinking": False,
        "filter_empty_tool_calls": False,
        "filter_bare_text_tool_calls": False,
        "no_tools": False,
        "split_mode": "pair",
        "dump_dir": None,
        "dump_n_samples": 0,
        "parse_tool_call_args": False,
    }


def _write_cache(
    cache_dir,
    input_ids,
    max_length=2,
    path="unused.jsonl",
    tokenizer=None,
):
    if tokenizer is None:
        tokenizer = object()
    meta = swe_sft._build_cache_metadata(
        str(path), tokenizer, _cache_process_kwargs(max_length)
    )
    cache_key = swe_sft._json_digest(meta)
    entry_dir = Path(swe_sft._cache_entry_path(str(cache_dir), cache_key))
    entry_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_dict(
        {
            "input_ids": input_ids,
            "loss_mask": [[1] * len(ids) for ids in input_ids],
        }
    )
    dataset.save_to_disk(str(entry_dir))
    (entry_dir / ".meta.json").write_text(json.dumps(meta, sort_keys=True))
    (entry_dir / ".done").write_text(str(len(dataset)))
    return entry_dir


def _entry_for(cache_dir, path="unused.jsonl", tokenizer=None, max_length=2):
    if tokenizer is None:
        tokenizer = object()
    meta = swe_sft._build_cache_metadata(
        str(path), tokenizer, _cache_process_kwargs(max_length)
    )
    return Path(swe_sft._cache_entry_path(str(cache_dir), swe_sft._json_digest(meta)))


def _coord_prefix_for(cache_dir, path="unused.jsonl", tokenizer=None, max_length=2):
    if tokenizer is None:
        tokenizer = object()
    meta = swe_sft._build_cache_metadata(
        str(path), tokenizer, _cache_process_kwargs(max_length)
    )
    return swe_sft._cache_coordination_prefix(
        str(cache_dir), swe_sft._json_digest(meta)
    )


def test_get_swe_sft_dataset_loads_distributed_cache(tmp_path, monkeypatch):
    cache_dir = tmp_path / "processed_dataset"
    _write_cache(cache_dir, [[1, 2], [1, 2, 3]])
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    dataset = swe_sft.get_swe_sft_dataset(
        "unused.jsonl",
        tokenizer=object(),
        cache_dir=str(cache_dir),
        max_length=2,
    )

    assert len(dataset) == 1
    assert dataset[0]["input_ids"] == [1, 2]


def test_write_swe_sft_artifact_marker_publishes_stable_contract(tmp_path):
    """Test that CLI artifacts carry a stable loader-dispatch marker."""
    artifact_path = tmp_path / "tokenized"
    artifact_path.mkdir()

    swe_sft._write_swe_sft_artifact_marker(str(artifact_path))

    metadata = json.loads(
        (artifact_path / ".areal_swe_sft.json").read_text(encoding="utf-8")
    )
    assert metadata == {"format": "areal.swe_sft.pretokenized", "version": 1}


def test_source_identity_detects_rewrite_with_preserved_size_and_mtime(tmp_path):
    """Test that cache identity hashes bytes beyond stat metadata."""
    source = tmp_path / "input.jsonl"
    source.write_bytes(b"abcdef")
    original_stat = source.stat()
    before = swe_sft._source_identity(str(source))

    source.write_bytes(b"abcxef")
    os.utime(
        source,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    after = swe_sft._source_identity(str(source))

    assert before["size"] == after["size"]
    assert before["mtime_ns"] == after["mtime_ns"]
    assert before["digest"] != after["digest"]


def test_tokenizer_identity_detects_backend_pipeline_change():
    """Test that normalizer/pre-tokenizer backend state invalidates caches."""
    before = swe_sft._tokenizer_identity(
        _FakeTokenizer(backend_state='{"normalizer":"v1"}')
    )
    after = swe_sft._tokenizer_identity(
        _FakeTokenizer(backend_state='{"normalizer":"v2"}')
    )

    assert before["backend_tokenizer_digest"] != after["backend_tokenizer_digest"]


def test_cache_attempt_id_uses_distributed_launcher_environment(monkeypatch):
    """Test direct SPMD launchers get a shared attempt namespace."""
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "elastic-run-7")
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "2")
    monkeypatch.setenv("SLURM_JOB_ID", "1234")

    assert (
        swe_sft._resolve_cache_attempt_id(None)
        == "TORCHELASTIC_RUN_ID:elastic-run-7:restart:2"
    )
    monkeypatch.delenv("TORCHELASTIC_RUN_ID")
    monkeypatch.setenv("SLURM_STEP_ID", "batch-9")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "3")
    assert (
        swe_sft._resolve_cache_attempt_id(None)
        == "SLURM_JOB_ID:1234:step:batch-9:restart:3"
    )
    assert swe_sft._resolve_cache_attempt_id("explicit") == "explicit"


def test_get_swe_sft_dataset_rebuilds_cache_filtered_to_empty(tmp_path, monkeypatch):
    cache_dir = tmp_path / "processed_dataset"
    _write_cache(cache_dir, [[1, 2, 3]])
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    def fake_process_swe_sft(*args, **kwargs):
        return Dataset.from_dict({"input_ids": [[1]], "loss_mask": [[1]]})

    monkeypatch.setattr(swe_sft, "_process_swe_sft", fake_process_swe_sft)

    dataset = swe_sft.get_swe_sft_dataset(
        "unused.jsonl",
        tokenizer=object(),
        cache_dir=str(cache_dir),
        max_length=2,
    )

    assert len(dataset) == 1
    assert (_entry_for(cache_dir) / ".done").read_text() == "1"


def test_get_swe_sft_dataset_ignores_legacy_root_cache(tmp_path, monkeypatch):
    """Test that a legacy v2 root is left intact while a v3 entry is built."""
    cache_dir = tmp_path / "processed_dataset"
    Dataset.from_dict({"input_ids": [[99]], "loss_mask": [[1]]}).save_to_disk(
        str(cache_dir)
    )
    (cache_dir / ".meta.json").write_text('{"version":2}', encoding="utf-8")
    (cache_dir / ".done").write_text("1", encoding="utf-8")

    def fake_process_swe_sft(*args, **kwargs):
        return Dataset.from_dict({"input_ids": [[1]], "loss_mask": [[1]]})

    monkeypatch.setattr(swe_sft, "_process_swe_sft", fake_process_swe_sft)

    dataset = swe_sft.get_swe_sft_dataset(
        "unused.jsonl",
        tokenizer=object(),
        cache_dir=str(cache_dir),
        max_length=2,
        cache_rank=0,
        cache_world_size=2,
        cache_attempt_id="new-layout",
    )

    assert dataset[0]["input_ids"] == [1]
    assert (cache_dir / ".done").read_text(encoding="utf-8") == "1"
    assert (_entry_for(cache_dir) / ".done").exists()


def test_get_swe_sft_dataset_filters_dataset_with_indices_mapping(
    tmp_path, monkeypatch
):
    """Test that the max-length filter handles .filter() views (indices mapping)."""
    cache_dir = tmp_path / "processed_dataset"
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    def fake_process_swe_sft(*args, **kwargs):
        # Mimic _tokenize_samples: a .filter() view whose underlying arrow
        # table has more rows than the visible dataset.
        ds = Dataset.from_dict(
            {
                "input_ids": [[1], [], [1, 2], [], [1, 2, 3]],
                "loss_mask": [[1], [], [1, 1], [], [1, 1, 1]],
            }
        )
        return ds.filter(lambda x: len(x["input_ids"]) > 0)

    monkeypatch.setattr(swe_sft, "_process_swe_sft", fake_process_swe_sft)

    dataset = swe_sft.get_swe_sft_dataset(
        "unused.jsonl",
        tokenizer=object(),
        cache_dir=str(cache_dir),
        max_length=2,
    )

    # 3 non-empty rows built, the len-3 row is filtered by max_length=2.
    assert len(dataset) == 2
    assert dataset[0]["input_ids"] == [1]
    assert dataset[1]["input_ids"] == [1, 2]


def test_get_swe_sft_dataset_refuses_to_cache_empty_processed_dataset(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "processed_dataset"
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    def fake_process_swe_sft(*args, **kwargs):
        return Dataset.from_dict({"input_ids": [], "loss_mask": []})

    monkeypatch.setattr(swe_sft, "_process_swe_sft", fake_process_swe_sft)

    with pytest.raises(RuntimeError, match="produced 0 samples"):
        swe_sft.get_swe_sft_dataset(
            "unused.jsonl",
            tokenizer=object(),
            cache_dir=str(cache_dir),
            max_length=2,
        )

    assert not (_entry_for(cache_dir) / ".done").exists()
    coordination_prefix = _coord_prefix_for(cache_dir)
    building_marker = swe_sft._building_marker_path(coordination_prefix, None)
    building = json.loads(Path(building_marker).read_text(encoding="utf-8"))
    failure = json.loads(
        Path(
            swe_sft._failed_marker_path(coordination_prefix, building["attempt_id"])
        ).read_text(encoding="utf-8")
    )
    assert failure["attempt_id"] == building["attempt_id"]
    assert failure["error_type"] == "RuntimeError"


def test_get_swe_sft_dataset_worker_raises_rank0_failure_without_timeout(
    tmp_path, monkeypatch
):
    """Test that a waiter observes the matching failed build attempt promptly."""
    cache_dir = tmp_path / "processed_dataset"
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    tokenizer = _FakeTokenizer()
    cache_meta = swe_sft._build_cache_metadata(
        str(source), tokenizer, _cache_process_kwargs()
    )
    cache_key = swe_sft._json_digest(cache_meta)
    coordination_prefix = swe_sft._cache_coordination_prefix(str(cache_dir), cache_key)
    building_marker = Path(swe_sft._building_marker_path(coordination_prefix, None))
    failed_marker = Path(
        swe_sft._failed_marker_path(coordination_prefix, "attempt-from-rank0")
    )

    monkeypatch.setattr(swe_sft, "_RANK0_CACHE_TIMEOUT", 5)
    monkeypatch.setattr(swe_sft, "_RANK0_CACHE_POLL_INTERVAL", 0.02)

    def publish_failure():
        attempt_id = "attempt-from-rank0"
        swe_sft._atomic_write_json(
            str(building_marker),
            {"attempt_id": attempt_id, "cache_key": cache_key},
        )
        swe_sft._atomic_write_json(
            str(failed_marker),
            {
                "attempt_id": attempt_id,
                "cache_key": cache_key,
                "error_type": "ValueError",
                "error": "bad trajectory",
            },
        )

    writer = threading.Timer(0.05, publish_failure)
    writer.start()
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="ValueError: bad trajectory"):
            swe_sft.get_swe_sft_dataset(
                str(source),
                tokenizer=tokenizer,
                cache_dir=str(cache_dir),
                max_length=2,
                cache_rank=1,
                cache_world_size=2,
            )
    finally:
        writer.join()

    assert time.monotonic() - started < 1


def test_get_swe_sft_dataset_worker_ignores_stale_failure_for_new_attempt(
    tmp_path, monkeypatch
):
    """Test that a nonzero worker does not fail on the previous run's markers."""
    cache_dir = tmp_path / "processed_dataset"
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    tokenizer = _FakeTokenizer()
    cache_meta = swe_sft._build_cache_metadata(
        str(source), tokenizer, _cache_process_kwargs()
    )
    cache_key = swe_sft._json_digest(cache_meta)
    coordination_prefix = swe_sft._cache_coordination_prefix(str(cache_dir), cache_key)
    building_marker = Path(swe_sft._building_marker_path(coordination_prefix, None))
    failed_marker = Path(
        swe_sft._failed_marker_path(coordination_prefix, "stale-attempt")
    )
    swe_sft._atomic_write_json(
        str(building_marker),
        {"attempt_id": "stale-attempt", "cache_key": cache_key},
    )
    swe_sft._atomic_write_json(
        str(failed_marker),
        {
            "attempt_id": "stale-attempt",
            "cache_key": cache_key,
            "error_type": "ValueError",
            "error": "failure from previous launcher run",
        },
    )

    monkeypatch.setattr(swe_sft, "_RANK0_CACHE_TIMEOUT", 5)
    monkeypatch.setattr(swe_sft, "_RANK0_CACHE_POLL_INTERVAL", 0.02)
    monkeypatch.setattr(swe_sft, "_RANK0_STALE_FAILURE_GRACE", 1)

    def publish_new_attempt():
        swe_sft._atomic_write_json(
            str(building_marker),
            {"attempt_id": "current-attempt", "cache_key": cache_key},
        )
        _write_cache(
            cache_dir,
            [[1, 2]],
            path=source,
            tokenizer=tokenizer,
        )

    writer = threading.Timer(0.05, publish_new_attempt)
    writer.start()
    try:
        dataset = swe_sft.get_swe_sft_dataset(
            str(source),
            tokenizer=tokenizer,
            cache_dir=str(cache_dir),
            max_length=2,
            cache_rank=1,
            cache_world_size=2,
        )
    finally:
        writer.join()

    assert dataset[0]["input_ids"] == [1, 2]


def test_get_swe_sft_dataset_explicit_attempt_ignores_other_launcher_failure(
    tmp_path, monkeypatch
):
    """Test that an explicit launch ID never consumes another launcher's failure."""
    cache_dir = tmp_path / "processed_dataset"
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    tokenizer = _FakeTokenizer()
    cache_meta = swe_sft._build_cache_metadata(
        str(source), tokenizer, _cache_process_kwargs()
    )
    cache_key = swe_sft._json_digest(cache_meta)
    coordination_prefix = swe_sft._cache_coordination_prefix(str(cache_dir), cache_key)
    swe_sft._atomic_write_json(
        swe_sft._building_marker_path(coordination_prefix, "launcher-b"),
        {"attempt_id": "launcher-b", "cache_key": cache_key},
    )
    swe_sft._atomic_write_json(
        swe_sft._failed_marker_path(coordination_prefix, "launcher-b"),
        {
            "attempt_id": "launcher-b",
            "cache_key": cache_key,
            "error_type": "ValueError",
            "error": "launcher b failed",
        },
    )
    monkeypatch.setattr(swe_sft, "_RANK0_CACHE_TIMEOUT", 5)
    monkeypatch.setattr(swe_sft, "_RANK0_CACHE_POLL_INTERVAL", 0.02)

    writer = threading.Timer(
        0.05,
        _write_cache,
        args=(cache_dir, [[1, 2]]),
        kwargs={"path": source, "tokenizer": tokenizer},
    )
    writer.start()
    try:
        dataset = swe_sft.get_swe_sft_dataset(
            str(source),
            tokenizer=tokenizer,
            cache_dir=str(cache_dir),
            max_length=2,
            cache_rank=1,
            cache_world_size=2,
            cache_attempt_id="launcher-a",
        )
    finally:
        writer.join()

    assert dataset[0]["input_ids"] == [1, 2]


def test_get_swe_sft_dataset_cleanup_preserves_newer_attempt_markers(
    tmp_path, monkeypatch
):
    """Test that a successful builder only removes its own coordination state."""
    cache_dir = tmp_path / "processed_dataset"
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    tokenizer = _FakeTokenizer()
    cache_meta = swe_sft._build_cache_metadata(
        str(source), tokenizer, _cache_process_kwargs()
    )
    cache_key = swe_sft._json_digest(cache_meta)
    coordination_prefix = swe_sft._cache_coordination_prefix(str(cache_dir), cache_key)
    newer_failed_marker = swe_sft._failed_marker_path(coordination_prefix, "launcher-b")
    newer_building_marker = swe_sft._building_marker_path(
        coordination_prefix, "launcher-b"
    )

    def fake_process_swe_sft(*args, **kwargs):
        swe_sft._atomic_write_json(
            newer_building_marker,
            {"attempt_id": "launcher-b", "cache_key": cache_key},
        )
        swe_sft._atomic_write_json(
            newer_failed_marker,
            {
                "attempt_id": "launcher-b",
                "cache_key": cache_key,
                "error_type": "ValueError",
                "error": "launcher b failure",
            },
        )
        return Dataset.from_dict({"input_ids": [[1]], "loss_mask": [[1]]})

    monkeypatch.setattr(swe_sft, "_process_swe_sft", fake_process_swe_sft)

    swe_sft.get_swe_sft_dataset(
        str(source),
        tokenizer=tokenizer,
        cache_dir=str(cache_dir),
        max_length=2,
        cache_rank=0,
        cache_world_size=2,
        cache_attempt_id="launcher-a",
    )

    building = json.loads(Path(newer_building_marker).read_text(encoding="utf-8"))
    newer_failure = json.loads(Path(newer_failed_marker).read_text(encoding="utf-8"))
    assert building["attempt_id"] == "launcher-b"
    assert newer_failure["attempt_id"] == "launcher-b"


def test_get_swe_sft_dataset_concurrent_different_keys_publish_independent_entries(
    tmp_path, monkeypatch
):
    """Test concurrent settings never overwrite each other's immutable entry."""
    cache_dir = tmp_path / "processed_dataset"
    source_a = tmp_path / "input-a.jsonl"
    source_b = tmp_path / "input-b.jsonl"
    source_a.write_text('{"source":"a"}\n', encoding="utf-8")
    source_b.write_text('{"source":"b"}\n', encoding="utf-8")
    tokenizer = _FakeTokenizer()
    build_barrier = threading.Barrier(2)
    results = {}
    errors = []

    def fake_process_swe_sft(path, *args, **kwargs):
        build_barrier.wait(timeout=5)
        token = 11 if path == str(source_a) else 22
        return Dataset.from_dict({"input_ids": [[token]], "loss_mask": [[1]]})

    def build(label, source, attempt_id):
        try:
            results[label] = swe_sft.get_swe_sft_dataset(
                str(source),
                tokenizer=tokenizer,
                cache_dir=str(cache_dir),
                max_length=2,
                cache_rank=0,
                cache_world_size=2,
                cache_attempt_id=attempt_id,
            )
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(swe_sft, "_process_swe_sft", fake_process_swe_sft)
    threads = [
        threading.Thread(target=build, args=("a", source_a, "launcher-a")),
        threading.Thread(target=build, args=("b", source_b, "launcher-b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    entry_a = _entry_for(cache_dir, source_a, tokenizer)
    entry_b = _entry_for(cache_dir, source_b, tokenizer)
    assert entry_a != entry_b
    assert (entry_a / ".done").exists()
    assert (entry_b / ".done").exists()
    assert results["a"][0]["input_ids"] == [11]
    assert results["b"][0]["input_ids"] == [22]

    def refuse_rebuild(*args, **kwargs):
        raise AssertionError("late worker should load its content-addressed entry")

    monkeypatch.setattr(swe_sft, "_process_swe_sft", refuse_rebuild)
    late_a = swe_sft.get_swe_sft_dataset(
        str(source_a),
        tokenizer=tokenizer,
        cache_dir=str(cache_dir),
        max_length=2,
        cache_rank=1,
        cache_world_size=2,
        cache_attempt_id="late-launcher-a",
    )
    late_b = swe_sft.get_swe_sft_dataset(
        str(source_b),
        tokenizer=tokenizer,
        cache_dir=str(cache_dir),
        max_length=2,
        cache_rank=1,
        cache_world_size=2,
        cache_attempt_id="late-launcher-b",
    )
    assert late_a[0]["input_ids"] == [11]
    assert late_b[0]["input_ids"] == [22]


def test_get_swe_sft_dataset_rebuilds_when_source_content_changes(
    tmp_path, monkeypatch
):
    """Test that replacing a JSONL at the same path invalidates the cache."""
    cache_dir = tmp_path / "processed_dataset"
    source = tmp_path / "input.jsonl"
    source.write_text('{"value": 1}\n', encoding="utf-8")
    tokenizer = _FakeTokenizer()
    _write_cache(cache_dir, [[1]], path=source, tokenizer=tokenizer)
    source.write_text('{"value": 2}\n', encoding="utf-8")
    rebuilt = []

    def fake_process_swe_sft(*args, **kwargs):
        rebuilt.append(True)
        return Dataset.from_dict({"input_ids": [[2]], "loss_mask": [[1]]})

    monkeypatch.setattr(swe_sft, "_process_swe_sft", fake_process_swe_sft)

    dataset = swe_sft.get_swe_sft_dataset(
        str(source),
        tokenizer=tokenizer,
        cache_dir=str(cache_dir),
        max_length=2,
        cache_rank=0,
        cache_world_size=2,
    )

    assert rebuilt == [True]
    assert dataset[0]["input_ids"] == [2]


def test_get_swe_sft_dataset_rebuilds_when_chat_template_changes(tmp_path, monkeypatch):
    """Test that tokenizer template changes invalidate an existing cache."""
    cache_dir = tmp_path / "processed_dataset"
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    tokenizer = _FakeTokenizer("template-v1")
    _write_cache(cache_dir, [[1]], path=source, tokenizer=tokenizer)
    tokenizer.chat_template = "template-v2"
    rebuilt = []

    def fake_process_swe_sft(*args, **kwargs):
        rebuilt.append(True)
        return Dataset.from_dict({"input_ids": [[2]], "loss_mask": [[1]]})

    monkeypatch.setattr(swe_sft, "_process_swe_sft", fake_process_swe_sft)

    dataset = swe_sft.get_swe_sft_dataset(
        str(source),
        tokenizer=tokenizer,
        cache_dir=str(cache_dir),
        max_length=2,
        cache_rank=0,
        cache_world_size=2,
    )

    assert rebuilt == [True]
    assert dataset[0]["input_ids"] == [2]


def test_get_swe_sft_dataset_worker_loads_cache_written_by_rank0(tmp_path, monkeypatch):
    """Test that a non-rank-0 worker loads the cache once rank 0 publishes it."""
    cache_dir = tmp_path / "processed_dataset"
    # Explicit data-worker topology must win over conflicting SPMD env values.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr(swe_sft, "_RANK0_CACHE_TIMEOUT", 10)
    monkeypatch.setattr(swe_sft, "_RANK0_CACHE_POLL_INTERVAL", 0.05)

    writer = threading.Timer(0.2, _write_cache, args=(cache_dir, [[1, 2]]))
    writer.start()
    try:
        dataset = swe_sft.get_swe_sft_dataset(
            "unused.jsonl",
            tokenizer=object(),
            cache_dir=str(cache_dir),
            max_length=2,
            cache_rank=1,
            cache_world_size=2,
        )
    finally:
        writer.join()

    assert len(dataset) == 1
    assert dataset[0]["input_ids"] == [1, 2]


@pytest.mark.parametrize("skip_length_filter", [False, True])
def test_get_swe_sft_dataset_filters_unsupervised_pretokenized_rows(
    tmp_path,
    skip_length_filter,
):
    dataset_path = tmp_path / "pretokenized"
    Dataset.from_dict(
        {
            "input_ids": [[1, 2], [3, 4], [5]],
            "loss_mask": [[0, 0], [0, 1], []],
        }
    ).save_to_disk(str(dataset_path))

    dataset = swe_sft.get_swe_sft_dataset(
        str(dataset_path),
        max_length=8,
        skip_pretokenized_filter=skip_length_filter,
    )

    assert len(dataset) == 1
    assert dataset[0]["input_ids"] == [3, 4]
    assert dataset[0]["loss_mask"] == [0, 1]


def test_get_swe_sft_dataset_rejects_pretokenized_data_without_supervision(tmp_path):
    dataset_path = tmp_path / "pretokenized"
    Dataset.from_dict({"input_ids": [[1, 2]], "loss_mask": [[0, 0]]}).save_to_disk(
        str(dataset_path)
    )

    with pytest.raises(ValueError, match="no samples with a non-empty"):
        swe_sft.get_swe_sft_dataset(str(dataset_path))


def test_get_swe_sft_dataset_rejects_pretokenized_data_filtered_to_empty(tmp_path):
    """Test that max-length filtering cannot return an empty training dataset."""
    dataset_path = tmp_path / "pretokenized"
    Dataset.from_dict(
        {"input_ids": [[1, 2, 3]], "loss_mask": [[1, 1, 1]]}
    ).save_to_disk(str(dataset_path))

    with pytest.raises(ValueError, match="0 samples after max_length=2"):
        swe_sft.get_swe_sft_dataset(str(dataset_path), max_length=2)


def test_get_swe_sft_dataset_rejects_unknown_split_mode(tmp_path):
    """Test that public callers cannot silently fall back on a split typo."""
    dataset_path = tmp_path / "pretokenized"
    Dataset.from_dict({"input_ids": [[1]], "loss_mask": [[1]]}).save_to_disk(
        str(dataset_path)
    )

    with pytest.raises(ValueError, match="split_mode must be either"):
        swe_sft.get_swe_sft_dataset(str(dataset_path), split_mode="trajectroy")


def test_get_swe_sft_dataset_nonzero_rank_does_not_write_dump_without_cache(
    tmp_path, monkeypatch
):
    """Test that direct multi-rank preprocessing reserves dumps for rank 0."""
    observed_dump_dirs = []

    def fake_process_swe_sft(*args, **kwargs):
        observed_dump_dirs.append(kwargs["dump_dir"])
        return Dataset.from_dict({"input_ids": [[1]], "loss_mask": [[1]]})

    monkeypatch.setattr(swe_sft, "_process_swe_sft", fake_process_swe_sft)

    swe_sft.get_swe_sft_dataset(
        str(tmp_path / "input.jsonl"),
        tokenizer=object(),
        dump_dir=str(tmp_path / "dump"),
        cache_rank=1,
        cache_world_size=2,
    )

    assert observed_dump_dirs == [None]


@pytest.mark.parametrize(
    ("cache_rank", "cache_world_size"),
    [(1, None), (None, 2), (-1, 2), (2, 2), (0, 0)],
)
def test_get_swe_sft_dataset_rejects_invalid_explicit_cache_topology(
    cache_rank,
    cache_world_size,
):
    with pytest.raises(ValueError):
        swe_sft.get_swe_sft_dataset(
            "unused.jsonl",
            tokenizer=object(),
            cache_rank=cache_rank,
            cache_world_size=cache_world_size,
        )


def test_get_swe_sft_dataset_worker_rejects_mismatched_cache(tmp_path, monkeypatch):
    """Test that a worker times out instead of loading a cache built with other settings."""
    cache_dir = tmp_path / "processed_dataset"
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(swe_sft, "_RANK0_CACHE_TIMEOUT", 0.5)
    monkeypatch.setattr(swe_sft, "_RANK0_CACHE_POLL_INTERVAL", 0.05)

    # Cache published for max_length=99; this worker asks for max_length=2.
    writer = threading.Timer(
        0.1, _write_cache, args=(cache_dir, [[1, 2]]), kwargs={"max_length": 99}
    )
    writer.start()
    try:
        with pytest.raises(TimeoutError):
            swe_sft.get_swe_sft_dataset(
                "unused.jsonl",
                tokenizer=object(),
                cache_dir=str(cache_dir),
                max_length=2,
            )
    finally:
        writer.join()

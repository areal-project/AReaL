import importlib.util
import json
import logging
import sys
import threading
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


def _write_cache(cache_dir, input_ids, max_length=2):
    dataset = Dataset.from_dict(
        {
            "input_ids": input_ids,
            "loss_mask": [[1] * len(ids) for ids in input_ids],
        }
    )
    dataset.save_to_disk(str(cache_dir))
    meta = {
        "version": 1,
        "path": "unused.jsonl",
        "tokenizer": None,
        "process_kwargs": {
            "max_length": max_length,
            "num_proc": None,
            "pre_split": False,
            "filter_errors": True,
            "strip_all_thinking": False,
            "filter_empty_tool_calls": False,
            "filter_bare_text_tool_calls": False,
            "truncate_task_notifications": False,
            "no_tools": False,
            "max_no_thinking_ratio": None,
            "split_mode": "pair",
            "random_strip_thinking_prob": 0.0,
            "random_strip_thinking_seed": 42,
            "n_thinking_variants": 1,
            "parse_tool_call_args": False,
        },
    }
    (cache_dir / ".meta.json").write_text(json.dumps(meta, sort_keys=True))
    (cache_dir / ".done").write_text(str(len(dataset)))


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
    assert (cache_dir / ".done").read_text() == "1"


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

    assert not (cache_dir / ".done").exists()


def test_get_swe_sft_dataset_worker_loads_cache_written_by_rank0(tmp_path, monkeypatch):
    """Test that a non-rank-0 worker loads the cache once rank 0 publishes it."""
    cache_dir = tmp_path / "processed_dataset"
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
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
        )
    finally:
        writer.join()

    assert len(dataset) == 1
    assert dataset[0]["input_ids"] == [1, 2]


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

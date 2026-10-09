import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import pytest


def _load_dataset_module():
    """Load areal/dataset/__init__.py standalone, stubbing heavy dependencies."""
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "areal" or name.startswith("areal.")
    }
    for name in list(sys.modules):
        if name == "areal" or name.startswith("areal."):
            del sys.modules[name]

    areal_module = types.ModuleType("areal")
    api_module = types.ModuleType("areal.api")
    cli_args_module = types.ModuleType("areal.api.cli_args")
    cli_args_module._DatasetConfig = object
    utils_module = types.ModuleType("areal.utils")
    utils_module.logging = logging
    areal_module.api = api_module
    areal_module.utils = utils_module
    api_module.cli_args = cli_args_module
    sys.modules["areal"] = areal_module
    sys.modules["areal.api"] = api_module
    sys.modules["areal.api.cli_args"] = cli_args_module
    sys.modules["areal.utils"] = utils_module

    path = Path(__file__).parents[1] / "areal" / "dataset" / "__init__.py"
    spec = importlib.util.spec_from_file_location("areal.dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["areal.dataset"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name in list(sys.modules):
            if name == "areal" or name.startswith("areal."):
                del sys.modules[name]
        sys.modules.update(saved_modules)
    return module


_DATASET_MODULE = _load_dataset_module()
_SWE_PATH_PATTERN = _DATASET_MODULE._SWE_PATH_PATTERN


@pytest.mark.parametrize(
    "path",
    [
        "/storage/datasets/swe_data/sft_test_dataset.jsonl",
        "/data/swe-bench/train.jsonl",
        "/data/swe_sft/processed",
        "swe.jsonl",
        "/exp/my_swe.jsonl",
        "/Data/SWE_data/file.jsonl",
    ],
)
def test_swe_path_pattern_matches_swe_token_paths(path):
    """Test that _SWE_PATH_PATTERN matches when 'swe' is a delimited path token."""
    assert _SWE_PATH_PATTERN.search(path.lower())


@pytest.mark.parametrize(
    "path",
    [
        "/data/answer_sft/dataset",
        "/home/swetha/my_sft_data",
        "/data/sweep_results/train.jsonl",
        "/corpora/swedish_sft",
        "/data/answers.jsonl",
    ],
)
def test_swe_path_pattern_ignores_incidental_trigram(path):
    """Test that _SWE_PATH_PATTERN does not fire on paths merely containing 'swe'."""
    assert not _SWE_PATH_PATTERN.search(path.lower())


def test_pretokenized_artifact_marker_dispatches_arbitrary_directory(
    tmp_path, monkeypatch
):
    """Test that CLI artifacts do not depend on a 'swe' pathname token."""
    artifact_path = tmp_path / "tokenized_training_data"
    artifact_path.mkdir()
    (artifact_path / ".areal_swe_sft.json").write_text(
        json.dumps({"format": "areal.swe_sft.pretokenized", "version": 1}),
        encoding="utf-8",
    )
    captured = {}

    def fake_get_swe_sft_dataset(**kwargs):
        captured.update(kwargs)
        return "swe-dataset"

    swe_module = types.ModuleType("areal.dataset.swe_sft")
    swe_module.get_swe_sft_dataset = fake_get_swe_sft_dataset
    monkeypatch.setitem(sys.modules, "areal.dataset", _DATASET_MODULE)
    monkeypatch.setitem(sys.modules, "areal.dataset.swe_sft", swe_module)

    result = _DATASET_MODULE._get_custom_dataset(
        path=str(artifact_path),
        type="sft",
        tokenizer="tokenizer",
        data_worker_rank=1,
        data_worker_world_size=2,
        data_worker_cache_attempt_id="launch-123",
    )

    assert result == "swe-dataset"
    assert captured["path"] == str(artifact_path)
    assert captured["cache_rank"] == 1
    assert captured["cache_world_size"] == 2
    assert captured["cache_attempt_id"] == "launch-123"

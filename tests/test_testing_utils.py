# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

from areal.utils import testing_utils


def test_importing_test_utils_when_model_root_is_unavailable_does_not_download(
    tmp_path: Path,
):
    script = """
import os

import huggingface_hub

def unexpected_download(*args, **kwargs):
    raise AssertionError("import attempted to download a model")

huggingface_hub.snapshot_download = unexpected_download

from areal.utils import testing_utils

model_root = os.environ["TEST_MODEL_ROOT"]

def with_model_root(specs):
    return {
        key: (os.path.join(model_root, key), hf_id)
        for key, (_, hf_id) in specs.items()
    }

dense_specs = with_model_root(testing_utils._DENSE_MODEL_PATH_SPECS)
moe_specs = with_model_root(testing_utils._MOE_MODEL_PATH_SPECS)
testing_utils.DENSE_MODEL_PATHS = testing_utils._LazyModelPaths(dense_specs)
testing_utils.MOE_MODEL_PATHS = testing_utils._LazyModelPaths(moe_specs)
testing_utils.MODEL_PATHS = testing_utils._LazyModelPaths({**dense_specs, **moe_specs})

import tests.utils
import tests.test_megatron_engine_vlm_distributed
"""
    env = {
        **os.environ,
        "HF_HOME": str(tmp_path),
        "HF_HUB_OFFLINE": "1",
        "TEST_MODEL_ROOT": str(tmp_path / "models"),
        "TRANSFORMERS_OFFLINE": "1",
    }

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_model_paths_when_one_key_is_accessed_resolve_and_cache_only_that_key(
    monkeypatch,
    tmp_path: Path,
):
    calls: list[tuple[str, str]] = []
    resolved_root = tmp_path / "resolved"

    def resolve(local_path: str, hf_id: str) -> str:
        calls.append((local_path, hf_id))
        return str(resolved_root / hf_id)

    monkeypatch.setattr(testing_utils, "get_model_path", resolve)
    small_local_path = str(tmp_path / "models" / "small")
    large_local_path = str(tmp_path / "models" / "large")
    paths = testing_utils._LazyModelPaths(
        {
            "small": (small_local_path, "org/small"),
            "large": (large_local_path, "org/large"),
        }
    )

    keys = list(paths)
    path_count = len(paths)
    calls_before_access = list(calls)
    first_result = paths["small"]
    cached_result = paths["small"]

    assert keys == ["small", "large"]
    assert path_count == 2
    assert calls_before_access == []
    assert first_result == str(resolved_root / "org/small")
    assert cached_result == first_result
    assert calls == [(small_local_path, "org/small")]

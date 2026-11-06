# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys

from areal.v2.data_engineering import load_snapshot


def test_module_entrypoint_runs_real_tir_recipe_and_exports_training_data(tmp_path):
    """Normal package imports and python -m work on CPU without an inference service."""
    from datasets import load_from_disk

    config = tmp_path / "job.yaml"
    config.write_text(
        "recipe: examples.data_engineering.tir.data_recipe:create_recipe\npurpose: harness\nseed: 42\ntarget_count: 16\nparams:\n  candidate_count: 32\n"
    )
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "areal.v2.data_engineering",
            "--config",
            str(config),
            "--output",
            str(tmp_path / "job"),
            "--export-hf",
            str(tmp_path / "hf-data"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout.splitlines()[-1])
    snapshot = load_snapshot(result["dataset_path"])
    assert result["dataset_id"] == snapshot.dataset_id
    assert (
        load_from_disk(result["training_dataset_path"]).to_list()
        == snapshot.training_rows()
    )
    manifest = json.loads((snapshot.root / "manifest.json").read_text())
    assert manifest["provenance"]["config"]["purpose"] == "harness"
    assert manifest["provenance"]["calibration"] is None


def test_module_rejects_invalid_config_without_creating_a_job(tmp_path):
    """Malformed settings fail before any execution state or artifact is created."""
    config = tmp_path / "job.json"
    config.write_text('{"recipe":"bad","target_count":0}')
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "areal.v2.data_engineering",
            "--config",
            str(config),
            "--output",
            str(tmp_path / "job"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode != 0
    assert not (tmp_path / "job").exists()

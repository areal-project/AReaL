# SPDX-License-Identifier: Apache-2.0

"""Exercise submission boundaries without scheduling jobs or contacting Arena."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "examples/swe/submit_arena.sh"


@pytest.fixture
def launch_env(tmp_path, monkeypatch):
    repo = tmp_path / "shared checkout"
    script = repo / "examples/swe/submit_arena.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT, script)
    config = repo / "config.yaml"
    config.touch()
    image = repo / "image.sif"
    image.touch()
    credentials = repo / "private.env"
    credentials.write_text(
        "export ARENA_OPENAPI_TOKEN=secret-sentinel\n"
        "export ARENA_LLM_API_KEY=secret-sentinel\n"
        "export SWE_RL_ADMIN_API_KEY=secret-sentinel\n"
        "export ARENA_OPENAPI_BASE=https://arena.example\n"
        "export MODEL_PATH=/obsolete/model\n"
        "export ARENA_STREAM_ID=obsolete-stream\n"
    )
    capture = tmp_path / "capture.json"
    fake = tmp_path / "sbatch"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({\n"
        "'args': sys.argv[1:], 'model': os.environ['MODEL_PATH'],\n"
        "'stream': os.environ['ARENA_STREAM_ID'],\n"
        "'bind': os.environ.get('APPTAINER_BIND')}))\n"
    )
    fake.chmod(0o755)
    values = {
        "AREAL_DIR": str(repo),
        "CONFIG_PATH": str(config),
        "AREAL_IMAGE": str(image),
        "AREAL_PYTHON": str(fake),
        "MODEL_PATH": str(repo),
        "AREAL_FILEROOT": str(repo / "output"),
        "ARENA_CREDENTIALS_FILE": str(credentials),
        "AREAL_CONTROLLER_MOUNTS": str(repo),
        "AREAL_WORKER_MOUNTS": str(repo),
        "ARENA_STREAM_ID": "selected-stream",
        "TRIAL_NAME": "test-run",
        "AREAL_CONTAINER_BIN": str(fake),
        "CAPTURE": str(capture),
        "SLURM_JOB_ID": "already-in-allocation",
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return script, capture


@pytest.mark.parametrize("mode", [[], ["--check-arena"]])
def test_submit_inside_existing_allocation_creates_cpu_controller(launch_env, mode):
    script, capture = launch_env
    result = subprocess.run(
        ["bash", str(script), *mode], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())["args"]
    assert "--nodes=1" in args and "--gres=gpu:0" in args
    assert args[-2:] == [str(script), "--controller"]
    assert "secret-sentinel" not in result.stdout + result.stderr + capture.read_text()


def test_controller_from_spool_uses_exported_checkout(launch_env, tmp_path):
    script, capture = launch_env
    spool = tmp_path / "slurm-script"
    shutil.copyfile(script, spool)
    subprocess.run(["bash", str(spool), "--controller"], check=True)
    assert json.loads(capture.read_text())["args"][-2:] == [str(script), "--run"]


def test_controller_credentials_preserve_launch_overrides(launch_env):
    script, capture = launch_env
    result = subprocess.run(
        ["bash", str(script), "--run"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(capture.read_text())
    assert data["model"] == os.environ["MODEL_PATH"]
    assert data["stream"] == "selected-stream"
    assert data["bind"] == os.environ["AREAL_WORKER_MOUNTS"]
    assert "secret-sentinel" not in result.stdout + result.stderr + capture.read_text()

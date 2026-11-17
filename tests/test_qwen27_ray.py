# SPDX-License-Identifier: Apache-2.0
"""Test the Qwen Ray launch boundary without GPUs or a running Ray cluster."""

import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from examples.swe.rl.qwen38_27b.ray_train import prepare_ray_config

RECIPE = Path(__file__).parents[1] / "examples/swe/rl/qwen38_27b"


def _config_env(tmp_path):
    return {
        "QWEN_ENV_FILE": str(tmp_path / "private profile.env"),
        "QWEN_ARENA_TRIAL": "ray-test",
        "QWEN_EXPERIMENT_NAME": "qwen-test",
        "QWEN_RUN_DIR": str(tmp_path / "run"),
        "QWEN_OUTPUT_ROOT": str(tmp_path / "output"),
        "QWEN_RUN_USER": "tester",
        "QWEN_ARENA_STREAM_ID": "test-stream",
        "QWEN_ARENA_HARNESS": "test-harness@1",
        "QWEN_REWARD_KEY": "test-reward",
        "QWEN_REWARD_VERSION": "1",
    }


def test_ray_config_preserves_training_and_pins_role_startup(tmp_path):
    original = yaml.safe_load((RECIPE / "train_fine_released64_bs16.yaml").read_text())
    before = copy.deepcopy(original)
    repo = tmp_path / "source with spaces"
    env = _config_env(tmp_path)
    config = prepare_ray_config(original, repo, env)

    assert original == before
    assert config["scheduler"]["type"] == "ray"
    for key in ("gconfig", "train_dataset", "recover", "sglang", "allocation_mode"):
        assert config[key] == original[key]
    assert config["actor"]["optimizer"] == original["actor"]["optimizer"]
    assert config["actor"]["megatron"] == original["actor"]["megatron"]
    for role in ("actor", "rollout"):
        spec = config[role]["scheduling_spec"][0]
        assert shlex.split(spec["cmd"]) == [
            "bash",
            str(repo / "examples/swe/rl/qwen38_27b/ray_worker.sh"),
            role,
        ]
        assert (
            not {"image", "mount", "reservation", "nodelist", "additional_bash_cmds"}
            & spec.keys()
        )
        assert (
            not {"PYTHONPATH", "LD_LIBRARY_PATH", "CUDNN_PATH"}
            & spec["env_vars"].keys()
        )
        assert spec["env_vars"]["QWEN_ENV_FILE"] == env["QWEN_ENV_FILE"]
        assert spec["env_vars"]["QWEN_SUBMITTED_TRIAL"] == "ray-test"
    assert config["econfig"]["arena_streams"][0]["expected_reward_ref"] == {
        "key": "test-reward",
        "version": "1",
    }


def test_ray_config_selects_native_images_and_independent_interpreters(tmp_path):
    original = yaml.safe_load((RECIPE / "train_fine_released64_bs16.yaml").read_text())
    env = _config_env(tmp_path)
    env.update(
        QWEN_RAY_PYTHON="/controller/bin/python",
        QWEN_RAY_ACTOR_IMAGE="registry.example/train:fixed",
        QWEN_RAY_ROLLOUT_IMAGE="registry.example/rollout:fixed",
        QWEN_RAY_ACTOR_PYTHON="/opt/train/bin/python",
        QWEN_RAY_ROLLOUT_PYTHON="/usr/bin/python3",
    )
    config = prepare_ray_config(original, RECIPE.parents[3], env)
    for role in ("actor", "rollout"):
        spec = config[role]["scheduling_spec"][0]
        assert spec["ray_runtime_env"] == {
            "image_uri": env[f"QWEN_RAY_{role.upper()}_IMAGE"]
        }
        assert (
            spec["env_vars"]["QWEN_RAY_PYTHON"]
            == env[f"QWEN_RAY_{role.upper()}_PYTHON"]
        )
        assert spec["env_vars"]["TMS_INIT_ENABLE"] == "0"
        assert (
            spec["env_vars"]["PYTORCH_CUDA_ALLOC_CONF"]
            == original[role]["scheduling_spec"][0]["env_vars"][
                "PYTORCH_CUDA_ALLOC_CONF"
            ]
        )
    # A later profile edit cannot mutate the already-generated deployment.
    env["QWEN_RAY_ACTOR_IMAGE"] = "registry.example/train:changed"
    assert (
        config["actor"]["scheduling_spec"][0]["ray_runtime_env"]["image_uri"]
        == "registry.example/train:fixed"
    )
    assert "ray_runtime_env" not in original["actor"]["scheduling_spec"][0]


def test_ray_config_preserves_native_container_mounts_and_rejects_image_conflict(
    tmp_path,
):
    original = yaml.safe_load((RECIPE / "train_fine_released64_bs16.yaml").read_text())
    runtime = {
        "container": {
            "image": "registry.example/train:fixed",
            "run_options": ["--volume=/shared:/shared"],
        },
        "env_vars": {"PYTHONPATH": "/shared/source"},
    }
    original["actor"]["scheduling_spec"][0]["ray_runtime_env"] = runtime
    env = _config_env(tmp_path)
    result = prepare_ray_config(original, RECIPE.parents[3], env)
    spec = result["actor"]["scheduling_spec"][0]
    assert spec["ray_runtime_env"] == runtime
    assert spec["env_vars"]["QWEN_RAY_PYTHON"] == "python3"
    env["QWEN_RAY_ACTOR_IMAGE"] = "registry.example/other:tag"
    with pytest.raises(ValueError, match="choose either"):
        prepare_ray_config(original, RECIPE.parents[3], env)


@pytest.mark.parametrize(
    "change, match",
    [
        (
            {"allocation_mode": "megatron[actor]:d1p4t4c2e1|sglang[rollout]:d8t4"},
            "requires actor",
        ),
        ({"cluster": {"n_nodes": 1, "n_gpus_per_node": 8}}, "8 nodes"),
    ],
)
def test_ray_config_rejects_uncovered_bootstrap_topologies(tmp_path, change, match):
    config = yaml.safe_load((RECIPE / "train_fine_released64_bs16.yaml").read_text())
    config.update(change)
    with pytest.raises(ValueError, match=match):
        prepare_ray_config(config, RECIPE.parents[3], _config_env(tmp_path))


def test_ray_config_requires_explicit_reward_version(tmp_path):
    config = yaml.safe_load((RECIPE / "train_fine_released64_bs16.yaml").read_text())
    env = _config_env(tmp_path)
    del env["QWEN_REWARD_VERSION"]
    with pytest.raises(ValueError, match="Pin the Arena harness"):
        prepare_ray_config(config, RECIPE.parents[3], env)


@pytest.mark.parametrize(
    "role, field, value, match",
    [
        ("actor", "backend", "megatron:d2p4t4c2e1", "actor.backend"),
        ("rollout", "backend", "sglang:d8t8", "rollout.backend"),
        ("actor", "scheduling_strategy", {"type": "colocation"}, "actor separation"),
        (
            "rollout",
            "scheduling_strategy",
            {"type": "colocation", "target": "actor", "fork": False},
            "forked rollout",
        ),
    ],
)
def test_ray_config_rejects_actual_backend_changes_with_old_allocation(
    tmp_path, role, field, value, match
):
    config = yaml.safe_load((RECIPE / "train_fine_released64_bs16.yaml").read_text())
    config[role][field] = value
    with pytest.raises(ValueError, match=match):
        prepare_ray_config(config, RECIPE.parents[3], _config_env(tmp_path))


def _worker_environment(tmp_path):
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("QWEN_", "SLURM_", "SGLANG_"))
    }
    env.pop("BASH_ENV", None)
    values = _config_env(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    values.update(
        QWEN_MODEL=str(model),
        ARENA_OPENAPI_BASE="https://example.invalid",
        ARENA_OPENAPI_TOKEN="test-token",
        ARENA_LLM_API_KEY="test-key",
        WANDB_BASE_URL="https://example.invalid",
        WANDB_API_KEY="test-key",
        QWEN_RAY_ACTOR_PYTHONPATH="/actor-only",
        QWEN_RAY_ROLLOUT_PYTHONPATH="/rollout-only",
        CUDA_VISIBLE_DEVICES="profile-must-not-reassign-devices",
        SLURM_LOCALID="99",
        QWEN_RAY_PYTHON="/wrong/profile/python",
    )
    profile = Path(values["QWEN_ENV_FILE"])
    profile.write_text(
        "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items()) + "\n"
    )
    fake_python = tmp_path / "image python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        + "import json, os, sys\nwith open(os.environ['TEST_WORKER_LOG'], 'a') as f:\n json.dump({'argv': sys.argv[1:], 'env': dict(os.environ)}, f); f.write('\\n')\n"
    )
    fake_python.chmod(0o755)
    env.update(
        QWEN_ENV_FILE=str(profile),
        QWEN_RAY_PYTHON=str(fake_python),
        TEST_WORKER_LOG=str(tmp_path / "worker.jsonl"),
        PYTHONPATH="/wrong-controller-deps",
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        SLURM_LOCALID="1",
        SLURM_NODEID="3",
        SLURM_NNODES="8",
        SLURM_JOB_NAME="old-job:actor",
        SLURM_PROCID="99",
        RANK="99",
        QWEN_SUBMITTED_TRIAL="pinned-trial",
    )
    return env


@pytest.mark.parametrize(
    "role,module",
    [
        ("actor", "qwen27_actor_worker"),
        ("rollout", "areal.infra.rpc.rpc_server"),
    ],
)
def test_ray_worker_initializes_role_before_python_and_preserves_rpc_args(
    tmp_path, role, module
):
    env = _worker_environment(tmp_path)
    if role == "actor":
        env["CUDA_VISIBLE_DEVICES"] = "5"
    args = [
        "--role",
        role,
        "--worker-index",
        "3",
        "--host",
        "0.0.0.0",
        "--port",
        "0",
        "--fileroot",
        "path with spaces",
    ]
    result = subprocess.run(
        ["bash", str(RECIPE / "ray_worker.sh"), role, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    calls = [
        json.loads(line)
        for line in Path(env["TEST_WORKER_LOG"]).read_text().splitlines()
    ]
    assert calls[-1]["argv"] == ["-m", module, *args]
    actual = calls[-1]["env"]
    assert actual["PYTHONPATH"].startswith(f"/{role}-only:")
    assert "/wrong-controller-deps" not in actual["PYTHONPATH"]
    assert actual["QWEN_ARENA_TRIAL"] == "pinned-trial"
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "SLURM_LOCALID",
        "SLURM_NODEID",
        "SLURM_NNODES",
    ):
        assert actual[key] == env[key]
    assert (
        actual["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"]
        == "areal.engine.sglang_processors"
    )
    assert "SLURM_PROCID" not in actual and "RANK" not in actual
    assert actual["TMS_INIT_ENABLE"] == "0"
    if role == "rollout":
        assert len(calls) == 2
        assert calls[0]["argv"] == [str(RECIPE / "configure_sglang_logging.py")]
        assert actual["SGLANG_VLM_CACHE_SIZE_MB"] == "100"
        assert "pinned-trial/native/sglang-" in actual["SGLANG_LOGGING_CONFIG_PATH"]
    else:
        assert len(calls) == 1


def test_ray_controller_prepare_only_writes_config_without_cluster_or_secrets(tmp_path):
    """Exercise the shell-to-Python boundary using the real config generator."""
    env = _worker_environment(tmp_path)
    env["QWEN_RAY_PYTHON"] = sys.executable
    env["RAY_ADDRESS"] = "invalid-for-prepare-only"
    result = subprocess.run(
        ["bash", str(RECIPE / "ray_worker.sh"), "controller", "--prepare-only"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    config_path = tmp_path / "run/train-ray.yaml"
    text = config_path.read_text()
    assert "test-token" not in text and "test-key" not in text
    config = yaml.safe_load(text)
    assert config["scheduler"]["type"] == "ray"
    assert (
        config["actor"]["scheduling_spec"][0]["env_vars"]["QWEN_SUBMITTED_TRIAL"]
        == "pinned-trial"
    )
    assert config_path.stat().st_mode & 0o777 == 0o600
    # Preparation can be repeated, but cannot mutate an existing run's config.
    with Path(env["QWEN_ENV_FILE"]).open("a") as profile:
        profile.write("QWEN_REWARD_VERSION=changed\n")
    result = subprocess.run(
        ["bash", str(RECIPE / "ray_worker.sh"), "controller", "--prepare-only"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "Ray config changed" in result.stderr
    assert config_path.read_text() == text


@pytest.mark.parametrize("failure", ["role", "profile"])
def test_ray_worker_rejects_invalid_setup_before_python(tmp_path, failure):
    env = _worker_environment(tmp_path)
    if failure == "profile":
        Path(env["QWEN_ENV_FILE"]).write_text("exit 23\n")
    result = subprocess.run(
        [
            "bash",
            str(RECIPE / "ray_worker.sh"),
            "unknown" if failure == "role" else "rollout",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert not Path(env["TEST_WORKER_LOG"]).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["ray", "slurm"])
async def test_qwen_entry_selects_backend_and_preserves_training_rpc(
    monkeypatch, backend
):
    from examples.swe import train_swe_rl
    from examples.swe.qwen38_flash_next import train_rl
    from examples.swe.utils import SWEPPOConfig

    import areal
    from areal.api import cli_args
    from areal.infra.scheduler import slurm

    config = SWEPPOConfig()
    config.scheduler.type = backend
    captured = []
    original_call = AsyncMock()
    scheduler = SimpleNamespace(async_call_engine=original_call)
    monkeypatch.setattr(cli_args, "load_expr_config", lambda *_: (config, None))
    monkeypatch.setattr(
        train_swe_rl, "get_arena_mixture_dataset", lambda *_, **__: ([], [])
    )
    monkeypatch.setattr(
        train_rl, "prepare_arena_training_dataset", lambda dataset, *_: dataset
    )
    monkeypatch.delenv("QWEN_ARENA_TASK_IDS_FILE", raising=False)
    monkeypatch.delenv("QWEN_REQUIRE_RECOVER", raising=False)
    if backend == "slurm":
        monkeypatch.setenv("QWEN_MOUNTS", "/shared")
    else:
        monkeypatch.delenv("QWEN_MOUNTS", raising=False)

    def make_slurm(**kwargs):
        captured.append(("slurm", kwargs))
        return scheduler

    monkeypatch.setattr(slurm, "SlurmScheduler", make_slurm)

    class Trainer:
        def __init__(self, config, **kwargs):
            self.config = config
            self.recover_info = None
            self.scheduler = self._init_scheduler()

        def _init_scheduler(self):
            captured.append(("base", {}))
            return scheduler

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def train(self, **kwargs):
            pass

    # Avoid triggering the lazy trainer import (and external metric clients)
    # when this test replaces the trainer with an in-memory double.
    monkeypatch.setitem(areal.__dict__, "PPOTrainer", Trainer)
    train_rl.main("swe", [])
    assert captured == (
        [
            (
                "slurm",
                {
                    "exp_config": config,
                    "container_mounts": "/shared",
                    "startup_timeout": 86400,
                },
            )
        ]
        if backend == "slurm"
        else [("base", {})]
    )
    await scheduler.async_call_engine("actor/0", "ppo_update", "actor", data="batch")
    assert original_call.call_args.kwargs == {
        "data": "batch",
        "http_timeout": 8 * 3600,
        "max_retries": 1,
    }
    await scheduler.async_call_engine("actor/0", "other_method", "actor")
    assert original_call.call_args.kwargs == {}

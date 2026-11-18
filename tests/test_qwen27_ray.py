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

from examples.swe.rl.qwen38_27b.ray_train import (
    prepare_ray_config,
    ray_gpu_bundles,
    validate_ray_capacity,
)

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
            "actor.backend",
        ),
        ({"cluster": {"n_nodes": 1, "n_gpus_per_node": 8}}, "at least 8 nodes"),
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
    with pytest.raises(ValueError, match="Pin expected_reward_ref"):
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


@pytest.mark.parametrize(
    "actor,rollout,gpus_per_node,node_limit,bundles",
    [
        ("d1p2t4c2", "d4t4", 8, 2, [8, 8]),
        ("d1p4t4c4", "d16t4", 8, 8, [8] * 8),
        ("d2p4t4c4", "d32t4", 8, 16, [8] * 16),
        ("d3t4", "d3t4", 8, 4, [8, 4]),
        ("d2t2", "d2t2", 4, 1, [4]),
        ("d2", "d2", 2, 1, [2]),
    ],
)
def test_ray_topology_uses_backend_dimensions(
    tmp_path, actor, rollout, gpus_per_node, node_limit, bundles
):
    config = yaml.safe_load((RECIPE / "train_fine_released64_bs16.yaml").read_text())
    config.pop("allocation_mode")
    config["actor"]["backend"] = f"megatron:{actor}"
    config["rollout"]["backend"] = f"sglang:{rollout}"
    config["cluster"].update(n_nodes=node_limit, n_gpus_per_node=gpus_per_node)
    env = _config_env(tmp_path)
    env["QWEN_RAY_ROLLOUT_IMAGE"] = "registry.example/rollout:fixed"
    result = prepare_ray_config(config, RECIPE.parents[3], env)
    assert ray_gpu_bundles(result) == bundles
    nodes = [{"Alive": True, "Resources": {"GPU": gpus}} for gpus in bundles]
    validate_ray_capacity(result, nodes)
    nodes[-1]["Alive"] = False
    with pytest.raises(RuntimeError, match="GPU node bundles"):
        validate_ray_capacity(result, nodes)


def test_ray_capacity_rejects_fragmentation_even_with_enough_total_gpus(tmp_path):
    config = yaml.safe_load((RECIPE / "train_fine_released64_bs16.yaml").read_text())
    result = prepare_ray_config(config, RECIPE.parents[3], _config_env(tmp_path))
    with pytest.raises(RuntimeError, match="GPU node bundles"):
        validate_ray_capacity(result, [{"Alive": True, "Resources": {"GPU": 4}}] * 16)


@pytest.mark.parametrize(
    "actor,rollout,gpus_per_node,match",
    [
        ("d1t8", "d1t4", 8, "same total GPUs"),
        ("d1t16", "d1t16", 8, "cross-node"),
        ("d1t12", "d2t6", 8, "divisible"),
        ("d1t4", "d2t1c2", 8, "Parsing error"),
        ("d1t4", "d4t1", 8, "Single-GPU rollout"),
    ],
)
def test_ray_topology_rejects_unsupported_colocation(
    tmp_path, actor, rollout, gpus_per_node, match
):
    config = yaml.safe_load((RECIPE / "train_fine_released64_bs16.yaml").read_text())
    config.pop("allocation_mode")
    config["actor"]["backend"] = f"megatron:{actor}"
    config["rollout"]["backend"] = f"sglang:{rollout}"
    config["cluster"]["n_gpus_per_node"] = gpus_per_node
    with pytest.raises(ValueError, match=match):
        prepare_ray_config(config, RECIPE.parents[3], _config_env(tmp_path))


def test_ray_config_preserves_multistream_recipe_with_omitted_defaults(tmp_path):
    config = yaml.safe_load((RECIPE / "train_fine_released64_bs16.yaml").read_text())
    config["cluster"].pop("n_gpus_per_node")
    config["actor"].pop("scheduling_strategy", None)
    config["rollout"]["scheduling_strategy"].pop("fork")
    config["scheduler"] = {}
    config["econfig"]["arena_streams"] = [
        {
            "name": f"stream-{i}",
            "stream_id": f"id-{i}",
            "sampling_weight": i + 1,
            "reward_mapping_file": f"${{oc.env:QWEN_RECIPE_DIR}}/reward-mappings/{i}.json",
        }
        for i in range(6)
    ]
    config["train_dataset"]["batch_size"] = 64
    config["rollout"].update(
        consumer_batch_size=64, max_concurrent_rollouts=16, queue_size=256
    )
    config["gconfig"].update(n_samples=12, reward_normalization_use_std=False)
    config["sglang"]["max_running_requests"] = 1000
    original = copy.deepcopy(config)
    env = {
        k: v
        for k, v in _config_env(tmp_path).items()
        if k
        not in (
            "QWEN_ARENA_STREAM_ID",
            "QWEN_ARENA_HARNESS",
            "QWEN_REWARD_KEY",
            "QWEN_REWARD_VERSION",
        )
    }
    env["QWEN_RECIPE_DIR"] = str(tmp_path / "external recipe")
    result = prepare_ray_config(config, RECIPE.parents[3], env)
    assert config == original
    for key in ("econfig", "gconfig", "train_dataset", "sglang", "should_accept_fn"):
        assert result[key] == original[key]
    for key in ("consumer_batch_size", "max_concurrent_rollouts", "queue_size"):
        assert result["rollout"][key] == original["rollout"][key]
    assert ray_gpu_bundles(result) == [8] * 8
    for role in ("actor", "rollout"):
        assert (
            result[role]["scheduling_spec"][0]["env_vars"]["QWEN_SUBMITTED_RECIPE_DIR"]
            == env["QWEN_RECIPE_DIR"]
        )
    env["QWEN_REWARD_KEY"] = "accidental-global-override"
    with pytest.raises(ValueError, match="exactly one inline stream"):
        prepare_ray_config(config, RECIPE.parents[3], env)


def test_ray_config_rejects_reward_override_on_mapping(tmp_path):
    config = yaml.safe_load((RECIPE / "train_fine_released64_bs16.yaml").read_text())
    config["econfig"]["arena_streams"][0]["reward_mapping_file"] = "external.json"
    with pytest.raises(ValueError, match="mutually exclusive"):
        prepare_ray_config(config, RECIPE.parents[3], _config_env(tmp_path))


@pytest.mark.parametrize("role", ["actor", "rollout"])
def test_ray_config_rejects_implicit_cpu_workers(tmp_path, role):
    config = yaml.safe_load((RECIPE / "train_fine_released64_bs16.yaml").read_text())
    config[role]["scheduling_spec"][0].pop("gpu")
    with pytest.raises(ValueError, match="GPU"):
        prepare_ray_config(config, RECIPE.parents[3], _config_env(tmp_path))


def test_ray_capacity_rejects_nodes_that_can_pack_multiple_launchers(tmp_path):
    config = yaml.safe_load((RECIPE / "train_fine_released64_bs16.yaml").read_text())
    config["cluster"]["n_gpus_per_node"] = 4
    config["cluster"]["n_nodes"] = 16
    result = prepare_ray_config(config, RECIPE.parents[3], _config_env(tmp_path))
    with pytest.raises(RuntimeError, match="multiple actor launchers"):
        validate_ray_capacity(result, [{"Alive": True, "Resources": {"GPU": 8}}] * 16)


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
        QWEN_SUBMITTED_RECIPE_DIR=str(tmp_path / "external recipe"),
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
    assert actual["QWEN_RECIPE_DIR"] == env["QWEN_SUBMITTED_RECIPE_DIR"]
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


def test_ray_prepare_only_pins_custom_yaml_directory(tmp_path):
    env = _worker_environment(tmp_path)
    env["QWEN_RAY_PYTHON"] = sys.executable
    recipe_dir = tmp_path / "external experiment"
    recipe_dir.mkdir()
    config = yaml.safe_load((RECIPE / "train_fine_released64_bs16.yaml").read_text())
    config["econfig"]["arena_streams"] = [
        {
            "name": f"stream-{i}",
            "stream_id": f"id-{i}",
            "reward_mapping_file": f"${{oc.env:QWEN_RECIPE_DIR}}/mapping-{i}.json",
        }
        for i in range(2)
    ]
    config_path = recipe_dir / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with Path(env["QWEN_ENV_FILE"]).open("a") as profile:
        profile.write(
            "unset QWEN_ARENA_STREAM_ID QWEN_ARENA_HARNESS QWEN_REWARD_KEY QWEN_REWARD_VERSION\n"
        )
    result = subprocess.run(
        [
            "bash",
            str(RECIPE / "ray_worker.sh"),
            "controller",
            "--prepare-only",
            "--config",
            str(config_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    generated = yaml.safe_load((tmp_path / "run/train-ray.yaml").read_text())
    assert generated["econfig"]["arena_streams"] == config["econfig"]["arena_streams"]
    for role in ("actor", "rollout"):
        assert generated[role]["scheduling_spec"][0]["env_vars"][
            "QWEN_SUBMITTED_RECIPE_DIR"
        ] == str(recipe_dir)


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

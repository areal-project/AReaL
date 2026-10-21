import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from examples.swe.rl.qwen38_27b.qwen27_actor_worker import _install_cpu_optimizer
from examples.swe.rl.qwen38_27b.qwen27_train_rl import decorate_dense_controller


@pytest.fixture
def submit_env(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for name in (
        "model/config.json",
        "awex/awex/models/qwen3_5.py",
        "dense/dependencies/megatron/bridge/models/qwen_vl/qwen35_vl_bridge.py",
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    values = {
        "QWEN_REPO": str(checkout),
        "QWEN_RUN_DIR": str(tmp_path / "run"),
        "QWEN_OUTPUT_ROOT": str(tmp_path / "output"),
        "QWEN_MODEL": str(tmp_path / "model"),
        "AWEX_RUNTIME_ROOT": str(tmp_path / "awex"),
        "QWEN_DENSE_RUNTIME": str(tmp_path / "dense"),
        "QWEN_ARENA_TRIAL": "q27-bs8gs12-1013-1",
    }
    for key in (
        "QWEN_ACTOR_IMAGE",
        "QWEN_ROLLOUT_IMAGE",
        "QWEN_PARTITION",
        "QWEN_MOUNTS",
        "QWEN_CONTROLLER_MOUNTS",
        "QWEN_ACTOR_EXTRA_PYTHONPATH",
        "QWEN_ROLLOUT_RUNTIME_DEPS",
        "QWEN_PROXY_DEPS",
        "MEGATRON_ROOT",
        "ARENA_OPENAPI_BASE",
        "ARENA_OPENAPI_TOKEN",
        "ARENA_LLM_API_KEY",
        "WANDB_BASE_URL",
        "WANDB_API_KEY",
        "QWEN_CONTROLLER_NODE",
    ):
        values[key] = "unused-test-value"
    profile = tmp_path / "site.env"

    def write_profile():
        profile.write_text(
            "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())
        )

    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        "if 'rev-parse' in sys.argv: print('test-revision')\n"
        "elif os.getenv('QWEN_TEST_DIRTY'): print(' M tracked.py')\n"
    )
    git.chmod(0o700)
    sbatch = binaries / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "with open(os.environ['QWEN_TEST_SBATCH_ARGS'], 'w') as f: json.dump(sys.argv[1:], f)\n"
        "print('12345')\n"
    )
    sbatch.chmod(0o700)
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "QWEN_ENV_FILE": str(profile),
        "QWEN_TEST_SBATCH_ARGS": str(tmp_path / "sbatch-args.json"),
    }
    env.pop("BASH_ENV", None)
    script = Path(__file__).parents[1] / "examples/swe/rl/qwen38_27b/submit.sh"

    def run():
        write_profile()
        return subprocess.run(
            ["bash", str(script)], env=env, capture_output=True, text=True
        )

    return values, env, run


@pytest.mark.parametrize("field", ["QWEN_RUN_DIR", "QWEN_OUTPUT_ROOT"])
def test_submit_rejects_output_in_checkout_before_sbatch(submit_env, field):
    values, env, run = submit_env
    values[field] = values["QWEN_REPO"] + "/data"
    result = run()
    assert result.returncode != 0
    assert "outside the Git checkout" in result.stderr
    assert not Path(env["QWEN_TEST_SBATCH_ARGS"]).exists()


def test_submit_rejects_dirty_source_before_sbatch(submit_env):
    _, env, run = submit_env
    env["QWEN_TEST_DIRTY"] = "1"
    result = run()
    assert result.returncode != 0
    assert "Commit the reviewed source" in result.stderr
    assert not Path(env["QWEN_TEST_SBATCH_ARGS"]).exists()


def test_submit_snapshots_config_and_pins_source(submit_env):
    values, env, run = submit_env
    result = run()
    assert result.returncode == 0, result.stderr
    args = json.loads(Path(env["QWEN_TEST_SBATCH_ARGS"]).read_text())
    assert "--export=ALL" in args
    assert "--job-name=gray19-q27-bs8gs12-1013-1" in args
    root = Path(values["QWEN_RUN_DIR"])
    assert (root / "source-revision").read_text().strip() == "test-revision"
    assert (root / "train.yaml").exists()
    result = run()
    assert result.returncode == 0, result.stderr


def test_dense_controller_preserves_native_backends_and_input():
    calls = []
    controller = SimpleNamespace(
        initialize=lambda **kwargs: calls.append(kwargs), start_proxy=lambda: None
    )
    args = {"model_path": "model", "ple_offload_embedding": True}
    decorate_dense_controller(controller).initialize(server_args=args, engine="sglang")
    assert calls == [{"server_args": {"model_path": "model"}, "engine": "sglang"}]
    assert args["ple_offload_embedding"] is True


@pytest.mark.parametrize("raises", [False, True])
def test_dense_controller_keeps_proxy_and_restores_scheduler(raises):
    commands = []

    def fork(**kwargs):
        commands.append(kwargs["command"])

    scheduler = SimpleNamespace(fork_workers=fork)

    def start_proxy():
        scheduler.fork_workers(
            command="areal.experimental.openai.proxy.proxy_rollout_server"
        )
        scheduler.fork_workers(command="another.module")
        if raises:
            raise RuntimeError("proxy failed")
        return "started"

    controller = SimpleNamespace(
        initialize=lambda **kwargs: None, start_proxy=start_proxy, scheduler=scheduler
    )
    decorate_dense_controller(controller)
    if raises:
        with pytest.raises(RuntimeError, match="proxy failed"):
            controller.start_proxy()
    else:
        assert controller.start_proxy() == "started"
    assert commands == ["examples.swe.qwen38_flash_next.proxy", "another.module"]
    assert scheduler.fork_workers is fork


def test_cpu_optimizer_keeps_fp32_offload_and_other_options():
    allocations = []
    engine = SimpleNamespace(MCoreOptimizerConfig=lambda **kwargs: kwargs)
    torch = SimpleNamespace(
        float32="fp32",
        cuda=SimpleNamespace(
            current_device=lambda: 3,
            set_per_process_memory_fraction=lambda *args: allocations.append(args),
        ),
    )
    _install_cpu_optimizer(engine, torch)
    config = engine.MCoreOptimizerConfig(lr=3e-6, optimizer_cpu_offload=False)
    assert config["lr"] == 3e-6
    assert config["optimizer_cpu_offload"] is True
    assert config["optimizer_offload_fraction"] == 1.0
    assert config["use_torch_optimizer_for_cpu_offload"] is True
    assert config["overlap_cpu_optimizer_d2h_h2d"] is True
    assert config["use_precision_aware_optimizer"] is False
    for field in (
        "main_grads_dtype",
        "main_params_dtype",
        "exp_avg_dtype",
        "exp_avg_sq_dtype",
    ):
        assert config[field] == "fp32"
    assert allocations == [(1.0, 3)]


def test_graybox_config_retains_sampling_cache_and_full_recovery():
    path = Path(__file__).parents[1] / "examples/swe/rl/qwen38_27b/train.yaml"
    config = yaml.safe_load(path.read_text())
    for name in ("gconfig", "eval_gconfig"):
        assert config[name]["temperature"] == 1.0
        assert config[name]["top_p"] == 1.0
        assert config[name]["top_k"] == 100000
    assert config["gconfig"]["n_samples"] == 12
    assert config["gconfig"]["reward_normalization_use_std"] is True
    assert config["train_dataset"]["batch_size"] == 8
    assert config["rollout"]["max_concurrent_rollouts"] == 16
    assert config["rollout"]["return_routed_experts"] is False
    assert config["sglang"]["disable_radix_cache"] is False
    assert config["sglang"]["disable_cuda_graph"] is False
    assert config["sglang"]["enable_memory_saver"] is True
    assert config["actor"]["megatron"]["enable_chunked_logits"] is False
    assert config["actor"]["megatron"]["lm_head_loss_chunk_size"] == 0
    assert config["recover"]["freq_secs"] == 7200
    assert config["recover"]["no_save_optim"] is False
    assert config["recover"]["no_load_optim"] is False

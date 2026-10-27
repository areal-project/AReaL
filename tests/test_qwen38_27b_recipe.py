import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from examples.swe.rl.qwen38_27b.qwen27_actor_worker import _install_cpu_optimizer
from examples.swe.rl.qwen38_27b.qwen27_train_rl import decorate_dense_controller
from examples.swe.rl.qwen38_27b.source_snapshot import stage, verify


@pytest.fixture
def submit_env(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    recipe = Path("examples/swe/rl/qwen38_27b")
    shutil.copytree(Path(__file__).parents[1] / recipe, checkout / recipe)
    (checkout / "areal").mkdir()
    (checkout / "areal/obsolete.py").touch()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
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
        "QWEN_EXPERIMENT_NAME": "chucai-qwen27-test",
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
    sudo = binaries / "sudo"
    sudo.write_text('#!/bin/sh\nexec "$@"\n')
    sudo.chmod(0o700)
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


def test_submit_freezes_local_edits_and_deletions(submit_env):
    """The snapshot captures local source without changing the checkout."""
    values, _, run = submit_env
    checkout = Path(values["QWEN_REPO"])
    (checkout / "areal/obsolete.py").unlink()
    (checkout / "areal/new.py").write_text("# local edit\n")
    (checkout / "scratch.txt").write_text("excluded")
    result = run()
    assert result.returncode == 0, result.stderr
    code = Path(values["QWEN_RUN_DIR"]) / "source/code"
    assert (code / "areal/new.py").read_text() == "# local edit\n"
    assert not (code / "areal/obsolete.py").exists()
    assert not (code / "scratch.txt").exists()
    assert not (code / ".git").exists()
    assert (checkout / "areal/new.py").exists()


def test_submit_snapshots_config_and_pins_source(submit_env):
    values, env, run = submit_env
    result = run()
    assert result.returncode == 0, result.stderr
    args = json.loads(Path(env["QWEN_TEST_SBATCH_ARGS"]).read_text())
    assert "--export=ALL" in args
    assert "--job-name=chucai-qwen27-test-q27-bs8gs12-1013-1" in args
    root = Path(values["QWEN_RUN_DIR"])
    revision = (root / "source-revision").read_text().strip()
    assert verify(root / "source", revision) == revision
    assert (
        root / "source/code/examples/swe/rl/qwen38_27b/site.env"
    ).stat().st_mode & 0o777 == 0o600
    assert (root / "train.yaml").exists()
    result = run()
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("change", ["profile", "source", "extra_file"])
def test_submit_rejects_changed_snapshot_before_sbatch(submit_env, change):
    """Neither changed inputs nor tampered shared files can overwrite a trial."""
    values, env, run = submit_env
    assert run().returncode == 0
    root = Path(values["QWEN_RUN_DIR"])
    original = (root / "source/manifest.json").read_bytes()
    Path(env["QWEN_TEST_SBATCH_ARGS"]).unlink()
    if change == "profile":
        values["QWEN_PARTITION"] = "changed"
    else:
        relative = (
            "examples/swe/rl/qwen38_27b/train.yaml"
            if change == "source"
            else "unexpected.py"
        )
        (root / "source/code" / relative).write_text("changed")
    result = run()
    assert result.returncode != 0
    assert "manifest mismatch" in result.stderr
    assert (root / "source/manifest.json").read_bytes() == original
    assert not Path(env["QWEN_TEST_SBATCH_ARGS"]).exists()


def test_snapshot_rejects_symlink_escape(submit_env, tmp_path):
    """Copied code cannot depend on a link outside the frozen directory."""
    values, _, _ = submit_env
    source = Path(values["QWEN_REPO"])
    (tmp_path / "site.env").write_text("QWEN_TEST=1\n")
    (tmp_path / "outside.py").touch()
    (source / "areal/escape.py").symlink_to("../../outside.py")
    with pytest.raises(ValueError, match="symlink escapes"):
        stage(source, tmp_path / "site.env", tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


@pytest.mark.parametrize("external", [False, True])
def test_snapshot_rejects_symlink_parent_before_writing(submit_env, tmp_path, external):
    values, _, _ = submit_env
    source = Path(values["QWEN_REPO"])
    profile = tmp_path / "site.env"
    profile.write_text("QWEN_TEST=1\n")
    target = (tmp_path if external else source) / "target"
    shutil.move(source / "areal", target)
    sentinel = target / "obsolete.py"
    sentinel.write_text("must remain unchanged\n")
    (source / "areal").symlink_to("../target" if external else "target")
    with pytest.raises(ValueError, match="symlink"):
        stage(source, profile, tmp_path / "snapshot")
    assert sentinel.read_text() == "must remain unchanged\n"
    assert not (tmp_path / "snapshot").exists()


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
    assert config["train_dataset"]["batch_size"] == 16
    assert config["rollout"]["consumer_batch_size"] == 16
    assert config["rollout"]["max_concurrent_rollouts"] == 1500
    assert config["rollout"]["queue_size"] == 256
    assert config["rollout"]["return_routed_experts"] is False
    assert config["sglang"]["disable_radix_cache"] is False
    assert config["sglang"]["disable_cuda_graph"] is False
    assert config["sglang"]["enable_memory_saver"] is True
    assert config["actor"]["megatron"]["enable_chunked_logits"] is False
    assert config["actor"]["megatron"]["lm_head_loss_chunk_size"] == 0
    assert config["recover"]["freq_secs"] == 7200
    assert config["recover"]["no_save_optim"] is False
    assert config["recover"]["no_load_optim"] is False

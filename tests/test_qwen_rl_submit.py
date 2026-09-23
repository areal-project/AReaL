# SPDX-License-Identifier: Apache-2.0
"""Exercise the submission boundary without installing packages or submitting jobs."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

RECIPE = Path(__file__).parents[1] / "examples/swe/qwen38_flash_next"


def _environment(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    scripts = {
        "git": """case "${3}" in
    rev-parse) printf '%s\\n' "${TEST_BRIDGE_REVISION}" ;;
    status) printf '%s' "${TEST_BRIDGE_DIRTY:-}" ;;
    *) exit 99 ;;
esac
""",
        "uv": "exit 99\n",
        "sbatch": """printf '%s\\n' "$QWEN_ACTOR_PYTHONPATH" "$QWEN_CONTROLLER_PYTHONPATH" "$QWEN_ROLLOUT_PYTHONPATH" >> "$TEST_SUBMITTED"
""",
    }
    for name, script in scripts.items():
        path = tools / name
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + script)
        path.chmod(0o755)
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("QWEN_")
    }
    env.pop("BASH_ENV", None)
    env.update(
        PATH=f"{tools}:{env['PATH']}",
        TEST_LOG=str(tmp_path / "commands"),
        TEST_SUBMITTED=str(tmp_path / "submitted"),
        QWEN_REPO=str(RECIPE.parents[2]),
        QWEN_OUTPUT_ROOT=str(tmp_path / "output"),
        QWEN_MODEL="model",
        QWEN_ACTOR_IMAGE="actor.sif",
        QWEN_ROLLOUT_IMAGE="rollout.sif",
        QWEN_RESERVATION="reservation",
        QWEN_NODELIST="nodes",
        QWEN_PARTITION="partition",
        QWEN_CONTROLLER_NODE="controller",
        QWEN_MOUNTS=str(tmp_path),
        QWEN_CONTROLLER_MOUNTS=str(tmp_path),
        MEGATRON_ROOT="/inference/megatron",
        MCORE_BRIDGE_ROOT="/training/bridge",
        QWEN_TRAIN_EXTRA_PYTHONPATH="/training/transformers",
        TEST_BRIDGE_REVISION="557aaf93b16d083fdec4f82a8251d47d47c76ccb",
        QWEN_INFER_EXTRA_PYTHONPATH="/inference/overlay",
    )
    for key in ("QWEN_PRIVATE_ENV", "QWEN_ARENA_STREAMS_FILE"):
        path = tmp_path / key
        path.touch()
        env[key] = str(path)
    return env


def _submit(env):
    return subprocess.run(
        ["bash", str(RECIPE / "submit_rl.sh"), "swe"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_submit_propagates_training_sources_without_changing_rollout(tmp_path):
    env = _environment(tmp_path)
    result = _submit(env)
    assert result.returncode == 0, result.stderr
    actor, controller, rollout = Path(env["TEST_SUBMITTED"]).read_text().splitlines()
    assert actor == f"/training/transformers:/training/bridge/src:{env['QWEN_REPO']}"
    assert controller == actor
    assert rollout == f"/inference/overlay:/inference/megatron:{env['QWEN_REPO']}"


@pytest.mark.parametrize(
    "invalid_bridge",
    [{"TEST_BRIDGE_REVISION": "wrong"}, {"TEST_BRIDGE_DIRTY": " M code.py"}],
)
def test_submit_does_not_schedule_with_invalid_bridge(tmp_path, invalid_bridge):
    env = _environment(tmp_path)
    env.update(invalid_bridge)
    assert _submit(env).returncode != 0
    assert not Path(env["TEST_SUBMITTED"]).exists()


def test_yaml_passes_separate_pythonpaths_to_image_interpreters():
    config = yaml.safe_load((RECIPE / "swe_mm_rl.yaml").read_text())
    for name, path in [
        ("actor", "QWEN_ACTOR_PYTHONPATH"),
        ("rollout", "QWEN_ROLLOUT_PYTHONPATH"),
    ]:
        spec = config[name]["scheduling_spec"][0]
        assert spec["cmd"].startswith("python3 -m ")
        assert spec["env_vars"]["PYTHONPATH"] == "${oc.env:" + path + "}"
        assert spec["image"] == "${oc.env:QWEN_" + name.upper() + "_IMAGE}"


@pytest.mark.parametrize("training_runtime", [False, True])
def test_rollout_startup_uses_image_python_for_patches_and_rpc(
    tmp_path, training_runtime
):
    config = yaml.safe_load((RECIPE / "swe_mm_rl.yaml").read_text())
    rollout = config["rollout"]["scheduling_spec"][0]
    image_bin = tmp_path / "image-bin"
    image_bin.mkdir()
    image_python = image_bin / "python3"
    image_python.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_LOG"\n')
    image_python.chmod(0o755)
    train_bin = tmp_path / "training-bin"
    train_bin.mkdir()
    for name in ("python", "python3"):
        executable = train_bin / name
        executable.write_text("#!/bin/sh\nexit 99\n")
        executable.chmod(0o755)
    private_env = tmp_path / "private.env"
    private_env.touch()
    env = dict(os.environ)
    env.pop("BASH_ENV", None)
    env.pop("QWEN_TRAIN_PYTHON", None)
    env.update(
        PATH=f"{image_bin}:{os.environ['PATH']}",
        QWEN_PRIVATE_ENV=str(private_env),
        TMS_INIT_ENABLE="0",
        TMS_INIT_ENABLE_CPU_BACKUP="0",
        TEST_LOG=str(tmp_path / "python-calls"),
    )
    if training_runtime:
        env["QWEN_TRAIN_PYTHON"] = str(train_bin / "python")
    script = "\n".join([*rollout["additional_bash_cmds"], rollout["cmd"]])
    script = script.replace("${oc.env:QWEN_REPO}", str(RECIPE.parents[2]))
    result = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
    assert Path(env["TEST_LOG"]).read_text().splitlines() == [
        "-m examples.swe.qwen38_flash_next.patch_sglang_qsa_topk",
        "-m examples.swe.qwen38_flash_next.patch_sglang_qsa_compress_gather",
        "-m areal.infra.rpc.rpc_server",
    ]

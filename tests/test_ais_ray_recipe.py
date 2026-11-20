# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import ais_ray_bootstrap as ray_cluster
import submit


@pytest.mark.parametrize(
    ("env", "role"),
    [
        ({"POD_NAME": "aistudio-example-ptjob-master-0"}, "head"),
        ({"HOSTNAME": "aistudio-example-ptjob-worker-2"}, "worker"),
        ({"AISTUDIO_TASK_ROLE_NAME": "worker"}, "worker"),
        ({"AREAL_NODE_RANK": "0", "AREAL_AIS_NODES": "4"}, "head"),
        ({"RANK": "2", "AREAL_AIS_NODES": "4"}, "worker"),
    ],
)
def test_ais_role_recognizes_platform_identity(env, role):
    """Only the master launches the single training driver."""
    assert ray_cluster.ais_role(env) == role


def test_ais_role_refuses_unknown_pod():
    """Do not assume an unrelated Pod is an AIS worker or head."""
    with pytest.raises(ValueError, match="Cannot identify"):
        ray_cluster.ais_role({"HOSTNAME": "dev-pod"})


def test_rendezvous_address_isolated_per_submission(tmp_path):
    """Workers read an atomic address scoped to their own submitted job."""
    first = ray_cluster.rendezvous_path(tmp_path, tmp_path, "areal-a")
    second = ray_cluster.rendezvous_path(tmp_path, tmp_path, "areal-b")
    ray_cluster.publish_head(first, "192.0.2.1")

    assert ray_cluster.wait_for_head(first, 1) == "192.0.2.1"
    assert not second.exists()
    with pytest.raises(FileExistsError):
        ray_cluster.publish_head(first, "192.0.2.2")
    with pytest.raises(TimeoutError):
        ray_cluster.wait_for_head(second, 0)


def test_rendezvous_refuses_local_only_output_path(tmp_path):
    """The bootstrap requires the mounted storage, not the Pod-local checkout."""
    with pytest.raises(ValueError, match="absolute path under"):
        ray_cluster.rendezvous_path(tmp_path / "local", tmp_path / "cpfs", "areal-a")


def test_ready_requires_32_available_gpus_in_four_nodes():
    """Cluster membership alone is insufficient to start training."""
    nodes = [{"Alive": True, "Resources": {"GPU": 8}} for _ in range(4)]

    ray_cluster.ready_for_training(nodes, 32, 4, 8)
    with pytest.raises(RuntimeError, match="Only 31"):
        ray_cluster.ready_for_training(nodes, 31, 4, 8)
    with pytest.raises(RuntimeError, match="Expected 4"):
        ray_cluster.ready_for_training(nodes[:3], 24, 4, 8)


@pytest.mark.parametrize("role", ["head", "worker"])
def test_ais_dispatch_runs_driver_only_after_head_bootstrap(
    monkeypatch, tmp_path, role
):
    """Both roles use the same packaged entrypoint, but only the head trains."""
    monkeypatch.setattr(os, "environ", os.environ.copy())
    job = "areal-a"
    for key, value in {
        "AREAL_AIS_JOB_NAME": job,
        "AREAL_AIS_NODES": "3",
        "AREAL_AIS_GPUS_PER_NODE": "8",
        "AREAL_AIS_STORAGE_ROOT": str(tmp_path),
        "AREAL_AIS_RAY_ROOT": str(tmp_path),
        "AREAL_NODE_RANK": "0" if role == "head" else "1",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("RAY_ADDRESS", "unused-existing-cluster")
    cluster = SimpleNamespace(
        ray_port=6379,
        ray_dashboard_port=8265,
        ray_bootstrap_timeout_seconds=1,
    )
    monkeypatch.setattr(ray_cluster, "check_ais_devices", Mock())
    user_command = Mock(return_value=7)
    monkeypatch.setattr(ray_cluster, "run_user_command", user_command)
    ray = Mock()
    ray.nodes.return_value = [{"Alive": True, "Resources": {"GPU": 8}}] * 3
    ray.available_resources.return_value = {"GPU": 24}
    args = ModuleType("areal.api.cli_args")
    args.ClusterSpecConfig = Mock(return_value=cluster)
    bootstrap = Mock()
    bootstrap.detect_node_ip.return_value = (
        "192.0.2.1" if role == "head" else "192.0.2.2"
    )
    launcher = ModuleType("areal.infra.launcher")
    launcher.ray_bootstrap = bootstrap
    for name, pkg in (
        ("ray", ray),
        ("areal", ModuleType("areal")),
        ("areal.api", ModuleType("areal.api")),
        ("areal.api.cli_args", args),
        ("areal.infra", ModuleType("areal.infra")),
        ("areal.infra.launcher", launcher),
    ):
        monkeypatch.setitem(sys.modules, name, pkg)
    path = ray_cluster.rendezvous_path(tmp_path, tmp_path, job)
    if role == "worker":
        monkeypatch.setenv("AREAL_MASTER_ADDR", "")
        ray_cluster.publish_head(path, "192.0.2.1")

    command = 'printf "%s\\n" "$RAY_ADDRESS"'
    code = ray_cluster.run_ais(command)

    if role == "head":
        bootstrap.bootstrap_head.assert_called_once()
        bootstrap.bootstrap_worker.assert_not_called()
        user_command.assert_called_once_with(command, "192.0.2.1:6379")
        assert code == 7
        bootstrap.stop_local_ray.assert_called_once()
    else:
        bootstrap.bootstrap_head.assert_not_called()
        bootstrap.bootstrap_worker.assert_called_once()
        user_command.assert_not_called()
        assert code == 0
        assert ray_cluster.os.environ["AREAL_MASTER_ADDR"] == "192.0.2.1"


def test_submit_embeds_shared_bootstrap_settings_and_quotes_arguments():
    """Every replica receives one job name, the CPFS root and identical args."""
    user_command = 'QWEN_REPO="$PWD" bash "examples/run this.sh" total_train_steps=10'
    command = submit.startup_command(
        user_command,
        job_name="areal-a",
        nodes=4,
        rdma=True,
        ray_root="/storage/output root",
        forwarded_env={"QWEN_ARENA_OUTPUT": "/storage/output root"},
    )
    bash_command = shlex.split(command)[2]

    assert "'QWEN_ARENA_OUTPUT=/storage/output root'" in bash_command
    assert "AREAL_AIS_JOB_NAME=areal-a" in bash_command
    assert "AREAL_AIS_NODES=4" in bash_command
    assert "AREAL_AIS_GPUS_PER_NODE=8" in bash_command
    assert "AREAL_AIS_STORAGE_ROOT=/storage" in bash_command
    assert "'AREAL_AIS_RAY_ROOT=/storage/output root'" in bash_command
    assert "AREAL_AIS_RDMA=1" in bash_command
    assert shlex.split(bash_command.split(" && ")[-1]) == [
        "exec",
        "python3",
        "ais_ray_bootstrap.py",
        "--",
        user_command,
    ]


def test_single_node_runs_user_command_without_bootstrap():
    """Keep the already working single-node startup path."""
    command = submit.startup_command(
        "exec bash -- examples/run.sh", launcher=Path("examples/run.sh")
    )
    script = shlex.split(command)[2]
    assert "AREAL_AIS_JOB_NAME" not in script
    assert "ais_ray_bootstrap.py" not in script
    assert "test -f examples/run.sh" in script
    assert shlex.split(script.split(" && ")[-1]) == [
        "exec",
        "bash",
        "-c",
        "exec bash -- examples/run.sh",
    ]


def test_user_command_gets_ray_address_without_spmd_rank(monkeypatch):
    """The user command is not the AIS worker process and uses Ray for ranks."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("MASTER_ADDR", "old-torch-head")
    runner = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(ray_cluster.subprocess, "run", runner)

    assert ray_cluster.run_user_command("exit 0", "192.0.2.1:6379") == 0
    assert runner.call_args.args == (("bash", "-c", "exit 0"),)
    env = runner.call_args.kwargs["env"]
    assert env["RAY_ADDRESS"] == "192.0.2.1:6379"
    assert "RANK" not in env
    assert "MASTER_ADDR" not in env


def test_multinode_archive_requires_bootstrap_file(tmp_path):
    """Refuse to upload a bundle that cannot run the generated command."""
    manifest = tmp_path / "pyproject.toml"
    manifest.touch()
    bundle = tmp_path / "source.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(manifest, arcname="AReaL/pyproject.toml")

    submit.check_archive(str(bundle), None)
    with pytest.raises(ValueError, match="ais_ray_bootstrap.py"):
        submit.check_archive(str(bundle), None, nodes=2)


def test_multi_node_ray_root_requires_shared_storage():
    """Resolve either explicit shared root or the old forwarded output root."""
    assert submit.resolve_ray_root("/storage/ray", {}) == "/storage/ray"
    assert (
        submit.resolve_ray_root("", {"QWEN_ARENA_OUTPUT": "/storage/output"})
        == "/storage/output"
    )
    with pytest.raises(ValueError, match="AREAL_RAY_ROOT"):
        submit.resolve_ray_root("/storage/../../tmp/not-shared", {})


def test_command_preview_uses_default_command_without_submitting(monkeypatch, capsys):
    """The top-level COMMAND exposes only the post-bootstrap shell command."""
    monkeypatch.setattr(submit, "COMMAND", "echo user-command")
    monkeypatch.setattr(
        sys, "argv", ["submit.py", "--nodes", "4", "--ray-root", "/storage/ray"]
    )
    called = Mock()
    monkeypatch.setattr(submit, "submit", called)

    assert submit.main() == 0
    preview = capsys.readouterr().out
    assert "Command: echo user-command" in preview
    assert "command runs on head only" in preview
    assert "ais_ray_bootstrap.py" not in preview
    called.assert_not_called()


def test_forwarded_env_blocks_credentials_and_internal_fields(monkeypatch):
    """Pod commands must not include AIS tokens or user overrides of runtime metadata."""
    monkeypatch.setenv("QWEN_MODEL", "/storage/model")

    assert submit.collect_forwarded_env(["QWEN_MODEL"]) == {
        "QWEN_MODEL": "/storage/model"
    }
    for invalid in ("AISTUDIO_TOKEN", "WANDB_API_KEY", "AREAL_AIS_JOB_NAME"):
        with pytest.raises(ValueError, match="Cannot forward"):
            submit.collect_forwarded_env([invalid])


def test_roce16_recipe_dry_run_supplies_training_defaults(monkeypatch, tmp_path):
    """The head command needs no W&B URL, step, or checkpoint arguments."""
    probe = tmp_path / "python3"
    probe.write_text(
        "#!/bin/bash\n"
        'printf "WANDB_BASE_URL=%s\\nQWEN_ARENA_TRIAL=%s\\n" '
        '"$WANDB_BASE_URL" "$QWEN_ARENA_TRIAL"\n',
        encoding="utf-8",
    )
    probe.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path), prepend=os.pathsep)
    monkeypatch.setenv("QWEN_MODEL", str(tmp_path / "model"))
    monkeypatch.setenv("QWEN_DATASET", str(tmp_path / "dataset"))
    monkeypatch.setenv("QWEN_ARENA_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("AREAL_AIS_JOB_NAME", "areal-a")
    for name in ("WANDB_BASE_URL", "QWEN_ARENA_EXPERIMENT", "QWEN_ARENA_TRIAL"):
        monkeypatch.delenv(name, raising=False)
    recipe = (
        Path(__file__).resolve().parents[1]
        / "examples/math/qwen38_27b_ray_32gpu/run_ais_roce16.sh"
    )

    result = subprocess.run(
        ["bash", str(recipe), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "WANDB_BASE_URL=https://slurm.alipay.com/\n" in result.stdout
    assert "QWEN_ARENA_TRIAL=areal-a\n" in result.stdout
    assert " total_train_steps=10 " in result.stdout
    assert " saver.freq_steps=2" in result.stdout
    assert not (tmp_path / "output").exists()

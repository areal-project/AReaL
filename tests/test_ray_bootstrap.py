# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ray") is None,
    reason="ray is required for importing the Ray bootstrap helpers",
)


def test_accelerator_cli_args_use_native_gpu_flag_and_custom_npu_resource():
    from areal.infra.launcher.ray_bootstrap import _accelerator_cli_args

    assert _accelerator_cli_args("GPU", 8) == ["--num-gpus=8"]

    npu_args = _accelerator_cli_args("NPU", 8)
    assert len(npu_args) == 1
    assert npu_args[0].startswith("--resources=")
    assert json.loads(npu_args[0].split("=", 1)[1]) == {"NPU": 8}


def test_bootstrap_head_registers_npu_resource(monkeypatch):
    import areal.infra.launcher.ray_bootstrap as bootstrap

    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)
    monkeypatch.setattr(bootstrap.ray, "init", lambda **kwargs: None)
    monkeypatch.setattr(bootstrap, "wait_for_ray_nodes", lambda *args, **kwargs: None)

    bootstrap.bootstrap_head(
        "10.0.0.1",
        n_nodes=2,
        n_gpus_per_node=8,
        accelerator_resource="NPU",
    )

    start_command = commands[0][0]
    assert not any(arg.startswith("--num-gpus") for arg in start_command)
    resources_arg = next(arg for arg in start_command if arg.startswith("--resources="))
    assert json.loads(resources_arg.split("=", 1)[1]) == {"NPU": 8}


def test_bootstrap_worker_times_out_and_cleans_up(monkeypatch):
    import areal.infra.launcher.ray_bootstrap as bootstrap

    stop_calls = []
    clock = iter([0.0, 0.0, 1.0])
    monkeypatch.setenv("AREAL_MASTER_ADDR", "10.0.0.1")
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    monkeypatch.setattr(bootstrap, "stop_local_ray", lambda: stop_calls.append(True))

    with pytest.raises(TimeoutError, match="Timed out joining Ray head"):
        bootstrap.bootstrap_worker(
            "10.0.0.2",
            n_gpus_per_node=8,
            wait_timeout=1,
        )

    assert stop_calls == [True]


def test_bootstrap_worker_limits_blocked_ray_cli_and_cleans_up(monkeypatch):
    import areal.infra.launcher.ray_bootstrap as bootstrap

    stop_calls = []
    monkeypatch.setenv("AREAL_MASTER_ADDR", "10.0.0.1")
    monkeypatch.setattr(bootstrap.time, "monotonic", lambda: 0.0)

    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(bootstrap.subprocess, "run", timeout_run)
    monkeypatch.setattr(bootstrap, "stop_local_ray", lambda: stop_calls.append(True))

    with pytest.raises(TimeoutError, match="Timed out joining Ray head"):
        bootstrap.bootstrap_worker(
            "10.0.0.2",
            n_gpus_per_node=8,
            wait_timeout=3,
        )

    assert stop_calls == [True]


def test_head_bootstrap_failure_is_cleaned_up_by_main(monkeypatch):
    import areal.infra.launcher.ray as ray_launcher

    config = SimpleNamespace(
        cluster=SimpleNamespace(
            n_nodes=2,
            n_gpus_per_node=8,
            ray_port=6379,
            ray_dashboard_port=8265,
            ray_bootstrap_timeout_seconds=30,
        )
    )
    stop_calls = []

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setattr(ray_launcher, "parse_cli_args", lambda args: (config, None))
    monkeypatch.setattr(ray_launcher, "to_structured_cfg", lambda value, cls: value)
    monkeypatch.setattr(ray_launcher, "detect_node_rank", lambda: 0)
    monkeypatch.setattr(ray_launcher, "detect_node_ip", lambda: "10.0.0.1")
    monkeypatch.setattr(
        ray_launcher,
        "current_platform",
        SimpleNamespace(ray_device_key="GPU"),
    )
    monkeypatch.setattr(
        ray_launcher,
        "stop_local_ray",
        lambda: stop_calls.append(True),
    )
    monkeypatch.setattr(
        ray_launcher,
        "bootstrap_head",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("bootstrap")),
    )

    with pytest.raises(TimeoutError, match="bootstrap"):
        ray_launcher.main()

    assert stop_calls == [True, True]


@pytest.mark.parametrize("ray_port", [0, -1, 65536, True])
def test_cluster_spec_config_rejects_invalid_ray_port(ray_port):
    """The public cluster config requires a fixed valid Ray head port."""
    from areal.api.cli_args import ClusterSpecConfig

    with pytest.raises(ValueError, match="cluster.ray_port"):
        ClusterSpecConfig(ray_port=ray_port)


def test_ray_launcher_main_rejects_dynamic_ray_port(monkeypatch):
    """CLI/YAML configuration rejects port zero before changing Ray state."""
    import areal.infra.launcher.ray as ray_launcher

    config = SimpleNamespace(
        cluster=OmegaConf.create(
            {
                "n_nodes": 2,
                "n_gpus_per_node": 8,
                "ray_port": 0,
            }
        )
    )
    stop_calls = []
    monkeypatch.setattr(ray_launcher, "parse_cli_args", lambda args: (config, None))
    monkeypatch.setattr(
        ray_launcher,
        "stop_local_ray",
        lambda: stop_calls.append(True),
    )

    with pytest.raises(ValueError, match="cluster.ray_port"):
        ray_launcher.main()

    assert stop_calls == []

"""Tests for hiding accelerators from the single-controller process.

In single-controller mode the trainer process only orchestrates remote workers and
should hold no device context. But merely importing areal pulls in transformers, which
eagerly imports the torchao quantizer; torchao probes the GPU at import
(has_triton -> torch.cuda.current_device) and creates a ~hundreds-of-MiB CUDA context.

The controller-hide makes the accelerator invisible to controller-like processes before
those imports run (areal/__init__.py), and the LocalScheduler recovers the worker
device pool + control env var from the stashed original visibility
(areal/infra/scheduler/local.py).
"""

import subprocess
import sys
import textwrap

import pytest
import torch

# NOTE: torch.cuda.is_available() does not create a CUDA context.
CUDA_AVAILABLE = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Scheduler resolves the worker device pool / control env var from the stash
# ---------------------------------------------------------------------------


def test_worker_device_control_env_var_prefers_stash(monkeypatch):
    from areal.infra.scheduler import local as local_sched

    monkeypatch.setenv("AREAL_CONTROLLER_HIDDEN_DEVICE_ENV", "CUDA_VISIBLE_DEVICES")
    assert local_sched._worker_device_control_env_var() == "CUDA_VISIBLE_DEVICES"

    monkeypatch.delenv("AREAL_CONTROLLER_HIDDEN_DEVICE_ENV", raising=False)
    # Falls back to the platform's own value.
    assert isinstance(local_sched._worker_device_control_env_var(), str)


def test_detect_gpus_reads_stashed_subset(monkeypatch):
    from areal.infra.scheduler.local import LocalScheduler

    monkeypatch.setenv("AREAL_CONTROLLER_HIDDEN_DEVICE_ENV", "CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("AREAL_CONTROLLER_ORIG_DEVICES_SET", "1")
    monkeypatch.setenv("AREAL_CONTROLLER_ORIG_DEVICES", "2,3,5")
    # _detect_gpus does not use self.
    assert LocalScheduler._detect_gpus(object()) == [2, 3, 5]


def test_detect_gpus_stashed_unset_uses_physical_count(monkeypatch):
    from areal.infra.scheduler import local as local_sched
    from areal.infra.scheduler.local import LocalScheduler

    monkeypatch.setenv("AREAL_CONTROLLER_HIDDEN_DEVICE_ENV", "CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("AREAL_CONTROLLER_ORIG_DEVICES_SET", "0")
    monkeypatch.setattr(local_sched, "_get_device_count_safely", lambda: 4)
    assert LocalScheduler._detect_gpus(object()) == [0, 1, 2, 3]


def test_detect_gpus_stashed_empty_returns_no_devices(monkeypatch):
    """An explicit empty original visibility means 'no devices', not 'all devices'."""
    from areal.infra.scheduler import local as local_sched
    from areal.infra.scheduler.local import LocalScheduler

    monkeypatch.setenv("AREAL_CONTROLLER_HIDDEN_DEVICE_ENV", "CUDA_VISIBLE_DEVICES")
    # Original CUDA_VISIBLE_DEVICES was explicitly "" (set, but empty).
    monkeypatch.setenv("AREAL_CONTROLLER_ORIG_DEVICES_SET", "1")
    monkeypatch.setenv("AREAL_CONTROLLER_ORIG_DEVICES", "")
    # Must not fall back to enumerating physical devices.
    monkeypatch.setattr(local_sched, "_get_device_count_safely", lambda: 8)
    assert LocalScheduler._detect_gpus(object()) == []


# ---------------------------------------------------------------------------
# Gating of _early_hide_controller_devices (controller default; worker signals never hide)
# ---------------------------------------------------------------------------


def _run_hide_probe(extra_env: dict, argv_tail=None) -> bool:
    """Import areal in a subprocess; return whether the controller-hide ran.

    Detected via the stash var AREAL_CONTROLLER_HIDDEN_DEVICE_ENV, which the hide
    logic only sets when it actually blanks a device env var. Runs under a fake
    /dev probe so the test does not require real accelerators.
    """
    import os

    code = textwrap.dedent(
        """
        import os, builtins
        # Pretend an NVIDIA accelerator exists without touching the real /dev.
        _real_listdir = os.listdir
        os.listdir = lambda p="/dev": ["nvidia0"] if p == "/dev" else _real_listdir(p)
        import areal  # noqa: F401  triggers _early_hide_controller_devices()
        print("HIDDEN" if os.environ.get("AREAL_CONTROLLER_HIDDEN_DEVICE_ENV")
              else "VISIBLE")
        """
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }
    pp = os.environ.get("PYTHONPATH")
    if pp:
        env["PYTHONPATH"] = pp
    env.update(extra_env)
    result = subprocess.run(
        [sys.executable, "-c", code, *(argv_tail or [])],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return "HIDDEN" in result.stdout


def test_hide_on_by_default():
    """A controller-like process hides devices without requiring an explicit env flag."""
    assert _run_hide_probe({}) is True


def test_hide_can_be_disabled():
    """AREAL_HIDE_CONTROLLER_DEVICES=0 keeps the controller's devices visible."""
    assert _run_hide_probe({"AREAL_HIDE_CONTROLLER_DEVICES": "0"}) is False


def test_hide_on_when_opted_in():
    """AREAL_HIDE_CONTROLLER_DEVICES=1 explicitly keeps the default hide behavior."""
    assert _run_hide_probe({"AREAL_HIDE_CONTROLLER_DEVICES": "1"}) is True


def test_hide_skipped_for_spmd_by_default():
    """A worker process with AREAL_SPMD_MODE is never hidden by default."""
    assert _run_hide_probe({"AREAL_SPMD_MODE": "1"}) is False


def test_hide_skipped_for_role_worker_by_default():
    """A worker process with --role is never hidden by default."""
    assert _run_hide_probe({}, argv_tail=["--role", "actor"]) is False


def test_hide_skipped_for_ray_role_worker_by_default():
    """A Ray worker process with AREAL_ROLE_WORKER is never hidden by default."""
    assert _run_hide_probe({"AREAL_ROLE_WORKER": "1"}) is False


def test_hide_skipped_for_spmd_even_when_opted_in():
    """A worker that inherited the opt-in flag is still safe via AREAL_SPMD_MODE."""
    assert (
        _run_hide_probe({"AREAL_HIDE_CONTROLLER_DEVICES": "1", "AREAL_SPMD_MODE": "1"})
        is False
    )


def test_hide_skipped_for_role_worker_even_when_opted_in():
    """A worker that inherited the opt-in flag is still safe via --role."""
    assert (
        _run_hide_probe(
            {"AREAL_HIDE_CONTROLLER_DEVICES": "1"}, argv_tail=["--role", "actor"]
        )
        is False
    )


def test_hide_skipped_for_ray_role_worker_even_when_opted_in():
    """Ray engine actors carry no --role, so they are skipped via AREAL_ROLE_WORKER."""
    assert (
        _run_hide_probe(
            {"AREAL_HIDE_CONTROLLER_DEVICES": "1", "AREAL_ROLE_WORKER": "1"}
        )
        is False
    )


# ---------------------------------------------------------------------------
# Ray resource type must survive a hidden controller
# ---------------------------------------------------------------------------


def test_ray_resource_type_recovers_from_stash(monkeypatch):
    """A hidden controller reports GPU/NPU from the stash, not torch.cuda (CPU).

    ray_resource_type() runs on the controller to decide whether Ray requests GPU or
    CPU bundles. When the controller has blanked its own visibility, torch.cuda would
    report CPU and Ray would reject the GPU request; the stash env var must win.
    """
    pytest.importorskip("ray")
    from areal.infra.utils import ray as ray_utils

    monkeypatch.setenv("AREAL_CONTROLLER_HIDDEN_DEVICE_ENV", "CUDA_VISIBLE_DEVICES")
    assert ray_utils.ray_resource_type() == "GPU"

    monkeypatch.setenv(
        "AREAL_CONTROLLER_HIDDEN_DEVICE_ENV", "ASCEND_RT_VISIBLE_DEVICES"
    )
    assert ray_utils.ray_resource_type() == "NPU"


# ---------------------------------------------------------------------------
# End-to-end: a hidden controller process must not create a CUDA context
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="requires CUDA to verify no context is created"
)
def test_hidden_controller_import_does_not_init_cuda():
    """With AREAL_HIDE_CONTROLLER_DEVICES=1, `import areal` must stay CUDA-free."""
    import os

    code = textwrap.dedent(
        """
        import torch
        import areal  # pulls transformers/torchao
        assert not torch.cuda.is_initialized(), "import areal created a CUDA context"
        assert not torch.cuda.is_available(), "devices should be hidden on controller"
        print("OK")
        """
    )
    env = {
        "AREAL_HIDE_CONTROLLER_DEVICES": "1",
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }
    # Preserve PYTHONPATH so the subprocess can import the editable install.
    pp = os.environ.get("PYTHONPATH")
    if pp:
        env["PYTHONPATH"] = pp
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout

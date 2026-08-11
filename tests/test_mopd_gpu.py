# SPDX-License-Identifier: Apache-2.0

"""Opt-in hardware regression tests for persistent MOPD teacher residency."""

from __future__ import annotations

import math
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import pytest
import torch

from areal.infra.utils.proc import kill_process_tree

_RUN_8GPU_SMOKE = os.environ.get("AREAL_RUN_MOPD_8GPU_TEST", "").strip() == "1"
_MODEL_PATH_VARS = (
    "MOPD_STUDENT_MODEL_PATH",
    "MOPD_TEACHER_MODEL_PATH",
    "MOPD_GSM8K_PATH",
)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _assigned_cuda_device_count() -> int:
    if os.environ.get("AREAL_CONTROLLER_HIDDEN_DEVICE_ENV") == "CUDA_VISIBLE_DEVICES":
        original = os.environ.get("AREAL_CONTROLLER_ORIG_DEVICES", "")
        return len([device for device in original.split(",") if device])
    return torch.cuda.device_count()


def _run_mopd_cuda_case(case: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    _restore_controller_hidden_devices(env)
    env["AREAL_ROLE_WORKER"] = "1"
    repo_root = str(Path(__file__).resolve().parents[1])
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = repo_root if not pythonpath else f"{repo_root}:{pythonpath}"
    runner = Path(__file__).parent / "torchrun" / "run_mopd_loss_logits.py"
    return subprocess.run(
        [sys.executable, str(runner), "--case", case],
        capture_output=True,
        text=True,
        env=env,
        cwd=repo_root,
        timeout=120,
    )


@pytest.mark.gpu
@pytest.mark.skipif(_assigned_cuda_device_count() == 0, reason="requires one CUDA GPU")
def test_mopd_cuda_logits_and_reverse_kl_match_oracles():
    """CUDA token logits and the MOPD surrogate match independent oracles."""
    result = _run_mopd_cuda_case("logits_reverse_kl")
    assert result.returncode == 0, (
        f"CUDA logits/reverse-KL worker failed\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "Passed case=logits_reverse_kl" in result.stdout


@pytest.mark.gpu
@pytest.mark.skipif(_assigned_cuda_device_count() == 0, reason="requires one CUDA GPU")
def test_mopd_cuda_masked_ratio_overflow_stays_finite():
    """CUDA masked and active ratio extremes stay finite and bounded."""
    result = _run_mopd_cuda_case("masked_overflow")
    assert result.returncode == 0, (
        f"CUDA masked-overflow worker failed\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "Passed case=masked_overflow" in result.stdout


def _restore_controller_hidden_devices(env: dict[str, str]) -> None:
    """Let a fresh test controller see, then independently hide, its GPUs."""
    hidden_env = env.pop("AREAL_CONTROLLER_HIDDEN_DEVICE_ENV", None)
    if hidden_env:
        if env.pop("AREAL_CONTROLLER_ORIG_DEVICES_SET", "0") == "1":
            env[hidden_env] = env.pop("AREAL_CONTROLLER_ORIG_DEVICES", "")
        else:
            env.pop(hidden_env, None)
            env.pop("AREAL_CONTROLLER_ORIG_DEVICES", None)


def _session_members(session_id: int) -> list[int]:
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getsid(pid) == session_id:
                members.append(pid)
        except (PermissionError, ProcessLookupError):
            continue
    return members


def _cleanup_process_session(process: subprocess.Popen) -> None:
    if process.poll() is None:
        kill_process_tree(process.pid, timeout=10, graceful=False)
    for pid in _session_members(process.pid):
        kill_process_tree(pid, timeout=5, graceful=False)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _tail(path: Path, line_count: int = 200) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
    )


def _assert_ports_released(ports: set[int]) -> None:
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))


@pytest.mark.integration
@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.skipif(
    not _RUN_8GPU_SMOKE,
    reason="set AREAL_RUN_MOPD_8GPU_TEST=1 to run the persistent MOPD smoke",
)
def test_persistent_teacher_8gpu_three_step_smoke_releases_memory(tmp_path, request):
    """Real Qwen3 phases reuse eight teachers and release their CUDA weights."""
    assert _assigned_cuda_device_count() >= 8, (
        "AREAL_RUN_MOPD_8GPU_TEST=1 requires at least eight visible CUDA GPUs"
    )
    paths = {name: Path(os.environ.get(name, "")) for name in _MODEL_PATH_VARS}
    missing = [name for name, path in paths.items() if not path.is_dir()]
    assert not missing, f"Missing required MOPD directories: {', '.join(missing)}"

    repo_root = Path(__file__).resolve().parents[1]
    run_root = tmp_path / "run"
    trial_name = f"mopd-gpu-{uuid.uuid4().hex[:10]}"
    output_path = tmp_path / "train.log"
    env = os.environ.copy()
    _restore_controller_hidden_devices(env)
    env.setdefault("AREAL_ADMIN_API_KEY", secrets.token_hex(32))
    command = [
        sys.executable,
        "-m",
        "examples.mopd.gsm8k_qwen3_14b_to_0_6b",
        "--config",
        "examples/mopd/gsm8k_qwen3_14b_to_0_6b_local.yaml",
        "total_train_steps=3",
        "total_train_epochs=1",
        "gconfig.n_samples=1",
        "gconfig.max_new_tokens=64",
        "train_dataset.batch_size=1",
        "valid_dataset.batch_size=1",
        "rollout.max_concurrent_rollouts=2",
        "sglang.max_running_requests=2",
        f"trial_name={trial_name}",
        f"cluster.fileroot={run_root}",
        f"cluster.name_resolve.nfs_record_root={run_root / 'name-resolve'}",
    ]

    with output_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        request.addfinalizer(lambda: _cleanup_process_session(process))
        try:
            return_code = process.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            _cleanup_process_session(process)
            pytest.fail(f"MOPD GPU smoke timed out\n{_tail(output_path)}")

    assert return_code == 0, (
        f"MOPD GPU smoke exited with code {return_code}\n{_tail(output_path)}"
    )
    console = output_path.read_text(encoding="utf-8", errors="replace")
    ownership_events = re.findall(
        r"\[MOPD\] teacher (onload|offload) complete|offload done, (onloading actor)",
        console,
    )
    assert ownership_events == [
        ("offload", ""),
        ("", "onloading actor"),
        ("onload", ""),
        ("offload", ""),
        ("", "onloading actor"),
        ("onload", ""),
        ("offload", ""),
        ("", "onloading actor"),
    ]

    teacher_logs = list(run_root.rglob("mopd-teacher.log"))
    actor_logs = list(run_root.rglob("actor.log"))
    assert len(teacher_logs) == 1, f"Expected one teacher log, found {teacher_logs}"
    assert len(actor_logs) == 1, f"Expected one actor log, found {actor_logs}"
    teacher_log = teacher_logs[0].read_text(encoding="utf-8", errors="replace")
    actor_log = actor_logs[0].read_text(encoding="utf-8", errors="replace")

    spawn_events = re.findall(
        r"Forked worker for role 'mopd-teacher' index (\d+) spawned \(pid=(\d+)\)",
        actor_log,
    )
    assert len(spawn_events) == 8, spawn_events
    spawned = {int(rank): int(pid) for rank, pid in spawn_events}
    assert set(spawned) == set(range(8)), f"Unexpected teacher workers: {spawned}"
    assert teacher_log.count("Created Megatron weight residency adapter") == 8

    residency_events = re.findall(
        r"\[Megatron residency\] rank=(\d+) "
        r"phase=(before_offload|after_offload|after_onload) "
        r"allocated_gb=([0-9.]+) reserved_gb=([0-9.]+)",
        teacher_log,
    )
    stats: dict[int, dict[str, list[tuple[float, float]]]] = {
        rank: {"before_offload": [], "after_offload": [], "after_onload": []}
        for rank in range(8)
    }
    for rank_text, phase, allocated, reserved in residency_events:
        stats[int(rank_text)][phase].append((float(allocated), float(reserved)))

    for rank, rank_stats in stats.items():
        before_offload = rank_stats["before_offload"]
        after_offload = rank_stats["after_offload"]
        after_onload = rank_stats["after_onload"]
        assert len(before_offload) == len(after_offload) == 3, (rank, rank_stats)
        assert len(after_onload) == 2, (rank, rank_stats)
        assert all(
            resident_allocated - offloaded_allocated >= 2.0
            and resident_reserved - offloaded_reserved >= 2.0
            for (resident_allocated, resident_reserved), (
                offloaded_allocated,
                offloaded_reserved,
            ) in zip(before_offload, after_offload, strict=True)
        ), (rank, rank_stats)
        assert max(allocated for allocated, _ in after_offload) < 0.5, (
            rank,
            rank_stats,
        )
        assert max(reserved for _, reserved in after_offload) < 0.5, (
            rank,
            rank_stats,
        )

    published_versions = [
        int(version)
        for version in re.findall(r"Put writer version .* version=(\d+)", actor_log)
    ]
    assert Counter(published_versions) == Counter({1: 8, 2: 8, 3: 8})

    merged_logs = list(run_root.rglob("merged.log"))
    assert len(merged_logs) == 1, merged_logs
    merged_log = merged_logs[0].read_text(encoding="utf-8", errors="replace")
    finite_metrics = {
        "mopd_loss": re.findall(
            r"ppo_actor/update/mopd_loss/(?:avg|max|min)\s+│\s+(\S+)",
            merged_log,
        ),
        "new_logp": re.findall(
            r"ppo_actor/update/new_logp/(?:avg|max|min)\s+│\s+(\S+)",
            merged_log,
        ),
        "grad_norm": re.findall(r"ppo_actor/update/grad_norm\s+│\s+(\S+)", merged_log),
    }
    assert {name: len(values) for name, values in finite_metrics.items()} == {
        "mopd_loss": 9,
        "new_logp": 9,
        "grad_norm": 3,
    }
    assert all(
        math.isfinite(float(value))
        for values in finite_metrics.values()
        for value in values
    ), finite_metrics

    deadline = time.monotonic() + 30
    while (
        any(_pid_exists(pid) for pid in spawned.values())
        and time.monotonic() < deadline
    ):
        time.sleep(0.1)
    live_pids = [pid for pid in spawned.values() if _pid_exists(pid)]
    assert not live_pids, f"Teacher worker PIDs survived shutdown: {live_pids}"

    deadline = time.monotonic() + 30
    while _session_members(process.pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    session_pids = _session_members(process.pid)
    assert not session_pids, f"MOPD subprocess descendants survived: {session_pids}"

    guard_events = re.findall(
        r"Starting Guard on [^: ]+:(\d+) for worker mopd-teacher/(\d+)",
        teacher_log,
    )
    assert len(guard_events) == 8, guard_events
    ports_by_rank = {int(rank): int(port) for port, rank in guard_events}
    assert set(ports_by_rank) == set(range(8)), ports_by_rank
    _assert_ports_released(set(ports_by_rank.values()))

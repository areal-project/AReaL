# SPDX-License-Identifier: Apache-2.0

"""Local engine process ownership and GPU release confirmation."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence

import psutil

# Keep the existing launch token so live engines and saved ownership records agree.
OWNER_ENV = "AREAL_ELASTIC_PROCESS_OWNER"


def owned_processes(owner: str) -> list[psutil.Process]:
    if not owner:
        raise ValueError("Process owner token must be nonempty")
    found = []
    for process in psutil.process_iter():
        try:
            if (
                process.status() != psutil.STATUS_ZOMBIE
                and process.environ().get(OWNER_ENV) == owner
            ):
                found.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def stop_owned_processes(
    owner: str, *, grace: float = 10, force_timeout: float = 10
) -> None:
    """Stop token-owned descendants and fence PID reuse while signalling."""
    for operation, timeout in (("terminate", grace), ("kill", force_timeout)):
        deadline = time.monotonic() + timeout
        signalled = set()
        while processes := owned_processes(owner):
            for process in processes:
                try:
                    identity = (process.pid, process.create_time())
                    if identity not in signalled:
                        getattr(process, operation)()
                        signalled.add(identity)
                except psutil.NoSuchProcess:
                    pass
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        if not owned_processes(owner):
            return
    raise RuntimeError(
        f"Owned processes remain alive: {[p.pid for p in owned_processes(owner)]}"
    )


def gpu_processes() -> dict[str, list[int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    processes: dict[str, list[int]] = {}
    for line in result.stdout.splitlines():
        gpu, pid = (part.strip() for part in line.split(","))
        processes.setdefault(gpu, []).append(int(pid))
    return processes


def resolve_gpu_uuids(devices: list[int]) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    mapping = {
        int(index.strip()): uuid.strip()
        for index, uuid in (line.split(",") for line in result.stdout.splitlines())
    }
    return [mapping[index] for index in devices]


def wait_for_gpu_release(devices: Sequence[int], *, timeout: float = 30) -> None:
    """Wait until the declared local GPUs have no remaining compute processes."""
    if not devices:
        return
    uuids = resolve_gpu_uuids(list(devices))
    deadline = time.monotonic() + timeout
    while True:
        processes = gpu_processes()
        occupied = {gpu: processes[gpu] for gpu in uuids if processes.get(gpu)}
        if not occupied:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU release is unconfirmed: {occupied}")
        time.sleep(0.1)

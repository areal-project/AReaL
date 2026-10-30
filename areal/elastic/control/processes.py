# SPDX-License-Identifier: Apache-2.0

"""Identify owned descendants even after their leader exits or they change sessions."""

import subprocess
import time

import psutil

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
    """Signal only processes carrying this launch's random token; fence PID reuse.

    Poll again for late descendants instead of treating leader exit as group exit.
    psutil's signal methods check process identity (PID and creation time).
    """
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

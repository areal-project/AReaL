# SPDX-License-Identifier: Apache-2.0

"""Assemble Ray clusters on exclusively allocated nodes.

``python -m areal.infra.launcher.ray`` calls into this module when it detects
that it runs inside a multi-node platform job without a pre-assembled Ray
cluster: the same launcher command is used as the job command on EVERY node,
rank 0 starts the Ray head and continues into the launcher, and the remaining
ranks join as Ray workers until the head shuts down.

Node rank is detected from ``AREAL_NODE_RANK``/``RANK``/``OMPI_COMM_WORLD_RANK``,
from a platform role name (``PAI_CURRENT_TASK_ROLE_NAME``/
``AISTUDIO_TASK_ROLE_NAME``), or from a ``POD_NAME`` containing "master".
If none of these exist, the process is not considered part of a platform job.

Only per-node, platform-injected identity is read from environment variables
(they differ per node and cannot live in the shared config):
    AREAL_NODE_RANK / RANK / OMPI_COMM_WORLD_RANK: node rank.
    AREAL_NODE_IP / POD_IP: node IP override (default: first ``hostname -I``).
    AREAL_MASTER_ADDR / MASTER_ADDR: head address, required on worker nodes.
Tunable knobs (Ray ports, bootstrap wait timeout) come from
``cluster.ray_port`` / ``cluster.ray_dashboard_port`` /
``cluster.ray_bootstrap_timeout_seconds`` in the experiment config.

Note: environment variables read by ray at import time (e.g.
``RAY_DEDUP_LOGS``) must be exported before starting python -- the areal
package imports ray transitively, so setting them here would be too late.
Likewise, dependency installation (``uv pip install -e .``) must happen
before invoking the launcher.

Other job runners can pass explicit node information to ``run_node`` (or the
CLI) to start Ray and run a separate command on the head. Workers can discover
the head using a per-job file on shared storage or an explicit address.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import psutil
import ray

import areal.utils.logging as logging

logger = logging.getLogger("RayBootstrap")

DEFAULT_PORT = 6379
DEFAULT_DASHBOARD_PORT = 8265
DEFAULT_TIMEOUT = 900
POLL_INTERVAL = 1

NODE_WAIT_CHECK_INTERVAL = 5  # seconds
WORKER_JOIN_RETRY_INTERVAL = 5  # seconds
WORKER_ALIVE_CHECK_INTERVAL = 30  # seconds
# Consecutive `ray status` failures before a worker deems the head gone.
# `ray status` also fails transiently while the GCS is merely overloaded
# (e.g. during NFS-heavy checkpoints), so this fallback window is long
# (20 x 30s = 10 min); the primary shutdown signal is the local raylet
# exiting, which happens by itself once the head is truly gone.
WORKER_ALIVE_MAX_FAILURES = 20

_RANK_ENVS = ("AREAL_NODE_RANK", "RANK", "OMPI_COMM_WORLD_RANK")
_ROLE_ENVS = ("PAI_CURRENT_TASK_ROLE_NAME", "AISTUDIO_TASK_ROLE_NAME")


def _accelerator_cli_args(resource_name: str, count: int) -> list[str]:
    if not resource_name:
        raise ValueError("Ray accelerator resource name must not be empty")
    if count < 0:
        raise ValueError(f"Ray accelerator count must be non-negative, got {count}")
    if resource_name == "GPU":
        return [f"--num-gpus={count}"]
    return [f"--resources={json.dumps({resource_name: count})}"]


def _ray_cli() -> str:
    """Path of the `ray` CLI matching the running interpreter's ray package.

    A bare "ray" resolves via PATH, which may belong to a different
    python/ray installation than this process (e.g. system conda vs the
    project venv) and then fails with a version mismatch when this process
    connects via `ray.init`.
    """
    candidate = Path(sys.executable).with_name("ray")
    if candidate.exists():
        return str(candidate)
    return "ray"


def detect_node_rank() -> int | None:
    """Detect this node's rank in a platform job (0 = master).

    Returns None when no rank/role environment variable exists at all,
    i.e. this process does not look like part of a gang-scheduled job.
    """
    for key in _RANK_ENVS:
        value = os.environ.get(key)
        if value:
            return int(value)
    has_role_signal = False
    for key in _ROLE_ENVS:
        value = os.environ.get(key)
        if value:
            has_role_signal = True
            if "master" in value.lower():
                return 0
    # POD_NAME exists on ANY k8s pod (including dev boxes), so its mere
    # presence is not evidence of a gang-scheduled job -- only a master-ish
    # name is meaningful.
    if "master" in os.environ.get("POD_NAME", "").lower():
        return 0
    return 1 if has_role_signal else None


def detect_node_ip() -> str:
    """Detect the IP address other nodes can reach this node at."""
    ip = os.environ.get("AREAL_NODE_IP") or os.environ.get("POD_IP")
    if ip:
        return ip
    result = subprocess.run(
        ["hostname", "-I"], capture_output=True, text=True, check=False
    )
    if result.returncode == 0 and result.stdout.split():
        return result.stdout.split()[0]
    return socket.gethostbyname(socket.gethostname())


def stop_local_ray():
    """Force-stop any Ray processes on this node (e.g. stale ones)."""
    subprocess.run(
        [_ray_cli(), "stop", "--force"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _local_raylet_alive() -> bool:
    """Whether a raylet process is running on this node.

    The raylet exits by itself once the head (GCS) is permanently gone, so
    its process liveness is a more reliable shutdown signal than `ray
    status`, which also fails transiently while the GCS is merely under
    load.
    """
    try:
        for proc in psutil.process_iter(["name"]):
            if proc.info["name"] == "raylet":
                return True
        return False
    except Exception:
        # If process inspection is unavailable, fall back to the
        # `ray status` failure counting alone.
        return True


def wait_for_ray_nodes(
    expected_nodes: int,
    timeout: int,
    accelerator_resource: str = "GPU",
):
    deadline = time.time() + timeout
    while time.time() < deadline:
        alive = [node for node in ray.nodes() if node.get("Alive")]
        accelerator_total = sum(
            node.get("Resources", {}).get(accelerator_resource, 0) for node in alive
        )
        logger.info(
            f"Ray cluster: {len(alive)}/{expected_nodes} nodes alive, "
            f"{accelerator_total} {accelerator_resource} devices visible"
        )
        if len(alive) >= expected_nodes:
            return
        time.sleep(NODE_WAIT_CHECK_INTERVAL)
    raise TimeoutError(f"Timed out waiting for {expected_nodes} Ray nodes")


def bootstrap_head(
    node_ip: str,
    n_nodes: int,
    n_gpus_per_node: int,
    ray_port: int = DEFAULT_PORT,
    dashboard_port: int = DEFAULT_DASHBOARD_PORT,
    wait_timeout: int = DEFAULT_TIMEOUT,
    accelerator_resource: str = "GPU",
):
    """Start the Ray head on this node and wait for all nodes to join.

    Returns with `ray.init` already connected to the assembled cluster, so
    the caller can proceed into the launcher.
    """
    logger.info(f"Starting Ray head at {node_ip}:{ray_port}")
    subprocess.run(
        [
            _ray_cli(),
            "start",
            "--head",
            f"--node-ip-address={node_ip}",
            f"--port={ray_port}",
            "--dashboard-host=0.0.0.0",
            f"--dashboard-port={dashboard_port}",
            *_accelerator_cli_args(accelerator_resource, n_gpus_per_node),
        ],
        check=True,
    )
    ray.init(address="auto")
    wait_for_ray_nodes(
        n_nodes,
        wait_timeout,
        accelerator_resource=accelerator_resource,
    )
    subprocess.run([_ray_cli(), "status"], check=False)


def _wait_until_cluster_gone(
    check_interval: int = WORKER_ALIVE_CHECK_INTERVAL,
    max_failures: int = WORKER_ALIVE_MAX_FAILURES,
):
    """Block until the Ray cluster this node has joined shuts down."""
    failures = 0
    while True:
        time.sleep(check_interval)
        # Primary signal: the local raylet exits on its own once the head
        # is permanently gone (it is also gone if someone ran `ray stop`).
        if not _local_raylet_alive():
            logger.info("Local raylet has exited; worker exiting")
            return
        # Fallback: a long window of consecutive `ray status` failures while
        # the raylet is still up (e.g. the raylet itself is wedged). Kept
        # deliberately long -- `ray status` also fails transiently when the
        # GCS is merely overloaded, and a false positive here would kill
        # healthy trainer ranks on this node.
        result = subprocess.run(
            [_ray_cli(), "status"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        failures = failures + 1 if result.returncode != 0 else 0
        if failures >= max_failures:
            logger.info(
                f"Ray head is unreachable ({failures} consecutive failures "
                f"over {failures * check_interval}s); worker exiting"
            )
            return


def bootstrap_worker(
    node_ip: str,
    n_gpus_per_node: int,
    ray_port: int = DEFAULT_PORT,
    wait_timeout: int = DEFAULT_TIMEOUT,
    accelerator_resource: str = "GPU",
):
    """Join the Ray head as a worker and block until the head shuts down."""
    master_addr = os.environ.get("AREAL_MASTER_ADDR") or os.environ.get("MASTER_ADDR")
    if not master_addr:
        raise RuntimeError(
            "Worker node cannot find the Ray head. "
            "Set AREAL_MASTER_ADDR or ensure MASTER_ADDR is provided."
        )
    if wait_timeout <= 0:
        raise ValueError(
            f"Ray worker join timeout must be positive, got {wait_timeout}"
        )

    logger.info(f"Starting Ray worker, connecting to {master_addr}:{ray_port}")
    deadline = time.monotonic() + wait_timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out joining Ray head at {master_addr}:{ray_port}"
                )
            try:
                result = subprocess.run(
                    [
                        _ray_cli(),
                        "start",
                        f"--address={master_addr}:{ray_port}",
                        f"--node-ip-address={node_ip}",
                        *_accelerator_cli_args(
                            accelerator_resource,
                            n_gpus_per_node,
                        ),
                    ],
                    check=False,
                    timeout=remaining,
                )
            except subprocess.TimeoutExpired as e:
                raise TimeoutError(
                    f"Timed out joining Ray head at {master_addr}:{ray_port}"
                ) from e
            if result.returncode == 0:
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out joining Ray head at {master_addr}:{ray_port}"
                )
            sleep_seconds = min(WORKER_JOIN_RETRY_INTERVAL, remaining)
            logger.info(f"Ray head is not ready yet; retrying in {sleep_seconds:.1f}s")
            time.sleep(sleep_seconds)

        logger.info("Ray worker is up; waiting for the cluster to shut down")
        _wait_until_cluster_gone()
    finally:
        stop_local_ray()


def split_address(address: str) -> tuple[str, int]:
    host, sep, port = address.rpartition(":")
    if not sep or not host or not port.isascii() or not port.isdigit():
        raise ValueError(f"Expected a Ray address of the form host:port: {address!r}")
    if not 1 <= int(port) <= 65535:
        raise ValueError("The Ray port must be between 1 and 65535.")
    return host, int(port)


@dataclass(frozen=True)
class BootstrapConfig:
    role: Literal["head", "worker"]
    node_ip: str
    nodes: int
    gpus_per_node: int
    port: int = DEFAULT_PORT
    dashboard_port: int = DEFAULT_DASHBOARD_PORT
    timeout: int = DEFAULT_TIMEOUT
    head_address: str | None = None
    rendezvous_file: Path | None = None

    def __post_init__(self) -> None:
        if self.role not in ("head", "worker") or not self.node_ip:
            raise ValueError("Specify a head or worker role and a node IP.")
        if self.nodes < 1 or self.gpus_per_node < 0 or self.timeout < 1:
            raise ValueError(
                "Specify a positive node count and timeout, and nonnegative GPUs."
            )
        if not 1 <= self.port <= 65535 or not 1 <= self.dashboard_port <= 65535:
            raise ValueError("Ray ports must be between 1 and 65535.")
        if self.role == "head" and self.head_address:
            raise ValueError("Only a worker can specify a head address.")
        if self.role == "worker" and bool(self.head_address) == bool(
            self.rendezvous_file
        ):
            raise ValueError(
                "A worker needs exactly one head address or rendezvous file."
            )
        if self.head_address:
            split_address(self.head_address)

    @property
    def local_address(self) -> str:
        host = f"[{self.node_ip}]" if ":" in self.node_ip else self.node_ip
        return f"{host}:{self.port}"


def reserve_rendezvous(path: Path) -> None:
    """Refuse to reuse either an active or a stale per-job address file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8"):
        pass


def publish_head(path: Path, address: str) -> None:
    staging = path.with_name(path.name + ".tmp")
    with staging.open("x", encoding="utf-8") as output:
        output.write(address + "\n")
    staging.replace(path)


def wait_for_head(path: Path, deadline: float) -> str:
    while time.monotonic() < deadline:
        try:
            address = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            pass
        else:
            if address:
                split_address(address)
                return address
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Timed out waiting for the Ray head at {path}.")


def ready_for_training(
    nodes: list[dict], available_gpus: float, expected_nodes: int, gpus_per_node: int
) -> None:
    gpu_counts = [node["Resources"].get("GPU", 0) for node in nodes if node["Alive"]]
    if len(gpu_counts) != expected_nodes or any(n != gpus_per_node for n in gpu_counts):
        raise RuntimeError(
            f"Expected {expected_nodes} × {gpus_per_node} GPUs; got {gpu_counts}."
        )
    if available_gpus < expected_nodes * gpus_per_node:
        raise RuntimeError(f"Only {available_gpus:g} Ray GPUs are available.")


@contextmanager
def termination_signals() -> Iterator[None]:
    """Run finally blocks when the job runner sends SIGTERM."""

    def stop(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_node(
    config: BootstrapConfig,
    command: Sequence[str] = (),
    *,
    command_env: Mapping[str, str] | None = None,
) -> int:
    """Start Ray on one exclusive node; execute the command only on the head."""
    path = config.rendezvous_file
    child_env = dict(os.environ if command_env is None else command_env)
    with patch.dict(os.environ):
        os.environ.pop("RAY_ADDRESS", None)
        if config.role == "worker":
            address = config.head_address
            if address is None:
                assert path is not None
                address = wait_for_head(path, time.monotonic() + config.timeout)
            host, port = split_address(address)
            os.environ["AREAL_MASTER_ADDR"] = host
            bootstrap_worker(
                config.node_ip,
                config.gpus_per_node,
                ray_port=port,
                wait_timeout=config.timeout,
            )
            return 0

        if not command:
            raise ValueError("The head needs a command to run after Ray is ready.")
        address = config.local_address
        if path is not None:
            reserve_rendezvous(path)
            # Workers retry while the GCS is starting, and the existing head
            # bootstrap waits for them before returning.
            publish_head(path, address)
        try:
            try:
                bootstrap_head(
                    config.node_ip,
                    config.nodes,
                    config.gpus_per_node,
                    ray_port=config.port,
                    dashboard_port=config.dashboard_port,
                    wait_timeout=config.timeout,
                )
                ready_for_training(
                    ray.nodes(),
                    ray.available_resources().get("GPU", 0),
                    config.nodes,
                    config.gpus_per_node,
                )
            finally:
                ray.shutdown()
            logger.info("Ray ready: %s × %s GPUs", config.nodes, config.gpus_per_node)
            child_env["RAY_ADDRESS"] = address
            code = subprocess.run(command, env=child_env, check=False).returncode
            return code if code >= 0 else 128 - code
        finally:
            # The caller must allocate exclusive nodes, one Ray session per node.
            stop_local_ray()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start Ray and run a command on the head.", allow_abbrev=False
    )
    parser.add_argument("--role", choices=("head", "worker"), required=True)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dashboard-port", type=int, default=DEFAULT_DASHBOARD_PORT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    endpoint = parser.add_mutually_exclusive_group()
    endpoint.add_argument("--head-address")
    endpoint.add_argument("--rendezvous-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    kwargs = vars(parser.parse_args())
    command = kwargs.pop("command")
    if command and command[0] == "--":
        command = command[1:]
    with termination_signals():
        return run_node(BootstrapConfig(**kwargs), command)


if __name__ == "__main__":
    raise SystemExit(main())

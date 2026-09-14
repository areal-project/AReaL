# SPDX-License-Identifier: Apache-2.0

"""Ray cluster bootstrap helpers for gang-scheduled platform jobs (e.g. AIS/PAI).

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
"""

import json
import os
import pathlib
import socket
import subprocess
import sys
import time

import psutil
import ray

import areal.utils.logging as logging

logger = logging.getLogger("RayBootstrap")

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
    candidate = pathlib.Path(sys.executable).with_name("ray")
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
    ray_port: int = 6379,
    dashboard_port: int = 8265,
    wait_timeout: int = 900,
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
    ray_port: int = 6379,
    wait_timeout: int = 900,
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

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import getpass
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import ray
import requests

from areal.infra.utils.proc import kill_process_tree, run_with_streaming_logs
from areal.utils import logging
from areal.utils.network import format_hostport, gethostip

logger = logging.getLogger("RayHTTPWorkerManager")


class _RayHTTPWorkerManagerImpl:
    """Ray-managed lifecycle manager for HTTP worker subprocesses.

    This is the RayScheduler counterpart of the Local parent guard's
    ``/fork`` / ``/kill_forked_worker`` path. The manager actor starts,
    health-checks, and tears down an existing HTTP server module such as
    ``areal.experimental.openai.proxy.proxy_rollout_server``. It is lifecycle
    only: engine/control requests must go directly to the managed HTTP worker
    endpoint instead of being proxied through this actor.
    """

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._host: str | None = None
        self._port: int | None = None
        self._log_file: str | None = None

    def get_node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()

    def _read_log_tail(self, lines: int = 80) -> str:
        if self._log_file is None:
            return "[No log file was created.]"
        try:
            with open(self._log_file) as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception as e:
            return f"[Could not read log file {self._log_file}: {e}]"

    def _health_url(self) -> str:
        if self._host is None or self._port is None:
            raise RuntimeError("HTTP server has not been launched")
        return f"http://{format_hostport(self._host, self._port)}/health"

    def _get_advertised_host(self, bind_host: str) -> str:
        if bind_host not in ("0.0.0.0", "::"):
            return bind_host
        try:
            node_ip = ray.util.get_node_ip_address()
            if node_ip:
                return str(node_ip)
        except Exception:
            pass
        return gethostip()

    def _wait_ready(self, startup_timeout: float) -> None:
        deadline = time.time() + startup_timeout
        last_error = "health check did not run"
        while time.time() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    "HTTP server exited during startup with code "
                    f"{self._process.returncode}.\nLog tail:\n{self._read_log_tail()}"
                )

            try:
                response = requests.get(self._health_url(), timeout=2.0)
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except Exception as e:
                last_error = str(e)
            time.sleep(0.2)

        raise TimeoutError(
            f"HTTP server did not become ready within {startup_timeout}s: {last_error}."
            f"\nLog tail:\n{self._read_log_tail()}"
        )

    def launch(
        self,
        *,
        module: str,
        host: str,
        port: int,
        experiment_name: str,
        trial_name: str,
        role: str,
        worker_index: int,
        name_resolve_type: str,
        nfs_record_root: str,
        etcd3_addr: str,
        fileroot: str,
        env: dict[str, str] | None = None,
        startup_timeout: float = 30.0,
    ) -> dict[str, Any]:
        if self._process is not None and self._process.poll() is None:
            raise RuntimeError(
                f"HTTP server is already running at {self._host}:{self._port} "
                f"(pid={self._process.pid})"
            )

        advertised_host = self._get_advertised_host(host)
        self._host = advertised_host
        self._port = int(port)

        log_dir = (
            Path(fileroot) / "logs" / getpass.getuser() / experiment_name / trial_name
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        log_label = f"{role}-{worker_index}"
        log_file = log_dir / f"{role}-{worker_index}.log"
        merged_log = log_dir / "merged.log"
        self._log_file = str(log_file)

        cmd = [
            sys.executable,
            "-m",
            module,
            "--host",
            host,
            "--port",
            str(port),
            "--experiment-name",
            experiment_name,
            "--trial-name",
            trial_name,
            "--role",
            role,
            "--worker-index",
            str(worker_index),
            "--name-resolve-type",
            name_resolve_type,
            "--nfs-record-root",
            nfs_record_root,
            "--etcd3-addr",
            etcd3_addr,
            "--fileroot",
            fileroot,
        ]

        process_env = os.environ.copy()
        if env:
            process_env.update({str(k): str(v) for k, v in env.items()})

        logger.info(
            "Launching HTTP worker %s/%s on %s:%s with module %s",
            role,
            worker_index,
            advertised_host,
            port,
            module,
        )
        try:
            self._process = run_with_streaming_logs(
                cmd,
                log_file,
                merged_log,
                log_label,
                env=process_env,
            )
            self._wait_ready(startup_timeout)
        except Exception:
            self.destroy(exit_actor=False)
            raise

        return {
            "host": advertised_host,
            "port": int(port),
            "pid": self._process.pid,
            "node_id": self.get_node_id(),
            "log_file": str(log_file),
        }

    def ping(self) -> str:
        if self._process is None:
            raise RuntimeError("HTTP server has not been launched")
        if self._process.poll() is not None:
            raise RuntimeError(
                "HTTP server process exited with code "
                f"{self._process.returncode}.\nLog tail:\n{self._read_log_tail()}"
            )
        response = requests.get(self._health_url(), timeout=2.0)
        if response.status_code != 200:
            raise RuntimeError(
                f"HTTP health check failed with status {response.status_code}: "
                f"{response.text[:500]}"
            )
        return "ok"

    def destroy(self, exit_actor: bool = True) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            try:
                kill_process_tree(process.pid, timeout=5, graceful=True)
            except Exception as e:
                logger.warning(
                    "Failed to gracefully stop HTTP worker process %s: %s",
                    process.pid,
                    e,
                )
                try:
                    process.kill()
                except Exception:
                    pass
        if exit_actor:
            ray.actor.exit_actor()


@ray.remote
class RayHTTPWorkerManager(_RayHTTPWorkerManagerImpl):
    pass

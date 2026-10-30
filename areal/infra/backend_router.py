# SPDX-License-Identifier: Apache-2.0

"""v1 lifecycle for the shared v2 Router CPU service."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import requests

from areal.api.cli_args import BackendRoutingConfig
from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging
from areal.utils.network import find_free_ports, format_hostport, gethostip
from areal.v2.inference_service.router.backend import RoutingError

logger = logging.getLogger("BackendRouter")


class BackendRouterService:
    def __init__(self, config: BackendRoutingConfig, log_dir: str, role: str):
        self.config = config
        self.run_id = uuid.uuid4().hex
        self.pool_id = f"{role}-{self.run_id}"
        self.admin_key = config.admin_api_key or secrets.token_urlsafe(32)
        self.address = config.router_addr
        self.process: subprocess.Popen | None = None
        self.log_file = None
        self.log_dir = Path(log_dir)
        self.role = role
        self.control_seq = -1
        self.state = "PAUSED"
        self.version = 0
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.headers["Authorization"] = f"Bearer {self.admin_key}"
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._monitor: threading.Thread | None = None
        self._error: BaseException | None = None

    def start(self, backends: list[dict]) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.address is None:
            port = find_free_ports(1)[0]
            host = gethostip()
            self.address = f"http://{format_hostport(host, port)}"
            env = os.environ.copy()
            env["AREAL_ROUTER_ADMIN_API_KEY"] = self.admin_key
            self.log_file = (self.log_dir / f"{self.role}-backend-router.log").open("a")
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "areal.v2.inference_service.router",
                    "--host",
                    host,
                    "--port",
                    str(port),
                ],
                env=env,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        deadline = time.monotonic() + self.config.control_timeout
        while True:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError("Router subprocess exited; inspect its log")
            try:
                self._session.get(
                    f"{self.address}/health", timeout=2
                ).raise_for_status()
                break
            except requests.RequestException:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Router did not become healthy") from None
                time.sleep(0.2)
        self._request(
            "POST",
            "/routing/backends/register",
            {
                "pool_id": self.pool_id,
                "run_id": self.run_id,
                "backends": backends,
                "weight_version": self.version,
            },
        )
        self.set_state("PAUSED")
        receipt = {
            "address": self.address,
            "pool_id": self.pool_id,
            "run_id": self.run_id,
            "pid": self.process.pid if self.process else None,
        }
        (self.log_dir / f"{self.role}-backend-router.json").write_text(
            json.dumps(receipt, indent=2)
        )
        logger.info("Backend Router ready for registration: %s", receipt)
        self._monitor = threading.Thread(target=self._watch, daemon=True)
        self._monitor.start()

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        with self._lock:
            deadline = time.monotonic() + self.config.metadata_timeout
            while True:
                try:
                    response = self._session.request(
                        method,
                        f"{self.address}{path}",
                        json=body if method == "POST" else None,
                        params={"run_id": self.run_id} if method == "GET" else None,
                        timeout=min(5.0, self.config.metadata_timeout),
                    )
                    data = response.json()
                    if response.status_code != 200:
                        detail = data.get("detail", {})
                        if isinstance(detail, dict) and "code" in detail:
                            raise RoutingError(detail["code"])
                        response.raise_for_status()
                    return data
                except (requests.RequestException, ValueError):
                    if time.monotonic() >= deadline:
                        raise RoutingError("ROUTER_UNREACHABLE") from None
                    time.sleep(0.1)

    def client_kwargs(self) -> dict:
        return dict(
            router_addr=self.address,
            pool_id=self.pool_id,
            run_id=self.run_id,
            admin_api_key=self.admin_key,
            metadata_timeout=self.config.metadata_timeout,
            control_timeout=self.config.control_timeout,
        )

    def set_state(self, state: str, version: int | None = None) -> None:
        with self._lock:
            self.check_health()
            version = self.version if version is None else version
            seq = self.control_seq + 1
            try:
                self._request(
                    "POST",
                    f"/routing/pools/{self.pool_id}/state",
                    {
                        "run_id": self.run_id,
                        "control_seq": seq,
                        "state": state,
                        "weight_version": version,
                    },
                )
            except Exception as exc:
                # The Router may have applied the transition before the reply
                # was lost. Never reopen it from an unconfirmed local state.
                self.fail(exc)
                raise
            self.state, self.version, self.control_seq = state, version, seq

    def drain(self, *, abort_inflight: Callable[[], None] | None = None) -> None:
        """Release admitted leases before closing the native generation gate.

        Repeat abort-only requests because a transport admitted before the sender
        fence can reach a backend after an earlier abort has already completed.
        The callback must keep native admission open until outstanding is zero.
        """
        if self.state != "PAUSED":
            raise RuntimeError("Close Router admission before draining")
        deadline = time.monotonic() + self.config.control_timeout
        next_abort = 0.0
        while True:
            self.check_health()
            state = self._request("GET", f"/routing/pools/{self.pool_id}/state")
            if state["state"] != "PAUSED":
                raise RoutingError(state["state"])
            if state["outstanding"] == 0:
                return
            if time.monotonic() >= deadline:
                self.set_state("FAILED")
                raise TimeoutError(
                    f"Router still owns {state['outstanding']} executions"
                )
            if abort_inflight is not None and time.monotonic() >= next_abort:
                try:
                    abort_inflight()
                except Exception as exc:
                    self.fail(exc)
                    raise
                next_abort = time.monotonic() + 1.0
            time.sleep(0.05)

    def check_health(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                "Backend routing failed; restart the whole experiment"
            ) from self._error

    def fail(self, error: BaseException) -> None:
        with self._lock:
            self._error = error
            self.state = "FAILED"
            try:
                state = self._request("GET", f"/routing/pools/{self.pool_id}/state")
                self._request(
                    "POST",
                    f"/routing/pools/{self.pool_id}/state",
                    {
                        "run_id": self.run_id,
                        "control_seq": state["control_seq"] + 1,
                        "state": "FAILED",
                        "weight_version": state["weight_version"],
                    },
                )
            except Exception:
                logger.exception("Could not confirm failed pool state")

    def _watch(self) -> None:
        while not self._stop.wait(2.0):
            try:
                if self.process is not None and self.process.poll() is not None:
                    raise RuntimeError("Router subprocess exited")
                state = self._request("GET", f"/routing/pools/{self.pool_id}/state")
                if state["state"] in ("FAILED", "CLOSED"):
                    raise RoutingError(state["state"])
            except Exception as exc:
                self.fail(exc)
                logger.exception("Backend Router failed")
                return

    def destroy(self, *, old_processes_stopped: bool = False) -> None:
        self._stop.set()
        if self._monitor is not None:
            self._monitor.join(timeout=self.config.metadata_timeout + 5)
        try:
            if self.process is not None:
                kill_process_tree(self.process.pid)
                self.process.wait(timeout=10)
            elif self.address:
                # Shared services outlive the experiment; never kill them.
                state = self._request("GET", f"/routing/pools/{self.pool_id}/state")
                self.control_seq = state["control_seq"]
                if old_processes_stopped:
                    self._request(
                        "POST",
                        f"/routing/pools/{self.pool_id}/state",
                        {
                            "run_id": self.run_id,
                            "control_seq": state["control_seq"] + 1,
                            "state": "CLOSED",
                            "weight_version": state["weight_version"],
                            "old_processes_stopped": True,
                        },
                    )
                elif state["state"] != "CLOSED":
                    # Leave the pool fenced until the operator confirms all
                    # old workers/backends exited and explicitly retires it.
                    self.fail(RuntimeError("Experiment is shutting down"))
        finally:
            self._session.close()
            if self.log_file is not None:
                self.log_file.close()

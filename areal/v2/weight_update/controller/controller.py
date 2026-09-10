# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

import httpx

from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging
from areal.utils.network import find_free_ports
from areal.v2.weight_update.controller.config import (
    WeightUpdateControllerConfig,
)
from areal.v2.weight_update.gateway.config import WeightUpdateResult

logger = logging.getLogger("WeightUpdateController")


class WeightUpdateError(RuntimeError):
    """A gateway-reported update failure with its inference safety status."""

    def __init__(
        self,
        message: str,
        *,
        inference_weights_may_be_mutated: bool,
    ) -> None:
        super().__init__(message)
        self.inference_weights_may_be_mutated = inference_weights_may_be_mutated


class WeightUpdateController:
    def __init__(self, config: WeightUpdateControllerConfig | None = None) -> None:
        self.config = config or WeightUpdateControllerConfig()
        self._gateway_url: str = ""
        self._gateway_proc: subprocess.Popen | None = None
        self._pair_name: str | None = None
        self._operation_id: str | None = None
        self._port_guard_url: str = ""
        self._port_token: str | None = None
        self._workers_cleaned = False
        self._train_worker_urls: list[str] = []
        self._inference_worker_urls: list[str] = []
        self._session: httpx.Client | None = None

    @property
    def gateway_url(self) -> str:
        return self._gateway_url

    @property
    def pair_name(self) -> str | None:
        return self._pair_name

    @property
    def train_worker_urls(self) -> list[str]:
        return list(self._train_worker_urls)

    @property
    def operation_id(self) -> str | None:
        return self._operation_id

    @property
    def port_token(self) -> str | None:
        return self._port_token

    def retain_port_reservation(self, guard_url: str, token: str) -> None:
        self._port_guard_url = guard_url
        self._port_token = token

    @property
    def workers_cleaned(self) -> bool:
        return self._workers_cleaned

    def mark_workers_cleaned(self) -> None:
        self._workers_cleaned = True
        self._pair_name = None
        self._operation_id = None
        self._train_worker_urls = []
        self._inference_worker_urls = []

    def retain_pending_connection(
        self,
        pair_name: str,
        operation_id: str,
        train_worker_urls: list[str],
        inference_worker_urls: list[str],
    ) -> None:
        self._pair_name = pair_name
        self._operation_id = operation_id
        self._train_worker_urls = list(train_worker_urls)
        self._inference_worker_urls = list(inference_worker_urls)
        self._workers_cleaned = False

    def release_port_reservation(self, timeout: float) -> None:
        if self._port_token is None:
            return
        resp = httpx.post(
            f"{self._port_guard_url}/release_ports",
            json={"token": self._port_token},
            timeout=timeout,
        )
        resp.raise_for_status()
        self._port_guard_url = ""
        self._port_token = None

    @property
    def inference_worker_urls(self) -> list[str]:
        return list(self._inference_worker_urls)

    @property
    def _http(self) -> httpx.Client:
        if self._session is None:
            raise RuntimeError("Controller not initialized. Call initialize() first.")
        return self._session

    def initialize(self, timeout: float | None = None) -> None:
        cfg = self.config
        port = cfg.port
        if port == 0:
            port = find_free_ports(1)[0]

        cmd = [
            sys.executable,
            "-m",
            "areal.v2.weight_update.gateway",
            "--host",
            cfg.host,
            "--port",
            str(port),
            "--admin-api-key",
            cfg.admin_api_key,
            "--init-timeout",
            str(cfg.init_timeout_s),
            "--update-timeout",
            str(cfg.update_timeout_s),
            "--log-level",
            cfg.log_level,
        ]

        try:
            self._gateway_proc = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stdout,
            )

            self._gateway_url = f"http://{cfg.host}:{port}"
            self._session = httpx.Client()
            self._session.headers["Authorization"] = f"Bearer {cfg.admin_api_key}"
            self._wait_for_health(timeout=timeout)
            logger.info("Gateway ready at %s", self._gateway_url)
        except BaseException:
            # /connect has not been sent yet, so there is no worker pair to
            # disconnect. Avoid the long request timeout and terminate only the
            # private process within the setup budget.
            cleanup_timeout = max(
                0.001,
                min(5.0, self.config.setup_timeout if timeout is None else timeout),
            )
            self.terminate_private_gateway(
                timeout=cleanup_timeout, raise_on_error=False
            )
            raise

    def _wait_for_health(self, timeout: float | None = None) -> None:
        timeout = self.config.setup_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._gateway_proc is not None and self._gateway_proc.poll() is not None:
                raise RuntimeError(
                    f"Gateway process exited prematurely "
                    f"(code {self._gateway_proc.returncode})"
                )
            try:
                remaining = deadline - time.monotonic()
                resp = self._http.get(
                    f"{self._gateway_url}/health",
                    timeout=min(2.0, max(0.001, remaining)),
                )
                if resp.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(f"Gateway did not become healthy within {timeout}s")

    def health_check(self) -> bool:
        try:
            resp = self._http.get(
                f"{self._gateway_url}/health",
                timeout=self.config.request_timeout,
            )
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    def connect(
        self,
        pair_name: str,
        train_worker_urls: list[str],
        inference_worker_urls: list[str],
        mode: str = "awex",
        save_path: str = "",
        use_lora: bool = False,
        lora_name: str = "",
        lora_keep_versions: int = 0,
        colocate: bool = False,
        nccl_master_addr: str = "",
        nccl_master_port: int = 0,
        setup_timeout_s: float | None = None,
        rollback_timeout_s: float = 30.0,
        request_timeout: float | None = None,
        operation_id: str | None = None,
    ) -> None:
        operation_id = operation_id or uuid4().hex
        payload: dict[str, Any] = {
            "pair_name": pair_name,
            "operation_id": operation_id,
            "train_worker_urls": train_worker_urls,
            "inference_worker_urls": inference_worker_urls,
            "mode": mode,
            "save_path": save_path,
            "use_lora": use_lora,
            "lora_name": lora_name,
            "lora_keep_versions": lora_keep_versions,
            "colocate": colocate,
            "nccl_master_addr": nccl_master_addr,
            "nccl_master_port": nccl_master_port,
            "setup_timeout_s": setup_timeout_s,
            "rollback_timeout_s": rollback_timeout_s,
        }
        # Retain the pending identity before /connect so a timeout or partial
        # gateway rollback can still be retried through /disconnect.
        self.retain_pending_connection(
            pair_name,
            operation_id,
            train_worker_urls,
            inference_worker_urls,
        )
        resp = self._http.post(
            f"{self._gateway_url}/connect",
            json=payload,
            timeout=(
                self.config.request_timeout
                if request_timeout is None
                else request_timeout
            ),
        )
        resp.raise_for_status()
        logger.info(
            "Connected pair '%s' (mode=%s, colocate=%s, use_lora=%s)",
            pair_name,
            mode,
            colocate,
            use_lora,
        )

    def update_weights(self, version: int) -> WeightUpdateResult:
        if self._pair_name is None:
            raise RuntimeError("Not connected. Call connect() first.")
        resp = self._http.post(
            f"{self._gateway_url}/update_weights",
            json={"pair_name": self._pair_name, "version": version},
            timeout=self.config.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        result = WeightUpdateResult(
            status=data["status"],
            version=data["version"],
            duration_ms=data["duration_ms"],
            error=data.get("error"),
            inference_weights_may_be_mutated=data.get(
                "inference_weights_may_be_mutated", True
            ),
        )
        if result.status != "ok":
            raise WeightUpdateError(
                f"Weight update failed for pair {self._pair_name!r}, "
                f"version={version}: {result.error or 'unknown error'}",
                inference_weights_may_be_mutated=(
                    result.inference_weights_may_be_mutated
                ),
            )
        return result

    def disconnect(self, timeout: float | None = None) -> None:
        if self._pair_name is None:
            return
        pair_name = self._pair_name
        resp = self._http.post(
            f"{self._gateway_url}/disconnect",
            json={
                "pair_name": pair_name,
                "operation_id": self._operation_id or "",
                "timeout_s": timeout,
            },
            timeout=self.config.request_timeout if timeout is None else timeout,
        )
        resp.raise_for_status()
        self.mark_workers_cleaned()
        logger.info("Disconnected pair '%s'", pair_name)

    def _gateway_get(self, path: str) -> Any:
        resp = self._http.get(
            f"{self._gateway_url}{path}", timeout=self.config.request_timeout
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Gateway {path} returned {resp.status_code}: {resp.text}"
            )
        return resp.json()

    def _gateway_post(self, path: str, payload: Any = None) -> Any:
        resp = self._http.post(
            f"{self._gateway_url}{path}",
            json=payload,
            timeout=self.config.request_timeout,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Gateway {path} returned {resp.status_code}: {resp.text}"
            )
        return resp.json()

    def terminate_private_gateway(
        self, *, timeout: float = 5.0, raise_on_error: bool = True
    ) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None

        error: Exception | None = None
        proc = self._gateway_proc
        if proc is not None:
            try:
                kill_process_tree(proc.pid, timeout=max(1, int(timeout)))
                if proc.poll() is None:
                    proc.wait(timeout=max(0.001, timeout))
                if proc.poll() is None:
                    raise RuntimeError("Gateway process is still running after kill")
            except Exception as exc:
                error = exc
                logger.warning("Failed to kill gateway process", exc_info=True)
            else:
                self._gateway_proc = None
                self._gateway_url = ""

        if error is not None and raise_on_error:
            raise error

    def destroy(
        self,
        *,
        raise_on_error: bool = False,
        timeout: float | None = None,
        disconnect: bool = True,
    ) -> None:
        deadline = None if timeout is None else time.monotonic() + max(0.001, timeout)

        def _remaining() -> float:
            assert deadline is not None
            return max(0.001, deadline - time.monotonic())

        disconnect_error: Exception | None = None
        if disconnect and self._pair_name is not None:
            try:
                self.disconnect(timeout=None if deadline is None else _remaining())
            except Exception as e:
                disconnect_error = e
                logger.warning("Failed to disconnect during destroy", exc_info=True)

        if disconnect_error is not None:
            # The private gateway is still the retry channel and owns the pair's
            # registry/KV state. Preserve all controller state until disconnect
            # has been confirmed.
            if raise_on_error:
                raise disconnect_error
            return

        self.terminate_private_gateway(
            timeout=5.0 if deadline is None else _remaining(),
            raise_on_error=raise_on_error,
        )
        logger.info("WeightUpdateController destroyed")

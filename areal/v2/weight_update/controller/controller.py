# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import time
from typing import Any

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
        self._session: httpx.Client | None = None

    @property
    def gateway_url(self) -> str:
        return self._gateway_url

    @property
    def pair_name(self) -> str | None:
        return self._pair_name

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
            self.destroy()
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
                resp = self._http.get(f"{self._gateway_url}/health", timeout=2.0)
                if resp.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.TimeoutException):
                pass
            time.sleep(0.5)
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
    ) -> None:
        payload: dict[str, Any] = {
            "pair_name": pair_name,
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
        resp = self._http.post(
            f"{self._gateway_url}/connect",
            json=payload,
            timeout=request_timeout or self.config.request_timeout,
        )
        resp.raise_for_status()
        self._pair_name = pair_name
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
            json={"pair_name": pair_name},
            timeout=timeout or self.config.request_timeout,
        )
        resp.raise_for_status()
        self._pair_name = None
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

    def destroy(self, *, raise_on_error: bool = False) -> None:
        disconnect_error: Exception | None = None
        if self._pair_name is not None:
            try:
                self.disconnect()
            except Exception as e:
                disconnect_error = e
                logger.warning("Failed to disconnect during destroy", exc_info=True)

        if self._session is not None:
            self._session.close()
            self._session = None

        if self._gateway_proc is not None:
            try:
                kill_process_tree(self._gateway_proc.pid)
            except Exception:
                logger.warning("Failed to kill gateway process", exc_info=True)
            self._gateway_proc = None
        self._gateway_url = ""
        logger.info("WeightUpdateController destroyed")
        if disconnect_error is not None and raise_on_error:
            raise disconnect_error

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
import time
from typing import Any

import httpx

from areal.experimental.weight_update.controller.config import (
    WeightUpdateControllerConfig,
)
from areal.experimental.weight_update.gateway.config import WeightUpdateResult
from areal.infra.utils.proc import kill_process_tree
from areal.utils import logging
from areal.utils.network import find_free_ports

logger = logging.getLogger("WeightUpdateController")


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
    def _http(self) -> httpx.Client:
        if self._session is None:
            raise RuntimeError("Controller not initialized. Call initialize() first.")
        return self._session

    def initialize(self) -> None:
        cfg = self.config
        port = cfg.port
        if port == 0:
            port = find_free_ports(1)[0]

        cmd = [
            sys.executable,
            "-m",
            "areal.experimental.weight_update.gateway",
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

        self._gateway_proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stdout,
        )

        self._gateway_url = f"http://{cfg.host}:{port}"
        self._session = httpx.Client()
        self._session.headers["Authorization"] = f"Bearer {cfg.admin_api_key}"
        self._wait_for_health()
        logger.info("Gateway ready at %s", self._gateway_url)

    def _wait_for_health(self) -> None:
        deadline = time.monotonic() + self.config.setup_timeout
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
        raise TimeoutError(
            f"Gateway did not become healthy within {self.config.setup_timeout}s"
        )

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
        nccl_master_addr: str = "",
        nccl_master_port: int = 0,
    ) -> None:
        self._pair_name = pair_name
        payload: dict[str, Any] = {
            "pair_name": pair_name,
            "train_worker_urls": train_worker_urls,
            "inference_worker_urls": inference_worker_urls,
            "mode": mode,
            "save_path": save_path,
            "use_lora": use_lora,
            "lora_name": lora_name,
            "nccl_master_addr": nccl_master_addr,
            "nccl_master_port": nccl_master_port,
        }
        resp = self._http.post(
            f"{self._gateway_url}/connect",
            json=payload,
            timeout=self.config.request_timeout,
        )
        resp.raise_for_status()
        logger.info("Connected pair '%s' (mode=%s)", pair_name, mode)

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
        return WeightUpdateResult(
            status=data["status"],
            version=data["version"],
            duration_ms=data["duration_ms"],
            error=data.get("error"),
        )

    def disconnect(self) -> None:
        if self._pair_name is None:
            return
        try:
            resp = self._http.post(
                f"{self._gateway_url}/disconnect",
                json={"pair_name": self._pair_name},
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
            logger.info("Disconnected pair '%s'", self._pair_name)
        finally:
            self._pair_name = None

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

    def destroy(self) -> None:
        if self._pair_name is not None:
            try:
                self.disconnect()
            except Exception:
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

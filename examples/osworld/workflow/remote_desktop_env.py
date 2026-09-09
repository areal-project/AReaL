"""HTTP proxy that mimics OSWorld's ``DesktopEnv`` on the training side.

The training container cannot run docker itself, so a companion
``remote_server.py`` runs on a machine that *does* have docker and exposes
``DesktopEnv`` operations over a small JSON API. Instances of this class are
drop-in replacements for ``DesktopEnv`` from the workflow's point of view —
same blocking method names, same return shapes.

All payloads are JSON; screenshots travel as base64-encoded PNG strings under
``screenshot_b64`` and are decoded back to ``bytes`` on the client side so the
rest of the pipeline (which expects raw PNG bytes) stays unchanged.
"""

from __future__ import annotations

import base64
from typing import Any

import requests

from areal.utils import logging

logger = logging.getLogger("RemoteDesktopEnv")


class RemoteDesktopEnvError(RuntimeError):
    """Raised when the remote server reports an error or is unreachable."""


def _decode_obs(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn the server's JSON obs back into the in-process obs shape."""
    screenshot_b64 = payload.get("screenshot_b64")
    obs: dict[str, Any] = {
        "screenshot": base64.b64decode(screenshot_b64) if screenshot_b64 else b"",
        "accessibility_tree": payload.get("accessibility_tree"),
        "terminal": payload.get("terminal"),
        "instruction": payload.get("instruction"),
    }
    return obs


class RemoteDesktopEnv:
    """Client-side stand-in for ``desktop_env.desktop_env.DesktopEnv``.

    Parameters
    ----------
    server_url
        Base URL of the remote ``remote_server.py`` (e.g. ``http://10.0.0.5:8000``).
    provider_name, path_to_vm, action_space, cache_dir, screen_size, headless,
    os_type, require_a11y_tree
        Forwarded verbatim to the server's ``DesktopEnv(...)`` constructor.
    request_timeout_secs
        Upper bound on every HTTP call. Reset/evaluate can take many minutes,
        so default generously.
    """

    def __init__(
        self,
        server_url: str,
        *,
        provider_name: str = "docker",
        path_to_vm: str | None = None,
        action_space: str = "pyautogui",
        cache_dir: str = "cache",
        screen_size: tuple[int, int] = (1920, 1080),
        headless: bool = True,
        os_type: str = "Ubuntu",
        require_a11y_tree: bool = False,
        request_timeout_secs: float = 1800.0,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.request_timeout_secs = request_timeout_secs
        self._last_obs: dict[str, Any] | None = None

        resp = self._post(
            "/envs",
            {
                "provider_name": provider_name,
                "path_to_vm": path_to_vm,
                "action_space": action_space,
                "cache_dir": cache_dir,
                "screen_size": list(screen_size),
                "headless": headless,
                "os_type": os_type,
                "require_a11y_tree": require_a11y_tree,
            },
        )
        self.session_id: str = resp["session_id"]
        logger.info(
            f"Opened remote DesktopEnv session {self.session_id} at {server_url}"
        )

    def reset(
        self, task_config: dict[str, Any] | None = None, **_: Any
    ) -> dict[str, Any]:
        payload = self._post(
            f"/envs/{self.session_id}/reset",
            {"task_config": task_config},
        )
        self._last_obs = _decode_obs(payload["obs"])
        return self._last_obs

    def _get_obs(self) -> dict[str, Any]:
        # The workflow calls `env._get_obs()` right after reset expecting the
        # cached observation. We return whatever the server sent last, and
        # only go back over the wire if nothing has been cached yet.
        if self._last_obs is not None:
            return self._last_obs
        payload = self._get(f"/envs/{self.session_id}/obs")
        self._last_obs = _decode_obs(payload["obs"])
        return self._last_obs

    def step(
        self, action: str, pause: float = 0.0
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        payload = self._post(
            f"/envs/{self.session_id}/step",
            {"action": action, "pause": pause},
        )
        self._last_obs = _decode_obs(payload["obs"])
        return (
            self._last_obs,
            float(payload.get("reward", 0.0)),
            bool(payload.get("done", False)),
            payload.get("info") or {},
        )

    def evaluate(self) -> float:
        payload = self._post(f"/envs/{self.session_id}/evaluate", {})
        return float(payload.get("reward", 0.0))

    def close(self) -> None:
        try:
            self._post(f"/envs/{self.session_id}/close", {})
        except RemoteDesktopEnvError as e:
            logger.warning(f"Remote close failed (session may already be gone): {e}")
        self._last_obs = None

    # ------------------------------------------------------------------ HTTP

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = self.server_url + path
        try:
            r = requests.post(url, json=body, timeout=self.request_timeout_secs)
        except requests.RequestException as e:
            raise RemoteDesktopEnvError(f"POST {url} failed: {e!r}") from e
        return self._unwrap(r, url)

    def _get(self, path: str) -> dict[str, Any]:
        url = self.server_url + path
        try:
            r = requests.get(url, timeout=self.request_timeout_secs)
        except requests.RequestException as e:
            raise RemoteDesktopEnvError(f"GET {url} failed: {e!r}") from e
        return self._unwrap(r, url)

    @staticmethod
    def _unwrap(response: requests.Response, url: str) -> dict[str, Any]:
        if response.status_code >= 400:
            try:
                msg = response.json().get("error", response.text)
            except ValueError:
                msg = response.text
            raise RemoteDesktopEnvError(f"{url} returned {response.status_code}: {msg}")
        try:
            return response.json()
        except ValueError as e:
            raise RemoteDesktopEnvError(
                f"{url} returned non-JSON body: {response.text[:200]}"
            ) from e

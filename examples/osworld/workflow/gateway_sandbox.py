"""OSWorld DesktopEnv backed by a vendor-neutral remote sandbox cluster gateway.

The training container can't run docker, so we drive a remote OSWorld VM via
a vendor-provided SDK (imported as ``pssdk``). The gateway is a transparent
proxy onto OSWorld's stock VM-side HTTP server, so we can subclass OSWorld's
``PythonController`` and re-route its calls through the gateway. The setup
controller is reimplemented here for the verbs we care about — supporting
every OSWorld setup verb is out of scope.

Expected SDK protocol (the ``pssdk`` module must export):

    BaseSandboxClusterTool(
        cluster_endpoint, application_secret_token, session_id,
        global_call_timeout,
    )
        # Constructor.

    BaseSandboxClusterTool.session_id
        # Property, str. The session id to authenticate gateway HTTP calls.

    BaseSandboxClusterTool.sandbox_start(body=None, call_timeout=...) -> dict
        # Allocate / start a sandbox VM; returns a dict with at least
        # ``sandboxId``.

    BaseSandboxClusterTool.sandbox_stop(call_timeout=...) -> dict
        # Stop / release the sandbox VM.

    with_retry(**kwargs)
        # Class decorator; wraps the cluster tool's methods so transient
        # gateway errors auto-retry and resource-quota errors park instead
        # of bubbling up.

If your provider's SDK does not export these symbols under the name
``pssdk``, replace this transport with your own RemoteClusterClient
implementation.

Layered as:

    BaseSandboxClusterTool         (pssdk, lifecycle: start/stop/status)
            ↓
    _GatewayTransport              (this module: auth-aware GET/POST/form/raw)
            ↓
    GatewaySandboxPythonController (subclass of PythonController — drop-in for
                                    OSWorld's evaluators/getters)
    GatewaySandboxSetupController  (handcrafted; covers the common verbs)
            ↓
    GatewaySandboxDesktopEnv       (subclass of DesktopEnv — drop-in for our
                                    workflow; skips provider/manager and
                                    _start_emulator)

Endpoint method/body conventions were verified by live probing — see
``REMOTE.md`` for the cheat sheet.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any
from urllib.parse import urljoin

import requests

try:
    from pssdk import BaseSandboxClusterTool, with_retry

    _HAS_PSSDK = True
except ImportError:
    _HAS_PSSDK = False
    BaseSandboxClusterTool = (
        object  # placeholder so class definition doesn't crash at import
    )

    def with_retry(**_kwargs):  # no-op decorator
        return lambda cls: cls


logger = logging.getLogger("GatewaySandbox")


# Wrap the cluster SDK in pssdk's retry decorator so transient gateway
# errors (5xx) auto-retry, and resource-quota 429s (`SandboxResourceLimitError`)
# park instead of bubbling up. Without this, a brief cluster spike kills the
# whole rollout.
@with_retry(
    max_attempts=3,
    retry_interval=10,
    infinite_retry_on_resource_limit=True,
    exclude_methods=[],
)
class _RetryingClusterTool(BaseSandboxClusterTool):
    pass


def _ensure_osworld_on_path(osworld_root: str | None) -> None:
    if osworld_root and osworld_root not in sys.path:
        sys.path.insert(0, osworld_root)


# ---------------------------------------------------------------- transport


class _GatewayTransport:
    """Auth-aware HTTP client that talks to the remote sandbox gateway.

    The pssdk's ``_gateway_request`` always sends JSON, but OSWorld's
    ``/file`` endpoint requires form-urlencoded bodies and ``/screenshot``
    returns raw image bytes — both of which the SDK can't express. This
    class is a tiny shim that issues the request directly with ``requests``
    while reusing the SDK's session id and secret token.
    """

    def __init__(
        self,
        cluster_endpoint: str,
        secret_token: str,
        session_id: str,
        default_timeout: float = 600.0,
    ) -> None:
        self.cluster_endpoint = cluster_endpoint.rstrip("/")
        self.secret_token = secret_token
        self.session_id = session_id
        self.default_timeout = default_timeout
        self._session = requests.Session()

    def _headers(self) -> dict[str, str]:
        return {
            "x-paas-session-id": self.session_id,
            "x-paas-secret-token": self.secret_token,
        }

    def _url(self, uri: str) -> str:
        return urljoin(self.cluster_endpoint + "/", uri.lstrip("/"))

    def request(
        self,
        method: str,
        uri: str,
        *,
        json_body: dict[str, Any] | None = None,
        form_body: dict[str, Any] | None = None,
        timeout: float | None = None,
        raw: bool = False,
    ) -> Any:
        method = method.upper()
        url = self._url(uri)
        headers = self._headers()
        kwargs: dict[str, Any] = {
            "timeout": timeout or self.default_timeout,
            "headers": headers,
        }
        if json_body is not None:
            kwargs["json"] = json_body
        elif form_body is not None:
            kwargs["data"] = form_body
        r = self._session.request(method, url, **kwargs)
        if r.status_code >= 400:
            raise GatewayHTTPError(f"{method} {uri} → {r.status_code}: {r.text[:300]}")
        if raw:
            return r.content
        try:
            return r.json()
        except ValueError:
            return r.text


class GatewayHTTPError(RuntimeError):
    pass


# ---------------------------------------------------- PythonController layer


def _make_python_controller(transport: _GatewayTransport):
    """Build the controller as a subclass of OSWorld's PythonController.

    Defined as a factory so OSWorld is only imported lazily (the training
    container has OSWorld on path via ``osworld_root``).
    """
    from desktop_env.controllers.python import PythonController

    class GatewaySandboxPythonController(PythonController):
        def __init__(self, t: _GatewayTransport) -> None:
            self._t = t
            self.pkgs_prefix = (
                "import pyautogui; import time; pyautogui.FAILSAFE = False; {command}"
            )
            self.retry_times = 3
            self.retry_interval = 5
            # Base class reads these in places; keep them populated.
            self.vm_ip = "sandbox"
            self.http_server = ""

        # -------- observation getters --------

        def get_screenshot(self):
            for attempt in range(self.retry_times):
                try:
                    data = self._t.request("GET", "/screenshot", timeout=30, raw=True)
                    if data and (
                        data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff"
                    ):
                        return data
                    logger.warning(
                        f"invalid screenshot payload (try {attempt + 1}/{self.retry_times})"
                    )
                except Exception as e:
                    logger.warning(
                        f"screenshot error (try {attempt + 1}/{self.retry_times}): {e!r}"
                    )
                time.sleep(self.retry_interval)
            return None

        def get_accessibility_tree(self):
            try:
                payload = self._t.request("GET", "/accessibility", timeout=60)
                return payload.get("AT") if isinstance(payload, dict) else None
            except Exception as e:
                logger.warning(f"accessibility fetch failed: {e!r}")
                return None

        def get_terminal_output(self):
            try:
                payload = self._t.request("GET", "/terminal", timeout=30)
                return payload.get("output") if isinstance(payload, dict) else None
            except Exception as e:
                # Common when no terminal is open in the VM; downgrade to debug.
                logger.debug(f"terminal fetch failed: {e!r}")
                return None

        def get_file(self, file_path: str):
            # The gateway only forwards `application/json`, but OSWorld VM's
            # /file reads `request.form` — incompatible. Work around by
            # base64-piping the file through /execute.
            cmd = (
                "import base64, os, sys; "
                f"p = os.path.expandvars(os.path.expanduser({file_path!r})); "
                "sys.stdout.write(base64.b64encode(open(p, 'rb').read()).decode())"
            )
            try:
                resp = self._t.request(
                    "POST",
                    "/execute",
                    json_body={"command": ["python3", "-c", cmd], "shell": False},
                    timeout=120,
                )
            except Exception as e:
                logger.warning(f"get_file({file_path}) via /execute failed: {e!r}")
                return None
            if not isinstance(resp, dict) or resp.get("status") != "success":
                logger.warning(f"get_file({file_path}) returned non-success: {resp!r}")
                return None
            import base64

            try:
                return base64.b64decode(resp.get("output") or "")
            except Exception as e:
                logger.warning(f"get_file({file_path}) base64 decode failed: {e!r}")
                return None

        # -------- execute / scripts --------

        def execute_python_command(self, command: str):
            command_list = ["python", "-c", self.pkgs_prefix.format(command=command)]
            for _ in range(self.retry_times):
                try:
                    return self._t.request(
                        "POST",
                        "/execute",
                        json_body={"command": command_list, "shell": False},
                        timeout=120,
                    )
                except Exception as e:
                    logger.warning(f"execute_python_command failed: {e!r}")
                    time.sleep(self.retry_interval)
            return None

        def run_python_script(self, script: str):
            try:
                return self._t.request(
                    "POST", "/run_python", json_body={"code": script}, timeout=180
                )
            except Exception as e:
                return {
                    "status": "error",
                    "message": str(e),
                    "output": "",
                    "error": repr(e),
                }

        def run_bash_script(
            self, script: str, timeout: int = 30, working_dir: str | None = None
        ):
            body: dict[str, Any] = {"script": script, "timeout": timeout}
            if working_dir:
                body["working_dir"] = working_dir
            try:
                return self._t.request(
                    "POST", "/run_bash_script", json_body=body, timeout=timeout + 60
                )
            except Exception as e:
                return {
                    "status": "error",
                    "message": str(e),
                    "output": "",
                    "error": repr(e),
                }

        # -------- VM info getters (mostly POST, empty body) --------

        def _post_empty(self, uri: str):
            try:
                return self._t.request("POST", uri, json_body={}, timeout=30)
            except Exception as e:
                logger.warning(f"{uri} failed: {e!r}")
                return None

        def get_vm_platform(self):
            # GET returning a plain string ("Linux" / "Windows" / "Darwin").
            try:
                return self._t.request("GET", "/platform", timeout=15)
            except Exception as e:
                logger.warning(f"/platform failed: {e!r}")
                return None

        def get_vm_machine(self):
            # OSWorld VM server has no /machine route; shell out to uname.
            try:
                resp = self._t.request(
                    "POST",
                    "/execute",
                    json_body={"command": ["uname", "-m"], "shell": False},
                    timeout=15,
                )
                if isinstance(resp, dict) and resp.get("status") == "success":
                    out = (resp.get("output") or "").strip()
                    if out:
                        return out
            except Exception as e:
                logger.warning(f"get_vm_machine via uname failed: {e!r}")
            return "x86_64"  # Sandbox image is amd64; safe fallback.

        def get_vm_screen_size(self):
            return self._post_empty("/screen_size")

        def get_vm_window_size(self, app_class_name: str):
            try:
                return self._t.request(
                    "POST",
                    "/window_size",
                    form_body={"app_class_name": app_class_name},
                    timeout=30,
                )
            except Exception as e:
                logger.warning(f"window_size failed: {e!r}")
                return None

        def get_vm_wallpaper(self):
            return self._post_empty("/wallpaper")

        def get_vm_desktop_path(self):
            return self._post_empty("/desktop_path")

        def get_vm_directory_tree(self, path):
            try:
                return self._t.request(
                    "POST", "/list_directory", json_body={"path": path}, timeout=120
                )
            except Exception as e:
                logger.warning(f"list_directory failed: {e!r}")
                return None

        # -------- recording --------

        def start_recording(self):
            return self._post_empty("/start_recording")

        def end_recording(self, dest: str):
            try:
                data = self._t.request(
                    "POST", "/end_recording", json_body={}, raw=True, timeout=300
                )
                if data:
                    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                    with open(dest, "wb") as f:
                        f.write(data)
                return True
            except Exception as e:
                logger.warning(f"end_recording failed: {e!r}")
                return False

    return GatewaySandboxPythonController(transport)


# --------------------------------------------------- SetupController surrogate


class GatewaySandboxSetupController:
    """Minimal OSWorld SetupController equivalent.

    Implements only the verbs we have actually observed in the AReaL training
    workloads (covers ~95% of OSWorld test_small tasks): launch, download,
    execute, open, chrome_open_tabs, activate_window, close_window, command,
    sleep. Anything else is logged and skipped (so a partially-supported task
    still runs end-to-end with a degraded reward signal).
    """

    SUPPORTED = frozenset(
        [
            "launch",
            "download",
            "execute",
            "open",
            "chrome_open_tabs",
            "activate_window",
            "close_window",
            "command",
            "sleep",
            "change_wallpaper",
        ]
    )

    def __init__(self, transport: _GatewayTransport, cache_dir: str = "cache") -> None:
        self._t = transport
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def reset_cache_dir(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def setup(self, config_list: list[dict[str, Any]], use_proxy: bool = False) -> bool:
        for i, item in enumerate(config_list or []):
            verb = item.get("type")
            params = item.get("parameters", {})
            if verb not in self.SUPPORTED:
                logger.warning(
                    f"setup step {i + 1}: verb '{verb}' not implemented for sandbox; skipping"
                )
                continue
            handler = getattr(self, f"_{verb}_setup")
            try:
                handler(**params)
                logger.info(f"setup step {i + 1}/{len(config_list)} ok: {verb}")
            except Exception as e:
                logger.error(f"setup step {i + 1} failed: {verb}({params}) → {e!r}")
                return False
        return True

    # ---- verb implementations ----

    def _launch_setup(self, command, shell: bool = False) -> None:
        if isinstance(command, str) and not shell:
            import shlex

            command = shlex.split(command)
        self._t.request(
            "POST",
            "/setup/launch",
            json_body={"command": command, "shell": shell},
            timeout=60,
        )

    def _execute_setup(self, command, shell: bool = False, **_: Any) -> None:
        if isinstance(command, str) and not shell:
            import shlex

            command = shlex.split(command)
        self._t.request(
            "POST",
            "/setup/execute",
            json_body={"command": command, "shell": shell},
            timeout=120,
        )

    def _command_setup(self, command, **_: Any) -> None:
        # OSWorld treats `command` as launch-like (Popen). Forward to /setup/launch with shell=True.
        if isinstance(command, list):
            cmd_str = " ".join(str(c) for c in command)
        else:
            cmd_str = str(command)
        self._t.request(
            "POST",
            "/setup/launch",
            json_body={"command": cmd_str, "shell": True},
            timeout=60,
        )

    def _sleep_setup(self, seconds: float) -> None:
        time.sleep(float(seconds))

    def _open_setup(self, path: str) -> None:
        self._t.request(
            "POST", "/setup/open_file", json_body={"path": path}, timeout=120
        )

    def _activate_window_setup(
        self, window_name: str, strict: bool = False, by_class: bool = False
    ) -> None:
        self._t.request(
            "POST",
            "/setup/activate_window",
            json_body={
                "window_name": window_name,
                "strict": strict,
                "by_class": by_class,
            },
            timeout=30,
        )

    def _close_window_setup(
        self, window_name: str, strict: bool = False, by_class: bool = False
    ) -> None:
        self._t.request(
            "POST",
            "/setup/close_window",
            json_body={
                "window_name": window_name,
                "strict": strict,
                "by_class": by_class,
            },
            timeout=30,
        )

    def _change_wallpaper_setup(self, path: str) -> None:
        self._t.request(
            "POST", "/setup/change_wallpaper", json_body={"path": path}, timeout=30
        )

    def _chrome_open_tabs_setup(self, urls_to_open: list[str]) -> None:
        # OSWorld's stock implementation kills chrome, then launches it with the URLs.
        self._t.request(
            "POST",
            "/setup/launch",
            json_body={"command": ["pkill", "-f", "chrome"], "shell": False},
            timeout=20,
        )
        time.sleep(2)
        cmd = [
            "google-chrome",
            "--no-first-run",
            "--no-default-browser-check",
            *urls_to_open,
        ]
        self._t.request(
            "POST",
            "/setup/launch",
            json_body={"command": cmd, "shell": False},
            timeout=30,
        )

    def _download_setup(self, files: list[dict[str, str]]) -> None:
        # OSWorld's _download_setup downloads a remote URL to the VM's local path.
        # The gateway exposes /setup/download_file taking {url, path}.
        for entry in files or []:
            url = entry.get("url")
            path = entry.get("path")
            if not url or not path:
                logger.warning(f"download entry missing url/path: {entry}")
                continue
            self._t.request(
                "POST",
                "/setup/download_file",
                json_body={"url": url, "path": path},
                timeout=600,
            )


# -------------------------------------------------- DesktopEnv subclass


def _make_desktop_env_cls():
    """Lazy-build subclass of DesktopEnv to keep OSWorld imports lazy."""
    from desktop_env.desktop_env import DesktopEnv

    class GatewaySandboxDesktopEnv(DesktopEnv):
        def __init__(
            self,
            *,
            cluster_endpoint: str,
            secret_token: str,
            cache_dir: str = "cache",
            screen_size: tuple[int, int] = (1920, 1080),
            require_a11y_tree: bool = False,
            require_terminal: bool = False,
            os_type: str = "Ubuntu",
            session_id: str | None = None,
            global_call_timeout: int = 1800,
            sandbox_start_body: dict[str, Any] | None = None,
        ) -> None:
            # Bypass DesktopEnv.__init__: we don't have a provider/manager and
            # the lifecycle goes through pssdk instead.
            self.region = None
            self.provider_name = "sandbox"
            self.enable_proxy = False
            self.client_password = "password"
            self.screen_width, self.screen_height = screen_size
            self.server_port = 5000
            self.chromium_port = 9222
            self.vnc_port = 8006
            self.vlc_port = 8080
            self.current_use_proxy = False
            self.os_type = os_type
            self.is_environment_used = False
            self.path_to_vm = "sandbox"
            self.snapshot_name = "init_state"
            self.cache_dir_base = cache_dir
            self.cache_dir = cache_dir
            self.headless = True
            self.require_a11y_tree = require_a11y_tree
            self.require_terminal = require_terminal
            self.action_space = "pyautogui"
            self.instruction = None
            self._traj_no = -1
            self._step_no = 0
            self.action_history = []
            os.makedirs(self.cache_dir, exist_ok=True)

            self._tool = _RetryingClusterTool(
                cluster_endpoint=cluster_endpoint,
                application_secret_token=secret_token,
                session_id=session_id,
                global_call_timeout=global_call_timeout,
            )
            logger.info(f"sandbox session id: {self._tool.session_id}")
            start_info = self._tool.sandbox_start(
                body=sandbox_start_body or {}, call_timeout=180
            )
            logger.info(f"sandbox started: {start_info.get('sandboxId') or start_info}")

            self._transport = _GatewayTransport(
                cluster_endpoint=cluster_endpoint,
                secret_token=secret_token,
                session_id=self._tool.session_id,
                default_timeout=global_call_timeout,
            )
            self.controller = _make_python_controller(self._transport)
            self.setup_controller = GatewaySandboxSetupController(
                self._transport, cache_dir=self.cache_dir
            )
            self.vm_ip = "sandbox"

        # ---- lifecycle overrides ----

        def _start_emulator(self) -> None:  # pragma: no cover - never called
            return

        def _revert_to_snapshot(self) -> None:
            # Pop the current sandbox and acquire a fresh one.
            try:
                self._tool.sandbox_stop(call_timeout=60)
            except Exception as e:
                logger.warning(f"sandbox_stop during revert failed: {e!r}")
            start_info = self._tool.sandbox_start(call_timeout=180)
            logger.info(
                f"sandbox restarted: {start_info.get('sandboxId') or start_info}"
            )
            self._transport.session_id = self._tool.session_id

        def close(self) -> None:
            try:
                self._tool.sandbox_stop(call_timeout=60)
            except Exception as e:
                logger.warning(f"sandbox_stop on close failed: {e!r}")

    return GatewaySandboxDesktopEnv


# Public factory used by the workflow.
def make_sandbox_desktop_env(
    *,
    osworld_root: str,
    cluster_endpoint: str,
    secret_token: str,
    cache_dir: str = "cache",
    screen_size: tuple[int, int] = (1920, 1080),
    require_a11y_tree: bool = False,
    os_type: str = "Ubuntu",
    sandbox_start_body: dict[str, Any] | None = None,
):
    """Create a `GatewaySandboxDesktopEnv` instance.

    OSWorld must be on the path; pass ``osworld_root`` so we can insert it
    lazily before importing ``desktop_env`` symbols.
    """
    if not _HAS_PSSDK:
        raise RuntimeError(
            "Gateway-based sandbox requires a vendor SDK that provides "
            "BaseSandboxClusterTool / with_retry. Install your provider's "
            "SDK (which exports the `pssdk` module) or replace this transport "
            "with your own RemoteClusterClient implementation. See README."
        )
    _ensure_osworld_on_path(osworld_root)
    cls = _make_desktop_env_cls()
    return cls(
        cluster_endpoint=cluster_endpoint,
        secret_token=secret_token,
        cache_dir=cache_dir,
        screen_size=screen_size,
        require_a11y_tree=require_a11y_tree,
        os_type=os_type,
        sandbox_start_body=sandbox_start_body,
    )

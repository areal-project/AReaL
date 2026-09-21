"""Flask server wrapping OSWorld's DesktopEnv for remote rollout.

Deploy this on a machine that can actually run docker (KVM optional but
recommended). The AReaL training container then sends HTTP requests here via
``RemoteDesktopEnv`` — each training trajectory opens its own server-side
session, resets with a task config, pushes pyautogui actions, and finally asks
for an ``evaluate()`` reward.

Run (on the remote docker machine):

    python remote_server.py \\
        --osworld-root /path/to/OSWorld \\
        --host 0.0.0.0 --port 8000 \\
        --max-envs 2

Concurrency is bounded by a global semaphore (``--max-envs``); creating more
sessions than that blocks until an existing one is closed. The workflow
creates a fresh env per trajectory and closes it when done, so the bound also
caps the simultaneously booted OSWorld docker VMs.

No auth — intended for trusted internal networks only.
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
import threading
import traceback
import uuid
from typing import Any

from flask import Flask, jsonify, request

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("osworld.remote_server")


def _import_desktop_env(osworld_root: str):
    if osworld_root and osworld_root not in sys.path:
        sys.path.insert(0, osworld_root)
    from desktop_env.desktop_env import DesktopEnv  # noqa: E402

    return DesktopEnv


class SessionRegistry:
    """Keeps per-session ``DesktopEnv`` instances with a global cap."""

    def __init__(self, max_envs: int, desktop_env_cls):
        self._max_envs = max_envs
        self._cls = desktop_env_cls
        self._sessions: dict[str, DesktopEnvSession] = {}
        self._registry_lock = threading.Lock()
        self._slots = threading.Semaphore(max_envs)

    def create(self, kwargs: dict[str, Any]) -> DesktopEnvSession:
        # Block here until a slot is available. ``close`` releases it.
        self._slots.acquire()
        try:
            env = self._cls(**kwargs)
        except Exception:
            self._slots.release()
            raise
        sid = uuid.uuid4().hex
        session = DesktopEnvSession(sid=sid, env=env, slot_release=self._slots.release)
        with self._registry_lock:
            self._sessions[sid] = session
        logger.info(
            f"[session {sid}] created; active={len(self._sessions)}/{self._max_envs}"
        )
        return session

    def get(self, sid: str) -> DesktopEnvSession:
        with self._registry_lock:
            session = self._sessions.get(sid)
        if session is None:
            raise KeyError(sid)
        return session

    def drop(self, sid: str) -> None:
        with self._registry_lock:
            session = self._sessions.pop(sid, None)
        if session is None:
            return
        session.close()
        logger.info(
            f"[session {sid}] closed; active={len(self._sessions)}/{self._max_envs}"
        )


class DesktopEnvSession:
    """Wraps a single ``DesktopEnv`` with a lock so callers can't race it."""

    def __init__(self, sid: str, env, slot_release):
        self.sid = sid
        self.env = env
        self._lock = threading.Lock()
        self._closed = False
        self._slot_release = slot_release

    def call(self, fn_name: str, *args, **kwargs):
        with self._lock:
            if self._closed:
                raise RuntimeError(f"session {self.sid} already closed")
            fn = getattr(self.env, fn_name)
            return fn(*args, **kwargs)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self.env.close()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[session {self.sid}] env.close() failed: {e!r}")
            finally:
                self._slot_release()


# ---------------------------------------------------------------------- helpers


def _encode_obs(obs: dict[str, Any] | None) -> dict[str, Any]:
    if not obs:
        return {}
    screenshot = obs.get("screenshot") or b""
    return {
        "screenshot_b64": base64.b64encode(screenshot).decode("ascii"),
        "accessibility_tree": obs.get("accessibility_tree"),
        "terminal": obs.get("terminal"),
        "instruction": obs.get("instruction"),
    }


def _error(status: int, message: str):
    return jsonify({"error": message}), status


# ----------------------------------------------------------------------- routes


def create_app(osworld_root: str, max_envs: int) -> Flask:
    DesktopEnv = _import_desktop_env(osworld_root)
    registry = SessionRegistry(max_envs=max_envs, desktop_env_cls=DesktopEnv)
    app = Flask("osworld-remote-server")

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "active_sessions": len(registry._sessions),
                "max_envs": max_envs,
            }
        )

    @app.post("/envs")
    def create_env():
        body = request.get_json(silent=True) or {}
        screen_size = body.get("screen_size") or [1920, 1080]
        try:
            kwargs = dict(
                provider_name=body.get("provider_name", "docker"),
                path_to_vm=body.get("path_to_vm"),
                action_space=body.get("action_space", "pyautogui"),
                cache_dir=body.get("cache_dir", "cache"),
                screen_size=tuple(screen_size),
                headless=bool(body.get("headless", True)),
                os_type=body.get("os_type", "Ubuntu"),
                require_a11y_tree=bool(body.get("require_a11y_tree", False)),
            )
            session = registry.create(kwargs)
        except Exception as e:  # noqa: BLE001
            logger.error(f"create_env failed: {e!r}\n{traceback.format_exc()}")
            return _error(500, f"DesktopEnv init failed: {e!r}")
        return jsonify({"session_id": session.sid})

    @app.post("/envs/<sid>/reset")
    def reset_env(sid):
        body = request.get_json(silent=True) or {}
        try:
            session = registry.get(sid)
            obs = session.call("reset", task_config=body.get("task_config"))
        except KeyError:
            return _error(404, f"unknown session {sid}")
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"[session {sid}] reset failed: {e!r}\n{traceback.format_exc()}"
            )
            return _error(500, f"reset failed: {e!r}")
        return jsonify({"obs": _encode_obs(obs)})

    @app.get("/envs/<sid>/obs")
    def get_obs(sid):
        try:
            session = registry.get(sid)
            obs = session.call("_get_obs")
        except KeyError:
            return _error(404, f"unknown session {sid}")
        except Exception as e:  # noqa: BLE001
            return _error(500, f"_get_obs failed: {e!r}")
        return jsonify({"obs": _encode_obs(obs)})

    @app.post("/envs/<sid>/step")
    def step_env(sid):
        body = request.get_json(silent=True) or {}
        action = body.get("action")
        pause = float(body.get("pause", 0.0))
        if action is None:
            return _error(400, "missing 'action' in body")
        try:
            session = registry.get(sid)
            obs, reward, done, info = session.call("step", action, pause)
        except KeyError:
            return _error(404, f"unknown session {sid}")
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"[session {sid}] step failed: {e!r}\n{traceback.format_exc()}"
            )
            return _error(500, f"step failed: {e!r}")
        return jsonify(
            {
                "obs": _encode_obs(obs),
                "reward": float(reward),
                "done": bool(done),
                "info": info if isinstance(info, dict) else {"raw": str(info)},
            }
        )

    @app.post("/envs/<sid>/evaluate")
    def evaluate_env(sid):
        try:
            session = registry.get(sid)
            reward = session.call("evaluate")
        except KeyError:
            return _error(404, f"unknown session {sid}")
        except Exception as e:  # noqa: BLE001
            logger.error(
                f"[session {sid}] evaluate failed: {e!r}\n{traceback.format_exc()}"
            )
            return _error(500, f"evaluate failed: {e!r}")
        return jsonify({"reward": float(reward)})

    @app.post("/envs/<sid>/close")
    def close_env(sid):
        try:
            registry.drop(sid)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[session {sid}] close error: {e!r}")
        return jsonify({"ok": True})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--osworld-root",
        default=os.environ.get("OSWORLD_ROOT", ""),
        help="Path to OSWorld checkout; also read from $OSWORLD_ROOT.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max-envs",
        type=int,
        default=2,
        help="Global cap on simultaneously-alive DesktopEnv sessions.",
    )
    args = parser.parse_args()

    if not args.osworld_root:
        parser.error("--osworld-root is required (or set $OSWORLD_ROOT)")
    if not os.path.isdir(args.osworld_root):
        parser.error(f"OSWorld root does not exist: {args.osworld_root}")

    app = create_app(osworld_root=args.osworld_root, max_envs=args.max_envs)

    logger.info(
        f"Serving OSWorld on http://{args.host}:{args.port} "
        f"(OSWorld={args.osworld_root}, max_envs={args.max_envs})"
    )
    # Flask's dev server with threaded=True is enough for our 1–2 concurrent
    # sessions; swap for gunicorn if you need more throughput later.
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()

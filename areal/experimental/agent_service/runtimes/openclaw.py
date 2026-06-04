# SPDX-License-Identifier: Apache-2.0

"""OpenClaw Agent for AReaL Agent Service (per-session subprocess).

Implements :class:`AgentRunnable` by spawning **one OpenClaw Gateway
subprocess per RL session**.  Each subprocess is bound to its own
upstream LLM (base URL + API key + model) so that, during training, a
session's turns can be attributed to a distinct per-episode key
(``sk-sess-*``).  OpenClaw config is process-global, so per-session
isolation requires one process per session.

Per turn the agent issues a single OpenAI-compatible
``POST /v1/chat/completions`` to the session's own subprocess, replaying
the conversation history that the DataProxy maintains.

Requires the ``openclaw`` CLI on ``PATH`` (``npm i -g openclaw``).

Upstream selection
------------------
At episode start the agent prefers the :class:`TrainingContext` injected
by the controller (``llm_base_url`` / ``llm_api_key`` / ``llm_model``).
Outside training (e.g. the interactive demo) it falls back to the
``OPENCLAW_UPSTREAM_*`` environment variables, then to the legacy
``OPENCLAW_GATEWAY_*`` names.

Environment variables
---------------------
    OPENCLAW_BIN                — openclaw executable (default ``openclaw``).
    OPENCLAW_UPSTREAM_BASE_URL  — upstream LLM base URL (fallback when no
        TrainingContext); legacy fallback ``OPENCLAW_GATEWAY_URL``.
    OPENCLAW_UPSTREAM_API_KEY   — upstream API key; legacy fallback
        ``OPENCLAW_GATEWAY_TOKEN``.
    OPENCLAW_UPSTREAM_MODEL     — upstream model id; legacy fallback
        ``OPENCLAW_MODEL``.
    OPENCLAW_UPSTREAM_API       — ``openai-completions`` (default) or
        ``anthropic-messages``.
    OPENCLAW_TIMEOUT            — per-request timeout in seconds (default 120).
    OPENCLAW_STARTUP_TIMEOUT    — subprocess health-wait seconds (default 60).
    OPENCLAW_NODE_EXTRA_CA_CERTS— path to a CA bundle for the upstream TLS
        cert (preferred over disabling verification).
    OPENCLAW_TLS_INSECURE       — ``1`` to set ``NODE_TLS_REJECT_UNAUTHORIZED=0``
        (dev only; needed for upstreams whose CA Node cannot verify).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import shutil
import socket
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import IO, Any

import httpx

from areal.experimental.agent_service.types import (
    AgentRequest,
    AgentResponse,
    EventEmitter,
    TrainingContext,
)
from areal.utils import logging

logger = logging.getLogger("OpenClawAgent")

_PROVIDER = "areal"


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class _Upstream:
    """Upstream LLM that an OpenClaw subprocess routes its calls to."""

    base_url: str
    api_key: str
    model: str
    api: str = "openai-completions"

    @classmethod
    def from_training_ctx(cls, ctx: TrainingContext) -> _Upstream | None:
        if not ctx.llm_base_url or not ctx.llm_api_key:
            return None
        api = str(ctx.extras.get("openclaw_api", "openai-completions"))
        model = ctx.llm_model or "default"
        return cls(
            base_url=ctx.llm_base_url,
            api_key=ctx.llm_api_key,
            model=model,
            api=api,
        )

    @classmethod
    def from_env(cls) -> _Upstream | None:
        base_url = os.environ.get("OPENCLAW_UPSTREAM_BASE_URL") or os.environ.get(
            "OPENCLAW_GATEWAY_URL", ""
        )
        api_key = os.environ.get("OPENCLAW_UPSTREAM_API_KEY") or os.environ.get(
            "OPENCLAW_GATEWAY_TOKEN", ""
        )
        if not base_url or not api_key:
            return None
        model = os.environ.get("OPENCLAW_UPSTREAM_MODEL") or os.environ.get(
            "OPENCLAW_MODEL", "default"
        )
        api = os.environ.get("OPENCLAW_UPSTREAM_API", "openai-completions")
        return cls(base_url=base_url.rstrip("/"), api_key=api_key, model=model, api=api)


@dataclass
class _SessionState:
    port: int
    gateway_token: str
    config_dir: str
    process: asyncio.subprocess.Process
    client: httpx.AsyncClient
    log_file: IO[str]
    training_ctx: TrainingContext | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OpenClawAgent:
    """AgentRunnable that runs one OpenClaw subprocess per session."""

    def __init__(self, **_: Any) -> None:
        self._bin = os.environ.get("OPENCLAW_BIN", "openclaw")
        self._timeout = float(os.environ.get("OPENCLAW_TIMEOUT", "120"))
        self._startup_timeout = float(os.environ.get("OPENCLAW_STARTUP_TIMEOUT", "60"))
        self._node_extra_ca_certs = os.environ.get("OPENCLAW_NODE_EXTRA_CA_CERTS", "")
        self._tls_insecure = _truthy(os.environ.get("OPENCLAW_TLS_INSECURE", ""))
        self._env_upstream = _Upstream.from_env()

        self._sessions: dict[str, _SessionState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

        logger.info(
            "OpenClawAgent initialized (bin=%s, env_upstream=%s, tls_insecure=%s)",
            self._bin,
            self._env_upstream is not None,
            self._tls_insecure,
        )

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    def _render_config(
        self, port: int, token: str, upstream: _Upstream
    ) -> dict[str, Any]:
        return {
            "gateway": {
                "mode": "local",
                "port": port,
                "auth": {"mode": "token", "token": token},
                "http": {"endpoints": {"chatCompletions": {"enabled": True}}},
            },
            "models": {
                "providers": {
                    _PROVIDER: {
                        "baseUrl": upstream.base_url,
                        "apiKey": upstream.api_key,
                        "api": upstream.api,
                        "models": [{"id": upstream.model, "name": upstream.model}],
                    }
                }
            },
            "agents": {"defaults": {"model": f"{_PROVIDER}/{upstream.model}"}},
        }

    async def _session_lock(self, session_key: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(session_key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[session_key] = lock
            return lock

    async def _spawn(self, session_key: str, upstream: _Upstream) -> _SessionState:
        port = _free_port()
        token = secrets.token_hex(16)
        config_dir = tempfile.mkdtemp(prefix="openclaw-")
        log_file: IO[str] | None = None
        client: httpx.AsyncClient | None = None
        proc: asyncio.subprocess.Process | None = None
        try:
            config_path = os.path.join(config_dir, "openclaw.json")
            state_dir = os.path.join(config_dir, "state")
            os.makedirs(state_dir, exist_ok=True)
            with open(config_path, "w") as fh:
                json.dump(self._render_config(port, token, upstream), fh)

            env = dict(os.environ)
            env["OPENCLAW_CONFIG_PATH"] = config_path
            env["OPENCLAW_STATE_DIR"] = state_dir
            if self._node_extra_ca_certs:
                env["NODE_EXTRA_CA_CERTS"] = self._node_extra_ca_certs
            if self._tls_insecure:
                env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"

            log_file = open(os.path.join(config_dir, "gateway.log"), "w")
            proc = await asyncio.create_subprocess_exec(
                self._bin,
                "gateway",
                "--port",
                str(port),
                "--auth",
                "token",
                "--token",
                token,
                "--force",
                "--allow-unconfigured",
                env=env,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
            )

            client = httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}",
                timeout=self._timeout,
                headers={"Authorization": f"Bearer {token}"},
            )
            state = _SessionState(
                port=port,
                gateway_token=token,
                config_dir=config_dir,
                process=proc,
                client=client,
                log_file=log_file,
            )
            await self._wait_healthy(state)
        except Exception:
            # Any failure before the session is fully healthy must not leak the
            # log file descriptor, the subprocess, or the temp config dir.
            if client is not None:
                with contextlib.suppress(Exception):
                    await client.aclose()
            if proc is not None and proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    with contextlib.suppress(Exception):
                        await proc.wait()
            if log_file is not None:
                with contextlib.suppress(Exception):
                    log_file.close()
            shutil.rmtree(config_dir, ignore_errors=True)
            raise

        logger.info(
            "Spawned OpenClaw subprocess (session=%s, port=%d, pid=%s, model=%s)",
            session_key,
            port,
            proc.pid,
            upstream.model,
        )
        return state

    async def _wait_healthy(self, state: _SessionState) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._startup_timeout
        while loop.time() < deadline:
            if state.process.returncode is not None:
                raise RuntimeError(
                    f"openclaw gateway exited early "
                    f"(rc={state.process.returncode}); see {state.config_dir}/gateway.log"
                )
            try:
                resp = await state.client.get("/v1/models")
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
        raise TimeoutError(
            f"openclaw gateway on port {state.port} did not become healthy "
            f"within {self._startup_timeout}s"
        )

    async def _teardown_state(self, state: _SessionState) -> None:
        with contextlib.suppress(Exception):
            await state.client.aclose()
        proc = state.process
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
        with contextlib.suppress(Exception):
            state.log_file.close()
        shutil.rmtree(state.config_dir, ignore_errors=True)

    async def _ensure_session(
        self,
        session_key: str,
        upstream: _Upstream,
        training_ctx: TrainingContext | None = None,
        *,
        respawn: bool = False,
    ) -> _SessionState:
        lock = await self._session_lock(session_key)
        async with lock:
            existing = self._sessions.get(session_key)
            if existing is not None:
                if not respawn:
                    return existing
                await self._teardown_state(self._sessions.pop(session_key))
            state = await self._spawn(session_key, upstream)
            state.training_ctx = training_ctx
            self._sessions[session_key] = state
            return state

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional members of the AgentRunnable protocol)
    # ------------------------------------------------------------------

    async def on_episode_start(
        self, session_key: str, training_ctx: TrainingContext
    ) -> None:
        """Spawn (or rebind) a per-session subprocess for the episode.

        Prefers the upstream carried by ``training_ctx`` so the session's
        LLM calls flow through AReaL's proxy gateway under a per-episode
        key; falls back to the env upstream otherwise.
        """
        upstream = _Upstream.from_training_ctx(training_ctx) or self._env_upstream
        if upstream is None:
            raise RuntimeError(
                "No upstream available: TrainingContext lacks llm_base_url/"
                "llm_api_key and no OPENCLAW_UPSTREAM_* env is set."
            )
        await self._ensure_session(session_key, upstream, training_ctx, respawn=True)
        logger.info(
            "Episode start (session=%s, ctx=%s)", session_key, training_ctx.session_id
        )

    async def on_episode_end(self, session_key: str, reward: float | None) -> None:
        logger.info("Episode end (session=%s, reward=%s)", session_key, reward)
        await self.close_session(session_key)

    async def close_session(self, session_key: str) -> None:
        state = self._sessions.pop(session_key, None)
        if state is not None:
            await self._teardown_state(state)
        # Drop the per-session lock so ``_locks`` does not grow unbounded as
        # sessions are created and destroyed over a long-running worker.
        async with self._locks_guard:
            self._locks.pop(session_key, None)

    async def close_all_sessions(self) -> None:
        for key in list(self._sessions.keys()):
            await self.close_session(key)

    # ------------------------------------------------------------------
    # Per-turn execution
    # ------------------------------------------------------------------

    async def run(
        self,
        request: AgentRequest,
        *,
        emitter: EventEmitter,
    ) -> AgentResponse:
        state = self._sessions.get(request.session_key)
        if state is None:
            # No episode opened (e.g. interactive demo): lazily spawn from env.
            if self._env_upstream is None:
                raise RuntimeError(
                    "No subprocess for session and no OPENCLAW_UPSTREAM_* env "
                    "configured; call /episode/start first or set the env."
                )
            state = await self._ensure_session(request.session_key, self._env_upstream)

        messages = list(request.history) + [
            {"role": "user", "content": request.message}
        ]

        text_parts: list[str] = []
        # Accumulate streamed tool calls by their ``index``: OpenAI-compatible
        # streaming sends the ``name`` only in the first chunk and streams
        # ``arguments`` across later chunks, so we must buffer per index and
        # emit once the stream completes.
        active_tool_calls: dict[int, dict[str, Any]] = {}

        # Use a fresh OpenClaw session per turn: OpenClaw keeps its own
        # per-session-key memory, but AReaL's DataProxy already replays the
        # full history below.  Reusing a stable key would feed the prior turns
        # twice (OpenClaw memory + replayed history) and corrupt the upstream
        # prompt that the training proxy captures.  A unique key makes the
        # replayed history the single source of truth.
        turn_key = f"{request.session_key}:{uuid.uuid4().hex}"

        async with state.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "openclaw/default",
                "messages": messages,
                "stream": True,
            },
            headers={"x-openclaw-session-key": turn_key},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                # The space after "data:" is optional per the SSE spec;
                # some OpenAI-compatible gateways omit it.
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed SSE chunk: %s", payload[:120])
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}

                text = delta.get("content")
                if text:
                    await emitter.emit_delta(text)
                    text_parts.append(text)

                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index")
                    if idx is None:
                        continue
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    args = fn.get("arguments", "")
                    slot = active_tool_calls.setdefault(
                        idx, {"name": "", "arguments": []}
                    )
                    if name:
                        slot["name"] = name
                    if args:
                        slot["arguments"].append(args)

        tool_calls: list[dict[str, Any]] = []
        for slot in active_tool_calls.values():
            name = slot["name"]
            full_args = "".join(slot["arguments"])
            await emitter.emit_tool_call(name=name, args=full_args)
            tool_calls.append({"name": name, "input": full_args})

        summary = "".join(text_parts)
        return AgentResponse(
            summary=summary[:200],
            metadata={"tool_calls": tool_calls},
        )

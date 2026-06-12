# SPDX-License-Identifier: Apache-2.0

"""Data Proxy — stateful session proxy between Gateway and Worker."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException

from areal.utils import logging

from .config import DataProxyConfig

logger = logging.getLogger("AgentDataProxy")


@dataclass
class _SessionData:
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_active: float = field(default_factory=time.monotonic)
    reward: float | None = None
    training_ctx: dict[str, Any] | None = None


def create_data_proxy_app(config: DataProxyConfig) -> FastAPI:
    app = FastAPI(title="AReaL Data Proxy")
    sessions: dict[str, _SessionData] = {}
    http_client = httpx.AsyncClient(timeout=config.request_timeout)

    async def _close_worker_session(session_key: str) -> None:
        try:
            await http_client.post(
                f"{config.worker_addr}/session/{session_key}/close", timeout=5
            )
        except Exception:
            logger.debug("Failed to close worker session %s", session_key)

    async def _relay_reward_to_training(session: _SessionData, reward: float) -> None:
        """Relay the episode's final reward into the training pipeline.

        The agent's LLM calls flow through the inference proxy gateway under
        a per-episode key (``sk-sess-*``), so the trajectory's tokens and
        logprobs are already captured there.  The reward, however, only lives
        in this DataProxy until we forward it.  We reuse the very same
        ``llm_base_url`` / ``llm_api_key`` that were injected at episode start
        (``training_ctx``) to POST the scalar to the proxy gateway's
        ``/rl/set_reward``, binding the reward to that captured trajectory.

        This is best-effort and never raises: a relay failure must not break
        episode teardown, and in serving mode (no ``training_ctx``) there is
        no training pipeline to relay to, so we simply skip.
        """
        ctx = session.training_ctx
        if not ctx:
            return
        base_url = ctx.get("llm_base_url")
        api_key = ctx.get("llm_api_key")
        if not base_url or not api_key:
            return
        try:
            resp = await http_client.post(
                f"{str(base_url).rstrip('/')}/rl/set_reward",
                json={"reward": reward},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
        except Exception:
            logger.warning(
                "Failed to relay reward to training pipeline for an episode "
                "(reward=%s); trajectory will train without it",
                reward,
            )

    async def _reap_idle_sessions() -> None:
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            stale = [
                k
                for k, s in sessions.items()
                if now - s.last_active > config.session_timeout
            ]
            for k in stale:
                del sessions[k]
                await _close_worker_session(k)
            if stale:
                logger.info("Reaped %d idle sessions", len(stale))

    @app.on_event("startup")
    async def startup():
        app.state.reaper_task = asyncio.create_task(_reap_idle_sessions())

    @app.on_event("shutdown")
    async def shutdown():
        await http_client.aclose()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "active_sessions": len(sessions),
            "worker_addr": config.worker_addr,
        }

    @app.post("/session/{session_key}/turn")
    async def turn(session_key: str, body: dict[str, Any]):
        session = sessions.get(session_key)
        if session is None:
            session = _SessionData()
            sessions[session_key] = session

        message = body.get("message", "")
        run_id = body.get("run_id", "")
        queue_mode = body.get("queue_mode", "collect")
        metadata = body.get("metadata", {})

        worker_request = {
            "message": message,
            "session_key": session_key,
            "run_id": run_id,
            "history": session.history.copy(),
            "queue_mode": queue_mode,
            "metadata": metadata,
        }

        resp = await http_client.post(f"{config.worker_addr}/run", json=worker_request)
        resp.raise_for_status()
        result = resp.json()

        session.history.append({"role": "user", "content": message})

        call_counter = 0
        for evt in result.get("events", []):
            if evt.get("type") == "tool_call":
                call_id = f"call_{evt.get('name', '')}_{run_id}_{call_counter}"
                call_counter += 1
                session.history.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": evt.get("name", ""),
                                    "arguments": evt.get("args", ""),
                                },
                            }
                        ],
                    }
                )
            elif evt.get("type") == "tool_result":
                result_call_id = (
                    f"call_{evt.get('name', '')}_{run_id}_{call_counter - 1}"
                    if call_counter > 0
                    else f"call_{evt.get('name', '')}_{run_id}_0"
                )
                session.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": result_call_id,
                        "content": evt.get("result", ""),
                    }
                )

        summary = result.get("summary", "")
        if summary:
            session.history.append({"role": "assistant", "content": summary})

        session.last_active = time.monotonic()
        return result

    @app.post("/session/{session_key}/close")
    async def close_session(session_key: str):
        sessions.pop(session_key, None)
        await _close_worker_session(session_key)
        return {"status": "ok"}

    @app.post("/session/{session_key}/episode/start")
    async def episode_start(session_key: str, body: dict[str, Any]):
        """Open a training episode for a session.

        Forwards a :class:`TrainingContext` payload to the worker so the
        agent can route internal LLM calls through the training proxy.
        """
        session = sessions.get(session_key)
        if session is None:
            session = _SessionData()
            sessions[session_key] = session

        # Start the episode on the worker first: only once the worker has
        # successfully (re)spawned its per-session state do we reset ours.
        # Resetting before this call would wipe the existing history/reward
        # even when the worker fails (500 from ``raise_for_status``), leaving
        # the session blanked for an episode that never actually started.
        resp = await http_client.post(
            f"{config.worker_addr}/session/{session_key}/episode/start",
            json=body,
        )
        resp.raise_for_status()

        # A new episode starts from a clean slate: the worker respawns a
        # fresh subprocess, so any history/reward carried over from a prior
        # episode on the same key would corrupt the new trajectory.
        session.history.clear()
        session.reward = None
        session.training_ctx = dict(body)
        session.last_active = time.monotonic()
        return resp.json()

    @app.post("/session/{session_key}/episode/end")
    async def episode_end(session_key: str, body: dict[str, Any]):
        """Close a training episode and forward final reward to the worker.

        ``body``: ``{"reward": <float|None>}``.  Reward defaults to the
        last value set via ``/session/{key}/reward``.
        """
        session = sessions.get(session_key)
        reward = body.get("reward")
        if reward is None and session is not None:
            reward = session.reward

        resp = await http_client.post(
            f"{config.worker_addr}/session/{session_key}/episode/end",
            json={"reward": reward},
        )
        resp.raise_for_status()

        if session is not None:
            # Relay the final reward into the training pipeline (best-effort;
            # no-op in serving mode where no training_ctx was injected). Done
            # only after the worker has successfully ended the episode.
            if reward is not None:
                await _relay_reward_to_training(session, reward)
            # The reward has been consumed by this episode; clear it so a
            # subsequent episode on the same key does not inherit a stale value.
            session.reward = None
            session.last_active = time.monotonic()
        return resp.json()

    @app.post("/session/{session_key}/reward")
    async def set_reward(session_key: str, body: dict[str, Any]):
        """Record a scalar reward for the session.

        The reward is buffered here and, on the next ``episode/end`` call,
        forwarded to the worker and relayed to the training pipeline's
        ``/rl/set_reward`` endpoint (see ``_relay_reward_to_training``).
        In serving mode (no ``training_ctx``) the relay is skipped.
        """
        session = sessions.get(session_key)
        if session is None:
            session = _SessionData()
            sessions[session_key] = session
        reward = body.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise HTTPException(status_code=400, detail="reward must be a number")
        session.reward = float(reward)
        session.last_active = time.monotonic()
        return {"status": "ok", "reward": session.reward}

    @app.get("/session/{session_key}/history")
    async def get_history(session_key: str):
        session = sessions.get(session_key)
        if session is None:
            return {"history": []}
        return {"history": session.history}

    return app

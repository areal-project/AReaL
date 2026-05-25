# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import aiohttp
import httpx
import openai

from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai.proxy.server import deserialize_interactions
from areal.infra import workflow_context
from areal.infra.utils.http import async_http_retry
from areal.utils import logging, stats_tracker

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine
    from areal.experimental.inference_service.controller.controller import (
        RolloutControllerV2,
    )
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

logger = logging.getLogger("InferenceServiceWorkflow")

_RL_START_SESSION_PATHNAME = "rl/start_session"
_RL_SET_REWARD_PATHNAME = "rl/set_reward"
_EXPORT_TRAJECTORIES_PATHNAME = "export_trajectories"

_CONNECTION_ERROR_TYPES: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    aiohttp.ClientConnectorError,
    aiohttp.ServerDisconnectedError,
    ConnectionRefusedError,
    ConnectionResetError,
    OSError,
    openai.APIConnectionError,
)


class InferenceServiceWorkflow(RolloutWorkflow):
    def __init__(
        self,
        controller: RolloutControllerV2,
        agent: Any | None = None,
        gateway_addr: str = "",
        admin_api_key: str = "areal-admin-key",
        discount: float = 1.0,
        export_style: str = "individual",
        timeout: float | None = None,
        group_size: int = 1,
    ):
        self.controller = controller
        self.agent = agent
        self.gateway_addr = gateway_addr.rstrip("/") if gateway_addr else ""
        self._admin_api_key = admin_api_key
        self.discount = discount
        self.export_style = export_style
        self.timeout = timeout
        self.group_size = group_size

    @async_http_retry
    async def _start_session(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
        group_size: int = 1,
    ) -> tuple[str | None, list[tuple[str, str]]]:
        """Start one or more sessions. Returns (group_id, [(session_id, api_key), ...])."""
        url = f"{self.gateway_addr}/{_RL_START_SESSION_PATHNAME}"
        headers = {"Authorization": f"Bearer {self._admin_api_key}"}
        payload: dict[str, Any] = {"task_id": task_id, "group_size": group_size}
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
        group_id = data.get("group_id")
        credentials = [
            (s["session_id"], s["session_api_key"]) for s in data["sessions"]
        ]
        return group_id, credentials

    @async_http_retry
    async def _set_last_reward(
        self,
        session: aiohttp.ClientSession,
        reward: float,
        session_api_key: str,
    ) -> int | None:
        url = f"{self.gateway_addr}/{_RL_SET_REWARD_PATHNAME}"
        headers = {"Authorization": f"Bearer {session_api_key}"}
        payload: dict[str, Any] = {"interaction_id": None, "reward": reward}
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
        trajectory_id = data.get("trajectory_id")
        return int(trajectory_id) if trajectory_id is not None else None

    @async_http_retry
    async def _export_interactions(
        self,
        session: aiohttp.ClientSession,
        session_ids: list[str],
        group_id: str | None = None,
        trajectory_id: int | None = None,
    ) -> dict[str, InteractionWithTokenLogpReward]:
        url = f"{self.gateway_addr}/{_EXPORT_TRAJECTORIES_PATHNAME}"
        headers = {"Authorization": f"Bearer {self._admin_api_key}"}
        payload: dict[str, Any] = {
            "session_ids": session_ids,
            "group_id": group_id,
            "trajectory_id": trajectory_id,
            "discount": self.discount,
            "style": self.export_style,
            "remove_session": True,
        }
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()

        return deserialize_interactions(data["interactions"])

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        del engine
        http_session = await workflow_context.get_aiohttp_session()

        if self.agent is not None:
            return await self._run_offline(http_session, data)
        return await self._run_online(http_session)

    async def _run_offline(
        self,
        http_session: aiohttp.ClientSession,
        data: dict[str, Any],
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        task_id = workflow_context.get().task_id
        group_id, sessions = await self._start_session(
            http_session, str(task_id), group_size=self.group_size
        )

        assert self.agent is not None
        http_client = await workflow_context.get_httpx_client()

        async def _run_one(session_id: str, session_api_key: str) -> float:
            try:
                rewards = await self.agent.run(
                    data,
                    base_url=self.gateway_addr,
                    http_client=http_client,
                    api_key=session_api_key,
                )
                if isinstance(rewards, dict):
                    final_reward = float(
                        next(reversed(rewards.values())) if rewards else 0.0
                    )
                elif isinstance(rewards, (int, float)):
                    final_reward = float(rewards)
                else:
                    raise ValueError(f"Invalid reward type: {type(rewards)}")

                await self._set_last_reward(http_session, final_reward, session_api_key)
                return final_reward
            except Exception as exc:
                is_conn_err = isinstance(exc, _CONNECTION_ERROR_TYPES) or (
                    exc.__cause__ is not None
                    and isinstance(exc.__cause__, _CONNECTION_ERROR_TYPES)
                )
                if is_conn_err:
                    logger.warning(
                        "Agent task failed (%s). Trajectory rejected (connection lost).",
                        type(exc).__name__,
                    )
                else:
                    logger.warning(
                        "Agent task failed (%s: %s). This trajectory will be rejected.",
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )
                try:
                    await self._set_last_reward(http_session, 0.0, session_api_key)
                except Exception:
                    logger.warning(
                        "Failed to set reward for session %s in group %s",
                        session_id,
                        group_id,
                    )
                return 0.0

        rewards = await asyncio.gather(
            *[_run_one(sid, api_key) for sid, api_key in sessions]
        )

        session_ids = [sid for sid, _ in sessions]
        interactions = await self._export_interactions(
            http_session,
            session_ids,
            group_id=group_id,
        )
        if not interactions:
            logger.warning(
                "Group %s has no interactions, all trajectories rejected.",
                group_id,
            )
            return None

        tracker = stats_tracker.get(workflow_context.stat_scope())
        for r in rewards:
            tracker.scalar(reward=r)

        return interactions

    async def _run_online(
        self,
        http_session: aiohttp.ClientSession,
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        logger.debug("Waiting for next ready online trajectory")
        export_request = await self.controller.wait_for_online_trajectory(
            timeout=self.timeout
        )
        if not export_request:
            return None

        interactions = await self._export_interactions(
            http_session,
            [export_request["session_id"]],
            trajectory_id=export_request["trajectory_id"],
        )
        if not interactions:
            return None

        last_id = next(reversed(interactions))
        last_reward = interactions[last_id].reward
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=last_reward)
        return interactions

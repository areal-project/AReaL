# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import aiohttp
import httpx
import openai
import torch

from areal.api.workflow_api import RolloutWorkflow
from areal.infra import workflow_context
from areal.infra.rpc.rtensor import RTensor
from areal.infra.rpc.serialization import deserialize_value
from areal.infra.utils.http import async_http_retry
from areal.utils import logging, stats_tracker
from areal.v2.inference_service.data_proxy.session import TrajectoryDeliveryMode

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine
    from areal.experimental.openai.types import InteractionWithTokenLogpReward
    from areal.v2.inference_service.controller.controller import (
        RolloutControllerV2,
    )

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


def validate_trajectory_policy_version(
    traj: dict[str, Any], expected_policy_version: int
) -> None:
    """Require every loss-bearing token to come from the expected policy."""
    for field in ("versions", "loss_mask"):
        if field not in traj:
            raise ValueError(
                f"trajectory is missing required provenance field '{field}'"
            )

    provenance = RTensor.localize(
        {
            "versions": traj["versions"],
            "loss_mask": traj["loss_mask"],
        }
    )
    versions = provenance["versions"]
    loss_mask = provenance["loss_mask"]

    if not isinstance(versions, torch.Tensor) or not isinstance(
        loss_mask, torch.Tensor
    ):
        raise ValueError("trajectory versions and loss_mask must be PyTorch tensors")
    if versions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("trajectory versions must use a signed integer dtype")
    if versions.shape != loss_mask.shape:
        raise ValueError("trajectory versions and loss_mask must have the same shape")

    selected_versions = versions[loss_mask == 1]
    if selected_versions.numel() == 0:
        raise ValueError("trajectory has no loss-bearing tokens")

    observed_versions = sorted(torch.unique(selected_versions.detach().cpu()).tolist())
    if not torch.all(selected_versions == expected_policy_version).item():
        raise ValueError(
            "trajectory policy provenance mismatch: "
            f"expected policy version {expected_policy_version}, "
            f"observed {observed_versions} on loss-bearing tokens"
        )


async def _clear_trajectory_rtensors(traj: dict[str, Any]) -> None:
    """Best-effort cleanup for an exported trajectory that will be discarded."""
    shards_by_node = RTensor.collect_shards(traj)
    if not shards_by_node:
        return

    results = await asyncio.gather(
        *(
            RTensor.clear_node(node_addr, shard_ids)
            for node_addr, shard_ids in shards_by_node.items()
        ),
        return_exceptions=True,
    )
    for node_addr, result in zip(shards_by_node, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                "Failed to clear rejected trajectory shards on %s: %s",
                node_addr,
                result,
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
        expected_policy_version: int | None = None,
    ):
        self.controller = controller
        self.agent = agent
        self.gateway_addr = gateway_addr.rstrip("/") if gateway_addr else ""
        self._admin_api_key = admin_api_key
        self.discount = discount
        self.export_style = export_style
        self.timeout = timeout
        self.group_size = group_size
        self.expected_policy_version = expected_policy_version

    @async_http_retry
    async def _start_session(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
        group_size: int = 1,
        delivery_mode: TrajectoryDeliveryMode = TrajectoryDeliveryMode.CALLBACK,
    ) -> tuple[str | None, list[tuple[str, str]]]:
        """Start one or more sessions. Returns (group_id, [(session_id, api_key), ...])."""
        url = f"{self.gateway_addr}/{_RL_START_SESSION_PATHNAME}"
        headers = {"Authorization": f"Bearer {self._admin_api_key}"}
        payload: dict[str, Any] = {
            "task_id": task_id,
            "group_size": group_size,
            "delivery_mode": delivery_mode.value,
        }
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
    ) -> dict[str, Any]:
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

        return deserialize_value(data["traj"])

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
            http_session,
            str(task_id),
            group_size=self.group_size,
            delivery_mode=TrajectoryDeliveryMode.PULL,
        )

        assert self.agent is not None
        http_client = await workflow_context.get_httpx_client()

        async def _run_one(session_id: str, session_api_key: str) -> float | None:
            """Run one agent session. Returns reward on success, ``None`` on failure."""
            try:
                rewards = await self.agent.run(
                    data,
                    base_url=self.gateway_addr,
                    http_client=http_client,
                    api_key=session_api_key,
                    policy_version=self.expected_policy_version,
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
                return None

        results = await asyncio.gather(
            *[_run_one(sid, api_key) for sid, api_key in sessions]
        )

        session_ids = [sid for sid, _ in sessions]

        # Always export to trigger session cleanup on the data proxy,
        # even when we intend to discard the trajectories.
        traj = await self._export_interactions(
            http_session,
            session_ids,
            group_id=group_id,
        )
        keep_trajectory = False
        try:
            if not traj:
                return None

            n_failed = sum(r is None for r in results)
            if n_failed > 0:
                logger.warning(
                    "Abandoning group %s: %d/%d sessions failed",
                    group_id,
                    n_failed,
                    len(sessions),
                )
                return None

            if self.expected_policy_version is not None:
                await asyncio.to_thread(
                    validate_trajectory_policy_version,
                    traj,
                    self.expected_policy_version,
                )

            tracker = stats_tracker.get(workflow_context.stat_scope())
            for r in results:
                metrics: dict[str, float | int | None] = {"reward": r}
                if self.expected_policy_version is not None:
                    metrics["policy_version"] = self.expected_policy_version
                tracker.scalar(**metrics)

            keep_trajectory = True
            return traj
        finally:
            if not keep_trajectory:
                await _clear_trajectory_rtensors(traj)

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

        traj = await self._export_interactions(
            http_session,
            [export_request["session_id"]],
            trajectory_id=export_request["trajectory_id"],
        )
        keep_trajectory = False
        try:
            if not traj:
                return None

            rewards_tensor = traj.get("rewards")
            if isinstance(rewards_tensor, RTensor):
                rewards_tensor = rewards_tensor.to_local()

            if rewards_tensor is not None and len(rewards_tensor) > 0:
                last_reward = float(rewards_tensor[-1])
            elif (
                "interactions" in traj
                and traj["interactions"]
                and traj["interactions"][-1].get("reward") is not None
            ):
                last_reward = float(traj["interactions"][-1]["reward"])
            else:
                logger.warning(
                    "Exported trajectory is missing rewards. "
                    "This trajectory will be rejected."
                )
                return None

            if self.expected_policy_version is not None:
                await asyncio.to_thread(
                    validate_trajectory_policy_version,
                    traj,
                    self.expected_policy_version,
                )

            metrics: dict[str, float | int] = {"reward": last_reward}
            if self.expected_policy_version is not None:
                metrics["policy_version"] = self.expected_policy_version
            stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)
            keep_trajectory = True
            return traj
        finally:
            if not keep_trajectory:
                await _clear_trajectory_rtensors(traj)

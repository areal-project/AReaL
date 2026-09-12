# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import atexit
import inspect
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

import aiohttp

from areal.api import RolloutWorkflow
from areal.infra import workflow_context
from areal.utils import logging, stats_tracker
from areal.utils.perf_tracer import session_context, trace_session

from .client_session import OpenAIProxyClient, post_json
from .server import (
    DEFAULT_ADMIN_API_KEY,
    GRANT_CAPACITY_PATHNAME,
    RL_END_PROCESSOR_CACHE_GROUP_PATHNAME,
    ProcessorCacheGroupRequest,
    derive_session_gateway_api_key,
    derive_session_gateway_token,
)
from .tensor_reference import SharedTensorResolver

if TYPE_CHECKING:
    from ..client import TRolloutEngine
    from ..types import InteractionWithTokenLogpReward
    from .proxy_gateway import CompletedSessionInfo

logger = logging.getLogger("OpenAIProxyWorkflow")


AgentFailureDisposition = Literal[
    "model_failure_zero",
    "system_failure_reject",
    "unknown_failure_reject",
]


# Lazy-initialized process pool for running agent tasks
_executor: ProcessPoolExecutor | None = None
_executor_lock = threading.Lock()
_executor_max_workers: int | None = None


def _get_executor(max_workers: int = 4) -> ProcessPoolExecutor:
    """Get or create the shared process pool executor.

    Parameters
    ----------
    max_workers : int
        Maximum number of worker processes for the pool. Only used when
        creating a new executor. If an executor already exists, this
        parameter is ignored.
    """
    global _executor, _executor_max_workers
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ProcessPoolExecutor(max_workers=max_workers)
                _executor_max_workers = max_workers
                # Register cleanup on process exit
                atexit.register(_shutdown_executor)
    return _executor


def _shutdown_executor() -> None:
    """Shutdown the shared process pool executor if it exists."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None


def _wrap_run(agent: Any, data: dict[str, Any], extra_envs: dict[str, str]):
    """Run agent in subprocess with environment variables."""
    for key, value in extra_envs.items():
        os.environ[key] = value
    return asyncio.run(agent.run(data))


class OpenAIProxyWorkflow(RolloutWorkflow):
    """Run an agent through the v1 proxy.

    Group-scoped processor caching is supported in ``inline`` and ``subproc``
    modes. Online mode owns its sessions externally and does not receive the
    rollout group's cache identity.
    """

    def __init__(
        self,
        mode: str,
        agent: Any | None = None,
        proxy_addr: str = "",
        admin_api_key: str = DEFAULT_ADMIN_API_KEY,
        discount: float = 1.0,
        export_style: str = "individual",
        subproc_max_workers: int = 4,
        proxy_gateway_addr: str | None = None,
        drop_retry_orphans: bool = False,
    ):
        if mode not in ("inline", "subproc", "online"):
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'inline', 'subproc', or 'online'"
            )

        if mode == "online":
            if proxy_gateway_addr is None:
                raise ValueError("proxy_gateway_addr is required when mode='online'")
            from .online_agent import _OnlineAgent

            agent = _OnlineAgent(
                proxy_gateway_addr=proxy_gateway_addr,
                admin_api_key=admin_api_key,
            )
        else:
            if agent is None:
                raise ValueError("agent is required when mode is 'inline' or 'subproc'")
            # Validate that agent has an async 'run' method
            if not hasattr(agent, "run") or not callable(getattr(agent, "run")):
                raise TypeError(
                    f"Agent must have a callable 'run' method. "
                    f"Got agent of type {type(agent).__name__} without a callable 'run' attribute."
                )
            if not asyncio.iscoroutinefunction(agent.run):
                raise TypeError(
                    f"Agent's 'run' method must be an async function. "
                    f"Got {type(agent).__name__}.run which is not a coroutine function."
                )

        self.mode = mode
        self.agent = agent
        self.proxy_addr = proxy_addr
        self._admin_api_key = admin_api_key
        self.discount = discount
        self.export_style = export_style
        self.subproc_max_workers = subproc_max_workers
        self.drop_retry_orphans = drop_retry_orphans
        self._shared_tensor_resolver = SharedTensorResolver()

    @trace_session("run_agent")
    async def _run_agent(
        self,
        session_api_key: str,
        data: dict,
        *,
        session_id: str | None = None,
        worker_runtime: Any | None = None,
    ):
        if self.mode == "inline":
            http_client = await workflow_context.get_httpx_client()
            extra_kwargs = {
                "base_url": self.proxy_addr,
                "http_client": http_client,
                "api_key": session_api_key,
                # The public gateway receives only a generation-scoped key,
                # never the control-plane admin credential.
                "proxy_gateway_api_key": derive_session_gateway_api_key(
                    self._admin_api_key
                ),
            }
            if session_id is not None:
                extra_kwargs["session_id"] = session_id
                extra_kwargs["proxy_session_token"] = derive_session_gateway_token(
                    self._admin_api_key, session_id
                )
            if worker_runtime is not None:
                extra_kwargs["worker_runtime"] = worker_runtime
            return await self.agent.run(data, **extra_kwargs)
        if self.mode == "subproc":
            extra_envs = {
                "OPENAI_BASE_URL": self.proxy_addr,
                "OPENAI_API_KEY": session_api_key,
                "ANTHROPIC_BASE_URL": self.proxy_addr,
                "ANTHROPIC_API_KEY": session_api_key,
            }
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                _get_executor(max_workers=self.subproc_max_workers),
                _wrap_run,
                self.agent,
                data,
                extra_envs,
            )
        if self.mode == "online":
            http_client = await workflow_context.get_httpx_client()
            extra_kwargs = {
                "base_url": self.proxy_addr,
                "http_client": http_client,
                "api_key": self._admin_api_key,
            }
            return await self.agent.run(data, **extra_kwargs)
        raise ValueError(f"Unsupported mode: {self.mode}")

    def record_group_metrics(
        self,
        data: dict[str, Any],
        rewards: list[float | None],
        group_size: int,
    ) -> None:
        """Delegate optional group-level metric recording to the inline agent."""
        recorder = getattr(self.agent, "record_group_metrics", None)
        if callable(recorder):
            recorder(data, rewards, group_size)

    def record_episode_metrics(
        self,
        data: dict[str, Any],
        reward: float,
    ) -> None:
        """Delegate optional episode metrics after interactions are exported."""
        recorder = getattr(self.agent, "record_episode_metrics", None)
        if callable(recorder):
            recorder(data, reward)

    async def _call_agent_hook(self, name: str, *args: Any, **kwargs: Any) -> None:
        """Call an optional agent lifecycle hook without blocking the event loop."""

        hook = getattr(self.agent, name, None)
        if not callable(hook):
            return
        result = hook(*args, **kwargs)
        if inspect.isawaitable(result):
            await result

    def _classify_agent_failure(
        self,
        error: Exception,
        *,
        context_overflow: bool,
        interaction_count: int,
        system_error: bool = False,
    ) -> AgentFailureDisposition:
        """Classify whether an agent error is a trainable model failure.

        Existing agents retain the historical behavior: a typed proxy context
        overflow is recoverable as reward zero. Agents backed by authoritative
        external graders may provide ``classify_proxy_failure`` to reject system
        and unknown failures even when an earlier model request overflowed.
        """

        if system_error:
            return "system_failure_reject"

        classifier = getattr(self.agent, "classify_proxy_failure", None)
        if classifier is None:
            return (
                "model_failure_zero" if context_overflow else "unknown_failure_reject"
            )
        disposition = classifier(
            error,
            context_overflow=context_overflow,
            interaction_count=interaction_count,
        )
        valid_dispositions = {
            "model_failure_zero",
            "system_failure_reject",
            "unknown_failure_reject",
        }
        if disposition not in valid_dispositions:
            raise ValueError(
                "classify_proxy_failure returned an invalid disposition: "
                f"{disposition!r}"
            )
        return disposition

    async def _grant_capacity(self, session: aiohttp.ClientSession) -> None:
        """Grant capacity via HTTP."""
        url = f"{self.proxy_addr}/{GRANT_CAPACITY_PATHNAME}"
        headers = {"Authorization": f"Bearer {self._admin_api_key}"}
        async with session.post(url, headers=headers) as resp:
            resp.raise_for_status()

    def _processor_cache_group_id(
        self, context: workflow_context.WorkflowContext
    ) -> str | None:
        if self.mode == "online" or context.group_size <= 1 or context.task_id is None:
            return None
        scope = "eval" if context.is_eval else "train"
        return f"{scope}:{context.task_id}"

    async def _afinalize_processor_cache_group(
        self, context: workflow_context.WorkflowContext
    ) -> None:
        """Release processor and tensor-ref state after an inline/subproc group."""
        group_id = self._processor_cache_group_id(context)
        if group_id is None:
            return

        http_session = await workflow_context.get_aiohttp_session()
        try:
            await post_json(
                http_session,
                url=(f"{self.proxy_addr}/{RL_END_PROCESSOR_CACHE_GROUP_PATHNAME}"),
                payload=ProcessorCacheGroupRequest(group_id=group_id),
                headers={"Authorization": f"Bearer {self._admin_api_key}"},
            )
        finally:
            self._shared_tensor_resolver.discard(group_id)

    @session_context()
    async def arun_episode(
        self, engine: TRolloutEngine, data: dict[str, Any]
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        context = workflow_context.get()
        task_id = context.task_id
        # Qualify the proxy session with the group sample index so each group
        # member owns a distinct, run-stable session namespace.
        proxy_task_id = (
            f"{task_id}:{context.sample_idx}"
            if context.sample_idx is not None
            else str(task_id)
        )
        processor_cache_group_id = self._processor_cache_group_id(context)

        http_session = await workflow_context.get_aiohttp_session()

        # Grant capacity for clients, otherwise agent sessions are rejected.
        # Designed for online mode. Users' requests do not have any staleness
        # control, which may be detrimental to RL training. We use a hacky way
        # to control the staleness. The staleness is always explicitly controlled
        # by the rollout controller and staleness manager. If the code runs
        # to this point, it means that we are within the allowed staleness window,
        # so we can grant capacity to let the agent session proceed.
        await self._grant_capacity(http_session)

        if self.mode == "online":
            # Online mode: _OnlineAgent waits for external user session.
            # Returns CompletedSessionInfo with session credentials.
            session_info: CompletedSessionInfo = await self._run_agent(
                self._admin_api_key, data
            )

            # Create proxy client for export only (no start/end session).
            proxy_client = OpenAIProxyClient(
                session=http_session,
                base_url=self.proxy_addr,
                task_id=proxy_task_id,
                admin_api_key=self._admin_api_key,
            )
            proxy_client.session_id = session_info.session_id

            interactions = await proxy_client.export_interactions(
                discount=self.discount,
                style=self.export_style,
                drop_retry_orphans=self.drop_retry_orphans,
            )

            # Return None if no interactions (empty session — user never sent chat/completions)
            if not interactions:
                logger.warning(
                    f"Session {session_info.session_id} has no interactions, "
                    "trajectory will be rejected."
                )
                return None

            # Record stats
            last_id = next(reversed(interactions))
            last_reward = interactions[last_id].reward
            stats_tracker.get(workflow_context.stat_scope()).scalar(reward=last_reward)
            return interactions

        # ---- Normal mode (inline / subproc) ----

        proxy_client = OpenAIProxyClient(
            session=http_session,
            base_url=self.proxy_addr,
            task_id=proxy_task_id,
            admin_api_key=self._admin_api_key,
            processor_cache_group_id=processor_cache_group_id,
            processor_cache_group_size=context.group_size,
            shared_tensor_resolver=self._shared_tensor_resolver,
        )
        agent_error: Exception | None = None
        async with proxy_client:
            # Run the user code.
            try:
                rewards = await self._run_agent(
                    proxy_client.session_api_key,
                    data,
                    session_id=proxy_client.session_id,
                    worker_runtime=engine,
                )
            except Exception as exc:
                agent_error = exc
            else:
                # Assign rewards back according to user code output
                if isinstance(rewards, dict):
                    for completion_id, reward in rewards.items():
                        await proxy_client.set_reward(completion_id, reward)
                elif isinstance(rewards, float):
                    await proxy_client.set_last_reward(rewards)
                else:
                    raise ValueError(f"Invalid reward type: {type(rewards)}")

        failure_disposition: AgentFailureDisposition | None = None
        if agent_error is not None:
            failure_disposition = self._classify_agent_failure(
                agent_error,
                context_overflow=proxy_client.context_overflow,
                interaction_count=proxy_client.interaction_count,
                system_error=proxy_client.system_error,
            )
            await self._call_agent_hook(
                "record_failure_disposition",
                data,
                agent_error,
                failure_disposition,
            )

        if agent_error is not None and failure_disposition != "model_failure_zero":
            logger.warning(
                "Agent task failed with disposition %s (%s: %s). This "
                "trajectory will be rejected.",
                failure_disposition,
                type(agent_error).__name__,
                agent_error,
                exc_info=agent_error,
            )
            stats_tracker.get(workflow_context.stat_scope()).scalar(
                context_overflow=float(proxy_client.context_overflow),
                proxy_system_error=float(proxy_client.system_error),
            )
            raise agent_error

        if proxy_client.context_overflow or failure_disposition == "model_failure_zero":
            logger.warning(
                "Recovering model failure with reward 0: context_overflow=%s, "
                "interactions=%d, agent_error=%s",
                proxy_client.context_overflow,
                proxy_client.interaction_count,
                agent_error,
            )
            stats_tracker.get(workflow_context.stat_scope()).scalar(
                context_overflow=float(proxy_client.context_overflow),
                proxy_system_error=float(proxy_client.system_error),
            )
            if proxy_client.interaction_count == 0:
                await self._call_agent_hook(
                    "persist_episode_result",
                    data,
                    None,
                )
                return None
            await proxy_client.set_last_reward(0.0)
        else:
            stats_tracker.get(workflow_context.stat_scope()).scalar(
                context_overflow=0.0,
                proxy_system_error=float(proxy_client.system_error),
            )

        # Apply turn-level discount and export interactions
        interactions = await proxy_client.export_interactions(
            discount=self.discount,
            style=self.export_style,
            drop_retry_orphans=self.drop_retry_orphans,
        )

        if not interactions:
            logger.warning(
                "Task %s exported no usable interactions; trajectory will be rejected.",
                task_id,
            )
            await self._call_agent_hook(
                "persist_episode_result",
                data,
                None,
            )
            return None

        # Record stats
        last_id = list(interactions.keys())[-1]
        if last_id:
            last_reward = interactions[last_id].reward
            await self._call_agent_hook(
                "persist_episode_result",
                data,
                last_reward,
            )
            tracker = stats_tracker.get(workflow_context.stat_scope())
            tracker.scalar(reward=last_reward)
            self.record_episode_metrics(data, last_reward)

        return interactions

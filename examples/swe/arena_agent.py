"""Arena Stream agent workflow using AReaL's OpenAI-compatible proxy."""

from __future__ import annotations

import asyncio
import hmac
import json
import math
import os
import re
import socket
import stat
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from examples.swe.arena_client import (
    ArenaAPIError,
    ArenaOpenAPIClient,
    ArenaTaskFailedError,
    ArenaTaskResult,
    LLMProtocol,
)
from examples.swe.arena_config import load_arena_stream_configs
from examples.swe.arena_types import ArenaStreamConfig

from areal.infra import workflow_context
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string

logger = logging.getLogger("ArenaStreamAgent")

_ARENA_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}$")
_WORKER_GATEWAY_REGISTRY_ATTR = "_arena_session_gateway_registry_v1"
_WORKER_GATEWAY_CLEANUP_KEY = "arena-session-gateway-registrations"


def _record_arena_metrics(**metrics: float) -> None:
    stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)


def _record_arena_domain_reward(reward: float, arena_task_type: str) -> None:
    _record_arena_metrics(**{f"{arena_task_type}/reward": float(reward)})


@dataclass
class _WorkerGatewayTarget:
    target: tuple[str, str]
    deployment_id: str
    client: ArenaOpenAPIClient
    http_client: httpx.AsyncClient
    proxy_gateway_api_key: str
    registration_timeout: float
    probe_interval: float
    last_probe_at: float
    known_missing: bool = False
    users: int = 0
    probe_task: asyncio.Task[None] | None = None


class _WorkerSessionGatewayRegistry:
    """Own Arena registrations for one rollout worker lifetime."""

    def __init__(self, *, persistent: bool) -> None:
        self.persistent = persistent
        self._lock = asyncio.Lock()
        self._targets: dict[tuple[str, str], _WorkerGatewayTarget] = {}
        self._closed = False

    async def acquire(
        self,
        *,
        arena_client: ArenaOpenAPIClient,
        proxy_base_url: str,
        proxy_gateway_api_key: str,
        http_client: httpx.AsyncClient,
        registration_timeout: float,
        probe_interval: float,
    ) -> tuple[str, str]:
        key = (arena_client.base_url, proxy_base_url)
        async with self._lock:
            if self._closed:
                raise RuntimeError("Arena worker session gateway registry is closed")
            entry = self._targets.get(key)
            if entry is None:
                suffix = uuid.uuid4().hex[:16]
                model_name = f"stream-areal-session-{suffix}"
                deployment_id = f"session-{suffix}"
                try:
                    target = await self._register_target(
                        arena_client=arena_client,
                        model_name=model_name,
                        deployment_id=deployment_id,
                        proxy_base_url=proxy_base_url,
                        proxy_gateway_api_key=proxy_gateway_api_key,
                        http_client=http_client,
                        registration_timeout=registration_timeout,
                    )
                except (Exception, asyncio.CancelledError):
                    _record_arena_metrics(**{"arena/registration_success": 0.0})
                    _record_arena_metrics(**{"arena/call_success": 0.0})
                    # A POST may have reached Arena even when its response was
                    # lost or the local task was cancelled. The generated name
                    # is exact and DELETE treats 404 as success, so remove any
                    # uncertain registration before propagating the failure.
                    try:
                        await asyncio.wait_for(
                            arena_client.delete_llm_proxy_async(
                                model_name,
                                client=http_client,
                                timeout=registration_timeout,
                            ),
                            timeout=registration_timeout,
                        )
                        _record_arena_metrics(**{"arena/cleanup_success": 1.0})
                    except Exception:
                        logger.exception(
                            "Failed to delete uncertain worker-scoped Arena "
                            "session gateway: model_id=%s",
                            model_name,
                        )
                        _record_arena_metrics(**{"arena/cleanup_success": 0.0})
                    raise
                entry = _WorkerGatewayTarget(
                    target=target,
                    deployment_id=deployment_id,
                    client=arena_client,
                    http_client=http_client,
                    proxy_gateway_api_key=proxy_gateway_api_key,
                    registration_timeout=registration_timeout,
                    probe_interval=probe_interval,
                    last_probe_at=asyncio.get_running_loop().time(),
                )
                self._targets[key] = entry
                _record_arena_metrics(**{"arena/registration_success": 1.0})
                logger.info(
                    "Registered worker-scoped Arena session gateway: "
                    "model_id=%s, proxy_base_url=%s",
                    target[1],
                    proxy_base_url,
                )
            elif not hmac.compare_digest(
                entry.proxy_gateway_api_key, proxy_gateway_api_key
            ):
                raise RuntimeError(
                    "Arena proxy gateway key changed within one rollout worker"
                )
            else:
                entry.client = arena_client
                entry.http_client = http_client
                entry.registration_timeout = registration_timeout
                entry.probe_interval = min(entry.probe_interval, probe_interval)
                await self._ensure_target_locked(key, entry)
            entry.users += 1
            if entry.probe_task is None:
                entry.probe_task = asyncio.create_task(
                    self._probe_while_active(key, entry),
                    name=f"arena-model-probe-{entry.target[1]}",
                )
            return entry.target

    @staticmethod
    async def _register_target(
        *,
        arena_client: ArenaOpenAPIClient,
        model_name: str,
        deployment_id: str,
        proxy_base_url: str,
        proxy_gateway_api_key: str,
        http_client: httpx.AsyncClient,
        registration_timeout: float,
    ) -> tuple[str, str]:
        return await asyncio.wait_for(
            arena_client.register_llm_proxy_async(
                model_name=model_name,
                upstream_base_url=proxy_base_url,
                upstream_api_key=proxy_gateway_api_key,
                deployment_id=deployment_id,
                protocol="chat_completions",
                client=http_client,
                timeout=registration_timeout,
            ),
            timeout=registration_timeout,
        )

    async def _ensure_target_locked(
        self,
        key: tuple[str, str],
        entry: _WorkerGatewayTarget,
    ) -> None:
        now = asyncio.get_running_loop().time()
        if now - entry.last_probe_at < entry.probe_interval:
            if entry.known_missing:
                raise ArenaAPIError(
                    "Arena session gateway registration is unavailable while "
                    f"restore retries are throttled: model_id={entry.target[1]}"
                )
            return

        model_name = entry.target[1]
        try:
            exists = await asyncio.wait_for(
                entry.client.llm_proxy_exists_async(
                    model_name,
                    client=entry.http_client,
                    timeout=entry.registration_timeout,
                ),
                timeout=entry.registration_timeout,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # The probe is advisory unless Arena explicitly confirms a 404.
            # Throttle failures too, otherwise task concurrency can become a
            # control-plane request storm during an Arena outage.
            entry.last_probe_at = asyncio.get_running_loop().time()
            _record_arena_metrics(**{"arena/registration_probe_success": 0.0})
            logger.warning(
                "Arena session gateway probe failed; retaining cached route: "
                "model_id=%s",
                model_name,
                exc_info=True,
            )
            if entry.known_missing:
                raise ArenaAPIError(
                    "Arena session gateway registration remains unavailable after "
                    f"a failed liveness probe: model_id={model_name}"
                )
            return

        entry.last_probe_at = asyncio.get_running_loop().time()
        _record_arena_metrics(**{"arena/registration_probe_success": 1.0})
        _record_arena_metrics(**{"arena/registration_present": float(exists)})
        if exists:
            entry.known_missing = False
            return

        entry.known_missing = True
        logger.warning(
            "Arena session gateway registration is missing (possibly garbage-"
            "collected); restoring the original model_id=%s",
            model_name,
        )
        try:
            restored_target = await self._register_target(
                arena_client=entry.client,
                model_name=model_name,
                deployment_id=entry.deployment_id,
                proxy_base_url=key[1],
                proxy_gateway_api_key=entry.proxy_gateway_api_key,
                http_client=entry.http_client,
                registration_timeout=entry.registration_timeout,
            )
        except asyncio.CancelledError:
            entry.last_probe_at = asyncio.get_running_loop().time()
            _record_arena_metrics(**{"arena/registration_restore_success": 0.0})
            raise
        except Exception:
            # Reusing the original name is required by already-queued tasks.
            # A concurrent restore or a lost POST response can therefore look
            # like a failed create even though the exact route now exists.
            try:
                restored = await asyncio.wait_for(
                    entry.client.llm_proxy_exists_async(
                        model_name,
                        client=entry.http_client,
                        timeout=entry.registration_timeout,
                    ),
                    timeout=entry.registration_timeout,
                )
            except Exception:
                restored = False
            if not restored:
                entry.last_probe_at = asyncio.get_running_loop().time()
                _record_arena_metrics(**{"arena/registration_restore_success": 0.0})
                raise
            restored_target = entry.target
        entry.target = restored_target
        entry.known_missing = False
        entry.last_probe_at = asyncio.get_running_loop().time()
        _record_arena_metrics(**{"arena/registration_restore_success": 1.0})
        logger.info(
            "Restored missing Arena session gateway: model_id=%s, proxy_base_url=%s",
            model_name,
            key[1],
        )

    async def _probe_while_active(
        self,
        key: tuple[str, str],
        expected_entry: _WorkerGatewayTarget,
    ) -> None:
        try:
            while True:
                await asyncio.sleep(expected_entry.probe_interval)
                async with self._lock:
                    entry = self._targets.get(key)
                    if self._closed or entry is not expected_entry or entry.users <= 0:
                        return
                    try:
                        await self._ensure_target_locked(key, entry)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.warning(
                            "Failed to restore Arena session gateway during "
                            "background probe: model_id=%s",
                            entry.target[1],
                            exc_info=True,
                        )
        except asyncio.CancelledError:
            return

    async def release(
        self,
        *,
        arena_base_url: str,
        proxy_base_url: str,
        http_client: httpx.AsyncClient,
        registration_timeout: float,
    ) -> None:
        key = (arena_base_url, proxy_base_url)
        probe_task: asyncio.Task[None] | None = None
        delete_entry: _WorkerGatewayTarget | None = None
        async with self._lock:
            entry = self._targets.get(key)
            if entry is None or entry.users <= 0:
                logger.error(
                    "Arena session gateway reference underflow: proxy_base_url=%s",
                    proxy_base_url,
                )
                _record_arena_metrics(**{"arena/cleanup_success": 0.0})
                return
            entry.users -= 1
            if entry.users == 0:
                probe_task = entry.probe_task
                entry.probe_task = None
            if not self.persistent and entry.users == 0:
                delete_entry = self._targets.pop(key)

        if probe_task is not None:
            probe_task.cancel()
            try:
                await probe_task
            except asyncio.CancelledError:
                pass

        if delete_entry is not None:
            try:
                await asyncio.wait_for(
                    delete_entry.client.delete_llm_proxy_async(
                        delete_entry.target[1],
                        client=http_client,
                        timeout=registration_timeout,
                    ),
                    timeout=registration_timeout,
                )
                logger.info(
                    "Deleted prompt-scoped Arena session gateway: model_id=%s",
                    delete_entry.target[1],
                )
                _record_arena_metrics(**{"arena/cleanup_success": 1.0})
            except Exception:
                logger.exception(
                    "Failed to delete prompt-scoped Arena session gateway: model_id=%s",
                    delete_entry.target[1],
                )
                _record_arena_metrics(**{"arena/cleanup_success": 0.0})

    def close(self) -> None:
        """Synchronously remove persistent registrations during worker destroy."""

        if self._closed:
            return
        self._closed = True
        entries = list(self._targets.values())
        self._targets.clear()
        for entry in entries:
            if entry.probe_task is not None:
                loop = entry.probe_task.get_loop()
                try:
                    if loop.is_running():
                        loop.call_soon_threadsafe(entry.probe_task.cancel)
                    elif not loop.is_closed():
                        entry.probe_task.cancel()
                except RuntimeError:
                    logger.warning(
                        "Failed to cancel Arena session gateway probe during "
                        "worker shutdown: model_id=%s",
                        entry.target[1],
                        exc_info=True,
                    )
            try:
                entry.client.delete_llm_proxy(entry.target[1])
                logger.info(
                    "Deleted worker-scoped Arena session gateway: model_id=%s",
                    entry.target[1],
                )
            except Exception:
                logger.exception(
                    "Failed to delete worker-scoped Arena session gateway: model_id=%s",
                    entry.target[1],
                )


def _open_directory_no_symlinks(path: Path) -> int:
    """Create/open an absolute directory path without traversing symlinks."""

    absolute_path = Path(os.path.abspath(path))
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(absolute_path.anchor, directory_flags)
    try:
        for part in absolute_path.parts[1:]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            next_fd = os.open(part, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def _arena_task_type(data: dict[str, Any]) -> str:
    arena_task_type = data.get("arena_task_type", "unknown")
    if not isinstance(arena_task_type, str) or not arena_task_type.strip():
        return "unknown"
    return arena_task_type.strip()


class ArenaStreamAgentWorkflow:
    """Launch an Arena online task and use its returned reward for RL."""

    _MODEL_FAILURE_STATUSES_WITH_INTERACTIONS = {"NO_OUTPUT", "TIMEOUT"}
    _SYSTEM_FAILURE_STATUSES = ArenaOpenAPIClient.FAILED_TASK_STATUSES - {
        "HARNESS_FAILED",
        *_MODEL_FAILURE_STATUSES_WITH_INTERACTIONS,
    }

    def __init__(
        self,
        econfig: dict[str, Any] | None = None,
        gen_args: dict[str, Any] | None = None,
        timeout: float = 3600.0,
    ) -> None:
        self.econfig = econfig or {}
        self.gen_args = gen_args or {}
        self.timeout = float(self.econfig.get("timeout", timeout))
        self.registration_timeout = float(
            self.econfig.get("arena_registration_timeout", 180.0)
        )
        self.registration_probe_interval = float(
            self.econfig.get("arena_registration_probe_interval", 60.0)
        )
        if not math.isfinite(self.registration_probe_interval) or (
            self.registration_probe_interval <= 0
        ):
            raise ValueError(
                "arena_registration_probe_interval must be finite and positive"
            )
        self.request_timeout = float(self.econfig.get("arena_request_timeout", 60.0))
        if self.request_timeout <= 0:
            raise ValueError("arena_request_timeout must be positive")
        self.llm_route_mode = str(
            self.econfig.get("arena_llm_route_mode", "gateway") or "gateway"
        ).strip()
        if self.llm_route_mode not in {"direct", "gateway", "session_gateway"}:
            raise ValueError(
                "arena_llm_route_mode must be 'direct', 'gateway', or 'session_gateway'"
            )
        stream_configs = load_arena_stream_configs(self.econfig)
        self.stream_configs = {config.name: config for config in stream_configs}
        self.default_stream_name = (
            stream_configs[0].name if len(stream_configs) == 1 else ""
        )
        self.reward_transforms = {
            config.name: (
                import_from_string(config.reward_transform_fn)
                if config.reward_transform_fn
                else None
            )
            for config in stream_configs
        }
        self._task_result: ContextVar[ArenaTaskResult | None] = ContextVar(
            "arena_task_result", default=None
        )
        self._task_result_dumped: ContextVar[bool] = ContextVar(
            "arena_task_result_dumped", default=False
        )
        self.result_dump_dir = str(
            self.econfig.get("arena_result_dump_dir", "") or ""
        ).strip()
        self.result_dump_max_bytes = int(
            self.econfig.get("arena_result_dump_max_bytes", 1_000_000)
        )
        if self.result_dump_max_bytes <= 0:
            raise ValueError("arena_result_dump_max_bytes must be positive")
        self._result_dump_lock = asyncio.Lock()
        # Unit tests and callers that invoke the agent directly do not provide a
        # rollout-worker runtime. Keep their registration scoped to this agent.
        # Normal inline RL attaches a persistent registry to RemoteInfEngine.
        self._local_session_gateway_registry = _WorkerSessionGatewayRegistry(
            persistent=False
        )
        self.client = ArenaOpenAPIClient(
            base_url=str(self.econfig.get("arena_base_url", "")),
            timeout=self.timeout,
            poll_interval=float(self.econfig.get("arena_poll_interval", 5.0)),
            request_retries=int(self.econfig.get("arena_request_retries", 3)),
        )

    def record_group_metrics(
        self,
        data: dict[str, Any],
        rewards: list[float | None],
        group_size: int,
    ) -> None:
        """Record whether any of the k rollouts solved the Arena task."""
        if len(rewards) != group_size:
            raise ValueError(
                f"Expected {group_size} Arena rewards for pass@k, got {len(rewards)}"
            )
        arena_task_type = _arena_task_type(data)
        stream_config = self._stream_config_for_data(data)
        pass_count = sum(
            reward is not None and self._is_solved_reward(reward, stream_config)
            for reward in rewards
        )
        group_pass_distribution = {
            f"group_pass_{count}": float(count == pass_count)
            for count in range(group_size + 1)
        }
        _record_arena_metrics(
            **{
                "pass@k": float(pass_count > 0),
                f"{arena_task_type}/pass@k": float(pass_count > 0),
                f"stream/{stream_config.name}/pass@k": float(pass_count > 0),
                **group_pass_distribution,
            }
        )

    @classmethod
    def _is_model_attributed_harness_failure(cls, error: ArenaTaskFailedError) -> bool:
        """Recognize the Harness' explicit Claude agent-phase failure envelope."""

        result = error.result
        raw = result.raw if result is not None else None
        if not isinstance(raw, dict):
            return False
        detail = raw.get("error")
        if not isinstance(detail, str):
            return False
        normalized = detail.lower()
        return (
            "harness: harness agent phase exited with code" in normalized
            and "harness: agent phase error: claude reported error" in normalized
        )

    @classmethod
    def classify_proxy_failure(
        cls,
        error: Exception,
        *,
        context_overflow: bool,
        interaction_count: int,
    ) -> str:
        """Classify Arena failures without parsing grader-owned ``raw`` text.

        A typed local context overflow or the Harness' explicit Claude
        agent-phase error envelope attributes the failure to the model-facing
        agent lifecycle. Everything ambiguous is rejected until Arena exposes
        versioned failure attribution and semantic codes in its stable result
        envelope.
        """

        if not isinstance(error, ArenaTaskFailedError):
            return "system_failure_reject"
        if error.status in cls._SYSTEM_FAILURE_STATUSES:
            return "system_failure_reject"
        if (
            error.status in cls._MODEL_FAILURE_STATUSES_WITH_INTERACTIONS
            and interaction_count > 0
        ):
            return "model_failure_zero"
        if (
            error.status == "HARNESS_FAILED"
            and interaction_count > 0
            and (context_overflow or cls._is_model_attributed_harness_failure(error))
        ):
            return "model_failure_zero"
        return "unknown_failure_reject"

    def record_episode_metrics(
        self,
        data: dict[str, Any],
        reward: float,
    ) -> None:
        """Record terminal result metrics after a valid interaction export."""
        result = self._task_result.get()
        stream_config = self._stream_config_for_data(data)
        arena_task_type = _arena_task_type(data)
        metrics = {
            "training_score": float(reward),
            f"stream/{stream_config.name}/training_score": float(reward),
        }
        if result is not None and result.score is not None:
            metrics.update(
                {
                    "arena_score": result.score,
                    # Compatibility alias: this has always meant the scalar score,
                    # never the heterogeneous Arena ``raw`` object.
                    "raw_reward": result.score,
                    f"{arena_task_type}/raw_reward": result.score,
                    "arena_raw_present": float(result.raw is not None),
                    "arena_trace_present": float(result.trace_id is not None),
                    "arena_artifacts_present": float(result.artifacts_uri is not None),
                    f"stream/{stream_config.name}/arena_score": result.score,
                }
            )
        try:
            _record_arena_metrics(**metrics)
            if result is not None and result.score is not None:
                data_id = str(data.get("data_id") or "").strip()
            else:
                data_id = ""
            if data_id:
                stats_tracker.get("debug").scalar(
                    **{f"task/{data_id}/raw_reward": result.score}
                )
            _record_arena_domain_reward(reward, arena_task_type)
        finally:
            self._task_result.set(None)
            self._task_result_dumped.set(False)

    async def persist_episode_result(
        self,
        data: dict[str, Any],
        reward: float | None,
    ) -> None:
        """Persist the final reward after proxy-level failure recovery."""

        result = self._task_result.get()
        if result is None or self._task_result_dumped.get():
            return
        await self._dump_task_result(
            data,
            self._stream_config_for_data(data),
            result,
            training_score=None if reward is None else float(reward),
        )
        self._task_result_dumped.set(True)

    async def record_failure_disposition(
        self,
        data: dict[str, Any],
        error: Exception,
        disposition: str,
    ) -> None:
        """Audit rejected results now; recovered failures use the final reward hook."""

        if isinstance(error, ArenaTaskFailedError) and error.result is not None:
            self._task_result.set(error.result)
        if disposition == "model_failure_zero":
            return
        await self.persist_episode_result(data, None)

    async def run(
        self,
        data: dict[str, Any],
        **extra_kwargs: Any,
    ) -> float:
        """Launch the row's task with the current rollout proxy session."""
        self._task_result.set(None)
        self._task_result_dumped.set(False)
        stream_config = self._stream_config_for_data(data)
        stream_id = str(data.get("stream_id") or stream_config.stream_id or "")
        data_id = str(data.get("data_id") or "")
        llm_protocol: LLMProtocol = data.get(
            "llm_protocol", stream_config.llm_protocol or "chat_completions"
        )
        proxy_base_url = extra_kwargs.get("base_url")
        proxy_api_key = extra_kwargs.get("api_key")
        proxy_session_id = str(extra_kwargs.get("session_id") or "").strip()
        arena_http_client: httpx.AsyncClient | None = extra_kwargs.get(
            "arena_http_client"
        ) or extra_kwargs.get("http_client")

        if not stream_id:
            raise ValueError("stream_id is required for ArenaStreamAgentWorkflow")
        if not data_id:
            raise ValueError("data_id is required for ArenaStreamAgentWorkflow")
        if not proxy_base_url:
            raise ValueError("base_url is required for ArenaStreamAgentWorkflow")
        if not proxy_api_key:
            raise ValueError("api_key is required for ArenaStreamAgentWorkflow")
        if self.llm_route_mode in {"direct", "session_gateway"} and not (
            proxy_session_id and _ARENA_SESSION_ID_PATTERN.fullmatch(proxy_session_id)
        ):
            raise ValueError(
                "a valid session_id is required when arena_llm_route_mode is "
                "'direct' or 'session_gateway'"
            )
        proxy_gateway_api_key = str(
            extra_kwargs.get("proxy_gateway_api_key") or ""
        ).strip()
        proxy_session_token = str(extra_kwargs.get("proxy_session_token") or "").strip()
        worker_runtime = extra_kwargs.get("worker_runtime")
        if self.llm_route_mode == "session_gateway" and not (
            proxy_gateway_api_key and proxy_session_token
        ):
            raise ValueError(
                "proxy_gateway_api_key and proxy_session_token are required when "
                "arena_llm_route_mode='session_gateway'"
            )
        self._validate_row_reward_ref(data, stream_config)
        self._validate_row_llm_protocol(llm_protocol, stream_config)
        if arena_http_client is None:
            async with httpx.AsyncClient(timeout=self.request_timeout) as check_client:
                await self._validate_live_reward_ref(
                    stream_id, stream_config, check_client
                )
        else:
            await self._validate_live_reward_ref(
                stream_id, stream_config, arena_http_client
            )

        suffix = uuid.uuid4().hex[:12]
        model_name = f"stream-areal-{suffix}"
        deployment_id = str(uuid.uuid4())
        registered_model_id = model_name
        registered_url = str(proxy_base_url).rstrip("/")
        launch_api_key = str(proxy_api_key)
        launch_task_envs = dict(stream_config.task_envs)
        registration_created = False
        session_gateway_acquired = False
        session_gateway_registry: _WorkerSessionGatewayRegistry | None = None
        session_gateway_key = str(proxy_base_url).rstrip("/")
        owns_client = arena_http_client is None
        client = arena_http_client or httpx.AsyncClient(timeout=self.timeout)
        launch_recorded = False
        launched_task_id: str | None = None

        def record_launch_success(task_id: str | None) -> None:
            nonlocal launch_recorded, launched_task_id
            launched_task_id = task_id
            if not launch_recorded:
                _record_arena_metrics(**{"arena/launch_success": 1.0})
                launch_recorded = True

        async def audit_incomplete_task(status: str) -> None:
            if not launched_task_id:
                return
            # TODO(agent): call Arena task cancellation here once a supported
            # endpoint is present in the published OpenAPI schema.
            incomplete_result = ArenaTaskResult(
                task_id=launched_task_id,
                status=status,
                score=None,
            )
            self._task_result.set(incomplete_result)
            await self._dump_task_result(
                data,
                stream_config,
                incomplete_result,
                training_score=None,
            )
            self._task_result_dumped.set(True)

        try:
            if self.llm_route_mode == "gateway":
                # Registration POST retries can have an uncertain-success
                # outcome, so always attempt exact-name cleanup afterward.
                registration_created = True
                try:
                    (
                        registered_url,
                        registered_model_id,
                    ) = await asyncio.wait_for(
                        self.client.register_llm_proxy_async(
                            model_name=model_name,
                            upstream_base_url=str(proxy_base_url),
                            upstream_api_key=str(proxy_api_key),
                            deployment_id=deployment_id,
                            protocol=llm_protocol,
                            client=client,
                            timeout=self.registration_timeout,
                        ),
                        timeout=self.registration_timeout,
                    )
                except Exception:
                    _record_arena_metrics(**{"arena/registration_success": 0.0})
                    _record_arena_metrics(**{"arena/call_success": 0.0})
                    raise
                launch_api_key = self.client.llm_gateway_api_key
                _record_arena_metrics(**{"arena/registration_success": 1.0})
                logger.info(
                    "Registered Arena LLM proxy: model_name=%s, model_id=%s, "
                    "registered_url=%s, protocol=%s",
                    model_name,
                    registered_model_id,
                    registered_url,
                    llm_protocol,
                )
            elif self.llm_route_mode == "direct":
                # The launch API injects these values into the Harness.  Route
                # directly with the existing least-privilege per-session key;
                # this avoids a control-plane model registration for every leaf.
                proxy_host = urlparse(registered_url).hostname
                if not proxy_host:
                    raise ValueError(
                        f"Direct Arena LLM route has no hostname: {registered_url!r}"
                    )
                no_proxy_hosts = [
                    host.strip()
                    for host in str(
                        launch_task_envs.get("NO_PROXY")
                        or launch_task_envs.get("no_proxy")
                        or ""
                    ).split(",")
                    if host.strip()
                ]
                if proxy_host not in no_proxy_hosts:
                    no_proxy_hosts.append(proxy_host)
                no_proxy = ",".join(no_proxy_hosts)
                launch_task_envs.update(
                    {
                        "ANTHROPIC_BASE_URL": registered_url,
                        "ANTHROPIC_API_KEY": launch_api_key,
                        "OPENAI_BASE_URL": registered_url,
                        "OPENAI_API_KEY": launch_api_key,
                        # ARCA installs an egress gateway in Harness sandboxes.
                        # The rollout proxy is a cluster-private address and
                        # must bypass that gateway to remain reachable.
                        "NO_PROXY": no_proxy,
                        "no_proxy": no_proxy,
                    }
                )
                _record_arena_metrics(**{"arena/direct_route": 1.0})
                logger.info(
                    "Using direct Arena LLM route: model_name=%s, protocol=%s, "
                    "session_id=%s",
                    model_name,
                    llm_protocol,
                    proxy_session_id,
                )
            else:
                session_gateway_registry = self._session_gateway_registry(
                    worker_runtime
                )
                (
                    registered_url,
                    registered_model_id,
                ) = await session_gateway_registry.acquire(
                    arena_client=self.client,
                    proxy_base_url=session_gateway_key,
                    proxy_gateway_api_key=proxy_gateway_api_key,
                    http_client=client,
                    registration_timeout=self.registration_timeout,
                    probe_interval=self.registration_probe_interval,
                )
                session_gateway_acquired = True
                launch_api_key = self.client.llm_gateway_api_key
                custom_headers = str(
                    launch_task_envs.get("ANTHROPIC_CUSTOM_HEADERS") or ""
                ).strip()
                session_headers = (
                    f"X-Session-Id: {proxy_session_id}\n"
                    f"X-Session-Token: {proxy_session_token}"
                )
                launch_task_envs["ANTHROPIC_CUSTOM_HEADERS"] = (
                    f"{custom_headers}\n{session_headers}"
                    if custom_headers
                    else session_headers
                )
                _record_arena_metrics(**{"arena/session_gateway_route": 1.0})
                logger.info(
                    "Using Arena session gateway route: model_id=%s, "
                    "protocol=%s, session_id=%s",
                    registered_model_id,
                    llm_protocol,
                    proxy_session_id,
                )
            logger.info(
                f"Launching Arena task: stream_id={stream_id}, data_id={data_id}"
            )
            try:
                task_result_or_score = await asyncio.wait_for(
                    self.client.launch_one_task_result(
                        stream_id=stream_id,
                        data_id=data_id,
                        model_name=registered_model_id,
                        proxy_base_url=registered_url,
                        proxy_api_key=launch_api_key,
                        session_id=proxy_session_id,
                        harness=stream_config.harness,
                        task_envs=launch_task_envs,
                        client=client,
                        on_launch_success=record_launch_success,
                    ),
                    timeout=self.timeout,
                )
            except ArenaTaskFailedError as exc:
                _record_arena_metrics(**{"arena/terminal_success": 0.0})
                _record_arena_metrics(**{"arena/call_success": 0.0})
                if exc.result is not None:
                    self._task_result.set(exc.result)
                raise
            except asyncio.CancelledError:
                await audit_incomplete_task("LOCAL_WAIT_CANCELLED")
                if launch_recorded:
                    _record_arena_metrics(**{"arena/terminal_success": 0.0})
                _record_arena_metrics(**{"arena/call_success": 0.0})
                logger.warning(
                    "Arena task wait cancelled locally: task_id=%s data_id=%s",
                    launched_task_id or "unknown",
                    data_id,
                )
                raise
            except (TimeoutError, asyncio.TimeoutError) as exc:  # noqa: UP041
                await audit_incomplete_task("LOCAL_WAIT_TIMEOUT")
                logger.warning(
                    "Arena task wait timed out: task_id=%s data_id=%s "
                    "timeout_seconds=%s",
                    launched_task_id or "unknown",
                    data_id,
                    self.timeout,
                )
                if launch_recorded:
                    _record_arena_metrics(**{"arena/terminal_success": 0.0})
                else:
                    _record_arena_metrics(**{"arena/launch_success": 0.0})
                _record_arena_metrics(**{"arena/call_success": 0.0})
                raise TimeoutError from exc
            except Exception:
                if launch_recorded:
                    _record_arena_metrics(**{"arena/terminal_success": 0.0})
                else:
                    _record_arena_metrics(**{"arena/launch_success": 0.0})
                _record_arena_metrics(**{"arena/call_success": 0.0})
                raise
            if isinstance(task_result_or_score, ArenaTaskResult):
                task_result = task_result_or_score
            elif isinstance(task_result_or_score, Real) and not isinstance(
                task_result_or_score, bool
            ):
                # Compatibility for existing custom fakes and older clients.
                task_result = ArenaTaskResult(
                    task_id=launched_task_id or "legacy-result",
                    status="DONE",
                    score=float(task_result_or_score),
                )
            else:
                raise ArenaAPIError(
                    "Arena launch returned neither ArenaTaskResult nor numeric score"
                )
            self._task_result.set(task_result)
            try:
                if task_result.score is None:
                    raise ArenaAPIError("Successful Arena task result is missing score")
                if not math.isfinite(task_result.score):
                    raise ArenaAPIError("Successful Arena task score must be finite")
                reward = self._transform_reward(task_result.score, data, stream_config)
            except Exception:
                await self._dump_task_result(
                    data,
                    stream_config,
                    task_result,
                    training_score=None,
                )
                self._task_result_dumped.set(True)
                _record_arena_metrics(**{"arena/terminal_success": 1.0})
                _record_arena_metrics(**{"arena/result_processing_success": 0.0})
                _record_arena_metrics(**{"arena/call_success": 0.0})
                raise
            _record_arena_metrics(**{"arena/result_processing_success": 1.0})
            _record_arena_metrics(**{"arena/terminal_success": 1.0})
            _record_arena_metrics(**{"arena/call_success": 1.0})
        finally:
            if session_gateway_acquired and session_gateway_registry is not None:
                await session_gateway_registry.release(
                    arena_base_url=self.client.base_url,
                    proxy_base_url=session_gateway_key,
                    http_client=client,
                    registration_timeout=self.registration_timeout,
                )
            elif registration_created:
                try:
                    await asyncio.wait_for(
                        self.client.delete_llm_proxy_async(
                            registered_model_id,
                            client=client,
                            timeout=self.registration_timeout,
                        ),
                        timeout=self.registration_timeout,
                    )
                    logger.info(
                        "Deleted Arena LLM proxy registration: model_id=%s",
                        registered_model_id,
                    )
                    _record_arena_metrics(**{"arena/cleanup_success": 1.0})
                except Exception:
                    logger.exception(
                        "Failed to delete Arena LLM proxy registration: model_id=%s",
                        registered_model_id,
                    )
                    _record_arena_metrics(**{"arena/cleanup_success": 0.0})
            if owns_client:
                await client.aclose()
        logger.info(
            f"Finished Arena task: stream_id={stream_id}, data_id={data_id}, "
            f"reward={reward}"
        )
        self._task_result.set(task_result)
        return reward

    def _session_gateway_registry(
        self, worker_runtime: Any | None
    ) -> _WorkerSessionGatewayRegistry:
        """Return the registry owned by this rollout worker.

        ``RemoteInfEngine`` survives across prompt submissions even though each
        submission reconstructs this agent. Keeping the registry on that
        runtime makes registration cardinality equal the number of proxy
        workers, not the number of prompts or samples.
        """

        if worker_runtime is None:
            return self._local_session_gateway_registry

        registry = getattr(worker_runtime, _WORKER_GATEWAY_REGISTRY_ATTR, None)
        if registry is None:
            registry = _WorkerSessionGatewayRegistry(persistent=True)
            setattr(worker_runtime, _WORKER_GATEWAY_REGISTRY_ATTR, registry)
        if not isinstance(registry, _WorkerSessionGatewayRegistry):
            raise TypeError(
                f"{_WORKER_GATEWAY_REGISTRY_ATTR} is not an Arena gateway registry"
            )

        register_cleanup = getattr(worker_runtime, "register_destroy_callback", None)
        if not callable(register_cleanup):
            raise TypeError(
                "worker_runtime must provide register_destroy_callback for "
                "persistent Arena session gateway cleanup"
            )
        register_cleanup(_WORKER_GATEWAY_CLEANUP_KEY, registry.close)
        return registry

    def _stream_config_for_data(self, data: dict[str, Any]) -> ArenaStreamConfig:
        stream_name = str(data.get("arena_stream_name") or self.default_stream_name)
        if not stream_name:
            raise ValueError("arena_stream_name is required for a multi-Stream run")
        try:
            stream_config = self.stream_configs[stream_name]
        except KeyError as exc:
            raise ValueError(f"Unknown Arena Stream name: {stream_name!r}") from exc
        row_stream_id = str(data.get("stream_id") or "")
        if (
            row_stream_id
            and stream_config.stream_id
            and row_stream_id != stream_config.stream_id
        ):
            raise ValueError(
                f"Arena row Stream mismatch for {stream_name!r}: "
                f"{row_stream_id!r} != {stream_config.stream_id!r}"
            )
        return stream_config

    @staticmethod
    def _validate_row_reward_ref(
        data: dict[str, Any], stream_config: ArenaStreamConfig
    ) -> None:
        expected = stream_config.expected_reward_ref
        row_key = str(data.get("reward_ref_key") or "")
        row_version = str(data.get("reward_ref_version") or "")
        if expected.key and (row_key, row_version) != (
            expected.key,
            expected.version,
        ):
            raise ValueError(
                f"Arena row reward_ref mismatch for {stream_config.name!r}: "
                f"{row_key}@{row_version} != {expected.key}@{expected.version}"
            )

    @staticmethod
    def _validate_row_llm_protocol(
        row_protocol: str, stream_config: ArenaStreamConfig
    ) -> None:
        if row_protocol not in ("anthropic", "responses", "chat_completions"):
            raise ValueError(f"Unsupported Arena row LLM protocol: {row_protocol!r}")
        if stream_config.llm_protocol and row_protocol != stream_config.llm_protocol:
            raise ValueError(
                f"Arena row LLM protocol mismatch for {stream_config.name!r}: "
                f"{row_protocol!r} != {stream_config.llm_protocol!r}"
            )

    async def _validate_live_reward_ref(
        self,
        stream_id: str,
        stream_config: ArenaStreamConfig,
        client: httpx.AsyncClient,
    ) -> None:
        """Fail before launch when a Stream's default grader has drifted."""

        expected = stream_config.expected_reward_ref
        if not expected.key:
            return
        try:
            stream = await asyncio.wait_for(
                self.client.resolve_stream_async(
                    stream_id,
                    client=client,
                    timeout=self.request_timeout,
                ),
                timeout=self.request_timeout,
            )
            actual = stream.get("default_reward_ref")
            if not isinstance(actual, dict):
                raise ArenaAPIError(
                    f"Arena Stream {stream_config.name!r} has no default_reward_ref"
                )
            actual_key = actual.get("key")
            actual_version = actual.get("version")
            if (actual_key, actual_version) != (expected.key, expected.version):
                raise ArenaAPIError(
                    f"Arena Stream {stream_config.name!r} reward_ref drifted before "
                    f"launch: expected {expected.key}@{expected.version}, got "
                    f"{actual_key}@{actual_version}"
                )
        except Exception:
            _record_arena_metrics(**{"arena/reward_ref_validation_success": 0.0})
            raise
        _record_arena_metrics(**{"arena/reward_ref_validation_success": 1.0})

    def _transform_reward(
        self,
        reward: float,
        data: dict[str, Any],
        stream_config: ArenaStreamConfig | None = None,
    ) -> float:
        """Apply the configured Arena reward mapping at the RL boundary."""
        if stream_config is None:
            stream_config = self._stream_config_for_data(data)
        reward_transform = self.reward_transforms[stream_config.name]
        if reward_transform is not None:
            transform_kwargs = (
                {"reward_threshold": stream_config.reward_threshold}
                if stream_config.reward_threshold is not None
                else {}
            )
            transformed = float(reward_transform(reward, data, **transform_kwargs))
            if not math.isfinite(transformed):
                raise ValueError(
                    f"Arena reward transform for {stream_config.name!r} returned "
                    f"a non-finite value: {transformed}"
                )
            return transformed
        if stream_config.reward_threshold is not None:
            return float(reward >= stream_config.reward_threshold)
        return reward

    async def _dump_task_result(
        self,
        data: dict[str, Any],
        stream_config: ArenaStreamConfig,
        result: ArenaTaskResult,
        *,
        training_score: float | None,
    ) -> None:
        """Optionally append one bounded result envelope to a private audit shard."""

        if not self.result_dump_dir:
            return
        expected_ref = stream_config.expected_reward_ref

        def encode_record() -> bytes:
            raw_json = json.dumps(result.raw, ensure_ascii=False, separators=(",", ":"))
            raw_size_bytes = len(raw_json.encode("utf-8"))
            raw_truncated = raw_size_bytes > self.result_dump_max_bytes
            record = {
                "arena_stream_name": stream_config.name,
                "stream_id": str(data.get("stream_id") or stream_config.stream_id),
                "data_id": str(data.get("data_id") or ""),
                "task_id": result.task_id,
                "status": result.status,
                "expected_reward_ref": {
                    "key": expected_ref.key,
                    "version": expected_ref.version,
                },
                "arena_score": result.score,
                "training_score": training_score,
                "raw": None if raw_truncated else result.raw,
                "raw_present": result.raw is not None,
                "raw_size_bytes": raw_size_bytes,
                "raw_truncated": raw_truncated,
                "artifacts_uri": result.artifacts_uri,
                "trace_id": result.trace_id,
                "computed_at": result.computed_at,
            }
            return (
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")

        try:
            encoded = await asyncio.to_thread(encode_record)
        except (TypeError, ValueError):
            logger.exception(
                "Failed to encode Arena result audit record: task_id=%s",
                result.task_id,
            )
            _record_arena_metrics(**{"arena/result_dump_success": 0.0})
            return
        dump_dir = Path(self.result_dump_dir)
        dump_path = (
            dump_dir / f"arena_results_{socket.gethostname()}_{os.getpid()}.jsonl"
        )

        def append_record() -> None:
            directory_fd = _open_directory_no_symlinks(dump_dir)
            try:
                directory_stat = os.fstat(directory_fd)
                if not stat.S_ISDIR(directory_stat.st_mode):
                    raise NotADirectoryError(str(dump_dir))
                if directory_stat.st_uid != os.geteuid():
                    raise PermissionError(
                        f"Arena result directory must be owned by uid "
                        f"{os.geteuid()}: {dump_dir}"
                    )
                os.fchmod(directory_fd, 0o700)
                file_descriptor = os.open(
                    dump_path.name,
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    file_stat = os.fstat(file_descriptor)
                    if not stat.S_ISREG(file_stat.st_mode):
                        raise OSError(
                            f"Arena result shard must be a regular file: {dump_path}"
                        )
                    if file_stat.st_uid != os.geteuid():
                        raise PermissionError(
                            f"Arena result shard must be owned by uid "
                            f"{os.geteuid()}: {dump_path}"
                        )
                    os.fchmod(file_descriptor, 0o600)
                    with os.fdopen(file_descriptor, "ab", closefd=False) as output:
                        output.write(encoded)
                finally:
                    os.close(file_descriptor)
            finally:
                os.close(directory_fd)

        try:
            async with self._result_dump_lock:
                await asyncio.to_thread(append_record)
        except Exception:
            logger.exception(
                "Failed to persist Arena result audit record: task_id=%s",
                result.task_id,
            )
            _record_arena_metrics(**{"arena/result_dump_success": 0.0})
        else:
            _record_arena_metrics(**{"arena/result_dump_success": 1.0})

    def _is_solved_reward(
        self, reward: float, stream_config: ArenaStreamConfig
    ) -> bool:
        if self.reward_transforms[stream_config.name] is None:
            # Threshold-only mapping has already converted the score to 0/1.
            return reward > 0.0
        if stream_config.reward_threshold is not None:
            return reward >= stream_config.reward_threshold
        return reward > 0.0

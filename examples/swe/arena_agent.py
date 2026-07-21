"""Arena Stream agent workflow using AReaL's OpenAI-compatible proxy."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx

from examples.swe.arena_client import ArenaOpenAPIClient, LLMProtocol

from areal.utils import logging

logger = logging.getLogger("ArenaStreamAgent")


class ArenaStreamAgentWorkflow:
    """Launch an Arena online task and use its returned reward for RL."""

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
        self.client = ArenaOpenAPIClient(
            base_url=str(self.econfig.get("arena_base_url", "")),
            timeout=self.timeout,
            poll_interval=float(self.econfig.get("arena_poll_interval", 5.0)),
            request_retries=int(self.econfig.get("arena_request_retries", 3)),
        )

    async def run(
        self,
        data: dict[str, Any],
        **extra_kwargs: Any,
    ) -> float:
        """Launch the row's task with the current rollout proxy session."""
        stream_id = str(data.get("stream_id") or self.econfig.get("stream_id") or "")
        data_id = str(data.get("data_id") or "")
        llm_protocol: LLMProtocol = data.get("llm_protocol", "chat_completions")
        proxy_base_url = extra_kwargs.get("base_url")
        proxy_api_key = extra_kwargs.get("api_key")
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

        suffix = uuid.uuid4().hex[:12]
        model_name = f"stream-areal-{suffix}"
        deployment_id = str(uuid.uuid4())
        registered_model_id = deployment_id
        owns_client = arena_http_client is None
        client = arena_http_client or httpx.AsyncClient(timeout=self.timeout)
        try:
            (
                registered_url,
                registered_model_id,
            ) = await self.client.register_llm_proxy_async(
                model_name=model_name,
                upstream_base_url=str(proxy_base_url),
                upstream_api_key=str(proxy_api_key),
                deployment_id=deployment_id,
                protocol=llm_protocol,
                client=client,
                timeout=self.registration_timeout,
            )
            logger.info(
                "Registered Arena LLM proxy: model_name=%s, model_id=%s, "
                "registered_url=%s, protocol=%s",
                model_name,
                registered_model_id,
                registered_url,
                llm_protocol,
            )
            logger.info(
                f"Launching Arena task: stream_id={stream_id}, data_id={data_id}"
            )
            reward = await asyncio.wait_for(
                self.client.launch_one_task(
                    stream_id=stream_id,
                    data_id=data_id,
                    model_name=registered_model_id,
                    proxy_base_url=registered_url,
                    proxy_api_key=self.client.llm_api_key,
                    client=client,
                ),
                timeout=self.timeout,
            )
        finally:
            try:
                await self.client.delete_llm_proxy_async(
                    registered_model_id,
                    client=client,
                    timeout=self.registration_timeout,
                )
                logger.info(
                    "Deleted Arena LLM proxy registration: model_id=%s",
                    registered_model_id,
                )
            finally:
                if owns_client:
                    await client.aclose()
        logger.info(
            f"Finished Arena task: stream_id={stream_id}, data_id={data_id}, "
            f"reward={reward}"
        )
        return reward

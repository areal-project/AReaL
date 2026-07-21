"""Client helpers for the Arena online Stream OpenAPI."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Mapping
from numbers import Real
from typing import Any, Literal
from urllib.parse import quote

import httpx

LLMProtocol = Literal["anthropic", "responses", "chat_completions"]


class ArenaAPIError(RuntimeError):
    """Raised when the Arena OpenAPI returns an invalid or failed response."""


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def resolve_arena_credentials(
    base_url: str = "",
    api_token: str = "",
) -> tuple[str, str]:
    """Resolve Arena connection settings without embedding credentials in configs."""
    resolved_base_url = base_url or os.getenv("ARENA_OPENAPI_BASE", "")
    resolved_api_token = api_token or os.getenv("ARENA_OPENAPI_TOKEN", "")
    if not resolved_base_url:
        raise ValueError(
            "Arena OpenAPI base URL is required; set econfig.arena_base_url or "
            "ARENA_OPENAPI_BASE"
        )
    if not resolved_api_token:
        raise ValueError("ARENA_OPENAPI_TOKEN is required")
    return resolved_base_url.rstrip("/"), resolved_api_token


def _response_json(response: httpx.Response) -> Any:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = response.text[:500]
        raise ArenaAPIError(
            f"Arena OpenAPI returned HTTP {response.status_code}: {body}"
        ) from exc
    try:
        return response.json()
    except ValueError as exc:
        raise ArenaAPIError("Arena OpenAPI returned a non-JSON response") from exc


def _extract_reward(value: Any) -> float | None:
    """Extract a numeric reward from common response envelope shapes."""
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, Mapping):
        return None

    for key in ("reward", "score"):
        reward = value.get(key)
        if isinstance(reward, Real) and not isinstance(reward, bool):
            return float(reward)

    for key in ("result", "output", "task", "data"):
        nested_reward = _extract_reward(value.get(key))
        if nested_reward is not None:
            return nested_reward
    return None


def infer_llm_protocol(stream: Mapping[str, Any]) -> LLMProtocol:
    """Select the native LLM protocol from a Stream's default Harness."""
    harness_ref = stream.get("default_harness_ref")
    harness_key = ""
    if isinstance(harness_ref, Mapping):
        key = harness_ref.get("key")
        if isinstance(key, str):
            harness_key = key.lower()

    if "claude" in harness_key:
        return "anthropic"
    if "codex" in harness_key:
        return "responses"
    return "chat_completions"


def _llm_registration_payload(
    model_name: str,
    upstream_base_url: str,
    upstream_api_key: str,
    deployment_id: str,
    protocol: LLMProtocol,
) -> dict[str, Any]:
    if protocol == "anthropic":
        provider = "anthropic"
        model = f"anthropic/{model_name}"
        mode = "chat"
    elif protocol == "responses":
        provider = "openai"
        model = f"openai/{model_name}"
        mode = "responses"
    elif protocol == "chat_completions":
        provider = "openai"
        model = f"openai/{model_name}"
        mode = "chat"
    else:
        raise ValueError(f"Unsupported Arena LLM protocol: {protocol!r}")

    return {
        "model_name": model_name,
        "litellm_params": {
            "model": model,
            "api_base": upstream_base_url.rstrip("/"),
            "api_key": upstream_api_key,
            "custom_llm_provider": provider,
        },
        "model_info": {
            "id": deployment_id,
            "mode": mode,
            "disable_background_health_check": True,
        },
    }


class ArenaOpenAPIClient:
    """Small sync/async client for Stream discovery, datasets, and task launch."""

    MAX_DATASET_LIMIT = 1000
    FAILED_TASK_STATUSES = {
        "CANCELLED",
        "COLLECT_FAILED",
        "EVAL_FAILED",
        "FAILED",
        "HARNESS_FAILED",
        "NO_OUTPUT",
        "SETUP_FAILED",
        "TIMEOUT",
    }

    def __init__(
        self,
        base_url: str = "",
        api_token: str = "",
        llm_api_key: str = "",
        timeout: float = 60.0,
        poll_interval: float = 5.0,
        request_retries: int = 3,
    ) -> None:
        self.base_url, self.api_token = resolve_arena_credentials(
            base_url=base_url,
            api_token=api_token,
        )
        self.llm_api_key = llm_api_key or os.getenv("ARENA_LLM_API_KEY", "")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.request_retries = request_retries

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

    @property
    def _llm_headers(self) -> dict[str, str]:
        if not self.llm_api_key:
            raise ValueError(
                "ARENA_LLM_API_KEY is required for /llm model registration"
            )
        return {"Authorization": f"Bearer {self.llm_api_key}"}

    def list_streams(
        self,
        status: str | None = "ACTIVE",
        *,
        client: httpx.Client | None = None,
    ) -> list[dict[str, Any]]:
        """Return online Streams, optionally filtered by status."""
        params = {"status": status} if status else None
        response = self._sync_request(
            "GET",
            f"{self.base_url}/openapi/v1/streams",
            client=client,
            params=params,
            headers=self._headers,
        )
        payload = _response_json(response)
        items = payload.get("items") if isinstance(payload, Mapping) else None
        if not isinstance(items, list):
            raise ArenaAPIError("Stream list response is missing an 'items' array")
        return [dict(item) for item in items if isinstance(item, Mapping)]

    def resolve_stream(
        self,
        stream_id: str = "",
        *,
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        """Resolve a Stream and retain metadata needed by the rollout Harness."""
        streams = self.list_streams(client=client)
        if not streams:
            raise ArenaAPIError("Arena OpenAPI returned no active Streams")

        if stream_id:
            for stream in streams:
                if stream.get("stream_id") == stream_id:
                    return stream
            raise ArenaAPIError(f"Active Stream {stream_id!r} was not found")
        return streams[0]

    def resolve_stream_id(
        self,
        stream_id: str = "",
        *,
        client: httpx.Client | None = None,
    ) -> str:
        """Use an explicit Stream id or fall back to the first active Stream."""
        stream = self.resolve_stream(stream_id, client=client)
        selected = stream.get("stream_id")
        if not isinstance(selected, str) or not selected:
            raise ArenaAPIError("The selected Stream is missing 'stream_id'")
        return selected

    def _select_dataset_page(
        self,
        stream_id: str,
        limit: int,
        offset: int = 0,
        *,
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        encoded_stream_id = quote(stream_id, safe="")
        url = f"{self.base_url}/openapi/v1/streams/{encoded_stream_id}/dataset"
        kwargs = {
            "params": {"limit": limit, "offset": offset},
            "headers": self._headers,
        }
        response = self._sync_request("POST", url, client=client, **kwargs)
        payload = _response_json(response)
        if not isinstance(payload, Mapping):
            raise ArenaAPIError("Dataset response must be a JSON object")
        return dict(payload)

    def get_all_dataset_rows(
        self,
        stream_id: str,
        llm_protocol: LLMProtocol = "chat_completions",
        *,
        client: httpx.Client | None = None,
    ) -> list[dict[str, str]]:
        """Load one Stream's complete dataset in one page after a size probe."""
        first_page = self._select_dataset_page(
            stream_id,
            limit=1,
            client=client,
        )
        total = first_page.get("total")
        if not isinstance(total, int) or isinstance(total, bool) or total < 0:
            raise ArenaAPIError("Dataset response has an invalid 'total'")
        if total == 0:
            raise ArenaAPIError(f"Stream {stream_id!r} contains no dataset rows")
        if total > self.MAX_DATASET_LIMIT:
            raise ArenaAPIError(
                f"Stream {stream_id!r} contains {total} rows, exceeding the "
                f"single-request limit {self.MAX_DATASET_LIMIT}; pagination is not "
                "implemented yet"
            )

        page = (
            first_page
            if total == 1
            else self._select_dataset_page(
                stream_id,
                limit=total,
                client=client,
            )
        )
        data_ids = page.get("data_ids")
        if not isinstance(data_ids, list) or not all(
            isinstance(data_id, str) and data_id for data_id in data_ids
        ):
            raise ArenaAPIError("Dataset response has an invalid 'data_ids' array")
        if len(data_ids) != total:
            raise ArenaAPIError(
                f"Dataset response returned {len(data_ids)} rows, expected {total}"
            )
        return [
            {
                "data_id": data_id,
                "stream_id": stream_id,
                "llm_protocol": llm_protocol,
            }
            for data_id in data_ids
        ]

    def register_llm_proxy(
        self,
        model_name: str,
        upstream_base_url: str,
        upstream_api_key: str,
        *,
        deployment_id: str | None = None,
        protocol: LLMProtocol = "chat_completions",
        client: httpx.Client | None = None,
    ) -> tuple[str, str]:
        """Register one AReaL proxy and return its external URL and model id."""
        if not model_name.startswith("stream-areal-"):
            raise ValueError("Arena model_name must start with 'stream-areal-'")
        if not upstream_base_url:
            raise ValueError("upstream_base_url is required")
        if not upstream_api_key:
            raise ValueError("upstream_api_key is required")

        resolved_deployment_id = deployment_id or str(uuid.uuid4())
        payload = _llm_registration_payload(
            model_name=model_name,
            upstream_base_url=upstream_base_url,
            upstream_api_key=upstream_api_key,
            deployment_id=resolved_deployment_id,
            protocol=protocol,
        )
        response = self._sync_request(
            "POST",
            f"{self.base_url}/llm/model/new",
            client=client,
            headers=self._llm_headers,
            json=payload,
        )
        registration = _response_json(response)
        registered_url, registered_model_id = self._registered_llm_target(
            registration,
            resolved_deployment_id,
        )
        return registered_url, registered_model_id

    def _registered_llm_target(
        self,
        registration: Any,
        deployment_id: str,
    ) -> tuple[str, str]:
        """Resolve the external API base and model id returned by registration."""
        if isinstance(registration, str) and registration:
            registered_url = registration
            registered_model_id = deployment_id
        elif isinstance(registration, Mapping):
            response_id = registration.get("model_id")
            if not isinstance(response_id, str) or not response_id:
                raise ArenaAPIError("LLM registration response is missing model_id")
            # The production API returns a deployment object rather than the
            # documented URL string. Its OpenAI-compatible endpoint is fixed.
            registered_url = f"{self.base_url}/llm"
            registered_model_id = response_id
        else:
            raise ArenaAPIError(
                "LLM registration response must be a URL or deployment object"
            )
        return registered_url.rstrip("/"), registered_model_id

    async def register_llm_proxy_async(
        self,
        model_name: str,
        upstream_base_url: str,
        upstream_api_key: str,
        deployment_id: str,
        *,
        protocol: LLMProtocol = "chat_completions",
        client: httpx.AsyncClient,
        timeout: float = 180.0,
    ) -> tuple[str, str]:
        """Asynchronously register one rollout proxy session."""
        if not model_name.startswith("stream-areal-"):
            raise ValueError("Arena model_name must start with 'stream-areal-'")
        if not upstream_base_url:
            raise ValueError("upstream_base_url is required")
        if not upstream_api_key:
            raise ValueError("upstream_api_key is required")
        payload = _llm_registration_payload(
            model_name=model_name,
            upstream_base_url=upstream_base_url,
            upstream_api_key=upstream_api_key,
            deployment_id=deployment_id,
            protocol=protocol,
        )
        response = await self._async_request(
            client,
            "POST",
            f"{self.base_url}/llm/model/new",
            headers=self._llm_headers,
            json=payload,
            timeout=timeout,
        )
        registration = _response_json(response)
        return self._registered_llm_target(registration, deployment_id)

    def delete_llm_proxy(
        self,
        deployment_id: str,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        """Delete one registered LLM deployment; missing deployments are clean."""
        response = self._sync_request(
            "POST",
            f"{self.base_url}/llm/model/delete",
            client=client,
            headers=self._llm_headers,
            json={"id": deployment_id},
        )
        if response.status_code == 404:
            return
        _response_json(response)

    async def delete_llm_proxy_async(
        self,
        deployment_id: str,
        *,
        client: httpx.AsyncClient,
        timeout: float = 180.0,
    ) -> None:
        """Asynchronously delete one registered LLM deployment."""
        response = await self._async_request(
            client,
            "POST",
            f"{self.base_url}/llm/model/delete",
            headers=self._llm_headers,
            json={"id": deployment_id},
            timeout=timeout,
        )
        if response.status_code == 404:
            return
        _response_json(response)

    def _sync_request(
        self,
        method: str,
        url: str,
        *,
        client: httpx.Client | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        for attempt in range(self.request_retries + 1):
            try:
                if client is not None:
                    response = client.request(method, url, **kwargs)
                else:
                    with httpx.Client(timeout=self.timeout) as owned_client:
                        response = owned_client.request(method, url, **kwargs)
            except httpx.RequestError as exc:
                if attempt == self.request_retries:
                    raise ArenaAPIError(
                        f"Arena OpenAPI request failed after {attempt + 1} attempts: "
                        f"{type(exc).__name__}"
                    ) from exc
                time.sleep(min(2**attempt, 10))
                continue
            if not _is_retryable_status(response.status_code):
                return response
            if attempt == self.request_retries:
                return response
            response.close()
            time.sleep(min(2**attempt, 10))
        raise AssertionError("unreachable")

    async def _async_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        for attempt in range(self.request_retries + 1):
            try:
                response = await client.request(method, url, **kwargs)
            except httpx.RequestError as exc:
                if attempt == self.request_retries:
                    raise ArenaAPIError(
                        f"Arena OpenAPI request failed after {attempt + 1} attempts: "
                        f"{type(exc).__name__}"
                    ) from exc
                await asyncio.sleep(min(2**attempt, 10))
                continue
            if not _is_retryable_status(response.status_code):
                return response
            if attempt == self.request_retries:
                return response
            await response.aclose()
            await asyncio.sleep(min(2**attempt, 10))
        raise AssertionError("unreachable")

    async def launch_one_task(
        self,
        stream_id: str,
        data_id: str,
        model_name: str,
        proxy_base_url: str,
        proxy_api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> float:
        """Launch one Arena task through the rollout proxy and return its reward."""
        encoded_stream_id = quote(stream_id, safe="")
        url = f"{self.base_url}/openapi/v1/streams/{encoded_stream_id}/launch_one_task"
        request = {
            "data_id": data_id,
            "model_name": model_name,
            "base_url": proxy_base_url,
            "api_key": proxy_api_key,
            "envs": {
                "MODEL_NAME": model_name,
                "BASE_URL": proxy_base_url,
                "API_KEY": proxy_api_key,
            },
        }
        if client is not None:
            return await self._launch_and_wait(client, url, request)
        async with httpx.AsyncClient(timeout=self.timeout) as owned_client:
            return await self._launch_and_wait(owned_client, url, request)

    async def _launch_and_wait(
        self,
        client: httpx.AsyncClient,
        url: str,
        request: dict[str, Any],
    ) -> float:
        response = await self._async_request(
            client,
            "POST",
            url,
            json=request,
            headers=self._headers,
        )
        payload = _response_json(response)
        immediate_reward = _extract_reward(payload)
        if immediate_reward is not None:
            return immediate_reward
        task_id = payload.get("task_id") if isinstance(payload, Mapping) else None
        if not isinstance(task_id, str) or not task_id:
            raise ArenaAPIError(
                "launch_one_task response contains neither a reward nor task_id"
            )
        return await self._poll_task_result(client, task_id)

    async def _poll_task_result(
        self,
        client: httpx.AsyncClient,
        task_id: str,
    ) -> float:
        encoded_task_id = quote(task_id, safe="")
        url = f"{self.base_url}/openapi/v1/tasks/{encoded_task_id}/result"
        while True:
            response = await self._async_request(
                client,
                "GET",
                url,
                headers=self._headers,
            )
            payload = _response_json(response)
            if not isinstance(payload, Mapping):
                raise ArenaAPIError("Task result response must be a JSON object")
            status = str(payload.get("status") or "").upper()
            if status in self.FAILED_TASK_STATUSES:
                raise ArenaAPIError(f"Arena task {task_id!r} failed with {status}")
            reward = _extract_reward(payload)
            if reward is not None and status in {"DONE", "OK"}:
                return reward
            await asyncio.sleep(self.poll_interval)

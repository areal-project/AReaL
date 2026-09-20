"""Client helpers for the Arena online Stream OpenAPI."""

from __future__ import annotations

import asyncio
import math
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal
from urllib.parse import quote

import httpx

LLMProtocol = Literal["anthropic", "responses", "chat_completions"]


class ArenaAPIError(RuntimeError):
    """Raised when the Arena OpenAPI returns an invalid or failed response."""


@dataclass(frozen=True)
class ArenaTaskResult:
    """Stable Arena task-result envelope.

    ``raw`` is deliberately opaque. Its schema is owned by the configured Arena
    reward implementation and may also be ``None`` for successful tasks.
    """

    task_id: str
    status: str
    score: float | None
    raw: Any = None
    artifacts_uri: str | None = None
    trace_id: str | None = None
    computed_at: str | None = None


def _arena_task_type_from_tags(tags: Any) -> str:
    """Extract a task type from tags, falling back when metadata is unusable."""
    if tags is None:
        return "unknown"
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        return "unknown"

    task_types = {
        tag.removeprefix("domain:").strip()
        for tag in tags
        if tag.startswith("domain:") and tag.removeprefix("domain:").strip()
    }
    if len(task_types) > 1:
        return "unknown"
    return next(iter(task_types), "unknown")


class ArenaTaskFailedError(ArenaAPIError):
    """Raised when an Arena task reaches a failed terminal status."""

    def __init__(
        self,
        task_id: str,
        status: str,
        payload: Mapping[str, Any] | None = None,
        result: ArenaTaskResult | None = None,
    ) -> None:
        self.task_id = task_id
        self.status = status
        self.payload = dict(payload or {})
        self.result = result
        super().__init__(f"Arena task {task_id!r} failed with {status}")


def _is_retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _is_retryable_response(response: httpx.Response) -> bool:
    if _is_retryable_status(response.status_code):
        return True
    if response.status_code != 403:
        return False
    return "spanner-http-ant-group-watch-all" in response.text[:1000]


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


def _stream_from_payload(payload: Any) -> dict[str, Any]:
    """Extract one Stream from direct-get response envelope variants."""

    if not isinstance(payload, Mapping):
        raise ArenaAPIError("Arena Stream response must be a JSON object")
    data = payload.get("data", payload)
    if not isinstance(data, Mapping):
        raise ArenaAPIError("Arena Stream response is missing the Stream object")
    stream = data.get("stream", data)
    if not isinstance(stream, Mapping):
        raise ArenaAPIError("Arena Stream response is missing the Stream object")
    return dict(stream)


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ArenaAPIError(f"Arena task result field {key!r} must be a string")
    return value


def _parse_task_result(
    payload: Any,
    *,
    fallback_task_id: str = "",
) -> ArenaTaskResult:
    """Parse only the stable top-level result envelope.

    In particular, no recursive reward discovery is performed: heterogeneous
    grader data under ``raw`` must never become an RL reward accidentally.
    """

    # Preserve the legacy launch API's explicit control-plane envelope support
    # without recursively searching opaque grader-owned ``raw`` data.
    result_payload = payload
    control_fields = {
        "task_id",
        "status",
        "score",
        "raw",
        "artifacts_uri",
        "trace_id",
        "computed_at",
    }
    for _ in range(4):
        if isinstance(result_payload, Real) and not isinstance(result_payload, bool):
            score = float(result_payload)
            if not math.isfinite(score):
                raise ArenaAPIError("Arena task result 'score' must be finite")
            return ArenaTaskResult(
                task_id=fallback_task_id or "launch_one_task",
                status="DONE",
                score=score,
            )
        if not isinstance(result_payload, Mapping):
            raise ArenaAPIError("Arena task result must be a JSON object or number")
        if control_fields.intersection(result_payload):
            break
        nested = next(
            (
                result_payload[key]
                for key in ("result", "output", "task", "data")
                if key in result_payload
            ),
            None,
        )
        if nested is None:
            break
        result_payload = nested
    if not isinstance(result_payload, Mapping):
        raise ArenaAPIError("Arena task result must be a JSON object or number")

    task_id = result_payload.get("task_id", fallback_task_id)
    if not isinstance(task_id, str) or not task_id:
        raise ArenaAPIError("Arena task result is missing a valid 'task_id'")

    status_value = result_payload.get("status")
    if status_value is None and "score" in result_payload:
        status = "DONE"
    elif status_value is None and isinstance(result_payload.get("task_id"), str):
        status = "PENDING"
    elif isinstance(status_value, str) and status_value.strip():
        status = status_value.strip().upper()
    else:
        raise ArenaAPIError("Arena task result is missing a valid 'status'")

    score_value = result_payload.get("score")
    if score_value is None:
        score = None
    elif isinstance(score_value, Real) and not isinstance(score_value, bool):
        score = float(score_value)
        if not math.isfinite(score):
            raise ArenaAPIError("Arena task result 'score' must be finite")
    else:
        raise ArenaAPIError("Arena task result 'score' must be numeric or null")

    return ArenaTaskResult(
        task_id=task_id,
        status=status,
        score=score,
        raw=result_payload.get("raw"),
        artifacts_uri=_optional_string(result_payload, "artifacts_uri"),
        trace_id=_optional_string(result_payload, "trace_id"),
        computed_at=_optional_string(result_payload, "computed_at"),
    )


def infer_llm_protocol_from_harness(harness_key: str) -> LLMProtocol:
    """Infer the Harness client protocol from a Harness key or reference."""

    harness_key = harness_key.lower()
    if "claude" in harness_key:
        return "anthropic"
    if "codex" in harness_key:
        return "responses"
    return "chat_completions"


def infer_llm_protocol(stream: Mapping[str, Any]) -> LLMProtocol:
    """Select the client protocol used by a Stream's default Harness."""
    harness_ref = stream.get("default_harness_ref")
    harness_key = ""
    if isinstance(harness_ref, Mapping):
        key = harness_ref.get("key")
        if isinstance(key, str):
            harness_key = key
    return infer_llm_protocol_from_harness(harness_key)


def resolve_llm_protocol(
    stream: Mapping[str, Any],
    override: str = "",
) -> LLMProtocol:
    """Return an explicit protocol override or infer one from the Harness."""
    if not override:
        return infer_llm_protocol(stream)
    if override not in ("anthropic", "responses", "chat_completions"):
        raise ValueError(f"Unsupported Arena LLM protocol override: {override!r}")
    return override


def _llm_registration_payload(
    model_name: str,
    upstream_base_url: str,
    upstream_api_key: str,
    deployment_id: str,
    protocol: LLMProtocol,
) -> dict[str, Any]:
    if protocol not in ("anthropic", "responses", "chat_completions"):
        raise ValueError(f"Unsupported Arena LLM protocol: {protocol!r}")

    # ``protocol`` describes the Arena Harness client. The AReaL rollout
    # proxy itself exposes an OpenAI-compatible Chat Completions endpoint, so
    # advertise only that native upstream capability. Arena converts Messages
    # or Responses requests to Chat before forwarding them to this endpoint.
    return {
        "model_name": model_name,
        "endpoints": [
            {
                "endpoint_id": deployment_id,
                "upstream_model": model_name,
                "base_url": upstream_base_url.rstrip("/"),
                "api_key": upstream_api_key,
                "inbound_protos": ["chat"],
                "enabled": True,
            }
        ],
        "enabled": True,
        "metadata": {"deployment_id": deployment_id},
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
    def llm_gateway_api_key(self) -> str:
        """Return the Arena gateway key required by launched Harness tasks."""
        if not self.llm_api_key:
            raise ValueError("ARENA_LLM_API_KEY is required for LLM gateway traffic")
        return self.llm_api_key

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

    async def list_streams_async(
        self,
        status: str | None = "ACTIVE",
        *,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> list[dict[str, Any]]:
        """Asynchronously return Streams, optionally filtered by status."""

        params = {"status": status} if status else None
        response = await self._async_request(
            client,
            "GET",
            f"{self.base_url}/openapi/v1/streams",
            params=params,
            headers=self._headers,
            timeout=timeout,
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
        if stream_id:
            encoded_stream_id = quote(stream_id, safe="")
            response = self._sync_request(
                "GET",
                f"{self.base_url}/openapi/v1/streams/{encoded_stream_id}",
                client=client,
                headers=self._headers,
            )
            stream = _stream_from_payload(_response_json(response))
            if stream.get("stream_id") != stream_id:
                raise ArenaAPIError(
                    f"Arena Stream get returned id {stream.get('stream_id')!r}, "
                    f"expected {stream_id!r}"
                )
            status = stream.get("status")
            if status is not None and status != "ACTIVE":
                raise ArenaAPIError(
                    f"Arena Stream {stream_id!r} is not ACTIVE; status={status!r}"
                )
            return stream

        streams = self.list_streams(client=client)
        if not streams:
            raise ArenaAPIError("Arena OpenAPI returned no active Streams")
        return streams[0]

    async def resolve_stream_async(
        self,
        stream_id: str = "",
        *,
        client: httpx.AsyncClient,
        timeout: float,
    ) -> dict[str, Any]:
        """Asynchronously resolve one active Stream."""
        if stream_id:
            encoded_stream_id = quote(stream_id, safe="")
            response = await self._async_request(
                client,
                "GET",
                f"{self.base_url}/openapi/v1/streams/{encoded_stream_id}",
                headers=self._headers,
                timeout=timeout,
            )
            stream = _stream_from_payload(_response_json(response))
            if stream.get("stream_id") != stream_id:
                raise ArenaAPIError(
                    f"Arena Stream get returned id {stream.get('stream_id')!r}, "
                    f"expected {stream_id!r}"
                )
            status = stream.get("status")
            if status is not None and status != "ACTIVE":
                raise ArenaAPIError(
                    f"Arena Stream {stream_id!r} is not ACTIVE; status={status!r}"
                )
            return stream

        streams = await self.list_streams_async(client=client, timeout=timeout)
        if not streams:
            raise ArenaAPIError("Arena OpenAPI returned no active Streams")
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
        """Load one Stream's complete dataset with stable, validated pagination."""
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

        if total == 1:
            page_specs = iter([(0, 1, first_page)])
        else:
            page_specs = (
                (
                    offset,
                    min(self.MAX_DATASET_LIMIT, total - offset),
                    None,
                )
                for offset in range(0, total, self.MAX_DATASET_LIMIT)
            )

        data_ids: list[str] = []
        item_by_data_id: dict[str, Mapping[str, Any]] = {}
        for expected_offset, expected_limit, page in page_specs:
            if page is None:
                page = self._select_dataset_page(
                    stream_id,
                    limit=expected_limit,
                    offset=expected_offset,
                    client=client,
                )
            page_total = page.get("total")
            if page_total != total:
                raise ArenaAPIError(
                    f"Stream {stream_id!r} dataset total changed during pagination: "
                    f"expected {total}, got {page_total!r}"
                )
            page_offset = page.get("offset")
            if page_offset != expected_offset:
                raise ArenaAPIError(
                    f"Stream {stream_id!r} dataset pagination returned offset "
                    f"{page_offset!r}, expected {expected_offset}"
                )
            page_limit = page.get("limit")
            if page_limit != expected_limit:
                raise ArenaAPIError(
                    f"Stream {stream_id!r} dataset pagination returned limit "
                    f"{page_limit!r}, expected {expected_limit}"
                )
            page_data_ids = page.get("data_ids")
            if not isinstance(page_data_ids, list) or not all(
                isinstance(data_id, str) and data_id for data_id in page_data_ids
            ):
                raise ArenaAPIError("Dataset response has an invalid 'data_ids' array")
            page_count = page.get("count")
            if page_count != len(page_data_ids):
                raise ArenaAPIError(
                    f"Stream {stream_id!r} dataset pagination returned count "
                    f"{page_count!r}, expected {len(page_data_ids)}"
                )
            duplicate_ids = set(data_ids).intersection(page_data_ids)
            if duplicate_ids or len(set(page_data_ids)) != len(page_data_ids):
                raise ArenaAPIError(
                    f"Stream {stream_id!r} dataset pagination returned duplicate ids"
                )
            data_ids.extend(page_data_ids)

            items = page.get("items")
            if items is None:
                continue
            if not isinstance(items, list) or len(items) != len(page_data_ids):
                raise ArenaAPIError("Dataset response has an invalid 'items' array")
            page_item_ids: set[str] = set()
            for item in items:
                if not isinstance(item, Mapping):
                    raise ArenaAPIError(
                        "Dataset response contains an invalid dataset item"
                    )
                item_data_id = item.get("data_id")
                if not isinstance(item_data_id, str) or not item_data_id:
                    raise ArenaAPIError("Dataset item is missing a valid 'data_id'")
                if item_data_id in item_by_data_id or item_data_id in page_item_ids:
                    raise ArenaAPIError(
                        f"Stream {stream_id!r} dataset items contain duplicate ids"
                    )
                item_by_data_id[item_data_id] = item
                page_item_ids.add(item_data_id)
            if page_item_ids != set(page_data_ids):
                raise ArenaAPIError(
                    "Dataset items do not match the returned 'data_ids'"
                )

        if len(data_ids) != total:
            raise ArenaAPIError(
                f"Dataset response returned {len(data_ids)} rows, expected {total}"
            )

        rows: list[dict[str, str]] = []
        for data_id in data_ids:
            item = item_by_data_id.get(data_id)
            rows.append(
                {
                    "data_id": data_id,
                    "stream_id": stream_id,
                    "llm_protocol": llm_protocol,
                    "arena_task_type": _arena_task_type_from_tags(
                        item.get("tags") if item is not None else None
                    ),
                }
            )
        return rows

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
        """Register an AReaL OpenAI proxy and return its URL and model alias."""
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
            f"{self.base_url}/openapi/v1/llm/models",
            client=client,
            headers=self._headers,
            json=payload,
        )
        registration = _response_json(response)
        registered_url, registered_model_id = self._registered_llm_target(
            registration,
            model_name,
        )
        return registered_url, registered_model_id

    def _registered_llm_target(
        self,
        registration: Any,
        model_name: str,
    ) -> tuple[str, str]:
        """Resolve the LLM gateway API base and model alias from registration."""
        if not isinstance(registration, Mapping):
            raise ArenaAPIError("LLM registration response must be a JSON object")
        registered_model_id = registration.get("model_name")
        if not isinstance(registered_model_id, str) or not registered_model_id:
            raise ArenaAPIError("LLM registration response is missing model_name")
        if registered_model_id != model_name:
            raise ArenaAPIError(
                "LLM registration response returned an unexpected model_name: "
                f"{registered_model_id!r}"
            )
        return f"{self.base_url}/api", registered_model_id

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
            f"{self.base_url}/openapi/v1/llm/models",
            headers=self._headers,
            json=payload,
            timeout=timeout,
        )
        registration = _response_json(response)
        return self._registered_llm_target(registration, model_name)

    async def llm_proxy_exists_async(
        self,
        model_name: str,
        *,
        client: httpx.AsyncClient,
        timeout: float = 180.0,
    ) -> bool:
        """Return whether one dynamic LLM route still exists in Arena."""
        if not model_name.startswith("stream-areal-"):
            raise ValueError("Arena model_name must start with 'stream-areal-'")
        response = await self._async_request(
            client,
            "GET",
            f"{self.base_url}/openapi/v1/llm/models/{quote(model_name, safe='')}",
            headers=self._headers,
            timeout=timeout,
        )
        if response.status_code == 404:
            return False
        payload = _response_json(response)
        if not isinstance(payload, Mapping):
            raise ArenaAPIError("LLM model response must be a JSON object")
        model = payload.get("data", payload)
        if not isinstance(model, Mapping):
            raise ArenaAPIError("LLM model response is missing the model object")
        returned_name = model.get("model_name")
        if returned_name != model_name:
            raise ArenaAPIError(
                "LLM model response returned an unexpected model_name: "
                f"{returned_name!r}"
            )
        return True

    def delete_llm_proxy(
        self,
        model_name: str,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        """Delete one registered LLM model; missing models are clean."""
        response = self._sync_request(
            "DELETE",
            f"{self.base_url}/openapi/v1/llm/models/{quote(model_name, safe='')}",
            client=client,
            headers=self._headers,
        )
        if response.status_code == 404:
            return
        if 200 <= response.status_code < 300:
            return
        _response_json(response)

    async def delete_llm_proxy_async(
        self,
        model_name: str,
        *,
        client: httpx.AsyncClient,
        timeout: float = 180.0,
    ) -> None:
        """Asynchronously delete one registered LLM model."""
        response = await self._async_request(
            client,
            "DELETE",
            f"{self.base_url}/openapi/v1/llm/models/{quote(model_name, safe='')}",
            headers=self._headers,
            timeout=timeout,
        )
        if response.status_code == 404:
            return
        if 200 <= response.status_code < 300:
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
        request_client = (
            client if client is not None else httpx.Client(timeout=self.timeout)
        )
        try:
            for attempt in range(self.request_retries + 1):
                try:
                    response = request_client.request(method, url, **kwargs)
                except httpx.RequestError as exc:
                    if attempt == self.request_retries:
                        raise ArenaAPIError(
                            "Arena OpenAPI request failed after "
                            f"{attempt + 1} attempts: {type(exc).__name__}"
                        ) from exc
                    time.sleep(min(2**attempt, 10))
                    continue
                if not _is_retryable_response(response):
                    return response
                if attempt == self.request_retries:
                    return response
                response.close()
                time.sleep(min(2**attempt, 10))
        finally:
            if client is None:
                request_client.close()
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
            if not _is_retryable_response(response):
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
        session_id: str = "",
        harness: str = "",
        task_envs: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        on_launch_success: Callable[[str | None], None] | None = None,
    ) -> float:
        """Launch one Arena task and return its terminal score.

        This compatibility wrapper retains the original scalar API. New callers
        that need ``raw`` and provenance metadata should use
        :meth:`launch_one_task_result`.
        """

        result = await self.launch_one_task_result(
            stream_id=stream_id,
            data_id=data_id,
            model_name=model_name,
            proxy_base_url=proxy_base_url,
            proxy_api_key=proxy_api_key,
            session_id=session_id,
            harness=harness,
            task_envs=task_envs,
            client=client,
            on_launch_success=on_launch_success,
        )
        if result.score is None:
            raise ArenaAPIError("Successful Arena task result is missing score")
        return result.score

    async def launch_one_task_result(
        self,
        stream_id: str,
        data_id: str,
        model_name: str,
        proxy_base_url: str,
        proxy_api_key: str,
        *,
        session_id: str = "",
        harness: str = "",
        task_envs: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
        on_launch_success: Callable[[str | None], None] | None = None,
    ) -> ArenaTaskResult:
        """Launch one Arena task and return its structured terminal result."""
        encoded_stream_id = quote(stream_id, safe="")
        url = f"{self.base_url}/openapi/v1/streams/{encoded_stream_id}/launch_one_task"
        envs = dict(task_envs or {})
        if not all(
            isinstance(key, str) and key and isinstance(value, str)
            for key, value in envs.items()
        ):
            raise ValueError("Arena task environment variables must be strings")
        envs.update(
            {
                "MODEL_NAME": model_name,
                "BASE_URL": proxy_base_url,
                "API_KEY": proxy_api_key,
            }
        )
        request = {
            "data_id": data_id,
            "model_name": model_name,
            "base_url": proxy_base_url,
            "api_key": proxy_api_key,
            "envs": envs,
        }
        if session_id:
            request["session_id"] = session_id
        if harness:
            request["harness"] = harness
        if client is not None:
            return await self._launch_and_wait(
                client,
                url,
                request,
                on_launch_success=on_launch_success,
            )
        async with httpx.AsyncClient(timeout=self.timeout) as owned_client:
            return await self._launch_and_wait(
                owned_client,
                url,
                request,
                on_launch_success=on_launch_success,
            )

    async def _launch_and_wait(
        self,
        client: httpx.AsyncClient,
        url: str,
        request: dict[str, Any],
        *,
        on_launch_success: Callable[[str | None], None] | None = None,
    ) -> ArenaTaskResult:
        response = await self._async_request(
            client,
            "POST",
            url,
            json=request,
            headers=self._headers,
        )
        payload = _response_json(response)
        result = _parse_task_result(payload, fallback_task_id="launch_one_task")
        status = result.status
        task_id = result.task_id
        has_task_id = task_id != "launch_one_task"
        if status in self.FAILED_TASK_STATUSES:
            raise ArenaTaskFailedError(
                result.task_id,
                status,
                payload if isinstance(payload, Mapping) else None,
                result,
            )
        if status and status not in {"DONE", "OK"}:
            if has_task_id:
                if on_launch_success is not None:
                    on_launch_success(task_id)
                return await self._poll_task_result(client, task_id)
            raise ArenaAPIError(
                f"launch_one_task response has status {status} but no task_id to poll"
            )

        if result.score is None:
            raise ArenaAPIError(
                "launch_one_task terminal response is missing top-level 'score'"
            )
        if on_launch_success is not None:
            on_launch_success(task_id if has_task_id else None)
        return result

    async def _poll_task_result(
        self,
        client: httpx.AsyncClient,
        task_id: str,
    ) -> ArenaTaskResult:
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
            result = _parse_task_result(payload, fallback_task_id=task_id)
            if result.task_id != task_id:
                raise ArenaAPIError(
                    "Arena task result returned an unexpected task_id: "
                    f"{result.task_id!r}"
                )
            if result.status in self.FAILED_TASK_STATUSES:
                raise ArenaTaskFailedError(
                    task_id,
                    result.status,
                    payload if isinstance(payload, Mapping) else None,
                    result,
                )
            if result.status in {"DONE", "OK"}:
                if result.score is None:
                    raise ArenaAPIError(
                        "Arena terminal task result is missing top-level 'score'"
                    )
                return result
            await asyncio.sleep(self.poll_interval)

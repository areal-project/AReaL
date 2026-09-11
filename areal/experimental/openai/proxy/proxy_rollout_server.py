# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import hmac
import inspect
import json
import os
import secrets
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import TYPE_CHECKING, Any

import uvicorn
from anthropic.types.message import Message
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.responses import Response
from openai.types.responses.response_create_params import ResponseCreateParams
from pydantic import BaseModel

from areal.api.cli_args import NameResolveConfig
from areal.experimental.openai.anthropic import (
    translate_anthropic_request,
    translate_anthropic_response,
    translate_anthropic_stream,
)
from areal.experimental.openai.client import ArealOpenAI, ContextLengthExceededError
from areal.infra.processor_cache import ProcessorCacheRegistry
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.infra.utils.http import validate_admin_api_key
from areal.utils import name_resolve, names, seeding
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.logging import getLogger
from areal.utils.network import find_free_ports, gethostip

from .server import (
    ANTHROPIC_MESSAGES_PATHNAME,
    CHAT_COMPLETIONS_PATHNAME,
    DEFAULT_ADMIN_API_KEY,
    EXPORT_TRAJECTORIES_PATHNAME,
    GRANT_CAPACITY_PATHNAME,
    OPENAI_CHAT_COMPLETIONS_PATHNAME,
    RESPONSES_PATHNAME,
    RL_END_PROCESSOR_CACHE_GROUP_PATHNAME,
    RL_END_SESSION_PATHNAME,
    RL_FETCH_SHARED_TENSORS_PATHNAME,
    RL_SET_REWARD_PATHNAME,
    RL_START_SESSION_PATHNAME,
    ExportTrajectoriesRequest,
    ExportTrajectoriesResponse,
    FetchSharedTensorsRequest,
    FetchSharedTensorsResponse,
    ProcessorCacheGroupRequest,
    SessionData,
    SetRewardRequest,
    StartSessionRequest,
    StartSessionResponse,
    derive_session_gateway_api_key,
    derive_session_gateway_token,
    serialize_interactions,
)
from .tensor_reference import GroupTensorStoreRegistry

if TYPE_CHECKING:
    from areal.api import InferenceEngine


logger = getLogger("ProxyRolloutServer")

_STREAM_HEARTBEAT_INTERVAL_SECONDS = 30.0


# =============================================================================
# Warning Deduplication
# =============================================================================


# Set AREAL_PROXY_WARN_ONCE=1 to avoid warning spamming
_warn_once_enabled = os.environ.get("AREAL_PROXY_WARN_ONCE", "0") == "1"
_warned_messages: set[str] = set()
_warn_lock = threading.Lock()


def _deterministic_sampling_seed(session_id: str, request_index: int) -> int:
    return seeding.derive_deterministic_seed(session_id, request_index)


def _warn_once(msg: str) -> None:
    """Log a warning message, optionally only once if AREAL_PROXY_WARN_ONCE=1."""
    if not _warn_once_enabled:
        logger.warning(msg)
        return

    with _warn_lock:
        if msg not in _warned_messages:
            _warned_messages.add(msg)
            logger.warning(msg)


# =============================================================================
# Module-Level Globals (like rpc_server.py)
# =============================================================================


# Engine and client (created via /create_engine and /call with method "initialize")
_engine: InferenceEngine | None = None
_openai_client: ArealOpenAI | None = None

# Session management
_session_cache: dict[str, SessionData] = {}
_lock = threading.Lock()
_capacity = 0
_last_cleanup_time: float = 0
_session_timeout_seconds: int = 3600  # Default timeout (overridden by config)
_processor_cache_registry = ProcessorCacheRegistry()
_group_tensor_store_registry = GroupTensorStoreRegistry()

# API key authentication
# Initialized to a random value so pre-configuration requests cannot bypass auth.
# Overwritten by _setup_openai_client() once the engine is configured.
_admin_api_key: str = secrets.token_urlsafe(32)
_api_key_to_session: dict[str, str] = {}
_session_to_api_key: dict[str, str] = {}  # Reverse mapping for O(1) cleanup

# Pluggable message preprocessors loaded from config at setup time.
# Applied in order after Anthropic-to-OpenAI translation, before content
# reaches the ArealOpenAI client.
_message_preprocessors: list = []

# Pluggable prefix matcher for InteractionCache parent-child matching.
_prefix_matcher = None

# Server address (set at startup)
_server_host: str = "0.0.0.0"
_server_port: int = 8000
_worker_role: str | None = None
_worker_index: int | None = None

# Port allocation tracking
_allocated_ports: set[int] = set()
_port_alloc_lock = asyncio.Lock()

# Deterministic sampling (set from InferenceEngineConfig at setup time).
_deterministic_sampling: bool = False

# Server config (needed for name_resolve registration)
_experiment_name: str | None = None
_trial_name: str | None = None
_name_resolve_type: str = "nfs"
_nfs_record_root: str = "/tmp/areal/name_resolve"
_etcd3_addr: str = "localhost:2379"


def _resolve_worker_index(cli_worker_index: int) -> int:
    """Resolve identity without clobbering an explicit scheduler value.

    A local launch can inherit ``SLURM_PROCID`` from its parent login shell.
    Treating that stale value as an unconditional override makes every local
    proxy identify as rank 0, so exact fork readiness checks reject ranks
    1..N. Slurm-only launchers still use the environment fallback when they
    leave ``--worker-index`` at its sentinel value.
    """
    worker_index = cli_worker_index
    if worker_index == -1 and "SLURM_PROCID" in os.environ:
        worker_index = int(os.environ["SLURM_PROCID"])
    if worker_index == -1:
        raise ValueError("Invalid worker index. Not found from SLURM environ or args.")
    return worker_index


# =============================================================================
# Request Validation
# =============================================================================


async def validate_json_request(raw_request: Request):
    """Validate that the request content-type is application/json."""
    content_type = raw_request.headers.get("content-type", "").lower()
    media_type = content_type.split(";", maxsplit=1)[0]
    if media_type != "application/json":
        raise RequestValidationError(
            errors=[
                {
                    "loc": ["header", "content-type"],
                    "msg": "Unsupported Media Type: Only 'application/json' is allowed",
                    "type": "value_error",
                }
            ]
        )


# =============================================================================
# API Key Authentication
# =============================================================================


def _extract_api_tokens(request: Request) -> tuple[str, ...]:
    """Extract all supported API credentials without assuming proxy precedence."""

    tokens: list[str] = []
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()
        if token:
            tokens.append(token)
    x_api_key = request.headers.get("x-api-key", "").strip()
    if x_api_key and x_api_key not in tokens:
        tokens.append(x_api_key)
    if not tokens:
        raise HTTPException(
            status_code=401,
            detail=(
                "Missing or malformed Authorization header. Expected "
                "'Bearer <token>' or 'x-api-key: <token>'."
            ),
        )
    return tuple(tokens)


def _extract_bearer_token(request: Request) -> str:
    """Extract API token from Authorization header or x-api-key header.

    Supports both 'Authorization: Bearer <token>' (OpenAI SDK, case-insensitive
    per RFC 6750) and 'x-api-key: <token>' (Anthropic SDK) for cross-SDK
    compatibility.
    """
    return _extract_api_tokens(request)[0]


def _require_admin_key(request: Request) -> str:
    """Validate that the request carries the admin API key.

    Uses constant-time comparison to prevent timing attacks.
    """
    token = _extract_bearer_token(request)
    if not hmac.compare_digest(token, _admin_api_key):
        raise HTTPException(status_code=403, detail="Invalid admin API key.")
    return token


def _require_session_key(request: Request) -> str:
    """Resolve an active session from a session key or trusted gateway header.

    Normal clients authenticate with their least-privilege per-session API key.
    A trusted upstream gateway may instead authenticate with a generation-only
    derived key and present a session-bound capability. This lets a public
    gateway keep one stable upstream registration per proxy worker without
    exposing the admin credential or allowing one Harness to select another
    concurrent RL session.
    """
    tokens = _extract_api_tokens(request)
    gateway_key = derive_session_gateway_api_key(_admin_api_key)
    with _lock:
        if any(hmac.compare_digest(token, gateway_key) for token in tokens):
            if request.url.path not in {
                f"/{CHAT_COMPLETIONS_PATHNAME}",
                f"/{OPENAI_CHAT_COMPLETIONS_PATHNAME}",
                f"/{RESPONSES_PATHNAME}",
                f"/{ANTHROPIC_MESSAGES_PATHNAME}",
            }:
                raise HTTPException(
                    status_code=403, detail="Gateway keys are generation-only."
                )
            session_id = request.headers.get("x-session-id", "").strip()
            session_token = request.headers.get("x-session-token", "").strip()
            expected_token = derive_session_gateway_token(_admin_api_key, session_id)
            if (
                session_id
                and session_id in _session_to_api_key
                and session_token
                and hmac.compare_digest(session_token, expected_token)
            ):
                return session_id
            raise HTTPException(
                status_code=401,
                detail="Missing, invalid, or expired session routing capability.",
            )
        session_id = next(
            (
                resolved
                for token in tokens
                if (resolved := _api_key_to_session.get(token)) is not None
            ),
            None,
        )
    if session_id is None:
        raise HTTPException(
            status_code=401, detail="Invalid or expired session API key."
        )
    return session_id


def _remove_api_keys_for_session(session_id: str) -> None:
    """Remove the API key mapping for the given session.

    Must be called with _lock held.
    """
    api_key = _session_to_api_key.pop(session_id, None)
    if api_key:
        _api_key_to_session.pop(api_key, None)


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "initialized": _engine is not None,
        "role": _worker_role,
        "worker_index": _worker_index,
    }


@app.post("/alloc_ports")
async def alloc_ports(raw_request: Request):
    """Allocate multiple free ports.

    Expected JSON payload:
    {
        "count": 5  # Number of ports to allocate
    }

    This endpoint is required for compatibility with SlurmScheduler's
    fork_workers mechanism, which may request additional ports after
    forking a new worker process.
    """
    global _allocated_ports

    try:
        data = await raw_request.json()
        count = data.get("count")
        if count is None:
            raise HTTPException(
                status_code=400, detail="Missing 'count' field in request"
            )

        if not isinstance(count, int) or count <= 0:
            raise HTTPException(
                status_code=400, detail="'count' must be a positive integer"
            )

        async with _port_alloc_lock:
            ports = find_free_ports(count, exclude_ports=_allocated_ports)
        _allocated_ports.update(ports)

        return {"status": "success", "ports": ports, "host": _server_host}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in alloc_ports: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def _setup_openai_client():
    global _openai_client, _session_timeout_seconds, _admin_api_key
    global _message_preprocessors, _prefix_matcher, _deterministic_sampling
    config = _engine.config
    _deterministic_sampling = bool(getattr(config, "deterministic_sampling", False))
    processor, tokenizer = load_hf_processor_and_tokenizer(config.tokenizer_path)
    if processor is not None and not hasattr(processor, "image_processor"):
        processor = None
    agent_cfg = config.agent
    _openai_client = ArealOpenAI(
        engine=_engine,
        tokenizer=tokenizer,
        processor=processor,
        tool_call_parser=agent_cfg.tool_call_parser,
        reasoning_parser=agent_cfg.reasoning_parser,
        engine_max_tokens=agent_cfg.engine_max_tokens,
        chat_template_type=agent_cfg.chat_template_type,
        lora_name=config.lora_name,
        require_multimodal_processor=True,
    )
    # Set session timeout from config
    _session_timeout_seconds = agent_cfg.session_timeout_seconds
    # Validate admin API key BEFORE assigning it to the global, so a
    # failed validation cannot leave the default key live on the server.
    # The default admin key is publicly known; refuse to use it when the
    # server is reachable from outside the local host (otherwise anyone
    # who can reach this port can call admin endpoints such as
    # grant_capacity, start_session, export_trajectories, ...).
    validate_admin_api_key(
        _server_host,
        agent_cfg.admin_api_key,
        default_key=DEFAULT_ADMIN_API_KEY,
        config_field="AgentConfig.admin_api_key",
    )
    # Only commit the key to the global after validation has passed.
    with _lock:
        _admin_api_key = agent_cfg.admin_api_key

    _message_preprocessors = []
    for path in agent_cfg.message_preprocessors:
        cls = import_from_string(path)
        _message_preprocessors.append(cls())
        logger.info("Loaded message preprocessor: %s", path)

    if agent_cfg.prefix_matcher:
        _prefix_matcher = import_from_string(agent_cfg.prefix_matcher)
        logger.info("Loaded prefix matcher: %s", agent_cfg.prefix_matcher)
    else:
        _prefix_matcher = None


@app.post("/configure")
async def configure(raw_request: Request):
    data = await raw_request.json()
    config = deserialize_value(data.get("config"))
    rank = data.get("rank", 0)
    seeding.set_random_seed(config.seed, key=f"proxy{rank}")
    return {"status": "success"}


@app.post("/set_env")
async def set_env(raw_request: Request):
    data = await raw_request.json()
    for key, value in data.get("env", {}).items():
        os.environ[key] = str(value)
    return {"status": "success"}


@app.post("/create_engine")
async def create_engine(raw_request: Request):
    global _engine
    if _engine is not None:
        raise HTTPException(status_code=400, detail="Engine already exists")

    data = await raw_request.json()
    engine_class = import_from_string(data.get("engine"))
    init_kwargs = deserialize_value(data.get("init_kwargs", {}))
    _engine = engine_class(**init_kwargs)
    return {"status": "success"}


@app.post("/call")
async def call_engine_method(raw_request: Request):
    global _engine, _openai_client
    if _engine is None:
        raise HTTPException(status_code=400, detail="Engine not initialized")

    data = await raw_request.json()
    method_name = data.get("method")
    args = deserialize_value(data.get("args", []))
    kwargs = deserialize_value(data.get("kwargs", {}))

    method = getattr(_engine, method_name)
    result = method(*args, **kwargs)

    if method_name == "initialize":
        _setup_openai_client()

    return {"status": "success", "result": serialize_value(result)}


# =============================================================================
# Capacity Management
# =============================================================================


@app.post(f"/{GRANT_CAPACITY_PATHNAME}", dependencies=[Depends(_require_admin_key)])
def grant_capacity():
    """Grant capacity for a new session."""
    global _capacity
    with _lock:
        _capacity += 1
        return {"capacity": _capacity}


# =============================================================================
# Session Management
# =============================================================================


def _cleanup_stale_sessions():
    """Remove stale sessions and orphaned API key mappings.

    Called periodically (at most once per minute) during start_session to clean up
    sessions that were started but never finished.

    Also sweeps orphaned API key mappings whose session_id is no longer in
    _session_cache (e.g. if a client crashes after end_session but before
    export_trajectories).

    Must be called with _lock held.
    """
    global _last_cleanup_time

    # Only run cleanup at most once per minute
    current_time = time.time()
    if current_time - _last_cleanup_time < 60:
        return

    _last_cleanup_time = current_time

    stale_sessions = []
    for session_id, session_data in _session_cache.items():
        if session_data.is_stale(timeout_seconds=_session_timeout_seconds):
            stale_sessions.append(session_id)

    for session_id in stale_sessions:
        logger.warning(f"Removing stale session: {session_id}")
        session_data = _session_cache.pop(session_id, None)
        if session_data is not None:
            cache_group_id = session_data.take_processor_cache_group_id()
            if cache_group_id is not None:
                _processor_cache_registry.release(cache_group_id)

    _processor_cache_registry.discard_stale(_session_timeout_seconds)
    _group_tensor_store_registry.discard_stale(_session_timeout_seconds)

    # Clean up API key mappings for stale sessions
    if stale_sessions:
        for sid in stale_sessions:
            _remove_api_keys_for_session(sid)
        logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")

    # Sweep orphaned API key mappings whose session_id is no longer in cache.
    # Handles edge case where client crashes after end_session but before
    # export_trajectories, leaving key mappings with no cleanup path.
    orphaned_sids = [sid for sid in _session_to_api_key if sid not in _session_cache]
    for sid in orphaned_sids:
        _remove_api_keys_for_session(sid)
    if orphaned_sids:
        logger.info(f"Cleaned up {len(orphaned_sids)} orphaned API key mappings")


@app.post(f"/{RL_START_SESSION_PATHNAME}", dependencies=[Depends(_require_admin_key)])
def start_session(request: StartSessionRequest) -> StartSessionResponse:
    """Start a new RL session and allocate a unique session API key.

    If ``request.api_key`` is provided, reuse that key instead of generating
    a new one.  When the key already maps to a *finished* session on this
    worker, the stale mapping is cleaned up first.  If it maps to an
    *active* (unfinished) session, the request is rejected with HTTP 409.
    """
    global _capacity
    task_id = request.task_id

    if (
        request.processor_cache_group_id is not None
        and request.processor_cache_group_size < 2
    ):
        raise HTTPException(
            status_code=400,
            detail="processor_cache_group_size must be at least 2 for grouped caching",
        )

    with _lock:
        # Periodically cleanup stale sessions
        _cleanup_stale_sessions()

        if _capacity <= 0:
            raise HTTPException(
                status_code=429,
                detail="No available capacity to start a new session",
            )

        # Generate unique session ID
        idx = 0
        while (session_id := f"{task_id}-{idx}") in _session_cache:
            idx += 1

        # Resolve session API key
        if request.api_key:
            # Gateway requested a specific key (refresh / reuse).
            session_api_key = request.api_key
            if session_api_key == _admin_api_key:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot use the admin API key as a session key.",
                )
            existing_sid = _api_key_to_session.get(session_api_key)
            if existing_sid is not None:
                existing_session = _session_cache.get(existing_sid)
                if existing_session is not None and not existing_session.is_completed:
                    raise HTTPException(
                        status_code=409,
                        detail=f"API key is already bound to active session {existing_sid}.",
                    )
                # Finished or orphaned — clean up the stale mapping.
                _remove_api_keys_for_session(existing_sid)
        else:
            # Generate unique session API key (current behaviour).
            session_api_key = secrets.token_urlsafe(32)
            while (
                session_api_key in _api_key_to_session
                or session_api_key == _admin_api_key
            ):
                session_api_key = secrets.token_urlsafe(32)

        processor_cache = None
        if request.processor_cache_group_id is not None:
            processor_cache = _processor_cache_registry.acquire(
                request.processor_cache_group_id,
                request.processor_cache_group_size,
            )

        _capacity -= 1
        _session_cache[session_id] = SessionData(
            session_id=session_id,
            prefix_matcher=_prefix_matcher,
            sampling_seed_identity=task_id,
            processor_cache=processor_cache,
            processor_cache_group_id=request.processor_cache_group_id,
        )
        _api_key_to_session[session_api_key] = session_id
        _session_to_api_key[session_id] = session_api_key

    return StartSessionResponse(session_id=session_id, api_key=session_api_key)


@app.post(f"/{RL_END_SESSION_PATHNAME}")
def end_session(session_id: str = Depends(_require_session_key)):
    """End an RL session.

    Returns the number of recorded interactions so callers (e.g. the proxy
    gateway refresh path) can decide whether meaningful trajectory data
    exists.
    """
    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=410, detail="Session already ended or expired"
            )
        session = _session_cache[session_id]
        interaction_count = len(session.completions)

    # finish() outside lock to avoid holding lock during potential I/O
    session.finish()
    cache_group_id = session.take_processor_cache_group_id()
    if cache_group_id is not None:
        _processor_cache_registry.release(cache_group_id)
    return {
        "message": "success",
        "interaction_count": interaction_count,
        "context_overflow": session.context_overflow,
        "context_overflow_message": session.context_overflow_message,
        "system_error": session.system_error,
        "system_error_message": session.system_error_message,
    }


@app.post(
    f"/{RL_END_PROCESSOR_CACHE_GROUP_PATHNAME}",
    dependencies=[Depends(_require_admin_key)],
)
def end_processor_cache_group(request: ProcessorCacheGroupRequest):
    """Discard processor and shared-tensor state after a rollout group finishes."""
    _processor_cache_registry.discard(request.group_id)
    _group_tensor_store_registry.discard(request.group_id)
    return {"message": "success"}


@app.post(
    f"/{RL_FETCH_SHARED_TENSORS_PATHNAME}",
    dependencies=[Depends(_require_admin_key)],
)
def fetch_shared_tensors(
    request: FetchSharedTensorsRequest,
) -> FetchSharedTensorsResponse:
    """Fetch each unique multimodal tensor once for a grouped rollout."""
    try:
        tensors = _group_tensor_store_registry.fetch(request.group_id, request.ref_ids)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FetchSharedTensorsResponse(tensors=serialize_value(tensors))


@app.post(f"/{RL_SET_REWARD_PATHNAME}")
def set_reward(
    request: SetRewardRequest, session_id: str = Depends(_require_session_key)
):
    """Set reward for an interaction in a session."""
    interaction_id = request.interaction_id
    reward = request.reward

    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=410, detail=f"Session {session_id} already ended or expired"
            )
        session_data = _session_cache[session_id]

    session_data.update_last_access()

    completions = session_data.completions
    if interaction_id is None:
        # Take the last interaction id
        if len(completions) == 0:
            logger.error(f"No interactions in session {session_id}")
            raise HTTPException(status_code=400, detail="No interactions in session")
        interaction_id = completions.last_interaction_id
    elif interaction_id not in completions:
        requested_interaction_id = interaction_id
        interaction_id = session_data.stream_completion_aliases.get(
            interaction_id, interaction_id
        )
        if interaction_id not in completions:
            logger.error(
                f"Interaction {requested_interaction_id} not found in session "
                f"{session_id}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Interaction {requested_interaction_id} not found",
            )
    session_data.completions.set_reward(interaction_id, reward)
    return {"message": "success"}


# =============================================================================
# OpenAI-Compatible Endpoints
# =============================================================================


def _contains_image_content(value: Any) -> bool:
    """Return whether a nested content value contains an image block."""
    if isinstance(value, (list, tuple)):
        return any(_contains_image_content(item) for item in value)
    if not isinstance(value, dict):
        return False
    if value.get("type") in {"image", "image_url", "input_image"}:
        return True
    return any(_contains_image_content(item) for item in value.values())


def _content_block_text(value: Any) -> str:
    """Extract text from nested Claude/OpenAI content blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(filter(None, (_content_block_text(item) for item in value)))
    if not isinstance(value, dict):
        return str(value)

    text = value.get("text")
    if isinstance(text, str):
        return text
    content = value.get("content")
    if isinstance(content, (str, list, tuple, dict)):
        return _content_block_text(content)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _flatten_text_content_lists(messages: list[dict]) -> None:
    """Flatten text-only content block lists to strings in-place.

    LiteLLM can forward Claude requests through the OpenAI chat-completions
    endpoint while retaining Anthropic-style text block lists. Preserve
    multimodal or otherwise structured content for the AReaL client.
    """
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        if msg.get("role") == "system" or not _contains_image_content(content):
            msg["content"] = _content_block_text(content)


def _preprocess_messages(messages: list[dict]) -> list[dict]:
    """Normalize messages shared by OpenAI and Anthropic proxy endpoints."""
    _flatten_text_content_lists(messages)
    for preprocessor in _message_preprocessors:
        messages = preprocessor(messages)
    return messages


def _materialize_iterables(value: Any) -> Any:
    """Recursively convert Pydantic validator iterators to plain containers."""
    if isinstance(value, BaseModel):
        return _materialize_iterables(value.model_dump())
    if isinstance(value, Mapping):
        return {key: _materialize_iterables(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, Iterable):
        return [_materialize_iterables(item) for item in value]
    return value


def _prepare_request_messages(messages: Any) -> Any:
    """Convert request message iterables to mutable dicts and preprocess them."""
    if isinstance(messages, (str, bytes, Mapping)) or not isinstance(
        messages, Iterable
    ):
        return messages

    message_list = list(messages)
    normalized: list[dict] = []
    for message in message_list:
        if isinstance(message, BaseModel):
            message = message.model_dump()
        elif isinstance(message, Mapping):
            message = dict(message)
        else:
            return message_list
        normalized.append(_materialize_iterables(message))
    return _preprocess_messages(normalized)


async def _call_client_create(
    create_fn,
    request: dict[str, Any] | BaseModel,
    session_id: str,
    extra_ignored_args: list[str] | None = None,
    stream: bool = False,
) -> ChatCompletion | Response | AsyncGenerator[ChatCompletionChunk, None]:
    """Common logic for chat completions and responses."""
    if _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail='Proxy server not initialized. Send requests to /create_engine then /call "initialize" first.',
        )

    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=410, detail=f"Session {session_id} already ended or expired"
            )
        session_data = _session_cache[session_id]
        session_data.update_last_access()

    request_index = (
        session_data.next_sampling_request_index() if _deterministic_sampling else None
    )

    sig = inspect.signature(create_fn)
    supports_processor_cache = "processor_cache" in sig.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in sig.parameters.values()
    )
    # Keep the request model when the AReaL client supports it. Anthropic's
    # LiteLLM response adapter uses this field to select the model family;
    # dropping it leaves the generated ChatCompletion with no usable model
    # identity and fails with ``Model type must be specified``.
    areal_client_ignored_args = extra_ignored_args or []
    areal_client_disallowed_args = ["areal_cache", "processor_cache"]
    areal_client_allowed_args = list(
        k
        for k in sig.parameters.keys()
        if k not in areal_client_ignored_args and k not in areal_client_disallowed_args
    )

    kwargs = request.model_dump() if isinstance(request, BaseModel) else dict(request)
    messages = kwargs.get("messages")
    if messages is not None:
        try:
            prepared_messages = _prepare_request_messages(messages)
        except Exception as e:
            message = f"{type(e).__name__}: {e}"
            session_data.mark_system_error(message)
            logger.exception("Failed to normalize proxy request messages")
            raise HTTPException(status_code=500, detail=message) from e
        kwargs["messages"] = prepared_messages

    dropped_args = []
    for k, v in kwargs.items():
        if k not in areal_client_allowed_args:
            dropped_args.append((k, v))

    for k, _ in dropped_args:
        del kwargs[k]

    def _is_default_value(k: str, v: Any) -> bool:
        if isinstance(request, BaseModel):
            return v == type(request).model_fields[k].default
        return False

    dropped_non_default_args = [
        (k, v)
        for k, v in dropped_args
        if k not in areal_client_ignored_args and not _is_default_value(k, v)
    ]
    if len(dropped_non_default_args):
        dropped_args_str = "\n".join(
            [f"  {k}: {v}" for k, v in dropped_non_default_args]
        )
        _warn_once(
            f"dropped unsupported non-default arguments for areal client:\n"
            f"{dropped_args_str}"
        )

    if "temperature" not in kwargs:
        kwargs["temperature"] = 1.0
        _warn_once("temperature not set in request, defaulting to 1.0")
    if "top_p" not in kwargs:
        kwargs["top_p"] = 1.0
        _warn_once("top_p not set in request, defaulting to 1.0")

    if (
        _deterministic_sampling
        and kwargs.get("seed") is None
        and "seed" in areal_client_allowed_args
    ):
        assert request_index is not None
        # The logical identity excludes the physical session collision suffix.
        # Reserve request indices at ingress so concurrent requests remain
        # distinct without holding a lock during inference.
        # TODO(agent): Strict mapping of concurrent sibling requests to seeds
        # requires a stable caller-provided request identity. Group samples use
        # separate sessions, so their sample_idx-based identities are stable.
        kwargs["seed"] = _deterministic_sampling_seed(
            session_data.sampling_seed_identity, request_index
        )

    # Strip stream from request body to prevent it from bypassing the explicit
    # `stream` parameter.  Without this, a request with {"stream": true} would
    # leak through kwargs and cause the client to return an AsyncGenerator even
    # when the caller did not ask for streaming.
    kwargs.pop("stream", None)
    if stream:
        kwargs["stream"] = True

    try:
        client_kwargs: dict[str, Any] = {
            "areal_cache": session_data.completions,
            **kwargs,
        }
        if supports_processor_cache:
            client_kwargs["processor_cache"] = session_data.processor_cache
        return await create_fn(**client_kwargs)
    except ContextLengthExceededError as e:
        session_data.mark_context_overflow(str(e))
        logger.warning("Session %s exceeded its context window: %s", session_id, e)
        raise HTTPException(
            status_code=400,
            detail={"type": "context_length_exceeded", "message": str(e)},
        )
    except ValueError as e:
        session_data.mark_system_error(f"{type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        session_data.mark_system_error(f"{type(e).__name__}: {e}")
        logger.exception("AReaL client request failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post(
    f"/{CHAT_COMPLETIONS_PATHNAME}",
    dependencies=[Depends(validate_json_request)],
    response_model=None,
)
@app.post(
    f"/{OPENAI_CHAT_COMPLETIONS_PATHNAME}",
    dependencies=[Depends(validate_json_request)],
    response_model=None,
)
async def chat_completions(
    request: dict[str, Any], session_id: str = Depends(_require_session_key)
) -> ChatCompletion | StreamingResponse:
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming (stream=True) and non-streaming requests.
    The wire body stays an untyped mapping here because OpenAI-compatible
    gateways may place multimodal content in roles that the SDK TypedDicts
    validate as text-only. The AReaL client performs the actual normalization
    after session routing.
    For streaming requests, returns a StreamingResponse with Server-Sent Events
    in the OpenAI streaming format (data: {json}\\n\\n ... data: [DONE]\\n\\n).
    """
    if _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail='Proxy server not initialized. Send requests to /create_engine then /call "initialize" first.',
        )

    # CompletionCreateParams is a TypedDict (dict subclass), so use dict access.
    is_streaming = request.get("stream") is True

    if is_streaming:
        safe_stream = _safe_stream_wrapper(
            _deferred_openai_sse_stream(request, session_id)
        )
        return StreamingResponse(
            safe_stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return await _call_client_create(
        create_fn=_openai_client.chat.completions.create,
        request=request,
        session_id=session_id,
    )


async def _deferred_openai_sse_stream(
    request: CompletionCreateParams,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """Start an SSE response before the non-streaming engine finishes.

    ``ArealOpenAI`` currently waits for ``agenerate()`` to finish before it
    returns its simulated chunk stream.  Emit a protocol-valid empty role
    chunk first so gateways and OpenAI-compatible clients can establish the
    stream, then send SSE comments while waiting.  Comments are ignored by
    compliant clients and do not become assistant content.
    """
    stream_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    stream_model = str(request.get("model") or "")
    initial_chunk = {
        "id": stream_id,
        "choices": [
            {
                "delta": {"role": "assistant", "content": ""},
                "index": 0,
                "finish_reason": None,
            }
        ],
        "created": int(time.time()),
        "model": stream_model,
        "object": "chat.completion.chunk",
    }
    yield f"data: {json.dumps(initial_chunk, separators=(',', ':'))}\n\n"

    create_task = asyncio.create_task(
        _call_client_create(
            create_fn=_openai_client.chat.completions.create,
            request=request,
            session_id=session_id,
            stream=True,
        )
    )
    openai_stream: AsyncGenerator[ChatCompletionChunk, None] | None = None
    real_completion_id: str | None = None
    try:
        while not create_task.done():
            try:
                openai_stream = await asyncio.wait_for(
                    asyncio.shield(create_task),
                    timeout=_STREAM_HEARTBEAT_INTERVAL_SECONDS,
                )
            except TimeoutError:
                yield ": ping\n\n"

        if openai_stream is None:
            openai_stream = await create_task

        async for chunk in openai_stream:
            if real_completion_id is None and chunk.id:
                real_completion_id = chunk.id
                if real_completion_id != stream_id:
                    with _lock:
                        session = _session_cache.get(session_id)
                        if session is not None:
                            session.stream_completion_aliases[stream_id] = (
                                real_completion_id
                            )
            stable_chunk = chunk.model_copy(
                update={"id": stream_id, "model": stream_model}
            )
            yield f"data: {stable_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Error producing streaming response")
        error = {
            "error": {
                "message": "AReaL stream generation failed",
                "type": "server_error",
            }
        }
        yield f"data: {json.dumps(error, separators=(',', ':'))}\n\n"
    finally:
        if not create_task.done():
            create_task.cancel()
            try:
                await create_task
            except asyncio.CancelledError:
                pass
        if openai_stream is not None:
            await openai_stream.aclose()


@app.post(
    f"/{RESPONSES_PATHNAME}",
    dependencies=[Depends(validate_json_request)],
)
async def responses(
    request: ResponseCreateParams, session_id: str = Depends(_require_session_key)
) -> Response:
    """OpenAI-compatible responses endpoint."""
    if _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail='Proxy server not initialized. Send requests to /create_engine then /call "initialize" first.',
        )
    return await _call_client_create(
        create_fn=_openai_client.responses.create,
        request=request,
        session_id=session_id,
    )


def _translate_anthropic_to_openai_request(anthropic_request: dict[str, Any]) -> dict:
    """Translate an Anthropic Messages API request to OpenAI format."""
    return translate_anthropic_request(
        anthropic_request,
        message_preprocessors=_message_preprocessors,
    )


async def _safe_stream_wrapper(
    stream: AsyncGenerator,
) -> AsyncGenerator:
    """Wrap an async generator to handle client disconnection gracefully.

    Ensures proper cleanup of the underlying stream when the client disconnects
    or an error occurs during streaming.

    Args:
        stream: The async generator to wrap.

    Yields:
        Chunks from the underlying stream.
    """
    try:
        async for chunk in stream:
            yield chunk
    except asyncio.CancelledError:
        logger.info("Streaming cancelled by client disconnect")
        raise
    finally:
        if hasattr(stream, "aclose"):
            await stream.aclose()


@app.post(
    f"/{ANTHROPIC_MESSAGES_PATHNAME}",
    dependencies=[Depends(validate_json_request)],
    response_model=None,
)
async def anthropic_messages(
    raw_request: Request, session_id: str = Depends(_require_session_key)
) -> Message | StreamingResponse:
    """Anthropic Messages API compatible endpoint.

    Converts Anthropic format requests to OpenAI format, processes through
    the OpenAI-compatible endpoint, then converts the response back to
    Anthropic format.

    Uses LiteLLM's AnthropicAdapter for format conversion.

    For streaming requests (stream=True in the request body), returns a
    StreamingResponse with Server-Sent Events (SSE) in Anthropic format.
    The SSE stream includes message_start, content_block_start,
    content_block_delta, content_block_stop, and message_stop events
    following the Anthropic streaming protocol.
    """

    if _openai_client is None:
        raise HTTPException(
            status_code=500,
            detail='Proxy server not initialized. Send requests to /create_engine then /call "initialize" first.',
        )

    # Parse Anthropic request
    anthropic_request = await raw_request.json()

    is_streaming = anthropic_request.get("stream", False)

    try:
        openai_request = _translate_anthropic_to_openai_request(anthropic_request)
    except (ValueError, TypeError, KeyError) as e:
        logger.warning(
            f"Failed to convert Anthropic request to OpenAI format due to invalid input: {e}"
        )
        raise HTTPException(
            status_code=400, detail=f"Invalid Anthropic request format: {e}"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error converting Anthropic request to OpenAI format: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Internal server error during request conversion."
        )

    if is_streaming:
        openai_stream = None
        try:
            # Get streaming response from OpenAI client
            openai_stream = await _call_client_create(
                create_fn=_openai_client.chat.completions.create,
                request=openai_request,
                session_id=session_id,
                stream=True,
            )

            # Use LiteLLM's adapter to convert to Anthropic SSE format
            anthropic_sse_stream = translate_anthropic_stream(
                openai_stream,
                model=anthropic_request.get("model", "default"),
            )

            # Wrap the stream to handle client disconnection gracefully
            safe_stream = _safe_stream_wrapper(anthropic_sse_stream)

            # Return streaming response
            return StreamingResponse(
                safe_stream,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        except HTTPException:
            raise
        except Exception as e:
            # Clean up stream on error during setup
            if openai_stream is not None and hasattr(openai_stream, "aclose"):
                await openai_stream.aclose()
            logger.error(f"Error setting up streaming response: {e}")
            raise HTTPException(status_code=500, detail=f"Streaming setup failed: {e}")

    # Non-streaming
    openai_response = await _call_client_create(
        create_fn=_openai_client.chat.completions.create,
        request=openai_request,
        session_id=session_id,
        stream=False,
    )

    # Convert OpenAI response to Anthropic format using LiteLLM's adapter
    try:
        return translate_anthropic_response(openai_response)
    except Exception as e:
        logger.error(f"Failed to convert OpenAI response to Anthropic format: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to convert response: {e}")


# =============================================================================
# Trajectory Export
# =============================================================================


@app.post(
    f"/{EXPORT_TRAJECTORIES_PATHNAME}",
    dependencies=[Depends(_require_admin_key)],
)
async def export_trajectories(
    request: ExportTrajectoriesRequest,
) -> ExportTrajectoriesResponse:
    """Export interactions for a completed session.

    The caller must supply an explicit ``session_id`` in the request body
    and authenticate with the admin key.  This avoids routing ambiguity
    when an API key has been reused across sessions.
    """
    session_id = request.session_id

    with _lock:
        if session_id not in _session_cache:
            raise HTTPException(
                status_code=404, detail=f"Session {session_id} not found"
            )
        session_data = _session_cache[session_id]

    # Wait for session to complete (non-blocking, outside lock)
    await session_data.wait_for_finish()

    # Export interactions
    interactions = session_data.export_interactions(
        discount=request.discount,
        style=request.style,
        drop_retry_orphans=request.drop_retry_orphans,
    )

    # Remove session from cache and clean up API key mapping
    with _lock:
        _session_cache.pop(session_id, None)
        _remove_api_keys_for_session(session_id)

    # Grouped inline/subproc sessions export only multimodal tensor references.
    # The workflow fetches each unique tensor once through the group endpoint.
    tensor_reference_group_id = (
        session_data.processor_cache_group_id
        if request.supports_shared_tensor_references
        else None
    )
    tensor_store = (
        _group_tensor_store_registry.get_or_create(tensor_reference_group_id)
        if tensor_reference_group_id is not None
        else None
    )
    serialized = serialize_interactions(interactions, tensor_store=tensor_store)
    return ExportTrajectoriesResponse(
        interactions=serialized,
        tensor_reference_group_id=tensor_reference_group_id,
    )


# =============================================================================
# Cleanup
# =============================================================================


def cleanup_engine():
    """Clean up engine on shutdown."""
    global _engine
    if _engine is not None:
        try:
            _engine.destroy()
            logger.info("Engine destroyed successfully")
        except Exception as e:
            logger.error(f"Error destroying engine: {e}")
        _engine = None


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for the proxy rollout server."""
    parser = argparse.ArgumentParser(description="Proxy Rollout Server")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to serve on (default: 0 = auto-assign)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    # name_resolve config (same as rpc_server.py)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--trial-name", type=str, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--worker-index", type=int, default=-1)
    parser.add_argument("--name-resolve-type", type=str, default="nfs")
    parser.add_argument(
        "--nfs-record-root", type=str, default="/tmp/areal/name_resolve"
    )
    parser.add_argument("--etcd3-addr", type=str, default="localhost:2379")
    parser.add_argument(
        "--fileroot",
        type=str,
        default=None,
        help="Root directory for log files (unused, for compatibility with rpc_server)",
    )

    args, _ = parser.parse_known_args()

    # Set global server address variables
    global _server_host, _server_port, _worker_role, _worker_index
    global \
        _experiment_name, \
        _trial_name, \
        _name_resolve_type, \
        _nfs_record_root, \
        _etcd3_addr
    _server_host = args.host
    if _server_host == "0.0.0.0":
        _server_host = gethostip()

    # Set global config for name_resolve
    _experiment_name = args.experiment_name
    _trial_name = args.trial_name
    _name_resolve_type = args.name_resolve_type
    _nfs_record_root = args.nfs_record_root
    _etcd3_addr = args.etcd3_addr

    # Get worker identity
    worker_role = args.role
    worker_index = _resolve_worker_index(args.worker_index)
    _worker_role = worker_role
    _worker_index = worker_index
    worker_id = f"{worker_role}/{worker_index}"

    # Determine port
    _server_port = args.port if args.port != 0 else find_free_ports(1)[0]
    _allocated_ports.add(_server_port)

    # Configure name_resolve and register this server
    name_resolve.reconfigure(
        NameResolveConfig(
            type=args.name_resolve_type,
            nfs_record_root=args.nfs_record_root,
            etcd3_addr=args.etcd3_addr,
        )
    )
    key = names.worker_discovery(
        args.experiment_name, args.trial_name, args.role, worker_index
    )
    name_resolve.add(key, f"{_server_host}:{_server_port}", replace=True)

    logger.info(
        f"Starting proxy rollout server on {_server_host}:{_server_port} for worker {worker_id}"
    )

    try:
        # Run uvicorn directly (blocking)
        uvicorn.run(
            app,
            host=_server_host,
            port=_server_port,
            log_level="warning",
            timeout_keep_alive=300,
            access_log=False,
        )
    except KeyboardInterrupt:
        logger.info("Shutting down proxy rollout server")
    finally:
        cleanup_engine()
        logger.info("Proxy rollout server stopped.")


if __name__ == "__main__":
    main()

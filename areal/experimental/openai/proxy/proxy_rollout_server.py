# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import hmac
import inspect
import os
import secrets
import threading
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import uvicorn
from anthropic.types.message import Message
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    AnthropicAdapter,
)
from litellm.types.utils import ModelResponse as LitellmModelResponse
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.completion_create_params import CompletionCreateParams
from openai.types.responses import Response
from openai.types.responses.response_create_params import ResponseCreateParams
from pydantic import BaseModel

from areal.api.cli_args import NameResolveConfig
from areal.experimental.openai.client import ArealOpenAI
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.infra.utils.http import validate_admin_api_key
from areal.utils import name_resolve, names, seeding
from areal.utils.dynamic_import import import_from_string
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.logging import getLogger
from areal.utils.network import find_free_ports, gethostip

from .server import (
    ANTHROPIC_MESSAGES_PATHNAME,
    CHAT_COMPLETIONS_PATHNAME,
    DEFAULT_ADMIN_API_KEY,
    EXPORT_TRAJECTORIES_PATHNAME,
    GRANT_CAPACITY_PATHNAME,
    RESPONSES_PATHNAME,
    RL_END_SESSION_PATHNAME,
    RL_SET_REWARD_PATHNAME,
    RL_START_SESSION_PATHNAME,
    ExportTrajectoriesRequest,
    ExportTrajectoriesResponse,
    SessionData,
    SetRewardRequest,
    StartSessionRequest,
    StartSessionResponse,
    serialize_interactions,
)

if TYPE_CHECKING:
    from areal.api import InferenceEngine


logger = getLogger("ProxyRolloutServer")


# =============================================================================
# Warning Deduplication
# =============================================================================


# Set AREAL_PROXY_WARN_ONCE=1 to avoid warning spamming
_warn_once_enabled = os.environ.get("AREAL_PROXY_WARN_ONCE", "0") == "1"
_warned_messages: set[str] = set()
_warn_lock = threading.Lock()


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

# API key authentication
# Initialized to a random value so pre-configuration requests cannot bypass auth.
# Overwritten by _setup_openai_client() once the engine is configured.
_admin_api_key: str = secrets.token_urlsafe(32)
_api_key_to_session: dict[str, str] = {}
_session_to_api_key: dict[str, str] = {}  # Reverse mapping for O(1) cleanup

# Server address (set at startup)
_server_host: str = "0.0.0.0"
_server_port: int = 8000

# Port allocation tracking
_allocated_ports: set[int] = set()
_port_alloc_lock = asyncio.Lock()

# Server config (needed for name_resolve registration)
_experiment_name: str | None = None
_trial_name: str | None = None
_name_resolve_type: str = "nfs"
_nfs_record_root: str = "/tmp/areal/name_resolve"
_etcd3_addr: str = "localhost:2379"

# Adapter to convert Anthropic request to OpenAI format
_adapter = AnthropicAdapter()

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


def _extract_bearer_token(request: Request) -> str:
    """Extract API token from Authorization header or x-api-key header.

    Supports both 'Authorization: Bearer <token>' (OpenAI SDK, case-insensitive
    per RFC 6750) and 'x-api-key: <token>' (Anthropic SDK) for cross-SDK
    compatibility.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()  # len("Bearer ") == 7
    # Fallback to x-api-key header (used by Anthropic SDK)
    x_api_key = request.headers.get("x-api-key", "")
    if x_api_key:
        return x_api_key
    raise HTTPException(
        status_code=401,
        detail="Missing or malformed Authorization header. Expected 'Bearer <token>' or 'x-api-key: <token>'.",
    )


def _require_admin_key(request: Request) -> str:
    """Validate that the request carries the admin API key.

    Uses constant-time comparison to prevent timing attacks.
    """
    token = _extract_bearer_token(request)
    if not hmac.compare_digest(token, _admin_api_key):
        raise HTTPException(status_code=403, detail="Invalid admin API key.")
    return token


def _require_session_key(request: Request) -> str:
    """Resolve session_id from the session API key in the Authorization header."""
    token = _extract_bearer_token(request)
    with _lock:
        session_id = _api_key_to_session.get(token)
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
    return {"status": "ok", "initialized": _engine is not None}


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
    config = _engine.config
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    agent_cfg = config.agent
    _openai_client = ArealOpenAI(
        engine=_engine,
        tokenizer=tokenizer,
        tool_call_parser=agent_cfg.tool_call_parser,
        reasoning_parser=agent_cfg.reasoning_parser,
        engine_max_tokens=agent_cfg.engine_max_tokens,
        chat_template_type=agent_cfg.chat_template_type,
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
        _session_cache.pop(session_id, None)

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

        _capacity -= 1
        _session_cache[session_id] = SessionData(session_id=session_id)
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
    return {"message": "success", "interaction_count": interaction_count}


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
        logger.error(f"Interaction {interaction_id} not found in session {session_id}")
        raise HTTPException(
            status_code=400, detail=f"Interaction {interaction_id} not found"
        )
    session_data.completions.set_reward(interaction_id, reward)
    return {"message": "success"}


# =============================================================================
# OpenAI-Compatible Endpoints
# =============================================================================


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

    sig = inspect.signature(create_fn)
    areal_client_ignored_args = ["model"] + (extra_ignored_args or [])
    areal_client_disallowed_args = ["areal_cache"]
    areal_client_allowed_args = list(
        k
        for k in sig.parameters.keys()
        if k not in areal_client_ignored_args and k not in areal_client_disallowed_args
    )

    kwargs = request.model_dump() if isinstance(request, BaseModel) else dict(request)
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

    # Strip stream from request body to prevent it from bypassing the explicit
    # `stream` parameter.  Without this, a request with {"stream": true} would
    # leak through kwargs and cause the client to return an AsyncGenerator even
    # when the caller did not ask for streaming.
    kwargs.pop("stream", None)
    if stream:
        kwargs["stream"] = True

    try:
        return await create_fn(areal_cache=session_data.completions, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post(
    f"/{CHAT_COMPLETIONS_PATHNAME}",
    dependencies=[Depends(validate_json_request)],
    response_model=None,
)
async def chat_completions(
    request: CompletionCreateParams, session_id: str = Depends(_require_session_key)
) -> ChatCompletion | StreamingResponse:
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming (stream=True) and non-streaming requests.
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
        openai_stream = None
        try:
            openai_stream = await _call_client_create(
                create_fn=_openai_client.chat.completions.create,
                request=request,
                session_id=session_id,
                stream=True,
            )

            # Convert ChatCompletionChunk objects to OpenAI SSE format
            async def _openai_sse_generator(
                chunk_stream: AsyncGenerator[ChatCompletionChunk, None],
            ) -> AsyncGenerator[str, None]:
                async for chunk in chunk_stream:
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            safe_stream = _safe_stream_wrapper(_openai_sse_generator(openai_stream))

            return StreamingResponse(
                safe_stream,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        except Exception as e:
            if openai_stream is not None and hasattr(openai_stream, "aclose"):
                await openai_stream.aclose()
            logger.error(f"Error setting up streaming response: {e}")
            raise HTTPException(status_code=500, detail=f"Streaming setup failed: {e}")

    return await _call_client_create(
        create_fn=_openai_client.chat.completions.create,
        request=request,
        session_id=session_id,
    )


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
    openai_request = _adapter.translate_completion_input_params(
        anthropic_request.copy()
    )
    if openai_request is None:
        raise ValueError("Failed to translate request")
    openai_request = dict(openai_request)

    # Fix message content if it's a list (Anthropic format with content blocks)
    # LiteLLM's adapter may not properly convert content from list to string
    # Claude Code CLI sends content as: [{"type":"text","text":"...","cache_control":{...}}, ...]
    if "messages" in openai_request:
        for msg in openai_request["messages"]:
            if isinstance(msg.get("content"), list):
                # Convert list of content blocks to string
                text_parts = []
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                msg["content"] = "\n".join(text_parts)

    return openai_request


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
            anthropic_sse_stream = (
                _adapter.translate_completion_output_params_streaming(
                    completion_stream=openai_stream,
                    model=anthropic_request.get("model", "default"),
                )
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
        # Convert ChatCompletion to LitellmModelResponse
        openai_response_dict = openai_response.model_dump()
        model_response = LitellmModelResponse(**openai_response_dict)
        anthropic_response = _adapter.translate_completion_output_params(model_response)
        if anthropic_response is None:
            raise ValueError("Failed to translate response")

        # LiteLLM returns Pydantic BaseModel objects in content list,
        # Convert them to dict.
        if "content" in anthropic_response and anthropic_response["content"]:
            anthropic_response["content"] = [
                block.model_dump() if hasattr(block, "model_dump") else block
                for block in anthropic_response["content"]
            ]
        return Message(**anthropic_response)
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
    )

    # Remove session from cache and clean up API key mapping
    with _lock:
        _session_cache.pop(session_id, None)
        _remove_api_keys_for_session(session_id)

    # Serialize for HTTP transport
    serialized = serialize_interactions(interactions)
    return ExportTrajectoriesResponse(interactions=serialized)


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
    global _server_host, _server_port
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
    worker_index = args.worker_index

    if "SLURM_PROCID" in os.environ:
        # Overwriting with slurm task id
        worker_index = int(os.environ["SLURM_PROCID"])
    if worker_index == -1:
        raise ValueError("Invalid worker index. Not found from SLURM environ or args.")
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

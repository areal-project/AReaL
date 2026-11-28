# SPDX-License-Identifier: Apache-2.0

"""Only backend-confirmed running deadlines may supply model-timeout evidence."""

import asyncio
import hashlib
import json
import pickle
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web

from examples.swe.arena_agent import ArenaStreamAgentWorkflow
from tests.experimental.openai.test_context_overflow_recovery import _FakeProxyClient
from tests.test_http_error_diagnostics import http_server
from tests.test_remote_inf_engine_r3 import (
    _make_engine,
    _make_request,
    _make_sglang_response,
)
from tests.test_swe_arena_multi_stream import (
    _core_tool_json_failure_error,
    _set_core_failure_outcome,
)
from tests.v2.inference_service import test_backend_router

from areal.engine.sglang_remote import SGLangBackend
from areal.experimental.openai.proxy import client_session as client_module
from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy import workflow as workflow_module
from areal.experimental.openai.proxy.server import SessionData
from areal.experimental.openai.proxy.workflow import OpenAIProxyWorkflow
from areal.infra import remote_inf_engine, workflow_context
from areal.infra.utils.http import arequest_with_retry
from areal.utils import stats_tracker
from areal.utils.errors import ModelExecutionTimeoutError
from areal.v2.inference_service.router.backend import (
    BackendSpec,
    RegisterBackends,
    SetPoolState,
)
from areal.v2.inference_service.router.client import RouteClient

live_client = test_backend_router.live_client


def _timeout_http_body(message="Request running timeout reached."):
    return {"object": "error", "message": message, "type": "503", "code": 503}


def _model_timeout_error():
    """Legacy Core preserves the terminal proxy message digest after label loss."""
    error = _core_tool_json_failure_error()
    outcome = deepcopy(error.result.raw["outcome"])
    message = ModelExecutionTimeoutError.message.encode("utf-8")
    outcome["details"]["diagnosticSignals"].update(
        messageSha256=hashlib.sha256(message).hexdigest(),
        messageBytes=len(message),
        messageScannedChars=len(message),
    )
    _set_core_failure_outcome(error, outcome)
    return error


def test_model_timeout_type_survives_exception_serialization():
    """Worker exception transport must preserve the canonical error type."""
    restored = pickle.loads(pickle.dumps(ModelExecutionTimeoutError()))
    assert isinstance(restored, ModelExecutionTimeoutError)
    assert str(restored) == ModelExecutionTimeoutError.message


@pytest.mark.parametrize(
    "status,body,accepted",
    [
        (503, _timeout_http_body(), True),
        (503, _timeout_http_body("Request waiting timeout reached."), False),
        (503, _timeout_http_body("Service unavailable"), False),
        (504, _timeout_http_body(), False),
        (503, {"detail": "Request running timeout reached."}, False),
        (503, {**_timeout_http_body(), "code": "503"}, False),
    ],
)
def test_backend_timeout_requires_exact_running_error(status, body, accepted):
    error = SGLangBackend.generation_error(status, json.dumps(body))
    assert isinstance(error, ModelExecutionTimeoutError) == accepted


@pytest.mark.asyncio
async def test_running_timeout_http_is_terminal_without_retry():
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        return web.json_response(_timeout_http_body(), status=503)

    async with http_server(handler) as addr:
        with pytest.raises(ModelExecutionTimeoutError):
            await arequest_with_retry(
                addr,
                "/generate",
                max_retries=3,
                generation_error=SGLangBackend.generation_error,
            )
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("wire", ["abort", "http"])
@pytest.mark.parametrize("sampling_mask", [False, True])
async def test_running_timeout_abort_never_resubmits_and_releases_route(
    monkeypatch, routed, wire, sampling_mask
):
    engine = _make_engine(return_routed_experts=False)
    req = _make_request()
    if sampling_mask:
        engine.config._sampling_mask_sampling = req.gconfig.new(top_k=8)
        req.metadata["_sampling_mask_lease"] = [1, 2, 3]
    result = {
        "meta_info": {
            "finish_reason": {
                "type": "abort",
                "status_code": 503,
                "message": "Request running timeout reached.",
            }
        }
    }

    async def handler(request):
        if wire == "http":
            return web.json_response(_timeout_http_body(), status=503)
        return web.json_response(result)

    async with http_server(handler) as addr, aiohttp.ClientSession() as session:
        monkeypatch.setattr(
            workflow_context, "get_aiohttp_session", AsyncMock(return_value=session)
        )
        outcomes = []
        calls = 0

        class Route:
            async def execute(self, session, session_id, generate):
                nonlocal calls
                calls += 1
                value, outcome = await generate(
                    SimpleNamespace(backend_addr=addr, weight_version=0)
                )
                outcomes.append(outcome)  # Real RouteClient releases before returning.
                return value, 0

        if routed:
            engine.route_client = Route()
            with pytest.raises(ModelExecutionTimeoutError):
                await engine._agenerate(req)
            assert outcomes == ["aborted"]
            assert calls == 1
        else:
            if wire == "http":
                engine.addresses = [addr]
                with pytest.raises(ModelExecutionTimeoutError):
                    await engine._agenerate(req)
                return
            mocked = AsyncMock(return_value=result)
            monkeypatch.setattr(remote_inf_engine, "arequest_with_retry", mocked)
            with pytest.raises(ModelExecutionTimeoutError):
                await engine._agenerate(req)
            mocked.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("auxiliary", [False, True])
async def test_proxy_records_only_training_timeout_and_fences_retries(
    monkeypatch, auxiliary
):
    session = SessionData(
        session_id="task", metadata={"auxiliary_system_prompts": ["Title"]}
    )
    monkeypatch.setattr(srv, "_openai_client", object())
    monkeypatch.setattr(srv, "_session_cache", {"task": session})
    calls = 0

    async def create_fn(messages, **kwargs):
        nonlocal calls
        calls += 1
        raise ModelExecutionTimeoutError()

    request = {
        "messages": [
            {
                "role": "system" if auxiliary else "user",
                "content": "Title" if auxiliary else "Solve",
            }
        ]
    }
    with pytest.raises(ModelExecutionTimeoutError):
        await srv._call_client_create(create_fn, request, "task")
    assert session.model_execution_timeout is (not auxiliary)
    assert session.recoverable_model_execution_timeout is (not auxiliary)
    assert session.system_error is False
    if not auxiliary:
        with pytest.raises(ModelExecutionTimeoutError):
            await srv._call_client_create(create_fn, request, "task")
        assert calls == 1


def test_concurrent_timeout_does_not_prove_the_terminal_cause():
    session = SessionData(session_id="task")
    session.begin_model_request()
    session.begin_model_request()
    session.mark_model_execution_timeout()
    session.end_model_request()
    assert not session.recoverable_model_execution_timeout
    session.end_model_request()
    assert not session.recoverable_model_execution_timeout


@pytest.mark.asyncio
async def test_timeout_stream_and_end_session_preserve_typed_evidence(monkeypatch):
    """Streaming errors and end-session metadata carry the same local proof."""
    session = SessionData(session_id="task")

    async def create_fn(messages, **kwargs):
        raise ModelExecutionTimeoutError()

    monkeypatch.setattr(srv, "_session_cache", {"task": session})
    monkeypatch.setattr(srv, "_engine", None)
    monkeypatch.setattr(
        srv,
        "_openai_client",
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_fn))
        ),
    )
    chunks = [
        chunk
        async for chunk in srv._deferred_openai_sse_stream(
            {"messages": [{"role": "user", "content": "solve"}]}, "task"
        )
    ]
    error = json.loads(chunks[-1].removeprefix("data: ").strip())["error"]
    assert error == {
        "message": ModelExecutionTimeoutError.message,
        "type": "invalid_request_error",
        "code": ModelExecutionTimeoutError.code,
    }
    ended = await srv.end_session("task")
    assert ended["model_execution_timeout"] is True
    assert ended["system_error"] is False

    response = await srv.invalid_tool_call_error_response(
        None, ModelExecutionTimeoutError()
    )
    assert response.status_code == 422
    assert response.headers["x-should-retry"] == "false"


@pytest.mark.asyncio
async def test_disconnected_stream_with_outstanding_lease_vetoes_later_timeout(
    monkeypatch, live_client
):
    """An SSE disconnect does not terminate the shielded backend generation."""
    state, _, http_session, controls = live_client
    identity = {"pool_id": "timeout-pool", "run_id": "run"}
    await state.register(
        RegisterBackends(
            **identity,
            weight_version=0,
            backends=[
                BackendSpec(
                    backend_id="backend", backend_addr=controls["address"], capacity=2
                )
            ],
        )
    )
    await state.set_state(
        identity["pool_id"],
        SetPoolState(run_id="run", control_seq=0, state="READY", weight_version=0),
    )
    route = RouteClient(controls["address"], **identity, admin_api_key="test")
    route.set_sender_state(False, 0)
    engine = _make_engine(return_routed_experts=False)
    engine.route_client = route
    session = SessionData(session_id="task")
    started, finish = asyncio.Event(), asyncio.Event()
    calls = 0

    async def generate(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await finish.wait()
            return web.json_response(_make_sglang_response([(-0.1, 7)]))
        return web.json_response(_timeout_http_body(), status=503)

    async def create_fn(messages, **kwargs):
        await engine._agenerate(_make_request())

        async def chunks():
            for chunk in ():
                yield chunk

        return chunks()

    async def consume():
        return [
            chunk
            async for chunk in srv._deferred_openai_sse_stream(
                {"messages": [{"role": "user", "content": "solve"}]}, "task"
            )
        ]

    controls["generate"] = generate
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=http_session)
    )
    monkeypatch.setattr(srv, "_session_cache", {"task": session})
    monkeypatch.setattr(srv, "_engine", engine)
    monkeypatch.setattr(srv, "_resume_streams", None)
    monkeypatch.setattr(
        srv,
        "_openai_client",
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create_fn))
        ),
    )
    first = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(started.wait(), 2)
        first.cancel()  # Same cancellation path as a disconnected StreamingResponse.
        with pytest.raises(asyncio.CancelledError):
            await first
        assert (await state.snapshot(**identity))["outstanding"] == 1

        chunks = await asyncio.wait_for(consume(), 2)
        assert ModelExecutionTimeoutError.code in chunks[-1]
        assert (await state.snapshot(**identity))["outstanding"] == 1
        ended = await srv.end_session("task")
        assert ended["system_error"] is False
        assert ended["model_execution_timeout"] is False
        assert (
            ArenaStreamAgentWorkflow.classify_proxy_failure(
                _model_timeout_error(),
                context_overflow=False,
                interaction_count=2,
                model_execution_timeout=ended["model_execution_timeout"],
            )
            == "unknown_failure_reject"
        )
    finally:
        finish.set()
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)
        await asyncio.wait_for(asyncio.gather(*list(route._tasks)), 3)
    assert (await state.snapshot(**identity))["outstanding"] == 0
    assert session.recoverable_model_execution_timeout is False


@pytest.mark.parametrize("auxiliary", [False, True])
@pytest.mark.asyncio
async def test_cancelled_request_veto_is_isolated_from_auxiliary_calls(
    monkeypatch, auxiliary
):
    """Cancelled auxiliary generation must not poison the training session."""
    session = SessionData(
        session_id="task", metadata={"auxiliary_system_prompts": ["Title"]}
    )
    monkeypatch.setattr(srv, "_openai_client", object())
    monkeypatch.setattr(srv, "_session_cache", {"task": session})

    async def create_fn(messages, **kwargs):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await srv._call_client_create(
            create_fn,
            {
                "messages": [
                    {
                        "role": "system" if auxiliary else "user",
                        "content": "Title" if auxiliary else "Solve",
                    }
                ]
            },
            "task",
        )
    session.mark_model_execution_timeout()
    assert session.recoverable_model_execution_timeout is auxiliary


@pytest.mark.parametrize(
    "terminal_message,accepted",
    [
        ("model HTTP status 422", True),
        ("model HTTP status 422 Unprocessable Entity", True),
        ("model HTTP status 422 Unprocessable Entity: upstream timeout", False),
        ("model HTTP status 503 Service Unavailable", False),
    ],
)
@pytest.mark.parametrize(
    "fault", [None, "no_local_proof", "wrong_digest", "wrong_status"]
)
def test_core_http_timeout_status_text_requires_matching_proof(
    terminal_message, accepted, fault
):
    """Source-derived Core HttpFailure fixture; no deployed-image provenance.

    AReaL-Harness 9a586229 core/engine/src/model.rs formats reqwest::StatusCode
    with its reason phrase. Keep legacy bare-status receipts supported too.
    """
    error = _model_timeout_error()
    outcome = deepcopy(error.result.raw["outcome"])
    outcome["source"] = "provider_http"
    outcome["details"].update(eventType="http_error", httpStatus=422)
    if fault == "wrong_digest":
        outcome["details"]["diagnosticSignals"]["messageSha256"] = "0" * 64
    elif fault == "wrong_status":
        outcome["details"]["httpStatus"] = 503
    _set_core_failure_outcome(error, outcome)
    error.harness_result["error"][-1]["message"] = terminal_message
    assert ArenaStreamAgentWorkflow.classify_proxy_failure(
        error,
        context_overflow=False,
        interaction_count=2,
        model_execution_timeout=fault != "no_local_proof",
    ) == (
        "model_failure_zero" if accepted and fault is None else "unknown_failure_reject"
    )


@pytest.mark.asyncio
async def test_proxy_client_reads_timeout_evidence_from_end_session(monkeypatch):
    """The optional field defaults safely for old proxies and reaches workflows."""
    proxy = client_module.OpenAIProxyClient(
        session=object(),
        base_url="http://proxy",
        task_id="task",
        admin_api_key="test-admin",
    )
    proxy.session_id = "task"
    proxy._session_api_key = "session-key"
    monkeypatch.setattr(
        client_module,
        "post_json_with_retry",
        AsyncMock(
            return_value={
                "interaction_count": 2,
                "model_execution_timeout": True,
            }
        ),
    )
    await proxy.__aexit__(None, None, None)
    assert proxy.model_execution_timeout is True
    assert proxy.interaction_count == 2


@pytest.mark.parametrize("interaction_count", [0, 2])
@pytest.mark.parametrize("local_timeout", [False, True])
def test_model_timeout_requires_local_and_terminal_proof(
    interaction_count, local_timeout
):
    disposition = ArenaStreamAgentWorkflow.classify_proxy_failure(
        _model_timeout_error(),
        context_overflow=False,
        interaction_count=interaction_count,
        model_execution_timeout=local_timeout,
    )
    assert disposition == (
        "model_failure_zero"
        if interaction_count and local_timeout
        else "unknown_failure_reject"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "runner_deadline",
        "missing_receipt",
        "wrong_digest",
        "wrong_terminal",
        "collection_failed",
        "external_interruption",
        "system_message",
    ],
)
def test_ambiguous_or_conflicting_timeout_stays_rejected(fault):
    error = _model_timeout_error()
    if fault == "runner_deadline":
        outcome = {
            "class": "agent",
            "code": "AGENT_RUN_TIMEOUT",
            "source": "runner_deadline",
            "details": {"reason": "process_deadline"},
        }
        _set_core_failure_outcome(error, outcome)
        summary = (
            "GAMEAGENT_OUTCOME_CODE=AGENT_RUN_TIMEOUT GAMEAGENT_OUTCOME_CLASS=agent"
        )
        error.result.raw["harness"]["summary"] = summary
        error.harness_result["summary"] = summary
    elif fault == "missing_receipt":
        error.harness_result = None
    elif fault == "wrong_digest":
        outcome = deepcopy(error.result.raw["outcome"])
        outcome["details"]["diagnosticSignals"]["messageSha256"] = "0" * 64
        _set_core_failure_outcome(error, outcome)
    elif fault == "wrong_terminal":
        error.harness_result["raw"]["outcome"]["source"] = "core"
    elif fault == "collection_failed":
        error.harness_result["collection"] = {"status": "ERROR"}
    elif fault == "system_message":
        error.harness_result["error"][-1]["message"] = "HTTP 503 service unavailable"
    else:
        error.harness_result["observed_external_interruption"] = True
    assert (
        ArenaStreamAgentWorkflow.classify_proxy_failure(
            error,
            context_overflow=False,
            interaction_count=2,
            model_execution_timeout=True,
        )
        == "unknown_failure_reject"
    )


@pytest.mark.asyncio
async def test_retry_after_uncertain_http_failure_cannot_become_model_timeout():
    """A prior failed POST can still be running when a retry reaches its limit."""
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return web.json_response({"error": "upstream unavailable"}, status=503)
        return web.json_response(_timeout_http_body(), status=503)

    async with http_server(handler) as addr:
        with pytest.raises(RuntimeError) as caught:
            await arequest_with_retry(
                addr,
                "/generate",
                max_retries=2,
                retry_delay=0,
                generation_error=SGLangBackend.generation_error,
            )
    assert not isinstance(caught.value, ModelExecutionTimeoutError)
    assert calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("resubmit", [False, True])
@pytest.mark.parametrize("timeout_wire", ["abort", "http"])
async def test_retry_then_successful_http_abort_keeps_prior_uncertainty(
    monkeypatch, resubmit, timeout_wire
):
    """A 200 abort envelope cannot erase an earlier unconfirmed execution."""
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return web.json_response({"error": "upstream unavailable"}, status=503)
        if resubmit and calls == 2:
            return web.json_response(
                {
                    "meta_info": {
                        "finish_reason": {"type": "abort", "message": "Aborted"},
                        "output_token_logprobs": [[-0.1, 7]],
                    }
                }
            )
        if timeout_wire == "http":
            return web.json_response(_timeout_http_body(), status=503)
        return web.json_response(
            {
                "meta_info": {
                    "finish_reason": {
                        "type": "abort",
                        "status_code": 503,
                        "message": "Request running timeout reached.",
                    }
                }
            }
        )

    engine = _make_engine(return_routed_experts=False)
    engine.config.request_retries = 2
    async with http_server(handler) as addr, aiohttp.ClientSession() as session:
        engine.addresses = [addr]
        monkeypatch.setattr(
            workflow_context, "get_aiohttp_session", AsyncMock(return_value=session)
        )
        with pytest.raises(RuntimeError) as caught:
            await engine._agenerate(_make_request())
    assert not isinstance(caught.value, ModelExecutionTimeoutError)
    assert calls == (3 if resubmit else 2)


@pytest.mark.parametrize("retained_code", [False, True])
@pytest.mark.parametrize("completed_turns", [0, 2])
def test_timeout_terminal_wrapper_survives_code_preservation(
    retained_code, completed_turns
):
    """Both legacy label loss and preserved typed errors bind to the final turn."""
    error = _model_timeout_error()
    if retained_code:
        outcome = deepcopy(error.result.raw["outcome"])
        outcome["details"]["code"] = ModelExecutionTimeoutError.code
        _set_core_failure_outcome(error, outcome)
        error.harness_result["error"][-1]["message"] = (
            "model reported a streaming error (event=error, "
            'code=Some("model_execution_timeout"), '
            'type=Some("invalid_request_error"), reason=None)'
        )
    error.harness_result["turn_statuses"] = ["completed"] * completed_turns + ["failed"]
    error.harness_result["error"] = [None] * completed_turns + error.harness_result[
        "error"
    ]
    error.harness_result["continuations"] = completed_turns
    assert (
        ArenaStreamAgentWorkflow.classify_proxy_failure(
            error,
            context_overflow=False,
            interaction_count=2,
            model_execution_timeout=True,
        )
        == "model_failure_zero"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("system_error", [False, True])
@pytest.mark.parametrize("style", ["individual", "concat"])
async def test_timeout_episode_exports_zero_only_with_system_veto_clear(
    monkeypatch, system_error, style
):
    """Actual Arena classification and proxy export preserve the system veto."""
    monkeypatch.setenv("ARENA_OPENAPI_TOKEN", "test-token")
    proxy = _FakeProxyClient()
    proxy.context_overflow = False
    proxy.model_execution_timeout = True
    proxy.system_error = system_error
    proxy.export_interactions = AsyncMock(
        return_value={"completion-1": proxy.interaction}
    )
    monkeypatch.setattr(workflow_module, "OpenAIProxyClient", lambda **kwargs: proxy)
    monkeypatch.setattr(
        workflow_context, "get_aiohttp_session", AsyncMock(return_value=object())
    )
    agent = ArenaStreamAgentWorkflow(
        econfig={"arena_base_url": "https://arena.example", "stream_id": "stream"}
    )
    agent.persist_episode_result = AsyncMock()
    workflow = OpenAIProxyWorkflow(mode="inline", agent=agent, export_style=style)
    error = _model_timeout_error()
    monkeypatch.setattr(workflow, "_run_agent", AsyncMock(side_effect=error))
    monkeypatch.setattr(workflow, "_grant_capacity", AsyncMock())
    try:
        if system_error:
            with pytest.raises(type(error)):
                await workflow.arun_episode(None, {})
            proxy.export_interactions.assert_not_awaited()
        else:
            result = await workflow.arun_episode(None, {})
            assert result["completion-1"].reward == 0.0
            if style == "individual":
                assert result["completion-1"].rollout_reward == 0.0
            else:
                assert (
                    proxy.export_interactions.await_args.kwargs["episode_reward"] == 0.0
                )
            agent.persist_episode_result.assert_awaited_once_with({}, 0.0)
    finally:
        stats_tracker.export_all(reset=True)

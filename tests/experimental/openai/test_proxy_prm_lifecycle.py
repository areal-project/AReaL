# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for PRM ownership in the v1 proxy's RPC and ASGI lifecycle."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import torch

from areal.api import ModelResponse
from areal.api.cli_args import AgentConfig
from areal.experimental.openai.proxy import proxy_rollout_server as srv
from areal.experimental.openai.proxy.server import SessionData, deserialize_interactions
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.reward.prm import BaseScorer, PRMConfig, PRMScorerConfig

_ADMIN_KEY = "test-admin-key"


class ResourceScorer(BaseScorer):
    """Real configured scorer with observable scoring and resource ownership."""

    name = "resource"
    instances = []
    events = []
    created = None

    def __init__(
        self,
        label="owned",
        fail_close=False,
        fail_construct=False,
        fail_validate=False,
        **kwargs,
    ):
        if fail_construct:
            raise ValueError("scorer construction failed")
        super().__init__(**kwargs)
        self.label = label
        self.fail_close = fail_close
        self.fail_validate = fail_validate
        self.fail_score = False
        self.loop = asyncio.get_running_loop()
        self.close_loop = None
        self.close_calls = 0
        self.close_started = asyncio.Event()
        self.close_release = None
        self.score_started = asyncio.Queue()
        self.score_release = None
        self.active = 0
        self.flushed = False
        self.instances.append(self)
        self.created.put_nowait(self)

    def validate_prm_config(self, config, *, training_enabled):
        if self.fail_validate:
            raise ValueError("scorer validation failed")

    async def evaluate(self, interaction, ctx):
        assert self.close_calls == 0
        self.active += 1
        self.score_started.put_nowait(None)
        try:
            if self.score_release is not None:
                await self.score_release.wait()
            if self.fail_score:
                raise ValueError("scoring failed")
            return 1.0
        finally:
            self.active -= 1

    async def aclose(self):
        assert self.active == 0
        self.close_loop = asyncio.get_running_loop()
        self.close_calls += 1
        self.events.append(f"close:{self.label}")
        self.close_started.set()
        if self.close_release is not None:
            await self.close_release.wait()
        if self.fail_close:
            raise ValueError(f"cannot flush {self.label}")
        self.flushed = True


def _spec(label="owned", *, enabled=True, **kwargs):
    return PRMScorerConfig(
        path=f"{__name__}.ResourceScorer",
        enabled=enabled,
        kwargs={"label": label, **kwargs},
    )


@pytest.fixture(autouse=True)
def reset_proxy(monkeypatch):
    """Give each test a fresh process/loop lifecycle without model dependencies."""
    ResourceScorer.instances.clear()
    ResourceScorer.events.clear()
    ResourceScorer.created = asyncio.Queue()
    for name, value in {
        "_engine": None,
        "_openai_client": None,
        "_prm_runner": None,
        "_engine_init_request": None,
        "_engine_init_result": None,
        "_engine_lifecycle_lock": asyncio.Lock(),
        "_proxy_closing": False,
        "_prm_required": False,
        "_active_prm_exports": 0,
        "_prm_idle": asyncio.Event(),
        "_session_cache": {},
        "_api_key_to_session": {},
        "_session_to_api_key": {},
        "_lock": threading.Lock(),
        "_admin_api_key": _ADMIN_KEY,
        "_message_preprocessors": [],
        "_prefix_matcher": None,
        "_session_timeout_seconds": 3600,
        "_deterministic_sampling": False,
    }.items():
        monkeypatch.setattr(srv, name, value)


@pytest.fixture
def engine(monkeypatch):
    """Mock only model resources; use actual setup, scorer factory and HTTP routes."""
    engine = SimpleNamespace(
        config=SimpleNamespace(
            tokenizer_path="mock-tokenizer",
            lora_name="",
            agent=AgentConfig(
                agent_cls_path="tests.unused.Agent",
                admin_api_key=_ADMIN_KEY,
                prm=PRMConfig(scorers=[_spec()]),
            ),
        ),
        initialize=MagicMock(return_value={"engine": "ready"}),
        destroy=MagicMock(side_effect=lambda: ResourceScorer.events.append("engine")),
    )
    monkeypatch.setattr(srv, "_engine", engine)
    monkeypatch.setattr(
        srv, "load_hf_processor_and_tokenizer", lambda _: (None, object())
    )
    monkeypatch.setattr(srv, "ArealOpenAI", MagicMock(return_value=object()))
    return engine


def _client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=srv.app), base_url="http://testserver"
    )


async def _rpc(client, method, **kwargs):
    return await client.post("/call", json={"method": method, "kwargs": kwargs})


def _session(session_id="session", *, finished=True):
    session = SessionData(session_id=session_id)
    interaction = InteractionWithTokenLogpReward(
        model_response=ModelResponse(
            input_tokens=[1],
            output_tokens=[2],
            output_logprobs=[0.0],
            output_versions=[0],
        ),
        messages=[{"role": "user", "content": "question"}],
        output_message_list=[{"role": "assistant", "content": "answer"}],
        reward=0.0,
        chat_template_type="concat",
    )
    interaction.interaction_id = "turn"
    session.completions["turn"] = interaction
    if finished:
        session.finish()
    srv._session_cache[session_id] = session
    return session


async def _export(client, session_id="session"):
    return await client.post(
        "/export_trajectories",
        headers={"Authorization": f"Bearer {_ADMIN_KEY}"},
        json={"session_id": session_id, "style": "concat"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_destroy", [False, True])
async def test_proxy_closes_configured_scorers_on_serving_loop(
    engine, explicit_destroy
):
    """RPC cleanup and lifespan fallback use the same ownership and ordering."""
    engine.config.agent.prm.scorers = [_spec("first"), _spec("disabled", enabled=False)]
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        assert ResourceScorer.instances == []
        assert not (await client.get("/health")).json()["initialized"]
        response = await _rpc(client, "initialize")
        assert response.status_code == 200
        assert response.json()["result"] == {"engine": "ready"}
        assert (await client.get("/health")).json()["initialized"]
        if explicit_destroy:
            assert (await _rpc(client, "destroy")).status_code == 200
            assert (await _rpc(client, "destroy")).status_code == 200
            assert not (await client.get("/health")).json()["initialized"]
    assert ResourceScorer.events == ["close:disabled", "close:first", "engine"]
    assert all(s.flushed and s.close_calls == 1 for s in ResourceScorer.instances)
    assert all(s.loop is s.close_loop for s in ResourceScorer.instances)
    engine.destroy.assert_called_once()
    assert srv._engine is None


@pytest.mark.asyncio
async def test_initialize_retries_reuse_engine_client_and_runner(engine):
    """Concurrent/repeated RPC retries preserve one successful initialization."""
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        first, second = await asyncio.gather(
            _rpc(client, "initialize", engine_id="worker"),
            _rpc(client, "initialize", engine_id="worker"),
        )
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json()
        runner, proxy_client = srv._prm_runner, srv._openai_client
        conflict = await _rpc(client, "initialize", engine_id="different-worker")
        assert conflict.status_code == 400
        assert srv._prm_runner is runner
        assert srv._openai_client is proxy_client
        assert len(ResourceScorer.instances) == 1
        engine.initialize.assert_called_once_with(engine_id="worker")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["fail_construct", "fail_validate"])
@pytest.mark.parametrize("fail_close", [False, True])
async def test_failed_scorer_setup_rolls_back_and_can_retry(
    engine, failure, fail_close
):
    """Original startup errors survive rollback; a retry does not rebuild the engine."""
    engine.config.agent.prm.scorers = [
        _spec("first", fail_close=fail_close),
        _spec("bad", **{failure: True}),
    ]
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        with pytest.raises(ValueError, match="scorer .* failed"):
            await _rpc(client, "initialize")
        assert srv._prm_runner is None
        assert srv._openai_client is None
        assert not (await client.get("/health")).json()["initialized"]
        assert all(s.close_calls == 1 for s in ResourceScorer.instances)
        _session()
        assert (await _export(client)).status_code == 503
        engine.config.agent.prm.scorers = [_spec("retry")]
        assert (await _rpc(client, "initialize")).status_code == 200
        assert (await _export(client)).status_code == 200
        engine.initialize.assert_called_once()
    assert ResourceScorer.instances[-1].flushed


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_close", [False, True])
async def test_client_setup_failure_closes_runner_without_publishing(
    engine, monkeypatch, fail_close
):
    """Failures after runner creation still roll back and preserve the setup error."""
    engine.config.agent.prm.scorers = [_spec(fail_close=fail_close)]
    monkeypatch.setattr(
        srv, "ArealOpenAI", MagicMock(side_effect=ValueError("client setup failed"))
    )
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        with pytest.raises(ValueError, match="client setup failed"):
            await _rpc(client, "initialize")
        assert srv._openai_client is srv._prm_runner is None
        assert ResourceScorer.instances[0].close_calls == 1
        monkeypatch.setattr(srv, "ArealOpenAI", MagicMock(return_value=object()))
        engine.config.agent.prm.scorers = [_spec("retry")]
        assert (await _rpc(client, "initialize")).status_code == 200
        engine.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_setup_rollback_blocks_readiness_and_serializes_destroy(
    engine, monkeypatch
):
    """Destroy waits for initialization rollback, and exports never skip missing PRM."""
    release = asyncio.Event()
    original_validate = ResourceScorer.validate_prm_config

    def delayed_failure(self, config, *, training_enabled):
        self.close_release = release
        self.fail_validate = True
        original_validate(self, config, training_enabled=training_enabled)

    monkeypatch.setattr(ResourceScorer, "validate_prm_config", delayed_failure)
    engine.config.agent.admin_api_key = "new-configured-key"
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        pending = asyncio.create_task(_rpc(client, "initialize"))
        scorer = await asyncio.wait_for(ResourceScorer.created.get(), timeout=2)
        await asyncio.wait_for(scorer.close_started.wait(), timeout=2)
        destroy = asyncio.create_task(_rpc(client, "destroy"))
        try:
            await asyncio.sleep(0)
            assert not destroy.done()
            engine.destroy.assert_not_called()
            assert srv._admin_api_key == _ADMIN_KEY
            assert srv._openai_client is srv._prm_runner is None
            assert not (await client.get("/health")).json()["initialized"]
            _session()
            assert (await _export(client)).status_code == 503
        finally:
            release.set()
            with pytest.raises(ValueError, match="scorer validation failed"):
                await pending
            assert (await destroy).status_code == 200
        assert scorer.flushed


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_destroy", [False, True])
async def test_shutdown_waits_for_async_scorer_flush(engine, explicit_destroy):
    """Neither shutdown entry point destroys the engine before the audit flush."""
    lifespan = srv.app.router.lifespan_context(srv.app)
    await lifespan.__aenter__()
    async with _client() as client:
        await _rpc(client, "initialize")
        scorer = ResourceScorer.instances[0]
        scorer.close_release = asyncio.Event()
        pending = asyncio.create_task(
            _rpc(client, "destroy")
            if explicit_destroy
            else lifespan.__aexit__(None, None, None)
        )
        try:
            await asyncio.wait_for(scorer.close_started.wait(), timeout=2)
            assert not pending.done()
            engine.destroy.assert_not_called()
        finally:
            scorer.close_release.set()
            await pending
            if explicit_destroy:
                await lifespan.__aexit__(None, None, None)
    assert scorer.flushed
    assert scorer.close_calls == 1
    engine.destroy.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_destroy", [False, True])
async def test_cleanup_failure_attempts_all_scorers_and_destroys_engine(
    engine, explicit_destroy
):
    """A failing scorer cannot suppress later cleanup or engine destruction."""
    engine.config.agent.prm.scorers = [_spec("good"), _spec("bad", fail_close=True)]
    lifespan = srv.app.router.lifespan_context(srv.app)
    await lifespan.__aenter__()
    async with _client() as client:
        await _rpc(client, "initialize")
        with pytest.raises(ExceptionGroup, match="PRM scorer cleanup failed"):
            if explicit_destroy:
                await _rpc(client, "destroy")
            else:
                await lifespan.__aexit__(None, None, None)
        assert ResourceScorer.events == ["close:bad", "close:good", "engine"]
        assert ResourceScorer.instances[0].flushed
        assert (await _rpc(client, "destroy")).status_code == 200
        if explicit_destroy:
            await lifespan.__aexit__(None, None, None)
    engine.destroy.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_drains_concurrent_scores_and_rejects_late_exports(engine):
    """Normal scoring is parallel; closing admission does not interrupt admitted work."""
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        await _rpc(client, "initialize")
        scorer = ResourceScorer.instances[0]
        scorer.score_release = asyncio.Event()
        for name in ("first", "second", "late"):
            _session(name)
        exports = [
            asyncio.create_task(_export(client, name)) for name in ("first", "second")
        ]
        destroy = None
        try:
            for _ in exports:
                await asyncio.wait_for(scorer.score_started.get(), timeout=2)
            assert scorer.active == srv._active_prm_exports == 2
            destroy = asyncio.create_task(_rpc(client, "destroy"))
            await asyncio.sleep(0)
            assert not destroy.done()
            assert scorer.close_calls == 0
            engine.destroy.assert_not_called()
            assert (await _export(client, "late")).status_code == 503
        finally:
            scorer.score_release.set()
            responses = await asyncio.gather(*exports)
            if destroy is not None:
                await destroy
        assert all(response.status_code == 200 for response in responses)
        for response in responses:
            interaction = deserialize_interactions(response.json()["interactions"])[
                "turn"
            ]
            torch.testing.assert_close(
                interaction.to_tensor_dict()["token_rewards"],
                torch.tensor([[0.0, 1.0]]),
                rtol=0,
                atol=0,
            )
        assert (await _export(client, "late")).status_code == 503
        assert srv._active_prm_exports == 0
        assert scorer.flushed


@pytest.mark.asyncio
async def test_unfinished_export_does_not_block_shutdown_or_bypass_prm(
    engine, monkeypatch
):
    """An export waiting for session completion must recheck scoring admission."""
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        await _rpc(client, "initialize")
        session = _session(finished=False)
        waiting, release = asyncio.Event(), asyncio.Event()

        async def wait_for_finish():
            waiting.set()
            await release.wait()

        monkeypatch.setattr(session, "wait_for_finish", wait_for_finish)
        pending = asyncio.create_task(_export(client))
        try:
            await asyncio.wait_for(waiting.wait(), timeout=2)
            assert (
                await asyncio.wait_for(_rpc(client, "destroy"), timeout=2)
            ).status_code == 200
            assert not pending.done()
        finally:
            release.set()
            response = await pending
        assert response.status_code == 503
        assert ResourceScorer.instances[0].score_started.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_score", [False, True])
async def test_failed_or_cancelled_scoring_releases_shutdown_waiter(
    engine, cancel_score
):
    """The scoring admission counter is released on failure and cancellation."""
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        await _rpc(client, "initialize")
        scorer = ResourceScorer.instances[0]
        scorer.fail_score = True
        scorer.score_release = asyncio.Event()
        _session()
        pending = asyncio.create_task(_export(client))
        await asyncio.wait_for(scorer.score_started.get(), timeout=2)
        destroy = asyncio.create_task(_rpc(client, "destroy"))
        await asyncio.sleep(0)
        if cancel_score:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            scorer.score_release.set()
            assert (await pending).json()["interactions"] == {}
        await asyncio.wait_for(destroy, timeout=2)
        assert srv._active_prm_exports == 0
        assert scorer.flushed


@pytest.mark.asyncio
async def test_self_cancelled_score_drains_siblings_before_shutdown(
    engine, monkeypatch
):
    """An internally cancelled turn cannot release resources used by another turn."""
    sibling_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    children = []

    async def evaluate(self, interaction, ctx):
        children.append(asyncio.current_task())
        self.active += 1
        try:
            if interaction.interaction_id == "turn":
                await sibling_started.wait()
                raise asyncio.CancelledError
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleanup_started.set()
                await release_cleanup.wait()
        finally:
            self.active -= 1

    monkeypatch.setattr(ResourceScorer, "evaluate", evaluate)
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        await _rpc(client, "initialize")
        scorer = ResourceScorer.instances[0]
        session = _session(finished=False)
        root = session.completions["turn"]
        leaf = InteractionWithTokenLogpReward(
            model_response=ModelResponse(
                input_tokens=[1, 2, 3],
                output_tokens=[4],
                output_logprobs=[0.0],
                output_versions=[0],
            ),
            parent=root,
            messages=root.messages
            + root.output_message_list
            + [{"role": "user", "content": "continue"}],
            output_message_list=[{"role": "assistant", "content": "done"}],
            reward=0.0,
            chat_template_type="concat",
        )
        leaf.interaction_id = "leaf"
        session.completions["leaf"] = leaf
        session.finish()
        pending = asyncio.create_task(_export(client))
        destroy = None
        try:
            await asyncio.wait_for(cleanup_started.wait(), timeout=2)
            assert not pending.done()
            assert srv._active_prm_exports == scorer.active == 1
            destroy = asyncio.create_task(_rpc(client, "destroy"))
            await asyncio.sleep(0)
            assert srv._proxy_closing
            assert not destroy.done()
            assert scorer.close_calls == 0
            engine.destroy.assert_not_called()
            release_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, timeout=2)
            assert (await asyncio.wait_for(destroy, timeout=2)).status_code == 200
            assert all(child.done() for child in children)
            assert srv._active_prm_exports == scorer.active == 0
            assert scorer.flushed and scorer.close_calls == 1
            assert root.token_rewards is leaf.token_rewards is None
            engine.destroy.assert_called_once()
        finally:
            release_cleanup.set()
            for task in [pending, *children]:
                if not task.done():
                    task.cancel()
            await asyncio.gather(pending, *children, return_exceptions=True)
            if destroy is not None:
                await asyncio.wait_for(destroy, timeout=2)


@pytest.mark.asyncio
async def test_cancelled_drain_leaves_engine_for_lifespan_cleanup(engine):
    """Cancelling a destroy waiter must not destroy resources underneath scoring."""
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        await _rpc(client, "initialize")
        scorer = ResourceScorer.instances[0]
        scorer.score_release = asyncio.Event()
        _session()
        pending = asyncio.create_task(_export(client))
        try:
            await asyncio.wait_for(scorer.score_started.get(), timeout=2)
            destroy = asyncio.create_task(_rpc(client, "destroy"))
            await asyncio.sleep(0)
            destroy.cancel()
            with pytest.raises(asyncio.CancelledError):
                await destroy
            engine.destroy.assert_not_called()
            assert scorer.close_calls == 0
            assert srv._proxy_closing
        finally:
            scorer.score_release.set()
            assert (await pending).status_code == 200
    assert scorer.flushed
    engine.destroy.assert_called_once()


@pytest.mark.asyncio
async def test_closed_proxy_rejects_initialization_and_engine_recreation(engine):
    """Clearing the engine at shutdown cannot enable reuse of stale init state."""
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        await _rpc(client, "initialize")
        await _rpc(client, "destroy")
        assert (await _rpc(client, "initialize")).status_code == 400
        response = await client.post("/create_engine", json={"engine": "unused.Engine"})
        assert response.status_code == 400
        assert len(ResourceScorer.instances) == 1


@pytest.mark.asyncio
async def test_borrowed_scorer_remains_caller_owned(engine):
    """The v1 integration preserves the shared runner's ownership rules."""
    borrowed = ResourceScorer("borrowed")
    engine.config.agent.prm.scorers = [borrowed, _spec()]
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        assert (await _rpc(client, "initialize")).status_code == 200
    assert borrowed.close_calls == 0
    assert ResourceScorer.instances[1].flushed
    await borrowed.aclose()


@pytest.mark.asyncio
async def test_engine_initialization_failure_is_not_cached(engine):
    """Only a successful engine call can be reused by later proxy-setup retries."""
    engine.initialize.side_effect = [ValueError("engine startup failed"), None]
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        with pytest.raises(ValueError, match="engine startup failed"):
            await _rpc(client, "initialize")
        assert srv._engine_init_request is None
        assert not (await client.get("/health")).json()["initialized"]
        assert ResourceScorer.instances == []
        assert (await _rpc(client, "initialize")).status_code == 200
        assert engine.initialize.call_count == 2


@pytest.mark.asyncio
async def test_concurrent_destroy_and_lifespan_wait_for_one_cleanup(engine):
    """Both entry points join a pending flush without repeating resource cleanup."""
    lifespan = srv.app.router.lifespan_context(srv.app)
    await lifespan.__aenter__()
    async with _client() as client:
        await _rpc(client, "initialize")
        scorer = ResourceScorer.instances[0]
        scorer.close_release = asyncio.Event()
        first = asyncio.create_task(_rpc(client, "destroy"))
        await asyncio.wait_for(scorer.close_started.wait(), timeout=2)
        repeated = asyncio.create_task(_rpc(client, "destroy"))
        fallback = asyncio.create_task(lifespan.__aexit__(None, None, None))
        try:
            await asyncio.sleep(0)
            assert not repeated.done()
            assert not fallback.done()
            engine.destroy.assert_not_called()
        finally:
            scorer.close_release.set()
            await asyncio.gather(first, repeated, fallback)
    assert scorer.flushed and scorer.close_calls == 1
    engine.destroy.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_path", ["rpc", "lifespan", "fallback"])
@pytest.mark.parametrize("failures", [1, 2])
async def test_engine_destroy_failure_preserves_engine_for_retry(
    engine, retry_path, failures
):
    """Every failed destroy remains visible and retriable without reclosing scorers."""
    engine.destroy.side_effect = ValueError("engine destroy failed")
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        await _rpc(client, "initialize")
        try:
            for _ in range(failures):
                with pytest.raises(ValueError, match="engine destroy failed"):
                    await _rpc(client, "destroy")
                assert srv._engine is engine
                assert ResourceScorer.instances[0].flushed
                assert ResourceScorer.instances[0].close_calls == 1
                assert not (await client.get("/health")).json()["initialized"]
        finally:
            engine.destroy.side_effect = None
            engine.destroy.return_value = None
        if retry_path == "rpc":
            assert (await _rpc(client, "destroy")).status_code == 200
        elif retry_path == "fallback":
            srv.cleanup_engine()
    assert engine.destroy.call_count == failures + 1
    assert srv._engine is None
    assert ResourceScorer.instances[0].close_calls == 1


@pytest.mark.asyncio
async def test_cancelled_cleanup_keeps_runner_cancellation_semantics(engine):
    """Cancellation propagates; interrupted scorer hooks are not retried at exit."""
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        await _rpc(client, "initialize")
        scorer = ResourceScorer.instances[0]
        scorer.close_release = asyncio.Event()
        pending = asyncio.create_task(_rpc(client, "destroy"))
        await asyncio.wait_for(scorer.close_started.wait(), timeout=2)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not scorer.flushed
        engine.destroy.assert_called_once()
    assert scorer.close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_scorers", [False, True])
async def test_disabled_or_empty_prm_keeps_unscored_export_behavior(
    engine, empty_scorers
):
    """PRM-disabled configurations do not instantiate scorers or block exports."""
    if empty_scorers:
        engine.config.agent.prm.scorers = []
    else:
        engine.config.agent.prm.enabled = False
    async with srv.app.router.lifespan_context(srv.app), _client() as client:
        assert (await _rpc(client, "initialize")).status_code == 200
        assert ResourceScorer.instances == []
        assert srv._prm_runner is None
        _session()
        response = await _export(client)
        assert response.status_code == 200
        interaction = deserialize_interactions(response.json()["interactions"])["turn"]
        assert "token_rewards" not in interaction.to_tensor_dict()
    engine.destroy.assert_called_once()

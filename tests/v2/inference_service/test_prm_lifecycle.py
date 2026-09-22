# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for scorer ownership and real data-proxy lifespan cleanup."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from areal.experimental.openai.cache import InteractionCache
from areal.reward.prm import BaseScorer, PRMConfig, PRMRunner, PRMScorerConfig
from areal.reward.prm import runner as prm_runner
from areal.v2.inference_service.data_proxy import app as data_proxy
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.session import SessionStore


class ResourceScorer(BaseScorer):
    """Importable configured scorer with observable, optionally delayed cleanup."""

    name = "resource"
    instances = []
    close_order = []

    def __init__(self, label="resource", fail=False, **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.fail = fail
        self.close_calls = 0
        self.started = asyncio.Event()
        self.release = None
        self.flushed = False
        self.instances.append(self)

    async def evaluate(self, interaction, ctx):
        return 1.0

    async def aclose(self):
        self.close_calls += 1
        self.close_order.append(self.label)
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.fail:
            raise ValueError(f"cannot flush {self.label}")
        self.flushed = True


@pytest.fixture(autouse=True)
def reset_scorer_state():
    ResourceScorer.instances.clear()
    ResourceScorer.close_order.clear()


def _spec(label="owned", *, fail=False, enabled=True):
    return PRMScorerConfig(
        path=f"{__name__}.ResourceScorer",
        enabled=enabled,
        kwargs={"label": label, "fail": fail},
    )


def _app(*specs, enabled=True):
    return data_proxy.create_app(
        DataProxyConfig(
            backend_addr="http://backend.invalid",
            tokenizer_path="mock-tokenizer",
            chat_template_type="concat",
            prm=PRMConfig(enabled=enabled, scorers=list(specs)),
        )
    )


@pytest.fixture
def services(monkeypatch):
    """Mock model/network resources, not the service lifecycle or scorer runner."""
    http_client = SimpleNamespace(aclose=AsyncMock())
    bridge = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(data_proxy, "create_httpx_client", lambda **kwargs: http_client)
    monkeypatch.setattr(data_proxy, "TokenizerProxy", MagicMock())
    monkeypatch.setattr(data_proxy, "_create_inf_bridge", lambda *args: bridge)
    monkeypatch.setattr(data_proxy, "_create_areal_client", MagicMock())
    return http_client, bridge


@pytest.mark.asyncio
async def test_base_scorer_cleanup_is_optional():
    """Existing computation-only scorers inherit an awaitable no-op hook."""
    from examples.prm.scorers import LengthBudgetScorer

    await LengthBudgetScorer(max_output_tokens=8).aclose()


@pytest.mark.asyncio
async def test_runner_closes_owned_scorers_but_not_borrowed_instances():
    """Only configured factories transfer ownership; disabled instances still close."""
    borrowed = ResourceScorer("borrowed")
    runner = PRMRunner(
        PRMConfig(scorers=[_spec("first"), borrowed, _spec("disabled", enabled=False)])
    )
    await runner.aclose()
    await runner.aclose()
    assert ResourceScorer.close_order == ["disabled", "first"]
    assert borrowed.close_calls == 0
    assert all(s.flushed for s in runner.scorers if s is not borrowed)
    with pytest.raises(RuntimeError, match="closed"):
        await runner.run(InteractionCache())
    # The external owner can still use and close its own instance.
    assert await borrowed.evaluate(None, {}) == 1.0
    await borrowed.aclose()


@pytest.mark.asyncio
async def test_runner_reports_failures_after_attempting_every_owned_scorer():
    """A failed flush cannot prevent other owned scorers from being closed."""
    runner = PRMRunner(
        PRMConfig(
            scorers=[
                _spec("good"),
                _spec("bad-1", fail=True),
                _spec("bad-2", fail=True),
            ]
        )
    )
    with pytest.raises(ExceptionGroup, match="PRM scorer cleanup") as caught:
        await runner.aclose()
    assert ResourceScorer.close_order == ["bad-2", "bad-1", "good"]
    assert [str(exc) for exc in caught.value.exceptions] == [
        "cannot flush bad-2",
        "cannot flush bad-1",
    ]
    assert runner.scorers[0].flushed
    await runner.aclose()
    assert all(s.close_calls == 1 for s in runner.scorers)


@pytest.mark.asyncio
async def test_concurrent_runner_closes_wait_for_one_cleanup():
    """Repeated callers wait for the in-progress close without repeating a flush."""
    runner = PRMRunner(PRMConfig(scorers=[_spec()]))
    scorer = runner.scorers[0]
    scorer.release = asyncio.Event()
    first = asyncio.create_task(runner.aclose())
    second = asyncio.create_task(runner.aclose())
    try:
        await asyncio.wait_for(scorer.started.wait(), timeout=2)
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()
        with pytest.raises(RuntimeError, match="closed"):
            await runner.run(InteractionCache())
    finally:
        scorer.release.set()
        await asyncio.gather(first, second)
    assert scorer.close_calls == 1
    assert scorer.flushed


@pytest.mark.asyncio
async def test_runner_cleanup_propagates_cancellation():
    """External cancellation is not converted into an ordinary cleanup error."""
    runner = PRMRunner(PRMConfig(scorers=[_spec()]))
    scorer = runner.scorers[0]
    scorer.release = asyncio.Event()
    pending = asyncio.create_task(runner.aclose())
    try:
        await asyncio.wait_for(scorer.started.wait(), timeout=2)
    finally:
        pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not scorer.flushed


@pytest.mark.asyncio
async def test_proxy_creates_and_closes_runner_with_its_lifespan(services):
    """Constructing an app acquires no scorer; each service lifetime owns one runner."""
    app = _app(_spec())
    assert ResourceScorer.instances == []
    for index in range(2):
        async with app.router.lifespan_context(app):
            assert len(ResourceScorer.instances) == index + 1
            scorer = ResourceScorer.instances[index]
            assert not scorer.flushed
        assert scorer.flushed
        assert scorer.close_calls == 1
    http_client, bridge = services
    assert http_client.aclose.await_count == 2
    assert bridge.aclose.await_count == 2


@pytest.mark.asyncio
async def test_proxy_shutdown_waits_for_audit_flush(services):
    """Shutdown cannot finish or close service clients before the scorer flushes."""
    app = _app(_spec())
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    scorer = ResourceScorer.instances[0]
    scorer.release = asyncio.Event()
    pending = asyncio.create_task(lifespan.__aexit__(None, None, None))
    try:
        await asyncio.wait_for(scorer.started.wait(), timeout=2)
        assert not pending.done()
        for resource in services:
            resource.aclose.assert_not_awaited()
    finally:
        scorer.release.set()
        await pending
    assert scorer.flushed
    for resource in services:
        resource.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_proxy_cleanup_failure_still_closes_other_resources(services):
    """Scorer failures are reported only after remaining scorers and clients close."""
    app = _app(_spec("good"), _spec("bad", fail=True))
    with pytest.raises(ExceptionGroup, match="PRM scorer cleanup"):
        async with app.router.lifespan_context(app):
            pass
    assert ResourceScorer.close_order == ["bad", "good"]
    assert ResourceScorer.instances[0].flushed
    for resource in services:
        resource.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_bridge_cleanup_failure_does_not_skip_http_client(
    services, caplog, monkeypatch
):
    """The last client closes even when both scorer and bridge cleanup fail."""
    http_client, bridge = services
    bridge.aclose.side_effect = RuntimeError("bridge close failed")
    app = _app(_spec(fail=True))
    # AReaL loggers may be attached to a different root than pytest's handler.
    monkeypatch.setattr(
        prm_runner.logger, "handlers", [*prm_runner.logger.handlers, caplog.handler]
    )
    with pytest.raises(RuntimeError, match="bridge close failed"):
        async with app.router.lifespan_context(app):
            pass
    # A later exit-stack callback can replace the propagated exception, but
    # the scorer failure must remain observable with its traceback.
    errors = [record for record in caplog.records if record.exc_info]
    assert any("cannot flush owned" in str(record.exc_info[1]) for record in errors)
    assert ResourceScorer.instances[0].close_calls == 1
    http_client.aclose.assert_awaited_once()
    bridge.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_export_cannot_silently_skip_uninitialized_prm(services):
    """An app served without startup must not export configured PRM data unscored."""
    app = _app(_spec())
    app.state.session_store = SessionStore()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/export_trajectories",
            json={"session_ids": ["session"], "style": "concat"},
            headers={"Authorization": "Bearer areal-admin-key"},
        )
    assert response.status_code == 503
    assert ResourceScorer.instances == []


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_proxy_without_active_prm_does_not_create_scorers(services, enabled):
    """Disabled PRM and empty configuration keep existing service behavior."""
    app = _app(*([] if enabled else [_spec()]), enabled=enabled)
    async with app.router.lifespan_context(app):
        assert ResourceScorer.instances == []
    for resource in services:
        resource.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_scorer_fails_startup_and_closes_service_resources(services):
    """A bad scorer import fails startup, not the first scoring request."""
    app = _app(PRMScorerConfig(path="missing_prm_test_module.Scorer"))
    with pytest.raises(ImportError):
        async with app.router.lifespan_context(app):
            pytest.fail("Invalid scorer must prevent startup")
    for resource in services:
        resource.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_proxy_exception_still_awaits_scorer_cleanup(services):
    """Unwinding the lifespan body also flushes a successfully created runner."""
    app = _app(_spec())
    with pytest.raises(RuntimeError, match="service failed"):
        async with app.router.lifespan_context(app):
            raise RuntimeError("service failed")
    assert ResourceScorer.instances[0].flushed
    for resource in services:
        resource.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_proxy_shutdown_closes_reconfigured_bridge(services, monkeypatch):
    """Shutdown follows the current bridge, not a stale startup-time reference."""
    app = _app(_spec())
    http_client, original_bridge = services
    replacement = SimpleNamespace(aclose=AsyncMock())
    async with app.router.lifespan_context(app):
        monkeypatch.setattr(data_proxy, "_create_inf_bridge", lambda *args: replacement)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/configure_backend",
                json={"backend_addr": "http://replacement.invalid"},
                headers={"Authorization": "Bearer areal-admin-key"},
            )
        assert response.status_code == 200
        original_bridge.aclose.assert_awaited_once()
        replacement.aclose.assert_not_awaited()
    original_bridge.aclose.assert_awaited_once()
    replacement.aclose.assert_awaited_once()
    http_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_factory_preserves_validation_and_ownership(monkeypatch):
    """Resolve once before validation, pass the original config, and own only factories."""
    borrowed = ResourceScorer("borrowed")
    config = PRMConfig(
        scorers=[asdict(_spec("first")), borrowed, _spec("disabled", enabled=False)]
    )
    validated = []

    def validate(scorer, received_config, *, training_enabled):
        assert received_config is config
        assert len(ResourceScorer.instances) == 3
        validated.append((scorer.label, training_enabled))

    monkeypatch.setattr(ResourceScorer, "validate_prm_config", validate, raising=False)
    runner = await PRMRunner.create(config)

    assert runner.config is config
    assert validated == [("first", True), ("borrowed", True), ("disabled", False)]
    await runner.aclose()
    await runner.aclose()
    assert ResourceScorer.close_order == ["disabled", "first"]
    assert borrowed.close_calls == 0
    assert len(ResourceScorer.instances) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["resolution", "construction", "validation"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_async_factory_rolls_back_without_replacing_startup_error(
    monkeypatch, phase, cleanup_fails
):
    """Every constructed owned scorer closes even if later setup and cleanup fail."""
    original_error = ValueError(f"{phase} failed")
    borrowed = ResourceScorer("borrowed")
    specs = [
        _spec("first"),
        borrowed,
        _spec("disabled", enabled=False, fail=cleanup_fails),
    ]
    if phase == "resolution":
        real_import = prm_runner.import_from_string

        def failing_import(path):
            if path == "missing.Scorer":
                raise original_error
            return real_import(path)

        monkeypatch.setattr(prm_runner, "import_from_string", failing_import)
        specs.append(PRMScorerConfig(path="missing.Scorer"))
    elif phase == "construction":
        real_init = ResourceScorer.__init__

        def failing_init(self, label, **kwargs):
            if label == "broken":
                raise original_error
            real_init(self, label=label, **kwargs)

        monkeypatch.setattr(ResourceScorer, "__init__", failing_init)
        specs.append(_spec("broken"))
    else:

        def failing_validation(self, config, *, training_enabled):
            # Even a failure in the first validator closes all constructed scorers.
            assert len(ResourceScorer.instances) == 3
            raise original_error

        monkeypatch.setattr(
            ResourceScorer, "validate_prm_config", failing_validation, raising=False
        )

    with pytest.raises(ValueError) as caught:
        await PRMRunner.create(PRMConfig(scorers=specs))

    assert caught.value is original_error
    assert ResourceScorer.close_order == ["disabled", "first"]
    assert borrowed.close_calls == 0
    assert ResourceScorer.instances[1].flushed
    assert all(
        s.close_calls == 1 for s in ResourceScorer.instances if s is not borrowed
    )
    if cleanup_fails:
        assert any("cleanup" in note for note in original_error.__notes__)


@pytest.mark.asyncio
async def test_async_factory_borrowed_validation_failure_only_closes_owned(monkeypatch):
    """A borrowed validator's failure does not transfer ownership to the runner."""
    borrowed = ResourceScorer("borrowed")
    error = ValueError("borrowed validation failed")
    monkeypatch.setattr(
        borrowed, "validate_prm_config", MagicMock(side_effect=error), raising=False
    )
    with pytest.raises(ValueError) as caught:
        await PRMRunner.create(PRMConfig(scorers=[_spec(), borrowed]))

    assert caught.value is error
    assert ResourceScorer.close_order == ["owned"]
    assert borrowed.close_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_cleanup", [False, True])
async def test_async_factory_waits_for_rollback_and_propagates_cancellation(
    monkeypatch, cancel_cleanup
):
    """Startup failure awaits audit flushing, but external cancellation still propagates."""
    started = asyncio.Event()
    release = asyncio.Event()
    original_error = ValueError("invalid configuration")

    def fail_validation(scorer, config, *, training_enabled):
        scorer.started = started
        scorer.release = release
        raise original_error

    monkeypatch.setattr(
        ResourceScorer, "validate_prm_config", fail_validation, raising=False
    )
    pending = asyncio.create_task(PRMRunner.create(PRMConfig(scorers=[_spec()])))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        assert not pending.done()
    finally:
        if cancel_cleanup:
            pending.cancel()
        release.set()

    if cancel_cleanup:
        with pytest.raises(asyncio.CancelledError):
            await pending
    else:
        with pytest.raises(ValueError) as caught:
            await pending
        assert caught.value is original_error
        assert ResourceScorer.instances[0].flushed


@pytest.mark.asyncio
async def test_async_factory_rolls_back_when_validation_raises_cancellation(
    monkeypatch,
):
    """A startup cancellation unwinds owned resources without changing its type."""
    error = asyncio.CancelledError("startup cancelled")
    monkeypatch.setattr(
        ResourceScorer,
        "validate_prm_config",
        MagicMock(side_effect=error),
        raising=False,
    )
    with pytest.raises(asyncio.CancelledError) as caught:
        await PRMRunner.create(PRMConfig(scorers=[_spec()]))

    assert caught.value is error
    assert ResourceScorer.instances[0].flushed
    assert ResourceScorer.instances[0].close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("phase", ["resolution", "validation"])
async def test_proxy_partial_startup_closes_all_resources_and_preserves_error(
    services, cleanup_fails, phase, caplog, monkeypatch
):
    """Scorer startup failures remain primary even when owned/service cleanup fails."""
    for logger in (prm_runner.logger, data_proxy.logger):
        monkeypatch.setattr(logger, "handlers", [*logger.handlers, caplog.handler])
    if cleanup_fails:
        for resource in services:
            resource.aclose.side_effect = RuntimeError("service cleanup failed")
    if phase == "resolution":
        last_spec = PRMScorerConfig(path="missing_prm_test.Scorer")
        error_type, message = ImportError, "missing_prm_test"
    else:
        last_spec = _spec("second")
        error_type, message = ValueError, "invalid PRM config"
        monkeypatch.setattr(
            ResourceScorer,
            "validate_prm_config",
            MagicMock(side_effect=ValueError(message)),
            raising=False,
        )
    app = _app(_spec(fail=cleanup_fails), last_spec)

    with pytest.raises(error_type, match=message) as caught:
        async with app.router.lifespan_context(app):
            pytest.fail("Partial PRM startup must not enter the service body")

    assert app.state.prm_runner is None
    assert all(s.close_calls == 1 for s in ResourceScorer.instances)
    for resource in services:
        resource.aclose.assert_awaited_once()
    if cleanup_fails:
        assert any("cleanup" in note for note in caught.value.__notes__)
        errors = [record for record in caplog.records if record.exc_info]
        assert any("cannot flush owned" in str(record.exc_info[1]) for record in errors)
        assert any(
            "service cleanup failed" in str(record.exc_info[1]) for record in errors
        )

"""A checkpoint drain must finish before any training or parent guard exits."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from areal.trainer.rl_trainer import PPOTrainer
from areal.v2.training_service.controller.controller import GatewayTrainController

MODULE = "areal.v2.training_service.controller.controller"


@pytest.mark.parametrize("failed", [False, True])
def test_drain_waits_for_all_ranks_before_returning(monkeypatch, failed):
    events = []
    controller = object.__new__(GatewayTrainController)
    controller.config = SimpleNamespace(request_timeout=3600)
    controller._worker_addrs = ["http://rank0", "http://rank1"]
    controller._async_checkpoint_pending = True

    class Response:
        def __init__(self, url):
            self.url = url

        async def __aenter__(self):
            if "rank0" in self.url:
                await asyncio.sleep(0.01)
            events.append(self.url)
            return self

        async def __aexit__(self, *args):
            pass

        def raise_for_status(self):
            if failed and "rank1" in self.url:
                raise RuntimeError("disk write failed")

    class Session:
        def __init__(self, timeout):
            assert timeout.total == 3600
            assert timeout.sock_connect == 30

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def post(self, url, json):
            return Response(url)

    monkeypatch.setattr(f"{MODULE}.aiohttp.ClientSession", Session)
    if failed:
        with pytest.raises(ExceptionGroup, match="checkpoint drain failed"):
            controller.wait_async_saves()
        assert controller._async_checkpoint_pending is True
    else:
        controller.wait_async_saves()
        assert controller._async_checkpoint_pending is False
        controller.wait_async_saves()  # idempotent
    assert sorted(events) == [f"http://rank{i}/wait_async_saves" for i in range(2)]


def test_cleanup_drain_failure_preserves_worker_addresses_and_guards():
    controller = object.__new__(GatewayTrainController)
    controller.wait_async_saves = Mock(side_effect=RuntimeError("drain failed"))
    controller._worker_addrs = ["http://rank0"]
    controller._graceful_shutdown_workers = Mock()
    controller._kill_forked_service = Mock()
    controller.scheduler = Mock()

    with pytest.raises(RuntimeError, match="drain failed"):
        controller._cleanup_runtime_state()

    controller._graceful_shutdown_workers.assert_not_called()
    controller._kill_forked_service.assert_not_called()
    controller.scheduler.delete_workers.assert_not_called()
    assert controller._worker_addrs == ["http://rank0"]


def test_trainer_close_drain_failure_does_not_destroy_parent_rollout():
    trainer = object.__new__(PPOTrainer)
    trainer.actor = object.__new__(GatewayTrainController)
    trainer.actor.wait_async_saves = Mock(side_effect=RuntimeError("drain failed"))
    trainer.actor.destroy = Mock()
    trainer.rollout = SimpleNamespace(destroy=Mock())

    with pytest.raises(RuntimeError, match="drain failed"):
        trainer.__exit__(None, None, None)

    trainer.actor.destroy.assert_not_called()
    trainer.rollout.destroy.assert_not_called()


def test_unreachable_rank_and_stalled_peer_fail_within_request_budget():
    from aiohttp import web

    async def check():
        release = asyncio.Event()

        async def stalled(request):
            await release.wait()
            return web.json_response({"status": "success"})

        async def failed(request):
            return web.Response(status=500)

        app = web.Application()
        app.router.add_post("/rank0/wait_async_saves", stalled)
        app.router.add_post("/rank1/wait_async_saves", failed)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        controller = object.__new__(GatewayTrainController)
        controller.config = SimpleNamespace(request_timeout=0.1)
        controller._worker_addrs = [
            f"http://127.0.0.1:{port}/rank{i}" for i in range(2)
        ]
        controller._async_checkpoint_pending = True
        try:
            with pytest.raises(ExceptionGroup) as error:
                await asyncio.wait_for(
                    asyncio.to_thread(controller.wait_async_saves), 3
                )
            assert len(error.value.exceptions) == 2
            assert controller._async_checkpoint_pending is True
        finally:
            release.set()
            await runner.cleanup()

    asyncio.run(check())


@pytest.mark.parametrize("colocate", [False, True])
@pytest.mark.parametrize("rpc_failed", [False, True])
def test_async_save_tracks_pending_even_when_rpc_fails(colocate, rpc_failed):
    from areal.api.io_struct import SaveLoadMeta
    from areal.infra.rpc.serialization import deserialize_value

    controller = object.__new__(GatewayTrainController)
    controller._colocate = colocate
    controller.config = SimpleNamespace(
        backend="megatron:d1", megatron=SimpleNamespace(async_save=True)
    )
    controller._gateway_post = Mock(
        side_effect=RuntimeError("lost response") if rpc_failed else None
    )
    meta = SaveLoadMeta(path="checkpoint", weight_format="dcp", with_optim=True)
    if rpc_failed:
        with pytest.raises(RuntimeError, match="lost response"):
            controller.save(meta)
    else:
        controller.save(meta)

    assert controller._async_checkpoint_pending is True
    route, payload = controller._gateway_post.call_args.args
    assert route == "/save"
    assert deserialize_value(payload["args"])[0] == meta

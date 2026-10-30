# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from areal.api.cli_args import BackendRoutingConfig
from areal.infra.backend_router import BackendRouterService
from areal.infra.controller.rollout_controller import RolloutController
from areal.trainer.rl_trainer import PPOTrainer
from areal.v2.inference_service.router.backend import RoutingError


def test_training_and_eval_share_pause_but_resume_only_the_selected_senders():
    events = []

    class Pool:
        control_seq = 0
        state = "READY"

        def set_state(self, state):
            self.control_seq += 1
            self.state = state
            events.append(("pool", state))

        def check_health(self):
            pass

    pool = Pool()
    members = []
    for role in ("train", "eval"):
        controller = object.__new__(RolloutController)
        controller._backend_router = pool
        controller._routing_active = True
        controller._routing_members = members
        controller._proxy_started = True
        controller._dispatcher = Mock()
        controller._collective_rpc = Mock(
            side_effect=lambda method, role=role, **kwargs: events.append(
                (role, method, kwargs)
            )
        )
        controller._proxy_collective_rpc = Mock(
            side_effect=lambda method, role=role, **kwargs: events.append(
                (role + "-proxy", method, kwargs)
            )
        )
        members.append(controller)

    train, evaluation = members
    train.pause()
    assert events[0] == ("pool", "PAUSED")
    for controller in members:
        controller._dispatcher.pause.assert_called_once()
        controller._collective_rpc.assert_any_call(
            "set_sender_state", paused=True, control_seq=1, http_timeout=60.0
        )
        controller._proxy_collective_rpc.assert_any_call(
            "set_sender_state", paused=True, control_seq=1, http_timeout=60.0
        )
    evaluation.resume()
    assert events[-1] == ("pool", "READY")
    train._collective_rpc.assert_any_call(
        "set_sender_state", paused=True, control_seq=2, http_timeout=60.0
    )
    evaluation._collective_rpc.assert_any_call(
        "set_sender_state", paused=False, control_seq=2, http_timeout=60.0
    )
    train._dispatcher.resume.assert_not_called()
    evaluation._dispatcher.resume.assert_called_once()


@pytest.mark.parametrize("failure", ["onload", "continue_generation"])
def test_failed_onload_never_opens_the_sender_fence(monkeypatch, failure):
    import areal.trainer.rl_trainer as trainer_module

    monkeypatch.setattr(
        trainer_module.stats_tracker,
        "record_timing",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        trainer_module.perf_tracer, "trace_scope", lambda *args, **kwargs: nullcontext()
    )
    rollout = Mock()
    getattr(rollout, failure).side_effect = RuntimeError("backend recovery failed")
    trainer = SimpleNamespace(rollout=rollout)
    with pytest.raises(RuntimeError, match="backend recovery failed"):
        PPOTrainer._onload_rollout(trainer)
    rollout.resume.assert_not_called()


def test_lost_control_reply_latches_failure_before_any_later_resume(tmp_path):
    manager = BackendRouterService(BackendRoutingConfig(), str(tmp_path), "rollout")
    # The READY transition reached the Router; its reply was lost.
    manager._request = Mock(
        side_effect=[
            RoutingError("ROUTER_UNREACHABLE"),
            {"control_seq": 4, "weight_version": 2},
            {},
        ]
    )
    with pytest.raises(RoutingError):
        manager.set_state("READY", version=2)
    assert manager.state == "FAILED"
    assert manager._request.call_args.args[2]["control_seq"] == 5
    assert manager._request.call_args.args[2]["state"] == "FAILED"
    with pytest.raises(RuntimeError, match="restart the whole experiment"):
        manager.set_state("READY")
    assert manager._request.call_count == 3
    manager._session.close()


@pytest.mark.parametrize("failed", [False, True])
def test_evaluation_opens_only_its_phase_and_always_closes_it(failed):
    events = []
    rollout = Mock()
    rollout.pause.side_effect = lambda: events.append("pause")
    rollout.resume.side_effect = lambda: events.append("resume")

    def collect(*args):
        events.append("evaluate")
        if failed:
            raise RuntimeError("evaluation failed")

    trainer = SimpleNamespace(
        config=SimpleNamespace(
            rollout=SimpleNamespace(backend_routing=BackendRoutingConfig(enabled=True))
        ),
        eval_rollout=rollout,
        _collect_eval_results=collect,
    )
    with pytest.raises(RuntimeError) if failed else nullcontext():
        PPOTrainer._evaluate_fn(trainer, "workflow", {})
    assert events == ["pause", "resume", "evaluate", "pause"]


@pytest.mark.slow
@pytest.mark.ci
def test_owned_router_subprocess_starts_publishes_and_exits(tmp_path, monkeypatch):
    import areal.infra.backend_router as manager_module

    monkeypatch.setattr(manager_module, "gethostip", lambda: "127.0.0.1")
    manager = BackendRouterService(
        BackendRoutingConfig(metadata_timeout=5, control_timeout=120),
        str(tmp_path),
        "rollout",
    )
    try:
        manager.start(
            [{"backend_id": "b0", "backend_addr": "http://127.0.0.1:9", "capacity": 24}]
        )
        manager.drain()
        manager.set_state("PAUSED", version=1)
        manager.set_state("READY")
        state = manager._request("GET", f"/routing/pools/{manager.pool_id}/state")
        assert state["state"] == "READY"
        assert state["weight_version"] == 1
        assert state["outstanding"] == 0
    finally:
        manager.destroy()
    assert manager.process.poll() is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_pause_drains_late_transport_before_native_gate_closes(synchronous):
    events = []
    controller = object.__new__(RolloutController)
    controller.pause = Mock(side_effect=lambda: events.append("fence"))
    controller.abort_all_requests = Mock(side_effect=lambda: events.append("abort"))

    def drain(*, abort_inflight):
        abort_inflight()
        events.append("drained")

    controller._backend_router = SimpleNamespace(state="READY", drain=drain)
    controller._collective_rpc = Mock(
        side_effect=lambda *a, **kw: events.append("native_pause")
    )
    controller._collective_rpc_async = AsyncMock(
        side_effect=lambda *a, **kw: events.append("native_pause")
    )
    if synchronous:
        controller.pause_generation_sync()
    else:
        await controller.pause_generation()
    assert events == ["fence", "abort", "drained", "native_pause"]


def test_drain_repeats_abort_until_late_request_releases(tmp_path, monkeypatch):
    import areal.infra.backend_router as manager_module

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        manager_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        ),
    )
    manager = BackendRouterService(BackendRoutingConfig(), str(tmp_path), "rollout")
    abort = Mock()
    manager._request = Mock(
        side_effect=lambda *args: {
            "state": "PAUSED",
            "outstanding": int(abort.call_count < 2),
        }
    )
    try:
        manager.drain(abort_inflight=abort)
        assert abort.call_count == 2
        assert clock.now >= 1.0
    finally:
        manager._session.close()

import asyncio
from types import SimpleNamespace

import areal.v2.training_service.controller.controller as controller_mod
from areal.v2.training_service.controller.controller import GatewayTrainController


def test_colocate_phase_offloads_and_restores_cuda_graph_in_order():
    events = []
    rollout = SimpleNamespace(
        pause=lambda: events.append("pause"),
        pause_generation_sync=lambda: events.append("drain"),
        offload=lambda tags: events.append(("offload", tuple(tags))),
        onload=lambda tags: events.append(("onload", tuple(tags))),
        resume=lambda: events.append("resume"),
        continue_generation=lambda: events.append("continue_generation"),
    )
    controller = SimpleNamespace(
        _colocate=True,
        rollout=rollout,
        _broadcast_awex_memory_op=lambda op, tags: events.append(
            ("actor", op, tuple(tags))
        ),
    )

    GatewayTrainController.enter_train_phase(controller)
    GatewayTrainController.exit_train_phase(controller)

    assert events == [
        "pause",
        "drain",
        ("offload", ("kv_cache",)),
        ("offload", ("cuda_graph",)),
        ("offload", ("weights",)),
        ("actor", "resume_memory", ("weights", "optimizer")),
        ("actor", "release_memory", ("weights", "optimizer")),
        ("onload", ("cuda_graph",)),
        ("onload", ("kv_cache",)),
        "continue_generation",
        "resume",
    ]


def _update_weights_events(colocate):
    """Record the rollout calls update_weights() makes around the transfer."""
    events = []
    rollout = SimpleNamespace(
        pause_generation=lambda: events.append("pause_generation"),
        continue_generation=lambda: events.append("continue_generation"),
    )
    controller = SimpleNamespace(
        _colocate=colocate,
        rollout=rollout,
        _weight_update_ctrl=SimpleNamespace(
            update_weights=lambda version: SimpleNamespace(status="ok", duration_ms=1.0)
        ),
    )

    GatewayTrainController.update_weights(controller, SimpleNamespace(version=1))
    return events


def test_colocate_update_weights_leaves_generation_paused():
    """train_phase owns resume: kv_cache is still released when transfer ends."""
    assert _update_weights_events(colocate=True) == ["pause_generation"]


def test_separation_update_weights_resumes_generation_itself():
    assert _update_weights_events(colocate=False) == [
        "pause_generation",
        "continue_generation",
    ]


def test_colocate_seed_delta_base_pauses_and_resumes_generation(monkeypatch):
    events = []
    rollout = SimpleNamespace(
        pause_generation=lambda: events.append("pause_generation"),
        continue_generation=lambda: events.append("continue_generation"),
    )
    controller = SimpleNamespace(
        _colocate=True,
        rollout=rollout,
        _weight_update_ctrl=object(),
        _async_seed_delta_base=lambda version: None,
    )

    def run_seed(_fn, version):
        events.append(("seed", version))

    monkeypatch.setattr(controller_mod, "run_async_task", run_seed)

    GatewayTrainController.seed_delta_base(controller, version=5)

    assert events == [
        "pause_generation",
        ("seed", 5),
        "continue_generation",
    ]


def test_seed_delta_base_skips_when_not_colocated(monkeypatch):
    events = []
    rollout = SimpleNamespace(
        pause_generation=lambda: events.append("pause_generation"),
        continue_generation=lambda: events.append("continue_generation"),
    )
    controller = SimpleNamespace(
        _colocate=False,
        rollout=rollout,
        _weight_update_ctrl=object(),
        _async_seed_delta_base=lambda version: None,
    )
    monkeypatch.setattr(
        controller_mod,
        "run_async_task",
        lambda _fn, _version: events.append("seed"),
    )

    GatewayTrainController.seed_delta_base(controller, version=5)

    assert events == []


def test_async_seed_delta_base_posts_to_train_and_inference_workers():
    events = []

    class Response:
        status_code = 200
        text = ""

    class Client:
        async def post(self, url, json, timeout):
            events.append((url, json, timeout))
            return Response()

    async def get_client():
        return Client()

    controller = SimpleNamespace(
        rollout=SimpleNamespace(inference_worker_urls=["http://inf-0"]),
        _worker_addrs=["http://train-0", "http://train-1"],
        config=SimpleNamespace(request_timeout=3.0),
        _get_async_client=get_client,
    )

    asyncio.run(GatewayTrainController._async_seed_delta_base(controller, version=9))

    assert events == [
        ("http://train-0/awex/seed_delta_base", {"version": 9}, 120.0),
        ("http://train-1/awex/seed_delta_base", {"version": 9}, 120.0),
        ("http://inf-0/awex/seed_delta_base", {"version": 9}, 120.0),
    ]

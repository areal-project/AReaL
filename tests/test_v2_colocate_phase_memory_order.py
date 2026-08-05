from types import SimpleNamespace

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

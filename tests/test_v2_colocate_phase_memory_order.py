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
        "resume",
    ]

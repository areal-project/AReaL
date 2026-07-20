import asyncio
from unittest.mock import Mock, patch

import pytest

pytest.importorskip("ray")

from areal.api import Worker  # noqa: E402
from areal.api.cli_args import BaseExperimentConfig  # noqa: E402
from areal.infra.scheduler import ray as ray_scheduler_mod  # noqa: E402
from areal.infra.scheduler.ray import RayScheduler, RayWorkerInfo  # noqa: E402


def _worker_info(worker_id: str, role: str) -> RayWorkerInfo:
    return RayWorkerInfo(
        worker=Worker(id=worker_id, ip="127.0.0.1", worker_ports=["12345"]),
        actor=Mock(),
        role=role,
        placement_group=Mock(),
        bundle_index=0,
        created_at=0.0,
    )


def test_fork_workers_with_command_uses_http_creator():
    scheduler = RayScheduler(exp_config=BaseExperimentConfig())
    target_wi = _worker_info("rollout/0", "rollout")
    scheduler._workers["rollout"] = [target_wi]

    command = "areal.experimental.openai.proxy.proxy_rollout_server"
    with patch.object(
        scheduler,
        "_create_managed_http_workers",
        return_value=["proxy-rollout/0"],
    ) as create_managed_http:
        worker_ids = scheduler.fork_workers(
            role="proxy-rollout",
            target_role="rollout",
            command=command,
        )

    assert worker_ids == ["proxy-rollout/0"]
    create_managed_http.assert_called_once_with(
        "proxy-rollout", "rollout", [target_wi], command
    )
    assert scheduler._colocated_roles["proxy-rollout"] == "rollout"


def test_fork_workers_without_command_keeps_ray_rpc_creator():
    scheduler = RayScheduler(exp_config=BaseExperimentConfig())
    target_wi = _worker_info("rollout/0", "rollout")
    scheduler._workers["rollout"] = [target_wi]

    with patch.object(
        scheduler,
        "_create_forked_workers_internal",
        return_value=["ref/0"],
    ) as create_ray_rpc:
        worker_ids = scheduler.fork_workers(role="ref", target_role="rollout")

    assert worker_ids == ["ref/0"]
    create_ray_rpc.assert_called_once()
    assert create_ray_rpc.call_args.args[0:3] == ("ref", "rollout", [target_wi])
    assert scheduler._colocated_roles["ref"] == "rollout"


def test_delete_target_role_deletes_http_child_role_first():
    scheduler = RayScheduler(exp_config=BaseExperimentConfig())
    target_wi = _worker_info("rollout/0", "rollout")
    proxy_wi = _worker_info("proxy-rollout/0", "proxy-rollout")
    proxy_wi.worker_kind = "http_server"
    proxy_wi.target_worker_id = "rollout/0"
    proxy_wi.target_node_id = "node-1"

    scheduler._workers["rollout"] = [target_wi]
    scheduler._workers["proxy-rollout"] = [proxy_wi]
    scheduler._colocated_roles["proxy-rollout"] = "rollout"

    with (
        patch.object(scheduler, "_cleanup_forked_workers") as cleanup_forked,
        patch.object(scheduler, "_cleanup_workers") as cleanup_workers,
    ):
        scheduler.delete_workers("rollout")

    cleanup_forked.assert_called_once_with([proxy_wi])
    cleanup_workers.assert_called_once_with([target_wi])
    assert "proxy-rollout" not in scheduler._workers
    assert "proxy-rollout" not in scheduler._colocated_roles
    assert "rollout" not in scheduler._workers


def test_create_engine_on_http_server_worker_posts_to_worker_endpoint():
    scheduler = RayScheduler(exp_config=BaseExperimentConfig())
    manager_actor = Mock()
    manager_actor.create_engine.remote = Mock(
        side_effect=AssertionError("manager must not proxy create_engine")
    )
    worker = RayWorkerInfo(
        worker=Worker(id="proxy-rollout/0", ip="10.0.0.8", worker_ports=["18080"]),
        actor=manager_actor,
        role="proxy-rollout",
        placement_group=Mock(),
        bundle_index=0,
        created_at=0.0,
        worker_kind="http_server",
        target_worker_id="rollout/0",
        target_node_id="node-1",
    )
    scheduler._workers["proxy-rollout"] = [worker]
    scheduler._worker_info_by_id[worker.worker.id] = worker

    post_calls = []

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def json(self):
            return {"result": "created"}

    class FakeSession:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, **kwargs):
            post_calls.append((url, kwargs))
            return FakeResponse()

    with patch.object(ray_scheduler_mod.aiohttp, "ClientSession", FakeSession):
        result = asyncio.run(
            scheduler.create_engine(
                "proxy-rollout/0",
                "tests.fake.Engine",
                engine_name="proxy-rollout/0",
                config={"dummy": True},
            )
        )

    assert result == "created"
    assert post_calls
    assert post_calls[0][0] == "http://10.0.0.8:18080/create_engine"
    manager_actor.create_engine.remote.assert_not_called()

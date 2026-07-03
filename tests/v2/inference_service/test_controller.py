"""Tests for RolloutControllerV2."""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import torch

from areal.api.cli_args import AgentConfig, InferenceEngineConfig
from areal.infra.rpc.rtensor import RTensor, TensorShardInfo
from areal.v2.inference_service.controller import workflow as workflow_module
from areal.v2.inference_service.controller.controller import (
    RolloutControllerV2,
)
from areal.v2.inference_service.controller.workflow import (
    InferenceServiceWorkflow,
)
from areal.v2.inference_service.data_proxy.session import TrajectoryDeliveryMode


def _make_scheduler(n_gpus_per_node: int = 8) -> MagicMock:
    scheduler = MagicMock()
    scheduler.n_gpus_per_node = n_gpus_per_node
    return scheduler


# =============================================================================
# InferenceEngineConfig
# =============================================================================


class TestInferenceEngineConfigForInferenceService:
    def test_defaults(self):
        cfg = InferenceEngineConfig(backend="sglang:d1")
        assert cfg.admin_api_key == "areal-admin-key"
        assert cfg.model == "default"
        assert cfg.consumer_batch_size == 1
        assert cfg.max_concurrent_rollouts is None
        assert cfg.max_head_offpolicyness == 0
        assert cfg.enable_rollout_tracing is False
        assert cfg.agent is not None
        assert (
            cfg.agent.agent_cls_path
            == "areal.experimental.openai.proxy.online_agent._OnlineAgent"
        )

    def test_custom_values(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="custom-key",
            consumer_batch_size=32,
            max_concurrent_rollouts=64,
            max_head_offpolicyness=5,
            agent=AgentConfig(
                agent_cls_path="tests.experimental.openai.utils.SimpleAgent",
                set_reward_finish_timeout=3.0,
            ),
        )
        assert cfg.admin_api_key == "custom-key"
        assert cfg.consumer_batch_size == 32
        assert cfg.max_concurrent_rollouts == 64
        assert cfg.max_head_offpolicyness == 5
        assert cfg.agent is not None
        assert cfg.agent.set_reward_finish_timeout == 3.0

    def test_scheduling_fields(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            request_timeout=60.0,
            setup_timeout=600.0,
        )
        assert cfg.request_timeout == 60.0
        assert cfg.setup_timeout == 600.0

    def test_dump_to_file_defaults_to_false(self):
        cfg = InferenceEngineConfig(backend="sglang:d1")
        assert cfg.dump_to_file is False


# =============================================================================
# RolloutControllerV2 — workflow resolution helpers
# =============================================================================


class TestControllerWorkflowResolution:
    def test_resolve_workflow_with_instance(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key"),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        with pytest.raises(TypeError, match=r"callable run\(\) method"):
            controller._resolve_workflow(12345)

    def test_resolve_workflow_none_creates_online_inference_service_workflow(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://test:8080"

        resolved = controller._resolve_workflow(
            None,
            workflow_kwargs={"timeout": 3.0},
        )

        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.controller is controller
        assert resolved.agent is None
        assert resolved.timeout == 3.0

    def test_resolve_workflow_agent_class_creates_offline_workflow(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://test:8080"

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        resolved = controller._resolve_workflow(
            MockAgent,
            workflow_kwargs={},
        )

        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.agent is not None
        assert isinstance(resolved.agent, MockAgent)

    def test_resolve_should_accept_fn_none(self):
        assert RolloutControllerV2._resolve_should_accept_fn(None) is None

    def test_resolve_should_accept_fn_callable(self):
        fn = lambda x: True  # noqa: E731
        assert RolloutControllerV2._resolve_should_accept_fn(fn) is fn

    def test_resolve_workflow_with_agent_class(self):
        """Test _resolve_workflow wraps agent-like classes in InferenceServiceWorkflow."""
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._gateway_addr = "http://test:8080"

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        resolved = controller._resolve_workflow(
            MockAgent,
            workflow_kwargs={},
        )
        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.agent is not None
        assert hasattr(resolved, "arun_episode")

    def test_resolve_workflow_agent_class_without_gateway_raises(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key"),
            scheduler=MagicMock(n_gpus_per_node=8),
        )

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        with pytest.raises(ValueError, match="Gateway address is unavailable"):
            controller._resolve_workflow(MockAgent, workflow_kwargs={})

    def test_resolve_workflow_rollout_workflow_instance_raises(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key"),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        controller._gateway_addr = "http://test:8080"

        workflow = InferenceServiceWorkflow(
            controller=controller,
            gateway_addr="http://test:8080",
        )

        with pytest.raises(
            TypeError,
            match="direct RolloutWorkflow instances are not supported",
        ):
            controller._resolve_workflow(workflow)

    def test_resolve_workflow_rollout_workflow_class_raises(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key"),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        controller._gateway_addr = "http://test:8080"

        with pytest.raises(
            TypeError,
            match="direct RolloutWorkflow classes are not supported",
        ):
            controller._resolve_workflow(
                "areal.v2.inference_service.controller.workflow.InferenceServiceWorkflow"
            )


class TestSubmitPolicyVersionSnapshot:
    @staticmethod
    def _make_controller(version: int = 7) -> RolloutControllerV2:
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key"),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        controller._gateway_addr = "http://test:8080"
        controller._workflow_executor = MagicMock()
        controller._version = version
        return controller

    def test_eval_submit_snapshots_current_policy_version(self):
        controller = self._make_controller(version=7)

        controller.submit(data={}, workflow=None, is_eval=True)

        resolved = controller.workflow_executor.submit.call_args.kwargs["workflow"]
        controller._version = 8
        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.expected_policy_version == 7

    def test_training_submit_has_no_expected_policy_version(self):
        controller = self._make_controller(version=7)

        controller.submit(data={}, workflow=None, is_eval=False)

        resolved = controller.workflow_executor.submit.call_args.kwargs["workflow"]
        assert resolved.expected_policy_version is None

    def test_workflow_kwargs_cannot_spoof_expected_policy_version(self):
        controller = self._make_controller(version=7)

        controller.submit(
            data={},
            workflow=None,
            workflow_kwargs={"expected_policy_version": 999},
            is_eval=True,
        )

        resolved = controller.workflow_executor.submit.call_args.kwargs["workflow"]
        assert resolved.expected_policy_version == 7


# =============================================================================
# RolloutControllerV2 — API surface
# =============================================================================


class TestRolloutControllerV2APISurface:
    def test_has_all_public_methods(self):
        methods = [
            "initialize",
            "destroy",
            "submit",
            "wait",
            "rollout_batch",
            "prepare_batch",
            "chat_completion",
            "set_version",
            "get_version",
            "get_capacity",
            "pause",
            "resume",
            "export_stats",
            "pause_generation",
            "continue_generation",
            "config_perf_tracer",
            "save_perf_tracer",
        ]
        for m in methods:
            assert hasattr(RolloutControllerV2, m), f"Missing method: {m}"

    def test_has_properties(self):
        properties = [
            "staleness_manager",
            "workflow_executor",
            "proxy_gateway_addr",
            "worker_ids",
        ]
        for p in properties:
            assert hasattr(RolloutControllerV2, p), f"Missing property: {p}"

    def test_not_subclass_of_rollout_controller(self):
        """RolloutControllerV2 must NOT be a subclass of RolloutController."""
        # Verify it doesn't inherit from any class except object
        bases = RolloutControllerV2.__bases__
        assert bases == (object,), f"Unexpected bases: {bases}"


# =============================================================================
# RolloutControllerV2 — construction + state
# =============================================================================


class TestRolloutControllerV2Construction:
    def test_admin_api_key_none_raises(self):
        cfg = InferenceEngineConfig(backend="sglang:d1")
        cfg.admin_api_key = ""
        with pytest.raises(ValueError, match="admin_api_key must be set"):
            RolloutControllerV2(config=cfg, scheduler=MagicMock(n_gpus_per_node=8))

    def test_model_empty_raises(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1", admin_api_key="test-key", model=""
        )
        with pytest.raises(ValueError, match="model must not be empty"):
            RolloutControllerV2(config=cfg, scheduler=MagicMock(n_gpus_per_node=8))

    def test_constructor(self):
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)

        assert controller.config is cfg
        assert controller.scheduler is scheduler
        assert controller.workers == []
        assert controller.server_infos == []
        assert controller.get_version() == 0
        assert controller.staleness_manager is None
        assert controller._worker_ids == {}
        assert controller.worker_ids == {}

    def test_admin_api_key_defaults(self):
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        assert controller.config.admin_api_key == "test-key"

    def test_version_management_without_services(self):
        """set_version / get_version work even without gateway services."""
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)

        # No gateway services started, but version management is local
        controller._version = 42
        assert controller.get_version() == 42

    def test_export_stats_returns_dict(self):
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        stats = controller.export_stats()
        assert isinstance(stats, dict)

    def test_proxy_gateway_addr(self):
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        # Before initialize, proxy_gateway_addr returns the empty _gateway_addr
        assert controller.proxy_gateway_addr == ""

    def test_callback_addr_formats_ipv6_hostport(self):
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._callback_host = "2001:db8::10"
        controller._callback_port = 19000

        assert controller.callback_addr == "[2001:db8::10]:19000"

    def test_workflow_executor_raises_before_init(self):
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        with pytest.raises(RuntimeError, match="initialize"):
            _ = controller.workflow_executor

    def test_config_perf_tracer_is_noop(self):
        cfg = InferenceEngineConfig(backend="sglang:d1", admin_api_key="test-key")
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        # Should not raise
        controller.config_perf_tracer()
        controller.save_perf_tracer()

    @pytest.mark.asyncio
    async def test_async_initialize_passes_callback_and_reward_timeout_to_data_proxy(
        self,
    ):
        from areal.api.cli_args import SchedulingSpec
        from areal.api.io_struct import LocalInfServerInfo

        worker = MagicMock()
        worker.ip = "127.0.0.1"
        worker.worker_ports = [18000]

        scheduler = MagicMock(n_gpus_per_node=8)
        scheduler.get_workers.return_value = [worker]

        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            tokenizer_path="mock-tokenizer",
            request_timeout=15.0,
            agent=AgentConfig(
                agent_cls_path="tests.experimental.openai.utils.SimpleAgent",
                set_reward_finish_timeout=7.5,
            ),
            scheduling_spec=(
                SchedulingSpec(
                    gpu=0,
                    cpu=1,
                    mem=1,
                    cmd="python -m areal.v2.inference_service.guard",
                ),
            ),
            admin_api_key="test-admin-key",
        )
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._callback_host = "127.0.0.1"
        controller._callback_port = 19000
        borrowed_server_info = LocalInfServerInfo(
            host="127.0.0.1", port=30000, process=MagicMock()
        )
        server_infos = [borrowed_server_info]

        with patch.object(controller, "_async_fork_on_guard") as mock_fork:
            mock_fork.side_effect = [
                ("127.0.0.1", 18081),
                ("127.0.0.1", 18082),
                ("127.0.0.1", 18080),
            ]

            await controller._async_initialize(
                server_args=None,
                server_infos=server_infos,
            )

        assert controller.server_infos == server_infos
        assert controller.server_infos is not server_infos
        controller.server_infos.clear()
        assert server_infos == [borrowed_server_info]

        data_proxy_calls = [
            c for c in mock_fork.call_args_list if c.kwargs.get("role") == "data-proxy"
        ]
        assert len(data_proxy_calls) == 1
        data_proxy_cmd = data_proxy_calls[0].kwargs["raw_cmd"]
        assert "--set-reward-finish-timeout" in data_proxy_cmd
        assert "7.5" in data_proxy_cmd
        assert "--callback-server-addr" in data_proxy_cmd
        assert "http://127.0.0.1:19000" in data_proxy_cmd


class TestOnlineCallbackFlow:
    @pytest.mark.asyncio
    async def test_online_callback_without_waiter_buffers_export_request(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._start_online_callback_server()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{controller.callback_addr}/callback/online_ready",
                    json={"session_id": "agent-a", "trajectory_id": 0},
                    headers={"Authorization": "Bearer test-admin-key"},
                )
            assert resp.status_code == 200
            buffered = await controller.wait_for_online_trajectory(timeout=1.0)
            assert buffered == {"session_id": "agent-a", "trajectory_id": 0}
        finally:
            controller._stop_online_callback_server()

    @pytest.mark.asyncio
    async def test_online_callback_settles_waiter_once(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._start_online_callback_server()

        waiter_task = asyncio.create_task(
            controller.wait_for_online_trajectory(timeout=1.0)
        )
        await asyncio.sleep(0)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{controller.callback_addr}/callback/online_ready",
                    json={"session_id": "agent-a", "trajectory_id": 0},
                    headers={"Authorization": "Bearer test-admin-key"},
                )
            assert resp.status_code == 200
            result = await waiter_task
            assert result == {"session_id": "agent-a", "trajectory_id": 0}
        finally:
            controller._stop_online_callback_server()

    @pytest.mark.asyncio
    async def test_online_callback_invalid_payload_keeps_waiter_pending(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._start_online_callback_server()

        waiter_task = asyncio.create_task(
            controller.wait_for_online_trajectory(timeout=1.0)
        )
        await asyncio.sleep(0)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{controller.callback_addr}/callback/online_ready",
                    json={"session_id": "agent-a"},
                    headers={"Authorization": "Bearer test-admin-key"},
                )
            assert resp.status_code == 425
            assert not waiter_task.done()
            waiter_task.cancel()
        finally:
            controller._stop_online_callback_server()

    @pytest.mark.asyncio
    async def test_cancelled_waiter_buffers_completed_online_result(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        )
        scheduler = MagicMock(n_gpus_per_node=8)
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._start_online_callback_server()

        waiter_task = asyncio.create_task(
            controller.wait_for_online_trajectory(timeout=1.0)
        )
        await asyncio.sleep(0)
        waiter_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_task

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"http://{controller.callback_addr}/callback/online_ready",
                    json={"session_id": "agent-a", "trajectory_id": 0},
                    headers={"Authorization": "Bearer test-admin-key"},
                )
            assert resp.status_code == 200

            buffered = await controller.wait_for_online_trajectory(timeout=1.0)
            assert buffered == {"session_id": "agent-a", "trajectory_id": 0}
        finally:
            controller._stop_online_callback_server()


class TestInferenceServiceWorkflow:
    @staticmethod
    def _versioned_trajectory(
        policy_version: int, reward: float = 1.25
    ) -> dict[str, torch.Tensor]:
        return {
            "rewards": torch.tensor([0.0, reward]),
            "versions": torch.tensor([-1, policy_version], dtype=torch.int32),
            "loss_mask": torch.tensor([0, 1], dtype=torch.int32),
        }

    @staticmethod
    def _versioned_rtensor_trajectory(
        policy_version: int, reward: float = 1.25
    ) -> dict[str, RTensor]:
        node_addr = "storage.test:9999"
        return {
            "rewards": RTensor(
                shard=TensorShardInfo(shard_id="rewards", node_addr=node_addr),
                data=torch.tensor([0.0, reward]),
            ),
            "versions": RTensor(
                shard=TensorShardInfo(shard_id="versions", node_addr=node_addr),
                data=torch.tensor([-1, policy_version], dtype=torch.int32),
            ),
            "loss_mask": RTensor(
                shard=TensorShardInfo(shard_id="loss-mask", node_addr=node_addr),
                data=torch.tensor([0, 1], dtype=torch.int32),
            ),
        }

    @pytest.mark.asyncio
    async def test_start_session_serializes_pull_delivery(self):
        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
        )
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "group_id": "grp",
                "sessions": [{"session_id": "s", "session_api_key": "k"}],
            }
        )
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_http_session = MagicMock()
        mock_http_session.post = MagicMock(return_value=mock_cm)

        result = await workflow._start_session(
            mock_http_session,
            "42",
            group_size=1,
            delivery_mode=TrajectoryDeliveryMode.PULL,
        )

        assert result == ("grp", [("s", "k")])
        mock_http_session.post.assert_called_once_with(
            "http://test:8080/rl/start_session",
            json={"task_id": "42", "group_size": 1, "delivery_mode": "pull"},
            headers={"Authorization": "Bearer test-key"},
        )

    @pytest.mark.skip(reason="pending /export_trajectories traj schema migration")
    @pytest.mark.asyncio
    async def test_online_mode_waits_on_controller(self):
        mock_interaction = MagicMock(reward=1.0)
        controller = MagicMock()
        controller.wait_for_online_trajectory = AsyncMock(
            return_value={"session_id": "sess-1", "trajectory_id": 7}
        )

        workflow = InferenceServiceWorkflow(
            controller=controller,
            agent=None,
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
            timeout=3.0,
        )

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
            patch(
                "areal.v2.inference_service.controller.workflow.deserialize_interactions"
            ) as mock_deserialize,
        ):
            mock_deserialize.return_value = {"chatcmpl-1": mock_interaction}

            # _run_online uses ``async with http_session.post(...)`` directly,
            # so the mock must support the async context-manager protocol.
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json = AsyncMock(
                return_value={"interactions": {"chatcmpl-1": {}}}
            )

            mock_cm = MagicMock()
            mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
            mock_cm.__aexit__ = AsyncMock(return_value=False)

            mock_http_session = MagicMock()
            mock_http_session.post = MagicMock(return_value=mock_cm)

            mock_wf_ctx.get_aiohttp_session = AsyncMock(return_value=mock_http_session)
            mock_wf_ctx.stat_scope.return_value = "rollout"
            mock_st.get.return_value = MagicMock()

            result = await workflow.arun_episode(engine=MagicMock(), data={})

        assert result is not None
        assert "chatcmpl-1" in result
        controller.wait_for_online_trajectory.assert_awaited_once_with(timeout=3.0)
        mock_http_session.post.assert_called_once()
        mock_deserialize.assert_called_once_with({"chatcmpl-1": {}})

    @pytest.mark.asyncio
    async def test_offline_mode_runs_agent(self):
        controller = MagicMock()

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        mock_interaction = MagicMock(reward=1.0)
        workflow = InferenceServiceWorkflow(
            controller=controller,
            agent=MockAgent(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
        )
        workflow._start_session = AsyncMock(
            return_value=("grp-test-1", [("sess-1", "sess-api-key-1")])
        )
        workflow._set_last_reward = AsyncMock(return_value=None)
        workflow._export_interactions = AsyncMock(
            return_value={"chatcmpl-1": mock_interaction}
        )

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
        ):
            mock_http_session = AsyncMock()
            mock_wf_ctx.get_aiohttp_session = AsyncMock(return_value=mock_http_session)
            mock_wf_ctx.get.return_value = MagicMock(task_id=42)
            mock_wf_ctx.get_httpx_client = AsyncMock(return_value=MagicMock())
            mock_wf_ctx.stat_scope.return_value = "rollout"
            mock_st.get.return_value = MagicMock()

            result = await workflow.arun_episode(engine=MagicMock(), data={})

        assert result is not None
        assert "chatcmpl-1" in result
        workflow._start_session.assert_awaited_once_with(
            mock_http_session,
            "42",
            group_size=1,
            delivery_mode=TrajectoryDeliveryMode.PULL,
        )
        workflow._set_last_reward.assert_awaited_once()
        workflow._export_interactions.assert_awaited_once_with(
            mock_http_session, ["sess-1"], group_id="grp-test-1"
        )

    @pytest.mark.asyncio
    async def test_offline_mode_records_reward_with_expected_policy_version(self):
        class MockAgent:
            async def run(self, data, **kwargs):
                self.observed_policy_version = kwargs.get("policy_version")
                return 1.25

        agent = MockAgent()
        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            agent=agent,
            gateway_addr="http://test:8080",
            expected_policy_version=7,
        )
        workflow._start_session = AsyncMock(
            return_value=("group-1", [("session-1", "key-1")])
        )
        workflow._set_last_reward = AsyncMock(return_value=None)
        traj = self._versioned_rtensor_trajectory(7)
        workflow._export_interactions = AsyncMock(return_value=traj)
        tracker = MagicMock()
        to_thread = AsyncMock(side_effect=lambda fn, *args: fn(*args))
        clear_node = AsyncMock()

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
            patch.object(workflow_module.asyncio, "to_thread", to_thread),
            patch.object(RTensor, "clear_node", clear_node),
        ):
            mock_wf_ctx.get.return_value = MagicMock(task_id=42)
            mock_wf_ctx.get_httpx_client = AsyncMock(return_value=MagicMock())
            mock_wf_ctx.stat_scope.return_value = "eval-rollout"
            mock_st.get.return_value = tracker

            result = await workflow._run_offline(MagicMock(), {})

        assert result is not None
        assert agent.observed_policy_version == 7
        to_thread.assert_awaited_once_with(
            workflow_module.validate_trajectory_policy_version,
            traj,
            7,
        )
        clear_node.assert_not_awaited()
        tracker.scalar.assert_called_once_with(reward=1.25, policy_version=7)

    @pytest.mark.asyncio
    async def test_offline_mode_rejects_version_before_recording_metrics(self):
        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.25

        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            agent=MockAgent(),
            gateway_addr="http://test:8080",
            expected_policy_version=7,
        )
        workflow._start_session = AsyncMock(
            return_value=("group-1", [("session-1", "key-1")])
        )
        workflow._set_last_reward = AsyncMock(return_value=None)
        traj = self._versioned_rtensor_trajectory(6)
        workflow._export_interactions = AsyncMock(return_value=traj)
        tracker = MagicMock()
        clear_node = AsyncMock()

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
            patch.object(RTensor, "clear_node", clear_node),
        ):
            mock_wf_ctx.get.return_value = MagicMock(task_id=42)
            mock_wf_ctx.get_httpx_client = AsyncMock(return_value=MagicMock())
            mock_wf_ctx.stat_scope.return_value = "eval-rollout"
            mock_st.get.return_value = tracker

            with pytest.raises(ValueError, match="expected policy version 7"):
                await workflow._run_offline(MagicMock(), {})

        tracker.scalar.assert_not_called()
        clear_node.assert_awaited_once_with(
            "storage.test:9999",
            ["rewards", "versions", "loss-mask"],
        )

    @pytest.mark.asyncio
    async def test_offline_mode_clears_exported_trajectory_when_group_fails(self):
        class FailingAgent:
            async def run(self, data, **kwargs):
                raise RuntimeError("agent failed")

        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            agent=FailingAgent(),
            gateway_addr="http://test:8080",
        )
        workflow._start_session = AsyncMock(
            return_value=("group-1", [("session-1", "key-1")])
        )
        workflow._set_last_reward = AsyncMock(return_value=None)
        traj = self._versioned_rtensor_trajectory(7)
        workflow._export_interactions = AsyncMock(return_value=traj)
        clear_node = AsyncMock()

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch.object(RTensor, "clear_node", clear_node),
        ):
            mock_wf_ctx.get.return_value = MagicMock(task_id=42)
            mock_wf_ctx.get_httpx_client = AsyncMock(return_value=MagicMock())

            result = await workflow._run_offline(MagicMock(), {})

        assert result is None
        clear_node.assert_awaited_once_with(
            "storage.test:9999",
            ["rewards", "versions", "loss-mask"],
        )

    @pytest.mark.asyncio
    async def test_offline_mode_without_expectation_accepts_missing_versions(self):
        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.25

        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            agent=MockAgent(),
            gateway_addr="http://test:8080",
            expected_policy_version=None,
        )
        workflow._start_session = AsyncMock(
            return_value=("group-1", [("session-1", "key-1")])
        )
        workflow._set_last_reward = AsyncMock(return_value=None)
        workflow._export_interactions = AsyncMock(return_value={"input_ids": []})
        tracker = MagicMock()

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
        ):
            mock_wf_ctx.get.return_value = MagicMock(task_id=42)
            mock_wf_ctx.get_httpx_client = AsyncMock(return_value=MagicMock())
            mock_wf_ctx.stat_scope.return_value = "rollout"
            mock_st.get.return_value = tracker

            result = await workflow._run_offline(MagicMock(), {})

        assert result is not None
        tracker.scalar.assert_called_once_with(reward=1.25)

    @pytest.mark.asyncio
    async def test_online_mode_records_reward_with_expected_policy_version(self):
        controller = MagicMock()
        controller.wait_for_online_trajectory = AsyncMock(
            return_value={"session_id": "session-1", "trajectory_id": 3}
        )
        workflow = InferenceServiceWorkflow(
            controller=controller,
            gateway_addr="http://test:8080",
            expected_policy_version=7,
        )
        traj = self._versioned_rtensor_trajectory(7)
        workflow._export_interactions = AsyncMock(return_value=traj)
        tracker = MagicMock()
        to_thread = AsyncMock(side_effect=lambda fn, *args: fn(*args))
        clear_node = AsyncMock()

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
            patch.object(workflow_module.asyncio, "to_thread", to_thread),
            patch.object(RTensor, "clear_node", clear_node),
        ):
            mock_wf_ctx.stat_scope.return_value = "eval-rollout"
            mock_st.get.return_value = tracker

            result = await workflow._run_online(MagicMock())

        assert result is not None
        to_thread.assert_awaited_once_with(
            workflow_module.validate_trajectory_policy_version,
            traj,
            7,
        )
        clear_node.assert_not_awaited()
        tracker.scalar.assert_called_once_with(reward=1.25, policy_version=7)

    @pytest.mark.asyncio
    async def test_online_mode_rejects_version_before_recording_metrics(self):
        controller = MagicMock()
        controller.wait_for_online_trajectory = AsyncMock(
            return_value={"session_id": "session-1", "trajectory_id": 3}
        )
        workflow = InferenceServiceWorkflow(
            controller=controller,
            gateway_addr="http://test:8080",
            expected_policy_version=7,
        )
        traj = self._versioned_rtensor_trajectory(6)
        workflow._export_interactions = AsyncMock(return_value=traj)
        tracker = MagicMock()
        clear_node = AsyncMock()

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
            patch.object(RTensor, "clear_node", clear_node),
        ):
            mock_wf_ctx.stat_scope.return_value = "eval-rollout"
            mock_st.get.return_value = tracker

            with pytest.raises(ValueError, match="expected policy version 7"):
                await workflow._run_online(MagicMock())

        tracker.scalar.assert_not_called()
        clear_node.assert_awaited_once_with(
            "storage.test:9999",
            ["rewards", "versions", "loss-mask"],
        )

    @pytest.mark.asyncio
    async def test_online_mode_clears_exported_trajectory_when_reward_is_missing(self):
        controller = MagicMock()
        controller.wait_for_online_trajectory = AsyncMock(
            return_value={"session_id": "session-1", "trajectory_id": 3}
        )
        workflow = InferenceServiceWorkflow(
            controller=controller,
            gateway_addr="http://test:8080",
        )
        traj = {
            "input_ids": RTensor(
                shard=TensorShardInfo(
                    shard_id="input-ids", node_addr="storage.test:9999"
                ),
                data=torch.tensor([1, 2], dtype=torch.int64),
            )
        }
        workflow._export_interactions = AsyncMock(return_value=traj)
        clear_node = AsyncMock()

        with patch.object(RTensor, "clear_node", clear_node):
            result = await workflow._run_online(MagicMock())

        assert result is None
        clear_node.assert_awaited_once_with(
            "storage.test:9999",
            ["input-ids"],
        )

    @pytest.mark.asyncio
    async def test_online_mode_without_expectation_accepts_missing_versions(self):
        controller = MagicMock()
        controller.wait_for_online_trajectory = AsyncMock(
            return_value={"session_id": "session-1", "trajectory_id": 3}
        )
        workflow = InferenceServiceWorkflow(
            controller=controller,
            gateway_addr="http://test:8080",
            expected_policy_version=None,
        )
        workflow._export_interactions = AsyncMock(
            return_value={"rewards": torch.tensor([0.0, 1.25])}
        )
        tracker = MagicMock()

        with (
            patch(
                "areal.v2.inference_service.controller.workflow.workflow_context"
            ) as mock_wf_ctx,
            patch(
                "areal.v2.inference_service.controller.workflow.stats_tracker"
            ) as mock_st,
        ):
            mock_wf_ctx.stat_scope.return_value = "rollout"
            mock_st.get.return_value = tracker

            result = await workflow._run_online(MagicMock())

        assert result is not None
        tracker.scalar.assert_called_once_with(reward=1.25)


class TestValidateTrajectoryPolicyVersion:
    @staticmethod
    def _trajectory(
        versions: list[int], loss_mask: list[int]
    ) -> dict[str, torch.Tensor]:
        return {
            "versions": torch.tensor(versions, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
        }

    def test_validate_trajectory_policy_version_accepts_expected_loss_tokens(self):
        traj = self._trajectory(
            versions=[-1, -1, 7, 7],
            loss_mask=[0, 0, 1, 1],
        )

        workflow_module.validate_trajectory_policy_version(traj, 7)

    def test_validate_trajectory_policy_version_ignores_untrained_positions(self):
        traj = self._trajectory(
            versions=[999, -1, 7, 123],
            loss_mask=[0, 0, 1, 0],
        )

        workflow_module.validate_trajectory_policy_version(traj, 7)

    @pytest.mark.parametrize(
        "versions, observed",
        [
            ([-1, 7, 6], "[6, 7]"),
            ([-1, 6, 6], "[6]"),
        ],
        ids=["mixed", "all-stale"],
    )
    def test_validate_trajectory_policy_version_rejects_mixed_or_stale_versions(
        self, versions: list[int], observed: str
    ):
        traj = self._trajectory(versions=versions, loss_mask=[0, 1, 1])

        with pytest.raises(
            ValueError,
            match=rf"expected policy version 7.*observed {re.escape(observed)}",
        ):
            workflow_module.validate_trajectory_policy_version(traj, 7)

    @pytest.mark.parametrize("missing_key", ["versions", "loss_mask"])
    def test_validate_trajectory_policy_version_rejects_missing_field(
        self, missing_key: str
    ):
        traj = self._trajectory(versions=[-1, 7], loss_mask=[0, 1])
        del traj[missing_key]

        with pytest.raises(ValueError, match=rf"missing.*{missing_key}"):
            workflow_module.validate_trajectory_policy_version(traj, 7)

    def test_validate_trajectory_policy_version_rejects_shape_mismatch(self):
        traj = self._trajectory(versions=[-1, 7, 7], loss_mask=[0, 1])

        with pytest.raises(ValueError, match="same shape"):
            workflow_module.validate_trajectory_policy_version(traj, 7)

    def test_validate_trajectory_policy_version_rejects_floating_versions(self):
        traj = {
            "versions": torch.tensor([-1.0, 7.0], dtype=torch.float32),
            "loss_mask": torch.tensor([0, 1], dtype=torch.int32),
        }

        with pytest.raises(ValueError, match="signed integer dtype"):
            workflow_module.validate_trajectory_policy_version(traj, 7)

    def test_validate_trajectory_policy_version_rejects_no_loss_tokens(self):
        traj = self._trajectory(versions=[-1, 7], loss_mask=[0, 0])

        with pytest.raises(ValueError, match="no loss-bearing tokens"):
            workflow_module.validate_trajectory_policy_version(traj, 7)

    def test_validate_trajectory_policy_version_accepts_local_rtensors(self):
        traj = {
            "versions": RTensor(
                shard=TensorShardInfo(shard_id="versions", node_addr="unused"),
                data=torch.tensor([-1, 7, 7], dtype=torch.int32),
            ),
            "loss_mask": RTensor(
                shard=TensorShardInfo(shard_id="loss-mask", node_addr="unused"),
                data=torch.tensor([0, 1, 1], dtype=torch.int32),
            ),
        }

        with patch(
            "areal.infra.rpc.rtensor.get_backend",
            side_effect=AssertionError("local RTensor must not fetch"),
        ):
            workflow_module.validate_trajectory_policy_version(traj, 7)


# =============================================================================
# Multi-node inference configuration
# =============================================================================


class TestMultiNodeConfig:
    def test_scheduler_zero_gpus_raises(self):
        cfg = InferenceEngineConfig(backend="sglang:d1t8", admin_api_key="test-key")
        scheduler = _make_scheduler()
        scheduler.n_gpus_per_node = 0
        with pytest.raises(ValueError, match="n_gpus_per_node must be >= 1"):
            RolloutControllerV2(config=cfg, scheduler=MagicMock(n_gpus_per_node=0))

    def test_gpus_not_divisible_raises(self):
        cfg = InferenceEngineConfig(backend="sglang:d1t8", admin_api_key="test-key")
        scheduler = _make_scheduler()
        scheduler.n_gpus_per_node = 3
        with pytest.raises(ValueError, match="must be divisible by n_gpus_per_node"):
            RolloutControllerV2(config=cfg, scheduler=MagicMock(n_gpus_per_node=3))

    def test_single_node_backward_compat(self):
        cfg = InferenceEngineConfig(backend="sglang:d2t4", admin_api_key="test-key")
        controller = RolloutControllerV2(
            config=cfg, scheduler=MagicMock(n_gpus_per_node=8)
        )
        assert controller._nnodes_per_instance == 1

    def test_multi_node_valid_config(self):
        # tp=16, n_gpus_per_node=8 → nnodes_per_instance=2
        cfg = InferenceEngineConfig(backend="sglang:d1t16", admin_api_key="test-key")
        controller = RolloutControllerV2(
            config=cfg, scheduler=MagicMock(n_gpus_per_node=8)
        )
        assert controller._nnodes_per_instance == 2

    @pytest.mark.asyncio
    async def test_async_initialize_multinode_worker_count(self):
        """With multi-node and pre-existing server_infos, should create dp_size workers."""
        from areal.api.cli_args import SchedulingSpec
        from areal.api.io_struct import LocalInfServerInfo

        worker0 = MagicMock()
        worker0.ip = "10.0.0.1"
        worker0.worker_ports = [18000]
        worker0.id = "w0"

        worker1 = MagicMock()
        worker1.ip = "10.0.0.2"
        worker1.worker_ports = [18000]
        worker1.id = "w1"

        scheduler = MagicMock(n_gpus_per_node=4)
        scheduler.get_workers.return_value = [worker0]

        # tp=8, n_gpus_per_node=4 → nnodes_per_instance=2
        cfg = InferenceEngineConfig(
            tokenizer_path="mock-tokenizer",
            backend="sglang:d1t8",
            scheduling_spec=(SchedulingSpec(gpu=1, cpu=1, mem=1, cmd="mock"),),
            admin_api_key="test-key",
        )
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._callback_host = "127.0.0.1"
        controller._callback_port = 19000

        with patch.object(controller, "_async_fork_on_guard") as mock_fork:
            mock_fork.side_effect = [
                ("127.0.0.1", 18081),  # router
                ("127.0.0.1", 18082),  # data proxy (only 1, on head)
                ("127.0.0.1", 18080),  # gateway
            ]

            await controller._async_initialize(
                server_args=None,
                server_infos=[
                    LocalInfServerInfo(
                        host="10.0.0.1", port=30000, process=MagicMock()
                    ),
                ],
            )

        # With server_infos, total_workers = dp_size = 1 (not dp_size * nnodes_per_instance)
        create_call = scheduler.create_workers.call_args
        job = create_call.kwargs.get("job") or create_call.args[0]
        assert job.replicas == 1

        # 3 forks: router + data-proxy + gateway (all on head worker)
        assert mock_fork.call_count == 3
        data_proxy_calls = [
            c for c in mock_fork.call_args_list if c.kwargs.get("role") == "data-proxy"
        ]
        assert len(data_proxy_calls) == 1

    @pytest.mark.asyncio
    async def test_async_initialize_multinode_fork_path(self):
        """Exercise the full multi-node fork path (server_infos=None)."""
        from areal.api.cli_args import SchedulingSpec

        worker0 = MagicMock()
        worker0.ip = "10.0.0.1"
        worker0.worker_ports = [18000]
        worker0.id = "w0"

        worker1 = MagicMock()
        worker1.ip = "10.0.0.2"
        worker1.worker_ports = [18000]
        worker1.id = "w1"

        scheduler = MagicMock(n_gpus_per_node=4)
        scheduler.get_workers.return_value = [worker0, worker1]

        # tp=8, n_gpus_per_node=4 → nnodes_per_instance=2
        cfg = InferenceEngineConfig(
            tokenizer_path="mock-tokenizer",
            backend="sglang:d1t8",
            scheduling_spec=(SchedulingSpec(gpu=1, cpu=1, mem=1, cmd="mock"),),
            admin_api_key="test-key",
        )
        controller = RolloutControllerV2(config=cfg, scheduler=scheduler)
        controller._callback_host = "127.0.0.1"
        controller._callback_port = 19000

        # Track async client .post calls to /alloc_ports and /fork
        alloc_port_counter = 0
        fork_calls = []

        async def mock_async_post(url, json=None, timeout=None):
            nonlocal alloc_port_counter
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "/alloc_ports" in url:
                alloc_port_counter += 1
                resp.json.return_value = {
                    "status": "success",
                    "host": url.split("//")[1].split(":")[0],
                    "ports": [30000 + alloc_port_counter],
                }
            elif "/fork" in url:
                fork_calls.append(json)
                resp.json.return_value = {"status": "success"}
            return resp

        mock_async_client = AsyncMock()
        mock_async_client.post = mock_async_post

        with (
            patch.object(
                controller, "_get_async_client", return_value=mock_async_client
            ),
            patch.object(controller, "_async_fork_on_guard") as mock_fork,
            patch.object(controller, "_async_wait_for_service"),
            patch(
                "areal.api.cli_args.pkg_version.is_version_greater_or_equal",
                return_value=True,
            ),
            patch("areal.api.cli_args.pkg_version.is_version_less", return_value=False),
        ):
            mock_fork.side_effect = [
                ("10.0.0.1", 18081),  # router
                ("10.0.0.1", 18082),  # data proxy
                ("10.0.0.1", 18080),  # gateway
            ]

            await controller._async_initialize(
                server_args=None,
                server_infos=None,
            )

        # dp_size=1, nnodes_per_instance=2: total_workers = 2
        create_call = scheduler.create_workers.call_args
        job = create_call.kwargs.get("job") or create_call.args[0]
        assert job.replicas == 2

        # Async client .post calls for inf server fork:
        # 1 rendezvous alloc (nnodes_per_instance > 1) + 2 node allocs + 2 forks = 5
        assert alloc_port_counter == 3  # 1 rendezvous + 2 per-node
        assert len(fork_calls) == 2  # 1 per node in the group

        # Verify fork payloads have correct worker_index and role
        assert fork_calls[0]["role"] == "inf-server"
        assert fork_calls[0]["worker_index"] == 0
        assert fork_calls[1]["role"] == "inf-server"
        assert fork_calls[1]["worker_index"] == 1

        # Verify dist_init_addr propagated to fork commands
        for fc in fork_calls:
            cmd_str = " ".join(fc["raw_cmd"])
            assert "--dist-init-addr" in cmd_str or "--dist_init_addr" in cmd_str

        # Only 1 data proxy (dp_size=1, on head worker only)
        data_proxy_calls = [
            c for c in mock_fork.call_args_list if c.kwargs.get("role") == "data-proxy"
        ]
        assert len(data_proxy_calls) == 1

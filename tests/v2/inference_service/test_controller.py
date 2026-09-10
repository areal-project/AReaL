"""Tests for RolloutControllerV2."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest

from areal.api.cli_args import AgentConfig, InferenceEngineConfig
from areal.utils import stats_tracker
from areal.v2.inference_service.controller.controller import (
    RolloutControllerV2,
)
from areal.v2.inference_service.controller.workflow import (
    InferenceServiceWorkflow,
)


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

    def test_deterministic_sampling_with_offpolicy_head_warns(self):
        with patch("areal.api.cli_args.logger") as mock_logger:
            InferenceEngineConfig(
                backend="sglang:d1",
                deterministic_sampling=True,
                max_head_offpolicyness=1,
            )

        mock_logger.warning.assert_called_once()
        assert "task-to-weight-version" in mock_logger.warning.call_args.args[0]

    def test_deterministic_sampling_onpolicy_does_not_warn(self):
        with patch("areal.api.cli_args.logger") as mock_logger:
            InferenceEngineConfig(
                backend="sglang:d1",
                deterministic_sampling=True,
                max_head_offpolicyness=0,
            )

        mock_logger.warning.assert_not_called()

    def test_nondeterministic_sampling_with_offpolicy_head_does_not_warn(self):
        with patch("areal.api.cli_args.logger") as mock_logger:
            InferenceEngineConfig(
                backend="sglang:d1",
                deterministic_sampling=False,
                max_head_offpolicyness=1,
            )

        mock_logger.warning.assert_not_called()


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
            agent=AgentConfig(
                agent_cls_path="tests.experimental.openai.utils.SimpleAgent",
                drop_retry_orphans=True,
            ),
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
        assert resolved.drop_retry_orphans is True

    def test_resolve_workflow_agent_class_creates_offline_workflow(self):
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
            agent=AgentConfig(
                agent_cls_path="tests.experimental.openai.utils.SimpleAgent",
                drop_retry_orphans=True,
            ),
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
        assert resolved.drop_retry_orphans is True

    def test_resolve_workflow_forwards_reward_normalization(self):
        controller = RolloutControllerV2(
            config=InferenceEngineConfig(
                backend="sglang:d1",
                admin_api_key="test-admin-key",
            ),
            scheduler=MagicMock(n_gpus_per_node=8),
        )
        controller._gateway_addr = "http://test:8080"

        class MockAgent:
            async def run(self, data, **kwargs):
                return 1.0

        resolved = controller._resolve_workflow(
            MockAgent,
            group_size=2,
            reward_normalization=True,
        )

        assert isinstance(resolved, InferenceServiceWorkflow)
        assert resolved.group_size == 2
        assert resolved.reward_normalization is True

    def test_resolve_should_accept_fn_none(self):
        assert RolloutControllerV2._resolve_should_accept_fn(None) is None

    def test_resolve_should_accept_fn_callable(self):
        fn = lambda x: True  # noqa: E731
        assert RolloutControllerV2._resolve_should_accept_fn(fn) is fn

    def test_resolve_workflow_with_agent_class(self):
        """Test _resolve_workflow wraps agent-like classes in InferenceServiceWorkflow."""
        cfg = InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-key",
            serialize_group_samples=True,
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
        assert hasattr(resolved, "arun_episode")
        assert resolved.serialize_group_samples is True

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

    def test_export_stats_drains_local_workflow_metrics(self):
        stats_tracker.export_all(reset=True)
        try:
            stats_tracker.get("rollout").scalar(reward=0.75)
            controller = RolloutControllerV2(
                config=InferenceEngineConfig(
                    backend="sglang:d1", admin_api_key="test-key"
                ),
                scheduler=MagicMock(n_gpus_per_node=8),
            )

            assert controller.export_stats() == {
                "rollout/reward": 0.75,
                "rollout/reward__count": 1,
            }
            assert controller.export_stats() == {}
        finally:
            stats_tracker.export_all(reset=True)

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

    @pytest.mark.parametrize("deterministic_sampling", [False, True])
    @pytest.mark.asyncio
    async def test_async_initialize_passes_config_to_data_proxy(
        self, deterministic_sampling
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
            deterministic_sampling=deterministic_sampling,
            agent=AgentConfig(
                agent_cls_path="tests.experimental.openai.utils.SimpleAgent",
                set_reward_finish_timeout=7.5,
                message_preprocessors=[
                    "examples.swe.preprocessors.StripAnthropicBillingHeader",
                    "examples.swe.preprocessors.StripAllSystemReminders",
                ],
                prefix_matcher="examples.swe.prefix_matchers.swe_prefix_matcher",
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

        with patch.object(controller, "_async_fork_on_guard") as mock_fork:
            mock_fork.side_effect = [
                ("127.0.0.1", 18081),
                ("127.0.0.1", 18082),
                ("127.0.0.1", 18080),
            ]

            await controller._async_initialize(
                server_args=None,
                server_infos=[
                    LocalInfServerInfo(
                        host="127.0.0.1", port=30000, process=MagicMock()
                    )
                ],
            )

        data_proxy_calls = [
            c for c in mock_fork.call_args_list if c.kwargs.get("role") == "data-proxy"
        ]
        assert len(data_proxy_calls) == 1
        data_proxy_cmd = data_proxy_calls[0].kwargs["raw_cmd"]
        assert "--set-reward-finish-timeout" in data_proxy_cmd
        assert "7.5" in data_proxy_cmd
        assert "--callback-server-addr" in data_proxy_cmd
        assert "http://127.0.0.1:19000" in data_proxy_cmd
        assert ("--deterministic-sampling" in data_proxy_cmd) is deterministic_sampling
        assert data_proxy_cmd.count("--message-preprocessor") == 2
        first = data_proxy_cmd.index("--message-preprocessor")
        second = data_proxy_cmd.index("--message-preprocessor", first + 1)
        assert data_proxy_cmd[first + 1].endswith("StripAnthropicBillingHeader")
        assert data_proxy_cmd[second + 1].endswith("StripAllSystemReminders")
        matcher = data_proxy_cmd.index("--prefix-matcher")
        assert data_proxy_cmd[matcher + 1] == (
            "examples.swe.prefix_matchers.swe_prefix_matcher"
        )


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
    async def _run_offline_group(
        self,
        *,
        serialize_group_samples: bool,
        failing_member: int | None = None,
    ):
        active = 0
        max_active = 0
        start_order: list[int] = []
        all_started = asyncio.Event()

        class MockAgent:
            async def run(self, data, **kwargs):
                del data
                nonlocal active, max_active
                member_index = int(kwargs["api_key"].rsplit("-", 1)[1])
                start_order.append(member_index)
                active += 1
                max_active = max(max_active, active)
                try:
                    if serialize_group_samples:
                        await asyncio.sleep(0)
                    else:
                        if active == 4:
                            all_started.set()
                        await asyncio.wait_for(all_started.wait(), timeout=1.0)
                    if member_index == failing_member:
                        raise RuntimeError(f"member {member_index} failed")
                    return float(member_index)
                finally:
                    active -= 1

        controller = MagicMock()
        controller.get_version.return_value = 3
        workflow = InferenceServiceWorkflow(
            controller=controller,
            agent=MockAgent(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
            group_size=4,
            serialize_group_samples=serialize_group_samples,
        )
        sessions = [(f"task-42-{i}", f"session-key-{i}") for i in range(4)]
        workflow._start_session = AsyncMock(return_value=("grp-test-42", sessions))
        workflow._set_last_reward = AsyncMock(return_value=None)
        workflow._export_interactions = AsyncMock(
            return_value={"chatcmpl-1": MagicMock(reward=1.0)}
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
            mock_http_session = AsyncMock()
            mock_wf_ctx.get_aiohttp_session = AsyncMock(return_value=mock_http_session)
            mock_wf_ctx.get.return_value = MagicMock(task_id=42)
            mock_wf_ctx.get_httpx_client = AsyncMock(return_value=MagicMock())
            mock_wf_ctx.stat_scope.return_value = "rollout"
            mock_st.get.return_value = tracker

            result = await workflow.arun_episode(engine=MagicMock(), data={})

        workflow._export_interactions.assert_awaited_once_with(
            mock_http_session,
            [session_id for session_id, _ in sessions],
            group_id="grp-test-42",
            discard_trajectory=failing_member is not None,
        )
        return result, max_active, start_order, tracker, workflow

    @pytest.mark.asyncio
    async def test_export_interactions_forwards_drop_retry_orphans(self):
        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
            drop_retry_orphans=True,
        )
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = AsyncMock(return_value={"traj": {}})
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=response)
        context.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.post = MagicMock(return_value=context)

        result = await workflow._export_interactions(session, ["session-1"])

        assert result == {}
        assert session.post.call_args.kwargs["json"]["drop_retry_orphans"] is True

    @pytest.mark.asyncio
    async def test_export_interactions_forwards_reward_normalization(self):
        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
            reward_normalization=True,
        )
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json = AsyncMock(return_value={"traj": {}})
        response_context = MagicMock()
        response_context.__aenter__ = AsyncMock(return_value=response)
        response_context.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.post.return_value = response_context

        result = await workflow._export_interactions(session, ["session-1"])

        assert result == {}
        assert session.post.call_args.kwargs["json"]["reward_normalization"] is True

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
        workflow._start_session.assert_awaited_once()
        workflow._set_last_reward.assert_awaited_once()
        workflow._export_interactions.assert_awaited_once_with(
            mock_http_session,
            ["sess-1"],
            group_id="grp-test-1",
            discard_trajectory=False,
        )

    @pytest.mark.asyncio
    async def test_offline_mode_discards_export_when_agent_fails(self):
        class FailingAgent:
            async def run(self, data, **kwargs):
                raise RuntimeError("agent failed")

        workflow = InferenceServiceWorkflow(
            controller=MagicMock(),
            agent=FailingAgent(),
            gateway_addr="http://test:8080",
            admin_api_key="test-key",
            group_size=2,
            reward_normalization=True,
        )
        workflow._start_session = AsyncMock(
            return_value=(
                "grp-test-1",
                [("sess-1", "key-1"), ("sess-2", "key-2")],
            )
        )
        workflow._set_last_reward = AsyncMock(return_value=None)
        workflow._export_interactions = AsyncMock(return_value={})

        with patch(
            "areal.v2.inference_service.controller.workflow.workflow_context"
        ) as context:
            context.get_aiohttp_session = AsyncMock(return_value=AsyncMock())
            context.get.return_value = MagicMock(task_id=42)
            context.get_httpx_client = AsyncMock(return_value=MagicMock())
            result = await workflow.arun_episode(engine=MagicMock(), data={})

        assert result is None
        workflow._export_interactions.assert_awaited_once_with(
            context.get_aiohttp_session.return_value,
            ["sess-1", "sess-2"],
            group_id="grp-test-1",
            discard_trajectory=True,
        )

    @pytest.mark.asyncio
    async def test_offline_group_is_concurrent_by_default(self):
        result, max_active, start_order, tracker, _ = await self._run_offline_group(
            serialize_group_samples=False,
        )

        assert result is not None
        assert max_active == 4
        assert sorted(start_order) == [0, 1, 2, 3]
        assert tracker.scalar.call_args_list == [
            call(reward=0.0),
            call(reward=1.0),
            call(reward=2.0),
            call(reward=3.0),
        ]

    @pytest.mark.asyncio
    async def test_offline_group_serial_flag_preserves_within_group_order(self):
        result, max_active, start_order, tracker, _ = await self._run_offline_group(
            serialize_group_samples=True,
        )

        assert result is not None
        assert max_active == 1
        assert start_order == [0, 1, 2, 3]
        assert tracker.scalar.call_args_list == [
            call(reward=0.0),
            call(reward=1.0),
            call(reward=2.0),
            call(reward=3.0),
        ]

    @pytest.mark.asyncio
    async def test_offline_group_serial_flag_exports_after_failure(self):
        (
            result,
            max_active,
            start_order,
            tracker,
            workflow,
        ) = await self._run_offline_group(
            serialize_group_samples=True,
            failing_member=1,
        )

        assert result is None
        assert max_active == 1
        assert start_order == [0, 1, 2, 3]
        assert workflow._set_last_reward.await_count == 4
        assert tracker.scalar.call_count == 0


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
    async def test_async_initialize_multinode_fork_path(self, monkeypatch):
        """Exercise the full multi-node fork path (server_infos=None)."""
        from areal.api.cli_args import SchedulingSpec

        monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/controller-triton-dir")
        monkeypatch.setenv("TRITON_CACHE_PATH", "/tmp/controller-triton-path")

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
        alloc_calls = []
        fork_calls = []

        async def mock_async_post(url, json=None, timeout=None):
            nonlocal alloc_port_counter
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if "/alloc_ports" in url:
                alloc_port_counter += 1
                alloc_calls.append(json)
                port_count = json["count"]
                first_port = 30000 + 2 * alloc_port_counter
                resp.json.return_value = {
                    "status": "success",
                    "host": url.split("//")[1].split(":")[0],
                    "ports": list(range(first_port, first_port + port_count)),
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

        # One owner-bound reservation per inference worker. The head reserves
        # both its HTTP port and the distributed rendezvous port.
        assert alloc_port_counter == 2
        assert alloc_calls == [
            {"count": 2, "role": "inf-server", "worker_index": 0},
            {"count": 1, "role": "inf-server", "worker_index": 1},
        ]
        assert len(fork_calls) == 2  # 1 per node in the group

        # Verify fork payloads have correct worker_index and role
        assert fork_calls[0]["role"] == "inf-server"
        assert fork_calls[0]["worker_index"] == 0
        assert fork_calls[1]["role"] == "inf-server"
        assert fork_calls[1]["worker_index"] == 1

        cache_dirs = [fc["env"]["TRITON_CACHE_DIR"] for fc in fork_calls]
        cache_paths = [fc["env"]["TRITON_CACHE_PATH"] for fc in fork_calls]
        assert all(
            path.startswith("/tmp/controller-triton-dir/inf-server-")
            for path in cache_dirs
        )
        assert all(
            path.startswith("/tmp/controller-triton-path/inf-server-")
            for path in cache_paths
        )
        assert len(set(cache_dirs)) == 2
        assert len(set(cache_paths)) == 2

        # Verify dist_init_addr propagated to fork commands
        for fc in fork_calls:
            cmd_str = " ".join(fc["raw_cmd"])
            assert "--dist-init-addr" in cmd_str or "--dist_init_addr" in cmd_str

        # Only 1 data proxy (dp_size=1, on head worker only)
        data_proxy_calls = [
            c for c in mock_fork.call_args_list if c.kwargs.get("role") == "data-proxy"
        ]
        assert len(data_proxy_calls) == 1

    @pytest.mark.asyncio
    async def test_async_fork_inf_servers_failure_releases_owned_ports(self):
        worker = MagicMock()
        worker.ip = "10.0.0.1"
        worker.worker_ports = [18000]

        cfg = InferenceEngineConfig(
            tokenizer_path="mock-tokenizer",
            backend="sglang:d1",
        )
        controller = RolloutControllerV2(
            config=cfg, scheduler=MagicMock(n_gpus_per_node=8)
        )
        requests = []

        async def mock_async_post(url, json=None, timeout=None):
            requests.append((url, json))
            resp = MagicMock()
            if url.endswith("/alloc_ports"):
                resp.json.return_value = {
                    "status": "success",
                    "host": "10.0.0.1",
                    "ports": [30000],
                }
            elif url.endswith("/fork"):
                resp.raise_for_status.side_effect = RuntimeError("fork failed")
            return resp

        mock_async_client = AsyncMock()
        mock_async_client.post = mock_async_post

        with (
            patch.object(
                controller, "_get_async_client", return_value=mock_async_client
            ),
            patch(
                "areal.api.cli_args.SGLangConfig.build_cmd_from_args",
                return_value=["python", "-m", "sglang.launch_server"],
            ),
        ):
            with pytest.raises(RuntimeError, match="fork failed"):
                await controller._async_fork_inf_servers(
                    cfg=cfg,
                    alloc=None,
                    inf_backend="sglang",
                    inf_workers=[worker],
                    dp_size=1,
                    nnodes_per_instance=1,
                    worker_env={},
                    server_args=None,
                )

        assert requests[0] == (
            "http://10.0.0.1:18000/alloc_ports",
            {"count": 1, "role": "inf-server", "worker_index": 0},
        )
        assert [url.rsplit("/", 1)[-1] for url, _ in requests[-2:]] == [
            "kill_forked_worker",
            "release_ports",
        ]
        assert controller._inf_addrs == []
        assert controller._server_infos == []
        assert controller._forked_services == []

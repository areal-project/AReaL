"""Tests for discovering the primary online rollout gateway."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from areal.api.cli_args import InferenceEngineConfig
from areal.infra import RolloutController
from areal.trainer.rl_trainer import PPOTrainer
from areal.v2.inference_service.controller.controller import (
    RolloutControllerV2,
)


class _ObservedFuture(Future):
    """Future that signals when a caller starts waiting for its result."""

    def __init__(self) -> None:
        super().__init__()
        self.result_entered = Event()

    def result(self, timeout: float | None = None):
        self.result_entered.set()
        return super().result(timeout=timeout)


def _make_v2_controller(
    gateway_addr: str,
    init_future: Future | None = None,
) -> RolloutControllerV2:
    controller = RolloutControllerV2(
        config=InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        ),
        scheduler=MagicMock(n_gpus_per_node=8),
    )
    controller._gateway_addr = gateway_addr
    controller._init_future = init_future
    return controller


def _make_trainer(rollout, eval_rollout=None) -> PPOTrainer:
    trainer = object.__new__(PPOTrainer)
    trainer._proxy_started = False
    trainer.rollout = rollout
    trainer.eval_rollout = eval_rollout
    trainer.config = SimpleNamespace(
        scheduler=SimpleNamespace(type="local"),
        rollout=SimpleNamespace(agent=SimpleNamespace(mode="online")),
    )
    return trainer


def test_v2_online_proxy_waits_for_initialization_before_logging():
    primary_gateway = "http://127.0.0.1:18080"
    init_future = _ObservedFuture()
    trainer = _make_trainer(
        _make_v2_controller(primary_gateway, init_future=init_future)
    )
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        with (
            patch("areal.trainer.rl_trainer.is_single_controller", return_value=True),
            patch("areal.trainer.rl_trainer.logger") as mock_logger,
        ):
            start = executor.submit(trainer._ensure_proxy_started)

            assert init_future.result_entered.wait(timeout=5)
            assert start.done() is False
            assert trainer._proxy_started is False
            mock_logger.info.assert_not_called()

            init_future.set_result(None)
            start.result(timeout=5)
    finally:
        if not init_future.done():
            init_future.set_result(None)
        executor.shutdown(wait=True)

    mock_logger.info.assert_called_once_with(
        "Proxy gateway available at %s (role=rollout)",
        primary_gateway,
    )
    assert trainer._proxy_started is True


def test_v2_online_proxy_propagates_initialization_failure_and_can_retry():
    initialization_error = RuntimeError("gateway initialization failed")
    failed_future = Future()
    failed_future.set_exception(initialization_error)
    trainer = _make_trainer(
        _make_v2_controller(
            "http://127.0.0.1:18080",
            init_future=failed_future,
        )
    )

    with (
        patch("areal.trainer.rl_trainer.is_single_controller", return_value=True),
        patch("areal.trainer.rl_trainer.logger") as mock_logger,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            trainer._ensure_proxy_started()

        assert exc_info.value is initialization_error
        assert trainer._proxy_started is False
        mock_logger.info.assert_not_called()

        healthy_gateway = "http://127.0.0.1:18081"
        trainer.rollout = _make_v2_controller(healthy_gateway)
        trainer._ensure_proxy_started()

    mock_logger.info.assert_called_once_with(
        "Proxy gateway available at %s (role=rollout)",
        healthy_gateway,
    )
    assert trainer._proxy_started is True


def test_v2_online_proxy_logs_primary_rollout_gateway_only_once():
    primary_gateway = "http://127.0.0.1:18080"
    eval_gateway = "http://127.0.0.1:18081"
    trainer = _make_trainer(
        _make_v2_controller(primary_gateway),
        eval_rollout=_make_v2_controller(eval_gateway),
    )

    with (
        patch("areal.trainer.rl_trainer.is_single_controller", return_value=True),
        patch("areal.trainer.rl_trainer.logger") as mock_logger,
    ):
        trainer._ensure_proxy_started()
        trainer._ensure_proxy_started()

    mock_logger.info.assert_called_once_with(
        "Proxy gateway available at %s (role=rollout)",
        primary_gateway,
    )
    assert trainer._proxy_started is True


def test_v1_online_proxy_start_behavior_is_unchanged():
    primary_gateway = "http://127.0.0.1:18080"
    rollout = MagicMock(spec=RolloutController)
    rollout.proxy_gateway_addr = primary_gateway
    eval_rollout = MagicMock(spec=RolloutController)
    trainer = _make_trainer(rollout, eval_rollout=eval_rollout)

    with (
        patch("areal.trainer.rl_trainer.is_single_controller", return_value=True),
        patch("areal.trainer.rl_trainer.logger") as mock_logger,
    ):
        trainer._ensure_proxy_started()

    rollout.start_proxy.assert_called_once_with()
    eval_rollout.start_proxy.assert_called_once_with()
    rollout.start_proxy_gateway.assert_called_once_with()
    eval_rollout.start_proxy_gateway.assert_not_called()
    assert mock_logger.info.call_args_list == [
        call("Initializing proxy workers for AgentWorkflow support"),
        call("Proxy gateway available at %s", primary_gateway),
    ]
    assert trainer._proxy_started is True

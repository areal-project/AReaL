"""Tests for discovering the primary online rollout gateway."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from areal.api.cli_args import InferenceEngineConfig
from areal.trainer.rl_trainer import PPOTrainer
from areal.v2.inference_service.controller.controller import (
    RolloutControllerV2,
)


def _make_v2_controller(gateway_addr: str) -> RolloutControllerV2:
    controller = RolloutControllerV2(
        config=InferenceEngineConfig(
            backend="sglang:d1",
            admin_api_key="test-admin-key",
        ),
        scheduler=MagicMock(n_gpus_per_node=8),
    )
    controller._gateway_addr = gateway_addr
    return controller


def test_v2_online_proxy_logs_primary_rollout_gateway_with_role():
    primary_gateway = "http://127.0.0.1:18080"
    eval_gateway = "http://127.0.0.1:18081"
    trainer = object.__new__(PPOTrainer)
    trainer._proxy_started = False
    trainer.rollout = _make_v2_controller(primary_gateway)
    trainer.eval_rollout = _make_v2_controller(eval_gateway)
    trainer.config = SimpleNamespace(
        scheduler=SimpleNamespace(type="local"),
        rollout=SimpleNamespace(agent=SimpleNamespace(mode="online")),
    )

    with (
        patch("areal.trainer.rl_trainer.is_single_controller", return_value=True),
        patch("areal.trainer.rl_trainer.logger") as mock_logger,
    ):
        trainer._ensure_proxy_started()

    mock_logger.info.assert_called_once_with(
        "Proxy gateway available at %s (role=rollout)",
        primary_gateway,
    )
    assert trainer._proxy_started is True

"""Unit tests for RolloutControllerV2 version management.

Tests set_version and get_version with mocked HTTP calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from areal.api.cli_args import InferenceEngineConfig, SchedulingSpec
from areal.v2.inference_service.controller.controller import (
    RolloutControllerV2,
)


def _make_scheduler(n_gpus_per_node: int = 8) -> MagicMock:
    scheduler = MagicMock()
    scheduler.n_gpus_per_node = n_gpus_per_node
    return scheduler


# =============================================================================
# Helpers
# =============================================================================


def _make_controller(
    gateway_addr: str = "",
    worker_ids: dict[str, str] | None = None,
    version: int = 0,
) -> RolloutControllerV2:
    """Create a controller with minimal config and manually injected state.

    Does NOT call initialize() — internal fields are set directly.
    """
    cfg = InferenceEngineConfig(
        backend="sglang:d1",
        admin_api_key="test-key",
        scheduling_spec=(SchedulingSpec(),),
    )
    scheduler = MagicMock(n_gpus_per_node=8)
    ctrl = RolloutControllerV2(config=cfg, scheduler=scheduler)
    ctrl._gateway_addr = gateway_addr
    ctrl._worker_ids = worker_ids or {}
    ctrl._version = version
    return ctrl


# =============================================================================
# TestControllerSetVersion
# =============================================================================


class TestControllerSetVersion:
    """Test RolloutControllerV2.set_version."""

    def test_set_version_updates_local(self):
        ctrl = _make_controller()
        ctrl.set_version(5)
        assert ctrl._version == 5

    def test_set_version_no_gateway_skips_broadcast(self):
        """When _gateway_addr is empty, set_version updates local but makes no HTTP calls."""
        ctrl = _make_controller(gateway_addr="", worker_ids={"dp0": "w1"})
        ctrl._data_proxy_addrs = ["http://dp0:8000"]
        with patch.object(
            ctrl, "_async_data_proxy_post", new_callable=AsyncMock
        ) as mock_post:
            ctrl.set_version(5)
            mock_post.assert_not_called()
        assert ctrl._version == 5

    def test_set_version_broadcasts_to_all_workers(self):
        """When gateway_addr is set and data proxies exist, broadcasts to all."""
        ctrl = _make_controller(
            gateway_addr="http://gateway:8000",
            worker_ids={"dp0": "w1", "dp1": "w2"},
        )
        ctrl._data_proxy_addrs = ["http://dp0:8000", "http://dp1:8000"]

        mock_post = AsyncMock()

        with patch.object(ctrl, "_async_data_proxy_post", mock_post):
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(ctrl._async_set_version(10))
            finally:
                loop.close()

        assert mock_post.call_count == 2
        call_addrs = [call.args[0] for call in mock_post.call_args_list]
        assert "http://dp0:8000" in call_addrs
        assert "http://dp1:8000" in call_addrs
        for call in mock_post.call_args_list:
            assert call.args[1] == "/set_version"
            assert call.args[2] == {"version": 10}


# =============================================================================
# TestControllerGetVersion
# =============================================================================


class TestControllerGetVersion:
    """Test RolloutControllerV2.get_version."""

    def test_get_version_returns_local(self):
        ctrl = _make_controller(version=0)
        assert ctrl.get_version() == 0

    def test_get_version_after_set(self):
        ctrl = _make_controller()
        ctrl.set_version(7)
        assert ctrl.get_version() == 7

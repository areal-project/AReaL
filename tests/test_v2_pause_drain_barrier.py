# SPDX-License-Identifier: Apache-2.0
"""Draining before offload must observe real in-flight counts, not sleep."""

import asyncio
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend


def test_colocate_control_fails_when_any_worker_fails():
    from areal.v2.inference_service.controller.controller import RolloutControllerV2

    controller = object.__new__(RolloutControllerV2)
    controller.awex_colocate = True

    with pytest.raises(RuntimeError, match="1/2"):
        controller._raise_control_failures("offload", [RuntimeError("failed")], 2)


def test_separation_control_keeps_partial_failure_compatibility():
    from areal.v2.inference_service.controller.controller import RolloutControllerV2

    controller = object.__new__(RolloutControllerV2)
    controller.awex_colocate = False

    controller._raise_control_failures("offload", [RuntimeError("failed")], 2)


def test_colocate_launch_waits_for_actor_memory_handoff():
    from areal.v2.inference_service.controller.controller import RolloutControllerV2

    controller = object.__new__(RolloutControllerV2)
    controller.awex_colocate = True
    controller._inference_launch_allowed = threading.Event()
    controller.config = SimpleNamespace(setup_timeout=0.0)

    with pytest.raises(TimeoutError, match="release GPU memory"):
        asyncio.run(controller._wait_for_inference_launch_permission(None))

    controller._inference_launch_allowed.set()
    asyncio.run(controller._wait_for_inference_launch_permission(None))


def test_colocate_handoff_allows_launch_before_waiting_for_initialization():
    from areal.v2.inference_service.controller.controller import RolloutControllerV2

    events = []
    controller = object.__new__(RolloutControllerV2)
    controller.awex_colocate = True
    controller._inference_launch_allowed = threading.Event()
    controller._ensure_initialized = lambda: events.append(
        controller._inference_launch_allowed.is_set()
    )

    controller.continue_initialization_after_colocate_handoff()

    assert events == [True]


class TestParseInFlight:
    @pytest.mark.parametrize(
        "body,expected",
        [
            ({"loads": [{"num_running_reqs": 0, "num_waiting_reqs": 0}]}, 0),
            ({"loads": [{"num_running_reqs": 3, "num_waiting_reqs": 4}]}, 7),
            (
                {
                    "loads": [
                        {"num_running_reqs": 1, "num_waiting_reqs": 0},
                        {"num_running_reqs": 0, "num_waiting_reqs": 2},
                    ]
                },
                3,
            ),
            ({"num_running_reqs": 5, "num_waiting_reqs": 1}, 6),
        ],
    )
    def test_counts_running_and_waiting_across_dp_ranks(self, body, expected):
        assert SGLangBridgeBackend.parse_in_flight(body) == expected

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"num_running_reqs": 0},
            {"num_waiting_reqs": 0},
            {"loads": [{"num_running_reqs": 0}]},
            {"loads": [{"num_waiting_reqs": 0}]},
        ],
    )
    def test_missing_fields_are_treated_as_busy_not_idle(self, body):
        """Unknown counts must not be mistaken for a drained engine."""
        assert SGLangBridgeBackend.parse_in_flight(body) > 0


@pytest.mark.parametrize(
    "counts,timeout,raises", [([0], 30, False), ([5, 2, 0], 30, False), ([9], 0, True)]
)
def test_pause_generation_waits_for_drain_or_times_out(counts, timeout, raises):
    from areal.v2.inference_service.controller.controller import RolloutControllerV2

    ctrl = mock.Mock(spec=RolloutControllerV2)
    ctrl._poll_in_flight.side_effect = counts
    if raises:
        with pytest.raises(RuntimeError, match="drain"):
            RolloutControllerV2.pause_generation_sync(
                ctrl, drain_timeout=timeout, poll_interval=0
            )
    else:
        RolloutControllerV2.pause_generation_sync(
            ctrl, drain_timeout=timeout, poll_interval=0
        )
        assert ctrl._poll_in_flight.call_count == len(counts)
        assert ctrl._nudge_abort.call_count == len(counts) - 1
    ctrl.pause_generation.assert_called_once()


def test_backend_without_load_probe_fails_closed():
    bridge = InfBridge(
        backend=object(),
        backend_addr="http://backend",
        pause_state=mock.Mock(),
    )

    async def run():
        try:
            with pytest.raises(NotImplementedError, match="safe drain probing"):
                await bridge.in_flight()
        finally:
            await bridge.aclose()

    import asyncio

    asyncio.run(run())


@pytest.mark.parametrize(
    "responses,expected",
    [
        ([{"in_flight": 0}], 0),
        ([{"in_flight": 2}, {"in_flight": 3}], 5),
        ([{}], 1),
        ([RuntimeError("unreachable")], 1),
    ],
)
def test_proxy_poll_counts_requests_and_treats_failed_probes_as_busy(
    responses, expected
):
    from areal.v2.inference_service.controller.controller import RolloutControllerV2

    ctrl = mock.Mock(spec=RolloutControllerV2)
    ctrl._data_proxy_addrs = [f"http://p{i}" for i in range(len(responses))]
    ctrl._async_data_proxy_get = mock.AsyncMock(side_effect=responses)

    assert asyncio.run(RolloutControllerV2._async_poll_in_flight(ctrl)) == expected
    assert ctrl._async_data_proxy_get.await_args_list == [
        mock.call(addr, "/in_flight") for addr in ctrl._data_proxy_addrs
    ]

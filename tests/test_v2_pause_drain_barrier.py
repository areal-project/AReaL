# SPDX-License-Identifier: Apache-2.0
"""Draining before offload must observe real in-flight counts, not sleep."""

from unittest import mock

import pytest

from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend


class TestLoadRequest:
    def test_load_request_targets_the_core_loads_endpoint(self):
        req = SGLangBridgeBackend().get_load_request()

        assert req.method == "GET"
        assert req.endpoint.startswith("/v1/loads")
        assert "include=core" in req.endpoint

    def test_pause_request_keeps_the_abort_mode_default(self):
        """Empty body means mode="abort", which is SGLang's blocking barrier."""
        assert SGLangBridgeBackend().get_pause_request().payload == {}


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

    def test_missing_fields_are_treated_as_busy_not_idle(self):
        """An unparseable payload must not be mistaken for a drained engine."""
        assert SGLangBridgeBackend.parse_in_flight({}) > 0


class TestDrainBarrier:
    """pause_generation_sync must poll real counts and bound its wait."""

    @staticmethod
    def _controller(sequence, elapsed=None):
        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )

        calls = {"pause": 0, "polls": 0, "nudges": 0}
        remaining = list(sequence)

        def _poll():
            calls["polls"] += 1
            return remaining.pop(0) if remaining else 0

        ctrl = mock.Mock(spec=RolloutControllerV2)
        ctrl.pause_generation = lambda: calls.__setitem__("pause", calls["pause"] + 1)
        ctrl._poll_in_flight = _poll
        ctrl._nudge_abort = lambda: calls.__setitem__("nudges", calls["nudges"] + 1)
        ctrl._calls = calls
        return ctrl, calls

    def test_returns_immediately_when_engines_report_idle(self):
        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )

        ctrl, calls = self._controller([0])

        RolloutControllerV2.pause_generation_sync(ctrl, drain_timeout=30.0)

        assert calls["pause"] == 1
        assert calls["polls"] == 1
        assert calls["nudges"] == 0

    def test_polls_until_drained(self):
        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )

        ctrl, calls = self._controller([5, 2, 0])

        RolloutControllerV2.pause_generation_sync(
            ctrl, drain_timeout=30.0, poll_interval=0.0
        )

        assert calls["polls"] == 3
        assert calls["nudges"] == 2

    def test_raises_when_drain_exceeds_timeout(self):
        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )

        ctrl, _ = self._controller([9] * 100)

        with pytest.raises(RuntimeError, match="drain"):
            RolloutControllerV2.pause_generation_sync(
                ctrl, drain_timeout=0.0, poll_interval=0.0
            )


class TestPollInFlightUsesRealAddrList:
    """Regression: _data_proxy_addrs is an attribute, not a method."""

    @staticmethod
    def _controller(per_proxy):
        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )

        ctrl = mock.Mock(spec=RolloutControllerV2)
        ctrl._data_proxy_addrs = [f"http://p{i}" for i in range(len(per_proxy))]

        async def _get(addr, endpoint):
            assert endpoint == "/in_flight"
            return per_proxy[ctrl._data_proxy_addrs.index(addr)]

        ctrl._async_data_proxy_get = _get
        return ctrl

    @pytest.mark.parametrize(
        "per_proxy,expected",
        [
            ([{"in_flight": 0}], 0),
            ([{"in_flight": 0}, {"in_flight": 0}], 0),
            ([{"in_flight": 2}, {"in_flight": 3}], 5),
            ([{}], 1),
        ],
    )
    def test_sums_in_flight_across_data_proxies(self, per_proxy, expected):
        import asyncio

        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )

        ctrl = self._controller(per_proxy)

        total = asyncio.run(RolloutControllerV2._async_poll_in_flight(ctrl))

        assert total == expected

    def test_probe_failure_counts_as_busy(self):
        import asyncio

        from areal.v2.inference_service.controller.controller import (
            RolloutControllerV2,
        )

        ctrl = mock.Mock(spec=RolloutControllerV2)
        ctrl._data_proxy_addrs = ["http://p0"]

        async def _boom(addr, endpoint):
            raise RuntimeError("unreachable")

        ctrl._async_data_proxy_get = _boom

        assert asyncio.run(RolloutControllerV2._async_poll_in_flight(ctrl)) == 1

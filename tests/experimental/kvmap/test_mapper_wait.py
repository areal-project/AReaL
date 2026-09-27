# SPDX-License-Identifier: Apache-2.0
"""The mapper wait returns as soon as the artifact exists and gives up exactly at the timeout."""

import pytest

from areal.experimental.kvmap.wait import wait_for_path


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.now += seconds


def test_wait_returns_without_sleeping_when_the_artifact_already_exists():
    clock = FakeClock()
    waited, timed_out = wait_for_path("/x", timeout_s=5.0, exists=lambda p: True, clock=clock, sleep=clock.sleep)
    assert (waited, timed_out) == (0.0, False)


def test_wait_stops_when_the_artifact_appears_and_reports_time_spent():
    clock = FakeClock()
    appears_at = 101.2
    waited, timed_out = wait_for_path("/x", timeout_s=5.0, exists=lambda p: clock.now >= appears_at, clock=clock, sleep=clock.sleep)
    assert timed_out is False and 1.2 <= waited <= 1.7


def test_wait_times_out_and_never_overshoots_the_budget():
    clock = FakeClock()
    calls = []
    # 1.3 is not exactly representable: the third sleep leaves a ~3e-15 remainder that must count as expired
    waited, timed_out = wait_for_path("/x", timeout_s=1.3, exists=lambda p: False, clock=clock, sleep=lambda s: (calls.append(s), clock.sleep(s)))
    assert timed_out is True and abs(waited - 1.3) < 1e-9
    assert [round(c, 6) for c in calls] == [0.5, 0.5, 0.3]


def test_non_positive_timeout_is_rejected():
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        wait_for_path("/x", timeout_s=0.0)

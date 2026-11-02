# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from areal.elastic.contracts import (
    ActorView,
    OnlineLayout,
    OnlineObservation,
    RequestView,
)
from areal.elastic.serving import (
    ServingSLO,
    request_slo_status,
    request_tpot_seconds,
    rolling_slo,
)


def view(*requests, now=10):
    return OnlineObservation(
        now,
        OnlineLayout(2, 1, 1, 1),
        (),
        requests,
        (),
        ActorView(),
        0,
        0,
        0,
        False,
        False,
    )


@pytest.mark.parametrize("tpot,passed", [(0.04, True), (0.08, False)])
def test_completed_request_uses_mean_tpot(tpot, passed):
    """One large gap does not replace the per-request decode mean."""
    request = RequestView(
        "r",
        "e",
        0,
        4,
        0,
        generated=11,
        first_token_at=1,
        last_token_at=1 + 10 * tpot,
        completed_at=1 + 10 * tpot,
        maximum_tbt_seconds=0.2,
    )
    slo = ServingSLO(5, 0.05, 0.8)
    assert request_tpot_seconds(request) == pytest.approx(tpot)
    assert request_slo_status(request, slo, now=10) is passed
    assert rolling_slo(view(request), slo, 120).fraction == float(passed)


def test_pending_mean_can_improve_before_request_completion():
    """Candidate pruning must not turn a partial mean into a permanent failure."""
    request = RequestView(
        "r", "e", 0, 4, 0, generated=2, first_token_at=1, last_token_at=2
    )
    slo = ServingSLO(5, 0.05, 0.8)
    assert request_slo_status(request, slo, now=2) is None
    assert rolling_slo(view(request, now=2), slo, 120).decided == 0
    complete = replace(request, generated=101, last_token_at=3, completed_at=3)
    assert request_slo_status(complete, slo, now=3) is True


def test_unmeasured_completed_requests_keep_window_slo_unknown():
    """An incomplete timing sample cannot justify a slack search."""
    request = RequestView("r", "e", 0, 4, 0, generated=10, completed_at=3)
    assert rolling_slo(view(request), ServingSLO(5, 0.05, 0.8), 120).fraction is None


def test_overdue_first_token_is_already_a_known_failure():
    """A still-waiting request can establish pressure without a final TPOT."""
    request = RequestView("r", "e", 0, 4, 0)
    statistics = rolling_slo(view(request, now=6), ServingSLO(5, 0.05, 0.8), 120)
    assert statistics.pending == statistics.decided == 1
    assert statistics.fraction == 0

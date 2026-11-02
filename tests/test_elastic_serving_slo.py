# SPDX-License-Identifier: Apache-2.0

"""Serving feasibility constrains candidate selection before GPU cost."""

from areal.elastic.contracts import CandidatePrediction, OnlineLayout
from areal.elastic.reconfigurator.basic.optimizer import select_prediction


def test_joint_search_filters_slo_before_comparing_gpu_cost():
    """A cheaper configuration cannot displace a serving-feasible one."""
    layouts = tuple(OnlineLayout(n, 1, 1, 1) for n in (1, 2, 4))
    predictions = tuple(
        CandidatePrediction(layout, True, fraction, cost, ("same-work",))
        for layout, fraction, cost in zip(
            layouts, (0.90, 0.96, 1.0), (20, 30, 50), strict=True
        )
    )
    result = select_prediction(predictions, layouts[0], 0.95)
    assert result.layout == layouts[1]
    assert (
        select_prediction(predictions, layouts[0], 0.8, trigger="slack").layout
        == layouts[0]
    )

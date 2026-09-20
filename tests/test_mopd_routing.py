# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from areal.api.cli_args import InferenceEngineConfig
from areal.dataset.mopd import MOPD_ROUTE_METADATA_KEY, DatasetRoute
from areal.infra.controller.rollout_controller import RolloutController


class _InferenceEngine:
    pass


class _Scheduler:
    pass


def _controller() -> RolloutController:
    controller = RolloutController(
        inf_engine=_InferenceEngine,
        config=InferenceEngineConfig(backend="sglang:d1"),
        scheduler=_Scheduler(),
    )
    controller.enable_mopd_routing()
    return controller


def test_source_route_metadata_is_removed_from_workflow_data():
    """The configured source route travels outside the workflow sample."""
    controller = _controller()

    prepared, route = controller._extract_mopd_route(
        {
            MOPD_ROUTE_METADATA_KEY: DatasetRoute(0, "gsm8k_single_a"),
            "instance_id": "unique-sample",
        },
        required=True,
    )

    assert route == "gsm8k_single_a"
    assert MOPD_ROUTE_METADATA_KEY not in prepared
    assert prepared["instance_id"] == "unique-sample"


def test_training_route_metadata_is_required():
    """No sample field substitutes for the configured source route."""
    controller = _controller()

    with pytest.raises(ValueError, match="route metadata is missing"):
        controller._extract_mopd_route({"task_type": "gsm8k_single_a"}, required=True)


@pytest.mark.parametrize("route", [None, "", 7, 1.5, True, [], {}])
def test_invalid_source_route_type_raises(route):
    """Only non-empty configured route strings are accepted."""
    controller = _controller()

    with pytest.raises(ValueError, match="DatasetRoute provenance"):
        controller._extract_mopd_route({MOPD_ROUTE_METADATA_KEY: route}, required=True)


def test_concat_trajectory_inherits_source_route():
    """A trajectory produced after OpenAI concat retains its source route."""
    controller = _controller()
    concat_trajectory = {"input_ids": "concat-output", "attention_mask": "mask"}

    result = controller._propagate_mopd_route("gsm8k_ensemble", concat_trajectory)

    assert result["mopd_route"] == "gsm8k_ensemble"


def test_multiple_derived_trajectories_inherit_same_route():
    """Every trajectory generated from one source sample receives one route."""
    controller = _controller()
    trajectories = [
        controller._propagate_mopd_route("gsm8k_single_b", {"trajectory_id": index})
        for index in range(3)
    ]

    assert [trajectory["mopd_route"] for trajectory in trajectories] == [
        "gsm8k_single_b",
        "gsm8k_single_b",
        "gsm8k_single_b",
    ]


def test_workflow_cannot_change_source_route():
    """A conflicting workflow route is rejected before teacher dispatch."""
    controller = _controller()

    with pytest.raises(ValueError, match="changed mopd_route"):
        controller._propagate_mopd_route(
            "gsm8k_single_a", {"mopd_route": "gsm8k_single_b"}
        )


def test_training_submit_keeps_route_outside_workflow_data():
    """Task metadata retains the source route while workflow data stays clean."""
    controller = _controller()
    controller._resolve_workflow_str = MagicMock(return_value="workflow")
    controller._resolve_should_accept_fn = MagicMock(return_value=None)
    controller._dispatcher = MagicMock()

    controller.submit(
        {
            "messages": [{"role": "user", "content": "train me"}],
            MOPD_ROUTE_METADATA_KEY: DatasetRoute(0, "gsm8k_single_a"),
        },
        object(),
    )

    task_input = controller._dispatcher.submit_task_input.call_args.args[0]
    assert task_input.mopd_route == "gsm8k_single_a"
    assert MOPD_ROUTE_METADATA_KEY not in task_input.data


def test_eval_submit_does_not_require_training_route():
    """Validation samples remain usable without the training-only route field."""
    controller = _controller()
    controller._resolve_workflow_str = MagicMock(return_value="workflow")
    controller._resolve_should_accept_fn = MagicMock(return_value=None)
    controller._dispatcher = MagicMock()

    controller.submit(
        {"messages": [{"role": "user", "content": "evaluate me"}]},
        object(),
        is_eval=True,
    )

    task_input = controller._dispatcher.submit_task_input.call_args.args[0]
    assert task_input.is_eval is True
    assert task_input.mopd_route is None
    assert "mopd_route" not in task_input.data

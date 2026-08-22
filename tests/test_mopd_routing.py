# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from areal.api.cli_args import InferenceEngineConfig
from areal.infra.controller.rollout_controller import RolloutController


class _InferenceEngine:
    pass


class _Scheduler:
    pass


def _controller(identifier: str = "task_type") -> RolloutController:
    controller = RolloutController(
        inf_engine=_InferenceEngine,
        config=InferenceEngineConfig(backend="sglang:d1"),
        scheduler=_Scheduler(),
    )
    controller.set_mopd_route_identifier(identifier)
    return controller


@pytest.mark.parametrize(
    ("source_route", "expected"),
    [("gsm8k_single_a", "gsm8k_single_a"), (7, "7")],
)
def test_source_route_is_copied_to_reserved_field(source_route, expected):
    """String and integer identifiers become a stable string mopd_route."""
    controller = _controller()

    prepared = controller._prepare_mopd_data(
        {"task_type": source_route, "instance_id": "unique-sample"}
    )

    assert prepared["mopd_route"] == expected
    assert prepared["instance_id"] == "unique-sample"


def test_instance_id_is_not_used_as_route():
    """A unique instance identifier never substitutes for a missing route."""
    controller = _controller()

    with pytest.raises(ValueError, match="task_type.*missing"):
        controller._prepare_mopd_data({"instance_id": "gsm8k_single_a"})


@pytest.mark.parametrize("route", [None, 1.5, True, [], {}])
def test_invalid_source_route_type_raises(route):
    """Only source strings and integers are valid route identifiers."""
    controller = _controller()

    with pytest.raises(ValueError, match="string or integer"):
        controller._prepare_mopd_data({"task_type": route})


def test_concat_trajectory_inherits_source_route():
    """A trajectory produced after OpenAI concat retains its source route."""
    controller = _controller()
    source = controller._prepare_mopd_data(
        {"task_type": "gsm8k_ensemble", "messages": []}
    )
    concat_trajectory = {"input_ids": "concat-output", "attention_mask": "mask"}

    result = controller._propagate_mopd_route(source, concat_trajectory)

    assert result["mopd_route"] == "gsm8k_ensemble"


def test_multiple_derived_trajectories_inherit_same_route():
    """Every trajectory generated from one source sample receives one route."""
    controller = _controller()
    source = controller._prepare_mopd_data({"task_type": "gsm8k_single_b"})
    trajectories = [
        controller._propagate_mopd_route(source, {"trajectory_id": index})
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
    source = controller._prepare_mopd_data({"task_type": "gsm8k_single_a"})

    with pytest.raises(ValueError, match="changed mopd_route"):
        controller._propagate_mopd_route(source, {"mopd_route": "gsm8k_single_b"})


def test_eval_submit_does_not_require_training_route():
    """Validation samples remain usable without the training-only route field."""
    controller = _controller()
    controller._prepare_mopd_data = MagicMock(
        side_effect=AssertionError("eval unexpectedly required an MOPD route")
    )
    controller._resolve_workflow_str = MagicMock(return_value="workflow")
    controller._resolve_should_accept_fn = MagicMock(return_value=None)
    controller._dispatcher = MagicMock()

    controller.submit(
        {"messages": [{"role": "user", "content": "evaluate me"}]},
        object(),
        is_eval=True,
    )

    controller._prepare_mopd_data.assert_not_called()
    task_input = controller._dispatcher.submit_task_input.call_args.args[0]
    assert task_input.is_eval is True
    assert "mopd_route" not in task_input.data

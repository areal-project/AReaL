from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from areal.infra.controller.train_controller import TrainController
from areal.utils.seqpack import balanced_greedy_partition


def test_balanced_greedy_partition_handles_remainders():
    groups = balanced_greedy_partition([10, 8, 6, 4, 2], K=3)

    assert len(groups) == 3
    assert sorted(len(group) for group in groups) == [1, 2, 2]
    assert sorted(idx for group in groups for idx in group) == [0, 1, 2, 3, 4]


def _make_controller() -> TrainController:
    controller = TrainController.__new__(TrainController)
    controller.train_alloc = SimpleNamespace(parallel=SimpleNamespace(dp_size=4))
    return controller


def _sample_item() -> dict[str, torch.Tensor]:
    return {"attention_mask": torch.ones(1, dtype=torch.long)}


def test_custom_function_call_trims_padded_eval_items():
    controller = _make_controller()
    sample = _sample_item()
    captured: dict[str, int] = {}

    def fake_prepare_dispatch(*args, **kwargs):
        captured["padded_len"] = len(args[0])
        return [], {}, None

    controller._prepare_dispatch = fake_prepare_dispatch
    controller._call_workers = lambda *args, **kwargs: []
    controller._collect_results = (
        lambda results, group_indices: list(range(captured["padded_len"]))
    )

    with patch(
        "areal.infra.controller.train_controller.make_dummy_eval_item",
        lambda template: {"attention_mask": torch.zeros_like(template["attention_mask"])},
    ), patch(
        "areal.infra.controller.train_controller.run_async_task",
        lambda fn, *args, **kwargs: [],
    ):
        result = controller._custom_function_call("noop", [sample] * 5)

    assert captured["padded_len"] == 8
    assert result == [0, 1, 2, 3, 4]


def test_custom_function_call_pads_kwargs_and_parallel_tensor_lists():
    controller = _make_controller()
    sample = _sample_item()
    captured: dict[str, list[int] | int] = {}

    def fake_prepare_dispatch(*args, **kwargs):
        captured["arg_lens"] = [len(args[0]), len(args[1])]
        captured["kwarg_len"] = len(kwargs["batch_kw"])
        return [], {}, None

    controller._prepare_dispatch = fake_prepare_dispatch
    controller._call_workers = lambda *args, **kwargs: []
    controller._collect_results = (
        lambda results, group_indices: list(range(captured["kwarg_len"]))
    )

    with patch(
        "areal.infra.controller.train_controller.make_dummy_eval_item",
        lambda template: {"attention_mask": torch.zeros_like(template["attention_mask"])},
    ), patch(
        "areal.infra.controller.train_controller.run_async_task",
        lambda fn, *args, **kwargs: [],
    ):
        result = controller._custom_function_call(
            "noop",
            [sample] * 5,
            [sample] * 5,
            batch_kw=[sample] * 5,
        )

    assert captured["arg_lens"] == [8, 8]
    assert captured["kwarg_len"] == 8
    assert result == [0, 1, 2, 3, 4]


def test_custom_function_call_uses_kwargs_length_for_trimming():
    controller = _make_controller()
    sample = _sample_item()
    captured: dict[str, int] = {}

    def fake_prepare_dispatch(*args, **kwargs):
        captured["padded_len"] = len(kwargs["batch_kw"])
        return [], {}, None

    controller._prepare_dispatch = fake_prepare_dispatch
    controller._call_workers = lambda *args, **kwargs: []
    controller._collect_results = (
        lambda results, group_indices: list(range(captured["padded_len"]))
    )

    with patch(
        "areal.infra.controller.train_controller.make_dummy_eval_item",
        lambda template: {"attention_mask": torch.zeros_like(template["attention_mask"])},
    ), patch(
        "areal.infra.controller.train_controller.run_async_task",
        lambda fn, *args, **kwargs: [],
    ):
        result = controller._custom_function_call("noop", batch_kw=[sample] * 5)

    assert captured["padded_len"] == 8
    assert result == [0, 1, 2, 3, 4]


def test_custom_function_call_rejects_inconsistent_tensor_lengths():
    controller = _make_controller()
    sample = _sample_item()

    try:
        controller._custom_function_call(
            "noop",
            [sample] * 5,
            [sample] * 4,
        )
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("expected inconsistent tensor-like inputs to fail")

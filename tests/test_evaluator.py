# SPDX-License-Identifier: Apache-2.0

import pytest

from areal.api import FinetuneSpec
from areal.api.cli_args import EvaluatorConfig
from areal.utils.evaluator import Evaluator


def _make_evaluator(
    *,
    eval_before_train: bool,
    freq_steps: int | None = None,
) -> Evaluator:
    config = EvaluatorConfig(
        experiment_name="test",
        trial_name="eval-before-train",
        fileroot="/tmp",
        eval_before_train=eval_before_train,
        freq_steps=freq_steps,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=4,
        train_batch_size=1,
    )
    return Evaluator(config, ft_spec)


def test_evaluate_before_train_runs_once_when_enabled():
    """The initial evaluation should run exactly once on a fresh evaluator."""
    evaluator = _make_evaluator(eval_before_train=True)
    calls: list[str] = []

    first_ran = evaluator.evaluate_before_train(lambda: calls.append("initial"))
    second_ran = evaluator.evaluate_before_train(lambda: calls.append("duplicate"))

    assert first_ran is True
    assert second_ran is False
    assert calls == ["initial"]


def test_evaluate_before_train_is_disabled_by_default():
    """A disabled initial evaluation should remain a no-op."""
    evaluator = _make_evaluator(eval_before_train=False)
    calls: list[str] = []

    ran = evaluator.evaluate_before_train(lambda: calls.append("initial"))

    assert ran is False
    assert calls == []


def test_evaluate_before_train_without_callback_consumes_initial_opportunity():
    """Missing evaluation inputs must not defer a baseline until after updates."""
    evaluator = _make_evaluator(eval_before_train=True)
    calls: list[str] = []

    missing_inputs_ran = evaluator.evaluate_before_train(None)
    later_ran = evaluator.evaluate_before_train(lambda: calls.append("late"))

    assert missing_inputs_ran is False
    assert later_ran is False
    assert calls == []


def test_evaluate_before_train_retries_after_callback_failure():
    """A failed callback should leave the one-shot evaluation pending."""
    evaluator = _make_evaluator(eval_before_train=True)
    calls: list[str] = []

    def fail_initial_evaluation() -> None:
        calls.append("failed")
        raise RuntimeError("evaluation failed")

    with pytest.raises(RuntimeError, match="evaluation failed"):
        evaluator.evaluate_before_train(fail_initial_evaluation)

    retry_ran = evaluator.evaluate_before_train(lambda: calls.append("retried"))
    duplicate_ran = evaluator.evaluate_before_train(lambda: calls.append("duplicate"))

    assert retry_ran is True
    assert duplicate_ran is False
    assert calls == ["failed", "retried"]


def test_evaluate_before_train_does_not_advance_step_frequency():
    """The initial evaluation must not advance any periodic cadence."""
    evaluator = _make_evaluator(eval_before_train=True, freq_steps=2)
    calls: list[str] = []
    state_before = evaluator.state_dict()

    evaluator.evaluate_before_train(lambda: calls.append("initial"))
    assert evaluator.state_dict() == state_before

    evaluator.evaluate(
        lambda: calls.append("scheduled"),
        epoch=0,
        step=0,
        global_step=0,
    )
    assert calls == ["initial"]

    evaluator.evaluate(
        lambda: calls.append("scheduled"),
        epoch=0,
        step=1,
        global_step=1,
    )
    assert calls == ["initial", "scheduled"]


def test_load_legacy_state_does_not_rearm_initial_evaluation():
    """Recovery should ignore the legacy deferred initial-trigger state."""
    evaluator = _make_evaluator(eval_before_train=True)
    legacy_state = evaluator.state_dict()
    legacy_state["epoch"]["initial_value"] = True

    recovered = _make_evaluator(eval_before_train=True)
    recovered.load_state_dict(legacy_state)
    calls: list[str] = []

    initial_ran = recovered.evaluate_before_train(lambda: calls.append("initial"))
    recovered.evaluate(
        lambda: calls.append("scheduled"),
        epoch=0,
        step=0,
        global_step=1,
    )

    assert initial_ran is False
    assert calls == []
    assert legacy_state["epoch"]["initial_value"] is True

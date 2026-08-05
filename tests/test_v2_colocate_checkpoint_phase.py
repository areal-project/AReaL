"""Guards that colocated checkpointing runs while train memory is resident."""

import ast
import pathlib

import pytest

import areal.trainer.rl_trainer as rl_trainer_module

SAVE_CALLS = ("_save_recover_checkpoint", "_save_hf")


def _train_method() -> ast.FunctionDef:
    source = pathlib.Path(rl_trainer_module.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "train":
            return node
    raise AssertionError("RLTrainer.train not found")


def _train_phase_with(train: ast.FunctionDef) -> ast.With:
    for node in ast.walk(train):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            expr = item.context_expr
            if isinstance(expr, ast.Name) and expr.id == "train_phase":
                return node
    raise AssertionError("`with train_phase:` not found in RLTrainer.train")


def _call_line(node: ast.AST, name: str) -> int:
    lines = [
        sub.lineno
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == name
    ]
    assert lines, f"no call to {name} inside the train phase"
    return min(lines)


def _called_method_names(node: ast.AST) -> set[str]:
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            names.add(sub.func.attr)
    return names


class TestCheckpointRunsInsideTrainPhase:
    @pytest.mark.parametrize("call_name", SAVE_CALLS)
    def test_save_call_is_nested_in_train_phase(self, call_name):
        train = _train_method()
        inside = _called_method_names(_train_phase_with(train))
        assert call_name in inside, (
            f"{call_name} runs outside `with train_phase:`; under AWEX "
            "colocation the train weights are released on phase exit, so "
            "saving there hits tensors whose storage was unmapped"
        )

    @pytest.mark.parametrize("call_name", SAVE_CALLS)
    def test_save_call_appears_exactly_once(self, call_name):
        train = _train_method()
        occurrences = [
            sub
            for sub in ast.walk(train)
            if isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == call_name
        ]
        assert len(occurrences) == 1, (
            f"expected a single {call_name} call site, found {len(occurrences)}"
        )

    @pytest.mark.parametrize("call_name", SAVE_CALLS)
    def test_save_call_precedes_the_weight_transfer(self, call_name):
        train = _train_method()
        phase = _train_phase_with(train)
        save_line = _call_line(phase, call_name)
        transfer_line = _call_line(phase, "update_weights")
        assert save_line < transfer_line, (
            f"{call_name} runs after update_weights; the AWEX colocate transfer "
            "releases the training weights, optimizer and grad buffers before it "
            "returns, so nothing is left on the GPU to checkpoint"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# SPDX-License-Identifier: Apache-2.0

"""Checkpoint ordering differs only for AWEX-colocated training."""

import ast
import pathlib

import areal.trainer.rl_trainer as rl_trainer_module


def _train_method() -> ast.FunctionDef:
    source = pathlib.Path(rl_trainer_module.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "train":
            return node
    raise AssertionError("PPOTrainer.train not found")


def _train_phase(train: ast.FunctionDef) -> ast.With:
    for node in ast.walk(train):
        if not isinstance(node, ast.With):
            continue
        if any(
            isinstance(item.context_expr, ast.Name)
            and item.context_expr.id == "train_phase"
            for item in node.items
        ):
            return node
    raise AssertionError("with train_phase not found")


def _method_calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        sub
        for sub in ast.walk(node)
        if isinstance(sub, ast.Call)
        and isinstance(sub.func, ast.Attribute)
        and sub.func.attr == name
    ]


def test_colocate_checkpoint_precedes_weight_publish_inside_train_phase():
    phase = _train_phase(_train_method())
    saves = _method_calls(phase, "_save_training_state")
    updates = _method_calls(phase, "update_weights")

    assert len(saves) == 1
    assert len(updates) == 1
    assert saves[0].lineno < updates[0].lineno


def test_noncolocate_checkpoint_path_remains_after_train_phase():
    train = _train_method()
    phase = _train_phase(train)
    saves = _method_calls(train, "_save_training_state")

    assert len(saves) == 2
    assert any(save.lineno > phase.end_lineno for save in saves)


def test_colocate_version_commit_is_marked_after_all_version_updates():
    phase = _train_phase(_train_method())
    set_versions = _method_calls(phase, "set_version")
    commits = _method_calls(phase, "mark_colocate_update_committed")

    assert len(set_versions) >= 2
    assert len(commits) == 1
    assert max(call.lineno for call in set_versions) < commits[0].lineno

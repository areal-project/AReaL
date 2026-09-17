# SPDX-License-Identifier: Apache-2.0
"""The gloo mirror group is only built when the engine can offload."""

import ast
import pathlib

import pytest

import areal.engine.megatron_engine as megatron_engine_module


def _init_group_method() -> ast.FunctionDef:
    source = pathlib.Path(megatron_engine_module.__file__).read_text()
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_init_context_and_model_parallel_group"
        ):
            return node
    raise AssertionError("_init_context_and_model_parallel_group not found")


def _gloo_group_calls(node: ast.AST) -> list[ast.Call]:
    calls = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        if not (isinstance(sub.func, ast.Attribute) and sub.func.attr == "new_group"):
            continue
        for kw in sub.keywords:
            if (
                kw.arg == "backend"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value == "gloo"
            ):
                calls.append(sub)
    return calls


class TestGlooMirrorGroupIsGated:
    def test_the_gloo_group_is_built_only_under_an_offload_guard(self):
        method = _init_group_method()
        gloo_calls = _gloo_group_calls(method)

        assert len(gloo_calls) == 1, (
            f"expected exactly one gloo new_group call, found {len(gloo_calls)}"
        )
        target = gloo_calls[0]

        guarded = False
        for node in ast.walk(method):
            if not isinstance(node, ast.If):
                continue
            if any(c is target for c in _gloo_group_calls(node)):
                test_src = ast.dump(node.test)
                if "offload" in test_src:
                    guarded = True
        assert guarded, (
            "the gloo mirror group is created unconditionally; a separation run "
            "pays one collective per data-parallel group for a group that "
            "resolve_broadcast_target never uses"
        )

    def test_the_mirror_group_defaults_to_none(self):
        source = pathlib.Path(megatron_engine_module.__file__).read_text()

        assert "self._cpu_model_parallel_group = None" in source, (
            "resolve_broadcast_target falls back on a None mirror group, so the "
            "attribute must exist even when the group is not built"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

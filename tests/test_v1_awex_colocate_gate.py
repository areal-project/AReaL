# SPDX-License-Identifier: Apache-2.0
"""Guards that v1 AWEX colocation never activates for controller v2."""

import ast
import pathlib

import pytest

import areal.trainer.rl_trainer as rl_trainer_module
from areal.api.cli_args import SchedulingStrategy
from areal.trainer.rl_trainer import PPOTrainer


def _config(version: str, mode: str, colocated: bool):
    actor = type(
        "Actor",
        (),
        {
            "_version": version,
            "weight_update_mode": mode,
            "scheduling_strategy": SchedulingStrategy(
                type="colocation" if colocated else "separation",
                target="rollout" if colocated else "",
            ),
        },
    )()
    rollout = type("Rollout", (), {"scheduling_strategy": None})()
    return type("Cfg", (), {"actor": actor, "rollout": rollout})()


class TestV1AwexColocateGate:
    @pytest.mark.parametrize(
        "version,mode,colocated,expected",
        [
            ("v1", "awex", True, True),
            ("v1", "awex", False, True),
            ("v2", "awex", True, False),
            ("v2", "awex", False, False),
            ("v1", "xccl", True, False),
            ("v1", "disk", True, False),
        ],
    )
    def test_gate_selects_v1_awex_whatever_the_strategy(
        self, version, mode, colocated, expected
    ):
        trainer = object.__new__(PPOTrainer)
        cfg = _config(version, mode, colocated)

        assert PPOTrainer._is_v1_awex_colocate(trainer, cfg) is expected

    def test_the_default_separation_strategy_still_selects_v1_awex(self):
        """AWEX runs opt in with weight_update_mode, not with a strategy.

        examples/math/gsm8k_grpo.yaml leaves actor and rollout on the dataclass
        default, so requiring colocation here skips the meta-server handoff and
        every worker starts its own; the run then waits on 'infer_conf' forever.
        """
        trainer = object.__new__(PPOTrainer)
        default = SchedulingStrategy()

        assert default.type == "separation"
        assert (
            PPOTrainer._is_v1_awex_colocate(
                trainer, _config("v1", "awex", colocated=False)
            )
            is True
        )


class TestNoUngatedAwexChecks:
    def test_weight_update_mode_awex_is_only_compared_in_the_meta_dispatch(self):
        source = pathlib.Path(rl_trainer_module.__file__).read_text()
        tree = ast.parse(source)

        gate = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_is_v1_awex_colocate"
        )
        gate_lines = range(gate.lineno, gate.end_lineno + 1)

        bare = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            if not (
                isinstance(node.left, ast.Attribute)
                and node.left.attr == "weight_update_mode"
            ):
                continue
            if node.lineno in gate_lines:
                continue
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and comparator.value == "awex":
                    bare.append(node.lineno)

        # The meta dispatch keeps its comparison: it sits in an elif chain that
        # controller v2 already short-circuits.
        assert len(bare) <= 1, (
            'ungated `weight_update_mode == "awex"` checks at lines '
            f"{bare}; use _is_v1_awex_colocate so controller v2 separation "
            "runs are unaffected"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

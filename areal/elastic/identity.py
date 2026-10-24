# SPDX-License-Identifier: Apache-2.0

"""Separate physical measurement identity from checkpoint recovery and acceptance."""

import ast
import dataclasses
from pathlib import Path

from areal.utils.atomic import json_digest


class _WithoutDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
        return node


def semantic_digest(source: str) -> str:
    """Ignore comments, formatting and docstrings for measurement compatibility."""
    return json_digest(
        ast.dump(
            _WithoutDocstrings().visit(ast.parse(source)), include_attributes=False
        )
    )


def measurement_identity(owner) -> str:
    """Physical/training facts and timing implementation; never an acceptance proof.

    Checkpoints continue to use owner.identity and their validated queue cursor.
    Accepted bundles separately retain their exact protected-source hashes.
    Offline trace parsing and policy search code do not define measured phases.
    """
    config = owner.config
    root = Path(__file__).resolve().parents[1]
    return json_digest(
        {
            "measurement_contract": 2,
            "training": owner.identity,
            "rollout": dataclasses.asdict(config.rollout),
            "sampling": dataclasses.asdict(config.gconfig),
            "sglang": dataclasses.asdict(config.sglang),
            "tokenizer": {
                name: sha
                for name, sha in owner.base_manifest.items()
                if "token" in name or name in {"vocab.json", "merges.txt"}
            },
            "actor_gpus": owner.gpu_uuids,
            "actor_layout": {
                "backend": getattr(owner, "actor_backend", config.actor.backend),
                "world_size": getattr(owner, "actor_world_size", len(owner.gpu_uuids)),
            },
            "rollout_gpus": list(owner.options.rollout_devices),
            "cpu_hooks": {
                name: dataclasses.asdict(getattr(config, name))
                if dataclasses.is_dataclass(getattr(config, name))
                else getattr(config, name)
                for name in (
                    "saver",
                    "stats_logger",
                    "perf_tracer",
                    "evaluator",
                    "recover",
                )
            },
            "timing_implementation": {
                name: semantic_digest((root / name).read_text())
                for name in (
                    "elastic/runtime.py",
                    "trainer/rl_trainer.py",
                    "elastic/monitor/collector.py",
                    "elastic/monitor/history.py",
                )
            },
            "accounting": "conditional-phases; resident-rollout; no-makespan-claim",
        }
    )

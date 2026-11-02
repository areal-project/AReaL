# SPDX-License-Identifier: Apache-2.0

"""Load accepted Online components in their original evaluated context."""

import json
from pathlib import Path

from areal.autopilot.revision.loop import RevisionStore

from .optimizer_evaluator import VerifiedOnlineOptimizer


def _accepted(root, evaluator):
    evaluator.assert_unchanged()
    store = RevisionStore(Path(root))
    context = json.loads((store.root / "context.json").read_text())
    if (
        context["target"] != evaluator.target
        or context["benchmark_sha256"] != evaluator.benchmark_sha256
    ):
        raise ValueError(
            "Accepted component belongs to another Online evaluation context"
        )
    active = store.active()
    if active is None:
        raise ValueError("No accepted Online revision")
    return active[0], json.loads((active[0].parent / "evaluation.json").read_text())


def load_accepted_model(root, evaluator):
    path, report = _accepted(root, evaluator)
    models = evaluator.build(path.read_text())
    evaluator.verify_loaded(
        models, report["metrics"]["prediction_sha256"], report["metrics"]["source_ids"]
    )
    return models


def load_accepted_optimizer(root, evaluator):
    path, _ = _accepted(root, evaluator)
    return VerifiedOnlineOptimizer(
        path.read_text(),
        evaluator.settings,
        max_queries=evaluator.max_queries,
        max_seconds=evaluator.max_seconds,
    )

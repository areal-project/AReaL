# SPDX-License-Identifier: Apache-2.0

"""Elastic revision instructions and public API source context."""

from pathlib import Path


def proposal_context(target, *, workload_change=None, objective=None):
    """Supply target-specific instructions and reference source for the Agent."""
    reference_files = [
        "elastic/contracts.py",
        "autopilot/domains/elastic/contracts.py",
    ]
    if target == "evaluator":
        factory = "build_cost_models(calibration_cases, settings)"
        instructions = (
            "Revise the Evaluator's CostModels subcomponent. Return frozen CostModels with "
            "source_ids, rollout_config(layout), "
            "service(layout, kv_tokens, slices), actor(layout, batch), publication(actor_layout, rollout_layouts). "
            "Calibration labels are public; validation and final-holdout labels are private. "
            "Preserve unknown coverage and fixed training semantics."
        )
        reference_files.append("autopilot/domains/elastic/evaluation/evaluator.py")
    elif target == "optimizer":
        factory = "build_optimizer(settings)"
        instructions = (
            "Your candidate module must define build_optimizer(settings). "
            "This function must return an optimizer object implementing "
            "search(evaluator, view, forecasts, layouts, preferred=(), trigger='pressure'). "
            "The evaluation worker will load your module, call build_optimizer(settings), "
            "and then call search(...) on the returned object. "
            "Use only the supplied evaluate-only port: prepare_context, historical, context_id, slo, "
            "order_candidates, evaluate, evaluate_wave, parallel_available and execution_stats. "
            "Return predictions supplied by that port. The parent owns SLO/Actor-supply selection, "
            "costs, query limits and acceptance; reduce cost and Evaluator queries at the fixed horizon."
        )
        reference_files += [
            "elastic/reconfigurator/reference/optimizer.py",
            "autopilot/domains/elastic/evaluation/optimizer.py",
        ]
    else:
        raise ValueError("Online revision target must be evaluator or optimizer")

    source_root = Path(__file__).resolve().parents[3]
    context = {
        "factory": factory,
        "instructions": instructions,
        "reference_sources": {
            f"areal/{name}": (source_root / name).read_text()
            for name in reference_files
        },
    }
    if workload_change is not None:
        context["workload_change"] = workload_change
    if objective is not None:
        context["objective"] = objective
        context["instructions"] += (
            " The objective specifies complete outer Simulation evaluation. "
            "Service success and training progress are checked per case. "
            "GPU seconds are compared first; at tied GPU cost, fewer Evaluator "
            "queries count as an improvement. Use the observed workload change "
            "and earlier revision outcomes to guide the search implementation."
        )
    return context

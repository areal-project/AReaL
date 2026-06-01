"""Offline reward statistics for generated responses.

This module computes pass rates and formatting metrics for math-style
datasets using the local deepscaler reward utilities. It mirrors the
core logic used by downstream evaluation scripts but is self-contained
so training/generation does not depend on external notebooks.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from deepscaler.rewards.math_rewardv2 import deepscaler_reward_fn


def _format_grid_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a lightweight ASCII grid table without external deps."""
    # Coerce to strings and compute column widths
    rows_str = [["" if v is None else str(v) for v in row] for row in rows]
    headers_str = ["" if h is None else str(h) for h in headers]
    cols = len(headers_str)
    widths = [len(headers_str[c]) for c in range(cols)]
    for row in rows_str:
        for c in range(cols):
            widths[c] = max(widths[c], len(row[c]) if c < len(row) else 0)

    def sep(char: str = "-"):
        parts = ["+" + char * (w + 2) for w in widths]
        return "".join(parts) + "+"

    def fmt_row(values: Sequence[str]):
        padded = [f" {values[c]:<{widths[c]}} " for c in range(cols)]
        return "|" + "|".join(padded) + "|"

    lines = [sep("-"), fmt_row(headers_str), sep("=")]
    for row in rows_str:
        lines.append(fmt_row(row + [""] * (cols - len(row))))
        lines.append(sep("-"))
    return "\n".join(lines)


def _normalize_think_tags(text: str) -> str:
    """Normalize <Think> casing to lowercase tags for consistency."""
    return text.replace("<Think>", "<think>").replace("</Think>", "</think>")


def _evaluate_single_response(
    response: str,
    ground_truth: Sequence[str] | str,
    *,
    tokenizer=None,
    version: str = "v2",
    allow_nonempty_whitespace: bool = True,
    treat_no_parallel_as_format_error: bool = False,
    skip_conclusion_check: bool = False,
    require_think_end: bool = False,
    allow_immediate_stop: bool = False,
) -> Mapping[str, Any]:
    """Run deepscaler reward fn and return extra_info for stats aggregation."""
    if isinstance(ground_truth, str):
        gt_list: Sequence[str] = [ground_truth]
    elif isinstance(ground_truth, int) or isinstance(ground_truth, float):
        gt_list = [str(ground_truth)]
    else:
        gt_list = list(ground_truth)

    # Keep the reward configuration minimal; we only consume extra_info fields
    # for stats rather than using the scalar reward.
    cfg = {
        "version": version,
        # v2 path uses parallel_format_correct_v2
        "parallel_format_error_v2_reward_enabled": False,
        "parallel_format_error_v2_allow_nonempty_whitespace": allow_nonempty_whitespace,
        "treat_no_parallel_as_format_error": treat_no_parallel_as_format_error,
        "parallel_format_error_v2_skip_conclusion_check": skip_conclusion_check,
        "strip_comma_from_answer": True,
        "require_think_end": require_think_end,
        "allow_immediate_stop": allow_immediate_stop,
    }

    _, extra_info = deepscaler_reward_fn(
        _normalize_think_tags(response),
        gt_list,
        config=cfg,
        correctness_as_reward=False,
        tokenizer=tokenizer,
        verbose=False,
    )

    return extra_info


def compute_reward_stats(
    *,
    dataset: pd.DataFrame,
    responses: Sequence[Sequence[str]],
    tokenizer=None,
    version: str = "v2",
) -> dict:
    """Compute reward statistics for a full dataset of responses.

    Args:
        dataset: DataFrame that contains a "reward_model" column with a
                 "ground_truth" entry per row.
        responses: List of length N where each item is a list of size n_samples
                   containing the generated responses for that prompt.
        tokenizer: Optional HF tokenizer for token-based metrics (acceleration).
        version: Reward config version (default "v2").

    Returns:
        A summary dictionary with key metrics (pass@1, pass@n, format_pass@1, ...).
    """
    total_questions = len(dataset)
    if total_questions == 0:
        return {"questions": 0}

    assert len(responses) == total_questions, "Length of responses must equal number of dataset rows"
    n_samples = max((len(r) for r in responses), default=0)

    total_scores: list[list[float]] = []
    total_scores_strict: list[list[float]] = []
    total_format_scores: list[list[float]] = []
    per_question_results: list[dict] = []

    # Per-response aggregations for comprehensive stats
    accel_ratios_all: list[Optional[float]] = []
    parallel_counts_all: list[Optional[float]] = []
    with_parallel_all: list[float] = []
    total_num_tokens_all: list[int] = []
    longest_thread_tokens_all: list[Optional[int]] = []

    passes = 0
    passes_strict = 0

    for i in range(total_questions):
        # Extract ground truth; support both dict-like and plain value
        rm = dataset.iloc[i].get("reward_model", {})
        gt = None
        if isinstance(rm, Mapping):
            gt = rm.get("ground_truth")
        if gt is None:
            # Skip this row if no ground truth
            total_scores.append([0.0] * len(responses[i]))
            total_scores_strict.append([0.0] * len(responses[i]))
            total_format_scores.append([0.0] * len(responses[i]))
            per_question_results.append(
                {
                    "question_id": i,
                    "pass@1": 0.0,
                    "pass@1_strict": 0.0,
                    f"pass@{n_samples}": 0.0,
                    f"pass@{n_samples}_strict": 0.0,
                    "correct_samples": 0,
                    "correct_samples_strict": 0,
                    "total_samples": len(responses[i]),
                    "format_pass@1": 0.0,
                    "format_correct_samples": 0,
                    "avg_acceleration_ratio": None,
                    "valid_ratio_samples": 0,
                    "avg_parallel_count": None,
                    "valid_parallel_count_samples": 0,
                    "with_parallel@1": 0.0,
                    "with_parallel_samples": 0,
                }
            )
            continue

        stats_lst = []
        for r in list(responses[i]):
            s = _evaluate_single_response(
                r,
                gt,
                tokenizer=tokenizer,
                version=version,
            )
            stats_lst.append(s)
            # Accumulate per-response stats
            accel_ratios_all.append(s.get("acceleration_ratio"))
            pc = s.get("parallel_count")
            parallel_counts_all.append(pc)
            with_parallel_all.append(1.0 if (pc is not None and pc > 0) else 0.0)
            total_num_tokens_all.append(int(s.get("total_num_tokens", 0) or 0))
            longest_thread_tokens_all.append(s.get("num_tokens_in_the_longest_thread"))

        score_lst = [1.0 if s.get("correct_lenient", False) else 0.0 for s in stats_lst]
        score_strict_lst = [1.0 if s.get("correct", False) else 0.0 for s in stats_lst]
        format_score_lst = [float(s.get("parallel_format_correct_v2", 0.0)) for s in stats_lst]
        acceleration_ratios_lst = [s.get("acceleration_ratio") for s in stats_lst]
        parallel_counts_lst = [s.get("parallel_count") for s in stats_lst]
        with_parallel_lst = [1.0 if (pc is not None and pc > 0) else 0.0 for pc in parallel_counts_lst]

        max_score = float(np.max(score_lst)) if score_lst else 0.0
        max_score_strict = float(np.max(score_strict_lst)) if score_strict_lst else 0.0
        avg_score = float(np.mean(score_lst)) if score_lst else 0.0
        avg_score_strict = float(np.mean(score_strict_lst)) if score_strict_lst else 0.0
        avg_format_score = float(np.mean(format_score_lst)) if format_score_lst else 0.0

        valid_ratios = [r for r in acceleration_ratios_lst if r is not None]
        avg_acceleration_ratio = float(np.mean(valid_ratios)) if valid_ratios else None

        valid_parallel_counts = [pc for pc in parallel_counts_lst if pc is not None]
        avg_parallel_count = float(np.mean(valid_parallel_counts)) if valid_parallel_counts else None

        avg_with_parallel = float(np.mean(with_parallel_lst)) if with_parallel_lst else 0.0

        total_scores.append(score_lst)
        total_scores_strict.append(score_strict_lst)
        total_format_scores.append(format_score_lst)

        per_question_results.append(
            {
                "question_id": i,
                "pass@1": avg_score,
                "pass@1_strict": avg_score_strict,
                f"pass@{n_samples}": max_score,
                f"pass@{n_samples}_strict": max_score_strict,
                "correct_samples": int(np.sum(score_lst)),
                "correct_samples_strict": int(np.sum(score_strict_lst)),
                "total_samples": len(score_lst),
                "format_pass@1": avg_format_score,
                "format_correct_samples": int(np.sum(format_score_lst)),
                "avg_acceleration_ratio": avg_acceleration_ratio,
                "valid_ratio_samples": len(valid_ratios),
                "avg_parallel_count": avg_parallel_count,
                "valid_parallel_count_samples": len(valid_parallel_counts),
                "with_parallel@1": avg_with_parallel,
                "with_parallel_samples": int(np.sum(with_parallel_lst)),
            }
        )

        if max_score == 1.0:
            passes += 1
        if max_score_strict == 1.0:
            passes_strict += 1

    # Summaries across questions
    if total_questions > 0:
        pass_at_n = passes / total_questions
        pass_at_n_strict = passes_strict / total_questions
        pass_at_1 = float(np.mean(total_scores)) if total_scores else 0.0
        pass_at_1_strict = float(np.mean(total_scores_strict)) if total_scores_strict else 0.0
        format_pass_at_1 = float(np.mean(total_format_scores)) if total_format_scores else 0.0
    else:
        pass_at_n = pass_at_n_strict = pass_at_1 = pass_at_1_strict = format_pass_at_1 = 0.0

    # Additional per-question aggregates
    per_q_pass1 = [r["pass@1"] for r in per_question_results]
    per_q_pass1_strict = [r["pass@1_strict"] for r in per_question_results]
    per_q_format1 = [r["format_pass@1"] for r in per_question_results]
    accel_vals = [r["avg_acceleration_ratio"] for r in per_question_results if r["avg_acceleration_ratio"] is not None]
    par_count_vals = [r["avg_parallel_count"] for r in per_question_results if r["avg_parallel_count"] is not None]
    with_parallel_vals = [r["with_parallel@1"] for r in per_question_results]

    # Build comprehensive stats (overall per-response)
    total_items = len(accel_ratios_all)
    valid_ratios = [r for r in accel_ratios_all if r is not None]
    non_zero_ratios = [r for r in valid_ratios if r > 0]
    valid_parallel_counts = [pc for pc in parallel_counts_all if pc is not None]
    non_zero_parallel_counts = [pc for pc in valid_parallel_counts if pc > 0]
    valid_longest_thread = [t for t in longest_thread_tokens_all if t is not None]

    overall = {
        "total_items": total_items,
        "acceleration_ratio": {
            "valid": len(valid_ratios),
            "non_zero": len(non_zero_ratios),
            "total": total_items,
            "mean": float(np.mean(valid_ratios)) if valid_ratios else None,
            "std": float(np.std(valid_ratios)) if valid_ratios else None,
            "min": float(np.min(valid_ratios)) if valid_ratios else None,
            "max": float(np.max(valid_ratios)) if valid_ratios else None,
        },
        "parallel_count": {
            "valid": len(valid_parallel_counts),
            "non_zero": len(non_zero_parallel_counts),
            "total": total_items,
            "mean": float(np.mean(valid_parallel_counts)) if valid_parallel_counts else None,
            "std": float(np.std(valid_parallel_counts)) if valid_parallel_counts else None,
            "min": float(np.min(valid_parallel_counts)) if valid_parallel_counts else None,
            "max": float(np.max(valid_parallel_counts)) if valid_parallel_counts else None,
        },
        "seq_len": {
            "has_any": any(total_num_tokens_all),
            "mean": float(np.mean(total_num_tokens_all)) if any(total_num_tokens_all) else None,
            "min": int(np.min(total_num_tokens_all)) if any(total_num_tokens_all) else None,
            "max": int(np.max(total_num_tokens_all)) if any(total_num_tokens_all) else None,
        },
        "longest_thread_tokens": {
            "valid": len(valid_longest_thread),
            "total": total_items,
            "mean": float(np.mean(valid_longest_thread)) if valid_longest_thread else None,
            "std": float(np.std(valid_longest_thread)) if valid_longest_thread else None,
            "min": int(np.min(valid_longest_thread)) if valid_longest_thread else None,
            "max": int(np.max(valid_longest_thread)) if valid_longest_thread else None,
        },
    }

    summary = {
        "questions": total_questions,
        "n_samples": n_samples,
        "pass@1": pass_at_1,
        "pass@1_strict": pass_at_1_strict,
        f"pass@{n_samples}": pass_at_n,
        f"pass@{n_samples}_strict": pass_at_n_strict,
        "format_pass@1": format_pass_at_1,
        # Per-question distribution summaries (optional consumers)
        "per_question": per_question_results,
        "pass@1_mean": float(np.mean(per_q_pass1)) if per_q_pass1 else 0.0,
        "pass@1_std": float(np.std(per_q_pass1)) if per_q_pass1 else 0.0,
        "pass@1_strict_mean": float(np.mean(per_q_pass1_strict)) if per_q_pass1_strict else 0.0,
        "pass@1_strict_std": float(np.std(per_q_pass1_strict)) if per_q_pass1_strict else 0.0,
        "format_pass@1_mean": float(np.mean(per_q_format1)) if per_q_format1 else 0.0,
        "format_pass@1_std": float(np.std(per_q_format1)) if per_q_format1 else 0.0,
        "avg_acceleration_ratio_mean": float(np.mean(accel_vals)) if accel_vals else None,
        "avg_parallel_count_mean": float(np.mean(par_count_vals)) if par_count_vals else None,
        "with_parallel@1_mean": float(np.mean(with_parallel_vals)) if with_parallel_vals else 0.0,
        "overall": overall,
    }

    return summary


def format_eval_results_table(stats: Mapping[str, Any]) -> str:
    """Return a grid table for the top-level evaluation results."""
    n = int(stats.get("n_samples", 0) or 0)
    rows = [
        ["pass@1", f"{stats.get('pass@1', 0.0):.4f}"],
        ["pass@1_strict", f"{stats.get('pass@1_strict', 0.0):.4f}"],
        [f"pass@{n}", f"{stats.get(f'pass@{n}', 0.0):.4f}"],
        [f"pass@{n}_strict", f"{stats.get(f'pass@{n}_strict', 0.0):.4f}"],
        ["format_pass@1", f"{stats.get('format_pass@1', 0.0):.4f}"],
    ]
    return _format_grid_table(["Metric", "Value"], rows)


def format_comprehensive_stats_table(stats: Mapping[str, Any]) -> str:
    """Return a grid table for comprehensive overall stats (no per-sample)."""
    overall = stats.get("overall", {}) or {}
    acc = overall.get("acceleration_ratio", {})
    pc = overall.get("parallel_count", {})
    seq = overall.get("seq_len", {})
    ltt = overall.get("longest_thread_tokens", {})

    def fmt(x, nd=4):
        return "N/A" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    rows = [["Total Items", str(overall.get("total_items", 0))]]

    rows.extend([
        ["-- Acceleration Ratio --", "---"],
        ["Items with Valid Ratio", f"{acc.get('valid', 0)} / {acc.get('total', 0)}"],
        ["Items with Non-Zero Ratio", f"{acc.get('non_zero', 0)} / {acc.get('total', 0)}"],
        ["Mean Ratio", fmt(acc.get("mean"))],
        ["Std Dev Ratio", fmt(acc.get("std"))],
        ["Min Ratio", fmt(acc.get("min"))],
        ["Max Ratio", fmt(acc.get("max"))],
    ])

    rows.extend([
        ["-- Parallel Count --", "---"],
        ["Items with Valid Parallel Count", f"{pc.get('valid', 0)} / {pc.get('total', 0)}"],
        ["Items with Non-Zero Parallel Count", f"{pc.get('non_zero', 0)} / {pc.get('total', 0)}"],
        ["Mean Parallel Count", fmt(pc.get("mean"))],
        ["Std Dev Parallel Count", fmt(pc.get("std"))],
        ["Min Parallel Count", fmt(pc.get("min"))],
        ["Max Parallel Count", fmt(pc.get("max"))],
    ])

    if seq.get("has_any"):
        rows.extend([
            ["-- Sequence Length --", "---"],
            ["Mean Seq Len", fmt(seq.get("mean"), nd=2)],
            ["Min Seq Len", fmt(seq.get("min"), nd=0)],
            ["Max Seq Len", fmt(seq.get("max"), nd=0)],
        ])

    if ltt.get("valid", 0) > 0:
        rows.extend([
            ["-- Longest Thread Tokens --", "---"],
            ["Items with Valid Thread Tokens", f"{ltt.get('valid', 0)} / {ltt.get('total', 0)}"],
            ["Mean Longest Thread Tokens", fmt(ltt.get("mean"), nd=2)],
            ["Std Dev Longest Thread Tokens", fmt(ltt.get("std"), nd=2)],
            ["Min Longest Thread Tokens", fmt(ltt.get("min"), nd=0)],
            ["Max Longest Thread Tokens", fmt(ltt.get("max"), nd=0)],
        ])

    return _format_grid_table(["Statistic", "Value"], rows)

#!/usr/bin/env python3
"""Tournament aggregation over pairwise comparison results.

Reads the eval JSONL (from test_training_data_eval.py) and the original pairwise
JSONL, groups pairwise comparisons by problem, tallies wins to select the best
solution per problem, then checks against ground truth to compute pass@1.

This mirrors the aggregation logic from:
  - LLM_Test-time_Scaling/src/scaling/aggregation/full_pairwise_comparison.py
  - TTS_JSONL/run_jsonl_experiment.py

Usage:
    python3 tournament_aggregate.py \
        --eval-jsonl eval_results_run1.jsonl \
        --pairwise-jsonl pairwise_from_direct_generation_AIME25.jsonl \
        --ground-truth LLM_Test-time_Scaling/imobench.json \
        --output tournament_results_run1.json
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Boxed-answer extraction (same as grade_against_aime25.py)
# ---------------------------------------------------------------------------

def extract_last_boxed(text: str) -> Optional[str]:
    """Extract the content of the last \\boxed{...} in text."""
    marker = "\\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def normalize_answer(value: str) -> str:
    """Normalize a boxed or ground-truth answer for comparison."""
    normalized = value.strip()
    normalized = normalized.replace("$", "")
    normalized = re.sub(r"\\left|\\right", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


# ---------------------------------------------------------------------------
# JSONL parsing helpers (same patterns as grade_against_aime25.py)
# ---------------------------------------------------------------------------

def extract_problem_text(text: str) -> str:
    """Extract the problem statement from the ChatML text field."""
    m = re.search(r"Problem:\s*(.+?)(?:\n<Chunk>)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_solution_contents(text: str) -> Tuple[str, str]:
    """Extract Solution 1 and Solution 2 content from <Chunk> blocks."""
    chunks = re.findall(
        r"<Chunk>\s*Solution\s+\d+:\s*\n(.*?)\n</Chunk>", text, re.DOTALL
    )
    sol1 = chunks[0].strip() if len(chunks) >= 1 else ""
    sol2 = chunks[1].strip() if len(chunks) >= 2 else ""
    return sol1, sol2


# ---------------------------------------------------------------------------
# Ground truth loading (same as grade_against_aime25.py)
# ---------------------------------------------------------------------------

def load_ground_truths(path: str) -> Dict[str, str]:
    """Load ground truth answers, supporting imobench.json and list formats."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_map: Dict[str, str] = {}

    if isinstance(data, list):
        problems = data
    elif isinstance(data, dict):
        problems = data.get("problems", data.get("results", []))
    else:
        problems = []

    for p in problems:
        problem_text = p.get("problem", "")
        ground_truth = str(p.get("ground_truth", ""))
        if problem_text and ground_truth:
            key = re.sub(r"\s+", " ", problem_text.strip())
            gt_map[key] = ground_truth

    return gt_map


def match_ground_truth(
    problem_text: str, gt_map: Dict[str, str]
) -> Optional[str]:
    """Match a problem to its ground truth answer."""
    normalized = re.sub(r"\s+", " ", problem_text.strip())
    if normalized in gt_map:
        return gt_map[normalized]
    # Fallback: substring match on first 200 chars
    prefix = normalized[:200]
    for key, val in gt_map.items():
        if key.startswith(prefix) or prefix.startswith(key[:200]):
            return val
    return None


# ---------------------------------------------------------------------------
# Tournament logic
# ---------------------------------------------------------------------------

def run_tournament(
    pairwise_lines: List[dict],
    eval_results: List[dict],
    gt_map: Dict[str, str],
) -> Tuple[List[dict], dict]:
    """Run tournament aggregation over pairwise comparison results.

    Groups pairwise comparisons by problem, tallies wins to select the best
    solution per problem, then checks if the best solution is correct.

    Returns (problem_results, aggregate_metrics).
    """
    # Build model_verdict lookup by line_index
    verdict_by_line: Dict[int, int] = {}
    for result in eval_results:
        line_idx = result.get("line_index", -1)
        verdict_by_line[line_idx] = result.get("model_verdict", 0)

    # Parse each pairwise line and group by problem
    problem_groups: Dict[str, List[Tuple[str, str, int]]] = {}
    for line_idx, record in enumerate(pairwise_lines):
        text = record.get("text", "")
        problem_text = extract_problem_text(text)
        sol1, sol2 = extract_solution_contents(text)
        if not problem_text:
            continue
        if problem_text not in problem_groups:
            problem_groups[problem_text] = []
        problem_groups[problem_text].append((sol1, sol2, line_idx))

    # Process each problem
    problem_results = []
    correct_count = 0
    total_with_gt = 0

    for problem_text, pairs in problem_groups.items():
        # Discover unique solutions in order of first appearance
        # (same approach as TTS_JSONL/jsonl_parser.py:reconstruct_problems)
        solution_texts: List[str] = []
        solution_index_map: Dict[str, int] = {}

        for sol1, sol2, _ in pairs:
            for content in [sol1, sol2]:
                if content not in solution_index_map:
                    solution_index_map[content] = len(solution_texts)
                    solution_texts.append(content)

        n = len(solution_texts)
        wins = [0.0] * n
        comparison_details = []

        # Tally wins from model verdicts
        # (same logic as full_pairwise_comparison.py:92-99)
        for sol1, sol2, line_idx in pairs:
            i = solution_index_map[sol1]
            j = solution_index_map[sol2]
            model_verdict = verdict_by_line.get(line_idx, 0)

            if model_verdict == 1:
                wins[i] += 1
            elif model_verdict == 2:
                wins[j] += 1
            else:
                wins[i] += 0.5
                wins[j] += 0.5

            comparison_details.append({
                "sol1_idx": i,
                "sol2_idx": j,
                "line_index": line_idx,
                "model_verdict": model_verdict,
            })

        # Select best solution (most wins)
        best_idx = max(range(n), key=lambda k: wins[k])
        best_solution = solution_texts[best_idx]

        # Check ground truth
        gt_answer = match_ground_truth(problem_text, gt_map)
        is_correct = None
        best_boxed = extract_last_boxed(best_solution)

        if gt_answer is not None:
            total_with_gt += 1
            if best_boxed is not None:
                is_correct = (
                    normalize_answer(best_boxed) == normalize_answer(gt_answer)
                )
            else:
                is_correct = False
            if is_correct:
                correct_count += 1

        # Check correctness of all solutions for diagnostic output
        solution_correctness = []
        for sol_idx, sol_text in enumerate(solution_texts):
            sol_boxed = extract_last_boxed(sol_text)
            sol_correct = None
            if gt_answer is not None and sol_boxed is not None:
                sol_correct = (
                    normalize_answer(sol_boxed) == normalize_answer(gt_answer)
                )
            elif gt_answer is not None:
                sol_correct = False
            solution_correctness.append({
                "solution_idx": sol_idx,
                "boxed_answer": sol_boxed,
                "is_correct": sol_correct,
                "wins": wins[sol_idx],
            })

        problem_results.append({
            "problem_text": problem_text[:120],
            "n_solutions": n,
            "n_pairs": len(pairs),
            "best_solution_idx": best_idx,
            "best_solution_wins": wins[best_idx],
            "best_solution_boxed": best_boxed,
            "ground_truth_answer": gt_answer,
            "is_correct": is_correct,
            "wins": wins,
            "solution_correctness": solution_correctness,
            "comparison_details": comparison_details,
        })

    pass_at_1 = correct_count / total_with_gt if total_with_gt > 0 else 0.0

    aggregate_metrics = {
        "pass@1": pass_at_1,
        "total_problems": len(problem_groups),
        "problems_with_gt": total_with_gt,
        "correct_problems": correct_count,
    }

    return problem_results, aggregate_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-jsonl", required=True,
        help="Path to JSONL file produced by test_training_data_eval.py",
    )
    parser.add_argument(
        "--pairwise-jsonl", required=True,
        help="Path to the original pairwise JSONL dataset",
    )
    parser.add_argument(
        "--ground-truth", required=True,
        help="Path to ground truth file (imobench.json)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: auto-generated with timestamp)",
    )
    args = parser.parse_args()

    # Load ground truths
    print(f"Loading ground truth: {args.ground_truth}")
    gt_map = load_ground_truths(args.ground_truth)
    print(f"  Loaded {len(gt_map)} problems")

    # Load pairwise JSONL
    print(f"Loading pairwise JSONL: {args.pairwise_jsonl}")
    pairwise_lines = []
    with open(args.pairwise_jsonl, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                pairwise_lines.append(json.loads(raw))
    print(f"  Loaded {len(pairwise_lines)} lines")

    # Load eval results
    print(f"Loading eval results: {args.eval_jsonl}")
    eval_results = []
    with open(args.eval_jsonl, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                eval_results.append(json.loads(raw))
    print(f"  Loaded {len(eval_results)} results")

    # Run tournament
    problem_results, metrics = run_tournament(pairwise_lines, eval_results, gt_map)

    # Output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.eval_jsonl).parent
        args.output = str(out_dir / f"tournament_results_{timestamp}.json")

    output_data = {
        "eval_jsonl": str(args.eval_jsonl),
        "pairwise_jsonl": str(args.pairwise_jsonl),
        "ground_truth_file": str(args.ground_truth),
        "timestamp": datetime.now().isoformat(),
        "aggregate_metrics": metrics,
        "problem_results": problem_results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    p = metrics
    print(f"\n{'=' * 70}")
    print("TOURNAMENT AGGREGATION SUMMARY")
    print(f"  Total problems:         {p['total_problems']}")
    print(f"  Problems with GT:       {p['problems_with_gt']}")
    print(f"  Correct (best sol):     {p['correct_problems']}")
    print(f"  ---")
    print(f"  Pass@1:                 {p['pass@1']:.4f} "
          f"({p['correct_problems']}/{p['problems_with_gt']})")
    print()

    # Per-problem breakdown
    for i, pr in enumerate(problem_results):
        if pr["is_correct"] is None:
            status = "?"
        elif pr["is_correct"]:
            status = "Y"
        else:
            status = "N"
        correct_sols = sum(
            1 for sc in pr["solution_correctness"] if sc["is_correct"]
        )
        print(
            f"  Problem {i+1}: [{status}] best=sol{pr['best_solution_idx']} "
            f"(wins={pr['best_solution_wins']:.1f}/{pr['n_pairs']}) "
            f"boxed={pr['best_solution_boxed']} gt={pr['ground_truth_answer']} "
            f"| {correct_sols}/{pr['n_solutions']} sols correct"
        )

    print(f"\nOutput: {args.output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

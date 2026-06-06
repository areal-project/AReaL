#!/usr/bin/env python3
"""Grade pairwise evaluation results against AIME25 ground truth.

Reads the JSONL output from test_training_data_eval.py, looks up each line in
the original pairwise JSONL to extract \boxed{} answers from both solutions,
compares them against the ground truth in imobench.json, and determines whether
the model's verdict agrees with the ground-truth-derived verdict.

Usage:
    python3 grade_against_aime25.py \
        --eval-jsonl eval_results_20260421_120000.jsonl \
        --pairwise-jsonl pairwise_from_direct_generation_AIME25.jsonl \
        --ground-truth LLM_Test-time_Scaling/imobench.json
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Boxed-answer extraction (from src/evaluation/boxed_imobench_evaluator.py)
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
# Pairwise JSONL parsing
# ---------------------------------------------------------------------------

def extract_problem_text(text: str) -> str:
    """Extract the problem statement from the ChatML text field."""
    m = re.search(r"Problem:\s*(.+?)(?:\n<Chunk>)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def extract_solution_contents(text: str) -> Tuple[str, str]:
    """Extract Solution 1 and Solution 2 content from <Chunk> blocks."""
    chunks = re.findall(r"<Chunk>\s*Solution\s+\d+:\s*\n(.*?)\n</Chunk>", text, re.DOTALL)
    sol1 = chunks[0].strip() if len(chunks) >= 1 else ""
    sol2 = chunks[1].strip() if len(chunks) >= 2 else ""
    return sol1, sol2


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truths(path: str) -> Dict[str, str]:
    """Load imobench.json and return dict mapping normalized problem text -> ground truth answer."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_map = {}
    problems = data.get("problems", [])
    for p in problems:
        problem_text = p.get("problem", "")
        ground_truth = str(p.get("ground_truth", ""))
        if problem_text and ground_truth:
            key = re.sub(r"\s+", " ", problem_text.strip())
            gt_map[key] = ground_truth
    return gt_map


def match_ground_truth(problem_text: str, gt_map: Dict[str, str]) -> Optional[str]:
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
# Main grading logic
# ---------------------------------------------------------------------------

def grade_results(
    eval_results: List[dict],
    pairwise_lines: List[dict],
    gt_map: Dict[str, str],
) -> Tuple[List[dict], dict]:
    """Grade each eval result against ground truth.

    Returns (graded_results, aggregate_metrics).
    """
    graded = []

    gt_match = 0
    gt_mismatch = 0
    gt_tie = 0
    gt_model_unparseable = 0
    gt_no_boxed = 0
    gt_no_ground_truth = 0
    ref_match = 0
    ref_mismatch = 0
    sol1_correct_count = 0
    sol2_correct_count = 0
    both_correct_count = 0
    neither_correct_count = 0
    errors = 0

    for result in eval_results:
        line_idx = result.get("line_index", -1)
        model_verdict = result.get("model_verdict", 0)
        reference_verdict = result.get("reference_verdict", 0)
        ref_status = result.get("status", "")

        # Count ref-based stats
        if ref_status == "MATCH":
            ref_match += 1
        elif ref_status == "MISMATCH":
            ref_mismatch += 1

        # Look up original pairwise data
        if line_idx < 0 or line_idx >= len(pairwise_lines):
            errors += 1
            graded.append({
                "line_index": line_idx,
                "gt_status": "ERROR",
                "error": f"line_index {line_idx} out of range",
            })
            continue

        original = pairwise_lines[line_idx]
        text = original.get("text", "")

        problem_text = extract_problem_text(text)
        sol1_content, sol2_content = extract_solution_contents(text)

        # Match to ground truth
        gt_answer = match_ground_truth(problem_text, gt_map)
        if gt_answer is None:
            gt_no_ground_truth += 1
            graded.append({
                "line_index": line_idx,
                "problem_text": problem_text[:120],
                "gt_status": "GT_NO_GROUND_TRUTH",
                "model_verdict": model_verdict,
                "reference_verdict": reference_verdict,
                "ref_status": ref_status,
            })
            continue

        # Extract boxed answers
        sol1_boxed = extract_last_boxed(sol1_content)
        sol2_boxed = extract_last_boxed(sol2_content)

        if sol1_boxed is None and sol2_boxed is None:
            gt_no_boxed += 1
            graded.append({
                "line_index": line_idx,
                "problem_text": problem_text[:120],
                "ground_truth_answer": gt_answer,
                "sol1_boxed": None,
                "sol2_boxed": None,
                "gt_status": "GT_NO_BOXED",
                "model_verdict": model_verdict,
                "reference_verdict": reference_verdict,
                "ref_status": ref_status,
            })
            continue

        # Check correctness
        sol1_correct = (
            sol1_boxed is not None
            and normalize_answer(sol1_boxed) == normalize_answer(gt_answer)
        )
        sol2_correct = (
            sol2_boxed is not None
            and normalize_answer(sol2_boxed) == normalize_answer(gt_answer)
        )

        if sol1_correct:
            sol1_correct_count += 1
        if sol2_correct:
            sol2_correct_count += 1
        if sol1_correct and sol2_correct:
            both_correct_count += 1
        if not sol1_correct and not sol2_correct:
            neither_correct_count += 1

        # Derive ground-truth verdict
        if sol1_correct and not sol2_correct:
            gt_verdict = 1
        elif sol2_correct and not sol1_correct:
            gt_verdict = 2
        else:
            gt_verdict = 0  # tie (both correct or both incorrect)

        # Compare model verdict against ground-truth verdict
        if gt_verdict == 0:
            gt_tie += 1
            gt_status = "GT_TIE"
        elif model_verdict == 0:
            gt_model_unparseable += 1
            gt_status = "GT_MODEL_UNPARSEABLE"
        elif model_verdict == gt_verdict:
            gt_match += 1
            gt_status = "GT_MATCH"
        else:
            gt_mismatch += 1
            gt_status = "GT_MISMATCH"

        graded.append({
            "line_index": line_idx,
            "problem_text": problem_text[:120],
            "ground_truth_answer": gt_answer,
            "sol1_boxed": sol1_boxed,
            "sol2_boxed": sol2_boxed,
            "sol1_correct": sol1_correct,
            "sol2_correct": sol2_correct,
            "gt_verdict": gt_verdict,
            "model_verdict": model_verdict,
            "reference_verdict": reference_verdict,
            "gt_status": gt_status,
            "ref_status": ref_status,
        })

    parseable = gt_match + gt_mismatch
    gt_accuracy = (gt_match / parseable) if parseable > 0 else 0.0

    aggregate_metrics = {
        "pass@1": gt_accuracy,
        "gt_match": gt_match,
        "gt_mismatch": gt_mismatch,
        "gt_tie": gt_tie,
        "gt_model_unparseable": gt_model_unparseable,
        "gt_no_boxed": gt_no_boxed,
        "gt_no_ground_truth": gt_no_ground_truth,
        "gt_parseable": parseable,
        "ref_match": ref_match,
        "ref_mismatch": ref_mismatch,
        "total_evaluated": len(eval_results),
        "errors": errors,
        "sol1_correct_count": sol1_correct_count,
        "sol2_correct_count": sol2_correct_count,
        "both_correct_count": both_correct_count,
        "neither_correct_count": neither_correct_count,
    }

    return graded, aggregate_metrics


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
        help="Path to imobench.json ground truth file",
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

    # Load original pairwise JSONL
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

    # Grade
    graded, metrics = grade_results(eval_results, pairwise_lines, gt_map)

    # Output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.eval_jsonl).parent
        args.output = str(out_dir / f"graded_results_{timestamp}.json")

    output_data = {
        "eval_jsonl": str(args.eval_jsonl),
        "pairwise_jsonl": str(args.pairwise_jsonl),
        "ground_truth_file": str(args.ground_truth),
        "timestamp": datetime.now().isoformat(),
        "aggregate_metrics": metrics,
        "results": graded,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    p = metrics
    print(f"\n{'=' * 70}")
    print("GROUND TRUTH GRADING SUMMARY")
    print(f"  Total evaluated:        {p['total_evaluated']}")
    print(f"  GT match:               {p['gt_match']}")
    print(f"  GT mismatch:            {p['gt_mismatch']}")
    print(f"  GT tie (both same):     {p['gt_tie']}")
    print(f"  GT model unparseable:   {p['gt_model_unparseable']}")
    print(f"  GT no boxed answer:     {p['gt_no_boxed']}")
    print(f"  GT no ground truth:     {p['gt_no_ground_truth']}")
    print(f"  Errors:                 {p['errors']}")
    print(f"  ---")
    print(f"  Ref match:              {p['ref_match']}")
    print(f"  Ref mismatch:           {p['ref_mismatch']}")
    print(f"  ---")
    print(f"  Sol1 correct:           {p['sol1_correct_count']}")
    print(f"  Sol2 correct:           {p['sol2_correct_count']}")
    print(f"  Both correct:           {p['both_correct_count']}")
    print(f"  Neither correct:        {p['neither_correct_count']}")
    print(f"  ---")
    parseable = p['gt_parseable']
    accuracy = p['pass@1']
    print(f"  GT Accuracy (pass@1):   {accuracy:.4f} ({p['gt_match']}/{parseable})")
    print(f"\nOutput: {args.output}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

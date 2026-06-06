"""JSONL parsing and solution reconstruction for pairwise comparison datasets.

Parses JSONL files where each line is a pre-formatted pairwise comparison entry
in ChatML format, reconstructing per-problem data structures for the pipeline.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SolutionPair:
    """A single pairwise comparison entry from the JSONL."""
    problem_text: str
    sol1_content: str
    sol2_content: str
    gt_label: int  # 1 or 2 (which solution is better according to ground truth)
    raw_prompt: str  # system+user portion to send to vLLM (up to <|im_start|>assistant\n)
    line_index: int


@dataclass
class ProblemData:
    """Reconstructed per-problem data."""
    problem_text: str
    solutions: List[str]  # Unique solution texts in first-appearance order
    pairs: List[Tuple[int, int, SolutionPair]]  # (sol_i_idx, sol_j_idx, pair)
    ground_truth_answer: Optional[str] = None


def parse_jsonl(jsonl_path: str) -> List[SolutionPair]:
    """Parse JSONL file and extract pairwise comparison entries.

    Each line has {"text": "<|im_start|>system\\n...assistant\\n..."}.
    We extract problem, solutions, ground-truth label, and the raw prompt.
    """
    pairs = []
    path = Path(jsonl_path)

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            text = record["text"]

            # Split into prompt (system+user) and assistant response
            assistant_marker = "<|im_start|>assistant\n"
            marker_pos = text.rfind(assistant_marker)
            if marker_pos < 0:
                print(f"  Warning: line {line_idx}: no assistant marker found, skipping")
                continue

            raw_prompt = text[: marker_pos + len(assistant_marker)]
            assistant_response = text[marker_pos + len(assistant_marker):]

            # Extract problem text: between "Problem: " and "\n<Chunk>"
            prob_match = re.search(r"Problem: (.*?)\n<Chunk>", text, re.DOTALL)
            if not prob_match:
                print(f"  Warning: line {line_idx}: could not extract problem text, skipping")
                continue
            problem_text = prob_match.group(1).strip()

            # Extract solutions from <Chunk> blocks
            chunk_pattern = re.compile(
                r"<Chunk>\nSolution \d+:\n\n(.*?)\n</Chunk>", re.DOTALL
            )
            chunks = chunk_pattern.findall(text)
            if len(chunks) < 2:
                print(f"  Warning: line {line_idx}: found {len(chunks)} chunks (expected 2), skipping")
                continue

            sol1_content = chunks[0]
            sol2_content = chunks[1]

            # Extract ground-truth label from assistant response
            label_match = re.search(
                r"Better Solution:\s*Solution\s+([12])", assistant_response, re.IGNORECASE
            )
            gt_label = int(label_match.group(1)) if label_match else 0

            pairs.append(SolutionPair(
                problem_text=problem_text,
                sol1_content=sol1_content,
                sol2_content=sol2_content,
                gt_label=gt_label,
                raw_prompt=raw_prompt,
                line_index=line_idx,
            ))

    return pairs


def group_by_problem(pairs: List[SolutionPair]) -> Dict[str, List[SolutionPair]]:
    """Group pairwise entries by problem text."""
    grouped: Dict[str, List[SolutionPair]] = {}
    for pair in pairs:
        key = pair.problem_text
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(pair)
    return grouped


def reconstruct_problems(
    grouped: Dict[str, List[SolutionPair]],
    ground_truths: Optional[Dict[str, str]] = None,
) -> List[ProblemData]:
    """Reconstruct per-problem data with solution indices and pair mappings.

    Discovers unique solutions in first-appearance order (matching the original
    indices from itertools.combinations(enumerate(solutions), 2) generation).
    """
    problems = []

    for problem_text, pair_list in grouped.items():
        # Discover unique solutions in order of first appearance
        solution_texts: List[str] = []
        solution_index_map: Dict[str, int] = {}  # content -> index

        for pair in pair_list:
            for content in [pair.sol1_content, pair.sol2_content]:
                if content not in solution_index_map:
                    solution_index_map[content] = len(solution_texts)
                    solution_texts.append(content)

        # Map each pair to (i, j, pair_data)
        indexed_pairs: List[Tuple[int, int, SolutionPair]] = []
        for pair in pair_list:
            i = solution_index_map[pair.sol1_content]
            j = solution_index_map[pair.sol2_content]
            indexed_pairs.append((i, j, pair))

        # Look up ground truth answer
        gt_answer = None
        if ground_truths:
            # Try exact match first
            gt_answer = ground_truths.get(problem_text)
            if gt_answer is None:
                # Try normalized match
                norm_problem = _normalize_text(problem_text)
                for gt_prob, gt_ans in ground_truths.items():
                    if _normalize_text(gt_prob) == norm_problem:
                        gt_answer = gt_ans
                        break

        problems.append(ProblemData(
            problem_text=problem_text,
            solutions=solution_texts,
            pairs=indexed_pairs,
            ground_truth_answer=gt_answer,
        ))

    return problems


def load_ground_truths(gt_path: str) -> Dict[str, str]:
    """Load ground truth answers from a JSON file.

    Supports two formats:
    1. direct_generation JSON: {"results": [{"problem": ..., "ground_truth": ...}]}
    2. imobench.json: [{"problem": ..., "ground_truth": ...}] (list of problems)
    """
    path = Path(gt_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ground_truths: Dict[str, str] = {}

    if isinstance(data, list):
        # imobench.json format: list of problem objects
        for item in data:
            problem = item.get("problem", "")
            gt = item.get("ground_truth", "")
            if problem and gt:
                ground_truths[problem] = str(gt)
    elif isinstance(data, dict):
        # direct_generation format
        results = data.get("results", [])
        if not results and "problems" in data:
            results = data["problems"]
        for item in results:
            problem = item.get("problem", "")
            gt = item.get("ground_truth", "")
            if problem and gt:
                ground_truths[problem] = str(gt)

    return ground_truths


def _normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching (collapse whitespace, strip)."""
    return " ".join(text.split()).strip()


# --- CLI for standalone testing ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse and validate JSONL pairwise dataset")
    parser.add_argument("--jsonl-file", required=True, help="Path to pairwise JSONL file")
    parser.add_argument("--ground-truth-file", help="Optional ground truth JSON file")
    args = parser.parse_args()

    print(f"Parsing: {args.jsonl_file}")
    pairs = parse_jsonl(args.jsonl_file)
    print(f"  Total pairs: {len(pairs)}")

    grouped = group_by_problem(pairs)
    print(f"  Unique problems: {len(grouped)}")

    gt = load_ground_truths(args.ground_truth_file) if args.ground_truth_file else None
    if gt:
        print(f"  Ground truths loaded: {len(gt)}")

    problems = reconstruct_problems(grouped, gt)

    print(f"\nPer-problem breakdown:")
    for i, prob in enumerate(problems):
        gt_status = f"GT={prob.ground_truth_answer}" if prob.ground_truth_answer else "no GT"
        print(f"  Problem {i+1}: {len(prob.solutions)} solutions, {len(prob.pairs)} pairs ({gt_status})")
        print(f"    Text: {prob.problem_text[:80]}...")

    # Sanity check
    total_expected_pairs = sum(
        len(p.solutions) * (len(p.solutions) - 1) // 2 for p in problems
    )
    print(f"\n  Expected total pairs (C(n,2) sum): {total_expected_pairs}")
    print(f"  Actual total pairs: {len(pairs)}")
    if total_expected_pairs == len(pairs):
        print("  PASS: pair count matches expected C(n,2) per problem")
    else:
        print("  WARNING: pair count mismatch!")

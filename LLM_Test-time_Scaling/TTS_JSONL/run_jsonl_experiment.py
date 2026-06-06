#!/usr/bin/env python3
"""Main orchestrator for JSONL pairwise test-time scaling experiments.

Parses a JSONL file containing pairwise comparison entries, sends prompts to a
vLLM server for model-based comparison, tallies wins to select the best solution
per problem, grades the selected solution, and computes aggregate metrics.

Usage:
    python3 TTS_JSONL/run_jsonl_experiment.py \
        --jsonl-file pairwise_from_direct_generation_AIME25.jsonl \
        --ground-truth-file LLM_Test-time_Scaling/imobench.json \
        --model-name qwen3-4b-newenc \
        --api-base http://127.0.0.1:8000/v1 \
        --eval-model-name qwen3-8b-eval \
        --eval-api-base http://127.0.0.1:8001/v1 \
        --output-dir results/tts_jsonl
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

from jsonl_parser import (
    ProblemData,
    load_ground_truths,
    group_by_problem,
    parse_jsonl,
    reconstruct_problems,
)
from vllm_client import extract_response_text, post_completion
from response_parser import parse_comparison
from grading import grade_solution


async def process_problem(
    problem: ProblemData,
    problem_idx: int,
    total_problems: int,
    session: aiohttp.ClientSession,
    api_base: str,
    model_name: str,
    eval_api_base: str,
    eval_model_name: str,
    temperature: float,
    max_tokens: int,
    request_timeout: int,
    semaphore: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
    """Process a single problem: run all pairwise comparisons, select best, grade it."""
    async with semaphore:
        n_solutions = len(problem.solutions)
        n_pairs = len(problem.pairs)
        problem_id = f"problem_{problem_idx + 1}"
        short_text = problem.problem_text[:60].replace("\n", " ")

        # Run all pairwise comparisons concurrently
        wins = [0.0] * n_solutions
        comparison_results = []

        async def compare_pair(i: int, j: int, pair):
            """Send one pairwise comparison to the model."""
            try:
                resp_json = await post_completion(
                    session=session,
                    api_base=api_base,
                    model_name=model_name,
                    prompt=pair.raw_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=request_timeout,
                )
                raw_text = resp_json.get("choices", [{}])[0].get("text", "")
                content, reasoning, usage = extract_response_text(resp_json)
                winner = parse_comparison(content)
                print(f"    [{problem_id}] Pair ({i},{j}) winner={winner}")
                print(f"      Raw response (first 500 chars): {raw_text[:500]}")
                if reasoning:
                    print(f"      Reasoning (first 300 chars): {reasoning[:300]}")
                print(f"      Parsed content (first 300 chars): {content[:300]}")
                return (i, j, winner, content, usage)
            except Exception as e:
                print(f"    [{problem_id}] Pair ({i},{j}) error: {e}")
                return (i, j, 0, "", {})

        # Dispatch all pair comparisons concurrently
        tasks = [compare_pair(i, j, pair) for i, j, pair in problem.pairs]
        results = await asyncio.gather(*tasks)

        # Tally wins
        for i, j, winner, content, usage in results:
            if winner == 1:
                wins[i] += 1
            elif winner == 2:
                wins[j] += 1
            else:
                wins[i] += 0.5
                wins[j] += 0.5

            comparison_results.append({
                "sol1_idx": i,
                "sol2_idx": j,
                "winner": winner,
                "response_snippet": content[:200] if content else "",
            })

        # Select best solution
        best_idx = max(range(n_solutions), key=lambda k: wins[k])
        best_solution = problem.solutions[best_idx]

        # Grade the best solution
        is_correct = False
        score = 0.0
        feedback = None

        if problem.ground_truth_answer:
            try:
                grade_result = await grade_solution(
                    session=session,
                    api_base=eval_api_base,
                    model_name=eval_model_name,
                    problem=problem.problem_text,
                    solution=best_solution,
                    ground_truth=problem.ground_truth_answer,
                    temperature=0.0,
                    max_tokens=4096,
                    timeout=request_timeout,
                )
                is_correct = grade_result["is_correct"]
                score = grade_result["score"]
                feedback = grade_result["feedback"]
            except Exception as e:
                print(f"    [{problem_id}] Grading error: {e}")
                feedback = f"grading_error: {e}"
        else:
            feedback = "no_ground_truth_available"

        status = "\u2713" if is_correct else "\u2717"
        print(
            f"  [{problem_idx + 1}/{total_problems}] {problem_id}: {status} "
            f"(best=sol{best_idx}, wins={wins[best_idx]:.1f}/{n_pairs}, score={score:.2f}) "
            f"| {short_text}...",
            flush=True,
        )

        return {
            "problem_id": problem_id,
            "problem": problem.problem_text,
            "ground_truth": problem.ground_truth_answer,
            "is_correct": is_correct,
            "score": score,
            "feedback": feedback,
            "solution_content": best_solution,
            "best_solution_idx": best_idx,
            "wins": wins,
            "n_solutions": n_solutions,
            "n_pairs": n_pairs,
            "comparison_results": comparison_results,
        }


async def run_experiment(args) -> Dict[str, Any]:
    """Run the full experiment pipeline."""
    start_time = datetime.now()

    # 1. Parse JSONL
    print(f"Parsing JSONL: {args.jsonl_file}")
    pairs = parse_jsonl(args.jsonl_file)
    print(f"  Total pairs: {len(pairs)}")

    # 2. Group by problem
    grouped = group_by_problem(pairs)
    print(f"  Unique problems: {len(grouped)}")

    # 3. Load ground truths
    print(f"Loading ground truths: {args.ground_truth_file}")
    ground_truths = load_ground_truths(args.ground_truth_file)
    print(f"  Ground truths loaded: {len(ground_truths)}")

    # 4. Reconstruct problems
    problems = reconstruct_problems(grouped, ground_truths)
    print(f"  Problems reconstructed: {len(problems)}")
    for i, p in enumerate(problems):
        gt_status = "GT" if p.ground_truth_answer else "no-GT"
        print(f"    {i+1}. {len(p.solutions)} solutions, {len(p.pairs)} pairs ({gt_status})")

    # 5. Process all problems
    print(f"\nRunning pairwise comparisons + grading...")
    print(f"  Model: {args.model_name} @ {args.api_base}")
    print(f"  Eval: {args.eval_model_name} @ {args.eval_api_base}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max concurrent: {args.max_concurrent}")
    print()

    semaphore = asyncio.Semaphore(args.max_concurrent)
    connector = aiohttp.TCPConnector(limit=args.max_concurrent * 2)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            process_problem(
                problem=prob,
                problem_idx=idx,
                total_problems=len(problems),
                session=session,
                api_base=args.api_base,
                model_name=args.model_name,
                eval_api_base=args.eval_api_base,
                eval_model_name=args.eval_model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                request_timeout=args.request_timeout,
                semaphore=semaphore,
            )
            for idx, prob in enumerate(problems)
        ]
        results = await asyncio.gather(*tasks)

    # Filter out None results
    results = [r for r in results if r is not None]

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 6. Compute aggregate metrics
    valid_results = [r for r in results if r.get("ground_truth") is not None]
    correct_count = sum(1 for r in valid_results if r["is_correct"])
    total_count = len(valid_results)
    avg_score = (
        sum(r["score"] for r in valid_results) / total_count if total_count > 0 else 0.0
    )
    pass_at_1 = correct_count / total_count if total_count > 0 else 0.0

    aggregate_metrics = {
        "pass@1": pass_at_1,
        "total_problems": total_count,
        "correct_problems": correct_count,
        "avg_score": avg_score,
    }

    # 7. Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"jsonl_pairwise_experiment_{timestamp}.json"

    output_data = {
        "source_file": str(args.jsonl_file),
        "ground_truth_file": str(args.ground_truth_file),
        "model_name": args.model_name,
        "eval_model_name": args.eval_model_name,
        "temperature": args.temperature,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "aggregate_metrics": aggregate_metrics,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Experiment completed in {duration:.1f}s")
    print(f"  Pass@1: {pass_at_1:.4f} ({correct_count}/{total_count})")
    print(f"  Avg Score: {avg_score:.4f}")
    print(f"  Output: {output_file}")
    print(f"{'=' * 60}")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Run JSONL pairwise test-time scaling experiment"
    )
    parser.add_argument(
        "--jsonl-file", required=True, help="Path to pairwise JSONL file"
    )
    parser.add_argument(
        "--ground-truth-file", required=True,
        help="Path to ground truth JSON (imobench.json or direct_generation JSON)",
    )
    parser.add_argument(
        "--model-name", required=True, help="Pairwise comparison model name"
    )
    parser.add_argument(
        "--api-base", default="http://127.0.0.1:8000/v1",
        help="vLLM server URL for pairwise comparison (default: http://127.0.0.1:8000/v1)",
    )
    parser.add_argument(
        "--eval-model-name", required=True, help="Evaluation/grading model name"
    )
    parser.add_argument(
        "--eval-api-base", default="http://127.0.0.1:8001/v1",
        help="vLLM eval server URL (default: http://127.0.0.1:8001/v1)",
    )
    parser.add_argument(
        "--output-dir", default="results/tts_jsonl",
        help="Output directory (default: results/tts_jsonl)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for pairwise comparisons (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=16384,
        help="Max tokens for generation (default: 16384)",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=40,
        help="Max concurrent requests (default: 40)",
    )
    parser.add_argument(
        "--request-timeout", type=int, default=600,
        help="HTTP request timeout in seconds (default: 600)",
    )

    args = parser.parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()

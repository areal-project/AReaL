"""Run aggregation experiments on LCB Pro result files.

This script reads LCB Pro result files containing multiple solutions per problem,
and applies different aggregation strategies to combine them.

The script extracts all solutions with metadata.iteration == 0 from all result files.
For each experiment, it randomly samples --solutions-per-exp solutions from all available
iteration=0 solutions.

Usage:
    # Single file:
    python -m scripts.run_lcb_pro_aggregation_experiment \
        --result-files results.json \
        --output-dir aggregation_experiments \
        --solutions-per-exp 4 \
        --service-url service_addr
    
    # Multiple files (all iteration=0 solutions are extracted):
    python -m aggregation_scripts.lcb_pro_aggregation_random \
        --result-files results_full_continuous/baseline_8_fixed/direct_evaluation_remote_lcb_pro_20260116_234529.json \
            results_full_continuous/fixed_effort/self_eval_7trials/lcb_pro_rollouts_gpt-oss-120b/self_eval_sequential_lcb_pro_20260118_112007.json \
            results_full_continuous/self_eval/lcb_pro_rollouts_gpt-oss-120b/code_execution_detailed_sequential_lcb_pro_20260113_160523.json \
        --output-dir aggregation_experiments/coding_random/16_traj_auto_effort \
        --solutions-per-exp 16 \
        --num-exps 1 \
        --reasoning-effort auto \
        --iteration 0 \
        --resume 
        
    python -m aggregation_scripts.lcb_pro_aggregation_random \
        --result-files results_full_continuous/baseline_8_fixed/direct_evaluation_remote_lcb_pro_20260116_234529.json \
        --output-dir aggregation_experiments/coding_random/2_traj_auto_effort_changed \
        --solutions-per-exp 2 \
        --num-exps 4 \
        --reasoning-effort auto \
        --service-url service_addr1,service_addr2,...

    python -m aggregation_scripts.lcb_pro_aggregation_random \
        --result-files path-to-results/LLM_Test-time_Scaling/results_full_continuous/fixed_effort/self_eval_7trials/lcb_pro_rollouts_gpt-oss-120b/self_eval_sequential_lcb_pro_20260118_112007.json \
            path-to-results/LLM_Test-time_Scaling/results_full_continuous/self_eval/lcb_pro_rollouts_gpt-oss-120b/code_execution_detailed_sequential_lcb_pro_20260113_160523.json \ 
        --output-dir aggregation_experiments/coding_random/8x8_traj_auto_effort \
        --solutions-per-exp 8 \
        --num-exps 4 \
        --iteration 8 \
        --reasoning-effort auto \
        --service-url service_addr1,service_addr2,...

    # Multiple experiments with random sampling:
    python -m scripts.run_lcb_pro_aggregation_experiment \
        --result-files results1.json results2.json \
        --output-dir aggregation_experiments \
        --solutions-per-exp 4 \
        --num-exps 5 \
        --service-url service_addr
"""
import argparse
import asyncio
import json
import os
import random
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.asyncio import tqdm

import sys
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import RemoteLCBProEvaluator
from src.evaluation.lcb_pro_evaluator import extract_code
from src.llm_service import create_llm_service
from src.prompts import PromptManager
from src.scaling.aggregation import (
    GenerateFromNAggregation,
    GTScoringAggregation,
    LLMScoringAggregation,
    LLMVotingAggregation,
    PairwiseComparisonAggregation,
    SelectBestAggregation,
    VotingAggregation,
)
from src.scaling.base import Solution
from src.utils.config import Config, LLMConfig, EvaluationConfig


def load_result_file(result_file: Path) -> Dict[str, Any]:
    """Load result file and extract problems with solutions.
    
    Args:
        result_file: Path to result JSON file
        
    Returns:
        Dictionary with config and results
    """
    print(f"Loading result file: {result_file}")
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    config_dict = data.get("config", {})
    results = data.get("results", [])
    benchmark = data.get("benchmark", "lcb_pro")
    
    print(f"Loaded {len(results)} problem results")
    print(f"Benchmark: {benchmark}")
    
    return {
        "config": config_dict,
        "results": results,
        "benchmark": benchmark,
    }


def merge_result_files(result_files: List[Path], iteration: int = 0) -> Dict[str, Any]:
    """Merge multiple result files, combining solutions by problem_id.
    
    For each problem, extracts all solutions with metadata.iteration == iteration from all files.
    
    Args:
        result_files: List of paths to result JSON files
        iteration: Iteration number to extract solutions from (default: 0)
        
    Returns:
        Dictionary with merged config, results, and benchmark
    """
    if not result_files:
        raise ValueError("At least one result file is required")
    
    print(f"\nMerging {len(result_files)} result files...")
    print(f"Extracting all solutions with metadata.iteration == {iteration}")
    
    # Load all files
    all_data = []
    for result_file in result_files:
        data = load_result_file(result_file)
        all_data.append(data)
        print(f"  File {len(all_data)}: {result_file.name}")
    
    # Use config and benchmark from first file
    config_dict = all_data[0]["config"]
    benchmark = all_data[0]["benchmark"]
    
    # Build problem_id -> results mapping
    problem_map: Dict[str, List[Dict[str, Any]]] = {}
    for data in all_data:
        for result in data["results"]:
            problem_id = result.get("problem_id", "")
            if problem_id not in problem_map:
                problem_map[problem_id] = []
            problem_map[problem_id].append(result)
    
    # Merge solutions for each problem
    merged_results = []
    total_iteration_solutions = 0
    for problem_id, problem_results in problem_map.items():
        if not problem_results:
            continue
        
        # Use problem info from first result
        merged_result = problem_results[0].copy()
        
        # Combine all solutions with iteration == iteration from all files
        combined_solutions = []
        for file_result_idx, result in enumerate(problem_results):
            all_solutions = result.get("all_solutions", [])
            
            for sol_idx, sol_data in enumerate(all_solutions):
                # Check if this solution has iteration == iteration in metadata
                sol_metadata = sol_data.get("metadata", {})
                sol_iteration = sol_metadata.get("iteration")
                
                if sol_iteration == iteration:
                    sol_data_copy = sol_data.copy()
                    # Add source file info to metadata
                    if "metadata" not in sol_data_copy:
                        sol_data_copy["metadata"] = {}
                    sol_data_copy["metadata"]["source_file_index"] = file_result_idx
                    sol_data_copy["metadata"]["solution_index_in_file"] = sol_idx
                    combined_solutions.append(sol_data_copy)
        
        merged_result["all_solutions"] = combined_solutions
        merged_results.append(merged_result)
        total_iteration_solutions += len(combined_solutions)
    
    # Calculate statistics
    if merged_results:
        avg_solutions = total_iteration_solutions / len(merged_results)
        print(f"Merged {len(merged_results)} problems")
        print(f"Average solutions per problem (iteration={iteration}): {avg_solutions:.1f}")
        print(f"Total iteration={iteration} solutions: {total_iteration_solutions}")
    else:
        print(f"Merged {len(merged_results)} problems")
    
    return {
        "config": config_dict,
        "results": merged_results,
        "benchmark": benchmark,
    }


def _save_incremental_result(
    new_result: Dict[str, Any],
    all_current_results: List[Dict[str, Any]],
    output_file_path: Path,
    source_files: List[Path],
    benchmark: str,
    config_dict: Dict[str, Any],
    experiment_config: Dict[str, Any],
    start_time: datetime,
    save_lock: threading.Lock,
    strategy_name: str,
) -> None:
    """Save results incrementally to disk for a single strategy.
    
    Args:
        new_result: New problem result to save
        all_current_results: All current results including the new one
        output_file_path: Path to output file
        source_files: List of source result files
        benchmark: Benchmark name
        config_dict: Configuration dictionary
        experiment_config: Experiment configuration
        start_time: Experiment start time
        save_lock: Thread lock for safe file writing
        strategy_name: Name of the strategy being run
    """
    with save_lock:
        # Load existing results from file if it exists
        existing_file_results = []
        if output_file_path.exists():
            try:
                with open(output_file_path, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                    existing_file_results = file_data.get("results", [])
            except Exception:
                existing_file_results = []
        
        # Merge: remove duplicates by problem_id, then add all current results
        # Keep results from file that are not in current results
        merged_results = [
            r for r in existing_file_results
            if r.get("problem_id") not in {res.get("problem_id") for res in all_current_results}
        ]
        # Add all current results
        merged_results.extend(all_current_results)
        
        # Compute current aggregate metrics for this strategy
        # Exclude errors and skipped results from metrics
        strategy_results = [
            r for r in merged_results
            if "error" not in r and not r.get("skipped", False)
        ]
        
        if strategy_results:
            total_count = len(strategy_results)
            
            # Calculate pass@1 using all experiments (problems * num_exps)
            # For num_exps > 1: is_correct is pass_rate (float 0.0-1.0), and we have exp_results
            # For num_exps == 1: is_correct is boolean (True/False)
            # We need to extract all individual experiments and calculate their average accuracy
            all_experiments = []
            for result in strategy_results:
                metadata = result.get("metadata", {})
                exp_results = metadata.get("exp_results", [])
                
                if exp_results:
                    # Multiple experiments case: add all experiments
                    for exp_result in exp_results:
                        all_experiments.append({
                            "is_correct": exp_result.get("is_correct", False),
                            "score": exp_result.get("score", 0.0),
                        })
                else:
                    # Single experiment case: add the problem's result
                    is_correct = result.get("is_correct", False)
                    score = result.get("score", 0.0)
                    all_experiments.append({
                        "is_correct": is_correct,
                        "score": score,
                    })
            
            if all_experiments:
                # Calculate pass@1 as average accuracy across all experiments
                total_experiments = len(all_experiments)
                correct_experiments = sum(1 for exp in all_experiments if exp["is_correct"])
                pass_at_1 = correct_experiments / total_experiments if total_experiments > 0 else 0.0
                avg_score = sum(exp["score"] for exp in all_experiments) / total_experiments if total_experiments > 0 else 0.0
                
                # Also calculate per-problem average accuracy for comparison
                def get_accuracy(result):
                    is_correct = result.get("is_correct", False)
                    if isinstance(is_correct, (int, float)):
                        return float(is_correct)
                    elif isinstance(is_correct, bool):
                        return 1.0 if is_correct else 0.0
                    else:
                        return 0.0
                
                per_problem_accuracies = [get_accuracy(r) for r in strategy_results]
                per_problem_avg_pass_at_1 = sum(per_problem_accuracies) / total_count if total_count > 0 else 0.0
                
                # Also calculate majority vote pass@1 for comparison (>= 0.5)
                majority_vote_correct = sum(1 for acc in per_problem_accuracies if acc >= 0.5)
                majority_vote_pass_at_1 = majority_vote_correct / total_count if total_count > 0 else 0.0
                
                # Count skipped problems
                skipped_count = sum(1 for r in merged_results if r.get("skipped", False))
                
                aggregate_metrics = {
                    "pass@1": pass_at_1,  # Average accuracy across all experiments (problems * num_exps)
                    "per_problem_avg_pass@1": per_problem_avg_pass_at_1,  # Average accuracy per problem, then averaged
                    "majority_vote_pass@1": majority_vote_pass_at_1,  # Majority vote (>=0.5) for comparison
                    "total_problems": total_count,
                    "total_experiments": total_experiments,  # Total number of experiments
                    "correct_experiments": correct_experiments,  # Number of correct experiments
                    "correct_problems": majority_vote_correct,  # Number of problems with >=50% accuracy
                    "avg_score": avg_score,
                    "skipped_problems": skipped_count,
                }
            else:
                # Fallback if no experiments found
                avg_score = sum(r.get("score", 0.0) for r in strategy_results) / total_count if total_count > 0 else 0.0
                skipped_count = sum(1 for r in merged_results if r.get("skipped", False))
                aggregate_metrics = {
                    "pass@1": 0.0,
                    "per_problem_avg_pass@1": 0.0,
                    "majority_vote_pass@1": 0.0,
                    "total_problems": total_count,
                    "total_experiments": 0,
                    "correct_experiments": 0,
                    "correct_problems": 0,
                    "avg_score": avg_score,
                    "skipped_problems": skipped_count,
                }
        else:
            aggregate_metrics = {}
        
        # Prepare output data
        end_time = datetime.now()
        output_data = {
            "source_files": [str(f) for f in source_files],
            "benchmark": benchmark,
            "config": config_dict,
            "experiment_config": {**experiment_config, "strategy": strategy_name},
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "aggregate_metrics": aggregate_metrics,
            "results": merged_results,
        }
        
        # Save to file (atomic write: write to temp file, then rename)
        temp_path = output_file_path.with_suffix('.json.tmp')
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Atomic rename
        temp_path.replace(output_file_path)
        
        problem_id = new_result.get("problem_id", "unknown")
        print(f"  ✓ Saved intermediate result for problem {problem_id} ({len(merged_results)} total)", flush=True)


def extract_first_n_solutions(result: Dict[str, Any], n: int = 4) -> List[Solution]:
    """Extract first N solutions from a result entry.
    
    Args:
        result: Result dictionary with all_solutions
        n: Number of solutions to extract (default: 4)
        
    Returns:
        List of Solution objects with evaluation results preserved in metadata
    """
    all_solutions_data = result.get("all_solutions", [])
    solutions = []
    
    for i, sol_data in enumerate(all_solutions_data[:n]):
        solution_content = sol_data.get("solution_content", "")
        # Handle null/None solution_content
        if solution_content is None:
            solution_content = ""
        
        # Extract evaluation results from sol_data
        is_correct = sol_data.get("is_correct")
        score = sol_data.get("score")
        feedback = sol_data.get("feedback")
        
        # Preserve all original data in metadata for gt_scoring to use
        original_metadata = sol_data.get("metadata", {}).copy()
        # Also include is_correct and score at top level of original_metadata for easy access
        if is_correct is not None:
            original_metadata["is_correct"] = is_correct
        if score is not None:
            original_metadata["score"] = score
        
        solution = Solution(
            content=solution_content,
            score=score,  # Store score in solution for easy access
            feedback=feedback,
            metadata={
                "original_index": i,
                "original_metadata": original_metadata,
                # Also store is_correct at top level for easy access
                "is_correct": is_correct,
            },
        )
        solutions.append(solution)
    
    return solutions


def extract_random_n_solutions(result: Dict[str, Any], n: int = 4, random_seed: Optional[int] = None) -> List[Solution]:
    """Extract N random solutions from a result entry.
    
    Args:
        result: Result dictionary with all_solutions
        n: Number of solutions to extract (default: 4)
        random_seed: Optional random seed for reproducibility
        
    Returns:
        List of Solution objects with evaluation results preserved in metadata
    """
    all_solutions_data = result.get("all_solutions", [])
    
    if len(all_solutions_data) <= n:
        # If we have fewer or equal solutions, return all
        return extract_first_n_solutions(result, len(all_solutions_data))
    
    # Randomly sample n solutions
    if random_seed is not None:
        rng = random.Random(random_seed)
    else:
        rng = random
    
    selected_indices = rng.sample(range(len(all_solutions_data)), n)
    solutions = []
    
    for idx in selected_indices:
        sol_data = all_solutions_data[idx]
        solution_content = sol_data.get("solution_content", "")
        # Handle null/None solution_content
        if solution_content is None:
            solution_content = ""
        
        # Extract evaluation results from sol_data
        is_correct = sol_data.get("is_correct")
        score = sol_data.get("score")
        feedback = sol_data.get("feedback")
        
        # Preserve all original data in metadata for gt_scoring to use
        original_metadata = sol_data.get("metadata", {}).copy()
        # Also include is_correct and score at top level of original_metadata for easy access
        if is_correct is not None:
            original_metadata["is_correct"] = is_correct
        if score is not None:
            original_metadata["score"] = score
        
        solution = Solution(
            content=solution_content,
            score=score,  # Store score in solution for easy access
            feedback=feedback,
            metadata={
                "original_index": idx,  # Keep original index in all_solutions
                "original_metadata": original_metadata,
                # Also store is_correct at top level for easy access
                "is_correct": is_correct,
            },
        )
        solutions.append(solution)
    
    return solutions


# Key used to detect whether a result has evaluation. If missing (at top-level or in exp_results),
# we re-eval only. Each exp_results[i] must have eval_key.
EVAL_KEY = "eval_result"


def _eval_result_to_dict(er: Any) -> Optional[Dict[str, Any]]:
    """Convert EvaluationResult or dict to JSON-serializable dict."""
    if er is None:
        return None
    if isinstance(er, dict):
        return er
    return {
        "is_correct": getattr(er, "is_correct", False),
        "score": getattr(er, "score", 0.0),
        "feedback": getattr(er, "feedback", None),
        "details": getattr(er, "details", None),
    }


def _load_existing_results_for_strategy(
    resume_file: Path, strategy_name: str, num_exps: int = 1, eval_key: str = EVAL_KEY
) -> tuple[set[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load existing results for a strategy and classify by resume action.

    - Results with "error": re-run full aggregation (not included in any return list).
    - Complete results with eval_key everywhere: skip (completed_problem_ids + existing_results).
      For num_exps > 1, every exp_results[i] must have eval_key. For num_exps == 1, top-level must have it.
    - Complete but any exp_result lacks eval_key: re-eval only those (reeval_results).

    Args:
        resume_file: Path to existing results JSON file
        strategy_name: Name of the strategy to filter by
        num_exps: Number of experiments expected per problem. If a problem has fewer
                  exp_results than num_exps, it will be considered incomplete (re-run).
        eval_key: Key that indicates evaluation exists (default "eval_result").

    Returns:
        Tuple of (completed_problem_ids, existing_results, reeval_results).
    """
    completed_problem_ids = set()
    existing_results = []
    reeval_results: List[Dict[str, Any]] = []
    incomplete_problems = []
    error_problems = []

    if not resume_file.exists():
        print(f"  Warning: Resume file not found: {resume_file}")
        return completed_problem_ids, existing_results, reeval_results

    try:
        with open(resume_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        exp_config = data.get("experiment_config", {})
        file_strategy = exp_config.get("strategy")

        if file_strategy != strategy_name:
            print(f"  Warning: Resume file strategy '{file_strategy}' doesn't match '{strategy_name}'")
            return completed_problem_ids, existing_results, reeval_results

        results = data.get("results", [])
        for result in results:
            if "problem_id" not in result:
                continue
            problem_id = result["problem_id"]

            # 1. Error results: re-run full aggregation (do not add to any list)
            if "error" in result:
                error_problems.append(problem_id)
                continue

            metadata = result.get("metadata", {})
            exp_results = metadata.get("exp_results", [])

            # 2. Incomplete (exp count mismatch): re-run
            if num_exps > 1:
                if len(exp_results) < num_exps:
                    incomplete_problems.append(problem_id)
                    print(f"  Problem {problem_id}: Only {len(exp_results)}/{num_exps} experiments, will re-run")
                    continue
            else:
                if exp_results:
                    incomplete_problems.append(problem_id)
                    print(f"  Problem {problem_id}: Has {len(exp_results)} exp_results but num_exps=1 (config mismatch), will re-run")
                    continue

            # 3. Check eval_key: each exp_results[i] must have it (multi-exp); else top-level (single-exp)
            skipped = result.get("skipped", False)
            if skipped:
                incomplete_problems.append(problem_id)
                print(f"  Problem {problem_id}: Skipped, will re-run")
                continue

            if num_exps > 1:
                # Multi-exp: every exp must have eval_result, and none should have "Remote evaluation error" in feedback
                missing = [i for i, ex in enumerate(exp_results) if eval_key not in ex]
                has_error_feedback = []
                
                # Check for "Remote evaluation error" in eval_result.feedback
                for i, ex in enumerate(exp_results):
                    if eval_key in ex:
                        eval_res = ex.get(eval_key, {})
                        if isinstance(eval_res, dict):
                            feedback = eval_res.get("feedback", "") or ""
                            # if "Remote evaluation error" in feedback:
                            #     has_error_feedback.append(i)
                
                # If any exp has error feedback or missing eval_result, need to re-eval
                needs_reeval = missing or has_error_feedback
                
                if not needs_reeval:
                    completed_problem_ids.add(problem_id)
                    existing_results.append(result)
                    continue
                
                # At least one exp needing re-eval must have a solution
                can_reeval = any(
                    (ex.get("solution_content") or ex.get("content") or "").strip()
                    for i, ex in enumerate(exp_results) if i in missing or i in has_error_feedback
                )
                if not can_reeval:
                    incomplete_problems.append(problem_id)
                    print(f"  Problem {problem_id}: exps {missing + has_error_feedback} need re-eval but no solution, will re-run")
                    continue
                
                if missing:
                    print(f"  Problem {problem_id}: exps {missing} missing {eval_key}, will re-eval only")
                if has_error_feedback:
                    print(f"  Problem {problem_id}: exps {has_error_feedback} have 'Remote evaluation error' in feedback, will re-eval only")
                reeval_results.append(result)
                continue

            # Single-exp: top-level must have eval_result, and no "Remote evaluation error" in feedback
            has_eval = eval_key in result
            has_error_feedback = False
            if has_eval:
                eval_res = result.get(eval_key, {})
                if isinstance(eval_res, dict):
                    feedback = eval_res.get("feedback", "") or ""
                    # if "Remote evaluation error" in feedback:
                    #     has_error_feedback = True
            
            solution = result.get("solution_content") or result.get("content") or ""
            if not has_eval or has_error_feedback:
                if not solution.strip():
                    incomplete_problems.append(problem_id)
                    reason = "no solution" if not has_eval else "has error feedback but no solution"
                    print(f"  Problem {problem_id}: {reason}, will re-run")
                    continue
                if not has_eval:
                    print(f"  Problem {problem_id}: Has result but no {eval_key}, will re-eval only")
                else:
                    print(f"  Problem {problem_id}: Has 'Remote evaluation error' in feedback, will re-eval only")
                reeval_results.append(result)
                continue

            completed_problem_ids.add(problem_id)
            existing_results.append(result)

        print(f"  Resuming from: {resume_file}")
        print(f"  Found {len(completed_problem_ids)} completed problems")
        if error_problems:
            print(f"  Found {len(error_problems)} error problems (will re-run): {error_problems[:5]}{'...' if len(error_problems) > 5 else ''}")
        if incomplete_problems:
            print(f"  Found {len(incomplete_problems)} incomplete problems (will re-run): {incomplete_problems[:5]}{'...' if len(incomplete_problems) > 5 else ''}")
        if reeval_results:
            print(f"  Found {len(reeval_results)} results without eval (will re-eval only)")

    except Exception as e:
        print(f"  Warning: Failed to load resume file: {e}")
        completed_problem_ids = set()
        existing_results = []
        reeval_results = []

    return completed_problem_ids, existing_results, reeval_results


def _find_latest_result_file(
    output_dir: Path, strategy_name: str, benchmark: str
) -> Optional[Path]:
    """Find the latest result file for a strategy.
    
    Args:
        output_dir: Output directory to search
        strategy_name: Name of the strategy
        benchmark: Benchmark name
        
    Returns:
        Path to latest result file, or None if not found
    """
    if not output_dir.exists():
        return None
    
    pattern = f"aggregation_experiment_{strategy_name}_{benchmark}_*.json"
    result_files = sorted(
        output_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    # Filter out temporary files
    result_files = [f for f in result_files if not f.name.endswith('.tmp')]
    
    return result_files[0] if result_files else None


def _load_solution_file_results(
    solution_file: Path, strategy_name: str, num_exps: int = 1, eval_key: str = EVAL_KEY
) -> Dict[str, Dict[str, Any]]:
    """Load results from solution file and index by problem_id.
    
    Args:
        solution_file: Path to solution result file
        strategy_name: Strategy name to filter by (if solution file has strategy info)
        num_exps: Number of experiments expected per problem
        eval_key: Key that indicates evaluation exists (default "eval_result")
        
    Returns:
        Dictionary mapping problem_id to result dictionary
    """
    solution_results_by_id: Dict[str, Dict[str, Any]] = {}
    
    if not solution_file.exists():
        print(f"  Warning: Solution file not found: {solution_file}")
        return solution_results_by_id
    
    try:
        with open(solution_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check if this file matches the strategy (if strategy info exists)
        exp_config = data.get("experiment_config", {})
        file_strategy = exp_config.get("strategy")
        
        if file_strategy and file_strategy != strategy_name:
            print(f"  Warning: Solution file strategy '{file_strategy}' doesn't match '{strategy_name}', but will still use it")
        
        results = data.get("results", [])
        for result in results:
            if "problem_id" not in result:
                continue
            problem_id = result["problem_id"]
            solution_results_by_id[problem_id] = result
        
        print(f"  Loaded {len(solution_results_by_id)} results from solution file: {solution_file}")
        
    except Exception as e:
        print(f"  Warning: Failed to load solution file: {e}")
        solution_results_by_id = {}
    
    return solution_results_by_id


async def run_aggregation_experiment(
    result_files: List[Path],
    output_dir: str = "aggregation_experiments",
    solutions_per_exp: int = 4,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    task_domain: str = "coding",
    max_concurrent: Optional[int] = None,
    resume: bool = False,
    resume_from: Optional[str] = None,
    service_url: Optional[str] = None,
    language: str = "cpp",
        num_exps: int = 1,
        reasoning_effort: str = "auto",
        iteration: int = 0,
        solution_file: Optional[Path] = None,
        solution_timestamp: Optional[str] = None,
        model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run aggregation experiment on LCB Pro result file(s).
    
    Args:
        result_files: List of paths to result JSON files (can be single file or multiple)
        output_dir: Output directory for results
        solutions_per_exp: Number of solutions to randomly sample for each experiment (default: 4)
        api_key: API key for LLM service
        api_base: API base URL for LLM service
        task_domain: Task domain for template selection (default: "coding" for lcb_pro)
        max_concurrent: Maximum number of problems to process concurrently
        resume: Auto-resume from latest results files for each strategy
        resume_from: Resume from specific results file (applies to all strategies)
        service_url: Remote code verify service URL(s), comma-separated
        language: Programming language for code evaluation (default: "cpp")
        num_exps: Number of experiments to run per problem
        reasoning_effort: Reasoning effort level for LLM aggregation strategies (default: "auto")
        iteration: Iteration number to extract solutions from (default: 0)
        solution_file: Optional solution result file. When a problem is not found in the latest
                      result file, this script will try to find it in the solution-file. If found
                      and exp_results contain 'Remote evaluation error', they will be re-evaluated.
                      If not found, the problem will be treated as new and aggregated.
                      If not provided, solution_timestamp will be used instead.
        solution_timestamp: Optional timestamp string (e.g., "20260126_120000"). When provided,
                           each strategy will automatically use solution-file at:
                           output_dir/aggregation_experiment_{strategy}_{benchmark}_{solution_timestamp}.json
                           Priority: solution_file > solution_timestamp
        
    Returns:
        Dictionary with experiment results
    """
    # Merge result files, extracting all solutions with iteration == iteration
    data = merge_result_files(result_files, iteration=iteration)
    
    config_dict = data["config"]
    results = data["results"]
    benchmark = data["benchmark"]
    
    # Extract LLM config from result file
    llm_config_dict = config_dict.get("llm", {})
    model_name = model_name
    api_key = api_key or llm_config_dict.get("api_key") or os.getenv("OPENAI_API_KEY", "None")
    api_base = os.getenv("OPENAI_API_BASE")
    
    if api_base is None:
        api_base = os.getenv("SGLANG_API_BASES")
    
    print(f"\nUsing model: {model_name}")
    if api_base:
        print(f"API Base: {api_base}")
    
    # Create LLM service
    llm_service = create_llm_service(
        provider="litellm",
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
    )
    
    # Create prompt manager
    prompt_manager = PromptManager()
    
    # Create remote LCB Pro evaluator (needed for evaluating aggregated solutions, not for gt_scoring)
    # Note: gt_scoring now uses stored evaluation results from result files, not re-evaluation
    if not service_url:
        service_url = os.getenv("CODE_VERIFY_URL")
    
    if not service_url:
        raise ValueError("service_url is required. Provide via --service-url or CODE_VERIFY_URL env var")
    
    evaluator = RemoteLCBProEvaluator(
        service_url=service_url,
        max_retries=3,
    )
    
    # Create aggregation strategies
    strategies = {}
    
    # 1. Generate from N (coding version)
    generate_template = prompt_manager.get_template("aggregation_generate_one_from_n_coding")
    if generate_template is None:
        # Fallback to general template
        generate_template = prompt_manager.get_template("aggregation_generate_one_from_n")
    # strategies["generate_from_n"] = GenerateFromNAggregation(
    #     llm_service=llm_service,
    #     generation_prompt_template=generate_template,
    #     temperature=0.7,
    #     reasoning_effort=reasoning_effort,
    # )
    
    # 2. Select best (coding version)
    select_template = prompt_manager.get_template("aggregation_select_one_from_n_coding")
    if select_template is None:
        # Fallback to general template
        select_template = prompt_manager.get_template("aggregation_select_one_from_n")
    # strategies["select_best"] = SelectBestAggregation(
    #     llm_service=llm_service,
    #     selection_prompt_template=select_template,
    #     temperature=0.0,
    #     reasoning_effort=reasoning_effort,
    # )
    
    # 3. LLM Scoring (coding version)
    scoring_template = prompt_manager.get_template(f"code_llm_scoring")
    if scoring_template is None:
        # Fallback to coding_scoring
        scoring_template = prompt_manager.get_template("code_llm_scoring")
    strategies["llm_scoring"] = LLMScoringAggregation(
        llm_service=llm_service,
        scoring_prompt_template=scoring_template,
        temperature=0.0,
        reasoning_effort=reasoning_effort,
    )
    
    # 4. GT Scoring (uses stored evaluation results from result files, implements pass@k)
    # Note: gt_scoring no longer re-evaluates, it directly uses is_correct from result files
    # strategies["gt_scoring"] = GTScoringAggregation(evaluator=evaluator)
    
    # 5. Voting (for code, use code hash as answer identifier) do not test now
    # Note: For code problems, voting compares code implementations rather than outputs
    # since outputs require execution. We use a code-based extractor.
    def _code_answer_extractor(solution: str) -> str:
        """Extract code from solution for voting comparison.
        
        For code problems, we compare the actual code implementation.
        This extracts the code block or uses the full solution as identifier.
        """
        # Try to extract code block
        import re
        code_pattern = r"```(?:cpp|c\+\+|c)?\s*\n?(.*?)\n?```"
        code_blocks = re.findall(code_pattern, solution, re.DOTALL)
        if code_blocks:
            # Use the last code block (usually the final solution)
            return code_blocks[-1].strip()
        # If no code block, use the full solution as identifier
        return solution.strip()
    
    # strategies["voting"] = VotingAggregation(answer_extractor=_code_answer_extractor)
    
    # # 6. LLM Voting (uses LLM to determine equivalent answers) do not test now
    # voting_template = prompt_manager.get_template("aggregation_llm_voting")
    # strategies["llm_voting"] = LLMVotingAggregation(
    #     llm_service=llm_service,
    #     voting_prompt_template=voting_template,
    #     temperature=0.0,
    #     reasoning_effort="auto",
    # )
    
    # only test pairwise comparison
    # strategies = {}
    # 7. Pairwise comparison (coding version)
    pairwise_template = prompt_manager.get_template("aggregation_pairwise_comparison_coding")
    if pairwise_template is None:
        # Fallback to general template
        pairwise_template = prompt_manager.get_template("aggregation_pairwise_comparison")
    # if reasoning_effort == "low":
    strategies = {}
    strategies["pairwise_comparison"] = PairwiseComparisonAggregation(
        llm_service=llm_service,
        comparison_prompt_template=pairwise_template,
        temperature=0.0,
        reasoning_effort=reasoning_effort,
    )
    
    # only gt_scoring:
    # strategies = {
    #     # "gt_scoring": strategies["gt_scoring"]
    # }
    # Prepare experiment config for saving
    experiment_config = {
        "num_result_files": len(result_files),
        "solutions_per_exp": solutions_per_exp,
        "task_domain": task_domain,
        "model_name": model_name,
        "language": language,
        "service_url": service_url,
        "num_exps": num_exps,  # Number of experiments to run
        "reasoning_effort": reasoning_effort,
        "iteration": iteration,  # Iteration number to extract solutions from
    }
    
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run each strategy separately on the entire benchmark
    print(f"\nRunning aggregation experiments on {len(results)} problems...")
    print(f"Extracting all solutions with metadata.iteration == {iteration} from {len(result_files)} file(s)")
    print(f"Each experiment will randomly sample {solutions_per_exp} solutions from all available iteration={iteration} solutions")
    print(f"Strategies: {list(strategies.keys())}")
    print(f"Each strategy will process all problems and save to separate files")
    print(f"Intermediate results will be saved incrementally")
    if resume:
        print(f"Resume mode: Auto-detect latest results for each strategy")
    elif resume_from:
        print(f"Resume mode: Using specified file: {resume_from}")
    if solution_file:
        print(f"Solution file: {solution_file} (will be used as fallback for all strategies)")
    elif solution_timestamp:
        print(f"Solution timestamp: {solution_timestamp} (each strategy will use its own solution-file)")
    print(f"Service URL: {service_url}")
    print(f"Language: {language}")
    
    all_strategy_results = {}
    
    # Prepare all tasks across all strategies for unified parallel execution
    all_tasks = []
    strategy_info = {}  # Store strategy-specific info for each task
    
    # Create semaphore for limiting concurrent executions (shared across all strategies)
    if max_concurrent is None:
        max_concurrent = int(os.getenv("MAX_CONCURRENT_PROBLEMS", "40"))
    global_semaphore = asyncio.Semaphore(max_concurrent)
    global_results_locks = {}  # One lock per strategy
    
    # Prepare strategy-specific data structures
    for strategy_idx, (strategy_name, strategy) in enumerate(strategies.items(), 1):
        print(f"\n{'=' * 80}")
        print(f"STRATEGY {strategy_idx}/{len(strategies)}: {strategy_name}")
        print(f"{'=' * 80}")
        
        # Determine resume file for this strategy
        strategy_resume_file = None
        if resume_from:
            # Use specified resume file
            strategy_resume_file = Path(resume_from)
        elif resume:
            # Auto-detect latest result file for this strategy
            strategy_resume_file = _find_latest_result_file(output_path, strategy_name, benchmark)
            if strategy_resume_file:
                print(f"  Auto-detected resume file: {strategy_resume_file}")
        
        # Load existing results if resuming
        completed_problem_ids = set()
        existing_results = []
        reeval_results: List[Dict[str, Any]] = []
        if strategy_resume_file:
            completed_problem_ids, existing_results, reeval_results = _load_existing_results_for_strategy(
                strategy_resume_file, strategy_name, num_exps=num_exps
            )
            if completed_problem_ids:
                print(f"  Will skip {len(completed_problem_ids)} completed problems")
            n_remaining = len(results) - len(completed_problem_ids)
            if n_remaining > 0:
                print(f"  Running {n_remaining} remaining (aggregate or re-eval only)")

        # Load solution file results if provided
        # Priority: solution_file > solution_timestamp (auto-generated per strategy)
        solution_results_by_id: Dict[str, Dict[str, Any]] = {}
        strategy_solution_file = None
        if solution_file:
            # Use explicitly provided solution_file
            strategy_solution_file = solution_file
            print(f"  Using solution file: {strategy_solution_file}")
        elif solution_timestamp:
            # Auto-generate solution_file path for this strategy
            strategy_solution_file = output_path / f"aggregation_experiment_{strategy_name}_{benchmark}_{solution_timestamp}.json"
            if strategy_solution_file.exists():
                print(f"  Using solution file: {strategy_solution_file}")
            else:
                print(f"  Solution file not found (will skip): {strategy_solution_file}")
                strategy_solution_file = None
        
        if strategy_solution_file:
            solution_results_by_id = _load_solution_file_results(
                strategy_solution_file, strategy_name, num_exps=num_exps
            )

        # Remaining = all problems not completed (includes error re-run, incomplete re-run, and re-eval-only)
        remaining_results = [
            r for r in results
            if r.get("problem_id", "") not in completed_problem_ids
        ]
        reeval_ids = {r["problem_id"] for r in reeval_results}
        
        # For problems not in completed_problem_ids, try to find them in solution_file
        # If found, check if they need re-evaluation due to remote evaluation error
        solution_file_reeval_results: List[Dict[str, Any]] = []
        aggregate_remaining = []
        for r in remaining_results:
            problem_id = r.get("problem_id", "")
            if problem_id in reeval_ids:
                continue  # Already in reeval_results
            
            # Try to find in solution_file
            if problem_id in solution_results_by_id:
                solution_result = solution_results_by_id[problem_id]
                # Check if this result needs re-evaluation due to remote evaluation error
                metadata = solution_result.get("metadata", {})
                exp_results = metadata.get("exp_results", [])
                
                needs_reeval_from_solution = False
                if num_exps > 1 and exp_results:
                    # Multi-exp: check if any exp has remote evaluation error
                    for ex in exp_results:
                        eval_res = ex.get(EVAL_KEY, {})
                        if isinstance(eval_res, dict):
                            feedback = eval_res.get("feedback", "") or ""
                            # do not do reeval now
                            # if "Remote evaluation error" in feedback:
                            #     needs_reeval_from_solution = True
                            #     break
                elif num_exps == 1:
                    # Single-exp: check top-level eval_result
                    eval_res = solution_result.get(EVAL_KEY, {})
                    if isinstance(eval_res, dict):
                        feedback = eval_res.get("feedback", "") or ""
                        # if "Remote evaluation error" in feedback:
                        #     needs_reeval_from_solution = True
                
                if needs_reeval_from_solution:
                    # Use solution result but mark for re-evaluation
                    solution_file_reeval_results.append(solution_result)
                    print(f"  Problem {problem_id}: Found in solution file with remote evaluation error, will re-eval")
                else:
                    # Use solution result as-is (completed)
                    completed_problem_ids.add(problem_id)
                    existing_results.append(solution_result)
                    print(f"  Problem {problem_id}: Found in solution file, using as completed")
            else:
                # Not found in solution_file, treat as new problem
                aggregate_remaining.append(r)
        
        # Add solution_file_reeval_results to reeval_results
        reeval_results.extend(solution_file_reeval_results)
        reeval_ids = {r["problem_id"] for r in reeval_results}

        if not remaining_results:
            print(f"\n  All problems already completed for strategy {strategy_name}!")
            print(f"  Using existing results from: {strategy_resume_file}")
            strategy_results = existing_results
        else:
            print(f"  Processing {len(aggregate_remaining)} aggregate + {len(reeval_results)} re-eval-only")
            strategy_results = existing_results.copy()

        # Create output file for this strategy
        if strategy_resume_file and not remaining_results:
            strategy_output_file = strategy_resume_file
        else:
            strategy_output_file = output_path / f"aggregation_experiment_{strategy_name}_{benchmark}_{timestamp}.json"

        save_lock = threading.Lock()
        strategy_start_time = datetime.now()
        results_lock = asyncio.Lock()
        global_results_locks[strategy_name] = results_lock

        strategy_info[strategy_name] = {
            "strategy": strategy,
            "strategy_name": strategy_name,
            "strategy_output_file": strategy_output_file,
            "strategy_results": strategy_results,
            "save_lock": save_lock,
            "strategy_start_time": strategy_start_time,
            "remaining_results": remaining_results,
            "existing_results": existing_results,
            "reeval_results": reeval_results,
            "results_lock": results_lock,
        }

        # Create tasks: aggregate (full re-run) + re-eval only
        total_tasks = len(aggregate_remaining) + len(reeval_results)
        if aggregate_remaining:
            for idx, result_data in enumerate(aggregate_remaining, 1):
                all_tasks.append({
                    "task_type": "aggregate",
                    "strategy_name": strategy_name,
                    "result_data": result_data,
                    "idx": idx,
                    "total": total_tasks,
                })
        if reeval_results:
            for idx, stored in enumerate(reeval_results, len(aggregate_remaining) + 1):
                all_tasks.append({
                    "task_type": "reeval",
                    "strategy_name": strategy_name,
                    "stored_result": stored,
                    "idx": idx,
                    "total": total_tasks,
                })
    
    # Define unified processing function
    async def process_problem_unified(task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single problem with its assigned strategy."""
        strategy_name = task_info["strategy_name"]
        task_type = task_info.get("task_type", "aggregate")
        idx = task_info["idx"]
        total = task_info["total"]
        info = strategy_info[strategy_name]
        strategy = info["strategy"]

        # Re-eval only: keep solutions, submit missing exps to remote evaluator, update result
        if task_type == "reeval":
            stored = task_info["stored_result"]
            problem_id = stored.get("problem_id", f"problem_{idx}")
            problem = stored.get("problem", "")
            ground_truth = stored.get("ground_truth", "")
            metadata = stored.get("metadata", {})
            exp_results_list = metadata.get("exp_results", [])
            async with global_semaphore:
                try:
                    if not exp_results_list:
                        # Single-exp: top-level solution
                        solution = stored.get("solution_content") or stored.get("content") or ""
                        if not solution.strip():
                            print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Re-eval skipped (no solution)", flush=True)
                            return stored
                        ev = await evaluator.evaluate(
                            problem=problem,
                            solution=solution,
                            ground_truth=ground_truth,
                            problem_id=problem_id,
                            language=language,
                        )
                        updated = {**stored}
                        updated["is_correct"] = ev.is_correct
                        updated["score"] = ev.score
                        updated["feedback"] = getattr(ev, "feedback", None)
                        if hasattr(ev, "details") and ev.details is not None:
                            updated["details"] = ev.details
                        updated["eval_result"] = _eval_result_to_dict(ev)
                    else:
                        # Multi-exp: re-eval exps missing eval_result or with "Remote evaluation error" in feedback
                        updated_metadata = {**metadata}
                        updated_exps = [dict(ex) for ex in exp_results_list]
                        for i, ex in enumerate(updated_exps):
                            # Check if needs re-evaluation: missing eval_result or has error feedback
                            needs_reeval = False
                            if EVAL_KEY not in ex:
                                needs_reeval = True
                            else:
                                # Check if eval_result has "Remote evaluation error" in feedback
                                eval_res = ex.get(EVAL_KEY, {})
                                if isinstance(eval_res, dict):
                                    feedback = eval_res.get("feedback", "") or ""
                                    # if "Remote evaluation error" in feedback:
                                    #     needs_reeval = True
                            
                            if not needs_reeval:
                                continue
                            
                            sol = ex.get("solution_content") or ex.get("content") or ""
                            if not sol.strip():
                                continue
                            ev = await evaluator.evaluate(
                                problem=problem,
                                solution=sol,
                                ground_truth=ground_truth,
                                problem_id=problem_id,
                                language=language,
                            )
                            ex["is_correct"] = ev.is_correct
                            ex["score"] = ev.score
                            ex["eval_result"] = _eval_result_to_dict(ev)
                            if hasattr(ev, "feedback") and ev.feedback is not None:
                                ex["feedback"] = ev.feedback
                            if hasattr(ev, "details") and ev.details is not None:
                                ex["details"] = ev.details
                        updated_metadata["exp_results"] = updated_exps
                        pass_rate = sum(1 for e in updated_exps if e.get("is_correct", False)) / len(updated_exps)
                        avg_score = sum(e.get("score", 0.0) for e in updated_exps) / len(updated_exps)
                        updated_metadata["pass_rate"] = pass_rate
                        updated_metadata["avg_score"] = avg_score
                        updated = {**stored}
                        updated["metadata"] = updated_metadata
                        updated["is_correct"] = pass_rate
                        updated["score"] = avg_score
                        updated["feedback"] = f"num_exps={len(updated_exps)}, pass_rate={pass_rate:.2f}, avg_score={avg_score:.2f}"
                    async with info["results_lock"]:
                        info["strategy_results"].append(updated)
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None,
                            _save_incremental_result,
                            updated,
                            info["strategy_results"],
                            info["strategy_output_file"],
                            result_files,
                            benchmark,
                            config_dict,
                            experiment_config,
                            info["strategy_start_time"],
                            info["save_lock"],
                            strategy_name,
                        )
                    rate = updated.get("is_correct", False)
                    sc = updated.get("score", 0.0)
                    print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Re-eval {'✓' if rate else '✗'} (score: {sc:.2f})", flush=True)
                    return updated
                except Exception as e:
                    print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Re-eval error - {e}", flush=True)
                    err = {**stored, "error": str(e)}
                    async with info["results_lock"]:
                        info["strategy_results"].append(err)
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None,
                            _save_incremental_result,
                            err,
                            info["strategy_results"],
                            info["strategy_output_file"],
                            result_files,
                            benchmark,
                            config_dict,
                            experiment_config,
                            info["strategy_start_time"],
                            info["save_lock"],
                            strategy_name,
                        )
                    return err

        # Aggregate (full run or re-run)
        result_data = task_info["result_data"]
        async with global_semaphore:
            problem_id = result_data.get("problem_id", f"problem_{idx}")
            problem = result_data.get("problem", "")
            ground_truth = result_data.get("ground_truth", "")

            try:
                # Determine if we need to run multiple experiments
                strategies_that_need_randomization = {"generate_from_n", "select_best", "pairwise_comparison"}
                needs_randomization = strategy_name in strategies_that_need_randomization
                
                if num_exps > 1:
                    # Run multiple experiments
                    print("Num of experiments: ", num_exps, flush=True)
                    exp_results = []
                    all_exp_solutions = []  # Store all experiment solutions
                    
                    for exp_idx in range(num_exps):
                        # Randomly sample solutions for each experiment
                        solutions = extract_random_n_solutions(
                            result_data, solutions_per_exp, random_seed=hash((problem_id, exp_idx)) % (2**31)
                        )
                        
                        # Skip if not enough solutions
                        if len(solutions) < solutions_per_exp:
                            if exp_idx == 0:
                                print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Skipping (only {len(solutions)} iteration={iteration} solutions, need {solutions_per_exp})", flush=True)
                            continue
                        
                        # Skip if all solutions are empty
                        if all(not sol.content.strip() for sol in solutions):
                            if exp_idx == 0:
                                print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Skipping (all solutions empty)", flush=True)
                            continue
                        
                        # Aggregate solutions with randomization if needed
                        randomize_order = needs_randomization
                        if strategy_name == "gt_scoring":
                            aggregated_solution = await strategy.aggregate(
                                problem=problem, solutions=solutions, ground_truth=ground_truth
                            )
                        else:
                            aggregated_solution = await strategy.aggregate(
                                problem=problem, solutions=solutions, randomize_order=randomize_order
                            )
                        
                        # Check if aggregation was skipped
                        if aggregated_solution.metadata.get("skipped", False):
                            continue
                        
                        # Evaluate aggregated solution
                        if strategy_name == "gt_scoring":
                            pass_at_k = aggregated_solution.metadata.get("pass_at_k", 0)
                            n_passed = aggregated_solution.metadata.get("n_passed", 0)
                            all_results = aggregated_solution.metadata.get("all_results", [])
                            is_correct = False
                            score = 0.0
                            if all_results:
                                selected_index = aggregated_solution.metadata.get("selected_index", 0)
                                for r in all_results:
                                    if r.get("index") == selected_index:
                                        is_correct = r.get("is_correct", False)
                                        score = r.get("score", 0.0)
                                        break
                            eval_result_dict = _eval_result_to_dict({"is_correct": is_correct, "score": score, "feedback": None, "details": None})
                        else:
                            eval_result = await evaluator.evaluate(
                                problem=problem,
                                solution=aggregated_solution.content,
                                ground_truth=ground_truth,
                                problem_id=problem_id,
                                language=language,
                            )
                            is_correct = eval_result.is_correct
                            score = eval_result.score
                            eval_result_dict = _eval_result_to_dict(eval_result)
                        
                        extracted_code = extract_code(aggregated_solution.content)
                        code = extracted_code if extracted_code is not None else ""
                        exp_token_usage = aggregated_solution.metadata.get("token_usage", {})

                        exp_result = {
                            "exp_idx": exp_idx,
                            "is_correct": is_correct,
                            "score": score,
                            "solution_content": aggregated_solution.content,
                            "content": aggregated_solution.content,
                            "code": code,
                            "eval_result": eval_result_dict,
                            "token_usage": exp_token_usage,
                            "metadata": aggregated_solution.metadata.copy(),
                        }
                        exp_results.append(exp_result)
                        all_exp_solutions.append(aggregated_solution)
                    
                    if not exp_results:
                        return None
                    
                    # Calculate average results
                    avg_score = sum(r["score"] for r in exp_results) / len(exp_results)
                    correct_count = sum(1 for r in exp_results if r["is_correct"])
                    pass_rate = correct_count / len(exp_results)
                    
                    # Use the last experiment's aggregated solution content for main fields
                    # (but all experiments are saved in exp_results)
                    last_exp_solution = all_exp_solutions[-1] if all_exp_solutions else None
                    
                    # Build metadata from last experiment or empty dict
                    result_metadata = last_exp_solution.metadata if last_exp_solution else {}
                    
                    # Extract code from last solution
                    extracted_code = extract_code(last_exp_solution.content if last_exp_solution else "")
                    code = extracted_code if extracted_code is not None else ""
                    
                    # Extract token usage from last experiment's metadata
                    token_usage = result_metadata.get("token_usage", {})
                    
                    problem_result = {
                        "problem_id": problem_id,
                        "problem": problem,
                        "ground_truth": ground_truth,
                        "solution_content": last_exp_solution.content if last_exp_solution else "",
                        "content": last_exp_solution.content if last_exp_solution else "",  # Final answer content (from last exp)
                        "code": code,  # Extracted code (from last exp)
                        "is_correct": pass_rate,  # Use average accuracy instead of majority vote
                        "score": avg_score,
                        "feedback": f"num_exps={num_exps}, pass_rate={pass_rate:.2f}, avg_score={avg_score:.2f}",
                        "token_usage": token_usage,  # Token usage from last experiment
                        "metadata": {
                            **result_metadata,
                            "num_exps": num_exps,
                            "exp_results": exp_results,  # All experiments with full details
                            "pass_rate": pass_rate,
                            "avg_score": avg_score,
                        },
                    }
                    
                    print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: {'✓' if problem_result['is_correct'] else '✗'} (num_exps={num_exps}, pass_rate={pass_rate:.2f}, avg_score={avg_score:.2f})", flush=True)
                else:
                    # Single experiment (original behavior)
                    # Randomly sample solutions for this experiment
                    solutions = extract_random_n_solutions(
                        result_data, solutions_per_exp, random_seed=hash(problem_id) % (2**31)
                    )
                    
                    # Skip if not enough solutions
                    if len(solutions) < solutions_per_exp:
                        print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Skipping (only {len(solutions)} iteration={iteration} solutions, need {solutions_per_exp})", flush=True)
                        return None
                    
                    # Skip if all solutions are empty
                    if all(not sol.content.strip() for sol in solutions):
                        print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Skipping (all solutions empty)", flush=True)
                        return None
                    
                    # Aggregate solutions
                    # Note: gt_scoring now uses stored evaluation results, ground_truth is not needed but kept for compatibility
                    randomize_order = needs_randomization
                    if strategy_name == "gt_scoring":
                        aggregated_solution = await strategy.aggregate(
                            problem=problem, solutions=solutions, ground_truth=ground_truth
                        )
                    else:
                        aggregated_solution = await strategy.aggregate(
                            problem=problem, solutions=solutions, randomize_order=randomize_order
                        )
                
                    # Check if aggregation was skipped due to token limit
                    if aggregated_solution.metadata.get("skipped", False):
                        # Extract code from solution_content
                        extracted_code = extract_code(aggregated_solution.content)
                        code = extracted_code if extracted_code is not None else ""
                        
                        # Extract token usage from metadata
                        token_usage = aggregated_solution.metadata.get("token_usage", {})
                        
                        problem_result = {
                            "problem_id": problem_id,
                            "problem": problem,
                            "ground_truth": ground_truth,
                            "solution_content": aggregated_solution.content,
                            "content": aggregated_solution.content,  # Final answer content
                            "code": code,  # Extracted code
                            "skipped": True,
                            "skip_reason": aggregated_solution.metadata.get("skip_reason", "unknown"),
                            "token_usage": token_usage,  # Token usage information
                            "metadata": aggregated_solution.metadata,
                        }
                        print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: ⏭️  Skipped ({aggregated_solution.metadata.get('skip_reason', 'unknown')})", flush=True)
                    elif strategy_name == "gt_scoring":
                        # For gt_scoring, use the stored pass_at_k result directly
                        # The aggregated_solution is already the first solution that passed (or first if none passed)
                        pass_at_k = aggregated_solution.metadata.get("pass_at_k", 0)
                        n_passed = aggregated_solution.metadata.get("n_passed", 0)
                        
                        # Get the is_correct from the selected solution's stored result
                        all_results = aggregated_solution.metadata.get("all_results", [])
                        is_correct = False
                        score = 0.0
                        if all_results:
                            # Find the result for the selected solution
                            # Use selected_index (index in solutions list) instead of original_index
                            # selected_index is set by GTScoringAggregation and corresponds to the index in all_results
                            selected_index = aggregated_solution.metadata.get("selected_index", 0)
                            for r in all_results:
                                if r.get("index") == selected_index:
                                    is_correct = r.get("is_correct", False)
                                    score = r.get("score", 0.0)
                                    break
                        
                        # Extract code from solution_content
                        extracted_code = extract_code(aggregated_solution.content)
                        code = extracted_code if extracted_code is not None else ""
                        
                        # Extract token usage from metadata
                        token_usage = aggregated_solution.metadata.get("token_usage", {})
                        
                        problem_result = {
                            "problem_id": problem_id,
                            "problem": problem,
                            "ground_truth": ground_truth,
                            "solution_content": aggregated_solution.content,
                            "content": aggregated_solution.content,
                            "code": code,
                            "is_correct": is_correct,
                            "score": score,
                            "feedback": f"pass@k={pass_at_k}, {n_passed}/{len(solutions)} passed",
                            "eval_result": _eval_result_to_dict({"is_correct": is_correct, "score": score, "feedback": None, "details": None}),
                            "token_usage": token_usage,
                            "metadata": aggregated_solution.metadata,
                        }
                        print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: {'✓' if is_correct else '✗'} (pass@k={pass_at_k}, score: {score:.2f})", flush=True)
                    else:
                        eval_result = await evaluator.evaluate(
                            problem=problem,
                            solution=aggregated_solution.content,
                            ground_truth=ground_truth,
                            problem_id=problem_id,
                            language=language,
                        )
                        extracted_code = extract_code(aggregated_solution.content)
                        code = extracted_code if extracted_code is not None else ""
                        token_usage = aggregated_solution.metadata.get("token_usage", {})

                        problem_result = {
                            "problem_id": problem_id,
                            "problem": problem,
                            "ground_truth": ground_truth,
                            "solution_content": aggregated_solution.content,
                            "content": aggregated_solution.content,
                            "code": code,
                            "is_correct": eval_result.is_correct,
                            "score": eval_result.score,
                            "feedback": eval_result.feedback,
                            "eval_result": _eval_result_to_dict(eval_result),
                            "token_usage": token_usage,
                            "metadata": aggregated_solution.metadata,
                        }
                        print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: {'✓' if eval_result.is_correct else '✗'} (score: {eval_result.score:.2f})", flush=True)
                
                # Save incrementally after each problem completes
                async with info["results_lock"]:
                    info["strategy_results"].append(problem_result)
                    # Save to disk (run in executor to avoid blocking async event loop)
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        _save_incremental_result,
                        problem_result,
                        info["strategy_results"],
                        info["strategy_output_file"],
                        result_files,
                        benchmark,
                        config_dict,
                        experiment_config,
                        info["strategy_start_time"],
                        info["save_lock"],
                        strategy_name,
                    )
                
                return problem_result
                
            except Exception as e:
                print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Error - {e}", flush=True)
                error_result = {
                    "problem_id": problem_id,
                    "problem": problem,
                    "ground_truth": ground_truth,
                    "error": str(e),
                }
                async with info["results_lock"]:
                    info["strategy_results"].append(error_result)
                return error_result
    
    # Process all strategies with a unified worker queue.
    # Tasks are ENQUEUED in strategy order, then consumed by workers with global concurrency.
    if all_tasks:
        print(
            f"\nProcessing {len(all_tasks)} tasks across {len(strategies)} strategies "
            f"with max {max_concurrent} concurrent..."
        )
        print("Tasks will be ENQUEUED in strategy order (worker queue).")

        task_queue: asyncio.Queue[Optional[Dict[str, Any]]] = asyncio.Queue()
        for task_info in all_tasks:
            task_queue.put_nowait(task_info)

        # Add sentinels to stop workers
        num_workers = max_concurrent
        for _ in range(num_workers):
            task_queue.put_nowait(None)

        pbar = tqdm(total=len(all_tasks), desc="All strategies")

        async def worker(worker_id: int) -> None:
            while True:
                item = await task_queue.get()
                try:
                    if item is None:
                        return
                    await process_problem_unified(item)
                except Exception as e:
                    print(f"\nError processing task in worker {worker_id}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    if item is not None:
                        pbar.update(1)
                    task_queue.task_done()

        workers = [asyncio.create_task(worker(i)) for i in range(num_workers)]
        await task_queue.join()
        for w in workers:
            # Ensure all workers finish (they should after consuming sentinels)
            await w
        pbar.close()
    
    # Process each strategy to finalize results
    for strategy_idx, (strategy_name, strategy) in enumerate(strategies.items(), 1):
        print(f"\n{'=' * 80}")
        print(f"STRATEGY {strategy_idx}/{len(strategies)}: {strategy_name}")
        print(f"{'=' * 80}")
        
        info = strategy_info[strategy_name]
        strategy_results = info["strategy_results"]
        strategy_output_file = info["strategy_output_file"]
        strategy_start_time = info["strategy_start_time"]
        
        strategy_end_time = datetime.now()
        strategy_duration = (strategy_end_time - strategy_start_time).total_seconds()
        
        # Compute final aggregate metrics for this strategy
        # Exclude errors and skipped results from metrics
        valid_results = [r for r in strategy_results if "error" not in r and not r.get("skipped", False)]
        if valid_results:
            total_count = len(valid_results)
            
            # Calculate pass@1 using all experiments (problems * num_exps)
            # For num_exps > 1: is_correct is pass_rate (float 0.0-1.0), and we have exp_results
            # For num_exps == 1: is_correct is boolean (True/False)
            # We need to extract all individual experiments and calculate their average accuracy
            all_experiments = []
            for result in valid_results:
                metadata = result.get("metadata", {})
                exp_results = metadata.get("exp_results", [])
                
                if exp_results:
                    # Multiple experiments case: add all experiments
                    for exp_result in exp_results:
                        all_experiments.append({
                            "is_correct": exp_result.get("is_correct", False),
                            "score": exp_result.get("score", 0.0),
                        })
                else:
                    # Single experiment case: add the problem's result
                    is_correct = result.get("is_correct", False)
                    score = result.get("score", 0.0)
                    all_experiments.append({
                        "is_correct": is_correct,
                        "score": score,
                    })
            
            if all_experiments:
                # Calculate pass@1 as average accuracy across all experiments
                total_experiments = len(all_experiments)
                correct_experiments = sum(1 for exp in all_experiments if exp["is_correct"])
                pass_at_1 = correct_experiments / total_experiments if total_experiments > 0 else 0.0
                avg_score = sum(exp["score"] for exp in all_experiments) / total_experiments if total_experiments > 0 else 0.0
                
                # Also calculate per-problem average accuracy for comparison
                def get_accuracy(result):
                    is_correct = result.get("is_correct", False)
                    if isinstance(is_correct, (int, float)):
                        return float(is_correct)
                    elif isinstance(is_correct, bool):
                        return 1.0 if is_correct else 0.0
                    else:
                        return 0.0
                
                per_problem_accuracies = [get_accuracy(r) for r in valid_results]
                per_problem_avg_pass_at_1 = sum(per_problem_accuracies) / total_count if total_count > 0 else 0.0
                
                # Also calculate majority vote pass@1 for comparison (>= 0.5)
                majority_vote_correct = sum(1 for acc in per_problem_accuracies if acc >= 0.5)
                majority_vote_pass_at_1 = majority_vote_correct / total_count if total_count > 0 else 0.0
                
                # Count skipped problems
                skipped_count = sum(1 for r in strategy_results if r.get("skipped", False))
                
                aggregate_metrics = {
                    "pass@1": pass_at_1,  # Average accuracy across all experiments (problems * num_exps)
                    "per_problem_avg_pass@1": per_problem_avg_pass_at_1,  # Average accuracy per problem, then averaged
                    "majority_vote_pass@1": majority_vote_pass_at_1,  # Majority vote (>=0.5) for comparison
                    "total_problems": total_count,
                    "total_experiments": total_experiments,  # Total number of experiments
                    "correct_experiments": correct_experiments,  # Number of correct experiments
                    "correct_problems": majority_vote_correct,  # Number of problems with >=50% accuracy
                    "avg_score": avg_score,
                    "skipped_problems": skipped_count,
                }
            else:
                # Fallback if no experiments found
                avg_score = sum(r.get("score", 0.0) for r in valid_results) / total_count if total_count > 0 else 0.0
                skipped_count = sum(1 for r in strategy_results if r.get("skipped", False))
                aggregate_metrics = {
                    "pass@1": 0.0,
                    "per_problem_avg_pass@1": 0.0,
                    "majority_vote_pass@1": 0.0,
                    "total_problems": total_count,
                    "total_experiments": 0,
                    "correct_experiments": 0,
                    "correct_problems": 0,
                    "avg_score": avg_score,
                    "skipped_problems": skipped_count,
                }
        else:
            aggregate_metrics = {}
        
        # Prepare final output for this strategy
        final_output_data = {
            "source_files": [str(f) for f in result_files],
            "benchmark": benchmark,
            "config": config_dict,
            "experiment_config": {**experiment_config, "strategy": strategy_name},
            "start_time": strategy_start_time.isoformat(),
            "end_time": strategy_end_time.isoformat(),
            "duration_seconds": strategy_duration,
            "aggregate_metrics": aggregate_metrics,
            "results": strategy_results,
        }
        
        # Save final results (overwrite with final metrics)
        with open(strategy_output_file, "w", encoding="utf-8") as f:
            json.dump(final_output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Strategy {strategy_name} completed")
        print(f"  Results saved to: {strategy_output_file}")
        print(f"  Duration: {strategy_duration:.2f} seconds")
        if aggregate_metrics:
            print(f"  Pass@1 (all experiments): {aggregate_metrics['pass@1']:.4f} ({aggregate_metrics['correct_experiments']}/{aggregate_metrics['total_experiments']} experiments)")
            if 'per_problem_avg_pass@1' in aggregate_metrics:
                print(f"  Per-problem avg Pass@1: {aggregate_metrics['per_problem_avg_pass@1']:.4f} ({aggregate_metrics['total_problems']} problems)")
            if 'majority_vote_pass@1' in aggregate_metrics:
                print(f"  Majority vote Pass@1 (>=0.5): {aggregate_metrics['majority_vote_pass@1']:.4f} ({aggregate_metrics['correct_problems']}/{aggregate_metrics['total_problems']})")
            print(f"  Avg Score: {aggregate_metrics['avg_score']:.4f}")
        
        all_strategy_results[strategy_name] = {
            "output_file": str(strategy_output_file),
            "metrics": aggregate_metrics,
            "duration": strategy_duration,
        }
    
    # Print final summary
    print("\n" + "=" * 80)
    print("ALL STRATEGIES COMPLETED")
    print("=" * 80)
    print(f"Total Problems: {len(results)}")
    print("\nStrategy Performance Summary:")
    for strategy_name, strategy_info in all_strategy_results.items():
        metrics = strategy_info["metrics"]
        if metrics:
            print(f"  {strategy_name}:")
            print(f"    Pass@1 (all experiments): {metrics['pass@1']:.4f} ({metrics['correct_experiments']}/{metrics['total_experiments']} experiments)")
            if 'per_problem_avg_pass@1' in metrics:
                print(f"    Per-problem avg Pass@1: {metrics['per_problem_avg_pass@1']:.4f} ({metrics['total_problems']} problems)")
            if 'majority_vote_pass@1' in metrics:
                print(f"    Majority vote Pass@1 (>=0.5): {metrics['majority_vote_pass@1']:.4f} ({metrics['correct_problems']}/{metrics['total_problems']})")
            print(f"    Avg Score: {metrics['avg_score']:.4f}")
            print(f"    Duration: {strategy_info['duration']:.2f}s")
            print(f"    Output: {strategy_info['output_file']}")
    print("=" * 80)
    
    return {
        "strategies": all_strategy_results,
        "total_problems": len(results),
    }


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run aggregation experiments on LCB Pro result files"
    )
    
    parser.add_argument(
        "--result-files",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to result JSON file(s) containing solutions. "
             "Multiple files can be provided to combine solutions from different runs.",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="aggregation_experiments",
        help="Output directory for experiment results (default: aggregation_experiments)",
    )
    
    parser.add_argument(
        "--solutions-per-exp",
        type=int,
        default=4,
        help="Number of solutions to randomly sample for each experiment from all available "
             "iteration=<iteration> solutions. (default: 4)",
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for LLM service (defaults to result file config or OPENAI_API_KEY env var)",
    )
    
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL for LLM service (defaults to result file config or OPENAI_API_BASE/SGLANG_API_BASES env var)",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/qwen3-4b",
        help="Model to use for experiments (default: qwen3)",
    )
        
    parser.add_argument(
        "--task-domain",
        type=str,
        default="coding",
        help="Task domain for template selection (default: coding for lcb_pro)",
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum number of problems to process concurrently (default: 40, or MAX_CONCURRENT_PROBLEMS env var)",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from latest results files for each strategy",
    )
    
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from specific results file (applies to all strategies). "
             "If not provided and --resume is set, auto-detects latest file for each strategy.",
    )
    
    parser.add_argument(
        "--service-url",
        type=str,
        default=None,
        help="Remote code verify service URL(s), comma-separated (defaults to CODE_VERIFY_URL env var)",
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="cpp",
        help="Programming language for code evaluation (default: cpp)",
    )
    
    parser.add_argument(
        "--num-exps",
        type=int,
        default=1,
        help="Number of experiments to run per problem. Each experiment randomly samples "
             "--solutions-per-exp solutions from all iteration=<iteration> solutions, and (for generate_from_n, "
             "select_best, pairwise_comparison) randomizes solution order. Results are averaged "
             "across experiments. (default: 1)",
    )
    
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="auto",
        help="Reasoning effort level for LLM aggregation strategies. "
             "Controls the amount of reasoning/thinking the LLM performs during aggregation. "
             "Common values: 'low', 'medium', 'high', 'auto'. (default: auto)",
    )
    
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Iteration number to extract solutions from. "
             "Only solutions with metadata.iteration == <iteration> will be used. (default: 0)",
    )
    
    parser.add_argument(
        "--solution-file",
        type=Path,
        default=None,
        help="Optional solution result file. When a problem is not found in the latest result file, "
             "this script will try to find it in the solution-file. If found and exp_results contain "
             "'Remote evaluation error', they will be re-evaluated. If not found, the problem will be "
             "treated as new and aggregated. If not provided, --solution-timestamp will be used instead.",
    )
    
    parser.add_argument(
        "--solution-timestamp",
        type=str,
        default=None,
        help="Optional timestamp string (e.g., '20260126_120000'). When provided, each strategy will "
             "automatically use solution-file at: output_dir/aggregation_experiment_{strategy}_{benchmark}_{timestamp}.json "
             "Priority: --solution-file > --solution-timestamp",
    )
    
    args = parser.parse_args()
    
    # Validate all result files exist
    for result_file in args.result_files:
        if not result_file.exists():
            print(f"Error: Result file not found: {result_file}")
            return
    
    # Validate solution file if explicitly provided (not via timestamp)
    if args.solution_file and not args.solution_file.exists():
        print(f"Error: Solution file not found: {args.solution_file}")
        return
    
    await run_aggregation_experiment(
        result_files=args.result_files,
        output_dir=args.output_dir,
        solutions_per_exp=args.solutions_per_exp,
        api_key=args.api_key,
        api_base=args.api_base,
        task_domain=args.task_domain,
        max_concurrent=args.max_concurrent,
        resume=args.resume,
        resume_from=args.resume_from,
        service_url=args.service_url,
        language=args.language,
        num_exps=args.num_exps,
        reasoning_effort=args.reasoning_effort,
        iteration=args.iteration,
        solution_file=args.solution_file,
        solution_timestamp=args.solution_timestamp,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    asyncio.run(main())

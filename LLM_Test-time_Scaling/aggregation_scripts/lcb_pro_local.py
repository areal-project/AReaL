"""Run aggregation experiments on LCB Pro result files using LOCAL evaluator.

This script is similar to lcb_pro_final.py but uses LCBProEvaluator (local evaluator)
instead of RemoteLCBProEvaluator. All evaluations go through a global queue that
limits concurrent evaluations to 8 at most.

Usage:
    # Basic usage:
    python -m aggregation_scripts.lcb_pro_local \
        --result-files results.json \
        --output-dir aggregation_experiments \
        --solutions-per-exp 4 \
        --local-data-dir /path/to/lcb_testcases/data
    
    # With solution file (fallback for missing problems):
    python -m aggregation_scripts.lcb_pro_local \
        --result-files results.json \
        --output-dir aggregation_experiments \
        --solutions-per-exp 4 \
        --local-data-dir /path/to/lcb_testcases/data \
        --solution-file previous_results.json
    
    # With solution timestamp (auto-generate solution file per strategy):
    python -m aggregation_scripts.lcb_pro_local \
        --result-files results.json \
        --output-dir aggregation_experiments \
        --solutions-per-exp 4 \
        --local-data-dir /path/to/lcb_testcases/data \
        --solution-timestamp 20260126_120000
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

from src.evaluation.lcb_pro_evaluator import LCBProEvaluator, extract_code
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

# Global evaluation queue: limits concurrent evaluations to 8
GLOBAL_EVAL_SEMAPHORE = None
GLOBAL_EVAL_SEMAPHORE_SIZE = 8

# Key used to detect whether a result has evaluation
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


async def evaluate_with_queue(
    evaluator: LCBProEvaluator,
    problem: str,
    solution: str,
    ground_truth: Optional[str] = None,
    problem_id: Optional[str] = None,
    language: str = "cpp",
    **kwargs: Any,
):
    """Evaluate using local evaluator through global queue.
    
    This ensures at most 8 evaluations run concurrently.
    
    Args:
        evaluator: LCBProEvaluator instance
        problem: Problem statement
        solution: Solution code
        ground_truth: Optional ground truth
        problem_id: Problem ID
        language: Programming language
        **kwargs: Additional evaluation parameters
        
    Returns:
        EvaluationResult
    """
    async with GLOBAL_EVAL_SEMAPHORE:
        return await evaluator.evaluate(
            problem=problem,
            solution=solution,
            ground_truth=ground_truth,
            problem_id=problem_id,
            language=language,
            **kwargs,
        )


def load_result_file(result_file: Path) -> Dict[str, Any]:
    """Load result file and extract problems with solutions."""
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
    """Merge multiple result files, combining solutions by problem_id."""
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
            if not problem_id:
                continue
            if problem_id not in problem_map:
                problem_map[problem_id] = []
            problem_map[problem_id].append(result)
    
    # Merge results by problem_id
    merged_results = []
    total_solutions_per_problem = 0
    for problem_id, problem_results in problem_map.items():
        if not problem_results:
            continue
        
        # Use problem info from first result
        merged_result = problem_results[0].copy()
        
        # Collect all solutions with iteration == iteration from all files
        all_solutions = []
        for file_result_idx, result in enumerate(problem_results):
            metadata = result.get("metadata", {})
            exp_results = metadata.get("exp_results", [])
            
            if exp_results:
                # Multi-exp format: extract from exp_results
                for exp_result in exp_results:
                    exp_metadata = exp_result.get("metadata", {})
                    if exp_metadata.get("iteration") == iteration:
                        solution_content = exp_result.get("solution_content") or exp_result.get("content", "")
                        if solution_content.strip():
                            all_solutions.append({
                                "solution_content": solution_content,
                                "content": solution_content,
                                "is_correct": exp_result.get("is_correct"),
                                "score": exp_result.get("score"),
                                "feedback": exp_result.get("feedback"),
                                "metadata": exp_metadata,
                            })
            else:
                # Single-exp format: check top-level metadata
                if metadata.get("iteration") == iteration:
                    solution_content = result.get("solution_content") or result.get("content", "")
                    if solution_content.strip():
                        all_solutions.append({
                            "solution_content": solution_content,
                            "content": solution_content,
                            "is_correct": result.get("is_correct"),
                            "score": result.get("score"),
                            "feedback": result.get("feedback"),
                            "metadata": metadata,
                        })
        
        # Store all solutions in metadata
        merged_result["metadata"] = {
            "all_solutions": all_solutions,
            "num_solutions": len(all_solutions),
        }
        
        merged_results.append(merged_result)
        total_solutions_per_problem += len(all_solutions)
    
    avg_solutions = total_solutions_per_problem / len(merged_results) if merged_results else 0
    print(f"Average solutions per problem: {avg_solutions:.1f} (total: {total_solutions_per_problem} from {len(result_files)} files)")
    
    return {
        "config": config_dict,
        "results": merged_results,
        "benchmark": benchmark,
    }


def extract_random_n_solutions(
    result_data: Dict[str, Any], n: int, random_seed: Optional[int] = None
) -> List[Solution]:
    """Extract n random solutions from result_data."""
    if random_seed is not None:
        random.seed(random_seed)
    
    metadata = result_data.get("metadata", {})
    all_solutions = metadata.get("all_solutions", [])
    
    if len(all_solutions) < n:
        return [Solution(content="", metadata={}) for _ in range(len(all_solutions))]
    
    selected = random.sample(all_solutions, n)
    
    solutions = []
    for idx, sol_data in enumerate(selected):
        solution_content = sol_data.get("solution_content") or sol_data.get("content", "")
        is_correct = sol_data.get("is_correct")
        score = sol_data.get("score")
        feedback = sol_data.get("feedback")
        
        original_metadata = sol_data.get("metadata", {}).copy()
        if is_correct is not None:
            original_metadata["is_correct"] = is_correct
        if score is not None:
            original_metadata["score"] = score
        
        solution = Solution(
            content=solution_content,
            score=score,
            feedback=feedback,
            metadata={
                "original_index": idx,
                "original_metadata": original_metadata,
                "is_correct": is_correct,
            },
        )
        solutions.append(solution)
    
    return solutions


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
    """Save results incrementally to disk."""
    with save_lock:
        existing_file_results = []
        if output_file_path.exists():
            try:
                with open(output_file_path, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                    existing_file_results = file_data.get("results", [])
            except Exception:
                existing_file_results = []
        
        merged_results = [
            r for r in existing_file_results
            if r.get("problem_id") not in {res.get("problem_id") for res in all_current_results}
        ]
        merged_results.extend(all_current_results)
        
        strategy_results = [
            r for r in merged_results
            if "error" not in r and not r.get("skipped", False)
        ]
        
        if strategy_results:
            total_count = len(strategy_results)
            all_experiments = []
            for result in strategy_results:
                metadata = result.get("metadata", {})
                exp_results = metadata.get("exp_results", [])
                
                if exp_results:
                    for exp_result in exp_results:
                        all_experiments.append({
                            "is_correct": exp_result.get("is_correct", False),
                            "score": exp_result.get("score", 0.0),
                        })
                else:
                    is_correct = result.get("is_correct", False)
                    score = result.get("score", 0.0)
                    all_experiments.append({
                        "is_correct": is_correct,
                        "score": score,
                    })
            
            if all_experiments:
                total_experiments = len(all_experiments)
                correct_experiments = sum(1 for exp in all_experiments if exp["is_correct"])
                pass_at_1 = correct_experiments / total_experiments if total_experiments > 0 else 0.0
                avg_score = sum(exp["score"] for exp in all_experiments) / total_experiments if total_experiments > 0 else 0.0
                
                def get_accuracy(result):
                    is_correct = result.get("is_correct", False)
                    if isinstance(is_correct, (int, float)):
                        return float(is_correct)
                    elif isinstance(is_correct, bool):
                        return 1.0 if is_correct else 0.0
                    return 0.0
                
                per_problem_avg = sum(get_accuracy(r) for r in strategy_results) / total_count if total_count > 0 else 0.0
                per_problem_score = sum(r.get("score", 0.0) for r in strategy_results) / total_count if total_count > 0 else 0.0
                
                aggregate_metrics = {
                    "total_problems": total_count,
                    "correct_problems": sum(1 for r in strategy_results if get_accuracy(r) > 0),
                    "pass@1": pass_at_1,
                    "avg_score": avg_score,
                    "per_problem_avg_accuracy": per_problem_avg,
                    "per_problem_avg_score": per_problem_score,
                }
            else:
                aggregate_metrics = {
                    "total_problems": total_count,
                    "correct_problems": 0,
                    "pass@1": 0.0,
                    "avg_score": 0.0,
                    "per_problem_avg_accuracy": 0.0,
                    "per_problem_avg_score": 0.0,
                }
        else:
            aggregate_metrics = {
                "total_problems": 0,
                "correct_problems": 0,
                "pass@1": 0.0,
                "avg_score": 0.0,
                "per_problem_avg_accuracy": 0.0,
                "per_problem_avg_score": 0.0,
            }
        
        duration = (datetime.now() - start_time).total_seconds()
        
        output_data = {
            "config": config_dict,
            "benchmark": benchmark,
            "experiment_config": {
                **experiment_config,
                "strategy": strategy_name,
            },
            "aggregate_metrics": aggregate_metrics,
            "results": merged_results,
            "source_files": [str(f) for f in source_files],
            "start_time": start_time.isoformat(),
            "last_update": datetime.now().isoformat(),
            "duration_seconds": duration,
        }
        
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        problem_id = new_result.get("problem_id", "unknown")
        print(f"  ✓ Saved intermediate result for problem {problem_id} ({len(merged_results)} total)", flush=True)


def _load_existing_results_for_strategy(
    resume_file: Path, strategy_name: str, num_exps: int = 1, eval_key: str = EVAL_KEY
) -> tuple[set[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load existing results for a strategy."""
    completed_problem_ids = set()
    existing_results = []
    reeval_results: List[Dict[str, Any]] = []

    if not resume_file.exists():
        return completed_problem_ids, existing_results, reeval_results

    try:
        with open(resume_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        exp_config = data.get("experiment_config", {})
        file_strategy = exp_config.get("strategy")

        if file_strategy != strategy_name:
            return completed_problem_ids, existing_results, reeval_results

        results = data.get("results", [])
        for result in results:
            if "problem_id" not in result:
                continue
            problem_id = result["problem_id"]

            if "error" in result:
                continue

            metadata = result.get("metadata", {})
            exp_results = metadata.get("exp_results", [])

            if num_exps > 1:
                if len(exp_results) < num_exps:
                    continue
                
                missing = [i for i, ex in enumerate(exp_results) if eval_key not in ex]
                has_error_feedback = []
                
                for i, ex in enumerate(exp_results):
                    if eval_key in ex:
                        eval_res = ex.get(eval_key, {})
                        if isinstance(eval_res, dict):
                            feedback = eval_res.get("feedback", "") or ""
                            if "Remote evaluation error" in feedback:
                                has_error_feedback.append(i)
                
                if missing or has_error_feedback:
                    reeval_results.append(result)
                else:
                    completed_problem_ids.add(problem_id)
                    existing_results.append(result)
            else:
                has_eval = eval_key in result
                has_error_feedback = False
                if has_eval:
                    eval_res = result.get(eval_key, {})
                    if isinstance(eval_res, dict):
                        feedback = eval_res.get("feedback", "") or ""
                        if "Remote evaluation error" in feedback:
                            has_error_feedback = True
                
                if not has_eval or has_error_feedback:
                    reeval_results.append(result)
                else:
                    completed_problem_ids.add(problem_id)
                    existing_results.append(result)

    except Exception as e:
        print(f"  Warning: Failed to load resume file: {e}")
        return completed_problem_ids, existing_results, reeval_results

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
    local_data_dir: Optional[str] = None,
    language: str = "cpp",
    num_exps: int = 1,
    reasoning_effort: str = "auto",
    iteration: int = 0,
    solution_file: Optional[Path] = None,
    solution_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Run aggregation experiment using LOCAL evaluator.
    
    Args:
        result_files: List of paths to result JSON files
        output_dir: Output directory for results
        solutions_per_exp: Number of solutions to randomly sample for each experiment
        api_key: API key for LLM service
        api_base: API base URL for LLM service
        task_domain: Task domain for template selection
        max_concurrent: Maximum number of problems to process concurrently
        resume: Auto-resume from latest results files for each strategy
        resume_from: Resume from specific results file
        local_data_dir: Local data directory for testcases
        language: Programming language for code evaluation
        num_exps: Number of experiments to run per problem
        reasoning_effort: Reasoning effort level for LLM aggregation strategies
        iteration: Iteration number to extract solutions from
        solution_file: Optional solution result file. When a problem is not found in the latest
                      result file, this script will try to find it in the solution-file. If found
                      and exp_results contain 'Remote evaluation error', they will be re-evaluated.
                      If not found, the problem will be treated as new and aggregated.
                      If not provided, solution_timestamp will be used instead.
        solution_timestamp: Optional timestamp string (e.g., "20260126_120000"). When provided,
                           each strategy will automatically use solution-file at:
                           output_dir/aggregation_experiment_{strategy}_{benchmark}_{solution_timestamp}.json
                           Priority: solution_file > solution_timestamp
    """
    global GLOBAL_EVAL_SEMAPHORE
    
    # Initialize global evaluation semaphore (max 8 concurrent evaluations)
    GLOBAL_EVAL_SEMAPHORE = asyncio.Semaphore(GLOBAL_EVAL_SEMAPHORE_SIZE)
    print(f"Global evaluation queue initialized: max {GLOBAL_EVAL_SEMAPHORE_SIZE} concurrent evaluations")
    
    # Merge result files
    data = merge_result_files(result_files, iteration=iteration)
    config_dict = data["config"]
    results = data["results"]
    benchmark = data["benchmark"]
    
    # Extract LLM config
    llm_config_dict = config_dict.get("llm", {})
    model_name = llm_config_dict.get("model_name", "openai/gpt-oss-120b")
    api_key = api_key or llm_config_dict.get("api_key") or os.getenv("OPENAI_API_KEY", "None")
    api_base = os.getenv("OPENAI_API_BASE") or os.getenv("SGLANG_API_BASES")
    
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
    
    # Create LOCAL LCB Pro evaluator
    if not local_data_dir:
        default_locations = [
            project_root / "data" / "local_data" / "lcb_testcases" / "data",
            project_root / "data" / "benchmarks" / "lcb_testcases" / "data",
            Path("path-to-results/llm_test_time_scaling/data/local_data/lcb_testcases/data"),
            Path("path-to-results/LLM_Test-time_Scaling/data/local_data/lcb_testcases/data"),
        ]
        
        for default_path in default_locations:
            if default_path.exists():
                local_data_dir = str(default_path)
                print(f"Using default data directory: {local_data_dir}")
                break
        
        if not local_data_dir:
            raise ValueError("local_data_dir is required. Provide via --local-data-dir or ensure default location exists")
    else:
        if not Path(local_data_dir).exists():
            raise ValueError(f"Specified data directory does not exist: {local_data_dir}")
        print(f"Using specified data directory: {local_data_dir}")
    
    evaluator = LCBProEvaluator(local_data_dir=local_data_dir)
    print("Using LOCAL evaluator (LCBProEvaluator)")
    
    # Create aggregation strategies
    strategies = {}
    
    generate_template = prompt_manager.get_template("aggregation_generate_one_from_n_coding")
    if generate_template is None:
        generate_template = prompt_manager.get_template("aggregation_generate_one_from_n")
    strategies["generate_from_n"] = GenerateFromNAggregation(
        llm_service=llm_service,
        generation_prompt_template=generate_template,
        temperature=0.7,
        reasoning_effort=reasoning_effort,
    )
    
    select_template = prompt_manager.get_template("aggregation_select_one_from_n_coding")
    if select_template is None:
        select_template = prompt_manager.get_template("aggregation_select_one_from_n")
    strategies["select_best"] = SelectBestAggregation(
        llm_service=llm_service,
        selection_prompt_template=select_template,
        temperature=0.0,
        reasoning_effort=reasoning_effort,
    )
    
    scoring_template = prompt_manager.get_template("code_llm_scoring")
    if scoring_template is None:
        scoring_template = prompt_manager.get_template("code_llm_scoring")
    strategies["llm_scoring"] = LLMScoringAggregation(
        llm_service=llm_service,
        scoring_prompt_template=scoring_template,
        temperature=0.0,
        reasoning_effort=reasoning_effort,
    )
    
    pairwise_template = prompt_manager.get_template("aggregation_pairwise_comparison_coding")
    if pairwise_template is None:
        pairwise_template = prompt_manager.get_template("aggregation_pairwise_comparison")
    # strategies = {}
    if reasoning_effort == "low":
        strategies["pairwise_comparison"] = PairwiseComparisonAggregation(
            llm_service=llm_service,
            comparison_prompt_template=pairwise_template,
            temperature=0.0,
            reasoning_effort=reasoning_effort,
        )
    
    # Prepare experiment config
    experiment_config = {
        "num_result_files": len(result_files),
        "solutions_per_exp": solutions_per_exp,
        "task_domain": task_domain,
        "model_name": model_name,
        "language": language,
        "num_exps": num_exps,
        "reasoning_effort": reasoning_effort,
        "iteration": iteration,
        "evaluator": "local",
    }
    
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nRunning aggregation experiments on {len(results)} problems...")
    print(f"Strategies: {list(strategies.keys())}")
    if solution_file:
        print(f"Solution file: {solution_file} (will be used as fallback for all strategies)")
    elif solution_timestamp:
        print(f"Solution timestamp: {solution_timestamp} (each strategy will use its own solution-file)")
    
    # Create semaphore for limiting concurrent problem processing
    if max_concurrent is None:
        max_concurrent = int(os.getenv("MAX_CONCURRENT_PROBLEMS", "40"))
    global_semaphore = asyncio.Semaphore(max_concurrent)
    
    all_strategy_results = {}
    
    # Store strategy-specific info for each task (defined outside loop like lcb_pro_final.py)
    strategy_info = {}  # Store strategy-specific info for each task
    all_tasks = []  # Store all tasks across all strategies
    
    # Run each strategy
    for strategy_idx, (strategy_name, strategy) in enumerate(strategies.items(), 1):
        print(f"\n{'=' * 80}")
        print(f"STRATEGY {strategy_idx}/{len(strategies)}: {strategy_name}")
        print(f"{'=' * 80}")
        
        strategy_output_file = output_path / f"aggregation_experiment_{strategy_name}_{benchmark}_{timestamp}.json"
        
        # Determine resume file for this strategy
        strategy_resume_file = None
        if resume_from:
            strategy_resume_file = Path(resume_from)
        elif resume:
            strategy_resume_file = _find_latest_result_file(output_path, strategy_name, benchmark)
            if strategy_resume_file:
                print(f"  Auto-detected resume file: {strategy_resume_file}")
        
        # Load existing results if resuming
        completed_problem_ids = set()
        existing_results = []
        reeval_results = []
        if strategy_resume_file and strategy_resume_file.exists():
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
                            if "Remote evaluation error" in feedback:
                                needs_reeval_from_solution = True
                                break
                elif num_exps == 1:
                    # Single-exp: check top-level eval_result
                    eval_res = solution_result.get(EVAL_KEY, {})
                    if isinstance(eval_res, dict):
                        feedback = eval_res.get("feedback", "") or ""
                        if "Remote evaluation error" in feedback:
                            needs_reeval_from_solution = True
                
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
            if strategy_resume_file:
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
        
        strategy_start_time = datetime.now()
        strategy_results = existing_results.copy()
        results_lock = asyncio.Lock()
        save_lock = threading.Lock()
        
        # Store strategy info in the global strategy_info dict (like lcb_pro_final.py)
        strategy_info[strategy_name] = {
            "strategy": strategy,
            "strategy_name": strategy_name,
            "strategy_results": strategy_results,
            "strategy_output_file": strategy_output_file,
            "save_lock": save_lock,
            "strategy_start_time": strategy_start_time,
            "remaining_results": aggregate_remaining,
            "existing_results": existing_results,
            "reeval_results": reeval_results,
            "results_lock": results_lock,
        }
        
        # Create tasks for this strategy
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
    
    # Define unified processing function (outside strategy loop, like lcb_pro_final.py)
    async def process_problem_unified(task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single problem with its assigned strategy."""
        strategy_name = task_info["strategy_name"]
        task_type = task_info.get("task_type", "aggregate")
        idx = task_info["idx"]
        total = task_info["total"]
        info = strategy_info[strategy_name]
        strategy = info["strategy"]
        
        # Re-eval only
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
                            # Single-exp
                            solution = stored.get("solution_content") or stored.get("content") or ""
                            if not solution.strip():
                                return stored
                            
                            # Use queue for evaluation
                            ev = await evaluate_with_queue(
                                evaluator=evaluator,
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
                            # Multi-exp
                            updated_metadata = {**metadata}
                            updated_exps = [dict(ex) for ex in exp_results_list]
                            for i, ex in enumerate(updated_exps):
                                needs_reeval = False
                                if EVAL_KEY not in ex:
                                    needs_reeval = True
                                else:
                                    eval_res = ex.get(EVAL_KEY, {})
                                    if isinstance(eval_res, dict):
                                        feedback = eval_res.get("feedback", "") or ""
                                        if "Remote evaluation error" in feedback:
                                            needs_reeval = True
                                
                                if not needs_reeval:
                                    continue
                                
                                sol = ex.get("solution_content") or ex.get("content") or ""
                                if not sol.strip():
                                    continue
                                
                                # Use queue for evaluation
                                ev = await evaluate_with_queue(
                                    evaluator=evaluator,
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
                        
                        return updated
                    except Exception as e:
                        print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Re-eval error - {e}", flush=True)
                        err = {**stored, "error": str(e)}
                        async with info["results_lock"]:
                            info["strategy_results"].append(err)
                        return err
        
        # Aggregate (full run)
        result_data = task_info["result_data"]
        async with global_semaphore:
            problem_id = result_data.get("problem_id", f"problem_{idx}")
            problem = result_data.get("problem", "")
            ground_truth = result_data.get("ground_truth", "")
            
            try:
                strategies_that_need_randomization = {"generate_from_n", "select_best", "pairwise_comparison"}
                needs_randomization = strategy_name in strategies_that_need_randomization
                
                if num_exps > 1:
                    exp_results = []
                    all_exp_solutions = []
                    
                    for exp_idx in range(num_exps):
                        solutions = extract_random_n_solutions(
                            result_data, solutions_per_exp, random_seed=hash((problem_id, exp_idx)) % (2**31)
                        )
                        
                        if len(solutions) < solutions_per_exp:
                            if exp_idx == 0:
                                print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Skipping (only {len(solutions)} solutions, need {solutions_per_exp})", flush=True)
                            continue
                        
                        if all(not sol.content.strip() for sol in solutions):
                            if exp_idx == 0:
                                print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Skipping (all solutions empty)", flush=True)
                            continue
                        
                        randomize_order = needs_randomization
                        if strategy_name == "gt_scoring":
                            aggregated_solution = await strategy.aggregate(
                                problem=problem, solutions=solutions, ground_truth=ground_truth
                            )
                        else:
                            aggregated_solution = await strategy.aggregate(
                                problem=problem, solutions=solutions, randomize_order=randomize_order
                            )
                        
                        if aggregated_solution.metadata.get("skipped", False):
                            continue
                        
                        # Evaluate using queue
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
                            # Use queue for evaluation
                            eval_result = await evaluate_with_queue(
                                evaluator=evaluator,
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
                    
                    avg_score = sum(r["score"] for r in exp_results) / len(exp_results)
                    correct_count = sum(1 for r in exp_results if r["is_correct"])
                    pass_rate = correct_count / len(exp_results)
                    
                    last_exp_solution = all_exp_solutions[-1] if all_exp_solutions else None
                    result_metadata = last_exp_solution.metadata if last_exp_solution else {}
                    extracted_code = extract_code(last_exp_solution.content if last_exp_solution else "")
                    code = extracted_code if extracted_code is not None else ""
                    token_usage = result_metadata.get("token_usage", {})
                    
                    problem_result = {
                        "problem_id": problem_id,
                        "problem": problem,
                        "ground_truth": ground_truth,
                        "solution_content": last_exp_solution.content if last_exp_solution else "",
                        "content": last_exp_solution.content if last_exp_solution else "",
                        "code": code,
                        "is_correct": pass_rate,
                        "score": avg_score,
                        "feedback": f"num_exps={num_exps}, pass_rate={pass_rate:.2f}, avg_score={avg_score:.2f}",
                        "token_usage": token_usage,
                        "metadata": {
                            **result_metadata,
                            "num_exps": num_exps,
                            "exp_results": exp_results,
                            "pass_rate": pass_rate,
                            "avg_score": avg_score,
                        },
                    }
                    
                    print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: {'✓' if problem_result['is_correct'] else '✗'} (num_exps={num_exps}, pass_rate={pass_rate:.2f}, avg_score={avg_score:.2f})", flush=True)
                else:
                    # Single experiment
                    solutions = extract_random_n_solutions(result_data, solutions_per_exp)
                    
                    if len(solutions) < solutions_per_exp:
                        print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Skipping (only {len(solutions)} solutions, need {solutions_per_exp})", flush=True)
                        return None
                    
                    if all(not sol.content.strip() for sol in solutions):
                        print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Skipping (all solutions empty)", flush=True)
                        return None
                    
                    randomize_order = needs_randomization
                    if strategy_name == "gt_scoring":
                        aggregated_solution = await strategy.aggregate(
                            problem=problem, solutions=solutions, ground_truth=ground_truth
                        )
                    else:
                        aggregated_solution = await strategy.aggregate(
                            problem=problem, solutions=solutions, randomize_order=randomize_order
                        )
                    
                    if aggregated_solution.metadata.get("skipped", False):
                        return None
                    
                    # Use queue for evaluation
                    eval_result = await evaluate_with_queue(
                        evaluator=evaluator,
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
                        "feedback": getattr(eval_result, "feedback", None),
                        "token_usage": token_usage,
                        "eval_result": eval_result_dict,
                        "metadata": aggregated_solution.metadata.copy(),
                    }
                    
                    print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: {'✓' if is_correct else '✗'} (score: {score:.2f})", flush=True)
                
                async with info["results_lock"]:
                    info["strategy_results"].append(problem_result)
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
                err = {
                    "problem_id": problem_id,
                    "error": str(e),
                }
                async with info["results_lock"]:
                    info["strategy_results"].append(err)
                return err
    
    # Execute all tasks across all strategies
    if all_tasks:
        tasks = [process_problem_unified(task_info) for task_info in all_tasks]
        await tqdm.gather(*tasks, desc="Processing all strategies")
    
    # Calculate final metrics for each strategy
    for strategy_idx, (strategy_name, strategy) in enumerate(strategies.items(), 1):
        info = strategy_info[strategy_name]
        strategy_results = info["strategy_results"]
        strategy_start_time = info["strategy_start_time"]
        strategy_output_file = info["strategy_output_file"]
        
        strategy_duration = (datetime.now() - strategy_start_time).total_seconds()
        valid_results = [r for r in strategy_results if "error" not in r]
        
        if valid_results:
            all_experiments = []
            for result in valid_results:
                metadata = result.get("metadata", {})
                exp_results = metadata.get("exp_results", [])
                if exp_results:
                    for exp_result in exp_results:
                        all_experiments.append({
                            "is_correct": exp_result.get("is_correct", False),
                            "score": exp_result.get("score", 0.0),
                        })
                else:
                    all_experiments.append({
                        "is_correct": result.get("is_correct", False),
                        "score": result.get("score", 0.0),
                    })
            
            if all_experiments:
                total_experiments = len(all_experiments)
                correct_experiments = sum(1 for exp in all_experiments if exp["is_correct"])
                pass_at_1 = correct_experiments / total_experiments if total_experiments > 0 else 0.0
                avg_score = sum(exp["score"] for exp in all_experiments) / total_experiments if total_experiments > 0 else 0.0
                
                aggregate_metrics = {
                    "total_problems": len(valid_results),
                    "correct_problems": sum(1 for r in valid_results if (r.get("is_correct", False) if isinstance(r.get("is_correct"), bool) else r.get("is_correct", 0) > 0)),
                    "pass@1": pass_at_1,
                    "avg_score": avg_score,
                }
            else:
                aggregate_metrics = {
                    "total_problems": len(valid_results),
                    "correct_problems": 0,
                    "pass@1": 0.0,
                    "avg_score": 0.0,
                }
        else:
            aggregate_metrics = {
                "total_problems": 0,
                "correct_problems": 0,
                "pass@1": 0.0,
                "avg_score": 0.0,
            }
        
        # Save final results
        final_output_data = {
            "config": config_dict,
            "benchmark": benchmark,
            "experiment_config": {
                **experiment_config,
                "strategy": strategy_name,
            },
            "aggregate_metrics": aggregate_metrics,
            "results": strategy_results,
        }
        
        with open(strategy_output_file, "w", encoding="utf-8") as f:
            json.dump(final_output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Strategy {strategy_name} completed")
        print(f"  Results saved to: {strategy_output_file}")
        print(f"  Duration: {strategy_duration:.2f} seconds")
        if aggregate_metrics:
            print(f"  Pass@1: {aggregate_metrics['pass@1']:.4f} ({aggregate_metrics['correct_problems']}/{aggregate_metrics['total_problems']})")
            print(f"  Avg Score: {aggregate_metrics['avg_score']:.4f}")
        
        all_strategy_results[strategy_name] = {
            "output_file": str(strategy_output_file),
            "metrics": aggregate_metrics,
            "duration": strategy_duration,
        }
    
    return {
        "strategies": all_strategy_results,
        "total_problems": len(results),
    }


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run aggregation experiments on LCB Pro result files using LOCAL evaluator"
    )
    
    parser.add_argument(
        "--result-files",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to result JSON file(s)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="aggregation_experiments",
        help="Output directory for experiment results",
    )
    
    parser.add_argument(
        "--solutions-per-exp",
        type=int,
        default=4,
        help="Number of solutions to randomly sample for each experiment",
    )
    
    parser.add_argument(
        "--local-data-dir",
        type=str,
        default=None,
        help="Local data directory for testcases (required if default locations don't exist)",
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for LLM service",
    )
    
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="API base URL for LLM service",
    )
    
    parser.add_argument(
        "--task-domain",
        type=str,
        default="coding",
        help="Task domain for template selection",
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum number of problems to process concurrently",
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
        help="Resume from specific results file",
    )
    
    parser.add_argument(
        "--language",
        type=str,
        default="cpp",
        help="Programming language for code evaluation",
    )
    
    parser.add_argument(
        "--num-exps",
        type=int,
        default=1,
        help="Number of experiments to run per problem",
    )
    
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="auto",
        help="Reasoning effort level for LLM aggregation strategies",
    )
    
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Iteration number to extract solutions from",
    )
    
    parser.add_argument(
        "--solution-file",
        type=Path,
        default=None,
        help="Optional solution result file. When a problem is not found in the latest "
             "result file, this script will try to find it in the solution-file. If found "
             "and exp_results contain 'Remote evaluation error', they will be re-evaluated. "
             "If not found, the problem will be treated as new and aggregated. "
             "If not provided, solution-timestamp will be used instead.",
    )
    
    parser.add_argument(
        "--solution-timestamp",
        type=str,
        default=None,
        help="Optional timestamp string (e.g., '20260126_120000'). When provided, "
             "each strategy will automatically use solution-file at: "
             "output_dir/aggregation_experiment_{strategy}_{benchmark}_{solution_timestamp}.json "
             "Priority: solution-file > solution-timestamp",
    )
    
    args = parser.parse_args()
    
    # Validate all result files exist
    for result_file in args.result_files:
        if not result_file.exists():
            print(f"Error: Result file not found: {result_file}")
            return
    
    # Validate solution_file if provided
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
        local_data_dir=args.local_data_dir,
        language=args.language,
        num_exps=args.num_exps,
        reasoning_effort=args.reasoning_effort,
        iteration=args.iteration,
        solution_file=args.solution_file,
        solution_timestamp=args.solution_timestamp,
    )


if __name__ == "__main__":
    asyncio.run(main())

"""Run aggregation experiments on existing result files.

This script reads result files containing multiple solutions per problem,
and applies different aggregation strategies to combine them.

The number of solutions extracted from each file is determined by:
1. config["reflection"]["n_samples_per_iteration"] from each file (if available)
2. --solutions-per-file parameter (if provided, overrides config)
3. Default: 4 (if neither is available)

Usage:
    # Single file (uses config.reflection.n_samples_per_iteration from file):
    python -m scripts.run_aggregation_experiment \
        --result-files path-to-results/llm_test_time_scaling/results/imobench_rollouts_gpt-oss-120b/self_eval_sequential_4seeds_imobench_20260110_024249.json \
        --output-dir aggregation_experiments 
    
    python -m scripts.run_aggregation_experiment \
        --result-files path-to-results/llm_test_time_scaling/results/imobench_rollouts_gpt-oss-120b/self_eval_sequential_4seeds_imobench_20260110_024249.json \
            path-to-results/llm_test_time_scaling/results/imobench_rollouts_gpt-oss-120b/self_eval_sequential_4seeds_trial2_imobench_20260111_151032.json \
        --output-dir aggregation_experiments/imobench_gt_scoring/8_traj

    # Multiple files (each file uses its own config.reflection.n_samples_per_iteration):
    python -m scripts.run_aggregation_experiment \
        --result-files path-to-results/llm_test_time_scaling/results/imobench_rollouts_gpt-oss-120b/self_eval_sequential_4seeds_imobench_20260110_024249.json \
        --solutions-per-file 2 \
        --output-dir aggregation_experiments/imobench_gt_scoring/2_traj

    python -m scripts.run_aggregation_experiment \
        --result-files results_full_continuous/baseline_8_fixed/direct_evaluation_remote_lcb_pro_20260116_234529.json \
        --solutions-per-file 2 \
        --output-dir aggregation_experiments/coding_random_gt_scoring/2_traj \

    python -m scripts.run_aggregation_experiment \
        --result-files path-to-results/llm_test_time_scaling/results/imobench_rollouts_gpt-oss-120b/self_eval_sequential_4seeds_imobench_20260110_024249.json \
            path-to-results/llm_test_time_scaling/results/imobench_rollouts_gpt-oss-120b/self_eval_sequential_4seeds_trial2_imobench_20260111_151032.json \
            path-to-results/llm_test_time_scaling/results/imobench_rollouts_gpt-oss-120b/baseline_imobench_20260109_190957.json \
        --output-dir aggregation_experiments/imobench_gt_scoring/16_traj
    
    # Override with --solutions-per-file (applies to all files):
    python -m scripts.run_aggregation_experiment \
        --result-files aggregation_results/file1.json aggregation_results/file2.json \
        --output-dir aggregation_experiments \
        --solutions-per-file 4
"""

import argparse
import asyncio
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm.asyncio import tqdm

import sys
# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import IMOBenchEvaluator
from src.evaluation import CodeExecutor, Evaluator, LLMJudge, IMOBenchEvaluator
from src.llm_service import create_llm_service
from src.prompts import PromptManager
from src.scaling.aggregation import (
    GenerateFromNAggregation,
    GTScoringAggregation,
    LLMScoringAggregation,
    PairwiseComparisonAggregation,
    SelectBestAggregation,
    VotingAggregation,
    LLMVotingAggregation
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
    benchmark = data.get("benchmark", "imobench")
    
    print(f"Loaded {len(results)} problem results")
    print(f"Benchmark: {benchmark}")
    
    return {
        "config": config_dict,
        "results": results,
        "benchmark": benchmark,
    }


def merge_result_files(result_files: List[Path], solutions_per_file: Optional[int] = None) -> Dict[str, Any]:
    """Merge multiple result files, combining solutions by problem_id.
    
    For each problem, extracts solutions from each file based on the file's config.
    The number of solutions extracted from each file is determined by:
    - config["reflection"]["n_samples_per_iteration"] if available
    - solutions_per_file parameter if provided (fallback)
    - 4 if neither is available (default)
    
    Solutions are extracted from the same position across files (e.g., 1st solution from 
    file1, 1st solution from file2, 2nd solution from file1, 2nd solution from file2, etc.)
    
    Args:
        result_files: List of paths to result JSON files
        solutions_per_file: Optional override for number of solutions per file.
                           If None, uses config["reflection"]["n_samples_per_iteration"] from each file.
        
    Returns:
        Dictionary with merged config, results, and benchmark
    """
    if not result_files:
        raise ValueError("At least one result file is required")
    
    print(f"\nMerging {len(result_files)} result files...")
    
    # Load all files
    all_data = []
    file_solutions_counts = []  # Track how many solutions per file
    for result_file in result_files:
        data = load_result_file(result_file)
        all_data.append(data)
        
        # Determine solutions per file for this file
        config_dict = data.get("config", {})
        reflection_config = config_dict.get("reflection", {})
        n_samples = reflection_config.get("n_samples_per_iteration")
        
        if n_samples is not None:
            file_solutions_count = n_samples
            print(f"  File {len(all_data)}: Using {file_solutions_count} solutions (from config.reflection.n_samples_per_iteration)")
        if solutions_per_file is not None:
            file_solutions_count = solutions_per_file
            print(f"  File {len(all_data)}: Using {file_solutions_count} solutions (from --solutions-per-file)")
        else:
            file_solutions_count = 4
            print(f"  File {len(all_data)}: Using {file_solutions_count} solutions (default)")
        
        file_solutions_counts.append(file_solutions_count)
    
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
    for problem_id, problem_results in problem_map.items():
        if not problem_results:
            continue
        
        # Use problem info from first result
        merged_result = problem_results[0].copy()
        
        # Combine solutions from all files
        # Keep solutions in order: all from file1, then all from file2, etc.
        combined_solutions = []
        for file_result_idx, result in enumerate(problem_results):
            all_solutions = result.get("all_solutions", [])
            n_solutions_for_this_file = file_solutions_counts[file_result_idx]
            
            for sol_idx in range(min(n_solutions_for_this_file, len(all_solutions))):
                sol_data = all_solutions[sol_idx].copy()
                # Add source file info to metadata
                if "metadata" not in sol_data:
                    sol_data["metadata"] = {}
                sol_data["metadata"]["source_file_index"] = file_result_idx
                sol_data["metadata"]["solution_index_in_file"] = sol_idx
                combined_solutions.append(sol_data)
        
        merged_result["all_solutions"] = combined_solutions
        merged_results.append(merged_result)
    
    # Calculate average solutions per problem
    if merged_results:
        avg_solutions = sum(len(r.get("all_solutions", [])) for r in merged_results) / len(merged_results)
        total_solutions_per_problem = sum(file_solutions_counts)
        print(f"Merged {len(merged_results)} problems")
        print(f"Average solutions per problem: {avg_solutions:.1f} (total: {total_solutions_per_problem} from {len(result_files)} files)")
        print(f"Solutions per file: {file_solutions_counts}")
    else:
        print(f"Merged {len(merged_results)} problems")
    
    return {
        "config": config_dict,
        "results": merged_results,
        "benchmark": benchmark,
        "solutions_per_file": file_solutions_counts,  # Store for later use
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
            correct_count = sum(1 for r in strategy_results if r.get("is_correct", False))
            total_count = len(strategy_results)
            avg_score = sum(r.get("score", 0.0) for r in strategy_results) / total_count if total_count > 0 else 0.0
            
            # Count skipped problems
            skipped_count = sum(1 for r in merged_results if r.get("skipped", False))
            
            aggregate_metrics = {
                "pass@1": correct_count / total_count if total_count > 0 else 0.0,
                "total_problems": total_count,
                "correct_problems": correct_count,
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
        solution_content = sanitize_solution_content(solution_content)
        
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


def sanitize_solution_content(solution_content: Any) -> str:
    """Strip reasoning traces enclosed by <think>...</think>.

    Requirement: if a solution contains think traces, keep only the answer content
    after the final </think>, and remove the think tags themselves.
    """
    if not isinstance(solution_content, str):
        return ""

    cleaned = solution_content

    # Primary rule for Qwen-style outputs: keep only content after the final </think>.
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]

    # Defensive cleanup in case stray think tags remain.
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")

    return cleaned.strip()


def _load_existing_results_for_strategy(
    resume_file: Path, strategy_name: str
) -> tuple[set[str], List[Dict[str, Any]]]:
    """Load existing results for a strategy and extract completed problem IDs.
    
    Args:
        resume_file: Path to existing results JSON file
        strategy_name: Name of the strategy to filter by
        
    Returns:
        Tuple of (completed_problem_ids set, existing_results list)
    """
    completed_problem_ids = set()
    existing_results = []
    
    if not resume_file.exists():
        print(f"  Warning: Resume file not found: {resume_file}")
        return completed_problem_ids, existing_results
    
    try:
        with open(resume_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check if this file matches the strategy
        exp_config = data.get("experiment_config", {})
        file_strategy = exp_config.get("strategy")
        
        if file_strategy != strategy_name:
            print(f"  Warning: Resume file strategy '{file_strategy}' doesn't match '{strategy_name}'")
            return completed_problem_ids, existing_results
        
        results = data.get("results", [])
        for result in results:
            if "error" not in result and "problem_id" in result:
                problem_id = result["problem_id"]
                completed_problem_ids.add(problem_id)
                existing_results.append(result)
        
        print(f"  Resuming from: {resume_file}")
        print(f"  Found {len(completed_problem_ids)} completed problems")
        
    except Exception as e:
        print(f"  Warning: Failed to load resume file: {e}")
        completed_problem_ids = set()
        existing_results = []
    
    return completed_problem_ids, existing_results


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


async def run_aggregation_experiment(
    result_files: List[Path],
    output_dir: str = "aggregation_experiments",
    solutions_per_file: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    task_domain: str = "math",
    max_concurrent: Optional[int] = None,
    resume: bool = False,
    resume_from: Optional[str] = None,
    model_name: Optional[str] = None,
    pairwise_template_name: Optional[str] = None,
    provider: str = "litellm",
    tokenizer_path: Optional[str] = None,
    eval_model_name: Optional[str] = None,
    eval_api_base: Optional[str] = None,
    eval_provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Run aggregation experiment on result file(s).

    Args:
        result_files: List of paths to result JSON files (can be single file or multiple)
        output_dir: Output directory for results
        solutions_per_file: Optional override for number of solutions per file.
                           If None, uses config["reflection"]["n_samples_per_iteration"] from each file.
        api_key: API key for LLM service
        api_base: API base URL for LLM service
        task_domain: Task domain for template selection (default: "math" for imobench)
        eval_model_name: Model name for the evaluation/grading server (default: same as model_name)
        eval_api_base: API base URL for the evaluation server (default: same as api_base)
        eval_provider: Provider for the evaluation server (default: same as provider)

    Returns:
        Dictionary with experiment results
    """
    # Merge result files if multiple, or load single file
    if len(result_files) > 1:
        data = merge_result_files(result_files, solutions_per_file)
        file_solutions_counts = data.get("solutions_per_file", [])
        num_solutions = sum(file_solutions_counts)
    else:
        data = load_result_file(result_files[0])
        # Determine solutions per file for single file
        config_dict = data.get("config", {})
        reflection_config = config_dict.get("reflection", {})
        n_samples = reflection_config.get("n_samples_per_iteration")
        
        if n_samples is not None:
            num_solutions = n_samples
            print(f"Using {num_solutions} solutions (from config.reflection.n_samples_per_iteration)")
        if solutions_per_file is not None:
            num_solutions = solutions_per_file
            print(f"Using {num_solutions} solutions (from --solutions-per-file)")
        else:
            num_solutions = 4
            print(f"Using {num_solutions} solutions (default)")
        
        file_solutions_counts = [num_solutions]
    
    config_dict = data["config"]
    results = data["results"]
    benchmark = data["benchmark"]
    
    # Extract LLM config from result file
    llm_config_dict = config_dict.get("llm", {})
    model_name = model_name
    api_key = api_key or llm_config_dict.get("api_key") or os.getenv("OPENAI_API_KEY", "None")
    # llm config dict is result file is old, should changed to new api base
    api_base = os.getenv("OPENAI_API_BASE")
    llm_config_dict[api_base] = api_base
    
    if api_base is None:
        api_base = os.getenv("SGLANG_API_BASES")
    
    print(f"\nUsing model: {model_name}")
    if api_base:
        print(f"API Base: {api_base}")
    
    # Create LLM service
    extra_service_kwargs = {}
    if provider == "vllm_direct" and tokenizer_path:
        extra_service_kwargs["tokenizer_name"] = tokenizer_path
    llm_service = create_llm_service(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        api_base=api_base,
        **extra_service_kwargs,
    )
    
    # Create prompt manager
    prompt_manager = PromptManager()
    
    # Create evaluator (needed for evaluating aggregated solutions, not for gt_scoring)
    # Note: gt_scoring now uses stored evaluation results from result files, not re-evaluation
    # The evaluator can optionally point to a separate server (e.g. different model/GPUs)
    eval_provider_actual = eval_provider or provider
    eval_model_actual = eval_model_name or model_name
    eval_api_base_actual = eval_api_base or api_base
    if eval_model_actual != model_name or eval_api_base_actual != api_base:
        print(f"\nUsing separate evaluator service:")
        print(f"  Eval model: {eval_model_actual}")
        print(f"  Eval API base: {eval_api_base_actual}")
        print(f"  Eval provider: {eval_provider_actual}")
    evaluator_service = create_llm_service(
        provider=eval_provider_actual,
        model_name=eval_model_actual,
        api_key=api_key,
        api_base=eval_api_base_actual,
    )
    llm_judge = LLMJudge(evaluator_service, benchmark="imobench")
    evaluator = IMOBenchEvaluator(llm_judge=llm_judge)
    
    # Create aggregation strategies
    strategies = {}
    
    # 1. Generate from N
    # generate_template = prompt_manager.get_template("aggregation_generate_one_from_n")
    # strategies["generate_from_n"] = GenerateFromNAggregation(
    #     llm_service=llm_service,
    #     generation_prompt_template=generate_template,
    #     temperature=0.7,
    #     reasoning_effort="auto",
    # )
    
    # 2. Select best
    # select_template = prompt_manager.get_template("aggregation_select_one_from_n")
    # strategies["select_best"] = SelectBestAggregation(
    #     llm_service=llm_service,
    #     selection_prompt_template=select_template,
    #     temperature=0.0,
    #     reasoning_effort="auto",
    # )
    
    # 3. LLM Scoring
    # scoring_template = prompt_manager.get_template(f"{task_domain}_llm_scoring")
    # if scoring_template is None:
    #     # Fallback to math if not found
    #     scoring_template = prompt_manager.get_template("math_llm_scoring")
    # strategies["llm_scoring"] = LLMScoringAggregation(
    #     llm_service=llm_service,
    #     scoring_prompt_template=scoring_template,
    #     temperature=0.0,
    #     reasoning_effort="auto",
    # )
    
    # 4. GT Scoring (uses stored evaluation results from result files, implements pass@k)
    # Note: gt_scoring no longer re-evaluates, it directly uses is_correct from result files
    # strategies["gt_scoring"] = GTScoringAggregation(evaluator=evaluator)
    
    # 5. Voting
    # strategies["voting"] = VotingAggregation()
    
    # 6. Pairwise comparison
    template_name = pairwise_template_name
    if not template_name:
        # Backward-compatible default selection.
        lowered_model_name = (model_name or "").lower()
        if "newenc" in lowered_model_name or "new-enc" in lowered_model_name or "new_encoding" in lowered_model_name:
            template_name = "aggregation_pairwise_new_encoding_comparison"
        else:
            template_name = "aggregation_pairwise_comparison"

    print(f"Using pairwise template: {template_name}")
    pairwise_template = prompt_manager.get_template(template_name)
    if pairwise_template is None:
        raise ValueError(f"Prompt template not found: {template_name}")

    strategies["pairwise_comparison"] = PairwiseComparisonAggregation(
        llm_service=llm_service,
        comparison_prompt_template=pairwise_template,
        temperature=0.7,
        reasoning_effort="auto",
    )
    
    # 7. llm voting
    # pairwise_template = prompt_manager.get_template("aggregation_llm_voting")
    # strategies["llm_voting"] = LLMVotingAggregation(
    #     llm_service=llm_service,
    #     voting_prompt_template=pairwise_template,
    #     temperature=0.0,
    #     reasoning_effort="auto",
    # )
    # Prepare experiment config for saving
    experiment_config = {
        "num_result_files": len(result_files),
        "solutions_per_file": file_solutions_counts if len(result_files) > 1 else file_solutions_counts[0],
        "solutions_per_file_list": file_solutions_counts,  # Keep list for multiple files
        "num_solutions": num_solutions,
        "task_domain": task_domain,
        "model_name": model_name,
        "pairwise_template_name": template_name,
    }
    
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run each strategy separately on the entire benchmark
    print(f"\nRunning aggregation experiments on {len(results)} problems...")
    if len(result_files) > 1:
        print(f"Using {file_solutions_counts} solutions per file from {len(result_files)} files ({num_solutions} total per problem)")
    else:
        print(f"Using first {num_solutions} solutions per problem")
    print(f"Strategies: {list(strategies.keys())}")
    print(f"Each strategy will process all problems and save to separate files")
    print(f"Intermediate results will be saved incrementally")
    if resume:
        print(f"Resume mode: Auto-detect latest results for each strategy")
    elif resume_from:
        print(f"Resume mode: Using specified file: {resume_from}")
    
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
        if strategy_resume_file:
            completed_problem_ids, existing_results = _load_existing_results_for_strategy(
                strategy_resume_file, strategy_name
            )
            if completed_problem_ids:
                print(f"  Will skip {len(completed_problem_ids)} completed problems")
                print(f"  Running {len(results) - len(completed_problem_ids)} remaining problems")
        
        # Filter out completed problems
        remaining_results = [
            r for r in results
            if r.get("problem_id", "") not in completed_problem_ids
        ]
        
        if not remaining_results:
            print(f"\n  All problems already completed for strategy {strategy_name}!")
            print(f"  Using existing results from: {strategy_resume_file}")
            # Use existing results and create a new file with updated timestamp
            strategy_results = existing_results
        else:
            print(f"  Processing {len(remaining_results)} remaining problems")
            strategy_results = existing_results.copy()  # Start with existing results
        
        # Create output file for this strategy
        if strategy_resume_file and not remaining_results:
            # If resuming and all done, use the same file
            strategy_output_file = strategy_resume_file
        else:
            # Create new file with timestamp
            strategy_output_file = output_path / f"aggregation_experiment_{strategy_name}_{benchmark}_{timestamp}.json"
        
        # Lock for thread-safe incremental saving
        save_lock = threading.Lock()
        strategy_start_time = datetime.now()
        
        # Create results lock for this strategy
        results_lock = asyncio.Lock()
        global_results_locks[strategy_name] = results_lock
        
        # Store strategy info
        strategy_info[strategy_name] = {
            "strategy": strategy,
            "strategy_name": strategy_name,
            "strategy_output_file": strategy_output_file,
            "strategy_results": strategy_results,
            "save_lock": save_lock,
            "strategy_start_time": strategy_start_time,
            "remaining_results": remaining_results,
            "existing_results": existing_results,
            "results_lock": results_lock,  # Store lock in info for unified processing
        }
        
        # Create tasks for remaining problems
        if remaining_results:
            for idx, result_data in enumerate(remaining_results, 1):
                task_info = {
                    "strategy_name": strategy_name,
                    "result_data": result_data,
                    "idx": idx,
                    "total": len(remaining_results),
                }
                all_tasks.append(task_info)
    
    # Define unified processing function
    async def process_problem_unified(task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single problem with its assigned strategy."""
        strategy_name = task_info["strategy_name"]
        result_data = task_info["result_data"]
        idx = task_info["idx"]
        total = task_info["total"]
        
        info = strategy_info[strategy_name]
        strategy = info["strategy"]
        
        async with global_semaphore:
            problem_id = result_data.get("problem_id", f"problem_{idx}")
            problem = result_data.get("problem", "")
            ground_truth = result_data.get("ground_truth", "")
            
            try:
                # Extract solutions (already merged if multiple files)
                solutions = extract_first_n_solutions(result_data, num_solutions)
                
                # Skip if not enough solutions
                if len(solutions) < num_solutions:
                    print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Skipping (only {len(solutions)} solutions)", flush=True)
                    return None
                
                # Skip if all solutions are empty
                if all(not sol.content.strip() for sol in solutions):
                    print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: Skipping (all solutions empty)", flush=True)
                    return None
                
                # Aggregate solutions
                # Note: gt_scoring now uses stored evaluation results, ground_truth is not needed but kept for compatibility
                if strategy_name == "gt_scoring":
                    aggregated_solution = await strategy.aggregate(
                        problem=problem, solutions=solutions, ground_truth=ground_truth
                    )
                else:
                    aggregated_solution = await strategy.aggregate(problem=problem, solutions=solutions)
                
                # Check if aggregation was skipped due to token limit
                if aggregated_solution.metadata.get("skipped", False):
                    problem_result = {
                        "problem_id": problem_id,
                        "problem": problem,
                        "ground_truth": ground_truth,
                        "solution_content": aggregated_solution.content,
                        "skipped": True,
                        "skip_reason": aggregated_solution.metadata.get("skip_reason", "unknown"),
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
                        selected_index = aggregated_solution.metadata.get("original_index", 0)
                        for r in all_results:
                            if r.get("index") == selected_index:
                                is_correct = r.get("is_correct", False)
                                score = r.get("score", 0.0)
                                break
                    
                    problem_result = {
                        "problem_id": problem_id,
                        "problem": problem,
                        "ground_truth": ground_truth,
                        "solution_content": aggregated_solution.content,
                        "is_correct": is_correct,
                        "score": score,
                        "feedback": f"pass@k={pass_at_k}, {n_passed}/{len(solutions)} passed",
                        "metadata": aggregated_solution.metadata,
                    }
                    
                    print(f"  [{strategy_name}] [{idx}/{total}] {problem_id}: {'✓' if is_correct else '✗'} (pass@k={pass_at_k}, score: {score:.2f})", flush=True)
                else:
                    # Evaluate aggregated solution for other strategies
                    eval_result = await evaluator.evaluate(
                        problem=problem,
                        solution=aggregated_solution.content,
                        ground_truth=ground_truth,
                    )
                    
                    problem_result = {
                        "problem_id": problem_id,
                        "problem": problem,
                        "ground_truth": ground_truth,
                        "solution_content": aggregated_solution.content,
                        "is_correct": eval_result.is_correct,
                        "score": eval_result.score,
                        "feedback": eval_result.feedback,
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
            correct_count = sum(1 for r in valid_results if r.get("is_correct", False))
            total_count = len(valid_results)
            avg_score = sum(r.get("score", 0.0) for r in valid_results) / total_count if total_count > 0 else 0.0
            
            # Count skipped problems
            skipped_count = sum(1 for r in strategy_results if r.get("skipped", False))
            
            aggregate_metrics = {
                "pass@1": correct_count / total_count if total_count > 0 else 0.0,
                "total_problems": total_count,
                "correct_problems": correct_count,
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
            print(f"  Pass@1: {aggregate_metrics['pass@1']:.4f} ({aggregate_metrics['correct_problems']}/{aggregate_metrics['total_problems']})")
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
            print(f"    Pass@1: {metrics['pass@1']:.4f} ({metrics['correct_problems']}/{metrics['total_problems']})")
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
        description="Run aggregation experiments on existing result files"
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
        "--solutions-per-file",
        type=int,
        default=None,
        help="Optional override for number of solutions to extract from each file. "
             "If not provided, uses config['reflection']['n_samples_per_iteration'] from each file. "
             "If that's also not available, defaults to 4.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/qwen3-4b",
        help="Model to use for experiments (default: qwen3)",
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
        "--task-domain",
        type=str,
        default="math",
        help="Task domain for template selection (default: math for imobench)",
    )

    parser.add_argument(
        "--pairwise-template",
        type=str,
        default=None,
        help="Prompt template name for pairwise comparison (overrides automatic template selection)",
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
        "--provider",
        type=str,
        default="litellm",
        choices=["litellm", "openai", "vllm_direct", "vllm_raw"],
        help="LLM service provider (default: litellm). Use 'vllm_direct' for client-side chat template application. Use 'vllm_raw' for manual prompt formatting matching training data format.",
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer directory for vllm_direct provider. "
             "Required when using vllm_direct with custom/local models.",
    )

    parser.add_argument(
        "--eval-model-name",
        type=str,
        default=None,
        help="Model name for the evaluation/grading server. "
             "If not provided, uses the main --model_name.",
    )

    parser.add_argument(
        "--eval-api-base",
        type=str,
        default=None,
        help="API base URL for the evaluation/grading server. "
             "If not provided, uses the main api_base.",
    )

    parser.add_argument(
        "--eval-provider",
        type=str,
        default=None,
        choices=["litellm", "openai", "vllm_direct", "vllm_raw"],
        help="Provider for the evaluation/grading server. "
             "If not provided, uses the main --provider.",
    )

    args = parser.parse_args()
    
    # Validate all result files exist
    for result_file in args.result_files:
        if not result_file.exists():
            print(f"Error: Result file not found: {result_file}")
            return
    
    await run_aggregation_experiment(
        result_files=args.result_files,
        output_dir=args.output_dir,
        solutions_per_file=args.solutions_per_file,
        api_key=args.api_key,
        api_base=args.api_base,
        task_domain=args.task_domain,
        max_concurrent=args.max_concurrent,
        resume=args.resume,
        resume_from=args.resume_from,
        model_name=args.model_name,
        pairwise_template_name=args.pairwise_template,
        provider=args.provider,
        tokenizer_path=args.tokenizer_path,
        eval_model_name=args.eval_model_name,
        eval_api_base=args.eval_api_base,
        eval_provider=args.eval_provider,
    )


if __name__ == "__main__":
    asyncio.run(main())

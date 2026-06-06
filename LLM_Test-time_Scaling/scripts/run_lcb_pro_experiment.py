"""LiveCodeBench-Pro experiment script using ExperimentRunner for generating rollouts.

This script runs test-time scaling experiments on LiveCodeBench-Pro and generates rollouts
(all intermediate solutions) for analysis of pass@k metrics.

Usage:
    # Run all experiments from scratch
    python scripts/run_lcb_pro_experiment.py

    # Auto-resume from latest results for each experiment
    python scripts/run_lcb_pro_experiment.py --resume

    # Resume from specific results file (applies to all experiments)
    python scripts/run_lcb_pro_experiment.py --resume-from results/lcb_pro_rollouts/baseline_20240115_103000.json

    # Filter by specific splits
    python scripts/run_lcb_pro_experiment.py --splits biannual_2024_7_12 quater_2025_4_6

Resume Functionality:
    The script supports resuming interrupted experiments by skipping already completed problems:
    - Loads existing results file
    - Extracts problem IDs that were successfully completed
    - Skips those problems and only runs remaining ones
    - Merges new results with existing results
    - Saves combined results to new file

    Use --resume to automatically detect and resume from the latest results file for each experiment.
    Use --resume-from to specify a specific results file to resume from.
"""

import asyncio
import os
from pathlib import Path
from typing import List, Optional

from src.experiment_runner import ExperimentRunner
from src.utils.config import (
    AggregationConfig,
    Config,
    EvaluationConfig,
    LLMConfig,
    ReflectionConfig,
)


def create_experiment_config(
    experiment_name: str,
    model_name: str,
    reflection_strategy: str,
    aggregation_strategy: str,
    n_samples: int,
    n_iterations: int,
    apply_agg_each_turn: bool,
    output_dir: str,
    api_key: str = None,
    api_base: str = None,
    local_data_dir: str = None,
    language: str = "cpp",
    splits: Optional[List[str]] = None,
    max_concurrent_problems: int = 40,
    service_url: str = None,
    reflection_effort: str = "auto",
    aggregation_effort: str = "auto",
    initial_effort: str = "auto",
    use_detailed_results: bool = True,
) -> Config:
    """Create experiment configuration.

    Args:
        experiment_name: Name of the experiment
        model_name: Model name (e.g., "gpt-oss-120b")
        reflection_strategy: Reflection strategy name
        aggregation_strategy: Aggregation strategy name
        n_samples: Number of samples per iteration
        n_iterations: Number of iterations
        apply_agg_each_turn: Whether to aggregate at each turn
        output_dir: Output directory for results
        api_key: API key (defaults to OPENAI_API_KEY env var)
        api_base: API base URL (defaults to OPENAI_API_BASE env var)
        local_data_dir: Local data directory containing testcases (for LCB-Pro)
        language: Programming language for code evaluation (default: "cpp")
        splits: List of split names to filter problems (e.g., ["biannual_2024_7_12"])
        service_url: Remote code verify service URL(s), comma-separated (defaults to CODE_VERIFY_URL env var)

    Returns:
        Config object
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY", "None")
    api_base = api_base or os.getenv("OPENAI_API_BASE")
    eval_api_base = os.getenv("EVAL_OPENAI_API_BASE") or api_base
    service_url = service_url or os.getenv("CODE_VERIFY_URL")
   
    return Config(
        experiment_name=experiment_name,
        llm=LLMConfig(
            provider="litellm",
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=0.7,
            reasoning_effort=initial_effort,  # Use initial_effort for initial solution generation
        ),
        evaluation=EvaluationConfig(
            benchmark="lcb_pro",
            evaluator_type="code_executor",  # Use code executor for LCB-Pro
            provider="litellm",
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            local_data_dir=local_data_dir,
            language=language,
            splits=splits,
            service_url=service_url,
        ),
        reflection=ReflectionConfig(
            strategy=reflection_strategy,
            n_iterations=n_iterations,
            n_samples_per_iteration=n_samples,
            reasoning_effort=reflection_effort,
            use_detailed_results=use_detailed_results
        ),
        aggregation=AggregationConfig(
            strategy=aggregation_strategy,
            apply_at_each_turn=apply_agg_each_turn,
            reasoning_effort=aggregation_effort
        ),
        output_dir=output_dir,
        max_concurrent_problems=max_concurrent_problems,
    )


async def run_baseline(
    model_name: str,
    output_dir: str,
    api_key: str = None,
    api_base: str = None,
    local_data_dir: str = None,
    language: str = "cpp",
    resume_from: str = None,
    splits: Optional[List[str]] = None,
    max_concurrent_problems: int = 40,
    service_url: str = None,
    n_samples: int = 8,
    experiment_name: str = "direct_generation",
) -> None:
    """Run baseline experiment (direct generation, no reflection/aggregation).

    Args:
        model_name: Model name
        output_dir: Output directory
        api_key: API key
        api_base: API base URL
        local_data_dir: Local data directory containing testcases
        language: Programming language for code evaluation
        resume_from: Path to existing results file to resume from
        splits: List of split names to filter problems
        max_concurrent_problems: Maximum number of problems to process concurrently
        service_url: Remote code verify service URL(s)
        n_samples: Number of parallel solutions to generate (default: 8)
        experiment_name: Name of the experiment (default: "direct_generation")
    """
    print("\n" + "=" * 80)
    print(f"DIRECT GENERATION EXPERIMENT: {experiment_name}")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"N Samples: {n_samples}")
    print(f"Reflection: none")
    print(f"Aggregation: none")
    print("=" * 80)

    config = create_experiment_config(
        experiment_name=experiment_name,
        model_name=model_name,
        reflection_strategy="none",
        aggregation_strategy="none",
        n_samples=n_samples,
        n_iterations=0,
        apply_agg_each_turn=False,
        output_dir=output_dir,
        api_key=api_key,
        api_base=api_base,
        local_data_dir=local_data_dir,
        language=language,
        splits=splits,
        max_concurrent_problems=max_concurrent_problems,
        service_url=service_url,
    )

    runner = ExperimentRunner(config, resume_from=resume_from)
    await runner.run()


async def run_scaling_experiment(
    experiment_name: str,
    model_name: str,
    reflection_strategy: str,
    aggregation_strategy: str,
    n_samples: int,
    n_iterations: int,
    apply_agg_each_turn: bool,
    output_dir: str,
    api_key: str = None,
    api_base: str = None,
    local_data_dir: str = None,
    language: str = "cpp",
    resume_from: str = None,
    splits: Optional[List[str]] = None,
    max_concurrent_problems: int = 40,
    service_url: str = None,
    reflection_effort: str = "auto",
    aggregation_effort: str = "auto",
    initial_effort: str = "auto",
    use_detailed_results: bool = True,
) -> None:
    """Run a test-time scaling experiment.

    Args:
        experiment_name: Name of the experiment
        model_name: Model name
        reflection_strategy: Reflection strategy
        aggregation_strategy: Aggregation strategy
        n_samples: Number of samples per iteration
        n_iterations: Number of iterations
        apply_agg_each_turn: Whether to aggregate at each turn
        output_dir: Output directory
        api_key: API key
        api_base: API base URL
        local_data_dir: Local data directory containing testcases
        language: Programming language for code evaluation
        resume_from: Path to existing results file to resume from
        splits: List of split names to filter problems
        max_concurrent_problems: Maximum number of problems to process concurrently
        service_url: Remote code verify service URL(s)
    """
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {experiment_name}")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Reflection: {reflection_strategy}")
    print(f"Aggregation: {aggregation_strategy}")
    print(f"Samples: {n_samples}")
    print(f"Iterations: {n_iterations}")
    print(f"Architecture: {'agg_each_turn' if apply_agg_each_turn else 'reflect_then_agg'}")
    print(f"Initial Effort: {initial_effort}")
    print(f"Refine Effort: {reflection_effort}")
    if reflection_strategy == "code_execution":
        print(f"Use Detailed Results: {use_detailed_results}")
    if splits:
        print(f"Splits: {splits}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    if service_url:
        print(f"Remote Service URL: {service_url}")
    print("=" * 80)

    config = create_experiment_config(
        experiment_name=experiment_name,
        model_name=model_name,
        reflection_strategy=reflection_strategy,
        aggregation_strategy=aggregation_strategy,
        n_samples=n_samples,
        n_iterations=n_iterations,
        apply_agg_each_turn=apply_agg_each_turn,
        output_dir=output_dir,
        api_key=api_key,
        api_base=api_base,
        local_data_dir=local_data_dir,
        language=language,
        splits=splits,
        max_concurrent_problems=max_concurrent_problems,
        service_url=service_url,
        reflection_effort=reflection_effort,
        aggregation_effort=aggregation_effort,
        initial_effort=initial_effort,
        use_detailed_results=use_detailed_results,
    )

    runner = ExperimentRunner(config, resume_from=resume_from)
    await runner.run()


async def main():
    """Main function to run all LiveCodeBench-Pro experiments."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run LiveCodeBench-Pro experiments with test-time scaling"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from latest results files for each experiment",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from specific results file (applies to all experiments)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        help="List of split names to filter problems (e.g., biannual_2024_7_12 quater_2025_4_6)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-120b",
        choices=["gpt-oss-120b", "gpt-oss-20b", "qwen3-235b", "qwen3-30b", "qwen3-4b", "qwen3-4b-newenc", "qwen3-30b-a3b", "qwen3-30b-a3b-newenc"],
        help="Model to use for experiments (default: gpt-oss-120b)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=128,
        help="Maximum number of problems to process concurrently (default: 40)",
    )
    args = parser.parse_args()

    # ========== Configuration ==========
    MODEL_PATHS = {
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "qwen3-235b": "openai/Qwen__Qwen3-235B-A22B",
        "qwen3-30b": "openai/Qwen__Qwen3-30B-A3B",
        "qwen3-30b-a3b":"openai/qwen3-30b-a3b",
        "qwen3-4b": "openai/qwen3-4b",
        "qwen3-4b-newenc": "openai/qwen3-4b-newenc",
        "qwen3-30b-a3b-newenc": "openai/qwen3-30b-a3b-newenc"
    }
    MODEL = MODEL_PATHS[args.model]
    OUTPUT_DIR = f"results/lcb_generation_4seed/lcb_pro_rollouts_{args.model}"
    API_KEY = os.getenv("OPENAI_API_KEY")
    API_BASE = os.getenv("OPENAI_API_BASE")  # Optional: Custom API endpoint
    LOCAL_DATA_DIR = os.getenv("LCB_PRO_DATA_DIR", "path-to-results/llm_test_time_scaling/data/local_data/lcb_testcases/data")  # Local data directory
    LANGUAGE = "cpp"  # Programming language for LCB-Pro (C++)
    SPLITS = args.splits  # List of split names to filter problems (from command line or None for all)
    SERVICE_URL = os.getenv("CODE_VERIFY_URL")  # Optional: Remote code verify service URL(s), comma-separated

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ========== Experiment Definitions ==========
    experiments = [
        # Direct generation: generate N solutions in parallel, no turn-wise scaling
        {
            "type": "baseline",
            "name": "direct_generation",
            "n_samples": 8,
        },
        # {
        #     "name": "code_execution_detailed_sequential",
        #     "reflection": "code_execution",
        #     "aggregation": "none",
        #     "n_samples": 3,
        #     "n_iterations": 8,
        #     "agg_each_turn": False,
        #     "use_detailed_results": True,  # Use detailed results (with input/output/answer)
        # },
        # {
        #     "name": "code_execution_sparse_sequential",
        #     "reflection": "code_execution",
        #     "aggregation": "none",
        #     "n_samples": 3,
        #     "n_iterations": 8,
        #     "agg_each_turn": False,
        #     "use_detailed_results": False,  # Use basic results (only passed/total, error type, and checker stdout)
        # },
    ]

    # ========== Run Experiments ==========
    print("\n" + "=" * 80)
    print("LIVECODEBENCH-PRO ROLLOUT GENERATION")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Local Data Directory: {LOCAL_DATA_DIR}")
    print(f"Language: {LANGUAGE}")
    print(f"API Base: {API_BASE or 'Default'}")
    print(f"Max Concurrent Problems: {args.max_concurrent}")
    if SERVICE_URL:
        print(f"Remote Service URL: {SERVICE_URL}")
    else:
        print("Remote Service: Not set (using local evaluator)")
    if SPLITS:
        print(f"Splits: {SPLITS}")
    else:
        print("Splits: All (no filtering)")
    print(f"Total Experiments: {len(experiments)}")
    if args.resume:
        print("Mode: Auto-resume from latest results")
    elif args.resume_from:
        print(f"Mode: Resume from {args.resume_from}")
    print("=" * 80)

    for idx, exp_config in enumerate(experiments, 1):
        print(f"\n[{idx}/{len(experiments)}] Starting experiment...")

        # Determine resume file for this experiment
        resume_from = args.resume_from

        if args.resume and not resume_from:
            # Auto-detect latest results file for this experiment
            exp_name = exp_config.get("name", exp_config.get("type", "unknown"))
            output_path = Path(OUTPUT_DIR)
            if output_path.exists():
                pattern = f"{exp_name}_lcb_pro_*.json"
                result_files = sorted(
                    output_path.glob(pattern), key=lambda p: p.stat().st_mtime
                )
                if result_files:
                    resume_from = str(result_files[-1])
                    print(f"  Auto-detected resume file: {resume_from}")

        try:
            if exp_config.get("type") == "baseline":
                await run_baseline(
                    MODEL, OUTPUT_DIR, API_KEY, API_BASE,
                    local_data_dir=LOCAL_DATA_DIR,
                    language=LANGUAGE,
                    resume_from=resume_from,
                    splits=SPLITS,
                    max_concurrent_problems=args.max_concurrent,
                    service_url=SERVICE_URL,
                    n_samples=exp_config.get("n_samples", 8),
                    experiment_name=exp_config.get("name", "direct_generation"),
                )
            else:
                await run_scaling_experiment(
                    experiment_name=exp_config["name"],
                    model_name=MODEL,
                    reflection_strategy=exp_config["reflection"],
                    aggregation_strategy=exp_config["aggregation"],
                    n_samples=exp_config["n_samples"],
                    n_iterations=exp_config["n_iterations"],
                    apply_agg_each_turn=exp_config["agg_each_turn"],
                    output_dir=OUTPUT_DIR,
                    api_key=API_KEY,
                    api_base=API_BASE,
                    local_data_dir=LOCAL_DATA_DIR,
                    language=LANGUAGE,
                    resume_from=resume_from,
                    splits=SPLITS,
                    max_concurrent_problems=args.max_concurrent,
                    service_url=SERVICE_URL,
                    reflection_effort=exp_config.get("refine_effort", "auto"),  # refine_effort maps to reflection_effort
                    aggregation_effort=exp_config.get("aggregation_effort", "auto"),
                    initial_effort=exp_config.get("initial_effort", "auto"),
                    use_detailed_results=exp_config.get("use_detailed_results", True),
                )

        except Exception as e:
            print(f"\n❌ Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Small delay between experiments
        await asyncio.sleep(2)

    # ========== Final Summary ==========
    print("\n" + "=" * 80)
    print("✅ ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"Results saved in: {OUTPUT_DIR}")
    print("\nGenerated rollouts include:")
    print("  • All intermediate solutions (n_samples × n_iterations)")
    print("  • Evaluation results for each solution")
    print("  • Pass@1 and Pass@k metrics")
    print("  • Token usage and timing information")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())


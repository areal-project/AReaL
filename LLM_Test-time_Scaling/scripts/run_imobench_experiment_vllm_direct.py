"""IMOBench experiment script using ExperimentRunner for generating rollouts.

This script runs test-time scaling experiments on IMOBench and generates rollouts
(all intermediate solutions) for analysis of pass@k metrics.

Usage:
    # Run all experiments from scratch (default: gpt-oss-120b)
    python scripts/run_imobench_experiment_vllm_direct.py

    # Run with specific model
    python scripts/run_imobench_experiment_vllm_direct.py --model qwen3-235b

    # Control concurrency (default: 128)
    python scripts/run_imobench_experiment_vllm_direct.py --model gpt-oss-20b --max-concurrent 64

    # Auto-resume from latest results for each experiment
    python scripts/run_imobench_experiment_vllm_direct.py --resume --model gpt-oss-20b

    # Resume from specific results file (applies to all experiments)
    python scripts/run_imobench_experiment_vllm_direct.py --resume-from results/imobench_rollouts_gpt-oss-120b/baseline_20240115_103000.json

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
from typing import Optional

from src.experiment_runner import ExperimentRunner
from src.utils.config import (
    AggregationConfig,
    Config,
    EvaluationConfig,
    LLMConfig,
    ReflectionConfig,
)


MODEL_PATHS = {
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "qwen3-235b": "openai/Qwen__Qwen3-235B-A22B",
    "qwen3-30b": "openai/Qwen__Qwen3-30B-A3B",
    "qwen3-30b-a3b": "openai/qwen3-30b-a3b",
    "qwen3-8b": "openai/qwen3-8b",
    "qwen3-4b": "openai/qwen3-4b",
    "qwen3-4b-newenc": "openai/qwen3-4b-newenc",
    "qwen3-30b-a3b-newenc": "openai/qwen3-30b-a3b-newenc",
}


def _sanitize_name(raw: str) -> str:
    """Sanitize arbitrary names for benchmark/result filenames."""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw)


def create_experiment_config(
    experiment_name: str,
    model_name: str,
    benchmark_name: str,
    reflection_strategy: str,
    aggregation_strategy: str,
    n_samples: int,
    n_iterations: int,
    apply_agg_each_turn: bool,
    output_dir: str,
    api_key: str = None,
    api_base: str = None,
    max_concurrent_problems: int = 128,
    reasoning_effort: Optional[str] = None,
    initial_effort: Optional[str] = None,
    benchmark_file: Optional[str] = None,
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
        max_concurrent_problems: Maximum number of problems to process concurrently

    Returns:
        Config object
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY", "None")
    api_base = api_base or os.getenv("OPENAI_API_BASE")
    eval_api_base = os.getenv("EVAL_OPENAI_API_BASE") or api_base

    return Config(
        experiment_name=experiment_name,
        llm=LLMConfig(
            provider="vllm_direct",
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            direct_prompt_template="math_direct_qwen_simple",
            reasoning_effort=initial_effort,  # Use initial_effort for initial solution generation
        ),
        evaluation=EvaluationConfig(
            benchmark=benchmark_name,
            evaluator_type="imobench_boxed",
            provider="vllm_direct",
            model_name=model_name,
            api_key=api_key,
            api_base=eval_api_base,
            benchmark_file=benchmark_file,
        ),
        reflection=ReflectionConfig(
            strategy=reflection_strategy,
            n_iterations=n_iterations,
            n_samples_per_iteration=n_samples,
            reasoning_effort=reasoning_effort  # Use reasoning_effort (refine_effort) for reflection
        ),
        aggregation=AggregationConfig(
            strategy=aggregation_strategy,
            apply_at_each_turn=apply_agg_each_turn,
            reasoning_effort=reasoning_effort
        ),
        output_dir=output_dir,
        max_concurrent_problems=max_concurrent_problems,
    )


async def run_baseline(
    model_name: str,
    benchmark_name: str,
    benchmark_file: Optional[str],
    output_dir: str,
    api_key: str = None,
    api_base: str = None,
    resume_from: str = None,
    max_concurrent_problems: int = 128,
    reasoning_effort: Optional[str] = None,
    initial_effort: Optional[str] = None,
    n_samples: int = 8,
    experiment_name: str = "direct_generation",
) -> None:
    """Run baseline experiment (direct generation, no reflection/aggregation).

    Args:
        model_name: Model name
        output_dir: Output directory
        api_key: API key
        api_base: API base URL
        resume_from: Path to existing results file to resume from
        max_concurrent_problems: Maximum number of problems to process concurrently
        reasoning_effort: Reasoning effort for reflection
        initial_effort: Reasoning effort for initial generation
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
        benchmark_name=benchmark_name,
        reflection_strategy="none",
        aggregation_strategy="none",
        n_samples=n_samples,
        n_iterations=0,
        apply_agg_each_turn=False,
        output_dir=output_dir,
        api_key=api_key,
        api_base=api_base,
        max_concurrent_problems=max_concurrent_problems,
        reasoning_effort=reasoning_effort,
        initial_effort=initial_effort,
        benchmark_file=benchmark_file,
    )

    runner = ExperimentRunner(config, resume_from=resume_from)
    await runner.run()


async def run_scaling_experiment(
    experiment_name: str,
    model_name: str,
    benchmark_name: str,
    benchmark_file: Optional[str],
    reflection_strategy: str,
    aggregation_strategy: str,
    n_samples: int,
    n_iterations: int,
    apply_agg_each_turn: bool,
    output_dir: str,
    api_key: str = None,
    api_base: str = None,
    resume_from: str = None,
    max_concurrent_problems: int = 128,
    reasoning_effort: Optional[str] = None,
    initial_effort: Optional[str] = None,
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
        resume_from: Path to existing results file to resume from
        max_concurrent_problems: Maximum number of problems to process concurrently
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
    print(f"Refine Effort: {reasoning_effort}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print("=" * 80)

    config = create_experiment_config(
        experiment_name=experiment_name,
        model_name=model_name,
        benchmark_name=benchmark_name,
        reflection_strategy=reflection_strategy,
        aggregation_strategy=aggregation_strategy,
        n_samples=n_samples,
        n_iterations=n_iterations,
        apply_agg_each_turn=apply_agg_each_turn,
        output_dir=output_dir,
        api_key=api_key,
        api_base=api_base,
        max_concurrent_problems=max_concurrent_problems,
        reasoning_effort=reasoning_effort,
        initial_effort=initial_effort,
        benchmark_file=benchmark_file,
    )

    runner = ExperimentRunner(config, resume_from=resume_from)
    await runner.run()


async def main():
    """Main function to run all IMOBench experiments."""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run IMOBench experiments with test-time scaling"
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
        "--model",
        type=str,
        default="gpt-oss-120b",
        choices=sorted(MODEL_PATHS.keys()),
        help="Predefined model alias (default: gpt-oss-120b)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Custom model alias to use in output naming (must be used with --model-path)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Custom full model path (must be used with --model-name)",
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=None,
        help=(
            "Optional list of benchmark JSON files to process. "
            "Each file will be parsed with IMOBench-compatible schema. "
            "If omitted, uses default imobench benchmark from benchmark data directory."
        ),
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=128,
        help="Maximum number of problems to process concurrently (default: 128)",
    )
    args = parser.parse_args()

    # ========== Configuration ==========
    if (args.model_name is None) != (args.model_path is None):
        parser.error("--model-name and --model-path must be provided together")

    if args.model_name and args.model_path:
        model_alias = _sanitize_name(args.model_name)
        model_path = args.model_path
    else:
        model_alias = args.model
        model_path = MODEL_PATHS[args.model]

    API_KEY = os.getenv("OPENAI_API_KEY")
    API_BASE = os.getenv("OPENAI_API_BASE")  # Optional: Custom API endpoint

    benchmark_targets = []
    if args.input_files:
        for file_path in args.input_files:
            resolved = Path(file_path).expanduser().resolve()
            if not resolved.exists():
                parser.error(f"Input file not found: {resolved}")
            benchmark_targets.append(resolved)
    else:
        benchmark_targets.append(None)

    # ========== Experiment Definitions ==========
    experiments = [
        # Direct generation: generate N solutions in parallel, no turn-wise scaling
        {
            "type": "baseline",
            "name": "direct_generation",
            "n_samples": 8,
        },

        # Self-Evaluation: Test different effort combinations for initial solution and refinement
        # Test with iteration 1 to see the effect of different effort combinations

        # Self-Evaluation: Generate → Evaluate → Refine (Sequential)
        # {
        #     "name": "self_eval_sequential_2*32",
        #     "reflection": "self_evaluation",
        #     "aggregation": "none",
        #     "n_samples": 2,
        #     "n_iterations": 32,
        #     "agg_each_turn": False,
        # },
        # {
        #     "name": "self_eval_sequential_2*16",
        #     "reflection": "self_evaluation",
        #     "aggregation": "none",
        #     "n_samples": 2,
        #     "n_iterations": 16,
        #     "agg_each_turn": False,
        # },
        # {
        #     "name": "self_eval_sequential_4*4",
        #     "reflection": "self_evaluation",
        #     "aggregation": "none",
        #     "n_samples": 4,
        #     "n_iterations": 4,
        #     "agg_each_turn": False,
        # },
        # {
        #     "name": "self_eval_sequential_16*2",
        #     "reflection": "self_evaluation",
        #     "aggregation": "none",
        #     "n_samples": 16,
        #     "n_iterations": 2,
        #     "agg_each_turn": False,
        # },

        # Self-Evaluation: Generate → Evaluate → Refine (Sequential)
        # {
        #     "name": "self_eval_sequential_trial2",
        #     "reflection": "self_evaluation",
        #     "aggregation": "none",
        #     "n_samples": 1,
        #     "n_iterations": 8,
        #     "agg_each_turn": False,
        # },


        # No-Feedback: Generate → Refine (Sequential)
        # {
        #     "name": "no_feedback_sequential_2*8",
        #     "reflection": "no_feedback",
        #     "aggregation": "none",
        #     "n_samples": 2,
        #     "n_iterations": 8,
        #     "agg_each_turn": False,
        # },

        # # Ground-truth: Generate → GT Evaluate -> Refine (Sequential)
        # {
        #     "name": "ground_truth_correctness_sequential_simple_feedback",
        #     "reflection": "ground_truth_simple",
        #     "aggregation": "none",
        #     "n_samples": 4,
        #     "n_iterations": 8,
        #     "agg_each_turn": False,
        # },
    ]

    # ========== Run Experiments ==========
    print("\n" + "=" * 80)
    print("IMOBENCH-COMPATIBLE ROLLOUT GENERATION")
    print("=" * 80)
    print(f"Model Alias: {model_alias}")
    print(f"Model Path: {model_path}")
    print(f"API Base: {API_BASE or 'Default'}")
    print(f"Max Concurrent Problems: {args.max_concurrent}")
    print(f"Total Experiments: {len(experiments)}")
    print(f"Total Benchmark Files: {len(benchmark_targets)}")
    if args.resume:
        print("Mode: Auto-resume from latest results")
    elif args.resume_from:
        print(f"Mode: Resume from {args.resume_from}")
    print("=" * 80)

    completed_targets = []

    for target_idx, benchmark_file in enumerate(benchmark_targets, 1):
        if benchmark_file is None:
            benchmark_name = "imobench"
            benchmark_label = "default_imobench"
            benchmark_file_str = None
        else:
            benchmark_label = _sanitize_name(benchmark_file.stem)
            benchmark_name = f"imobench_{benchmark_label}"
            benchmark_file_str = str(benchmark_file)

        output_dir = (
            f"results/test_time_compute/{benchmark_name}_rollouts_{model_alias}"
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        print("\n" + "-" * 80)
        print(f"[{target_idx}/{len(benchmark_targets)}] Benchmark target: {benchmark_label}")
        if benchmark_file_str:
            print(f"Input File: {benchmark_file_str}")
        else:
            print("Input File: default benchmark data directory (imobench.json)")
        print(f"Output Directory: {output_dir}")
        print("-" * 80)

        for idx, exp_config in enumerate(experiments, 1):
            print(f"\n[{idx}/{len(experiments)}] Starting experiment...")

            # Determine resume file for this experiment
            resume_from = args.resume_from

            if args.resume and not resume_from:
                exp_name = exp_config.get("name", exp_config.get("type", "unknown"))
                output_path = Path(output_dir)
                if output_path.exists():
                    pattern = f"{exp_name}_{benchmark_name}_*.json"
                    result_files = sorted(
                        output_path.glob(pattern), key=lambda p: p.stat().st_mtime
                    )
                    if result_files:
                        resume_from = str(result_files[-1])
                        print(f"  Auto-detected resume file: {resume_from}")

            try:
                if exp_config.get("type") == "baseline":
                    await run_baseline(
                        model_name=model_path,
                        benchmark_name=benchmark_name,
                        benchmark_file=benchmark_file_str,
                        output_dir=output_dir,
                        api_key=API_KEY,
                        api_base=API_BASE,
                        resume_from=resume_from,
                        max_concurrent_problems=args.max_concurrent,
                        reasoning_effort=exp_config.get("reasoning_effort"),
                        initial_effort=exp_config.get("initial_effort"),
                        n_samples=exp_config.get("n_samples", 8),
                        experiment_name=exp_config.get("name", "direct_generation"),
                    )
                else:
                    await run_scaling_experiment(
                        experiment_name=exp_config["name"],
                        model_name=model_path,
                        benchmark_name=benchmark_name,
                        benchmark_file=benchmark_file_str,
                        reflection_strategy=exp_config["reflection"],
                        aggregation_strategy=exp_config["aggregation"],
                        n_samples=exp_config["n_samples"],
                        n_iterations=exp_config["n_iterations"],
                        apply_agg_each_turn=exp_config["agg_each_turn"],
                        output_dir=output_dir,
                        api_key=API_KEY,
                        api_base=API_BASE,
                        resume_from=resume_from,
                        max_concurrent_problems=args.max_concurrent,
                        reasoning_effort=exp_config.get(
                            "refine_effort", exp_config.get("reasoning_effort")
                        ),
                        initial_effort=exp_config.get("initial_effort"),
                    )

            except Exception as e:
                print(f"\n❌ Experiment failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            await asyncio.sleep(2)

        completed_targets.append({
            "benchmark_name": benchmark_name,
            "output_dir": output_dir,
        })

    # ========== Final Summary ==========
    print("\n" + "=" * 80)
    print("✅ ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print("Results saved in:")
    for item in completed_targets:
        print(f"  • {item['benchmark_name']}: {item['output_dir']}")
    print("\nGenerated rollouts include:")
    print("  • All intermediate solutions (n_samples × n_iterations)")
    print("  • Evaluation results for each solution")
    print("  • Pass@1 and Pass@k metrics")
    print("  • Token usage and timing information")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

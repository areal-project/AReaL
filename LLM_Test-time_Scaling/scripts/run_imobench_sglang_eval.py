"""Standalone IMOBench evaluator using direct SGLang calls.

This script intentionally does NOT use ExperimentRunner or the test-time scaling
pipeline. It directly:
1) Loads IMOBench-format JSON problems.
2) Formats prompts with a chat template.
3) Sends requests to an OpenAI-compatible SGLang backend.
4) Evaluates each problem with 8 independent samples.
5) Saves results in the same top-level JSON format used by run_imobench_experiment.py,
   with benchmark and model names in the output path.

Usage:
  python scripts/run_imobench_sglang_eval.py \
      --input-file imobench.json \
      --model-path openai/Qwen__Qwen3-30B-A3B

  python scripts/run_imobench_sglang_eval.py \
      --input-file imobench.json \
      --model-name qwen3-30b \
      --model-path openai/Qwen__Qwen3-30B-A3B \
      --n-samples 8 \
      --max-concurrent 16
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from src.evaluation.boxed_imobench_evaluator import BoxedIMOBenchEvaluator
from src.utils.config import (
    AggregationConfig,
    Config,
    EvaluationConfig,
    LLMConfig,
    ReflectionConfig,
)


DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


@dataclass
class Problem:
    id: str
    problem: str
    ground_truth: str
    difficulty: Optional[str]
    metadata: Dict[str, Any]


def _sanitize_name(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw)


def _resolve_model_alias(model_name: Optional[str], model_path: str) -> str:
    if model_name and model_name.strip():
        alias = _sanitize_name(model_name.strip())
        if alias:
            return alias

    normalized_path = model_path.strip().rstrip("/")
    tail = normalized_path.split("/")[-1] if normalized_path else ""
    alias = _sanitize_name(tail)
    if alias:
        return alias

    fallback = _sanitize_name(normalized_path)
    if fallback:
        return fallback

    return "unknown_model"


def _load_imobench_problems(input_file: Path) -> List[Problem]:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "problems" not in data:
        raise ValueError("Input JSON must be an object with a 'problems' field")

    problems: List[Problem] = []
    for item in data["problems"]:
        if not isinstance(item, dict):
            continue
        problems.append(
            Problem(
                id=str(item.get("id", "")),
                problem=str(item.get("problem", "")),
                ground_truth=str(item.get("ground_truth", "")),
                difficulty=item.get("difficulty"),
                metadata=item.get("metadata") or {},
            )
        )

    return problems


def _resolve_tokenizer_name(model_path: str, tokenizer_name: Optional[str]) -> str:
    if tokenizer_name:
        return tokenizer_name
    if model_path.startswith("openai/"):
        return model_path[len("openai/") :]
    return model_path


def _apply_chat_template(
    tokenizer: Optional[Any],
    system_prompt: str,
    user_problem: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_problem},
    ]

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    # Fallback prompt if tokenizer/template is unavailable.
    return f"SYSTEM: {system_prompt}\n\nUSER: {user_problem}\n\nASSISTANT:"


async def _generate_one(
    client: AsyncOpenAI,
    model_path: str,
    api_base: str,
    prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
) -> Dict[str, Any]:
    start = time.time()
    response = await client.completions.create(
        model=model_path,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={"top_k": top_k},
    )
    elapsed = time.time() - start

    text = response.choices[0].text or ""
    usage = {
        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        "total_tokens": response.usage.total_tokens if response.usage else 0,
        "llm_call_time_sec": elapsed,
        "model": model_path,
        "api_base": api_base,
    }
    return {"content": text, "usage": usage}


async def _evaluate_problem(
    problem: Problem,
    client: AsyncOpenAI,
    tokenizer: Optional[Any],
    model_path: str,
    api_base: str,
    n_samples: int,
    sample_concurrency: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
) -> Dict[str, Any]:
    evaluator = BoxedIMOBenchEvaluator()

    prompt = _apply_chat_template(
        tokenizer=tokenizer,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        user_problem=problem.problem,
    )

    all_solutions = []
    total_tokens = 0
    total_llm_time = 0.0
    total_eval_time = 0.0

    sample_semaphore = asyncio.Semaphore(max(1, min(sample_concurrency, n_samples)))

    async def _run_single_sample(sample_idx: int) -> Dict[str, Any]:
        async with sample_semaphore:
            generation = await _generate_one(
                client=client,
                model_path=model_path,
                api_base=api_base,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )
            solution_text = generation["content"]
            usage = generation["usage"]

            eval_start = time.time()
            eval_result = await evaluator.evaluate(
                problem=problem.problem,
                solution=solution_text,
                ground_truth=problem.ground_truth,
            )
            eval_elapsed = time.time() - eval_start

            return {
                "solution": {
                    "solution_content": solution_text,
                    "is_correct": bool(eval_result.is_correct),
                    "score": float(eval_result.score),
                    "feedback": eval_result.feedback,
                    "metadata": {
                        "iteration": 0,
                        "sample_index": sample_idx,
                        "usage": usage,
                    },
                },
                "eval_time": eval_elapsed,
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
                "llm_time": float(usage.get("llm_call_time_sec", 0.0) or 0.0),
            }

    sample_tasks = [_run_single_sample(sample_idx) for sample_idx in range(n_samples)]
    sample_outputs = await asyncio.gather(*sample_tasks)

    for item in sample_outputs:
        all_solutions.append(item["solution"])
        total_eval_time += item["eval_time"]
        total_tokens += item["total_tokens"]
        total_llm_time += item["llm_time"]

    final_solution = all_solutions[0]
    correct_solutions = [s for s in all_solutions if s["is_correct"]]
    pass_at_1 = 1 if final_solution["is_correct"] else 0
    pass_at_k = 1 if correct_solutions else 0
    avg_accuracy = len(correct_solutions) / n_samples if n_samples > 0 else 0.0

    total_problem_time = total_llm_time + total_eval_time
    llm_time_percentage = (total_llm_time / total_problem_time * 100.0) if total_problem_time > 0 else 0.0
    eval_time_percentage = (total_eval_time / total_problem_time * 100.0) if total_problem_time > 0 else 0.0

    return {
        "problem_id": problem.id,
        "problem": problem.problem,
        "ground_truth": problem.ground_truth,
        "difficulty": problem.difficulty or problem.metadata.get("difficulty", "unknown"),
        "final_solution": {
            "content": final_solution["solution_content"],
            "is_correct": final_solution["is_correct"],
            "score": final_solution["score"],
            "feedback": final_solution["feedback"],
            "metadata": final_solution["metadata"],
            "code": "",
        },
        "all_solutions": all_solutions,
        "metrics": {
            "pass@1": pass_at_1,
            "pass@k": pass_at_k,
            "num_correct": len(correct_solutions),
            "num_total": n_samples,
            "final_score": final_solution["score"],
            "best_score_all": 1.0 if correct_solutions else 0.0,
            "avg_accuracy_8": avg_accuracy,
        },
        "total_tokens": total_tokens,
        "iterations": [],
        "scaling_metadata": {
            "mode": "direct_sglang_eval",
            "n_samples": n_samples,
            "test_time_scaling": False,
        },
        "timing": {
            "total_time_sec": total_problem_time,
            "llm_time_sec": total_llm_time,
            "evaluation_time_sec": total_eval_time,
            "llm_time_percentage": llm_time_percentage,
            "evaluation_time_percentage": eval_time_percentage,
        },
    }


def _compute_experiment_metrics(results: List[Dict[str, Any]], start_time: datetime, end_time: datetime) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "total_problems": 0,
            "successful_problems": 0,
            "failed_problems": 0,
            "duration_seconds": (end_time - start_time).total_seconds(),
        }

    successful_results = [r for r in results if "error" not in r]
    num_successful = len(successful_results)

    pass_at_1 = sum(r["metrics"]["pass@1"] for r in successful_results) / num_successful
    pass_at_k = sum(r["metrics"]["pass@k"] for r in successful_results) / num_successful
    avg_final_score = sum(r["final_solution"]["score"] for r in successful_results) / num_successful
    avg_best_score = sum(r["metrics"]["best_score_all"] for r in successful_results) / num_successful
    total_solutions = sum(r["metrics"]["num_total"] for r in successful_results)
    total_correct = sum(r["metrics"]["num_correct"] for r in successful_results)
    total_tokens = sum(r.get("total_tokens", 0) for r in successful_results)

    total_llm_time = sum(r.get("timing", {}).get("llm_time_sec", 0.0) for r in successful_results)
    total_eval_time = sum(r.get("timing", {}).get("evaluation_time_sec", 0.0) for r in successful_results)
    total_problem_time = sum(r.get("timing", {}).get("total_time_sec", 0.0) for r in successful_results)

    avg_llm_time = total_llm_time / num_successful if num_successful > 0 else 0.0
    avg_eval_time = total_eval_time / num_successful if num_successful > 0 else 0.0
    avg_problem_time = total_problem_time / num_successful if num_successful > 0 else 0.0

    llm_time_percentage = (total_llm_time / total_problem_time * 100.0) if total_problem_time > 0 else 0.0
    eval_time_percentage = (total_eval_time / total_problem_time * 100.0) if total_problem_time > 0 else 0.0

    avg_accuracy_8 = total_correct / total_solutions if total_solutions > 0 else 0.0

    return {
        "pass@1": pass_at_1,
        "pass@k": pass_at_k,
        "avg_final_score": avg_final_score,
        "avg_best_score": avg_best_score,
        "total_problems": total,
        "successful_problems": num_successful,
        "failed_problems": total - num_successful,
        "total_solutions_generated": total_solutions,
        "total_correct_solutions": total_correct,
        "total_tokens": total_tokens,
        "avg_tokens_per_problem": total_tokens / num_successful if num_successful > 0 else 0.0,
        "duration_seconds": (end_time - start_time).total_seconds(),
        "by_difficulty": {},
        "timing": {
            "total_llm_time_sec": total_llm_time,
            "total_evaluation_time_sec": total_eval_time,
            "total_problem_time_sec": total_problem_time,
            "avg_llm_time_sec": avg_llm_time,
            "avg_evaluation_time_sec": avg_eval_time,
            "avg_problem_time_sec": avg_problem_time,
            "llm_time_percentage": llm_time_percentage,
            "evaluation_time_percentage": eval_time_percentage,
        },
        "avg_accuracy_8": avg_accuracy_8,
    }


def _build_config(
    experiment_name: str,
    model_path: str,
    api_base: str,
    benchmark_name: str,
    benchmark_file: str,
    output_dir: str,
    max_concurrent: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Config:
    return Config(
        experiment_name=experiment_name,
        llm=LLMConfig(
            provider="sglang_direct",
            model_name=model_path,
            api_key="EMPTY",
            api_base=api_base,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            direct_prompt_template="math_direct_qwen_simple",
        ),
        evaluation=EvaluationConfig(
            benchmark=benchmark_name,
            evaluator_type="imobench_boxed",
            provider="sglang_direct",
            model_name=model_path,
            api_key="EMPTY",
            api_base=api_base,
            benchmark_file=benchmark_file,
        ),
        reflection=ReflectionConfig(
            strategy="none",
            n_iterations=0,
            n_samples_per_iteration=1,
            reasoning_effort="N/A",
        ),
        aggregation=AggregationConfig(
            strategy="none",
            apply_at_each_turn=False,
            reasoning_effort="N/A",
        ),
        output_dir=output_dir,
        max_concurrent_problems=max_concurrent,
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone IMOBench eval via direct SGLang requests")
    parser.add_argument("--input-file", type=str, required=True, help="IMOBench-format JSON file")
    parser.add_argument("--model-name", type=str, default=None, help="Model alias for output naming")
    parser.add_argument("--model-path", type=str, required=True, help="Model path served by SGLang")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer name/path for chat template")
    parser.add_argument("--output-dir", type=str, default="results/test_time_compute", help="Base output directory")
    parser.add_argument("--experiment-name", type=str, default="direct_generation", help="Experiment name")
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="OpenAI-compatible SGLang API base, e.g. http://127.0.0.1:30000/v1",
    )
    parser.add_argument("--n-samples", type=int, default=8, help="Number of trials per problem (default: 8)")
    parser.add_argument("--max-concurrent", type=int, default=16, help="Max concurrent problems")
    parser.add_argument(
        "--sample-concurrency",
        type=int,
        default=8,
        help="Number of parallel samples per problem (default: 8)",
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--max-tokens", type=int, default=35000, help="Max completion tokens")
    args = parser.parse_args()

    input_file = Path(args.input_file).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    api_base = (
        args.api_base
        or os.getenv("SGLANG_API_BASE")
        or os.getenv("OPENAI_API_BASE")
        or "http://127.0.0.1:30000/v1"
    )

    model_alias = _resolve_model_alias(args.model_name, args.model_path)
    benchmark_label = _sanitize_name(input_file.stem)
    benchmark_name = f"imobench_{benchmark_label}"
    output_dir = Path(args.output_dir) / f"{benchmark_name}_rollouts_{model_alias}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(
        experiment_name=args.experiment_name,
        model_path=args.model_path,
        api_base=api_base,
        benchmark_name=benchmark_name,
        benchmark_file=str(input_file),
        output_dir=str(output_dir),
        max_concurrent=args.max_concurrent,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    print("\n" + "=" * 80)
    print("IMOBENCH DIRECT EVALUATION (NO PIPELINE)")
    print("=" * 80)
    print(f"Input File: {input_file}")
    print(f"Benchmark Name: {benchmark_name}")
    print(f"Model Alias: {model_alias}")
    print(f"Model Path: {args.model_path}")
    print(f"SGLang API Base: {api_base}")
    print(f"Samples per Problem: {args.n_samples}")
    print(f"Parallel Samples per Problem: {args.sample_concurrency}")
    print(f"Max Concurrent Problems: {args.max_concurrent}")
    print("=" * 80)

    problems = _load_imobench_problems(input_file)
    print(f"Loaded {len(problems)} problems")

    tokenizer = None
    tokenizer_name = _resolve_tokenizer_name(args.model_path, args.tokenizer)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        print(f"Using tokenizer for chat template: {tokenizer_name}")
    except Exception as e:
        print(f"Tokenizer load failed ({tokenizer_name}), fallback prompt formatting will be used: {e}")

    client = AsyncOpenAI(api_key="EMPTY", base_url=api_base)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    start_time = datetime.now()

    async def _run_with_limit(problem: Problem) -> Dict[str, Any]:
        async with semaphore:
            try:
                result = await _evaluate_problem(
                    problem=problem,
                    client=client,
                    tokenizer=tokenizer,
                    model_path=args.model_path,
                    api_base=api_base,
                    n_samples=args.n_samples,
                    sample_concurrency=args.sample_concurrency,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_tokens=args.max_tokens,
                )
                print(
                    f"  ✓ {problem.id}: correct {result['metrics']['num_correct']}/{args.n_samples} "
                    f"(avg={result['metrics']['avg_accuracy_8']:.3f})"
                )
                return result
            except Exception as e:
                print(f"  ✗ {problem.id} failed: {e}")
                return {
                    "problem_id": problem.id,
                    "problem": problem.problem,
                    "ground_truth": problem.ground_truth,
                    "difficulty": problem.difficulty or problem.metadata.get("difficulty", "unknown"),
                    "error": str(e),
                }

    tasks = [_run_with_limit(p) for p in problems]
    results = await asyncio.gather(*tasks)

    end_time = datetime.now()
    metrics = _compute_experiment_metrics(results, start_time, end_time)

    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{args.experiment_name}_{benchmark_name}_{model_alias}_{timestamp}.json"
    output_payload = {
        "config": config.to_dict(),
        "benchmark": benchmark_name,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "metrics": metrics,
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"pass@1: {metrics.get('pass@1', 0):.4f}")
    print(f"pass@k: {metrics.get('pass@k', 0):.4f}")
    print(f"avg_accuracy_8: {metrics.get('avg_accuracy_8', 0):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
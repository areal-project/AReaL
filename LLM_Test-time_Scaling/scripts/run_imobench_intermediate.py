"""IMOBench intermediate experiment script.

This script implements three intermediate experiment patterns:
1. pairwise_comparison + 4 aggregation + 4 refinement
2. generate_1_from_n + 4 aggregation + 4 refinement  
3. generate_n_from_n + 4 aggregation

Usage:
    python -m scripts.run_imobench_intermediate --model gpt-oss-120b --resume
"""

import asyncio
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.experiment_runner import ExperimentRunner
from src.scaling.base import ScalingResult
from src.scaling.pipeline import PipelineFactory, ScalingPipeline
from src.utils.config import (
    AggregationConfig,
    Config,
    EvaluationConfig,
    LLMConfig,
    ReflectionConfig,
)


class IntermediateScalingPipeline(ScalingPipeline):
    """Custom pipeline for intermediate experiments with aggregation + refinement pattern."""

    def __init__(
        self,
        llm_service,
        reflection_strategy,
        aggregation_strategy,
        prompt_manager,
        task_domain: str = "general",
        n_iterations: int = 4,
        n_initial_solutions: int = 4,
    ):
        """Initialize intermediate scaling pipeline.

        Args:
            llm_service: LLM service
            reflection_strategy: Reflection strategy (for refinement)
            aggregation_strategy: Aggregation strategy
            prompt_manager: Prompt manager
            task_domain: Task domain
            n_iterations: Number of aggregation + refinement iterations
            n_initial_solutions: Number of initial solutions to generate
        """
        super().__init__(
            llm_service=llm_service,
            reflection_strategy=reflection_strategy,
            aggregation_strategy=aggregation_strategy,
            prompt_manager=prompt_manager,
            apply_aggregation_each_turn=False,  # We handle aggregation manually
            task_domain=task_domain,
        )
        self.n_iterations = n_iterations
        self.n_initial_solutions = n_initial_solutions

    async def run_intermediate_pattern(
        self,
        problem: str,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = "auto",
        **kwargs: Any,
    ) -> ScalingResult:
        """Run intermediate pattern: aggregate then refine for n_iterations.

        Args:
            problem: Problem statement
            temperature: Sampling temperature
            reasoning_effort: Reasoning effort for initial generation
            **kwargs: Additional parameters

        Returns:
            ScalingResult with final solution and all intermediate solutions
        """
        from src.scaling.base import ScalingResult

        # Generate initial solutions
        solutions = await self._generate_initial_solutions(
            problem, self.n_initial_solutions, temperature, self.task_domain, reasoning_effort
        )

        all_solutions = list(solutions)
        total_tokens = sum(
            [sol.metadata.get("usage", {}).get("total_tokens", 0) for sol in solutions]
        )

        current_solutions = solutions

        # Run n_iterations of aggregation + refinement
        for iteration in range(1, self.n_iterations + 1):
            print(f"Iteration {iteration}/{self.n_iterations}: Aggregation + Refinement", flush=True)

            # Step 1: Aggregate to get best solution
            if self.aggregation_strategy and len(current_solutions) > 1:
                best_solution = await self.aggregation_strategy.aggregate(
                    problem, current_solutions, **kwargs
                )
            else:
                best_solution = current_solutions[0] if current_solutions else solutions[0]

            # Step 2: Refine the best solution using reflection
            if self.reflection_strategy:
                # Use best solution for refinement (generate 1 refined solution)
                solutions_to_reflect = [best_solution]
                refined_solutions = await self.reflection_strategy.reflect(
                    problem, solutions_to_reflect, **kwargs
                )

                # Post-process refined solutions
                refined_solutions = [self._post_process_solution(sol) for sol in refined_solutions]

                for sol in refined_solutions:
                    sol.metadata["iteration"] = iteration
                    sol.metadata["aggregation_iteration"] = iteration

                all_solutions.extend(refined_solutions)
                current_solutions = refined_solutions
            else:
                # No refinement, just keep best solution
                best_solution.metadata["iteration"] = iteration
                best_solution.metadata["aggregation_iteration"] = iteration
                current_solutions = [best_solution]

        # Final solution is the last refined solution
        final_solution = current_solutions[0] if current_solutions else solutions[0]

        return ScalingResult(
            final_solution=final_solution,
            all_solutions=all_solutions,
            iterations=self.n_iterations,
            total_tokens=total_tokens,
            metadata={
                "n_initial_solutions": self.n_initial_solutions,
                "pattern": "aggregate_then_refine",
                "reflection_strategy": (
                    self.reflection_strategy.get_strategy_name()
                    if self.reflection_strategy
                    else None
                ),
                "aggregation_strategy": (
                    self.aggregation_strategy.get_strategy_name()
                    if self.aggregation_strategy
                    else None
                ),
            },
        )


class SparseRefinementPipeline(ScalingPipeline):
    """Pipeline for sparse refinement pattern: aggregate then refine twice per round.

    Pattern: init -> aggregate -> refine -> refine -> aggregate -> refine -> refine -> ...
    Each round: aggregate once, then refine twice.
    """

    def __init__(
        self,
        llm_service,
        reflection_strategy,
        aggregation_strategy,
        prompt_manager,
        task_domain: str = "general",
        n_aggregation_rounds: int = 2,
        n_refinements_per_round: int = 2,
        n_initial_solutions: int = 4,
        generate_n_from_n: bool = False,
        n_generate_per_round: int = 4,
    ):
        """Initialize sparse refinement pipeline.

        Args:
            llm_service: LLM service
            reflection_strategy: Reflection strategy (for refinement)
            aggregation_strategy: Aggregation strategy
            prompt_manager: Prompt manager
            task_domain: Task domain
            n_aggregation_rounds: Number of aggregation rounds (each round: aggregate + 2*refine)
            n_refinements_per_round: Number of refinements per aggregation round (default: 2)
            n_initial_solutions: Number of initial solutions to generate
            generate_n_from_n: If True, each round performs n independent aggregates, each refined separately
            n_generate_per_round: Number of independent aggregates per round when generate_n_from_n=True
        """
        super().__init__(
            llm_service=llm_service,
            reflection_strategy=reflection_strategy,
            aggregation_strategy=aggregation_strategy,
            prompt_manager=prompt_manager,
            apply_aggregation_each_turn=False,
            task_domain=task_domain,
        )
        self.n_aggregation_rounds = n_aggregation_rounds
        self.n_refinements_per_round = n_refinements_per_round
        self.n_initial_solutions = n_initial_solutions
        self.generate_n_from_n = generate_n_from_n
        self.n_generate_per_round = n_generate_per_round

    async def run_sparse_refinement_pattern(
        self,
        problem: str,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = "auto",
        **kwargs: Any,
    ) -> ScalingResult:
        """Run sparse refinement pattern: aggregate then refine twice per round.

        Pattern: init -> aggregate -> refine -> refine -> aggregate -> refine -> refine -> ...
        Each round: aggregate once, then refine n_refinements_per_round times.

        Args:
            problem: Problem statement
            temperature: Sampling temperature
            reasoning_effort: Reasoning effort for initial generation
            **kwargs: Additional parameters

        Returns:
            ScalingResult with final solution and all intermediate solutions
        """
        from src.scaling.base import ScalingResult

        # Generate initial solutions
        solutions = await self._generate_initial_solutions(
            problem, self.n_initial_solutions, temperature, self.task_domain, reasoning_effort
        )

        all_solutions = list(solutions)
        total_tokens = sum(
            [sol.metadata.get("usage", {}).get("total_tokens", 0) for sol in solutions]
        )

        current_solutions = solutions

        # Run aggregation rounds: each round = aggregate + n_refinements_per_round * refine
        for agg_round in range(1, self.n_aggregation_rounds + 1):
            if self.generate_n_from_n:
                print(
                    f"Round {agg_round}/{self.n_aggregation_rounds}: {self.n_generate_per_round} Aggregates + {self.n_refinements_per_round} Refinements each",
                    flush=True,
                )
            else:
                print(
                    f"Round {agg_round}/{self.n_aggregation_rounds}: Aggregate + {self.n_refinements_per_round} Refinements",
                    flush=True,
                )

            if self.generate_n_from_n:
                # Generate n_from_n pattern: perform n independent aggregates, each refined separately
                round_solutions = []
                
                # Step 1: Generate n_generate_per_round independent aggregates
                for gen_idx in range(1, self.n_generate_per_round + 1):
                    print(
                        f"  Aggregate {gen_idx}/{self.n_generate_per_round}",
                        flush=True,
                    )
                    
                    if self.aggregation_strategy and len(current_solutions) > 1:
                        aggregated_solution = await self.aggregation_strategy.aggregate(
                            problem, current_solutions, **kwargs
                        )
                    else:
                        aggregated_solution = current_solutions[0] if current_solutions else solutions[0]
                    
                    # Mark aggregation round and generation index
                    aggregated_solution.metadata["aggregation_round"] = agg_round
                    aggregated_solution.metadata["iteration"] = agg_round
                    aggregated_solution.metadata["generation_index"] = gen_idx
                    
                    # Step 2: Refine this aggregated solution n_refinements_per_round times
                    solution_to_refine = aggregated_solution
                    
                    if self.reflection_strategy:
                        for refine_idx in range(1, self.n_refinements_per_round + 1):
                            print(
                                f"    Refinement {refine_idx}/{self.n_refinements_per_round} (for aggregate {gen_idx})",
                                flush=True,
                            )
                            
                            # Use current solution for refinement
                            solutions_to_reflect = [solution_to_refine]
                            refined_solutions = await self.reflection_strategy.reflect(
                                problem, solutions_to_reflect, **kwargs
                            )
                            
                            # Post-process refined solutions
                            refined_solutions = [
                                self._post_process_solution(sol) for sol in refined_solutions
                            ]
                            
                            for sol in refined_solutions:
                                sol.metadata["iteration"] = agg_round
                                sol.metadata["aggregation_round"] = agg_round
                                sol.metadata["refinement_round"] = refine_idx
                                sol.metadata["refinement_index"] = refine_idx
                                sol.metadata["generation_index"] = gen_idx
                            
                            all_solutions.extend(refined_solutions)
                            
                            # Use the refined solution for next refinement (if any)
                            if refined_solutions:
                                solution_to_refine = refined_solutions[0]
                        
                        # After all refinements, add the final refined solution to round_solutions
                        if solution_to_refine:
                            round_solutions.append(solution_to_refine)
                    else:
                        # No refinement, use aggregated solution
                        round_solutions.append(aggregated_solution)
                
                # Update current_solutions to all refined solutions from this round
                current_solutions = round_solutions
            else:
                # Original pattern: single aggregate then refine n times
                # Step 1: Aggregate to get best solution
                if self.aggregation_strategy and len(current_solutions) > 1:
                    best_solution = await self.aggregation_strategy.aggregate(
                        problem, current_solutions, **kwargs
                    )
                else:
                    best_solution = current_solutions[0] if current_solutions else solutions[0]

                # Mark aggregation round
                best_solution.metadata["aggregation_round"] = agg_round
                best_solution.metadata["iteration"] = agg_round

                # Step 2: Refine n_refinements_per_round times
                if self.reflection_strategy:
                    solution_to_refine = best_solution
                    
                    for refine_idx in range(1, self.n_refinements_per_round + 1):
                        print(
                            f"  Refinement {refine_idx}/{self.n_refinements_per_round}",
                            flush=True,
                        )

                        # Use current solution for refinement
                        solutions_to_reflect = [solution_to_refine]
                        refined_solutions = await self.reflection_strategy.reflect(
                            problem, solutions_to_reflect, **kwargs
                        )

                        # Post-process refined solutions
                        refined_solutions = [
                            self._post_process_solution(sol) for sol in refined_solutions
                        ]

                        for sol in refined_solutions:
                            sol.metadata["iteration"] = agg_round
                            sol.metadata["aggregation_round"] = agg_round
                            sol.metadata["refinement_round"] = refine_idx
                            sol.metadata["refinement_index"] = refine_idx

                        all_solutions.extend(refined_solutions)
                        
                        # Use the refined solution for next refinement (if any)
                        if refined_solutions:
                            solution_to_refine = refined_solutions[0]
                            current_solutions = refined_solutions
                else:
                    # No refinement, just keep best solution
                    all_solutions.append(best_solution)
                    current_solutions = [best_solution]

        # Final aggregation if we have multiple solutions
        print("Final Aggregation", flush=True)
        if self.aggregation_strategy and len(current_solutions) > 1:
            final_solution = await self.aggregation_strategy.aggregate(
                problem, current_solutions, **kwargs
            )
            final_solution.metadata["iteration"] = self.n_aggregation_rounds + 1
            final_solution.metadata["aggregation_round"] = self.n_aggregation_rounds + 1
            final_solution.metadata["is_final_aggregation"] = True
            all_solutions.append(final_solution)
        else:
            # No final aggregation needed, use last solution
            final_solution = current_solutions[0] if current_solutions else solutions[0]

        return ScalingResult(
            final_solution=final_solution,
            all_solutions=all_solutions,
            iterations=self.n_aggregation_rounds,
            total_tokens=total_tokens,
            metadata={
                "n_initial_solutions": self.n_initial_solutions,
                "pattern": "sparse_refinement",
                "n_aggregation_rounds": self.n_aggregation_rounds,
                "n_refinements_per_round": self.n_refinements_per_round,
                "generate_n_from_n": self.generate_n_from_n,
                "n_generate_per_round": self.n_generate_per_round if self.generate_n_from_n else None,
                "reflection_strategy": (
                    self.reflection_strategy.get_strategy_name()
                    if self.reflection_strategy
                    else None
                ),
                "aggregation_strategy": (
                    self.aggregation_strategy.get_strategy_name()
                    if self.aggregation_strategy
                    else None
                ),
            },
        )


class GenerateNFromNPipeline(ScalingPipeline):
    """Pipeline for generate n from n pattern: generate multiple solutions each iteration."""

    def __init__(
        self,
        llm_service,
        aggregation_strategy,
        prompt_manager,
        task_domain: str = "general",
        n_iterations: int = 4,
        n_initial_solutions: int = 4,
        n_generate_per_iteration: int = 4,
    ):
        """Initialize generate n from n pipeline.

        Args:
            llm_service: LLM service
            aggregation_strategy: Aggregation strategy for generating from n
            prompt_manager: Prompt manager
            task_domain: Task domain
            n_iterations: Number of iterations
            n_initial_solutions: Number of initial solutions
            n_generate_per_iteration: Number of solutions to generate per iteration
        """
        super().__init__(
            llm_service=llm_service,
            reflection_strategy=None,  # No reflection, only aggregation
            aggregation_strategy=aggregation_strategy,
            prompt_manager=prompt_manager,
            apply_aggregation_each_turn=False,
            task_domain=task_domain,
        )
        self.n_iterations = n_iterations
        self.n_initial_solutions = n_initial_solutions
        self.n_generate_per_iteration = n_generate_per_iteration

    async def run_generate_n_from_n(
        self,
        problem: str,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = "auto",
        **kwargs: Any,
    ) -> ScalingResult:
        """Run generate n from n pattern: generate multiple solutions each iteration.

        Args:
            problem: Problem statement
            temperature: Sampling temperature
            reasoning_effort: Reasoning effort
            **kwargs: Additional parameters

        Returns:
            ScalingResult with final solution
        """
        from src.scaling.base import ScalingResult

        # Generate initial solutions
        solutions = await self._generate_initial_solutions(
            problem, self.n_initial_solutions, temperature, self.task_domain, reasoning_effort
        )

        all_solutions = list(solutions)
        total_tokens = sum(
            [sol.metadata.get("usage", {}).get("total_tokens", 0) for sol in solutions]
        )

        current_solutions = solutions

        # Run n_iterations of generate n from n
        for iteration in range(1, self.n_iterations + 1):
            print(
                f"Iteration {iteration}/{self.n_iterations}: Generate {self.n_generate_per_iteration} from {len(current_solutions)}",
                flush=True,
            )

            # Generate n_generate_per_iteration new solutions from current solutions
            new_solutions = []
            for _ in range(self.n_generate_per_iteration):
                if self.aggregation_strategy and len(current_solutions) > 1:
                    new_solution = await self.aggregation_strategy.aggregate(
                        problem, current_solutions, **kwargs
                    )
                else:
                    new_solution = current_solutions[0] if current_solutions else solutions[0]

                new_solution.metadata["iteration"] = iteration
                new_solutions.append(new_solution)

            # Post-process new solutions
            new_solutions = [self._post_process_solution(sol) for sol in new_solutions]

            all_solutions.extend(new_solutions)
            current_solutions = new_solutions

        # Final aggregation if we have multiple solutions
        if self.aggregation_strategy and len(current_solutions) > 1:
            final_solution = await self.aggregation_strategy.aggregate(
                problem, current_solutions, **kwargs
            )
        else:
            final_solution = current_solutions[0] if current_solutions else solutions[0]

        return ScalingResult(
            final_solution=final_solution,
            all_solutions=all_solutions,
            iterations=self.n_iterations,
            total_tokens=total_tokens,
            metadata={
                "n_initial_solutions": self.n_initial_solutions,
                "pattern": "generate_n_from_n",
                "n_generate_per_iteration": self.n_generate_per_iteration,
                "aggregation_strategy": (
                    self.aggregation_strategy.get_strategy_name()
                    if self.aggregation_strategy
                    else None
                ),
            },
        )


def create_experiment_config(
    experiment_name: str,
    model_name: str,
    output_dir: str,
    api_key: str = None,
    api_base: str = None,
    max_concurrent_problems: int = 128,
    reasoning_effort: str = "auto",
    initial_effort: str = "auto",
) -> Config:
    """Create experiment configuration.

    Args:
        experiment_name: Name of the experiment
        model_name: Model name
        output_dir: Output directory
        api_key: API key
        api_base: API base URL
        max_concurrent_problems: Maximum concurrent problems
        reasoning_effort: Reasoning effort for refinement
        initial_effort: Reasoning effort for initial generation

    Returns:
        Config object
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY", "None")
    api_base = api_base or os.getenv("OPENAI_API_BASE")
    eval_api_base = os.getenv("EVAL_OPENAI_API_BASE") or api_base

    return Config(
        experiment_name=experiment_name,
        llm=LLMConfig(
            provider="litellm",
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=0.7,
            reasoning_effort=initial_effort,
        ),
        evaluation=EvaluationConfig(
            benchmark="imobench",
            evaluator_type="llm_judge",
            provider="litellm",
            model_name=model_name,
            api_key=api_key,
            api_base=eval_api_base,
        ),
        reflection=ReflectionConfig(
            strategy="self_evaluation",
            n_iterations=4,  # Not used directly, but kept for compatibility
            n_samples_per_iteration=1,  # Generate 1 refined solution per iteration
            reasoning_effort=reasoning_effort,
        ),
        aggregation=AggregationConfig(
            strategy="none",  # Will be set per experiment
            apply_at_each_turn=False,
            reasoning_effort=reasoning_effort,
        ),
        output_dir=output_dir,
        max_concurrent_problems=max_concurrent_problems,
    )


class IntermediateExperimentRunner(ExperimentRunner):
    """Custom experiment runner for intermediate experiments."""

    def __init__(self, config: Config, resume_from: Optional[str] = None, pattern: str = "aggregate_then_refine", aggregation_strategy_name: str = "pairwise_comparison"):
        """Initialize intermediate experiment runner.

        Args:
            config: Experiment configuration
            resume_from: Path to resume from
            pattern: Pattern type ("aggregate_then_refine" or "generate_n_from_n")
            aggregation_strategy_name: Name of aggregation strategy
        """
        self.pattern = pattern
        self.aggregation_strategy_name = aggregation_strategy_name
        super().__init__(config, resume_from=resume_from)

    def _create_pipeline(self) -> ScalingPipeline:
        """Create custom pipeline based on pattern."""
        factory = PipelineFactory(self.llm_service, self.prompt_manager)
        task_domain = self._infer_task_domain(self.config.evaluation.benchmark)

        # Create reflection strategy (for aggregate_then_refine pattern)
        reflection = None
        if self.pattern == "aggregate_then_refine":
            reflection_kwargs = {
                "task_domain": task_domain,
                "reasoning_effort": self.config.reflection.reasoning_effort,
            }
            reflection = factory.create_reflection_strategy(
                self.config.reflection.strategy,
                **reflection_kwargs
            )

        # Create aggregation strategy
        aggregation_kwargs = {
            "task_domain": task_domain,
            "reasoning_effort": self.config.aggregation.reasoning_effort,
        }
        aggregation = factory.create_aggregation_strategy(
            self.aggregation_strategy_name,
            **aggregation_kwargs
        )

        # Create appropriate pipeline
        if self.pattern == "generate_n_from_n":
            return GenerateNFromNPipeline(
                self.llm_service,
                aggregation,
                self.prompt_manager,
                task_domain=task_domain,
                n_iterations=4,
                n_initial_solutions=4,
                n_generate_per_iteration=4,
            )
        elif self.pattern == "sparse_refinement":
            # Sparse refinement pattern: aggregate then refine twice per round
            # Parameters are determined by experiment name or passed via metadata
            n_agg_rounds = getattr(self, 'n_aggregation_rounds', 2)
            n_refinements_per_round = getattr(self, 'n_refinements_per_round', 2)
            return SparseRefinementPipeline(
                self.llm_service,
                reflection,
                aggregation,
                self.prompt_manager,
                task_domain=task_domain,
                n_aggregation_rounds=n_agg_rounds,
                n_refinements_per_round=n_refinements_per_round,
                n_initial_solutions=4,
            )
        else:  # aggregate_then_refine
            return IntermediateScalingPipeline(
                self.llm_service,
                reflection,
                aggregation,
                self.prompt_manager,
                task_domain=task_domain,
                n_iterations=4,
                n_initial_solutions=4,
            )

    async def run_single_problem(
        self, problem_data: Any, task_domain: str = "general", **kwargs: Any
    ) -> Dict[str, Any]:
        """Run single problem with custom pipeline pattern."""
        import time
        problem = problem_data.problem
        ground_truth = problem_data.ground_truth
        test_cases = problem_data.test_cases

        problem_start_time = time.time()

        # Prepare kwargs for pipeline
        pipeline_kwargs = {
            "ground_truth": ground_truth,
            "test_cases": test_cases,
        }
        if hasattr(problem_data, 'id'):
            pipeline_kwargs["problem_id"] = problem_data.id
        if hasattr(self.evaluator, 'language'):
            pipeline_kwargs["language"] = getattr(self.evaluator, 'language', 'cpp')
        elif hasattr(self.config.evaluation, 'language'):
            pipeline_kwargs["language"] = getattr(self.config.evaluation, 'language', 'cpp')

        # Run pipeline based on pattern
        if self.pattern == "generate_n_from_n":
            scaling_result = await self.pipeline.run_generate_n_from_n(
                problem=problem,
                temperature=self.config.llm.temperature,
                reasoning_effort=self.config.llm.reasoning_effort,
                **pipeline_kwargs,
            )
        elif self.pattern == "sparse_refinement":
            scaling_result = await self.pipeline.run_sparse_refinement_pattern(
                problem=problem,
                temperature=self.config.llm.temperature,
                reasoning_effort=self.config.llm.reasoning_effort,
                **pipeline_kwargs,
            )
        else:  # aggregate_then_refine
            scaling_result = await self.pipeline.run_intermediate_pattern(
                problem=problem,
                temperature=self.config.llm.temperature,
                reasoning_effort=self.config.llm.reasoning_effort,
                **pipeline_kwargs,
            )

        # Collect LLM call times
        total_llm_time = 0.0
        for solution in scaling_result.all_solutions:
            solution_metadata = solution.metadata or {}
            usage = solution_metadata.get("usage", {})
            if isinstance(usage, dict):
                llm_time = usage.get("llm_call_time_sec", 0.0)
                if isinstance(llm_time, (int, float)):
                    total_llm_time += llm_time
            improve_usage = solution_metadata.get("improve_usage", {})
            if isinstance(improve_usage, dict):
                llm_time = improve_usage.get("llm_call_time_sec", 0.0)
                if isinstance(llm_time, (int, float)):
                    total_llm_time += llm_time
            eval_usage = solution_metadata.get("eval_usage", {})
            if isinstance(eval_usage, dict):
                llm_time = eval_usage.get("llm_call_time_sec", 0.0)
                if isinstance(llm_time, (int, float)):
                    total_llm_time += llm_time

        # Evaluate ALL solutions
        all_evaluations = []
        total_evaluation_time = 0.0
        if self.evaluator:
            for solution in scaling_result.all_solutions:
                eval_kwargs = {
                    "problem": problem,
                    "solution": solution.content,
                    "ground_truth": ground_truth,
                }
                if hasattr(problem_data, 'id'):
                    eval_kwargs["problem_id"] = problem_data.id
                if test_cases and not (hasattr(self.evaluator, 'local_data_dir') and self.evaluator.local_data_dir):
                    eval_kwargs["test_cases"] = test_cases
                
                eval_result = await self.evaluator.evaluate(**eval_kwargs)
                eval_time = 0.0
                if eval_result.details and isinstance(eval_result.details, dict):
                    eval_time = eval_result.details.get("evaluation_time_sec", 0.0) or 0.0
                    if isinstance(eval_time, (int, float)):
                        total_evaluation_time += eval_time
                
                all_evaluations.append({
                    "solution_content": solution.content,
                    "is_correct": eval_result.is_correct,
                    "score": eval_result.score,
                    "feedback": eval_result.feedback,
                    "metadata": solution.metadata,
                })

        # Evaluate final solution
        final_eval_result = None
        if self.evaluator:
            eval_kwargs = {
                "problem": problem,
                "solution": scaling_result.final_solution.content,
                "ground_truth": ground_truth,
            }
            if hasattr(problem_data, 'id'):
                eval_kwargs["problem_id"] = problem_data.id
            if test_cases and not (hasattr(self.evaluator, 'local_data_dir') and self.evaluator.local_data_dir):
                eval_kwargs["test_cases"] = test_cases
            
            final_eval_result = await self.evaluator.evaluate(**eval_kwargs)
            if final_eval_result and final_eval_result.details and isinstance(final_eval_result.details, dict):
                final_eval_time = final_eval_result.details.get("evaluation_time_sec", 0.0) or 0.0
                if isinstance(final_eval_time, (int, float)):
                    total_evaluation_time += final_eval_time

        problem_end_time = time.time()
        total_problem_time = problem_end_time - problem_start_time

        # Calculate pass@k metrics
        correct_solutions = [e for e in all_evaluations if e["is_correct"]]
        pass_at_1 = 1 if final_eval_result and final_eval_result.is_correct else 0
        pass_at_k = 1 if len(correct_solutions) > 0 else 0
        best_score_all = max((e["score"] for e in all_evaluations), default=0.0)

        difficulty = getattr(problem_data, 'difficulty', None) or problem_data.metadata.get('difficulty', 'unknown')
        
        llm_time_percentage = (total_llm_time / total_problem_time * 100) if total_problem_time > 0 else 0.0
        eval_time_percentage = (total_evaluation_time / total_problem_time * 100) if total_problem_time > 0 else 0.0
        
        return {
            "problem_id": problem_data.id,
            "problem": problem,
            "ground_truth": ground_truth,
            "difficulty": difficulty,
            "final_solution": {
                "content": scaling_result.final_solution.content,
                "is_correct": final_eval_result.is_correct if final_eval_result else None,
                "score": final_eval_result.score if final_eval_result else None,
                "feedback": final_eval_result.feedback if final_eval_result else None,
                "metadata": scaling_result.final_solution.metadata,
            },
            "all_solutions": all_evaluations,
            "metrics": {
                "pass@1": pass_at_1,
                "pass@k": pass_at_k,
                "num_correct": len(correct_solutions),
                "num_total": len(all_evaluations),
                "final_score": final_eval_result.score if final_eval_result else 0.0,
                "best_score_all": best_score_all,
            },
            "total_tokens": scaling_result.total_tokens,
            "iterations": scaling_result.iterations,
            "scaling_metadata": scaling_result.metadata,
            "timing": {
                "total_time_sec": total_problem_time,
                "llm_time_sec": total_llm_time,
                "evaluation_time_sec": total_evaluation_time,
                "llm_time_percentage": llm_time_percentage,
                "evaluation_time_percentage": eval_time_percentage,
            },
        }


@contextmanager
def setup_experiment_logging(experiment_name: str, log_dir: Path):
    """Set up logging for a single experiment.
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to save log files
        
    Yields:
        Logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{experiment_name}.log"
    
    # Create logger for this experiment
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Remove any existing handlers
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (with prefix to identify experiment)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        f'[{experiment_name}] %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    try:
        yield logger
    finally:
        # Clean up handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


async def run_intermediate_experiment(
    experiment_name: str,
    model_name: str,
    pattern: str,
    aggregation_strategy: str,
    output_dir: str,
    api_key: str = None,
    api_base: str = None,
    resume_from: str = None,
    max_concurrent_problems: int = 40,
    reasoning_effort: str = "auto",
    initial_effort: str = "auto",
    log_dir: Optional[Path] = None,
    sparse_refinement_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Run an intermediate experiment.

    Args:
        experiment_name: Name of the experiment
        model_name: Model name
        pattern: Pattern type ("aggregate_then_refine", "generate_n_from_n", or "sparse_refinement")
        aggregation_strategy: Aggregation strategy name
        output_dir: Output directory
        api_key: API key
        api_base: API base URL
        resume_from: Path to resume from
        max_concurrent_problems: Max concurrent problems (default: 40)
        reasoning_effort: Reasoning effort for refinement
        initial_effort: Reasoning effort for initial generation
        log_dir: Directory for log files
        sparse_refinement_config: Configuration for sparse refinement pattern
    """
    # Set up logging for this experiment
    if log_dir is None:
        log_dir = Path(output_dir) / "logs"
    
    with setup_experiment_logging(experiment_name, log_dir) as logger:
        logger.info("\n" + "=" * 80)
        logger.info(f"EXPERIMENT: {experiment_name}")
        logger.info("=" * 80)
        logger.info(f"Model: {model_name}")
        logger.info(f"Pattern: {pattern}")
        logger.info(f"Aggregation: {aggregation_strategy}")
        logger.info(f"Initial Effort: {initial_effort}")
        logger.info(f"Refine Effort: {reasoning_effort}")
        logger.info(f"Max Concurrent: {max_concurrent_problems}")
        if resume_from:
            logger.info(f"Resuming from: {resume_from}")
        logger.info("=" * 80)

        try:
            config = create_experiment_config(
                experiment_name=experiment_name,
                model_name=model_name,
                output_dir=output_dir,
                api_key=api_key,
                api_base=api_base,
                max_concurrent_problems=max_concurrent_problems,
                reasoning_effort=reasoning_effort,
                initial_effort=initial_effort,
            )

            runner = IntermediateExperimentRunner(
                config,
                resume_from=resume_from,
                pattern=pattern,
                aggregation_strategy_name=aggregation_strategy,
            )
            
            # Set sparse refinement parameters if applicable
            if sparse_refinement_config:
                runner.n_aggregation_rounds = sparse_refinement_config.get("n_aggregation_rounds", 2)
                runner.n_refinements_per_round = sparse_refinement_config.get("n_refinements_per_round", 2)
                runner.generate_n_from_n = sparse_refinement_config.get("generate_n_from_n", False)
                runner.n_generate_per_round = sparse_refinement_config.get("n_generate_per_round", 4)
            
            await runner.run()
            
            logger.info(f"\n✅ Experiment {experiment_name} completed successfully")
        except Exception as e:
            logger.error(f"\n❌ Experiment {experiment_name} failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


async def main():
    """Main function to run all intermediate experiments."""
    import argparse
    from typing import Optional

    parser = argparse.ArgumentParser(
        description="Run IMOBench intermediate experiments"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Auto-resume from latest results files",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from specific results file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-120b",
        choices=["gpt-oss-120b", "gpt-oss-20b", "qwen3-235b", "qwen3-30b"],
        help="Model to use (default: gpt-oss-120b)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=40,
        help="Maximum concurrent problems per experiment (default: 40)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run experiments sequentially instead of in parallel",
    )
    args = parser.parse_args()

    # Configuration
    MODEL_PATHS = {
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "qwen3-235b": "openai/Qwen__Qwen3-235B-A22B",
        "qwen3-30b": "openai/Qwen__Qwen3-30B-A3B",
    }

    MODEL = MODEL_PATHS[args.model]
    OUTPUT_DIR = f"results/intermediate_experiments/imobench_{args.model}"
    API_KEY = os.getenv("OPENAI_API_KEY")
    API_BASE = os.getenv("OPENAI_API_BASE")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Experiment definitions
    experiments = [
        # {
        #     "name": "pairwise_comparison_4agg_4refine",
        #     "pattern": "aggregate_then_refine",
        #     "aggregation": "pairwise_comparison",
        # },
        # {
        #     "name": "generate_1_from_n_4agg_4refine",
        #     "pattern": "aggregate_then_refine",
        #     "aggregation": "generate_from_n",
        # },
        # {
        #     "name": "generate_n_from_n_4agg",
        #     "pattern": "generate_n_from_n",
        #     "aggregation": "generate_from_n",
        # },
        # Sparse refinement experiments (aggregate then refine twice per round)
        {
            "name": "pairwise_comparison_2agg_4refine_sparse",
            "pattern": "sparse_refinement",
            "aggregation": "pairwise_comparison",
            "n_aggregation_rounds": 2,
            "n_refinements_per_round": 2,  # 2 rounds * 2 refinements = 4 total refinements
        },
        {
            "name": "generate_1_from_n_2agg_4refine_sparse",
            "pattern": "sparse_refinement",
            "aggregation": "generate_from_n",
            "n_aggregation_rounds": 2,
            "n_refinements_per_round": 2,  # 2 rounds * 2 refinements = 4 total refinements
        },
        {
            "name": "generate_n_from_n_2agg_2refine_sparse",
            "pattern": "sparse_refinement",
            "aggregation": "generate_from_n",
            "n_aggregation_rounds": 2,
            "n_refinements_per_round": 1,  # 2 rounds * 1 refinement = 2 total refinements
            "generate_n_from_n": True,  # Each round: n independent aggregates, each refined separately
            "n_generate_per_round": 4,  # Generate 4 independent aggregates per round
        },
    ]

    LOG_DIR = Path(OUTPUT_DIR) / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("IMOBENCH INTERMEDIATE EXPERIMENTS")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Log Directory: {LOG_DIR}")
    print(f"Max Concurrent per Experiment: {args.max_concurrent}")
    print(f"Total Experiments: {len(experiments)}")
    print(f"Execution Mode: {'Sequential' if args.sequential else 'Parallel'}")
    if args.resume:
        print("Mode: Auto-resume")
    elif args.resume_from:
        print(f"Mode: Resume from {args.resume_from}")
    print("=" * 80)

    # Prepare experiment tasks
    async def run_single_experiment(exp_config: Dict[str, Any], idx: int) -> None:
        """Run a single experiment with error handling."""
        exp_name = exp_config["name"]
        resume_from = args.resume_from
        
        if args.resume and not resume_from:
            output_path = Path(OUTPUT_DIR)
            if output_path.exists():
                pattern = f"{exp_name}_imobench_*.json"
                result_files = sorted(
                    output_path.glob(pattern), key=lambda p: p.stat().st_mtime
                )
                if result_files:
                    resume_from = str(result_files[-1])
                    print(f"[{exp_name}] Auto-detected resume file: {resume_from}")

        await run_intermediate_experiment(
            experiment_name=exp_name,
            model_name=MODEL,
            pattern=exp_config["pattern"],
            aggregation_strategy=exp_config["aggregation"],
            output_dir=OUTPUT_DIR,
            api_key=API_KEY,
            api_base=API_BASE,
            resume_from=resume_from,
            max_concurrent_problems=args.max_concurrent,
            reasoning_effort="auto",
            initial_effort="auto",
            log_dir=LOG_DIR,
            sparse_refinement_config=exp_config if exp_config["pattern"] == "sparse_refinement" else None,
        )

    # Run experiments
    if args.sequential:
        # Sequential execution
        for idx, exp_config in enumerate(experiments, 1):
            print(f"\n[{idx}/{len(experiments)}] Starting experiment: {exp_config['name']}")
            try:
                await run_single_experiment(exp_config, idx)
            except Exception as e:
                print(f"\n❌ Experiment {exp_config['name']} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
            await asyncio.sleep(2)
    else:
        # Parallel execution
        print(f"\n🚀 Starting all {len(experiments)} experiments in parallel...")
        tasks = [
            run_single_experiment(exp_config, idx)
            for idx, exp_config in enumerate(experiments, 1)
        ]
        
        # Run all experiments in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for failures
        for idx, (exp_config, result) in enumerate(zip(experiments, results), 1):
            if isinstance(result, Exception):
                print(f"\n❌ Experiment {exp_config['name']} failed: {result}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ ALL EXPERIMENTS COMPLETED")
    print("=" * 80)
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Logs saved in: {LOG_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

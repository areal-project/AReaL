"""Test-time scaling pipeline orchestrator."""

import asyncio
from typing import Any, Dict, List, Optional

from ..llm_service import LLMService, Message
from ..prompts import PromptManager, PromptTemplate
from .aggregation.base import AggregationStrategy
from .base import ScalingResult, Solution
from .reflection.base import ReflectionStrategy
from .utils import extract_detailed_solution, has_structured_format


class ScalingPipeline:
    """Pipeline for combining reflection and aggregation strategies."""

    def __init__(
        self,
        llm_service: LLMService,
        reflection_strategy: Optional[ReflectionStrategy] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None,
        prompt_manager: Optional[PromptManager] = None,
        apply_aggregation_each_turn: bool = False,
        extract_detailed: bool = True,
        task_domain: str = "general",
        direct_prompt_template_name: Optional[str] = None,
    ):
        """Initialize the scaling pipeline.

        Args:
            llm_service: LLM service for generation
            reflection_strategy: Strategy for reflection (None = no reflection)
            aggregation_strategy: Strategy for aggregation (None = return first)
            prompt_manager: Prompt manager for accessing templates
            apply_aggregation_each_turn: If True, aggregate at each reflection turn
                                       If False, reflect first then aggregate once
            extract_detailed: If True, extract "Detailed Solution" section from structured outputs
            task_domain: Task domain (math, coding, etc.) for domain-specific processing
            direct_prompt_template_name: Optional direct-generation template name override
        """
        self.llm_service = llm_service
        self.reflection_strategy = reflection_strategy
        self.aggregation_strategy = aggregation_strategy
        self.prompt_manager = prompt_manager
        self.apply_aggregation_each_turn = apply_aggregation_each_turn
        self.extract_detailed = extract_detailed
        self.task_domain = task_domain
        self.direct_prompt_template_name = direct_prompt_template_name

    async def run(
        self,
        problem: str,
        n_initial_solutions: int = 1,
        n_iterations: int = 1,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        task_domain: str = "general",
        reasoning_effort: Optional[str] = "auto",
        initial_solutions: Optional[List[Solution]] = None,
        **kwargs: Any,
    ) -> ScalingResult:
        """Run the test-time scaling pipeline.

        Args:
            problem: Problem statement
            n_initial_solutions: Number of initial solutions to generate (ignored if initial_solutions provided)
            n_iterations: Number of reflection iterations
            temperature: Sampling temperature
            task_domain: Task domain (math, coding, professional_reasoning, etc.)
            reasoning_effort: Reasoning effort for initial generation ("auto", "low", "medium", "high", or None)
            initial_solutions: Optional list of existing solutions to continue from (if provided, skips initial generation)
            **kwargs: Additional parameters (ground_truth, test_cases, etc.)

        Returns:
            ScalingResult with final solution and metadata
        """
        # Use provided initial solutions or generate new ones
        if initial_solutions is not None:
            solutions = initial_solutions
        else:
            solutions = await self._generate_initial_solutions(
                problem,
                n_initial_solutions,
                temperature,
                task_domain,
                reasoning_effort,
                top_p,
                top_k,
            )

        all_solutions = list(solutions)
        total_tokens = sum(
            [
                sol.metadata.get("usage", {}).get("total_tokens", 0)
                for sol in solutions
            ]
        )

        if self.apply_aggregation_each_turn:
            # Pattern: Aggregation at each reflection turn
            final_solution = await self._run_with_aggregation_each_turn(
                problem, solutions, n_iterations, all_solutions, n_initial_solutions, **kwargs
            )
        else:
            # Pattern: Reflection first, then aggregation
            final_solution = await self._run_reflection_then_aggregation(
                problem, solutions, n_iterations, all_solutions, **kwargs
            )

        return ScalingResult(
            final_solution=final_solution,
            all_solutions=all_solutions,
            iterations=n_iterations,
            total_tokens=total_tokens,
            metadata={
                "n_initial_solutions": n_initial_solutions,
                "apply_aggregation_each_turn": self.apply_aggregation_each_turn,
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

    def _post_process_solution(self, solution: Solution) -> Solution:
        """Post-process a solution by extracting the detailed solution section if needed.

        This follows the pattern from IMO25-reflection where only the "Detailed Solution"
        section is used for evaluation and further processing, while the full solution
        (including Summary) is preserved in metadata.

        Args:
            solution: The solution to post-process

        Returns:
            Solution with extracted content if applicable
        """
        if not self.extract_detailed:
            return solution

        # Only extract for math domain and if solution has structured format
        if self.task_domain == "math" and has_structured_format(solution.content):
            detailed = extract_detailed_solution(solution.content)

            # Only replace if extraction actually found something
            if detailed and detailed != solution.content:
                return Solution(
                    content=detailed,
                    score=solution.score,
                    feedback=solution.feedback,
                    metadata={
                        **solution.metadata,
                        "full_solution": solution.content,
                        "extracted_detailed": True,
                    }
                )

        return Solution(
            content=solution.content,
            score=solution.score,
            feedback=solution.feedback,
            metadata={
                **solution.metadata,
                "extracted_detailed": False,
            }
        )

    async def _generate_initial_solutions(
        self, problem: str, n: int, temperature: float, task_domain: str = "general",
        reasoning_effort: Optional[str] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> List[Solution]:
        """Generate initial solutions for the problem.

        Args:
            problem: Problem statement
            n: Number of solutions to generate
            temperature: Sampling temperature
            task_domain: Task domain for template selection
            reasoning_effort: Reasoning effort ("auto", "low", "medium", "high", or None)

        Returns:
            List of initial solutions
        """
        # print(problem)

        print("Initial solution:", "Problem:", [problem[:100]], flush=True)

        if task_domain == "professional_reasoning":
            # Input is a conversation in PRBench. Direct generate next response
            import json
            convos = json.loads(problem)
            messages = [Message(role=turn["role"], content=turn["content"]) for turn in convos]
            print("[INFO] feeding the conversation as input in PRBench")
        else:
            # Try to use prompt template if available
            user_content = problem
            system_content = None

            if self.prompt_manager:
                template: Optional[PromptTemplate] = None

                # Prefer explicit template override when configured.
                if self.direct_prompt_template_name:
                    override_template = self.prompt_manager.get_template(
                        self.direct_prompt_template_name
                    )
                    if (
                        override_template
                        and override_template.prompt_type.value == "direct_generation"
                    ):
                        template = override_template

                # Fallback: select the first direct-generation template for the task domain.
                if template is None:
                    templates = self.prompt_manager.get_templates_by_domain(task_domain)
                    direct_templates = [
                        t for t in templates if t.prompt_type.value == "direct_generation"
                    ]
                    if direct_templates:
                        template = direct_templates[0]

                if template is not None:
                    formatted = template.format_with_system(problem=problem)
                    system_content = formatted["system"]
                    user_content = formatted["user"]

            # Build messages
            messages = []
            if system_content:
                messages.append(Message(role="system", content=system_content))
            messages.append(Message(role="user", content=user_content))

        # Build generation kwargs with reasoning effort
        gen_kwargs = {"temperature": temperature}

        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

        if reasoning_effort:
            # Determine actual effort based on model and config
            if reasoning_effort == "auto":
                effort = "high" if "gpt-oss" in self.llm_service.model_name.lower() else None
            elif reasoning_effort in ["low", "medium", "high"]:
                effort = reasoning_effort
            else:
                effort = None

            if effort:
                gen_kwargs["extra_body"] = {"reasoning_effort": effort}

        tasks = [
            self.llm_service.generate(messages, **gen_kwargs) for _ in range(n)
        ]
        responses = await asyncio.gather(*tasks)

        solutions = [
            Solution(
                content=resp.content,
                metadata={
                    "model": resp.model,
                    "usage": resp.usage,
                    "iteration": 0,
                    "reasoning_content": resp.reasoning_content,
                },
            )
            for resp in responses
        ]

        # Post-process solutions to extract detailed sections if needed
        return [self._post_process_solution(sol) for sol in solutions]

    async def _run_reflection_then_aggregation(
        self,
        problem: str,
        solutions: List[Solution],
        n_iterations: int,
        all_solutions: List[Solution],
        **kwargs: Any,
    ) -> Solution:
        """Run reflection iterations, then aggregate once at the end."""
        current_solutions = solutions

        # print(problem ,flush=True)

        # Apply reflection for n_iterations
        for iteration in range(1, n_iterations + 1):
            print("Iteration:", iteration, "Problem:", [problem[:100]], flush=True)
            if self.reflection_strategy:
                # print(">>>>> Current Solution >>>>>\n\n", current_solutions[0].content, flush=True)
                current_solutions = await self.reflection_strategy.reflect(
                    problem, current_solutions, **kwargs
                )
                # Post-process reflected solutions
                current_solutions = [self._post_process_solution(sol) for sol in current_solutions]

                for sol in current_solutions:
                    sol.metadata["iteration"] = iteration
                all_solutions.extend(current_solutions)

        # Aggregate once at the end
        if self.aggregation_strategy:
            final_solution = await self.aggregation_strategy.aggregate(
                problem, current_solutions, **kwargs
            )
        else:
            final_solution = current_solutions[0] if current_solutions else solutions[0]

        return final_solution

    async def _run_with_aggregation_each_turn(
        self,
        problem: str,
        solutions: List[Solution],
        n_iterations: int,
        all_solutions: List[Solution],
        n_samples_per_iteration: int,
        **kwargs: Any,
    ) -> Solution:
        """Apply aggregation at each reflection turn.

        Args:
            problem: Problem statement
            solutions: Initial solutions
            n_iterations: Number of reflection iterations
            all_solutions: List to accumulate all solutions
            n_samples_per_iteration: Number of solutions to generate at each iteration
            **kwargs: Additional parameters

        Returns:
            Final aggregated solution
        """
        current_solutions = solutions

        for iteration in range(1, n_iterations + 1):
            # Aggregate current solutions to select the best
            if self.aggregation_strategy and len(current_solutions) > 1:
                best_solution = await self.aggregation_strategy.aggregate(
                    problem, current_solutions, **kwargs
                )
            else:
                best_solution = current_solutions[0] if current_solutions else solutions[0]

            # Apply reflection to generate n_samples_per_iteration new solutions
            # from the best solution
            if self.reflection_strategy:
                # Create n_samples_per_iteration copies of the best solution to reflect on
                solutions_to_reflect = [best_solution] * n_samples_per_iteration

                # Pass n_samples hint for strategies that support sampling
                kwargs_with_n = {**kwargs, "n_samples": n_samples_per_iteration}

                current_solutions = await self.reflection_strategy.reflect(
                    problem, solutions_to_reflect, **kwargs_with_n
                )

                # Post-process reflected solutions
                current_solutions = [self._post_process_solution(sol) for sol in current_solutions]

                for sol in current_solutions:
                    sol.metadata["iteration"] = iteration
                all_solutions.extend(current_solutions)
            else:
                # If no reflection strategy, just keep the best solution
                current_solutions = [best_solution]

        # Final aggregation if needed
        if self.aggregation_strategy and len(current_solutions) > 1:
            final_solution = await self.aggregation_strategy.aggregate(
                problem, current_solutions, **kwargs
            )
        else:
            final_solution = current_solutions[0] if current_solutions else solutions[0]

        return final_solution


class PipelineFactory:
    """Factory for creating test-time scaling pipelines."""

    def __init__(self, llm_service: LLMService, prompt_manager: PromptManager):
        """Initialize the pipeline factory.

        Args:
            llm_service: LLM service for generation
            prompt_manager: Prompt manager for accessing templates
        """
        self.llm_service = llm_service
        self.prompt_manager = prompt_manager

    def _get_prompt_template(self, template_name: str) -> Optional[PromptTemplate]:
        """Get prompt template from manager.

        Args:
            template_name: Name of template to look up

        Returns:
            PromptTemplate object or None if not found
        """
        return self.prompt_manager.get_template(template_name)

    def create_reflection_strategy(
        self, strategy_name: str, task_domain: str = "general", **kwargs: Any
    ) -> Optional[ReflectionStrategy]:
        """Create a reflection strategy by name.

        Args:
            strategy_name: Name of the reflection strategy
            task_domain: Task domain (math, coding, etc.) for selecting appropriate templates
            **kwargs: Additional parameters (evaluator, code_executor, etc.)

        Returns:
            ReflectionStrategy instance or None
        """
        from .reflection import (
            CodeExecutionReflection,
            GroundTruthReflection,
            GroundTruthSimpleReflection,
            NoFeedbackReflection,
            SelfEvaluationReflection,
        )

        if strategy_name == "no_feedback":
            # Use direct generation template
            prompt_template = kwargs.get("prompt_template")
            if not prompt_template:
                template_name = f"{task_domain}_no_feedback_reflection"
                prompt_template = self._get_prompt_template(template_name)
            temperature = kwargs.get("temperature", 1.0)
            reasoning_effort = kwargs.get("reasoning_effort", "auto")
            return NoFeedbackReflection(
                self.llm_service, prompt_template, temperature, reasoning_effort
            )

        elif strategy_name == "self_evaluation":
            # Use self-evaluation and reflection templates
            eval_template = kwargs.get("eval_template")
            if not eval_template:
                template_name = f"{task_domain}_self_evaluation"
                eval_template = self._get_prompt_template(template_name)

            improve_template = kwargs.get("improve_template")
            if not improve_template:
                template_name = f"{task_domain}_generation_with_reflection"
                improve_template = self._get_prompt_template(template_name)

            eval_temperature = kwargs.get("eval_temperature", 0.0)
            improve_temperature = kwargs.get("improve_temperature", 0.7)
            reasoning_effort = kwargs.get("reasoning_effort", "auto")
            return SelfEvaluationReflection(
                self.llm_service,
                eval_template,
                improve_template,
                eval_temperature,
                improve_temperature,
                reasoning_effort,
            )

        elif strategy_name == "ground_truth":
            evaluator = kwargs.get("evaluator")
            if evaluator is None:
                raise ValueError("Evaluator required for ground_truth strategy")

            improve_template = kwargs.get("improve_template")
            if not improve_template:
                template_name = f"{task_domain}_generation_with_reflection"
                improve_template = self._get_prompt_template(template_name)

            temperature = kwargs.get("temperature", 0.7)
            reasoning_effort = kwargs.get("reasoning_effort", "auto")
            return GroundTruthReflection(
                self.llm_service, evaluator, improve_template, temperature, reasoning_effort
            )

        elif strategy_name == "ground_truth_simple":
            evaluator = kwargs.get("evaluator")
            if evaluator is None:
                raise ValueError("Evaluator required for ground_truth strategy")

            improve_template = kwargs.get("improve_template")
            if not improve_template:
                template_name = f"{task_domain}_generation_with_reflection"
                improve_template = self._get_prompt_template(template_name)

            temperature = kwargs.get("temperature", 0.7)
            reasoning_effort = kwargs.get("reasoning_effort", "auto")
            return GroundTruthSimpleReflection(
                self.llm_service, evaluator, improve_template, temperature, reasoning_effort
            )

        elif strategy_name == "code_execution":
            evaluator = kwargs.get("evaluator")
            if evaluator is None:
                raise ValueError("Evaluator required for code_execution strategy")

            improve_template = kwargs.get("improve_template")
            if not improve_template:
                template_name = f"{task_domain}_code_execution"
                improve_template = self._get_prompt_template(template_name)

            temperature = kwargs.get("temperature", 0.7)
            use_detailed_results = kwargs.get("use_detailed_results", True)
            reasoning_effort = kwargs.get("reasoning_effort", "auto")
            return CodeExecutionReflection(
                self.llm_service, evaluator, improve_template, temperature, use_detailed_results, reasoning_effort
            )

        elif strategy_name is None or strategy_name == "none":
            return None

        else:
            raise ValueError(f"Unknown reflection strategy: {strategy_name}")

    def create_aggregation_strategy(
        self, strategy_name: str, task_domain: str = "general", **kwargs: Any
    ) -> Optional[AggregationStrategy]:
        """Create an aggregation strategy by name.

        Args:
            strategy_name: Name of the aggregation strategy
            task_domain: Task domain for selecting appropriate templates
            **kwargs: Additional parameters

        Returns:
            AggregationStrategy instance or None
        """
        from .aggregation import (
            GenerateFromNAggregation,
            LLMScoringAggregation,
            LLMVotingAggregation,
            PairwiseComparisonAggregation,
            SelectBestAggregation,
            VotingAggregation,
        )

        if strategy_name == "select_best":
            selection_template = kwargs.get("selection_template")
            if not selection_template:
                template_name = "aggregation_select_one_from_n"
                selection_template = self._get_prompt_template(template_name)
            temperature = kwargs.get("temperature", 0.0)
            reasoning_effort = kwargs.get("reasoning_effort", "auto")
            return SelectBestAggregation(
                self.llm_service, selection_template, temperature, reasoning_effort
            )

        elif strategy_name == "generate_from_n":
            generation_template = kwargs.get("generation_template")
            if not generation_template:
                template_name = "aggregation_generate_one_from_n"
                generation_template = self._get_prompt_template(template_name)
            temperature = kwargs.get("temperature", 0.7)
            reasoning_effort = kwargs.get("reasoning_effort", "auto")
            return GenerateFromNAggregation(
                self.llm_service, generation_template, temperature, reasoning_effort
            )

        elif strategy_name == "llm_scoring":
            scoring_template = kwargs.get("scoring_template")
            if not scoring_template:
                template_name = f"{task_domain}_llm_scoring"
                scoring_template = self._get_prompt_template(template_name)
            temperature = kwargs.get("temperature", 0.0)
            reasoning_effort = kwargs.get("reasoning_effort", "auto")
            return LLMScoringAggregation(
                self.llm_service, scoring_template, temperature, reasoning_effort
            )

        elif strategy_name == "voting":
            answer_extractor = kwargs.get("answer_extractor")
            return VotingAggregation(answer_extractor)

        elif strategy_name == "pairwise_comparison":
            comparison_template = kwargs.get("comparison_template")
            if not comparison_template:
                template_name = "aggregation_pairwise_comparison"
                comparison_template = self._get_prompt_template(template_name)
            temperature = kwargs.get("temperature", 0.0)
            reasoning_effort = kwargs.get("reasoning_effort", "auto")
            return PairwiseComparisonAggregation(
                self.llm_service, comparison_template, temperature, reasoning_effort
            )

        elif strategy_name == "llm_voting":
            voting_template = kwargs.get("voting_template")
            if not voting_template:
                template_name = "aggregation_llm_voting"
                voting_template = self._get_prompt_template(template_name)
            temperature = kwargs.get("temperature", 0.0)
            reasoning_effort = kwargs.get("reasoning_effort", "auto")
            return LLMVotingAggregation(
                self.llm_service, voting_template, temperature, reasoning_effort
            )

        elif strategy_name is None or strategy_name == "none":
            return None

        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy_name}")

    def create_pipeline(
        self,
        reflection_strategy: str = "self_evaluation",
        aggregation_strategy: str = "select_best",
        apply_aggregation_each_turn: bool = False,
        task_domain: str = "general",
        extract_detailed: bool = True,
        **kwargs: Any,
    ) -> ScalingPipeline:
        """Create a complete scaling pipeline.

        Args:
            reflection_strategy: Name of reflection strategy
            aggregation_strategy: Name of aggregation strategy
            apply_aggregation_each_turn: Whether to aggregate at each turn
            task_domain: Task domain (math, coding, etc.) for template selection
            extract_detailed: Whether to extract "Detailed Solution" section (default: True)
            **kwargs: Additional parameters for strategies

        Returns:
            ScalingPipeline instance
        """
        reflection = self.create_reflection_strategy(
            reflection_strategy, task_domain=task_domain, **kwargs
        )
        aggregation = self.create_aggregation_strategy(
            aggregation_strategy, task_domain=task_domain, **kwargs
        )

        return ScalingPipeline(
            self.llm_service,
            reflection,
            aggregation,
            self.prompt_manager,
            apply_aggregation_each_turn,
            extract_detailed,
            task_domain,
        )

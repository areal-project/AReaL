"""Ground truth evaluation reflection strategy."""

import asyncio
from typing import Any, List, Optional

from ...evaluation import Evaluator
from ...llm_service import LLMService, Message
from ...prompts import PromptTemplate
from ..base import Solution
from .base import ReflectionStrategy


class GroundTruthSimpleReflection(ReflectionStrategy):
    """Reflection strategy using ground truth evaluation."""

    def __init__(
        self,
        llm_service: LLMService,
        evaluator: Evaluator,
        improvement_prompt_template: Optional[PromptTemplate],
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = "auto",
    ):
        """Initialize ground truth reflection.

        Args:
            llm_service: LLM service for generation
            evaluator: Evaluator with ground truth
            improvement_prompt_template: PromptTemplate for generating improved solutions
            temperature: Temperature for improvement generation (default: 0.7)
            reasoning_effort: Reasoning effort ("auto", "low", "medium", "high", or None)
        """
        super().__init__(llm_service, temperature=temperature, reasoning_effort=reasoning_effort)
        self.evaluator = evaluator
        self.improvement_prompt_template = improvement_prompt_template

    async def reflect(
        self, problem: str, solutions: List[Solution], **kwargs: Any
    ) -> List[Solution]:
        """Apply ground truth evaluation to improve solutions."""
        ground_truth = kwargs.get("ground_truth")
        if ground_truth is None:
            raise ValueError("Ground truth is required for GroundTruthReflection")

        tasks = [
            self._reflect_single(problem, solution, ground_truth) for solution in solutions
        ]
        return await asyncio.gather(*tasks)

    async def _check_correctness_with_llm(
        self, problem: str, solution: str, ground_truth: str
    ) -> str:
        """Use LLM to check if solution is correct based on ground truth.
        
        Returns only "correct" or "incorrect" to avoid leaking the answer.
        
        Args:
            problem: Problem statement
            solution: Solution to check
            ground_truth: Ground truth answer
            
        Returns:
            "correct" or "incorrect"
        """
        # Create a simple prompt that only asks for correctness judgment
        check_prompt = f"""You are evaluating a solution to a problem. Compare the solution with the ground truth answer and determine if the solution is correct.

Problem:
{problem}

Solution:
{solution}

Ground Truth Answer:
{ground_truth}

Based on your comparison, respond with ONLY one word: "correct" if the solution is correct, or "incorrect" if the solution is incorrect. Do not provide any additional explanation or feedback."""

        check_messages = [Message(role="user", content=check_prompt)]
        
        # Use low temperature for deterministic correctness judgment
        check_kwargs = self._build_generation_kwargs(temperature=0.0)
        check_response = await self.llm_service.generate(check_messages, **check_kwargs)
        
        # Extract and normalize the response
        response_text = check_response.content.strip().lower()
        if "correct" in response_text and "incorrect" not in response_text:
            return "correct"
        elif "incorrect" in response_text:
            return "incorrect"
        else:
            # Fallback: try to determine from the response
            # If it's unclear, default to "incorrect" to be safe
            return "incorrect"

    async def _reflect_single(
        self, problem: str, solution: Solution, ground_truth: str
    ) -> Solution:
        """Reflect on a single solution using ground truth."""
        eval_result = await self.evaluator.evaluate(problem, solution.content, ground_truth)
        
        # Use LLM to check correctness without leaking detailed feedback
        correctness = await self._check_correctness_with_llm(
            problem, solution.content, ground_truth
        )
        
        # Create a simple feedback message that only indicates correctness
        simple_feedback = f"The solution is {correctness}."

        print("simple feedback: ", simple_feedback)
        if self.improvement_prompt_template:
            improve_formatted = self.improvement_prompt_template.format_with_system(
                problem=problem,
                solution=solution.content,
                feedback=simple_feedback,  # Use simple feedback instead of detailed eval_result.feedback
                ground_truth=ground_truth,
            )
            improve_messages = [
                Message(role="system", content=improve_formatted["system"]),
                Message(role="user", content=improve_formatted["user"])
            ]
            # print("system message", improve_formatted["system"][:1000])
            # print("user message", improve_formatted["user"][:1000])
        else:
            # Fallback
            improve_messages = [Message(role="user", content=f"Problem: {problem}\nSolution: {solution.content}\nFeedback: {simple_feedback}\nGround Truth: {ground_truth}\nImprove:")]

        # Build kwargs with reasoning effort
        improve_kwargs = self._build_generation_kwargs()
        improved_response = await self.llm_service.generate(improve_messages, **improve_kwargs)

        return Solution(
            content=improved_response.content,
            score=eval_result.score,
            feedback=simple_feedback,  # Store simple feedback instead of detailed feedback
            metadata={
                "strategy": "ground_truth",
                "original_solution": solution.content,
                "original_score": eval_result.score,
                "correctness": correctness,  # Store the correctness judgment
                "model": improved_response.model,
                "reasoning_content": improved_response.reasoning_content,
                "usage": improved_response.usage,  # Store usage including timing
            },
        )

    def get_strategy_name(self) -> str:
        """Get the name of the reflection strategy."""
        return "ground_truth_simple"

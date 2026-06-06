"""Ground truth evaluation reflection strategy."""

import asyncio
from typing import Any, List, Optional

from ...evaluation import Evaluator
from ...llm_service import LLMService, Message
from ...prompts import PromptTemplate
from ..base import Solution
from .base import ReflectionStrategy


class GroundTruthReflection(ReflectionStrategy):
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

    async def _reflect_single(
        self, problem: str, solution: Solution, ground_truth: str
    ) -> Solution:
        """Reflect on a single solution using ground truth."""
        eval_result = await self.evaluator.evaluate(problem, solution.content, ground_truth)

        if self.improvement_prompt_template:
            improve_formatted = self.improvement_prompt_template.format_with_system(
                problem=problem,
                solution=solution.content,
                feedback=eval_result.feedback,
                ground_truth=ground_truth,
            )
            improve_messages = [
                Message(role="system", content=improve_formatted["system"]),
                Message(role="user", content=improve_formatted["user"])
            ]
        else:
            # Fallback
            improve_messages = [Message(role="user", content=f"Problem: {problem}\nSolution: {solution.content}\nFeedback: {eval_result.feedback}\nGround Truth: {ground_truth}\nImprove:")]

        # Build kwargs with reasoning effort
        improve_kwargs = self._build_generation_kwargs()
        improved_response = await self.llm_service.generate(improve_messages, **improve_kwargs)

        return Solution(
            content=improved_response.content,
            score=eval_result.score,
            feedback=eval_result.feedback,
            metadata={
                "strategy": "ground_truth",
                "original_solution": solution.content,
                "original_score": eval_result.score,
                "model": improved_response.model,
                "reasoning_content": improved_response.reasoning_content,
                "usage": improved_response.usage,  # Store usage including timing
            },
        )

    def get_strategy_name(self) -> str:
        """Get the name of the reflection strategy."""
        return "ground_truth"

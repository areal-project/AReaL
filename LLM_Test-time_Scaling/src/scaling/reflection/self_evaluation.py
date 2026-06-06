"""Self-evaluation reflection strategy."""

import asyncio
from typing import Any, List, Optional

from ...llm_service import LLMService, Message
from ...prompts import PromptTemplate
from ..base import Solution
from ..utils import extract_detailed_solution
from .base import ReflectionStrategy


class SelfEvaluationReflection(ReflectionStrategy):
    """Reflection strategy using self-evaluation with conditional refinement.

    Follows the workflow from IMO25-reflection reference implementation:
    1. Evaluate the solution
    2. Check if evaluation indicates errors
    3. If errors found, use correction prompt to refine
    4. If no errors, return original solution
    """

    def __init__(
        self,
        llm_service: LLMService,
        evaluation_prompt_template: Optional[PromptTemplate],
        improvement_prompt_template: Optional[PromptTemplate],
        eval_temperature: float = 0.0,
        improve_temperature: float = 0.7,
        reasoning_effort: Optional[str] = "auto",
    ):
        """Initialize self-evaluation reflection.

        Args:
            llm_service: LLM service for generation
            evaluation_prompt_template: PromptTemplate for self-evaluation
            improvement_prompt_template: PromptTemplate for correction prompt
                Should include placeholders for: {problem}, {solution}, {feedback}
            eval_temperature: Temperature for evaluation (default: 0.0 for deterministic)
            improve_temperature: Temperature for improvement generation (default: 0.7)
            reasoning_effort: Reasoning effort ("auto", "low", "medium", "high", or None)
        """
        super().__init__(llm_service, temperature=improve_temperature, reasoning_effort=reasoning_effort)
        self.evaluation_prompt_template = evaluation_prompt_template
        self.improvement_prompt_template = improvement_prompt_template
        self.eval_temperature = eval_temperature
        self.improve_temperature = improve_temperature

    async def reflect(
        self, problem: str, solutions: List[Solution], **kwargs: Any
    ) -> List[Solution]:
        """Apply self-evaluation to improve solutions."""
        tasks = [self._reflect_single(problem, solution) for solution in solutions]
        return await asyncio.gather(*tasks)

    async def _reflect_single(self, problem: str, solution: Solution) -> Solution:
        """Reflect on a single solution.

        Workflow (matches agent_oai.py):
        1. Evaluate the solution to identify errors
        2. Check if the evaluation indicates errors/issues
        3. If errors found: Extract bug report and use correction prompt
        4. If no errors: Return original solution unchanged
        """
        # Step 1: Evaluate the solution
        if self.evaluation_prompt_template:
            eval_formatted = self.evaluation_prompt_template.format_with_system(
                problem=problem, solution=solution.content
            )
            eval_messages = [
                Message(role="system", content=eval_formatted["system"]),
                Message(role="user", content=eval_formatted["user"])
            ]
        else:
            # Fallback
            eval_messages = [Message(role="user", content=f"Problem: {problem}\n\nCurrent Solution:\n{solution.content}\n\nEvaluate this solution:")]

        # Build kwargs for evaluation with reasoning effort
        eval_kwargs = self._build_generation_kwargs(temperature=self.eval_temperature)
        try:
            eval_response = await self.llm_service.generate(eval_messages, **eval_kwargs)
        except ValueError as e:
            # If input tokens exceed limit, skip reflection and return original solution
            # This error is raised in litellm_service.py when input_tokens > 120000
            print(f"  Warning: Input tokens exceeded limit during evaluation, skipping reflection")
            return Solution(
                content=solution.content,
                feedback="[Reflection skipped due to input token limit]",
                metadata={
                    "strategy": "self_evaluation",
                    "original_solution": solution.content,
                    "model": solution.metadata.get("model", "unknown"),
                    "has_errors": None,
                    "was_refined": False,
                    "reflection_skipped": True,
                    "skip_reason": "input_tokens_exceeded",
                },
            )

        # Step 2: Check if evaluation indicates errors
        # Note: _check_if_has_errors handles ValueError internally and returns True if error occurs
        has_errors = await self._check_if_has_errors(eval_response.content)

        # Step 3: Conditionally refine using correction prompt with bug report
        if has_errors:
            # Extract bug report: the part before "Detailed Verification Log"
            # This contains the Summary and List of Findings
            # Matches agent_oai.py line 359: extract_detailed_solution(out, "Detailed Verification", False)
            bug_report = extract_detailed_solution(
                eval_response.content,
                marker="Detailed Verification",
                after=False
            )

            # If extraction didn't find the marker, use full evaluation as bug report
            if not bug_report or bug_report == eval_response.content:
                bug_report = eval_response.content

            # Truncate bug_report if it's too long to prevent token limit issues
            # LLM evaluation can sometimes generate very long repetitive content
            MAX_BUG_REPORT_LENGTH = 10000  # Maximum characters for bug report
            if len(bug_report) > MAX_BUG_REPORT_LENGTH:
                bug_report = bug_report[:MAX_BUG_REPORT_LENGTH]
                # Add truncation indicator
                bug_report += "\n\n[Note: Bug report was truncated due to length]"

            # Build correction prompt with bug report
            if self.improvement_prompt_template:
                correction_formatted = self.improvement_prompt_template.format_with_system(
                    problem=problem,
                    solution=solution.content,
                    feedback=bug_report
                )
                correction_messages = [
                    Message(role="system", content=correction_formatted["system"]),
                    Message(role="user", content=correction_formatted["user"])
                ]
            else:
                # Fallback
                correction_messages = [Message(role="user", content=f"Problem: {problem}\nSolution: {solution.content}\nFeedback: {bug_report}\nImprove:")]

            # Build kwargs for improvement with reasoning effort
            improve_kwargs = self._build_generation_kwargs(temperature=self.improve_temperature)
            try:
                improved_response = await self.llm_service.generate(correction_messages, **improve_kwargs)
            except ValueError as e:
                # If input tokens exceed limit during improvement, return original solution
                print(f"  Warning: Input tokens exceeded limit during improvement, returning original solution")
                return Solution(
                    content=solution.content,
                    feedback=bug_report,
                    metadata={
                        "strategy": "self_evaluation",
                        "original_solution": solution.content,
                        "model": solution.metadata.get("model", "unknown"),
                        "has_errors": True,
                        "was_refined": False,
                        "full_evaluation": eval_response.content,
                        "eval_usage": eval_response.usage,
                        "eval_reasoning_content": eval_response.reasoning_content,
                        "improvement_skipped": True,
                        "skip_reason": "input_tokens_exceeded",
                    },
                )

            return Solution(
                content=improved_response.content,
                feedback=bug_report,
                metadata={
                    "strategy": "self_evaluation",
                    "original_solution": solution.content,
                    "model": improved_response.model,
                    "has_errors": True,
                    "was_refined": True,
                    "full_evaluation": eval_response.content,
                    "improve_usage": improved_response.usage,
                    "eval_usage": eval_response.usage,
                    "reasoning_content": improved_response.reasoning_content,
                    "eval_reasoning_content": eval_response.reasoning_content,
                    "eval_content": eval_response.content,
                },
            )
        else:
            # No errors found, return original solution
            return Solution(
                content=solution.content,
                feedback=eval_response.content,
                metadata={
                    "strategy": "self_evaluation",
                    "original_solution": solution.content,
                    "model": solution.metadata.get("model", "unknown"),
                    "has_errors": False,
                    "was_refined": False,
                    "eval_usage": eval_response.usage,  # Store eval usage including timing
                    "eval_reasoning_content": eval_response.reasoning_content,
                    "eval_content": eval_response.content,
                },
            )

    async def _check_if_has_errors(self, evaluation_text: str) -> bool:
        """Check if the evaluation indicates errors or issues.

        Uses LLM to parse the evaluation and determine if it indicates
        the solution is correct or contains errors/gaps.

        Matches agent_oai.py lines 346-358 logic.

        Args:
            evaluation_text: The evaluation feedback text

        Returns:
            True if errors/issues found, False if solution is correct
        """
        check_prompt = (
            'Response in "yes" or "no". Is the following statement saying '
            'the solution is correct, or does not contain critical error '
            'or a major justification gap?\n\n'
            f'{evaluation_text}'
        )

        check_messages = [Message(role="user", content=check_prompt)]

        # Use low temperature (0.0) for deterministic consistency, with reasoning effort
        check_kwargs = self._build_generation_kwargs(temperature=0.0)
        try:
            check_response = await self.llm_service.generate(check_messages, **check_kwargs)
            response_lower = check_response.content.lower().strip()

            # If response says "yes" (solution is correct), then has_errors = False
            # If response says "no" (solution has issues), then has_errors = True
            # Matches: if("yes" not in good_verify.lower()) from agent_oai.py
            return "yes" not in response_lower
        except ValueError as e:
            # If input tokens exceed limit during error check, default to assuming errors exist
            # This is conservative: if we can't check, assume there might be errors
            print(f"  Warning: Input tokens exceeded limit during error check, defaulting to has_errors=True")
            return True

    def get_strategy_name(self) -> str:
        """Get the name of the reflection strategy."""
        return "self_evaluation"

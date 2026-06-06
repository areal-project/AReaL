"""Code execution reflection strategy using evaluator test results."""

import asyncio
from typing import Any, List, Optional

from ...evaluation import Evaluator
from ...llm_service import LLMService, Message
from ...prompts import PromptTemplate
from ..base import Solution
from .base import ReflectionStrategy
from .test_result_giver import TestResultGiver


class CodeExecutionReflection(ReflectionStrategy):
    """Reflection strategy using code execution feedback from evaluator.
    
    Similar to self-evaluation reflection but uses actual test case results
    from an evaluator (e.g., RemoteLCBProEvaluator) instead of LLM self-evaluation.
    """

    def __init__(
        self,
        llm_service: LLMService,
        evaluator: Evaluator,
        improvement_prompt_template: Optional[PromptTemplate],
        temperature: float = 0.7,
        use_detailed_results: bool = True,
        reasoning_effort: Optional[str] = "auto",
    ):
        """Initialize code execution reflection.

        Args:
            llm_service: LLM service for generation
            evaluator: Evaluator for running test cases (e.g., RemoteLCBProEvaluator)
            improvement_prompt_template: PromptTemplate for generating improved solutions
                Should include placeholders for: {problem}, {solution}, {feedback}
                The feedback will contain the formatted test result summary from TestResultGiver
            temperature: Temperature for improvement generation (default: 0.7)
            use_detailed_results: If True, use detailed test results (with input/output/answer).
                If False, use basic results (only passed count and error type)
            reasoning_effort: Reasoning effort ("auto", "low", "medium", "high", or None)
        """
        super().__init__(llm_service, temperature=temperature, reasoning_effort=reasoning_effort)
        self.evaluator = evaluator
        self.improvement_prompt_template = improvement_prompt_template
        self.use_detailed_results = use_detailed_results

    async def reflect(
        self, problem: str, solutions: List[Solution], **kwargs: Any
    ) -> List[Solution]:
        """Apply code execution feedback to improve solutions."""
        tasks = [self._reflect_single(problem, solution, **kwargs) for solution in solutions]
        return await asyncio.gather(*tasks)

    async def _reflect_single(
        self, problem: str, solution: Solution, **kwargs: Any
    ) -> Solution:
        """Reflect on a single solution using code execution test results.
        
        Workflow:
        1. Check if solution is already correct (skip evaluation if so)
        2. Evaluate the solution using the evaluator
        3. Extract test results using TestResultGiver
        4. If errors found, use improvement prompt with test results as feedback to refine
        5. If all passed, return original solution
        """
        # Step 0: Check if solution is already correct (skip evaluation)
        # This optimization avoids re-evaluating solutions that have already passed all tests
        metadata = solution.metadata or {}
        test_results = metadata.get("test_results", {})
        
        # Check 1: test_results in metadata (most reliable)
        is_already_correct = False
        if test_results:
            passed = test_results.get("passed", 0)
            total = test_results.get("total", 0)
            if total > 0 and passed == total:
                is_already_correct = True
        
        # Check 2: solution.score and eval_details (if score is 1.0, likely correct)
        if not is_already_correct and solution.score is not None:
            if solution.score >= 1.0:
                # Check if we have evaluation details to confirm
                eval_details = metadata.get("eval_details")
                if eval_details and isinstance(eval_details, dict):
                    eval_passed = eval_details.get("passed", 0)
                    eval_total = eval_details.get("total", 0)
                    if eval_total > 0 and eval_passed == eval_total:
                        is_already_correct = True
        
        # Check 3: explicit early_success_marked flag (from result processing)
        if not is_already_correct:
            # Check if solution was marked as correct in a previous iteration
            # This can happen when we process results and mark early success
            if metadata.get("early_success_marked", False):
                is_already_correct = True
        
        # Check 4: Check if solution content matches a previously correct solution
        # (This handles cases where the same solution is passed through multiple iterations)
        if not is_already_correct:
            original_solution = metadata.get("original_solution")
            # If this solution's content matches a previously correct solution's content
            # and we have test_results indicating it was correct, skip evaluation
            if original_solution and solution.content == original_solution:
                if test_results:
                    passed = test_results.get("passed", 0)
                    total = test_results.get("total", 0)
                    if total > 0 and passed == total:
                        is_already_correct = True
        
        if is_already_correct:
            # Solution already passed all tests, reuse the result without re-evaluation
            passed = test_results.get("passed", 0) if test_results else 0
            total = test_results.get("total", 0) if test_results else 0
            print(f"  Solution already correct (passed {passed}/{total}), skipping evaluation", flush=True)
            # Return solution with same evaluation results
            return Solution(
                content=solution.content,
                score=solution.score if solution.score is not None else 1.0,
                feedback=solution.feedback or f"Test Results: {passed}/{total} passed (reused from previous evaluation)",
                metadata={
                    **metadata,
                    "strategy": "code_execution",
                    "was_refined": False,
                    "evaluation_skipped": True,
                    "reused_from_previous": True,
                },
            )
        
        # Step 1: Evaluate the solution
        eval_kwargs = {
            "problem": problem,
            "solution": solution.content,
        }
        # Pass through any additional kwargs (e.g., problem_id, language, test_cases)
        if "ground_truth" in kwargs:
            eval_kwargs["ground_truth"] = kwargs["ground_truth"]
        if "problem_id" in kwargs:
            eval_kwargs["problem_id"] = kwargs["problem_id"]
        if "language" in kwargs:
            eval_kwargs["language"] = kwargs["language"]
        if "test_cases" in kwargs:
            eval_kwargs["test_cases"] = kwargs["test_cases"]
        
        eval_result = await self.evaluator.evaluate(**eval_kwargs)
        
        # Step 2: Extract test results and format as feedback
        if eval_result.details and isinstance(eval_result.details, dict):
            if self.use_detailed_results:
                test_result = TestResultGiver.extract_detailed_result(eval_result.details)
            else:
                test_result = TestResultGiver.extract_basic_result(eval_result.details)
            
            # Use test result summary as feedback directly (no eval model needed)
            test_result_feedback = TestResultGiver.format_test_result_summary(
                test_result, detailed=self.use_detailed_results
            )
            
            # Store full eval_result.details in metadata for later use (contains metadata with first_error_input/output/answer)
            eval_details_for_metadata = eval_result.details
        else:
            # Fallback if details are not available
            test_result = {"passed": 0, "total": 0, "first_error_type": None}
            test_result_feedback = f"Test Results: Evaluation completed\n{eval_result.feedback or 'No details available'}"
            eval_details_for_metadata = None
        
        # Step 3: Check if all tests passed
        passed = test_result.get("passed", 0)
        total = test_result.get("total", 0)
        all_passed = (passed == total) and total > 0
        
        if all_passed or eval_result.is_correct:
            # All tests passed, return original solution
            return Solution(
                content=solution.content,
                score=eval_result.score,
                feedback=test_result_feedback,  # Use test result as feedback
                metadata={
                    "strategy": "code_execution",
                    "original_solution": solution.content,
                    "model": solution.metadata.get("model", "unknown"),
                    "was_refined": False,
                    "test_results": test_result,
                    "eval_details": eval_details_for_metadata,  # Store full evaluation details
                },
            )
        
        # Step 4: Errors found, refine using improvement prompt with test result as feedback
        if self.improvement_prompt_template:
            print("len of test result feedback: ", len(test_result_feedback))
            # print("sample: ", test_result_feedback[:100])
            improve_formatted = self.improvement_prompt_template.format_with_system(
                problem=problem,
                solution=solution.content,
                feedback=test_result_feedback,  # Use test result summary as feedback
            )
            improve_messages = [
                Message(role="system", content=improve_formatted["system"]),
                Message(role="user", content=improve_formatted["user"])
            ]
        else:
            # Fallback
            improve_messages = [
                Message(role="user", content=(
                    f"Problem: {problem}\n\n"
                    f"Current Solution:\n{solution.content}\n\n"
                    f"{test_result_feedback}\n\n"
                    f"Improve the solution to fix the test case failures:"
                ))
            ]
        
        # Build kwargs with reasoning effort
        improve_kwargs = self._build_generation_kwargs()
        improved_response = await self.llm_service.generate(improve_messages, **improve_kwargs)
        
        # print("test result feedback: ", test_result_feedback[:100], flush=True)
        return Solution(
            content=improved_response.content,
            score=eval_result.score,
            feedback=test_result_feedback,  # Use test result as feedback
            metadata={
                "strategy": "code_execution",
                "original_solution": solution.content,
                "original_score": eval_result.score,
                "model": improved_response.model,
                "was_refined": True,
                "test_results": test_result,
                "eval_details": eval_details_for_metadata,  # Store full evaluation details
                "usage": improved_response.usage,  # Store usage including timing
                "reasoning_content": improved_response.reasoning_content,
            },
        )

    def get_strategy_name(self) -> str:
        """Get the name of the reflection strategy."""
        return "code_execution"

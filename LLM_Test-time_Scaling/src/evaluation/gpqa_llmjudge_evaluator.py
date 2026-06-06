"""GPQA Diamond evaluator - rule-based answer matching."""

import asyncio
import re
from typing import Any, List, Optional

from .base import EvaluationResult, Evaluator
from .llm_judge import LLMJudge


class GPQALLMEvaluator(Evaluator):
    """LLM-based evaluator for GPQA Diamond.
    """

    def __init__(self, max_concurrent: int = 100, llm_judge: Optional[LLMJudge] = None):
        """Initialize GPQA evaluator.

        Args:
            max_concurrent: Maximum number of concurrent evaluations
            llm_judge: Optional LLM judge for answer evaluation (if None, uses rule-based extraction)
        """
        self.sem = asyncio.Semaphore(max_concurrent)
        self.llm_judge = llm_judge

    def _extract_answer_from_solution(self, solution: str) -> Optional[str]:
        """Extract answer letter (A, B, C, or D) from model solution.

        Args:
            solution: Model's solution text

        Returns:
            "A", "B", "C", or "D" if found, None otherwise
        """
        solution_upper = solution.upper()

        # Look for explicit answer patterns
        # Pattern 1: "The answer is X" or "Answer: X" or "The correct answer is X"
        patterns = [
            r'(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s+IS\s+([ABCD])',
            r'ANSWER:\s*([ABCD])',
            r'(?:OPTION\s+)?([ABCD])\s+IS\s+CORRECT',
            r'(?:CHOOSE|SELECT)\s+(?:OPTION\s+)?([ABCD])',
            r'^\s*([ABCD])\s*$',  # Just the letter alone
            r'\b([ABCD])\b.*(?:CORRECT|RIGHT)',  # Letter followed by "correct" or "right"
        ]

        for pattern in patterns:
            match = re.search(pattern, solution_upper)
            if match:
                return match.group(1)

        # Pattern 2: Look for answer at the end
        # "Therefore, the answer is A" or ending with "A."
        end_pattern = r'([ABCD])[\.\s]*$'
        match = re.search(end_pattern, solution_upper.strip())
        if match:
            return match.group(1)

        # Pattern 3: If solution contains only one letter A/B/C/D, use that
        letters = re.findall(r'\b([ABCD])\b', solution_upper)
        if len(letters) == 1:
            return letters[0]

        # Pattern 4: Take the last occurrence of A/B/C/D
        if letters:
            return letters[-1]

        return None

    async def _evaluate_single(
        self,
        problem: str,
        solution: str,
        ground_truth: Optional[str] = None,
        **kwargs: Any
    ) -> EvaluationResult:
        """Evaluate a single GPQA solution by checking answer match.

        Args:
            problem: Problem statement (multiple-choice question)
            solution: Model's solution
            ground_truth: Correct answer letter (A, B, C, or D)
            **kwargs: Additional arguments (ignored)

        Returns:
            EvaluationResult with is_correct and score
        """
        if not ground_truth:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback="No ground truth provided",
            )

        try:
            # Normalize ground truth
            expected_answer = ground_truth.strip().upper()
            if expected_answer not in ['A', 'B', 'C', 'D']:
                return EvaluationResult(
                    is_correct=False,
                    score=0.0,
                    feedback=f"Invalid ground truth answer: {ground_truth}",
                )

            # Use LLM judge if available, otherwise use rule-based extraction
            if self.llm_judge is not None:
                # Use LLM judge to evaluate the solution
                result = await self.llm_judge.evaluate(
                    problem=problem,
                    solution=solution,  # Pass full solution for context
                    ground_truth=ground_truth,
                    **kwargs,
                )
                
                # Add extracted answer to details (merge with existing details)
                if result.details is None:
                    result.details = {}
                
                # Also extract answer using rule-based method for comparison
                predicted_answer = self._extract_answer_from_solution(solution)
                result.details.update({
                    "predicted_answer": predicted_answer,
                    "expected_answer": expected_answer,
                    "evaluation_method": "llm_judge",
                })
                
                return result
            else:
                # Rule-based extraction (original behavior)
                predicted_answer = self._extract_answer_from_solution(solution)

                if predicted_answer is None:
                    return EvaluationResult(
                        is_correct=False,
                        score=0.0,
                        feedback="Could not extract answer letter (A/B/C/D) from solution",
                        details={
                            "predicted_answer": None,
                            "expected_answer": expected_answer,
                            "evaluation_method": "rule_based",
                        }
                    )

                # Check if prediction matches ground truth
                is_correct = (predicted_answer == expected_answer)

                feedback = f"Predicted: {predicted_answer}, Expected: {expected_answer}"
                if is_correct:
                    feedback = f"Correct! {feedback}"
                else:
                    feedback = f"Incorrect. {feedback}"

                return EvaluationResult(
                    is_correct=is_correct,
                    score=1.0 if is_correct else 0.0,
                    feedback=feedback,
                    details={
                        "predicted_answer": predicted_answer,
                        "expected_answer": expected_answer,
                        "evaluation_method": "rule_based",
                    }
                )

        except Exception as e:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback=f"Evaluation error: {e}",
            )

    async def evaluate(
        self,
        problem: str,
        solution: str,
        ground_truth: Optional[str] = None,
        **kwargs: Any
    ) -> EvaluationResult:
        """Evaluate a GPQA solution by checking answer match.

        Args:
            problem: Problem statement (multiple-choice question)
            solution: Model's solution
            ground_truth: Correct answer letter (A, B, C, or D)
            **kwargs: Additional arguments (ignored)

        Returns:
            EvaluationResult with is_correct and score
        """
        async with self.sem:
            return await self._evaluate_single(problem, solution, ground_truth, **kwargs)

    async def evaluate_batch(
        self,
        problems: List[str],
        solutions: List[str],
        ground_truths: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[EvaluationResult]:
        """Evaluate a batch of GPQA solutions by checking answer matches.

        Args:
            problems: List of problem statements
            solutions: List of solutions to evaluate
            ground_truths: List of ground truth answer letters
            **kwargs: Additional arguments (ignored)

        Returns:
            List of EvaluationResult objects
        """
        if ground_truths is None:
            ground_truths = [None] * len(problems)

        # Create evaluation tasks
        tasks = [
            self.evaluate(
                problem=problem,
                solution=solution,
                ground_truth=ground_truth,
                **kwargs,
            )
            for problem, solution, ground_truth in zip(problems, solutions, ground_truths)
        ]

        return await asyncio.gather(*tasks)

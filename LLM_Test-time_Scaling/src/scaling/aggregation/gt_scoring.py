"""Ground truth scoring aggregation strategy."""

from typing import Any, List, Optional

from ...evaluation import Evaluator
from ..base import Solution
from .base import AggregationStrategy


class GTScoringAggregation(AggregationStrategy):
    """Aggregation strategy that uses stored evaluation results (pass@k logic).
    
    This strategy directly reads evaluation results (is_correct, score) from the
    solution metadata stored in result files, without re-evaluating. It implements
    pass@k: if any initial solution passed, return the first one that passed;
    otherwise return the first solution.
    """

    def __init__(
        self,
        evaluator: Evaluator = None,  # Not used anymore, kept for backward compatibility
        llm_service: Any = None,  # Not used but required by base class
    ):
        """Initialize ground truth scoring aggregation.

        Args:
            evaluator: Not used anymore, kept for backward compatibility
            llm_service: Not used, but required by base class (can be None)
        """
        # GT scoring doesn't need LLM service, but base class requires it
        # Create a dummy object with model_name attribute to avoid errors
        dummy_llm = object()
        if llm_service is None:
            class DummyLLM:
                model_name = "dummy"
            dummy_llm = DummyLLM()
        else:
            dummy_llm = llm_service
        super().__init__(llm_service=dummy_llm, temperature=0.0)
        self.evaluator = evaluator

    async def aggregate(
        self, problem: str, solutions: List[Solution], ground_truth: Optional[str] = None, **kwargs: Any
    ) -> Solution:
        """Select solution based on stored evaluation results (pass@k).
        
        This method directly uses the evaluation results stored in the solution metadata
        from the original result file, without re-evaluating. It implements pass@k logic:
        - If any initial solution passed (is_correct=True), return the first one that passed
        - Otherwise, return the first solution (all failed)

        Args:
            problem: The problem statement (not used, kept for interface compatibility)
            solutions: List of candidate solutions with stored evaluation results
            ground_truth: Ground truth answer (not used, kept for interface compatibility)
            **kwargs: Additional parameters

        Returns:
            The first solution that passed, or the first solution if none passed
        """
        if len(solutions) == 1:
            solution = solutions[0]
            solution.metadata["aggregation"] = "gt_scoring"
            solution.metadata["n_candidates"] = 1
            solution.metadata["pass_at_k"] = 1 if solution.metadata.get("original_metadata", {}).get("is_correct") or solution.score == 1.0 else 0
            return solution

        # Extract stored evaluation results from solution metadata
        # Check both solution.score and original_metadata for is_correct
        solution_results = []
        for i, sol in enumerate(solutions):
            # Try to get is_correct from multiple sources
            is_correct = None
            
            # 1. Check original_metadata (from result file)
            original_metadata = sol.metadata.get("original_metadata", {})
            if "is_correct" in original_metadata:
                is_correct = original_metadata["is_correct"]
            elif isinstance(original_metadata, dict):
                # Sometimes is_correct might be at top level of metadata
                if "is_correct" in sol.metadata:
                    is_correct = sol.metadata["is_correct"]
            
            # 2. Check solution.score (score == 1.0 typically means correct)
            if is_correct is None and sol.score is not None:
                is_correct = (sol.score >= 1.0)
            
            # 3. Check if score is explicitly stored in original_metadata
            if is_correct is None and "score" in original_metadata:
                score = original_metadata["score"]
                is_correct = (score >= 1.0) if score is not None else False
            
            # Default to False if we can't determine
            if is_correct is None:
                is_correct = False
            
            solution_results.append({
                "index": i,
                "solution": sol,
                "is_correct": is_correct,
                "score": sol.score if sol.score is not None else original_metadata.get("score", 0.0),
            })

        # Find first solution that passed (pass@k logic)
        passed_solutions = [r for r in solution_results if r["is_correct"]]
        
        if passed_solutions:
            # Return first solution that passed
            best_result = passed_solutions[0]
            best_solution = best_result["solution"]
            pass_at_k = 1
        else:
            # No solution passed, return first solution
            best_result = solution_results[0]
            best_solution = best_result["solution"]
            pass_at_k = 0

        # Update metadata
        best_solution.metadata["aggregation"] = "gt_scoring"
        best_solution.metadata["pass_at_k"] = pass_at_k
        best_solution.metadata["n_candidates"] = len(solutions)
        best_solution.metadata["n_passed"] = len(passed_solutions)
        best_solution.metadata["all_results"] = [
            {
                "index": r["index"],
                "is_correct": r["is_correct"],
                "score": r["score"],
            }
            for r in solution_results
        ]

        return best_solution

    def get_strategy_name(self) -> str:
        """Get the name of the aggregation strategy."""
        return "gt_scoring"

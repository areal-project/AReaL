"""Code executor for evaluating code solutions using functioncall."""

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add functioncall to path
_functioncall_path = Path(__file__).parent.parent.parent / "functioncall"
if str(_functioncall_path) not in sys.path:
    sys.path.insert(0, str(_functioncall_path))

try:
    from functioncall.code.function.testing_util import run_test
    _has_functioncall = True
except ImportError:
    _has_functioncall = False
    run_test = None

from .base import EvaluationResult, Evaluator
from .cpp_runner import run_cpp_tests


def extract_code(text: str, min_length: int = 20) -> Optional[str]:
    """Extract code from text by finding code blocks.
    
    Args:
        text: Text that may contain code blocks
        min_length: Minimum length of code block to be considered valid
        
    Returns:
        Extracted code string, or None if no valid code block found
    """
    # Pattern to match code blocks with optional language specifier
    code_pattern = r"(?i)```(?:python|py|cpp|CPP|c\+\+|c)?\s*\n?(.*?)\n?```"
    code_blocks = re.findall(code_pattern, text, re.DOTALL)
    valid_blocks = []
    for block in code_blocks:
        clean_block = block.strip()
        if len(clean_block) < min_length:
            continue
        valid_blocks.append(clean_block)
    
    if not valid_blocks:
        return None
    # Return the last code block (usually the final solution)
    return valid_blocks[-1]


class CodeExecutor(Evaluator):
    """Code executor for evaluating code solutions using functioncall."""

    def __init__(
        self,
        language: str = "python",
        use_local_verify: bool = False,
    ):
        """Initialize the code executor.

        Args:
            language: Programming language (python, javascript, etc.)
            use_local_verify: Whether to use local_verify instead of remote functioncall
        """
        # timeout and memory limit should be different for each problem
        self.language = language
        self.use_local_verify = use_local_verify

    async def evaluate(
        self,
        problem: str,
        problem_id: str,
        solution: str,
        timeout: int = 2000,
        memory_limit: int = 1024,
        input_output: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate a code solution by executing it using functioncall.

        Args:
            problem: Problem statement (not used for execution)
            id: Problem id, use id to get its testcase from storage
            solution: Code solution to execute
            timeout: Timeout in milliseconds
            memory_limit: Memory limit in MB
            input_output: Functioncall format with 'inputs', 'outputs', 'fn_name' (preferred)
            **kwargs: Additional parameters

        Returns:
            EvaluationResult with execution results
        """
        # Extract code from solution if it contains code blocks
        extracted_code = extract_code(solution)
        if extracted_code is not None:
            solution = extracted_code
        # Convert test_cases to input_output format if needed
        if input_output is None and problem_id is not None:
            input_output = self._get_input_output(problem_id)
            
        if input_output is None:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback="No test cases or input_output provided",
                details={"error": "No test cases", "extracted_code": extracted_code},
            )

        try:
            # Check if we need to use C++ local execution
            if self.language.lower() in ["cpp", "c++", "c"]:
                # Use local C++ execution
                loop = asyncio.get_event_loop()
                results, metadata = await loop.run_in_executor(
                    None, self._run_cpp_local, input_output, solution, timeout, memory_limit
                )
            else:
                # Prepare problem dict for functioncall (Python and other languages)
                problem_dict = {
                    "input_output": json.dumps(input_output),
                    "query_id": kwargs.get("query_id", "unknown"),
                }
                
                if self.use_local_verify:
                    # Use local verify (synchronous, run in executor)
                    loop = asyncio.get_event_loop()
                    results, metadata = await loop.run_in_executor(
                        None, self._run_local_verify, problem_dict, solution
                    )
                else:
                    # Use run_test directly (synchronous, run in executor)
                    if not _has_functioncall or run_test is None:
                        raise ImportError(
                            "functioncall not available. Please ensure functioncall is in the LLM_Test-time_Scaling directory."
                        )
                    loop = asyncio.get_event_loop()
                    results, metadata = await loop.run_in_executor(
                        None, run_test, problem_dict, solution, False, self.timeout
                    )

            # Parse results
            if not results:
                return EvaluationResult(
                    is_correct=False,
                    score=0.0,
                    feedback="No results from code execution",
                    details={"error": "No results", "metadata": metadata, "extracted_code": extracted_code},
                )

            # Check if all tests passed
            passed = sum(1 for r in results if r is True)
            total = len(results)
            is_correct = all(r is True for r in results)
            score = passed / total if total > 0 else 0.0

            # Build feedback
            feedback = f"Passed {passed}/{total} test cases"
            if metadata:
                if "error_message" in metadata:
                    feedback += f"\nError: {metadata['error_message']}"
                if "error" in metadata:
                    feedback += f"\nError details: {metadata['error']}"

            details = {
                "passed": passed,
                "total": total,
                "results": results,
                "metadata": metadata,
                "extracted_code": extracted_code,
            }

            return EvaluationResult(
                is_correct=is_correct,
                score=score,
                feedback=feedback,
                details=details,
            )

        except Exception as e:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback=f"Execution error: {str(e)}",
                details={"error": str(e), "exception_type": type(e).__name__},
            )

    def _run_local_verify(self, problem_dict: Dict[str, Any], solution: str) -> tuple:
        """Run local verify (if available)."""
        try:
            from functioncall.code.local_verify import call_verify
            result, metadata = call_verify(problem_dict, solution, False, self.timeout)
            # Convert to list format
            if isinstance(result, list):
                return result, metadata
            else:
                return [result], metadata
        except ImportError:
            # Fallback to run_test
            from functioncall.code.function.testing_util import run_test
            return run_test(problem_dict, solution, False, self.timeout)

    def _run_cpp_local(
        self, input_output: Dict[str, Any], solution: str, timeout: int = 2000, memory_limit: int = 1024
    ) -> tuple:
        """Run C++ code locally by compiling and executing.
        
        Args:
            input_output: Dictionary with 'inputs', 'outputs', 'fn_name'
            solution: C++ code to compile and execute
            timeout: Timeout in milliseconds
            memory_limit: Memory limit in MB
            
        Returns:
            Tuple of (results list, metadata dict)
        """
        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])
        
        # Use the cpp_runner module to handle compilation and execution
        results, metadata = run_cpp_tests(
            solution=solution,
            inputs=inputs,
            outputs=outputs,
            timeout_ms=timeout,
            memory_limit_mb=memory_limit,
            compile_timeout=30,
        )
        
        return results, metadata

    def _convert_test_cases_to_input_output(
        self, test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert legacy test_cases format to functioncall input_output format."""
        inputs = []
        outputs = []

        for test_case in test_cases:
            inputs.append(test_case.get("input", ""))
            outputs.append(test_case.get("expected_output", ""))

        return {
            "inputs": inputs,
            "outputs": outputs,
            "fn_name": "",  # Standard input mode
        }

    async def evaluate_batch(
        self,
        problems: list[str],
        solutions: list[str],
        ground_truths: Optional[list[str]] = None,
        test_cases_batch: Optional[list[List[Dict[str, Any]]]] = None,
        input_output_batch: Optional[list[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> list[EvaluationResult]:
        """Evaluate a batch of code solutions."""
        if input_output_batch is None:
            input_output_batch = [None] * len(solutions)

        tasks = [
            self.evaluate(
                problem,
                solution,
                None,
                test_cases if test_cases_batch else None,
                input_output=input_output,
                query_id=f"batch_{i}",
                **kwargs,
            )
            for i, (problem, solution, test_cases, input_output) in enumerate(
                zip(
                    problems,
                    solutions,
                    test_cases_batch or [None] * len(solutions),
                    input_output_batch,
                )
            )
        ]

        return await asyncio.gather(*tasks)

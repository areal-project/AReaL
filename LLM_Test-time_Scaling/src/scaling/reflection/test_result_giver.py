"""Test result giver for extracting test case information from evaluation results."""

from typing import Any, Dict, Optional, Tuple


class TestResultGiver:
    """Extract and format test case results from evaluation results."""

    @staticmethod
    def extract_basic_result(eval_result_details: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic test result information.
        
        Args:
            eval_result_details: Evaluation result details dict containing:
                - passed: Number of passed test cases
                - total: Total number of test cases
                - results: List of test case results
        
        Returns:
            Dictionary with:
                - passed: Number of passed test cases
                - total: Total number of test cases
                - first_error_type: Type of first error (WA/TLE/MLE/RE/CE) or None if all passed
                - first_error_checker_stdout: Checker stdout for first error test case (if available)
        """
        passed = eval_result_details.get("passed", 0)
        total = eval_result_details.get("total", 0)
        metadata = eval_result_details.get("metadata", None)
        
        first_error_type = None
        first_error_testcase = None
        first_error_checker_stdout = None
        
        if passed < total and metadata:
            if metadata.get("error") == "Compilation Error":
                first_error_type = "CE"
            else:
                # Find first failed test case by checking test metadata
                for idx in range(total):
                    error_type = metadata.get(f"test_{idx}_error_type")
                    error = metadata.get(f"test_{idx}_error")
                    
                    if error_type == "TimeLimitExceeded" or error_type == "Time Limit Exceeded":
                        first_error_type = "TLE"
                        first_error_testcase = idx
                        break
                    elif error_type == "MemoryLimitExceeded" or error_type == "Memory Limit Exceeded":
                        first_error_type = "MLE"
                        first_error_testcase = idx
                        break
                    elif error_type == "Runtime Error" or error_type == "Wrong Answer (Checker)":
                        if error_type == "Runtime Error":
                            first_error_type = "RE"
                        else:
                            first_error_type = "WA"
                        first_error_testcase = idx
                        break
                    elif error is True or metadata.get(f"test_{idx}_output_mismatch") is True:
                        first_error_type = "WA"
                        first_error_testcase = idx
                        break
            
            # Extract checker stdout for the first error (similar to extract_detailed_result)
            if first_error_type:
                # Get checker output based on error type
                if first_error_type == "CE":
                    first_error_checker_stdout = metadata.get("error_message") or metadata.get("err_message")
                elif first_error_type == "WA":
                    if first_error_testcase is not None:
                        first_error_checker_stdout = metadata.get(f"test_{first_error_testcase}_checker_stderr") or metadata.get(f"test_{first_error_testcase}_checker_message")
                elif first_error_type == "TLE":
                    if first_error_testcase is not None:
                        first_error_checker_stdout = metadata.get(f"test_{first_error_testcase}_timeout_message") or metadata.get(f"test_{first_error_testcase}_timeout_limit_ms")
                elif first_error_type == "MLE":
                    first_error_checker_stdout = "Memory used by User Program Exceeds Limit"
                elif first_error_type == "RE":
                    if first_error_testcase is not None:
                        first_error_checker_stdout = metadata.get(f"test_{first_error_testcase}_stderr") or "Runtime Error"
                
                if first_error_checker_stdout and len(first_error_checker_stdout) > 1000:
                    first_error_checker_stdout = first_error_checker_stdout[:1000]
        
        return {
            "passed": passed,
            "total": total,
            "first_error_type": first_error_type,
            "first_error_testcase": first_error_testcase,
            "first_error_checker_stdout": first_error_checker_stdout,
        }
    
    @staticmethod
    def extract_detailed_result(eval_result_details: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed test result information including first error details.
        
        Args:
            eval_result_details: Evaluation result details dict containing:
                - passed: Number of passed test cases
                - total: Total number of test cases
                - "first_error_type": first_error_type,
                - "first_error_testcase": first_error_testcase
        
        Returns:
            Dictionary with:
                - passed: Number of passed test cases
                - total: Total number of test cases
                - first_error_type: Type of first error (WA/TLE/MLE/RE/CE) or None if all passed
                - first_error_input: First 1000 characters of first error test case input (if available)
                - first_error_output: First 1000 characters of first error test case output (if available)
                - first_error_answer: First 1000 characters of first error test case expected answer (if available)
                - first_error_checker_stdout: Checker stdout for first error test case (if available)
        """
        basic_result = TestResultGiver.extract_basic_result(eval_result_details)
        
        first_error_input = None
        first_error_output = None
        first_error_answer = None
        first_error_checker_stdout = None
        
        if basic_result["first_error_type"] and eval_result_details.get("metadata"):
            first_error_type = basic_result["first_error_type"]
            first_error_testcase = basic_result["first_error_testcase"]
            metadata = eval_result_details.get("metadata")
            
            # Get input/output/answer from metadata (added by cpp_runner)
            first_error_input = metadata.get("first_error_input")
            first_error_output = metadata.get("first_error_output")
            first_error_answer = metadata.get("first_error_answer")
            
            # Get checker output based on error type
            if first_error_type == "CE":
                first_error_checker_stdout = metadata.get("error_message") or metadata.get("err_message")
            elif first_error_type == "WA":
                if first_error_testcase is not None:
                    first_error_checker_stdout = metadata.get(f"test_{first_error_testcase}_checker_stderr") or metadata.get(f"test_{first_error_testcase}_checker_message")
            elif first_error_type == "TLE":
                if first_error_testcase is not None:
                    first_error_checker_stdout = metadata.get(f"test_{first_error_testcase}_timeout_message") or metadata.get(f"test_{first_error_testcase}_timeout_limit_ms")
            elif first_error_type == "MLE":
                first_error_checker_stdout = "Memory used by User Program Exceeds Limit"
            elif first_error_type == "RE":
                if first_error_testcase is not None:
                    first_error_checker_stdout = metadata.get(f"test_{first_error_testcase}_stderr") or "Runtime Error"

            if first_error_checker_stdout and len(first_error_checker_stdout) > 1000:
                first_error_checker_stdout = first_error_checker_stdout[:1000]

        return {
            **basic_result,
            "first_error_input": first_error_input,
            "first_error_output": first_error_output,
            "first_error_answer": first_error_answer,
            "first_error_checker_stdout": first_error_checker_stdout,
        }
    
    @staticmethod
    def format_test_result_summary(test_result: Dict[str, Any], detailed: bool = False) -> str:
        """Format test result as a string summary.
        
        Args:
            test_result: Test result dictionary from extract_basic_result or extract_detailed_result
            detailed: If True, include detailed error information
        
        Returns:
            Formatted string summary
        """
        passed = test_result.get("passed", 0)
        total = test_result.get("total", 0)
        first_error_type = test_result.get("first_error_type")
        
        summary = f"Test Results: {passed}/{total} passed"
        
        if first_error_type == "CE":
            first_error_type = "Compilation Error"
        elif first_error_type == "WA":
            first_error_type = "Wrong Answer"
        elif first_error_type == "TLE":
            first_error_type = "Time Limit Exceeded"
        elif first_error_type == "MLE":
            first_error_type = "Memory Limit Exceeded"
        elif first_error_type == "RE":
            first_error_type = "Runtime Error"
        else:
            first_error_type = "Unknown Error"

        if first_error_type:
            summary += f"\nFirst Error Type: {first_error_type}"
            
            # Always include checker stdout (available in both basic and detailed results)
            if test_result.get("first_error_checker_stdout"):
                summary += f"\nChecker Output:\n{test_result['first_error_checker_stdout']}"
            
            if detailed:
                if test_result.get("first_error_input") is not None:
                    summary += f"\nFirst Error Input (first 1000 chars):\n{test_result['first_error_input']}"
                if test_result.get("first_error_output") is not None:
                    summary += f"\nFirst Error Output (first 1000 chars):\n{test_result['first_error_output']}"
                if test_result.get("first_error_answer") is not None:
                    summary += f"\nFirst Error Expected Answer (first 1000 chars):\n{test_result['first_error_answer']}"
        else:
            summary += "\nAll tests passed!"
        
        return summary

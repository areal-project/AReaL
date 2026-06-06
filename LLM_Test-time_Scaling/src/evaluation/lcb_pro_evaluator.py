"""LiveCodeBench-Pro evaluator."""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

from .base import EvaluationResult, Evaluator
from .code_executor import CodeExecutor
from .cpp_runner import run_cpp_tests_with_checker


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


class LCBProEvaluator(Evaluator):
    """Evaluator for LiveCodeBench-Pro benchmark."""

    def __init__(
        self,
        code_executor: Optional[CodeExecutor] = None,
        dataset_name: str = "QAQAQAQAQ/LiveCodeBench-Pro",
        use_judge: bool = False,
        judge_worker: int = 8,
        data_path: Optional[Path] = None,
        local_data_dir: Optional[str] = None,
    ):
        """Initialize LCB-Pro evaluator.

        Args:
            code_executor: Code executor instance (uses functioncall)
            dataset_name: HuggingFace dataset name
            use_judge: Whether to use LightCPVerifier judge (requires Docker)
            judge_worker: Number of judge workers
            data_path: Path to local lcb_pro.json file (for loading problems)
            local_data_dir: Local data directory containing testcases (similar to cpp_runner)
        """
        self.code_executor = code_executor or CodeExecutor(language="python")
        self.dataset_name = dataset_name
        self.use_judge = use_judge
        self.judge_worker = judge_worker
        self.data_path = data_path
        self.local_data_dir = local_data_dir
        self._judge = None
        self._problems = None

    def _get_judge(self):
        """Get or create judge instance."""
        if self._judge is None and self.use_judge:
            try:
                # Try to import judge from LiveCodeBench-Pro
                lcb_path = Path(__file__).parent.parent.parent.parent / "LiveCodeBench-Pro"
                if str(lcb_path) not in sys.path:
                    sys.path.insert(0, str(lcb_path))

                from judge import LightCPVerifierJudge, SupportedLanguage

                self._judge = LightCPVerifierJudge(worker=self.judge_worker)
                self._judge.__enter__()
                self._SupportedLanguage = SupportedLanguage
            except Exception as e:
                print(f"Warning: Could not initialize judge: {e}")
                print("Falling back to code executor only.")
                self.use_judge = False
        return self._judge

    def load_problems(self, data_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Load problems from HuggingFace dataset or local JSON file.
        
        Args:
            data_path: Path to data file (overrides self.data_path)
            
        Returns:
            List of problem dictionaries
        """
        if data_path is None:
            data_path = self.data_path

        if data_path is None:
            # Try default location
            default_path = (
                Path(__file__).parent.parent.parent.parent
                / "data"
                / "benchmarks"
                / "lcb_pro.json"
            )
            if default_path.exists():
                data_path = default_path
            else:
                # Fallback to HuggingFace dataset
                return self._load_from_huggingface()

        data_path = Path(data_path)
        
        if data_path.suffix == ".json":
            return self._load_from_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

    def _load_from_json(self, json_path: Path) -> List[Dict[str, Any]]:
        """Load problems from JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        problems = []
        for item in data.get("problems", []):
            problem = {
                "problem_id": item.get("problem_id", ""),
                "problem_title": item.get("problem_title", ""),
                "difficulty": item.get("difficulty", "unknown"),
                "platform": item.get("platform", "unknown"),
                "problem_statement": item.get("problem_statement", ""),
                "metadata": item.get("metadata", {}),
            }
            problems.append(problem)
        
        return problems

    def _load_from_huggingface(self) -> List[Dict[str, Any]]:
        """Load problems from HuggingFace dataset."""
        if load_dataset is None:
            raise ImportError(
                "datasets package required. Install with: pip install datasets"
            )

        dataset = load_dataset(self.dataset_name)
        problems = []

        # Flatten dataset splits
        for split_name, split in dataset.items():
            for row in split:
                problem = {
                    "problem_id": row["problem_id"],
                    "problem_title": row.get("problem_title", ""),
                    "difficulty": row.get("difficulty", "unknown"),
                    "platform": row.get("platform", "unknown"),
                    "problem_statement": row["problem_statement"],
                    "metadata": {
                        "split": split_name,
                        **{k: v for k, v in row.items() if k not in ["problem_id", "problem_statement"]},
                    },
                }
                problems.append(problem)

        return problems

    async def evaluate(
        self,
        problem: str,
        solution: str,
        ground_truth: Optional[str] = None,
        problem_id: Optional[str] = None,
        language: str = "python",
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate a code solution.

        Args:
            problem: Problem statement
            solution: Code solution (may contain code blocks)
            ground_truth: Not used for code evaluation
            problem_id: Problem ID (required for local data evaluation)
            language: Programming language (python, cpp, c++)
            **kwargs: Additional parameters

        Returns:
            EvaluationResult
        """
        # Extract code from solution if it contains code blocks
        extracted_code = extract_code(solution)
        if extracted_code is not None:
            solution = extracted_code
        
        if self.use_judge and problem_id:
            return await self._evaluate_with_judge(problem_id, solution, language)
        
        # Check if we should use local data directory (similar to cpp_runner)
        if self.local_data_dir and problem_id and language.lower() in ["cpp", "c++", "c"]:
            return await self._evaluate_with_local_data(problem_id, solution, language)
        
        # Use code executor with test cases
        # Try to get input_output from kwargs or use code executor's default
        input_output = kwargs.get("input_output")
        timeout = kwargs.get("timeout", 2000)
        memory_limit = kwargs.get("memory_limit", 1024)
        
        return await self.code_executor.evaluate(
            problem=problem,
            problem_id=problem_id,
            solution=solution,
            timeout=timeout,
            memory_limit=memory_limit,
            input_output=input_output,
            **{k: v for k, v in kwargs.items() if k not in ["input_output", "timeout", "memory_limit"]},
        )

    async def _evaluate_with_local_data(
        self, problem_id: str, solution: str, language: str
    ) -> EvaluationResult:
        """Evaluate using local data directory (similar to cpp_runner)."""
        import asyncio
        
        if not self.local_data_dir:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback="Local data directory not configured",
                details={"error": "No local_data_dir", "code": solution},
            )
        
        # Use cpp_runner to evaluate
        loop = asyncio.get_event_loop()
        results, metadata = await loop.run_in_executor(
            None,
            run_cpp_tests_with_checker,
            solution,
            self.local_data_dir,
            problem_id,
            30,  # compile_timeout
        )
        # print("run metadata: ", metadata)
        
        if not results:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback=metadata.get("error_message", "Evaluation failed"),
                details=metadata,
            )
        
        # Calculate metrics
        passed = sum(1 for r in results if r is True)
        total = len(results)
        is_correct = all(r is True for r in results)
        score = passed / total if total > 0 else 0.0
        
        feedback = f"Passed {passed}/{total} test cases"
        if metadata:
            if "error_message" in metadata:
                feedback += f"\nError: {metadata['error_message']}"
            if "error" in metadata:
                feedback += f"\nError details: {metadata['error']}"
        
        return EvaluationResult(
            is_correct=is_correct,
            score=score,
            feedback=feedback,
            details={
                "passed": passed,
                "total": total,
                "results": results,
                "metadata": metadata,
                "code": solution,
            },
        )

    async def _evaluate_with_judge(
        self, problem_id: str, code: str, language: str
    ) -> EvaluationResult:
        """Evaluate using LightCPVerifier judge."""
        judge = self._get_judge()
        if not judge:
            return await self.code_executor.evaluate("", code, None)

        try:
            # Map language
            lang_map = {
                "python": self._SupportedLanguage.PYTHON3,
                "cpp": self._SupportedLanguage.CPP,
                "c++": self._SupportedLanguage.CPP,
            }
            supported_lang = lang_map.get(language.lower(), self._SupportedLanguage.CPP)

            # Submit to judge
            submission_id = judge.submit(problem_id, supported_lang, code)

            # Wait for result
            import time
            max_wait = 60  # 60 seconds max wait
            waited = 0
            while waited < max_wait:
                result = judge.get_result(submission_id)
                if result != "Judging":
                    break
                import asyncio
                await asyncio.sleep(1)
                waited += 1

            is_correct = result == "Accepted"
            score = 1.0 if is_correct else 0.0

            return EvaluationResult(
                is_correct=is_correct,
                score=score,
                feedback=f"Judge result: {result}",
                details={"judge_result": result, "submission_id": submission_id},
            )
        except Exception as e:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback=f"Judge error: {str(e)}",
                details={"error": str(e)},
            )

    async def evaluate_batch(
        self,
        problems: list[str],
        solutions: list[str],
        ground_truths: Optional[list[str]] = None,
        problem_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> list[EvaluationResult]:
        """Evaluate a batch of solutions."""
        if problem_ids is None:
            problem_ids = [None] * len(problems)

        tasks = [
            self.evaluate(problem, solution, gt, problem_id=pid, **kwargs)
            for problem, solution, gt, pid in zip(
                problems, solutions, ground_truths or [None] * len(problems), problem_ids
            )
        ]

        import asyncio
        return await asyncio.gather(*tasks)

    def __del__(self):
        """Cleanup judge if used."""
        if self._judge:
            try:
                self._judge.__exit__(None, None, None)
            except Exception:
                pass


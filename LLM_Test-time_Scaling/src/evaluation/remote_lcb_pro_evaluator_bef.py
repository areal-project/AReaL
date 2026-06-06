"""Remote LCB-Pro evaluator using code verify service."""

import asyncio
import random
from typing import Any, Dict, List, Optional
from pathlib import Path
import time

try:
    import aiohttp
    from aiohttp import ClientSession, ClientError
except ImportError:
    print("Error: aiohttp is required. Install with: pip install aiohttp")
    aiohttp = None
    ClientSession = None
    ClientError = Exception

from .base import EvaluationResult, Evaluator
from .lcb_pro_evaluator import extract_code


class RemoteLCBProSimpleEvaluator(Evaluator):
    """Remote evaluator for LiveCodeBench-Pro using code verify service.
    
    This evaluator calls a remote code verification service via HTTP API
    instead of running evaluations locally.
    """

    def __init__(
        self,
        service_url: str,
        data_dir: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: int = 3,
    ):
        """Initialize remote LCB-Pro evaluator.

        Args:
            service_url: URL(s) of the code verify service. Can be a single URL or multiple URLs
                        separated by commas
            data_dir: Optional data directory path (if not provided, service will use default)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
        """
        # Parse service_url: support multiple comma-separated URLs
        if service_url:
            self.service_urls = [url.strip().rstrip("/") for url in service_url.split(',') if url.strip()]
        else:
            self.service_urls = []
        
        if not self.service_urls:
            raise ValueError("At least one service_url must be provided")
        
        self.data_dir = data_dir
        self.timeout = aiohttp.ClientTimeout(total=timeout) if aiohttp and timeout != None else None
        self.max_retries = max_retries
        self._session: Optional[ClientSession] = None

    async def _get_session(self) -> ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            if aiohttp is None:
                raise RuntimeError("aiohttp is not installed")
            self._session = ClientSession(timeout=self.timeout)
        return self._session

    async def _close_session(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _get_service_url(self) -> str:
        """Get a random service URL from the list.

        Returns:
            A service URL string
        """
        if not self.service_urls:
            raise ValueError("No service URLs available")
        if len(self.service_urls) == 1:
            return self.service_urls[0]
        return random.choice(self.service_urls)

    async def _verify_code_remote(
        self, code: str, problem_id: str
    ) -> Dict[str, Any]:
        """Call remote verify service.

        Args:
            code: C++ source code
            problem_id: Problem ID

        Returns:
            Response dictionary from the service

        Raises:
            Exception: If request fails after retries
        """
        session = await self._get_session()
        
        # Randomly select a service URL for this request
        service_url = self._get_service_url()
        url = f"{service_url}/verify"

        payload = {
            "code": code,
            "problem_id": problem_id,
            "compile_timeout": 30,
        }
        if self.data_dir:
            payload["data_dir"] = self.data_dir

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"Service returned status {response.status}: {error_text}"
                        )
            except (ClientError, asyncio.TimeoutError, Exception) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    # On retry, try a different service URL (if multiple available)
                    if len(self.service_urls) > 1:
                        service_url = self._get_service_url()
                        url = f"{service_url}/verify"
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue
                raise

        raise last_exception if last_exception else Exception("Request failed")

    async def evaluate(
        self,
        problem: str,
        solution: str,
        ground_truth: Optional[str] = None,
        problem_id: Optional[str] = None,
        language: str = "cpp",
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate a code solution using remote service.

        Args:
            problem: Problem statement (not used, but kept for interface compatibility)
            solution: Code solution (may contain code blocks)
            ground_truth: Not used
            problem_id: Problem ID (required)
            language: Programming language (should be "cpp", "c++", or "c")
            **kwargs: Additional parameters

        Returns:
            EvaluationResult
        """
        if problem_id is None:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback="problem_id is required for remote evaluation",
                details={"error": "Missing problem_id"},
            )

        # Check language
        if language.lower() not in ["cpp", "c++", "c"]:
            return EvaluationResult(
                is_correct=False,
                score=0.0,
                feedback=f"Language {language} not supported (only C++ supported)",
                details={"error": f"Unsupported language: {language}"},
            )

        # Extract code from solution if it contains code blocks
        extracted_code = extract_code(solution)
        if extracted_code is not None:
            solution = extracted_code

        try:
            # Call remote service and measure time
            eval_start_time = time.time()
            result = await self._verify_code_remote(solution, problem_id)
            eval_end_time = time.time()
            eval_duration = eval_end_time - eval_start_time

            # Parse response
            is_correct = result.get("is_correct", False)
            score = result.get("score", 0.0)
            feedback = result.get("feedback", "")
            metadata = result.get("metadata", {})
            # skip metadata
            # metadata = None
            details = {
                "passed": result.get("passed", 0),
                "total": result.get("total", 0),
                "results": result.get("results", []),
                "metadata": metadata,
                "evaluation_time_sec": eval_duration,  # Add evaluation time
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
                feedback=f"Remote evaluation error: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
            )

    async def evaluate_batch(
        self,
        problems: list[str],
        solutions: list[str],
        ground_truths: Optional[list[str]] = None,
        problem_ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> list[EvaluationResult]:
        """Evaluate a batch of solutions.

        Args:
            problems: List of problem statements
            solutions: List of solutions
            ground_truths: Not used
            problem_ids: List of problem IDs (required)
            **kwargs: Additional parameters

        Returns:
            List of EvaluationResult objects
        """
        if problem_ids is None:
            problem_ids = [None] * len(problems)

        tasks = [
            self.evaluate(
                problem, solution, gt, problem_id=pid, **kwargs
            )
            for problem, solution, gt, pid in zip(
                problems,
                solutions,
                ground_truths or [None] * len(problems),
                problem_ids,
            )
        ]

        return await asyncio.gather(*tasks)

    async def __aenter__(self):
        """Async context manager entry."""
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._close_session()

    def __del__(self):
        """Cleanup session on deletion."""
        if self._session and not self._session.closed:
            # Try to close session synchronously (not ideal, but better than leaking)
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule close
                    asyncio.create_task(self._close_session())
                else:
                    loop.run_until_complete(self._close_session())
            except Exception:
                pass
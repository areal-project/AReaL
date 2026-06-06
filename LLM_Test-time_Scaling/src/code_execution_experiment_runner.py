"""Experiment runner for test-time scaling evaluation."""

import asyncio
import json
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
    module="pydantic",
)

from tqdm.asyncio import tqdm

from .benchmarks import Benchmark, BenchmarkLoader
from .evaluation import CodeExecutor, Evaluator, LLMJudge, IMOBenchEvaluator, PRBenchEvaluator, SATBenchEvaluator, GPQAEvaluator
from .llm_service import create_llm_service
from .prompts import PromptManager
from .scaling import ScalingResult
from .scaling.pipeline import PipelineFactory, ScalingPipeline
from .utils import Config


class CodeExecutionExperimentResult:
    """Result of running an experiment on a benchmark."""

    def __init__(
        self,
        config: Config,
        benchmark_name: str,
        results: List[Dict[str, Any]],
        start_time: datetime,
        end_time: datetime,
        has_new_results: bool = True
    ):
        """Initialize experiment result.

        Args:
            config: Experiment configuration
            benchmark_name: Name of the benchmark
            results: List of per-problem results
            start_time: Experiment start time
            end_time: Experiment end time
        """
        self.config = config
        self.benchmark_name = benchmark_name
        self.results = results
        self.start_time = start_time
        self.end_time = end_time
        self.has_new_results = has_new_results

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics from results including pass@k and difficulty breakdown."""
        total = len(self.results)
        if total == 0:
            return {}

        successful_results = [r for r in self.results if "error" not in r]
        if not successful_results:
            return {
                "total_problems": total,
                "successful_problems": 0,
                "failed_problems": total,
                "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            }

        num_successful = len(successful_results)

        # Pass@k metrics (if available)
        if "metrics" in successful_results[0] and "pass@1" in successful_results[0]["metrics"]:
            # Results have pass@k metrics
            pass_at_1 = sum(r["metrics"]["pass@1"] for r in successful_results) / num_successful
            pass_at_k = sum(r["metrics"]["pass@k"] for r in successful_results) / num_successful
            avg_final_score = sum(r["final_solution"]["score"] for r in successful_results) / num_successful
            avg_best_score = sum(r["metrics"]["best_score_all"] for r in successful_results) / num_successful
            total_solutions = sum(r["metrics"]["num_total"] for r in successful_results)
            total_correct = sum(r["metrics"]["num_correct"] for r in successful_results)
        else:
            # Legacy format: only final solution evaluated
            pass_at_1 = sum(1 for r in successful_results if r.get("is_correct", False)) / num_successful
            pass_at_k = pass_at_1  # Same as pass@1 if we only have final solution
            avg_final_score = sum(r.get("score", 0.0) for r in successful_results) / num_successful
            avg_best_score = avg_final_score
            total_solutions = num_successful
            total_correct = sum(1 for r in successful_results if r.get("is_correct", False))

        total_tokens = sum(r.get("total_tokens", 0) for r in successful_results)

        # Compute timing metrics
        total_llm_time = 0.0
        total_evaluation_time = 0.0
        total_problem_time = 0.0
        timing_results = [r for r in successful_results if "timing" in r]
        if timing_results:
            total_llm_time = sum(r["timing"].get("llm_time_sec", 0.0) for r in timing_results)
            total_evaluation_time = sum(r["timing"].get("evaluation_time_sec", 0.0) for r in timing_results)
            total_problem_time = sum(r["timing"].get("total_time_sec", 0.0) for r in timing_results)
        
        avg_llm_time = total_llm_time / num_successful if num_successful > 0 else 0.0
        avg_evaluation_time = total_evaluation_time / num_successful if num_successful > 0 else 0.0
        avg_problem_time = total_problem_time / num_successful if num_successful > 0 else 0.0
        
        llm_time_percentage = (total_llm_time / total_problem_time * 100) if total_problem_time > 0 else 0.0
        eval_time_percentage = (total_evaluation_time / total_problem_time * 100) if total_problem_time > 0 else 0.0

        # Compute metrics by difficulty
        difficulty_metrics = {}
        difficulty_groups = defaultdict(list)
        
        for r in successful_results:
            difficulty = r.get("difficulty", "unknown")
            difficulty_groups[difficulty].append(r)
        
        for difficulty, group_results in difficulty_groups.items():
            group_count = len(group_results)
            if group_count == 0:
                continue
            
            # Calculate pass@1 and pass@k for this difficulty
            if "metrics" in group_results[0] and "pass@1" in group_results[0]["metrics"]:
                group_pass_at_1 = sum(r["metrics"]["pass@1"] for r in group_results) / group_count
                group_pass_at_k = sum(r["metrics"]["pass@k"] for r in group_results) / group_count
            else:
                group_pass_at_1 = sum(1 for r in group_results if r.get("is_correct", False)) / group_count
                group_pass_at_k = group_pass_at_1
            
            difficulty_metrics[difficulty] = {
                "count": group_count,
                "pass@1": group_pass_at_1,
                "pass@k": group_pass_at_k,
            }

        return {
            "pass@1": pass_at_1,
            "pass@k": pass_at_k,
            "avg_final_score": avg_final_score,
            "avg_best_score": avg_best_score,
            "total_problems": total,
            "successful_problems": num_successful,
            "failed_problems": total - num_successful,
            "total_solutions_generated": total_solutions,
            "total_correct_solutions": total_correct,
            "total_tokens": total_tokens,
            "avg_tokens_per_problem": total_tokens / num_successful if num_successful > 0 else 0,
            "duration_seconds": (self.end_time - self.start_time).total_seconds(),
            "by_difficulty": difficulty_metrics,  # Add difficulty breakdown
            "timing": {
                "total_llm_time_sec": total_llm_time,
                "total_evaluation_time_sec": total_evaluation_time,
                "total_problem_time_sec": total_problem_time,
                "avg_llm_time_sec": avg_llm_time,
                "avg_evaluation_time_sec": avg_evaluation_time,
                "avg_problem_time_sec": avg_problem_time,
                "llm_time_percentage": llm_time_percentage,
                "evaluation_time_percentage": eval_time_percentage,
            },
        }

    def save(self, output_dir: Path) -> None:
        """Save experiment results to disk.

        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.experiment_name}_{self.benchmark_name}_{timestamp}.json"
        output_path = output_dir / filename

        data = {
            "config": self.config.to_dict(),
            "benchmark": self.benchmark_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "metrics": self.compute_metrics(),
            "results": self.results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")


class ExperimentRunner:
    """Runner for test-time scaling experiments."""

    def __init__(self, config: Config, resume_from: Optional[str] = None, continue_reflections_from: Optional[str] = None):
        """Initialize experiment runner.

        Args:
            config: Experiment configuration
            resume_from: Optional path to existing results file to resume from
            continue_reflections_from: Optional path to existing results file to continue reflections from
        """
        self.config = config
        self.resume_from = resume_from
        self.continue_reflections_from = continue_reflections_from
        self.completed_problem_ids = set()
        self.existing_results_by_problem_id: Dict[str, Dict[str, Any]] = {}
        self.processed_results: List[Dict[str, Any]] = []  # Store processed results for code_execution

        # Load existing results if resuming
        if resume_from:
            self._load_existing_results(resume_from)
        
        # Load existing results for continuing reflections
        if continue_reflections_from:
            self._load_existing_results_for_continuation(continue_reflections_from)

        # Initialize LLM service
        self.llm_service = create_llm_service(
            provider=config.llm.provider,
            model_name=config.llm.model_name,
            api_key=config.llm.api_key,
            api_base=config.llm.api_base,
        )

        # Initialize prompt manager
        self.prompt_manager = PromptManager()

        # Initialize benchmark loader
        self.benchmark_loader = BenchmarkLoader()

        # Initialize evaluator based on benchmark
        self.evaluator = self._create_evaluator()

        # Initialize pipeline
        self.pipeline = self._create_pipeline()
        
        # For incremental saving
        self._save_lock = threading.Lock()
        self._results_file_path: Optional[Path] = None
        self._start_time: Optional[datetime] = None
        self._valid_results: List[Dict[str, Any]] = []  # Store valid (no error) results to save first



    def _load_existing_results(self, results_path: str) -> None:
        """Load existing results and extract completed problem IDs.

        For code_execution experiments, also processes results to:
        1. Mark early success (if any iteration in a group is correct, mark subsequent as correct)
        2. Identify failed groups (groups that never achieved correct)
        3. Re-run problems with failed groups

        Args:
            results_path: Path to existing results JSON file
        """
        results_file = Path(results_path)
        if not results_file.exists():
            print(f"Warning: Resume file not found: {results_path}")
            return

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            results = data.get("results", [])
            
            # Check if this is a code_execution experiment
            is_code_execution = (
                self.config.reflection.strategy == "code_execution" and
                self.config.reflection.n_samples_per_iteration == 3
            )
            
            if is_code_execution:
                # Process results to mark early success and identify failed groups
                # This modifies results in-place
                failed_problem_ids = self._process_code_execution_results(results)
                
                # Separate valid results (no errors) from results that need re-run
                valid_results = []
                for result in results:
                    problem_id = result.get("problem_id")
                    if problem_id and problem_id in failed_problem_ids:
                        # Don't add to completed_problem_ids - these need to be re-run
                        continue
                    elif "error" not in result and problem_id:
                        self.completed_problem_ids.add(problem_id)
                        valid_results.append(result)
                
                # Store valid results to save first
                self._valid_results = valid_results
                
                # Store processed results for later merging (but we'll use _valid_results for saving)
                self.processed_results = results
                
                print(f"\nResuming from: {results_path}")
                print(f"Found {len(results)} total problems")
                print(f"  Valid (no errors, will save first): {len(valid_results)}")
                print(f"  Need re-run (has failed groups): {len(failed_problem_ids)}")
                print(f"Will re-run {len(failed_problem_ids)} problems with failed groups")
                print(f"Will save {len(valid_results)} valid results first, then append re-tested problems\n")
            else:
                # Original behavior for non-code_execution experiments
                for result in results:
                    if "error" not in result and "problem_id" in result:
                        self.completed_problem_ids.add(result["problem_id"])

                print(f"\nResuming from: {results_path}")
                print(f"Found {len(self.completed_problem_ids)} completed problems")
                print(f"Will skip these problems and run remaining ones\n")

        except Exception as e:
            print(f"Warning: Failed to load resume file: {e}")
            self.completed_problem_ids = set()
    
    def _process_code_execution_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """Process code execution results to mark early success and identify failed groups.

        Args:
            results: List of result dictionaries

        Returns:
            List of problem IDs that have failed groups and need re-running
        """
        n_samples = self.config.reflection.n_samples_per_iteration
        n_iterations = self.config.reflection.n_iterations + 1  # iterations 0-8 = 9 total
        
        failed_problem_ids = []
        
        for result in results:
            problem_id = result.get("problem_id")
            if not problem_id:
                continue
            
            all_solutions = result.get("all_solutions", [])
            if not all_solutions:
                continue
            
            # Group solutions by sample index
            groups = self._group_solutions_by_sample(all_solutions, n_samples, n_iterations)
            
            # Process each group
            has_failed_group = False
            for group_idx, group in enumerate(groups):
                has_any_correct, was_modified = self._mark_early_success_in_group(group)
                if not has_any_correct:
                    has_failed_group = True
                    break  # If any group failed, the problem needs re-run
            
            if has_failed_group:
                failed_problem_ids.append(problem_id)
        
        return failed_problem_ids
    
    def _group_solutions_by_sample(
        self, all_solutions: List[Dict[str, Any]], n_samples_per_iteration: int, n_iterations: int
    ) -> List[List[Dict[str, Any]]]:
        """Group solutions by sample index across iterations."""
        # Group solutions by iteration first
        solutions_by_iteration = {}
        for sol in all_solutions:
            metadata = sol.get("metadata", {})
            iteration = metadata.get("iteration", 0)
            if iteration not in solutions_by_iteration:
                solutions_by_iteration[iteration] = []
            solutions_by_iteration[iteration].append(sol)
        
        # Sort solutions within each iteration by their original index
        for iteration in solutions_by_iteration:
            solutions_by_iteration[iteration].sort(key=lambda s: all_solutions.index(s))
        
        # Group by sample index across iterations
        groups = [[] for _ in range(n_samples_per_iteration)]
        for iteration in range(n_iterations):
            if iteration in solutions_by_iteration:
                iter_solutions = solutions_by_iteration[iteration]
                for sample_idx in range(n_samples_per_iteration):
                    if sample_idx < len(iter_solutions):
                        groups[sample_idx].append(iter_solutions[sample_idx])
                    else:
                        groups[sample_idx].append(None)
            else:
                for sample_idx in range(n_samples_per_iteration):
                    groups[sample_idx].append(None)
        
        return groups
    
    def _mark_early_success_in_group(self, group: List[Dict[str, Any]]) -> Tuple[bool, bool]:
        """Mark subsequent iterations as correct if any earlier iteration is correct.
        
        Returns:
            Tuple of (has_any_correct, was_modified)
        """
        has_any_correct = False
        was_modified = False
        first_correct_iteration = None
        
        # Find first correct iteration
        for i, sol in enumerate(group):
            if sol is None:
                continue
            if sol.get("is_correct", False):
                has_any_correct = True
                first_correct_iteration = i
                break
        
        # Mark all subsequent iterations as correct
        if first_correct_iteration is not None:
            for i in range(first_correct_iteration + 1, len(group)):
                if group[i] is not None and not group[i].get("is_correct", False):
                    group[i]["is_correct"] = True
                    group[i]["score"] = 1.0
                    group[i]["feedback"] = "Marked as correct due to early success in this group"
                    if "metadata" not in group[i]:
                        group[i]["metadata"] = {}
                    group[i]["metadata"]["early_success_marked"] = True
                    group[i]["metadata"]["first_correct_iteration"] = first_correct_iteration
                    was_modified = True
        
        return has_any_correct, was_modified
    
    def _load_existing_results_list(self, results_path: str) -> List[Dict[str, Any]]:
        """Load existing results as a list.

        Args:
            results_path: Path to existing results JSON file

        Returns:
            List of existing results
        """
        results_file = Path(results_path)
        if not results_file.exists():
            return []

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("results", [])
        except Exception as e:
            print(f"Warning: Failed to load existing results list: {e}")
            return []
    
    def _load_existing_results_for_continuation(self, results_path: str) -> None:
        """Load existing results for continuing reflections.

        Args:
            results_path: Path to existing results JSON file
        """
        results_file = Path(results_path)
        if not results_file.exists():
            print(f"Warning: Continue reflections file not found: {results_path}")
            return

        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            results = data.get("results", [])
            for result in results:
                if "error" not in result and "problem_id" in result:
                    problem_id = result["problem_id"]
                    self.existing_results_by_problem_id[problem_id] = result

            print(f"\nContinuing reflections from: {results_path}")
            print(f"Found {len(self.existing_results_by_problem_id)} problems with existing results")
            print(f"Will continue reflections from existing solutions\n")

        except Exception as e:
            print(f"Warning: Failed to load continue reflections file: {e}")
            self.existing_results_by_problem_id = {}
    
    def _load_solutions_from_existing_result(self, existing_result: Dict[str, Any]) -> List:
        """Load solutions from existing result for continuing reflections.

        Args:
            existing_result: Existing result dictionary

        Returns:
            List of Solution objects from the existing result
        """
        from .scaling.base import Solution
        
        solutions = []
        
        # Try to load from all_solutions first (contains all intermediate solutions)
        all_solutions_data = existing_result.get("all_solutions", [])
        
        if all_solutions_data:
            # Extract the last iteration's solutions (most recent)
            # Group by iteration if available
            solutions_by_iteration = {}
            for sol_data in all_solutions_data:
                # Try to get iteration from metadata
                metadata = sol_data.get("metadata", {})
                iteration = metadata.get("iteration", 0)
                if iteration not in solutions_by_iteration:
                    solutions_by_iteration[iteration] = []
                solutions_by_iteration[iteration].append(sol_data)
            
            if solutions_by_iteration:
                # Get the highest iteration number
                max_iteration = max(solutions_by_iteration.keys())
                last_iteration_solutions = solutions_by_iteration[max_iteration]
                
                # Convert to Solution objects
                for sol_data in last_iteration_solutions:
                    solution = Solution(
                        content=sol_data.get("solution_content", ""),
                        score=sol_data.get("score", 0.0),
                        feedback=sol_data.get("feedback", ""),
                        metadata=sol_data.get("metadata", {}),
                    )
                    solutions.append(solution)
            else:
                # Fallback: use all solutions from the last batch
                # Take the last n solutions where n = samples_per_iteration
                n_samples = self.config.reflection.n_samples_per_iteration
                last_solutions = all_solutions_data[-n_samples:] if len(all_solutions_data) >= n_samples else all_solutions_data
                
                for sol_data in last_solutions:
                    solution = Solution(
                        content=sol_data.get("solution_content", ""),
                        score=sol_data.get("score", 0.0),
                        feedback=sol_data.get("feedback", ""),
                        metadata=sol_data.get("metadata", {}),
                    )
                    solutions.append(solution)
        else:
            # Fallback: use final solution
            final_solution_data = existing_result.get("final_solution", {})
            if final_solution_data:
                solution = Solution(
                    content=final_solution_data.get("content", ""),
                    score=final_solution_data.get("score", 0.0),
                    feedback=final_solution_data.get("feedback", ""),
                    metadata=final_solution_data.get("metadata", {}),
                )
                solutions.append(solution)
        
        return solutions



    def _create_evaluator(self) -> Optional[Evaluator]:
        """Create evaluator based on benchmark type."""
        benchmark_name = self.config.evaluation.benchmark
        evaluator_type = self.config.evaluation.evaluator_type

        if evaluator_type == "llm_judge":
            # Create LLM judge evaluator
            evaluator_service = create_llm_service(
                provider=self.config.evaluation.provider,
                model_name=self.config.evaluation.model_name,
                api_key=self.config.evaluation.api_key,
                api_base=self.config.evaluation.api_base,
            )
            return LLMJudge(evaluator_service, benchmark=benchmark_name)
        elif evaluator_type == "imobench_evaluator":
            evaluator_service = create_llm_service(
                provider=self.config.evaluation.provider,
                model_name=self.config.evaluation.model_name,
                api_key=self.config.evaluation.api_key,
                api_base=self.config.evaluation.api_base,
            )
            llm_judge = LLMJudge(evaluator_service, benchmark=benchmark_name)
            imobench_evaluator = IMOBenchEvaluator(llm_judge=llm_judge)
            return imobench_evaluator
        elif evaluator_type == "prbench":
            evaluator_service = create_llm_service(
                provider=self.config.evaluation.provider,
                model_name=self.config.evaluation.model_name,
                api_key=self.config.evaluation.api_key,
                api_base=self.config.evaluation.api_base,
            )
            prbench_evaluator = PRBenchEvaluator(
                llm_service = evaluator_service,
                max_concurrent = 32,
            )
            return prbench_evaluator
        elif evaluator_type == "satbench":
            # SAT solving evaluator - rule-based label matching
            return SATBenchEvaluator()
        elif evaluator_type == "gpqa":
            # GPQA Diamond evaluator - rule-based answer matching
            return GPQAEvaluator()
        elif evaluator_type == "code_executor":
            if benchmark_name == "lcb_pro":
                # from .evaluation import LCBProEvaluator
                # # Get local_data_dir from config if available
                # local_data_dir = getattr(self.config.evaluation, 'local_data_dir', None)
                # return LCBProEvaluator(
                #     local_data_dir=local_data_dir,
                # )
                service_url = getattr(self.config.evaluation, 'service_url', None)
                if service_url:
                    from src.evaluation.remote_lcb_pro_evaluator import RemoteLCBProEvaluator
                    
                    if RemoteLCBProEvaluator is None:
                        raise ImportError(
                            f"RemoteLCBProEvaluator not found at {remote_eval_path}. "
                            "Please ensure the remote evaluator module is available."
                        )
                    
                    data_dir = getattr(self.config.evaluation, 'local_data_dir', None)
                    language = getattr(self.config.evaluation, 'language', 'cpp')
                    return RemoteLCBProEvaluator(
                        service_url=service_url,
                        data_dir=data_dir,
                        timeout=300,
                        max_retries=3,
                    )
                else:
                    # Use local evaluator
                    from .evaluation import LCBProEvaluator
                    local_data_dir = getattr(self.config.evaluation, 'local_data_dir', None)
                    return LCBProEvaluator(
                        local_data_dir=local_data_dir,
                    )
            return CodeExecutor(language="python")

        return None

    def _infer_task_domain(self, benchmark_name: str) -> str:
        """Infer task domain from benchmark name.

        Args:
            benchmark_name: Name of the benchmark

        Returns:
            Task domain string
        """
        domain_mapping = {
            "imobench": "math",
            "imo": "math",
            "lcb_pro": "coding",
            "livecodebench": "coding",
            "prbench": "professional_reasoning",
            "hle": "general",
            "satbench": "sat_solving",
            "gpqa_diamond": "science_qa",
        }
        # Try exact match first
        if benchmark_name in domain_mapping:
            return domain_mapping[benchmark_name]

        # Try partial match
        for key, domain in domain_mapping.items():
            if key in benchmark_name.lower():
                return domain

        return "general"

    def _create_pipeline(self) -> ScalingPipeline:
        """Create scaling pipeline based on config."""
        factory = PipelineFactory(self.llm_service, self.prompt_manager)

        # Infer task domain from benchmark
        task_domain = self._infer_task_domain(self.config.evaluation.benchmark)

        # Prepare kwargs for reflection and aggregation
        reflection_kwargs = {
            "task_domain": task_domain,
            "reasoning_effort": self.config.reflection.reasoning_effort,
        }
        if self.config.reflection.strategy == "ground_truth":
            reflection_kwargs["evaluator"] = self.evaluator
        elif self.config.reflection.strategy == "code_execution":
            if not self.evaluator:
                raise ValueError("Evaluator is required for code_execution reflection strategy")
            reflection_kwargs["evaluator"] = self.evaluator

        # Create strategies separately to pass different reasoning_effort
        reflection = factory.create_reflection_strategy(
            self.config.reflection.strategy,
            **reflection_kwargs
        )

        aggregation_kwargs = {
            "task_domain": task_domain,
            "reasoning_effort": self.config.aggregation.reasoning_effort,
        }
        aggregation = factory.create_aggregation_strategy(
            self.config.aggregation.strategy,
            **aggregation_kwargs
        )

        # Create pipeline with both strategies
        from .scaling.pipeline import ScalingPipeline
        return ScalingPipeline(
            self.llm_service,
            reflection,
            aggregation,
            self.prompt_manager,
            apply_aggregation_each_turn=self.config.aggregation.apply_at_each_turn,
            task_domain=task_domain
        )

    async def run_single_problem(
        self, problem_data: Any, task_domain: str = "general", **kwargs: Any
    ) -> Dict[str, Any]:
        """Run test-time scaling on a single problem with pass@k evaluation.

        Args:
            problem_data: Problem data from benchmark
            task_domain: Task domain for prompt selection
            **kwargs: Additional parameters

        Returns:
            Dictionary with problem results including pass@k metrics
        """
        import time
        problem = problem_data.problem
        ground_truth = problem_data.ground_truth
        test_cases = problem_data.test_cases

        problem_start_time = time.time()

        # Check if we should continue from existing results
        initial_solutions = None
        if self.continue_reflections_from and hasattr(problem_data, 'id'):
            problem_id = problem_data.id
            if problem_id in self.existing_results_by_problem_id:
                existing_result = self.existing_results_by_problem_id[problem_id]
                initial_solutions = self._load_solutions_from_existing_result(existing_result)
                print(f"  Continuing reflections from existing result for problem {problem_id}")
                print(f"  Loaded {len(initial_solutions)} solutions from previous iterations")

        # Run scaling pipeline
        # Prepare kwargs for pipeline (including problem_id for code_execution strategy)
        pipeline_kwargs = {
            "ground_truth": ground_truth,
            "test_cases": test_cases,
        }
        # Add problem_id if available (needed for code_execution reflection strategy)
        if hasattr(problem_data, 'id'):
            pipeline_kwargs["problem_id"] = problem_data.id
        # Add language if evaluator has it (needed for code evaluation)
        if hasattr(self.evaluator, 'language'):
            pipeline_kwargs["language"] = getattr(self.evaluator, 'language', 'cpp')
        elif hasattr(self.config.evaluation, 'language'):
            pipeline_kwargs["language"] = getattr(self.config.evaluation, 'language', 'cpp')
        
        scaling_result: ScalingResult = await self.pipeline.run(
            problem=problem,
            n_initial_solutions=self.config.reflection.n_samples_per_iteration,
            n_iterations=self.config.reflection.n_iterations,
            temperature=self.config.llm.temperature,
            reasoning_effort=self.config.llm.reasoning_effort,
            task_domain=task_domain,
            initial_solutions=initial_solutions,
            **pipeline_kwargs,
        )

        # Collect LLM call times from all solutions
        total_llm_time = 0.0
        for solution in scaling_result.all_solutions:
            solution_metadata = solution.metadata or {}
            # Check various usage dict fields in metadata for LLM call time
            # Initial solutions use "usage"
            usage = solution_metadata.get("usage", {})
            if isinstance(usage, dict):
                llm_time = usage.get("llm_call_time_sec", 0.0)
                if isinstance(llm_time, (int, float)):
                    total_llm_time += llm_time
            
            # Reflection strategies may use "improve_usage" and "eval_usage"
            improve_usage = solution_metadata.get("improve_usage", {})
            if isinstance(improve_usage, dict):
                llm_time = improve_usage.get("llm_call_time_sec", 0.0)
                if isinstance(llm_time, (int, float)):
                    total_llm_time += llm_time
            
            eval_usage = solution_metadata.get("eval_usage", {})
            if isinstance(eval_usage, dict):
                llm_time = eval_usage.get("llm_call_time_sec", 0.0)
                if isinstance(llm_time, (int, float)):
                    total_llm_time += llm_time

        # Check if using code_execution strategy (which already evaluates solutions)
        is_code_execution = (
            self.pipeline.reflection_strategy and 
            self.pipeline.reflection_strategy.get_strategy_name() == "code_execution"
        )
        
        # Evaluate ALL solutions (not just final)
        all_evaluations = []
        total_evaluation_time = 0.0
        if self.evaluator and not is_code_execution:
            # For code_execution strategy, evaluation is done during reflection, so skip here
            for solution in scaling_result.all_solutions:
                # For lcb_pro, pass problem_id and language
                eval_kwargs = {
                    "problem": problem,
                    "solution": solution.content,
                    "ground_truth": ground_truth,
                }
                # Add problem_id and language for code evaluation
                if hasattr(problem_data, 'id'):
                    eval_kwargs["problem_id"] = problem_data.id
                if hasattr(self.evaluator, 'language'):
                    eval_kwargs["language"] = getattr(self.evaluator, 'language', 'cpp')
                elif hasattr(self.config.evaluation, 'language'):
                    eval_kwargs["language"] = getattr(self.config.evaluation, 'language', 'cpp')
                # Add test_cases if not using local data
                if test_cases and not (hasattr(self.evaluator, 'local_data_dir') and self.evaluator.local_data_dir):
                    eval_kwargs["test_cases"] = test_cases
                
                eval_result = await self.evaluator.evaluate(**eval_kwargs)
                # Extract evaluation time from details
                eval_time = 0.0
                if eval_result.details and isinstance(eval_result.details, dict):
                    eval_time = eval_result.details.get("evaluation_time_sec", 0.0) or 0.0
                    if isinstance(eval_time, (int, float)):
                        total_evaluation_time += eval_time
                
                all_evaluations.append({
                    "solution_content": solution.content,
                    "is_correct": eval_result.is_correct,
                    "score": eval_result.score,
                    "feedback": eval_result.feedback,
                    "metadata": solution.metadata,
                    # "eval_metadata": eval_result.details,  # Include evaluation metadata from code executor
                })
        elif is_code_execution:
            # For code_execution, extract evaluation results from solution metadata
            # Evaluation was already done during reflection, so we extract from metadata
            for solution in scaling_result.all_solutions:
                solution_metadata = solution.metadata or {}
                test_results = solution_metadata.get("test_results", {})
                eval_details = solution_metadata.get("eval_details")
                
                # Extract evaluation info from test_results
                passed = test_results.get("passed", 0)
                total = test_results.get("total", 0)
                is_correct = (passed == total) and total > 0
                score = solution.score if solution.score is not None else (1.0 if is_correct else 0.0)
                
                # Extract evaluation time from eval_details if available
                eval_time = 0.0
                if eval_details and isinstance(eval_details, dict):
                    eval_time = eval_details.get("evaluation_time_sec", 0.0) or 0.0
                    if isinstance(eval_time, (int, float)):
                        total_evaluation_time += eval_time
                
                all_evaluations.append({
                    "solution_content": solution.content,
                    "is_correct": is_correct,
                    "score": score,
                    "feedback": solution.feedback or "",
                    "metadata": solution.metadata,
                    # "eval_metadata": eval_details,  # Store evaluation details from reflection
                })

        # Evaluate final solution separately
        final_eval_result = None
        if self.evaluator:
            # final_eval_result = await self.evaluator.evaluate(
            #     problem=problem,
            #     solution=scaling_result.final_solution.content,
            #     ground_truth=ground_truth,
            #     test_cases=test_cases,
            # )
            eval_kwargs = {
                "problem": problem,
                "solution": scaling_result.final_solution.content,
                "ground_truth": ground_truth,
            }
            # Add problem_id and language for code evaluation
            if hasattr(problem_data, 'id'):
                eval_kwargs["problem_id"] = problem_data.id
            if hasattr(self.evaluator, 'language'):
                eval_kwargs["language"] = getattr(self.evaluator, 'language', 'cpp')
            elif hasattr(self.config.evaluation, 'language'):
                eval_kwargs["language"] = getattr(self.config.evaluation, 'language', 'cpp')
            # Add test_cases if not using local data
            if test_cases and not (hasattr(self.evaluator, 'local_data_dir') and self.evaluator.local_data_dir):
                eval_kwargs["test_cases"] = test_cases
            
            final_eval_result = await self.evaluator.evaluate(**eval_kwargs)
            # Add final evaluation time
            if final_eval_result and final_eval_result.details and isinstance(final_eval_result.details, dict):
                final_eval_time = final_eval_result.details.get("evaluation_time_sec", 0.0) or 0.0
                if isinstance(final_eval_time, (int, float)):
                    total_evaluation_time += final_eval_time

        problem_end_time = time.time()
        total_problem_time = problem_end_time - problem_start_time

        # Calculate pass@k metrics
        correct_solutions = [e for e in all_evaluations if e["is_correct"]]
        pass_at_1 = 1 if final_eval_result and final_eval_result.is_correct else 0
        pass_at_k = 1 if len(correct_solutions) > 0 else 0
        best_score_all = max((e["score"] for e in all_evaluations), default=0.0)

        # Get difficulty from problem_data
        difficulty = getattr(problem_data, 'difficulty', None) or problem_data.metadata.get('difficulty', 'unknown')
        
        # Calculate time percentages
        llm_time_percentage = (total_llm_time / total_problem_time * 100) if total_problem_time > 0 else 0.0
        eval_time_percentage = (total_evaluation_time / total_problem_time * 100) if total_problem_time > 0 else 0.0
        
        return {
            "problem_id": problem_data.id,
            "problem": problem,
            "ground_truth": ground_truth,
            "difficulty": difficulty,  # Add difficulty to results
            "final_solution": {
                "content": scaling_result.final_solution.content,
                "is_correct": final_eval_result.is_correct if final_eval_result else None,
                "score": final_eval_result.score if final_eval_result else None,
                "feedback": final_eval_result.feedback if final_eval_result else None,
                "metadata": scaling_result.final_solution.metadata,
                "code": final_eval_result.details.get("code", "") if final_eval_result and final_eval_result.details else "",
                # "eval_metadata": final_eval_result.details if final_eval_result else None,  # Include evaluation metadata from code executor
           },
            "all_solutions": all_evaluations,
            "metrics": {
                "pass@1": pass_at_1,
                "pass@k": pass_at_k,
                "num_correct": len(correct_solutions),
                "num_total": len(all_evaluations),
                "final_score": final_eval_result.score if final_eval_result else 0.0,
                "best_score_all": best_score_all,
            },
            "total_tokens": scaling_result.total_tokens,
            "iterations": scaling_result.iterations,
            "scaling_metadata": scaling_result.metadata,
            "timing": {
                "total_time_sec": total_problem_time,
                "llm_time_sec": total_llm_time,
                "evaluation_time_sec": total_evaluation_time,
                "llm_time_percentage": llm_time_percentage,
                "evaluation_time_percentage": eval_time_percentage,
            },
        }

    def _save_incremental_result(
        self,
        new_result: Dict[str, Any],
        benchmark_name: str,
        start_time: datetime,
        all_current_results: List[Dict[str, Any]],
    ) -> None:
        """Save results incrementally to disk.
        
        Args:
            new_result: New problem result to save
            benchmark_name: Name of the benchmark
            start_time: Experiment start time
            all_current_results: All current results including the new one
        """
        with self._save_lock:
            # Determine output file path
            if self._results_file_path is None:
                output_dir = Path(self.config.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = start_time.strftime("%Y%m%d_%H%M%S")
                filename = f"{self.config.experiment_name}_{benchmark_name}_{timestamp}.json"
                self._results_file_path = output_dir / filename
                
                # If we have valid results to save first, save them now
                if self._valid_results:
                    print(f"  Saving {len(self._valid_results)} valid results first...")
                    # Save valid results to file
                    end_time = datetime.now()
                    temp_result = ExperimentResult(
                        config=self.config,
                        benchmark_name=benchmark_name,
                        results=self._valid_results,
                        start_time=start_time,
                        end_time=end_time,
                    )
                    temp_path = self._results_file_path.with_suffix('.json.tmp')
                    data = {
                        "config": self.config.to_dict(),
                        "benchmark": benchmark_name,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat(),
                        "metrics": temp_result.compute_metrics(),
                        "results": self._valid_results,
                    }
                    with open(temp_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    temp_path.replace(self._results_file_path)
                    print(f"  ✓ Saved {len(self._valid_results)} valid results to {self._results_file_path}")
            
            # Load existing results from file if it exists
            existing_file_results = []
            if self._results_file_path.exists():
                try:
                    with open(self._results_file_path, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                        existing_file_results = file_data.get("results", [])
                except Exception:
                    existing_file_results = []
            
            # Merge: remove duplicates by problem_id, then add new result
            existing_ids = {r.get("problem_id") for r in existing_file_results}
            new_problem_id = new_result.get("problem_id")
            
            # Remove old result for this problem_id if exists
            merged_results = [
                r for r in existing_file_results
                if r.get("problem_id") != new_problem_id
            ]
            # Add new result
            merged_results.append(new_result)
            
            # Compute current metrics
            end_time = datetime.now()
            temp_result = ExperimentResult(
                config=self.config,
                benchmark_name=benchmark_name,
                results=merged_results,
                start_time=start_time,
                end_time=end_time,
            )
            
            # Save to file (atomic write: write to temp file, then rename)
            temp_path = self._results_file_path.with_suffix('.json.tmp')
            data = {
                "config": self.config.to_dict(),
                "benchmark": benchmark_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "metrics": temp_result.compute_metrics(),
                "results": merged_results,
            }
            
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_path.replace(self._results_file_path)
            
            problem_id = new_result.get("problem_id")
            print(f"  ✓ Saved intermediate result for problem {problem_id} ({len(merged_results)} total) to {temp_path}")

    async def run_benchmark(self, benchmark_name: str) -> ExperimentResult:
        """Run experiment on a benchmark.

        Args:
            benchmark_name: Name of the benchmark to evaluate

        Returns:
            ExperimentResult with all results
        """
        start_time = datetime.now()
        self._start_time = start_time

        # Load benchmark
        print(f"\nLoading benchmark: {benchmark_name}")
        splits = getattr(self.config.evaluation, 'splits', None)
        if splits:
            print(f"Filtering by splits: {splits}")
        benchmark = self.benchmark_loader.load(benchmark_name, splits=splits)
        print(f"Loaded {len(benchmark)} problems")

        # Filter out completed problems if resuming
        if self.completed_problem_ids:
            original_count = len(benchmark.problems)
            benchmark.problems = [
                p for p in benchmark.problems
                if p.id not in self.completed_problem_ids
            ]
            skipped_count = original_count - len(benchmark.problems)
            print(f"Skipped {skipped_count} completed problems")
            print(f"Running {len(benchmark.problems)} remaining problems")

        # Load existing results if resuming (for incremental saving)
        # Note: We don't use existing_results in _save_incremental_result anymore
        # Instead, we save _valid_results first, then append new results one by one
        existing_results = []
        if self.resume_from:
            # Use processed results if available (for code_execution experiments)
            # But we'll use _valid_results for initial save, not for merging
            if self.processed_results:
                existing_results = self.processed_results
            else:
                existing_results = self._load_existing_results_list(self.resume_from)

        # If no problems to run, load and return existing results
        if len(benchmark.problems) == 0:
            print("\nAll problems already completed! Loading existing results...")
            if self.resume_from:
                # existing_results = self._merge_with_existing_results([])
                end_time = datetime.now()
                return ExperimentResult(
                    config=self.config,
                    benchmark_name=benchmark_name,
                    results=existing_results,
                    start_time=start_time,
                    end_time=end_time,
                    has_new_results=False,
                )

        # Infer task domain
        task_domain = self._infer_task_domain(benchmark_name)
        print(f"Task domain: {task_domain}")

        # Create semaphore for limiting concurrent executions
        semaphore = asyncio.Semaphore(self.config.max_concurrent_problems)
        
        # Track results for incremental saving
        results = []
        results_lock = asyncio.Lock()

        async def run_with_semaphore(problem):
            """Wrapper to run single problem with semaphore and save incrementally."""
            async with semaphore:
                try:
                    result = await self.run_single_problem(problem, task_domain=task_domain)
                    # Save incrementally after each problem completes
                    async with results_lock:
                        results.append(result)
                        # Save to disk (run in executor to avoid blocking async event loop)
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None,
                            self._save_incremental_result,
                            result,
                            benchmark_name,
                            start_time,
                            results,  # Only new results (valid results saved separately)
                        )
                    return result
                except Exception as e:
                    print(f"\nError processing problem: {e}")
                    raise e

        # Run on all problems
        print(f"\nRunning {self.config.experiment_name} on {benchmark_name}...")
        print(f"Max concurrent problems: {self.config.max_concurrent_problems}")
        print(f"Intermediate results will be saved incrementally")
        tasks = [
            run_with_semaphore(problem)
            for problem in benchmark.problems
        ]

        # Wait for all tasks to complete
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Processing"):
            try:
                await coro
            except Exception as e:
                print(f"\nError processing problem: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Merge with existing results if resuming
        if self.resume_from:
            results = self._merge_with_existing_results(results)

        end_time = datetime.now()

        return ExperimentResult(
            config=self.config,
            benchmark_name=benchmark_name,
            results=results,
            start_time=start_time,
            end_time=end_time,
        )

    def _merge_with_existing_results(self, new_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge new results with existing results from resume file.

        Args:
            new_results: Newly computed results

        Returns:
            Combined list of existing and new results
        """
        if not self.resume_from:
            return new_results

        try:
            with open(self.resume_from, "r", encoding="utf-8") as f:
                data = json.load(f)

            existing_results = data.get("results", [])

            # Combine: existing results + new results
            all_results = existing_results + new_results

            print(f"\nMerged results:")
            print(f"  Existing: {len(existing_results)} problems")
            print(f"  New: {len(new_results)} problems")
            print(f"  Total: {len(all_results)} problems")

            return all_results

        except Exception as e:
            print(f"Warning: Failed to merge with existing results: {e}")
            return new_results

    async def run(self) -> ExperimentResult:
        """Run the full experiment.

        Returns:
            ExperimentResult
        """
        benchmark_name = self.config.evaluation.benchmark
        result = await self.run_benchmark(benchmark_name)

        # Results are already saved incrementally, but save final version to ensure it's up to date
        if result.has_new_results:
            output_dir = Path(self.config.output_dir)
            # If incremental saving was used, update the file with final end_time and metrics
            if self._results_file_path and self._results_file_path.exists():
                # Final save to update end_time and metrics
                result.save(output_dir)
                print(f"\nFinal results saved to: {self._results_file_path}")
            else:
                # If incremental saving didn't happen (e.g., no problems), save normally
                result.save(output_dir)

        # Print summary
        metrics = result.compute_metrics()
        print("\n" + "=" * 50)
        print("EXPERIMENT SUMMARY")
        print("=" * 50)
        print(f"Benchmark: {benchmark_name}")
        print(f"Experiment: {self.config.experiment_name}")
        print(f"Pass@1: {metrics.get('pass@1', 0):.2%}")
        print(f"Pass@k: {metrics.get('pass@k', 0):.2%}")
        print(f"Avg Final Score: {metrics.get('avg_final_score', 0):.3f}")
        print(f"Avg Best Score: {metrics.get('avg_best_score', 0):.3f}")
        print(f"Total Problems: {metrics.get('total_problems', 0)}")
        print(f"Successful: {metrics.get('successful_problems', 0)}")
        print(f"Failed: {metrics.get('failed_problems', 0)}")
        print(f"Total Solutions: {metrics.get('total_solutions_generated', 0)}")
        print(f"Correct Solutions: {metrics.get('total_correct_solutions', 0)}")
        print(f"Total Tokens: {metrics.get('total_tokens', 0):,}")
        print(f"Duration: {metrics.get('duration_seconds', 0):.1f}s")
        
        # Print timing statistics
        timing = metrics.get("timing", {})
        if timing:
            print("\n" + "-" * 50)
            print("TIMING STATISTICS")
            print("-" * 50)
            print(f"Total LLM Time: {timing.get('total_llm_time_sec', 0):.2f}s")
            print(f"Total Evaluation Time: {timing.get('total_evaluation_time_sec', 0):.2f}s")
            print(f"Total Problem Time: {timing.get('total_problem_time_sec', 0):.2f}s")
            print(f"Avg LLM Time/Problem: {timing.get('avg_llm_time_sec', 0):.2f}s")
            print(f"Avg Evaluation Time/Problem: {timing.get('avg_evaluation_time_sec', 0):.2f}s")
            print(f"Avg Problem Time: {timing.get('avg_problem_time_sec', 0):.2f}s")
            print(f"LLM Time Percentage: {timing.get('llm_time_percentage', 0):.1f}%")
            print(f"Evaluation Time Percentage: {timing.get('evaluation_time_percentage', 0):.1f}%")
            print("-" * 50)
        
        # Print difficulty breakdown if available (for lcb_pro)
        if benchmark_name == "lcb_pro" and "by_difficulty" in metrics:
            print("\n" + "-" * 50)
            print("BY DIFFICULTY")
            print("-" * 50)
            difficulty_metrics = metrics["by_difficulty"]
            # Sort by difficulty: easy, medium, hard, unknown
            difficulty_order = ["easy", "medium", "hard", "unknown"]
            sorted_difficulties = sorted(
                difficulty_metrics.keys(),
                key=lambda x: (difficulty_order.index(x) if x in difficulty_order else 999, x)
            )
            
            for difficulty in sorted_difficulties:
                diff_metrics = difficulty_metrics[difficulty]
                count = diff_metrics["count"]
                pass_at_1 = diff_metrics["pass@1"]
                pass_at_k = diff_metrics["pass@k"]
                print(f"{difficulty.capitalize():<10} | Count: {count:<5} | Pass@1: {pass_at_1:.2%} | Pass@k: {pass_at_k:.2%}")
        
        print("=" * 50)

        return result


async def run_experiment_from_config(
    config_path: Path, resume_from: Optional[str] = None
) -> ExperimentResult:
    """Run experiment from configuration file.

    Args:
        config_path: Path to YAML configuration file
        resume_from: Optional path to existing results file to resume from

    Returns:
        ExperimentResult
    """
    config = Config.from_yaml(config_path)
    runner = ExperimentRunner(config, resume_from=resume_from)
    return await runner.run()


def main():
    """Main entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run test-time scaling experiments")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to existing results file to resume from (skip completed problems)",
    )
    args = parser.parse_args()

    asyncio.run(run_experiment_from_config(args.config, resume_from=args.resume_from))


if __name__ == "__main__":
    main()
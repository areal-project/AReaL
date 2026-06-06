"""Benchmark data loader."""

import json
import os
from pathlib import Path
from typing import Optional, List

from .base import Benchmark, BenchmarkProblem


class BenchmarkLoader:
    """Loader for benchmark datasets."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize benchmark loader.

        Args:
            data_dir: Directory containing benchmark data files
        """
        if data_dir is None:
            env_data_dir = os.getenv("BENCHMARK_DATA_DIR")
            if env_data_dir:
                data_dir = Path(env_data_dir)
            else:
                data_dir = Path(__file__).parent.parent.parent / "local_data" / "benchmarks"
                if not data_dir.exists():
                    data_dir = Path(__file__).parent.parent.parent / "data" / "benchmarks"

        self.data_dir = Path(data_dir)

    def load(
        self,
        benchmark_name: str,
        splits: Optional[List[str]] = None,
        benchmark_file: Optional[str] = None,
    ) -> Benchmark:
        """Load a benchmark by name.

        Args:
            benchmark_name: Name of the benchmark (imobench, imo2025, lcb_pro, prbench, hle, satbench)
            splits: Optional list of split names to filter (e.g., ["biannual_2024_7_12"])
                   For satbench, can use ["satisfiable"] or ["unsatisfiable"]
                   If None, loads all problems

        Returns:
            Benchmark object
        """
        benchmark_path: Path
        if benchmark_file:
            benchmark_path = Path(benchmark_file).expanduser()
        else:
            benchmark_as_path = Path(benchmark_name).expanduser()
            if benchmark_as_path.suffix == ".json" and benchmark_as_path.exists():
                benchmark_path = benchmark_as_path
                benchmark_name = benchmark_as_path.stem
            else:
                benchmark_path = self.data_dir / f"{benchmark_name}.json"
                if splits is not None:
                    benchmark_path = self.data_dir / f"{benchmark_name}_with_splits_filtered.json"

        if not benchmark_path.exists():
            raise FileNotFoundError(
                f"Benchmark file not found: {benchmark_path}\n"
                f"Please download the benchmark data and place it in {self.data_dir}"
            )

        with open(benchmark_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        parse_as = self._resolve_parser_name(benchmark_name, data)
        return self._parse_benchmark(benchmark_name, data, splits=splits, parse_as=parse_as)

    def _resolve_parser_name(self, benchmark_name: str, data: dict) -> str:
        """Select parser type from benchmark name or data shape.

        Custom AIME files use IMOBench-compatible schema, so we parse them as
        IMOBench when schema matches even if file name differs.
        """
        normalized = benchmark_name.lower()
        known = {
            "imobench",
            "imo2025",
            "lcb_pro",
            "prbench",
            "satbench",
            "gpqa_diamond",
            "simpleqa",
            "hle_text_only",
        }
        if normalized in known:
            return normalized

        if normalized.startswith("imobench") or normalized.startswith("imo"):
            return "imobench"

        problems = data.get("problems", []) if isinstance(data, dict) else []
        if problems and isinstance(problems[0], dict) and "problem" in problems[0]:
            return "imobench"

        return normalized

    def _parse_benchmark(
        self, 
        name: str, 
        data: dict,
        splits: Optional[List[str]] = None,
        parse_as: Optional[str] = None,
    ) -> Benchmark:
        """Parse benchmark data into Benchmark object.
        
        Args:
            name: Benchmark name
            data: Benchmark data dictionary
            splits: Optional list of split names to filter
        """
        problems = []
        parser_name = (parse_as or name).lower()
        inferred_domain_name = parser_name if parser_name in {
            "imobench",
            "imo2025",
            "lcb_pro",
            "prbench",
            "satbench",
            "gpqa_diamond",
            "simpleqa",
            "hle_text_only",
        } else name

        if parser_name == "lcb_pro":
            for item in data.get("problems", []):
                # Filter by splits if specified
                if splits is not None:
                    item_split = item.get("metadata", {}).get("split")
                    if item_split not in splits:
                        continue
                
                if item.get("problem_title", None) is not None:
                    item["metadata"]["problem_data"] = item["problem_title"]
                if item.get("platform", None) is not None:
                    item["metadata"]["platform"] = item["platform"]
                problem = BenchmarkProblem(
                    id=str(item.get("problem_id", "")),
                    problem=item["problem_statement"],
                    ground_truth=None,
                    domain=item.get("domain", self._infer_domain(inferred_domain_name)),
                    difficulty=item.get("difficulty", "unknown"),
                    metadata=item.get("metadata", {}),
                )
                problems.append(problem)
        elif parser_name == "imobench":
            for item in data.get("problems", []):
                # Filter by splits if specified
                if splits is not None:
                    item_split = item.get("metadata", {}).get("split")
                    if item_split not in splits:
                        continue
                
                problem = BenchmarkProblem(
                    id=str(item.get("id", len(problems))),
                    problem=item["problem"],
                    ground_truth=item.get("ground_truth"),
                    domain=item.get("domain", self._infer_domain(inferred_domain_name)),
                    difficulty=item.get("difficulty"),
                    test_cases=item.get("test_cases"),
                    metadata=item.get("metadata", {}),
                )
                problems.append(problem)
        elif parser_name == "imo2025":
            for item in data.get("problems", []):
                # IMO 2025 uses the same structure as imobench
                problem = BenchmarkProblem(
                    id=str(item.get("id", len(problems))),
                    problem=item["problem"],
                    ground_truth=item.get("ground_truth"),
                    domain=item.get("domain", self._infer_domain(inferred_domain_name)),
                    difficulty=item.get("difficulty"),
                    test_cases=item.get("test_cases"),
                    metadata=item.get("metadata", {}),
                )
                problems.append(problem)
        elif parser_name == "prbench":
            unique_problems = set()
            for item in data.get("problems", []):
                if splits is not None:
                    item_split = item["field"]
                    if item_split not in splits:
                        continue

                conversation = item["conversation"]
                conversation_str = json.dumps(conversation, indent=4)
                rubric_str = json.dumps(item["rubric"], indent=4)

                if conversation_str in unique_problems:
                    continue

                unique_problems.add(conversation_str)

                problem = BenchmarkProblem(
                    id = str(item.get("task")),
                    problem = conversation_str,
                    ground_truth = rubric_str,
                    domain="professional_reasoning",
                    difficulty=None,
                    metadata=item["metadata"]
                )
                problems.append(problem)
        elif parser_name == "satbench":
            for idx, item in enumerate(data.get("problems", [])):
                # Filter by splits if specified (e.g., satisfiable, unsatisfiable)
                if splits is not None:
                    item_satisfiable = "satisfiable" if item.get("satisfiable") else "unsatisfiable"
                    if item_satisfiable not in splits:
                        continue

                # Extract components
                scenario = item.get("scenario", "")
                conditions = item.get("conditions", [])
                question = item.get("question", "")
                variable_mapping = item.get("variable_mapping", "")

                # Format conditions as a numbered list string
                conditions_str = "\n".join(conditions) if conditions else ""

                # Build the problem text using the specified template
                problem_text = f"<scenario>\n{scenario}\n\n<conditions>\n{conditions_str}\n\n<question>\n{question}"

                # Ground truth includes satisfiability and explanation
                ground_truth_dict = {
                    "satisfiable": item.get("satisfiable"),
                    "reason": item.get("sat_reason") if item.get("satisfiable") else item.get("unsat_reason"),
                    "readable_formula": item.get("readable"),
                }
                ground_truth_str = json.dumps(ground_truth_dict, indent=2, ensure_ascii=False)

                # Prepare metadata
                metadata = {
                    "dims": item.get("dims"),
                    "num_vars": item.get("num_vars"),
                    "num_clauses": item.get("num_clauses"),
                    "clauses": item.get("clauses"),
                    "readable": item.get("readable"),
                    "satisfiable": item.get("satisfiable"),
                    "scenario": scenario,
                }

                problem = BenchmarkProblem(
                    id=str(idx),
                    problem=problem_text,
                    ground_truth=ground_truth_str,
                    domain=self._infer_domain(inferred_domain_name),
                    difficulty=None,
                    metadata=metadata,
                )
                problems.append(problem)
        elif parser_name == "gpqa_diamond":
            for idx, item in enumerate(data.get("problems", [])):
                # Extract question and answer
                question = item.get("question", "")
                answer = item.get("answer", "")

                # Problem text is the multiple-choice question
                problem_text = question

                # Ground truth is just the answer letter
                ground_truth_str = answer

                # Prepare metadata
                metadata = {
                    "question": question,
                    "correct_answer": answer,
                }

                problem = BenchmarkProblem(
                    id=str(idx),
                    problem=problem_text,
                    ground_truth=ground_truth_str,
                    domain=self._infer_domain(inferred_domain_name),
                    difficulty=None,
                    metadata=metadata,
                )
                problems.append(problem)
        elif parser_name == "simpleqa":
            for idx, item in enumerate(data.get("problems", [])):
                # Extract fields
                problem_text = item.get("problem", "")
                answer = item.get("answer", "")
                topic = item.get("topic", "")
                answer_type = item.get("answer_type", "")

                # Ground truth is the answer
                ground_truth_str = answer

                # Prepare metadata
                metadata = {
                    "original_index": item.get("original_index"),
                    "topic": topic,
                    "answer_type": answer_type,
                    "multi_step": item.get("multi_step", False),
                    "requires_reasoning": item.get("requires_reasoning", False),
                    "urls": item.get("urls", ""),
                }

                problem = BenchmarkProblem(
                    id=str(idx),
                    problem=problem_text,
                    ground_truth=ground_truth_str,
                    domain=self._infer_domain(inferred_domain_name),
                    difficulty=None,
                    metadata=metadata,
                )
                problems.append(problem)
        elif parser_name == "hle_text_only":
            # HLE (Humanity's Last Exam) text-only benchmark
            for idx, item in enumerate(data):
                # Extract fields
                question = item.get("question", "")
                answer = item.get("answer", "")
                answer_type = item.get("answer_type", "")

                # For multiple choice, the question already includes choices
                problem_text = question
                ground_truth_str = answer

                # Prepare metadata
                metadata = {
                    "id": item.get("id"),
                    "answer_type": answer_type,
                    "raw_subject": item.get("raw_subject", ""),
                    "category": item.get("category", ""),
                    "author_name": item.get("author_name", ""),
                    "rationale": item.get("rationale", ""),
                }

                problem = BenchmarkProblem(
                    id=item.get("id", str(idx)),
                    problem=problem_text,
                    ground_truth=ground_truth_str,
                    domain=self._infer_domain(inferred_domain_name),
                    difficulty=None,
                    metadata=metadata,
                )
                problems.append(problem)

        return Benchmark(
            name=name,
            problems=problems,
            description=data.get("description"),
            metadata=data.get("metadata", {}),
        )

    def _infer_domain(self, benchmark_name: str) -> str:
        """Infer domain from benchmark name."""
        domain_mapping = {
            "imobench": "math",
            "imo2025": "math",
            "lcb_pro": "coding",
            "prbench": "professional_reasoning",
            "hle": "general",
            "hle_text_only": "general",
            "satbench": "sat_solving",
            "gpqa_diamond": "science_qa",
            "simpleqa": "factual_qa",
        }
        return domain_mapping.get(benchmark_name, "general")

    def list_available(self) -> list[str]:
        """List available benchmarks in the data directory."""
        if not self.data_dir.exists():
            return []

        return [f.stem for f in self.data_dir.glob("*.json")]

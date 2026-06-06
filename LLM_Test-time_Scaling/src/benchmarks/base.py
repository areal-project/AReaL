"""Base classes for benchmarks."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkProblem:
    """A single problem from a benchmark."""

    id: str
    problem: str
    ground_truth: Optional[str] = None
    domain: str = "general"
    difficulty: Optional[str] = None
    test_cases: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Benchmark:
    """A benchmark dataset."""

    name: str
    problems: List[BenchmarkProblem]
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        """Get number of problems."""
        return len(self.problems)

    def __getitem__(self, idx: int) -> BenchmarkProblem:
        """Get problem by index."""
        return self.problems[idx]

    def filter_by_domain(self, domain: str) -> "Benchmark":
        """Filter problems by domain."""
        filtered = [p for p in self.problems if p.domain == domain]
        return Benchmark(
            name=f"{self.name}_{domain}",
            problems=filtered,
            description=self.description,
            metadata=self.metadata,
        )

    def filter_by_difficulty(self, difficulty: str) -> "Benchmark":
        """Filter problems by difficulty."""
        filtered = [p for p in self.problems if p.difficulty == difficulty]
        return Benchmark(
            name=f"{self.name}_{difficulty}",
            problems=filtered,
            description=self.description,
            metadata=self.metadata,
        )

    def sample(self, n: int, seed: int = 42) -> "Benchmark":
        """Sample n problems from the benchmark."""
        import random

        random.seed(seed)
        sampled = random.sample(self.problems, min(n, len(self.problems)))
        return Benchmark(
            name=f"{self.name}_sample_{n}",
            problems=sampled,
            description=self.description,
            metadata=self.metadata,
        )

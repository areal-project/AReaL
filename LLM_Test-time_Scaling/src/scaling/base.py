"""Base classes for test-time scaling."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Solution:
    """A solution with optional feedback and metadata."""

    content: str
    score: Optional[float] = None
    feedback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingResult:
    """Result of a test-time scaling operation."""

    final_solution: Solution
    all_solutions: List[Solution]
    iterations: int
    total_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

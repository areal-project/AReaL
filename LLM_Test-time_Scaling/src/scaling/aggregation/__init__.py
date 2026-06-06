"""Aggregation strategies for test-time scaling."""

from .base import AggregationStrategy
from .select_best import SelectBestAggregation
from .generate_from_n import GenerateFromNAggregation
from .llm_scoring import LLMScoringAggregation
from .voting import VotingAggregation
from .pairwise_comparison import PairwiseComparisonAggregation
from .gt_scoring import GTScoringAggregation
from .llm_voting import LLMVotingAggregation

__all__ = [
    "AggregationStrategy",
    "SelectBestAggregation",
    "GenerateFromNAggregation",
    "LLMScoringAggregation",
    "VotingAggregation",
    "PairwiseComparisonAggregation",
    "GTScoringAggregation",
    "LLMVotingAggregation",
]

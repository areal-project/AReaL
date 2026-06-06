"""Test-time scaling functions for reflection and aggregation."""

from .base import Solution, ScalingResult
from .reflection import (
    ReflectionStrategy,
    NoFeedbackReflection,
    SelfEvaluationReflection,
    GroundTruthReflection,
    GroundTruthSimpleReflection,
    CodeExecutionReflection,
)
from .aggregation import (
    AggregationStrategy,
    SelectBestAggregation,
    GenerateFromNAggregation,
    LLMScoringAggregation,
    VotingAggregation,
    PairwiseComparisonAggregation,
)

__all__ = [
    "Solution",
    "ScalingResult",
    "ReflectionStrategy",
    "NoFeedbackReflection",
    "SelfEvaluationReflection",
    "GroundTruthReflection",
    "CodeExecutionReflection",
    "GroundTruthSimpleReflection",
    "AggregationStrategy",
    "SelectBestAggregation",
    "GenerateFromNAggregation",
    "LLMScoringAggregation",
    "VotingAggregation",
    "PairwiseComparisonAggregation",
]

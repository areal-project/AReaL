"""Prompt templates for test-time scaling operations."""

from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel


class PromptType(Enum):
    """Types of prompts used in test-time scaling."""

    DIRECT_GENERATION = "direct_generation"
    SELF_EVALUATION = "self_evaluation"
    GENERATION_WITH_REFLECTION = "generation_with_reflection"
    LLM_SCORING = "llm_scoring"
    GENERATE_ONE_FROM_N = "generate_one_from_n"
    SELECT_ONE_FROM_N = "select_one_from_n"
    PAIRWISE_COMPARISON = "pairwise_comparison"
    GENERATION_WITHOUT_FEEDBACK = "no_feedback_reflection"
    GENERATION_WITH_EXECUTION = "code_execution"
    LLM_VOTING = "llm_voting"
    LLM_CODING_SCORING = "code_llm_scoring"
    SCIENCE_QA_SCORING = "science_qa_llm_scoring"
    GENERATE_ONE_FROM_N_CODING = "aggregation_generate_one_from_n_coding"
    SELECT_ONE_FROM_N_CODING = "aggregation_select_one_from_n_coding"
    PAIRWISE_COMPARISON_CODING = "aggregation_pairwise_comparison_coding"
    GENERATE_ONE_FROM_N_SCIENCE_QA = "aggregation_generate_one_from_n_science_qa"
    SELECT_ONE_FROM_N_SCIENCE_QA = "aggregation_select_one_from_n_science_qa"
    PAIRWISE_COMPARISON_SCIENCE_QA = "aggregation_pairwise_comparison_science_qa"

class PromptTemplate(BaseModel):
    """A prompt template with placeholders."""

    name: str
    prompt_type: PromptType
    system_prompt: str
    user_prompt_template: str
    task_domain: str  # "math", "coding", "finance", "law", "general"

    def format(self, **kwargs: Any) -> str:
        """Format the user prompt with given parameters.

        Args:
            **kwargs: Parameters to fill in the template

        Returns:
            Formatted prompt string
        """
        return self.user_prompt_template.format(**kwargs)

    def format_with_system(self, **kwargs: Any) -> Dict[str, str]:
        """Format both system and user prompts.

        Args:
            **kwargs: Parameters to fill in the template

        Returns:
            Dictionary with 'system' and 'user' keys
        """
        return {"system": self.system_prompt, "user": self.format(**kwargs)}

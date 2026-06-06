"""Configuration management."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for LLM service."""

    provider: str = "litellm"
    model_name: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    direct_prompt_template: Optional[str] = None
    reasoning_effort: Optional[str] = Field(
        default="N/A",
        description="Reasoning effort for aggregation workflows: 'auto', 'low', 'medium', 'high', or None. "
                    "'auto' enables 'high' for gpt-oss models, None for others."
    )


class EvaluationConfig(BaseModel):
    """Configuration for evaluation."""

    benchmark: str = "general"
    evaluator_type: str = "llm_judge"
    provider: str = "litellm"
    model_name: Optional[str] = ""
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    local_data_dir: Optional[str] = None  # For LCB-Pro: local data directory with testcases
    benchmark_file: Optional[str] = None  # Optional: explicit benchmark JSON file path
    language: Optional[str] = None  # For code evaluation: language (e.g., "cpp", "python")
    splits: Optional[List[str]] = None  # For LCB-Pro: list of split names to load (e.g., ["biannual_2024_7_12"])
    service_url: Optional[str] = None  # For remote code verify service: service URL(s), comma-separated for load balancing

class ReflectionConfig(BaseModel):
    """Configuration for reflection strategy."""

    strategy: str = "self_evaluation"
    n_iterations: int = 1
    n_samples_per_iteration: int = 1
    reasoning_effort: Optional[str] = Field(
        default="auto",
        description="Reasoning effort for reflection workflows: 'auto', 'low', 'medium', 'high', or None. "
                    "'auto' enables 'high' for gpt-oss models, None for others."
    )
    use_detailed_results: bool = Field(
        default=True,
        description="For code_execution strategy: If True, use detailed test results (with input/output/answer). "
                    "If False, use basic results (only passed count, error type, and checker stdout)."
    )


class AggregationConfig(BaseModel):
    """Configuration for aggregation strategy."""

    strategy: str = "select_best"
    apply_at_each_turn: bool = False
    reasoning_effort: Optional[str] = Field(
        default="N/A",
        description="Reasoning effort for aggregation workflows: 'auto', 'low', 'medium', 'high', or None. "
                    "'auto' enables 'high' for gpt-oss models, None for others."
    )


class Config(BaseModel):
    """Main configuration."""

    llm: LLMConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    experiment_name: str = "default"
    output_dir: str = "outputs"
    seed: int = 42
    max_concurrent_problems: int = Field(
        default=128,
        description="Maximum number of problems to process concurrently"
    )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Load config from YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

    def to_yaml(self, yaml_path: Path) -> None:
        """Save config to YAML file."""
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from file or environment variables.

    Args:
        config_path: Path to YAML config file. If None, uses default config.

    Returns:
        Config object
    """
    if config_path and config_path.exists():
        return Config.from_yaml(config_path)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL", "gpt-4")

    return Config(
        llm=LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "litellm"),
            model_name=model_name,
            api_key=api_key,
        )
    )

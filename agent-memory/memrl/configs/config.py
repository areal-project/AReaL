"""
Configuration management for the Memp system.

This module defines Pydantic models for configuration management,
supporting both YAML and JSON configuration files.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ConfigDict
import yaml
import json


class LLMConfig(BaseModel):
    """Configuration for LLM provider."""
    
    provider: str = Field(default="openai", description="LLM provider name")
    api_key: str = Field(default="sk-8", 
                        description="API key for authentication")
    base_url: Optional[str] = Field(default="https://api.openai.com/v1", description="Base URL for API")
    # Optional Azure/OpenAI-style API versioning. Some runners pass this through.
    api_version: Optional[str] = Field(default=None, description="Optional API version for some providers (e.g. Azure OpenAI)")
    model: str = Field(default="gpt-4.1-mini", description="Model name")
    temperature: float = Field(default=0.7, ge=0, le=2, description="Generation temperature")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Maximum tokens")
    
    @field_validator('api_key')
    @classmethod
    def api_key_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('API key cannot be empty')
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for embedding provider."""
    
    provider: str = Field(default="openai", description="Embedding provider name")
    api_key: str = Field(default="sk-8",
                        description="API key for authentication")
    base_url: Optional[str] = Field(default="https://api.openai.com/v1", description="Base URL for API")
    api_version: Optional[str] = Field(default=None, description="Optional API version for some providers (e.g. Azure OpenAI)")
    model: str = Field(default="text-embedding-3-large", description="Embedding model name")
    dimension: int = Field(
        default=4096,
        gt=0,
        description="Embedding vector dimension (must match the embedding model; e.g. 4096 for Qwen3-Embedding-8B, 3072 for text-embedding-3-large)",
    )
    max_text_len: int = Field(
        default=8196,
        ge=0,
        description="Maximum characters per query before chunked embedding (0 disables chunking)",
    )
    
    @field_validator('api_key')
    @classmethod
    def api_key_must_not_be_empty(cls, v):
        if not v:
            raise ValueError('API key cannot be empty')
        return v


class MemoryConfig(BaseModel):
    """Configuration for memory system."""
    
    # Strategy configuration
    build_strategy: str = Field(default="proceduralization", 
                               description="Build strategy: trajectory, script, proceduralization")
    retrieve_strategy: str = Field(default="query",
                                  description="Retrieve strategy: random, query, avefact") 
    update_strategy: str = Field(default="adjustment",
                                description="Update strategy: vanilla, validation, adjustment")
    
    # Memory parameters
    k_retrieve: int = Field(default=1, gt=-1, description="Number of memories to retrieve")
    max_keywords: int = Field(default=8, gt=0, description="Maximum keywords for AveFact")
    confidence_threshold: float = Field(default=0.0, ge=0, le=1, 
                                       description="Minimum similarity threshold")
    memory_confidence: float = Field(default=100.0, ge=0, le=100,
                                    description="Confidence score for new memories")
    add_similarity_threshold: float = Field(default=0.9,
                                    description="similarity_threshold for add memory")
    memory_budget_tokens: int = Field(default=0,
                                      description="Token budget (character-level) for injected memory context. 0 means unlimited (no truncation).")
    # MemOS configuration
    mos_config_path: str = Field(default="configs/mos_config.json",
                                description="Path to MemOS configuration file")
    user_id: str = Field(default="memp_user", description="User ID for memory management")
    sim_norm_mean: float = Field(default=0, description="Mean for similarity normalization")
    sim_norm_std: float = Field(default=0, description="Standard deviation for similarity normalization")

    # Optional checkpoint loading for runners that support resuming memory state.
    load_from_checkpoint: bool = Field(default=False, description="Whether to load memory state from a checkpoint snapshot")
    checkpoint_path: Optional[str] = Field(default=None, description="Path to a memory checkpoint snapshot to load (if enabled)")

    script_detail_level: str = Field(
        default="abstract",
        description=(
            "Proceduralization/script build only: how detailed the LLM-generated success "
            "memory should be. 'abstract' (default, unchanged) = high-level strategy steps, "
            "no concrete commands. 'detailed' = keep exact commands/flags/paths so the stored "
            "memory is command-level replayable (helps OS/DB where vague steps aren't reusable). "
            "Env var MEMRL_LLB_SCRIPT_DETAIL overrides this."
        ),
    )


class EnvironmentConfig(BaseModel):
    """Configuration for environment-specific settings."""
    
    # ALFWorld settings  
    alfworld_config_path: str = Field(default="configs/base_config.yaml",
                                     description="Path to ALFWorld configuration")
    alfworld_env_type: str = Field(default="AlfredTWEnv",
                                  description="ALFWorld environment type")


class ExperimentConfig(BaseModel):
    """Configuration for experimental settings."""

    # Experiment parameters
    experiment_name: str = Field(..., description="Name for the experiment trail")
    algorithm: str = Field(
        default="rl", description="Algorithm to run: memp, rl, mdp, rlmdp, slow_rl"
    )
    val_before_train: bool = Field(
        default=True, description="Whether to run validation before training starts"
    )
    enable_value_driven: bool = Field(default=False, description="Whether to use rl")
    random_seed: Optional[int] = Field(
        default=42, description="Random seed for reproducibility"
    )
    mode: str = Field(
        default="train", description="Control whether to only train or test"
    )

    # BCB evaluation toggles (used only by run/run_bcb.py).
    bcb_run_validation: bool = Field(
        default=False,
        description=(
            "BigCodeBench only: whether to run the validation phase. "
            "If false, the BCB runner will run train-only by default."
        ),
    )

    # Optional tracing (LLB JSONL) controlled by YAML (YAML overrides env vars when set).
    trace_jsonl_path: Optional[str] = Field(
        default=None,
        description="If set, enable LLB per-task JSONL tracing and write to this path.",
    )
    trace_sample_filter: Optional[str] = Field(
        default=None,
        description="Optional task filter for tracing: digits=first N tasks; or comma-separated sample_index list.",
    )

    # LLB retrieval behavior toggles (used only by LLB runner entrypoints).
    llb_use_z_score_normalization: bool = Field(
        default=True,
        description=(
            "LLB only: whether to z-score normalize similarity/q before hybrid scoring. "
            "Set false to rank by raw similarity + raw q (closer to legacy behavior)."
        ),
    )
    llb_q_floor: Optional[float] = Field(
        default=None,
        description=(
            "LLB only: if set, overrides rl_config.q_floor for LLB runs. "
            "Mimics memory_rl's q_floor to prevent Q from dropping below this value "
            "(applied during both Q update and retrieval scoring)."
        ),
    )
    llb_dedup_by_task_id: bool = Field(
        default=False,
        description=(
            "LLB only: when selecting the final top-k retrieved memories, deduplicate by task_id "
            "(fallback to sample_index/id; if still missing, treated as unique). "
            "This mimics memory_rl's task-level Phase-B behavior and reduces same-task repeats."
        ),
    )
    llb_reflection_prompt: str = Field(
        default="legacy",
        description=(
            "LLB DB/OS only: which failure-reflection prompt variant to use. "
            "'legacy' = original prompt (default, unchanged, DB-worded for both). "
            "'v2' = corrected, TASK-AWARE prompt: for DB it states the real "
            "tuple/Decimal/order contract; for OS it states the real grading contract "
            "(a hidden check command verifies system state via exit_code==0; no answer/"
            "tuple/SQL) and steers to OS failure modes (perm/ownership bits, symlink, "
            "unverified state). Env var MEMRL_LLB_REFLECTION_PROMPT overrides this."
        ),
    )

    # LLB-specific parameters
    task: str = Field(default="db", description="Task type for LLB: db, os, kg")
    split_file: str = Field(default="", description="Path to LLB dataset split file")
    valid_file: Optional[str] = Field(
        default=None, description="Path to LLB validation dataset file"
    )

    num_sections: int = Field(default=5, description="Number of sections to split the training data into")
    batch_size: int = Field(default=32, description="Number of parallel environments for sampling")
    max_steps: int = Field(default=30, description="Max steps per episode during training and evaluation")
    max_recent_turns: int = Field(default=20, description="Max recent interaction turns (obs+action pairs) kept in agent context window. Lower for models with long outputs.")
    strip_thinking: bool = Field(default=False, description="Strip <think> blocks from trajectory before building memories. Enable for thinking models (e.g. Qwen3.6).")
    max_trajectory_len: int = Field(default=0, description="If > 0, truncate trajectory to this many chars before memory builder LLM call. 0 = no truncation.")
    max_history_response_chars: int = Field(default=0, description="If > 0, cap stored assistant response in history to this many chars (tail). 0 = no cap. Use for thinking models where fallback reasoning text leaks into content.")
    no_think: bool = Field(default=False, description="Append /no_think to system prompt to disable model thinking mode (Qwen3 family).")
    force_think: bool = Field(default=False, description="Append /think to system prompt to force model to output <think> tags (Qwen3 family). Enables client-side thinking strip.")
    valid_interval: int = Field(default=1, description="Run evaluation on the validation set every N sections. Set to 0 to disable.")
    test_interval: int = Field(default=1, description="Run evaluation on the test set every N sections. Set to 0 to disable.")
    eval_runs: int = Field(default=1, description="Number of independent eval passes at the final section (>1 enables multi-eval with mean/CI/CSR). Intermediate sections always run a single eval.")
    eval_temperature: float = Field(default=0.0, description="Sampling temperature for the stochastic eval passes in multi-eval (run 1 is always temp=0.0/deterministic).")
    ckpt_save_every_n_batches: int = Field(default=0, description="LLB only: save a mid-section memory snapshot every N mini-batches (0=disabled, section-level ckpt only). Enables resume after mid-section preemption.")
    ckpt_max_keep: int = Field(default=3, description="LLB only: max number of recent batch-level snapshots to keep per section (older ones pruned).")
    region_cluster_init_step: int = Field(
        default=500,
        ge=0,
        description=(
            "LLB Region only: global training step at which to run the first "
            "mid-section clustering (default 500). Set to 0 to disable the "
            "mid-section trigger; the end-of-section clustering fallback remains active. "
            "MEMRL_REGION_CLUSTER_INIT_STEP overrides this value at runtime."
        ),
    )
    batch_checkpoint_interval: int = Field(
        default=10,
        ge=1,
        description="ALFWorld: write a full resumable batch snapshot every N mini-batches.",
    )
    batch_checkpoint_keep: int = Field(
        default=3,
        ge=1,
        description="ALFWorld: keep this many recent full batch snapshots per section.",
    )
    dataset_ratio: float = Field(default=0.7, description="Proportion of files randomly selected for training (rest used for validation)")
    few_shot_path: str = Field(default='data/alfworld/alfworld_examples.json', description="Path for alfworld examples")

    # WebShop data paths
    file_path: Optional[str] = Field(default=None, description="Path to IN-DIST product file for WebShop")
    ood_file_path: Optional[str] = Field(default=None, description="Path to OOD product file for WebShop")
    split_info_path: Optional[str] = Field(default=None, description="Path to split_info.json for WebShop train/val/ood goals")

    bon: int = Field(default=0, description="Run BoN-evaluation on the val/test for N trails")
    hle_categories: Optional[List[str]] = Field(default=None, description="Subset of HLE categories to keep")
    hle_category_ratio: Optional[float] = Field(default=None, description="Per-category sampling ratio (0,1]")
    train_valid_split: float = Field(default=0.8, description="Ratio to split training and validation sets")
    ckpt_eval_enabled: bool = Field(default=False, description="Whether to evaluate by loading historical checkpoints")
    ckpt_eval_path: Optional[str] = Field(default=None, description="Path to experiment or snapshot directory for ckpt eval")
    ckpt_resume_enabled: bool = Field(default=False, description="Whether to resume training from a checkpoint snapshot")
    ckpt_resume_path: Optional[str] = Field(default=None, description="Path to experiment or snapshot directory for ckpt resume")
    ckpt_resume_epoch: Optional[int] = Field(default=None, description="Epoch index (1-based) to resume from")
    baseline_mode: Optional[str] = Field(default=None, description="Baseline mode: passk or reflection")
    baseline_k: int = Field(default=10, description="Baseline rounds (k) for pass@k/reflection")
    # Output settings
    # Default points at the shared admin checkpoint area so disk usage doesn't pile up
    # inside the repo (/storage/openpsi/users/yl/agent-memory/MemRL). Per-benchmark
    # subdirs are: bigcodebench/, alfworld/. Override via cfg.experiment.output_dir
    # or --output_dir for one-off runs. See docs/CHECKPOINT_STORAGE.md.
    output_dir: str = Field(default="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench", description="Directory for experiment outputs")
    save_trajectories: bool = Field(default=True, description="Save detailed trajectories")
    save_memories: bool = Field(default=True, description="Save memory snapshots")

    # Logging settings
    enable_logging: bool = Field(default=True, description="Enable detailed logging")
    log_level: str = Field(default="INFO", description="Logging level")

    # ALFWorld single-bank zero-shot holdout (used by run_alfworld.py)
    holdout_subtask: Optional[str] = Field(
        default=None,
        description=(
            "ALFWorld only: if set (e.g. 'alf/pick_and_place_simple'), games of "
            "this subtask are excluded from train and memory pool. The runner "
            "evaluates zero-shot transfer to this subtask. CLI --holdout_subtask "
            "overrides this."
        ),
    )
    val_lambda_max: Optional[float] = Field(
        default=None,
        description=(
            "If set, eval phase temporarily lowers region_manager.shrinkage_lambda_max "
            "so retrieval is region-utility-dominated. CLI --val_lambda_max overrides."
        ),
    )

    # v10 ALFWorld holdout retrieval (see docs/ALFWORLD_V10_HOLDOUT_IMPL.md)
    # When set, bypasses zero-shot transfer (info-empty per offline meta-eval)
    # and routes holdout queries through D1-based retrieval modes.
    holdout_retrieval_mode: Optional[str] = Field(
        default=None,
        description=(
            "ALFWorld holdout only: 'pure_d1' / 'hybrid' / 'sim_d1' / None. "
            "None = default zero-shot transfer (current behavior). "
            "pure_d1 = inject fixed top-k by region D1 quality. "
            "hybrid = top-N anchors by D1 + remaining by sim*D1 (recommended). "
            "sim_d1 = all top-k by sim*D1 with pool=holdout_pool_size. "
            "CLI --holdout_retrieval_mode overrides."
        ),
    )
    holdout_pool_size: int = Field(
        default=500,
        description=(
            "v10 holdout retrieval: candidate pool size for sim recall before D1 rerank. "
            "Used by hybrid/sim_d1 modes. Default 500 (analysis/region_quality_uplift.py "
            "shows uplift converges at pool>=500). CLI --holdout_pool_size overrides."
        ),
    )
    holdout_d1_anchors: int = Field(
        default=3,
        description=(
            "v10 hybrid mode: number of D1 anchor memories injected without sim. "
            "Remaining (k - holdout_d1_anchors) slots filled by sim*D1. "
            "Default 3 (Codex recommendation). CLI --holdout_d1_anchors overrides."
        ),
    )

class RLConfig(BaseModel):
    """Configuration for reinforcement learning parameters."""

    epsilon: float = Field(default=0.1, description="ε-greedy exploration probability")
    tau: float = Field(default=0.35, description="Unknown detection threshold on similarity")
    alpha: float = Field(default=0.3, description="Q-learning step size (learning rate)")
    gamma: float = Field(default=0.0, description="Discount factor (default 0 for single-step)")
    q_init_pos: float = Field(default=0.0, description="Optimistic initialization for positive Q-values")
    q_init_neg: float = Field(default=0.0, description="Initialization for negative Q-values")
    q_floor: Optional[float] = Field(
        default=None,
        description=(
            "Minimum allowed Q value (optional). "
            "Only applied by runners that explicitly enable it (e.g., LLB via experiment.llb_q_floor)."
        ),
    )
    success_reward: float = Field(default=1.0, description="Reward for successful outcome")
    failure_reward: float = Field(default=-1.0, description="Reward for failure outcome")
    # Retrieval filtering threshold used by runners when calling MemoryService.retrieve_query(...).
    # (Kept separate from `tau` to avoid conflating unknown-detection vs retrieval filtering.)
    sim_threshold: float = Field(default=0.5, description="Similarity threshold for retrieval filtering")
    topk: int = Field(default=5, description="Candidate set size for value-aware selection")
    novelty_threshold: float = Field(default=0.85, description="Similarity threshold to treat as non-novel (merge)")
    recency_boost: float = Field(default=0.0, description="Optional recency weight for prioritization")
    reward_merge_gain: float = Field(default=0.1, description="Gain for attributing success to close memories")
    q_min_threshold: float = Field(default=-0.8, description="Threshold for q min")
    weight_sim: float = Field(default=0.5, description="Weight for similarity in combined score")
    weight_q: float = Field(default=0.5, description="Weight for Q-value in combined score")

class MempConfig(BaseModel):
    """Main configuration class for the Memp system."""
    
    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    rl_config: RLConfig = Field(default_factory=RLConfig)
    # Global settings
    project_name: str = Field(default="memp", description="Project name")
    version: str = Field(default="0.1.0", description="Project version")
    
    model_config = ConfigDict(extra="forbid")  # Don't allow extra fields
        
    @classmethod
    def from_yaml(cls, config_path: str) -> "MempConfig":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            MempConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                config_data = {}
                
            return cls(**config_data)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {e}")
    
    @classmethod
    def from_json(cls, config_path: str) -> "MempConfig":
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            MempConfig instance
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            return cls(**config_data)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {e}")
    
    def to_yaml(self, output_path: str) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            output_path: Path to save YAML configuration
        """
        config_dict = self.model_dump()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_json(self, output_path: str, indent: int = 2) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            output_path: Path to save JSON configuration
            indent: JSON indentation level
        """
        config_dict = self.model_dump()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=indent)
    
    def get_strategy_config(self):
        """Get strategy configuration for MemoryService."""
        from ..service.strategies import StrategyConfiguration
        
        return StrategyConfiguration.from_strings(
            build=self.memory.build_strategy,
            retrieve=self.memory.retrieve_strategy,
            update=self.memory.update_strategy
        )
    
    def validate_paths(self) -> None:
        """
        Validate that all specified paths exist.
        
        Raises:
            FileNotFoundError: If required paths don't exist
        """
        paths_to_check = [
            (self.memory.mos_config_path, "MemOS config file"),
            (self.environment.alfworld_config_path, "ALFWorld config file"),
        ]
        
        # Only check TravelPlanner data dir if it's not the default relative path
        if not self.environment.travelplanner_data_dir.startswith("../"):
            paths_to_check.append((self.environment.travelplanner_data_dir, "TravelPlanner data directory"))
        
        for path, description in paths_to_check:
            if not Path(path).exists():
                print(f"Warning: {description} not found at {path}")
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        strategy_str = f"{self.memory.build_strategy}+{self.memory.retrieve_strategy}+{self.memory.update_strategy}"
        return f"MempConfig(strategy={strategy_str}, llm={self.llm.model}, embedding={self.embedding.model})"

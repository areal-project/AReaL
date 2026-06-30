"""Utilities for SWE-bench agent training with AReaL."""

from dataclasses import dataclass, field

from areal.api.cli_args import PPOConfig


@dataclass
class SWEEnvConfig:
    """Environment configuration for AReaL-SWEAgent-backed SWE-bench training.

    Attributes:
        dataset_path: Path to the SWE-bench JSONL dataset file.
        agent_type: AReaL-SWEAgent agent type to train, e.g. ``swe`` or ``cc``.
        agent_config: Generic AReaL-SWEAgent config name. When set, this overrides
            the compatibility fields below.
        swe_agent_config: Compatibility config field for ``agent_type=swe``.
        cc_agent_config: Compatibility config field for ``agent_type=cc``.
        agent_root: Root directory of the external AReaL-SWEAgent checkout.
        swe_agent_root: Legacy alias for ``agent_root``.
        llm_model: Optional LLM model override for OH/OpenCode/Codex agents.
        opencode_provider: Optional OpenCode provider override.
        codex_provider: Optional Codex provider override.
        step_limit: Maximum number of agent interaction steps per episode.
        max_completion_tokens: Maximum completion tokens for the agent LLM.
        timeout: Maximum time allowed for a single episode in seconds.
    """

    dataset_path: str = field(
        default="",
        metadata={"help": "Path to the SWE-bench JSONL dataset file."},
    )
    agent_type: str = field(
        default="swe",
        metadata={
            "help": (
                "AReaL-SWEAgent agent type to run. Supported by AReaL-SWEAgent main: "
                "'swe', 'cc', 'oh', 'opencode', and 'codex'."
            )
        },
    )
    agent_config: str = field(
        default="",
        metadata={
            "help": (
                "Generic AReaL-SWEAgent YAML config name. When non-empty, overrides "
                "swe_agent_config / cc_agent_config."
            )
        },
    )
    swe_agent_config: str = field(
        default="1_0_0/min-swe-agent-train-top1",
        metadata={
            "help": (
                "Name of the AReaL-SWEAgent YAML config under the external AReaL-SWEAgent "
                "checkout. Defaults to the Qwen SWE-RL training config."
            )
        },
    )
    cc_agent_config: str = field(
        default="train_cc_time3600",
        metadata={
            "help": (
                "Name of the AReaL-SWEAgent YAML config used when agent_type='cc'. "
                "Kept separate for compatibility with swe/main configs."
            )
        },
    )
    agent_root: str = field(
        default="",
        metadata={
            "help": (
                "Root directory of the external AReaL-SWEAgent checkout. Defaults to "
                "../AReaL-SWEAgent relative to the AReaL repository when unset."
            )
        },
    )
    swe_agent_root: str = field(
        default="",
        metadata={
            "help": (
                "Legacy alias for agent_root / AWEAGENT_ROOT. Kept so older "
                "SWE launch scripts keep working."
            )
        },
    )
    llm_model: str = field(
        default="",
        metadata={
            "help": "Optional model name override for OH/OpenCode/Codex agents."
        },
    )
    opencode_provider: str = field(
        default="",
        metadata={"help": "Optional provider override for OpenCode agents."},
    )
    codex_provider: str = field(
        default="",
        metadata={"help": "Optional provider override for Codex agents."},
    )
    step_limit: int = field(
        default=100,
        metadata={"help": "Maximum number of agent interaction steps per episode."},
    )
    max_completion_tokens: int = field(
        default=16384,
        metadata={"help": "Maximum completion tokens for the agent LLM."},
    )
    timeout: float = field(
        default=1800.0,
        metadata={"help": "Maximum time allowed for a single episode in seconds."},
    )


@dataclass
class SWEPPOConfig(PPOConfig):
    """PPO configuration with SWE-bench-specific settings."""

    econfig: SWEEnvConfig = field(default_factory=SWEEnvConfig)
    should_accept_fn: str | None = field(
        default=None,
        metadata={
            "help": "Import path of the filter function for accepting rollout samples."
        },
    )

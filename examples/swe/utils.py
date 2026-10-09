"""Utilities for SWE-bench agent training with AReaL."""

from dataclasses import dataclass, field

from examples.swe.arena_types import ArenaRewardRefConfig as ArenaRewardRefConfig
from examples.swe.arena_types import ArenaStreamConfig

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

    dataset_source: str = field(
        default="jsonl",
        metadata={
            "help": "Dataset and agent backend: 'jsonl' or 'arena'.",
            "choices": ["jsonl", "arena"],
        },
    )
    dataset_path: str = field(
        default="",
        metadata={"help": "Path to the SWE-bench JSONL dataset file."},
    )
    stream_id: str = field(
        default="",
        metadata={
            "help": (
                "Arena Stream id. When empty, the first active Stream returned by "
                "the Arena OpenAPI is used."
            )
        },
    )
    arena_streams: list[ArenaStreamConfig] = field(
        default_factory=list,
        metadata={
            "help": (
                "Optional inline Arena Stream mixture. The default epoch uses "
                "every Stream row once; sampling_weight controls deterministic "
                "interleaving and explicit subset allocation."
            )
        },
    )
    arena_streams_yaml_b64: str = field(
        default="",
        metadata={
            "help": (
                "Optional base64-encoded inline YAML containing a top-level "
                "'streams' list. "
                "Mutually exclusive with arena_streams and arena_streams_file."
            )
        },
    )
    arena_streams_file: str = field(
        default="",
        metadata={
            "help": (
                "Optional YAML/JSON file containing a top-level 'streams' list. "
                "Mutually exclusive with inline arena_streams."
            )
        },
    )
    arena_mixture_epoch_size: int = field(
        default=0,
        metadata={
            "help": (
                "Prompt rows in a deterministic weighted Arena virtual epoch. The "
                "value cannot exceed the unique source rows; zero uses the complete "
                "raw union without batch-padding repeats."
            )
        },
    )
    arena_result_dump_dir: str = field(
        default="",
        metadata={
            "help": (
                "Optional directory for mode-0600 per-process Arena result JSONL "
                "audit shards. Empty disables full raw-result persistence."
            )
        },
    )
    arena_result_dump_max_bytes: int = field(
        default=1_000_000,
        metadata={
            "help": "Maximum serialized Arena raw payload bytes per audit record."
        },
    )
    arena_base_url: str = field(
        default="",
        metadata={
            "help": (
                "Arena OpenAPI base URL. Defaults to the ARENA_OPENAPI_BASE "
                "environment variable."
            )
        },
    )
    arena_request_timeout: float = field(
        default=60.0,
        metadata={"help": "Arena Stream and dataset request timeout in seconds."},
    )
    arena_request_retries: int = field(
        default=3,
        metadata={"help": "Retries for transient Arena HTTP request failures."},
    )
    arena_poll_interval: float = field(
        default=5.0,
        metadata={"help": "Arena task-result polling interval in seconds."},
    )
    arena_registration_timeout: float = field(
        default=180.0,
        metadata={"help": "Arena LLM registration request timeout in seconds."},
    )
    arena_registration_probe_interval: float = field(
        default=60.0,
        metadata={
            "help": (
                "Seconds between worker-scoped Arena Session Gateway liveness "
                "probes while tasks are unfinished."
            )
        },
    )
    arena_llm_route_mode: str = field(
        default="gateway",
        metadata={
            "help": (
                "Arena LLM route: 'gateway' registers one temporary Arena model "
                "per rollout; 'session_gateway' reuses one public Arena route per "
                "rollout worker and selects each sample with a session capability; "
                "'direct' "
                "injects the rollout proxy's per-session URL and key into the "
                "Harness task."
            ),
            "choices": ["gateway", "session_gateway", "direct"],
        },
    )
    arena_llm_protocol: str = field(
        default="",
        metadata={
            "help": (
                "Optional Arena Harness client protocol label. Leave empty to infer "
                "from the Stream Harness. The AReaL proxy is always registered as "
                "an OpenAI Chat Completions upstream."
            ),
            "choices": ["", "anthropic", "responses", "chat_completions"],
        },
    )
    arena_harness: str = field(
        default="",
        metadata={
            "help": (
                "Optional Arena Harness key and version passed to launch_one_task, "
                "for example 'claude-code-with-skills@5.0.1'."
            )
        },
    )
    arena_task_envs: dict[str, str] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Additional environment variables passed to launch_one_task. "
                "MODEL_NAME, BASE_URL, and API_KEY are managed by AReaL."
            )
        },
    )
    arena_reward_threshold: float | None = field(
        default=None,
        metadata={
            "help": (
                "If set, map Arena rewards >= this threshold to 1.0 and lower "
                "rewards to 0.0 when no transform is configured. Configured "
                "transforms receive it as reward_threshold, and it also classifies "
                "transformed rewards for pass@k metrics."
            )
        },
    )
    arena_reward_transform_fn: str = field(
        default="",
        metadata={
            "help": (
                "Optional import path for a callable reward transform receiving "
                "(reward, data) and returning a float. The callable must accept "
                "reward_threshold when arena_reward_threshold is configured."
            )
        },
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
        metadata={"help": "Optional model name override for OH/OpenCode/Codex agents."},
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

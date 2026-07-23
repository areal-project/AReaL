"""Configuration for SWE SFT training with AReaL."""

from dataclasses import dataclass, field

from areal.api.cli_args import SFTConfig


@dataclass
class SweDataConfig:
    """SWE-specific data processing configuration."""

    filter_errors: bool = field(
        default=True,
        metadata={
            "help": "Discard pairs whose current segment contains a tool result "
            "with is_error=True. Set to false to keep all pairs."
        },
    )
    pre_split: bool = field(
        default=False,
        metadata={
            "help": "Input JSONL is already in pair format "
            '(each line: {"messages": [...]}). '
            "Skip trajectory splitting and error filtering."
        },
    )
    num_proc: int = field(
        default=4,
        metadata={"help": "Number of parallel workers for tokenization."},
    )
    strip_all_thinking: bool = field(
        default=False,
        metadata={
            "help": "Strip <think>...</think> from ALL assistant turns "
            "including the training target. By default only context "
            "turns are stripped."
        },
    )
    no_tools: bool = field(
        default=False,
        metadata={
            "help": "Do not pass tool definitions to apply_chat_template. "
            "By default, tools are auto-extracted from the data and "
            "rendered in the system prompt (e.g. Qwen3 '# Tools' block)."
        },
    )

    skip_pretokenized_filter: bool = field(
        default=False,
        metadata={
            "help": "Skip max_length filtering when loading a pre-tokenized "
            "dataset. Useful when the dataset was already filtered during "
            "pretokenization to avoid NFS cache conflicts from concurrent "
            "dataset.filter() calls across ranks."
        },
    )
    filter_empty_tool_calls: bool = field(
        default=False,
        metadata={
            "help": "Discard pairs whose training-target assistant turn has "
            "no text content but has tool_calls (silent tool invocations)."
        },
    )
    filter_bare_text_tool_calls: bool = field(
        default=False,
        metadata={
            "help": "Discard pairs whose training-target assistant turn has "
            "text content without <think> tags and has tool_calls."
        },
    )
    truncate_task_notifications: bool = field(
        default=False,
        metadata={
            "help": "Truncate trajectories at the first <task-notification> "
            "that follows a pure-text assistant turn. Removes noise from "
            "background task completions."
        },
    )
    parse_tool_call_args: bool = field(
        default=False,
        metadata={
            "help": "Convert OpenAI JSON-string tool_calls.arguments to dicts "
            "before apply_chat_template. Required by GLM-4.x / GLM-5.x "
            "templates; leave at the default (False) for Qwen / Llama / "
            "Bailing, which expect the standard string form."
        },
    )
    max_no_thinking_ratio: float | None = field(
        default=None,
        metadata={
            "help": "Maximum ratio of non-thinking pairs to thinking pairs. "
            "For example, 1.0 gives 1:1 balance, 2.0 allows up to 2x "
            "non-thinking pairs per thinking pair. "
            "None (default) disables balancing."
        },
    )
    split_mode: str = field(
        default="pair",
        metadata={
            "help": "Sample construction mode: 'pair' (default) splits "
            "trajectories into progressive pairs; 'trajectory' keeps "
            "the full trajectory as a single training sample."
        },
    )
    random_strip_thinking_prob: float = field(
        default=0.0,
        metadata={
            "help": "Probability of stripping thinking from each target assistant "
            "turn. 0.0 = no stripping (default), 1.0 = strip all. "
            "Works in both pair mode and trajectory mode."
        },
    )
    random_strip_thinking_seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducible thinking stripping decisions."},
    )
    n_thinking_variants: int = field(
        default=1,
        metadata={
            "help": "Number of thinking-pattern variants per trajectory. "
            "1 = no augmentation (default). K > 1 = augment each "
            "trajectory into K variants: the first preserves all "
            "thinking, the rest randomly strip with "
            "random_strip_thinking_prob."
        },
    )
    cleanup_processed_dataset: bool = field(
        default=True,
        metadata={
            "help": "Remove the processed dataset cache directory after training. "
            "Set to false to keep it for faster restarts."
        },
    )
    dump_samples: int = field(
        default=50,
        metadata={
            "help": "Number of random samples to dump for inspection after "
            "dataset processing. Each sample is saved as .txt and .json "
            "in a dumped_samples/ directory alongside logs. "
            "Set to 0 to disable, -1 to dump all."
        },
    )


@dataclass
class SweSFTConfig(SFTConfig):
    """SFT configuration with SWE-specific data processing settings."""

    swe: SweDataConfig = field(default_factory=SweDataConfig)

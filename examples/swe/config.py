"""Configuration for SWE SFT training with AReaL."""

from dataclasses import dataclass, field
from typing import Literal

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
    parse_tool_call_args: bool = field(
        default=False,
        metadata={
            "help": "Convert OpenAI JSON-string tool_calls.arguments to dicts "
            "before apply_chat_template for templates that require structured "
            "arguments. Leave disabled for templates that expect the standard "
            "OpenAI string form."
        },
    )
    split_mode: Literal["pair", "trajectory"] = field(
        default="pair",
        metadata={
            "help": "Sample construction mode: 'pair' (default) splits "
            "trajectories into progressive pairs; 'trajectory' keeps "
            "the full trajectory as a single training sample."
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

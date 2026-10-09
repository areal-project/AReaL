# SPDX-License-Identifier: Apache-2.0

"""Command-line tools for inspecting and preprocessing SWE SFT datasets."""

import json
import os

from .pipeline import (
    _load_full_trajectories,
    _load_presplit_pairs,
    _load_trajectory_pairs,
    _tokenize_samples,
    _write_swe_sft_artifact_marker,
)
from .tokenization import (
    DATASET_NUM_PROC,
    _dump_samples,
    _patch_chat_template_for_training,
)


def main():
    import argparse
    import sys

    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(
        description="Verify SWE SFT pair generation and loss masking.",
    )
    parser.add_argument("path", help="Path to SWE trajectory JSONL file")
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-8B",
        help="HuggingFace tokenizer name or path (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Filter samples exceeding this token length",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=None,
        help="Number of pairs to process.  Controls loading, tokenization,"
        " display, and export.  Default: all pairs.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help=f"Number of parallel workers (default: min(cpu_count, {DATASET_NUM_PROC}))",
    )
    parser.add_argument(
        "--save-pairs",
        "-o",
        default=None,
        metavar="FILE",
        help='Save cleaned pairs to FILE (JSONL, each line: {"messages": [...]}).',
    )
    parser.add_argument(
        "--pre-split",
        action="store_true",
        help='Input is already in pair format (each line: {"messages": [...]}).'
        " Skip trajectory splitting and error filtering.",
    )
    parser.add_argument(
        "--no-filter-errors",
        action="store_true",
        help="Keep pairs whose current segment contains tool results with "
        "is_error=True (by default these are discarded).",
    )
    parser.add_argument(
        "--save-tokenized",
        default=None,
        metavar="DIR",
        help="Save the tokenized dataset to DIR (Arrow format). "
        "The saved directory can be used directly as the dataset path "
        "during training, skipping all processing.",
    )
    parser.add_argument(
        "--strip-all-thinking",
        action="store_true",
        help="Strip <think>...</think> from ALL assistant turns including "
        "the training target. By default only context turns are stripped.",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Do not pass tool definitions to apply_chat_template. "
        "By default, tools are auto-extracted from the data and rendered "
        "in the system prompt (e.g. Qwen3 '# Tools' block).",
    )
    parser.add_argument(
        "--parse-tool-call-args",
        action="store_true",
        help="Convert OpenAI JSON-string tool_calls.arguments to dicts "
        "before apply_chat_template. Enable this only when the selected "
        "tokenizer template expects structured argument objects.",
    )
    parser.add_argument(
        "--filter-empty-tool-calls",
        action="store_true",
        help="Discard pairs whose training-target assistant turn has no "
        "text content but has tool_calls (silent tool invocations).",
    )
    parser.add_argument(
        "--filter-bare-text-tool-calls",
        action="store_true",
        help="Discard pairs whose training-target assistant turn has text "
        "content without <think> tags and has tool_calls.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["pair", "trajectory"],
        default="pair",
        help="Sample construction mode. 'pair' (default): split trajectories "
        "into progressive pairs. 'trajectory': keep the full trajectory "
        "as a single sample with all assistant turns as targets.",
    )
    parser.add_argument(
        "--dump-samples",
        default=None,
        metavar="DIR",
        help="Save sampled pairs to DIR, one file per pair. Each file "
        "contains the rendered text and a token-by-token table with "
        "token id, decoded text, and loss_mask.",
    )
    parser.add_argument(
        "--dump-n",
        type=int,
        default=None,
        help="Number of pairs to dump when --dump-samples is set. "
        "Default: all pairs. -1 also means all.",
    )
    args = parser.parse_args()

    filter_errors = not args.no_filter_errors
    strip_all_thinking = args.strip_all_thinking
    filter_empty_tool_calls = args.filter_empty_tool_calls
    filter_bare_text_tool_calls = args.filter_bare_text_tool_calls

    # --- Load ---
    split_mode = args.split_mode
    error_indices_list = None

    if split_mode == "trajectory":
        samples, error_indices_list, tools_list = _load_full_trajectories(
            args.path,
            filter_errors=filter_errors,
            filter_empty_tool_calls=filter_empty_tool_calls,
            filter_bare_text_tool_calls=filter_bare_text_tool_calls,
        )
        label = "trajectories"
    elif args.pre_split:
        samples, tools_list = _load_presplit_pairs(
            args.path,
            strip_all_thinking=strip_all_thinking,
        )
        label = "pairs"
    else:
        samples, tools_list = _load_trajectory_pairs(
            args.path,
            filter_errors=filter_errors,
            strip_all_thinking=strip_all_thinking,
            filter_empty_tool_calls=filter_empty_tool_calls,
            filter_bare_text_tool_calls=filter_bare_text_tool_calls,
        )
        label = "pairs"

    # --- Slice + stats ---
    total = len(samples)
    if args.num_samples is not None:
        samples = samples[: args.num_samples]
        tools_list = tools_list[: args.num_samples] if tools_list else tools_list
        if error_indices_list is not None:
            error_indices_list = error_indices_list[: args.num_samples]

    print(f"Total {label}:  {total}")
    if args.num_samples is not None:
        print(f"Using:          {len(samples)}")

    if samples:
        lengths = [len(s) for s in samples]
        print(
            f"Messages/sample: min={min(lengths)}, "
            f"max={max(lengths)}, avg={sum(lengths) / len(lengths):.1f}"
        )
    if error_indices_list is not None:
        n_masked = sum(len(e) for e in error_indices_list)
        print(f"Masked segments: {n_masked} (loss=0)")

    # --- Save cleaned samples as JSONL ---
    if args.save_pairs:
        with open(args.save_pairs, "w", encoding="utf-8") as fout:
            err_iter = error_indices_list or [None] * len(samples)
            tl_iter = tools_list if tools_list else [None] * len(samples)
            for sample, sample_tools, err_idxs in zip(samples, tl_iter, err_iter):
                record = {"messages": sample}
                if sample_tools is not None:
                    record["tools"] = sample_tools
                if err_idxs:
                    record["error_indices"] = err_idxs
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(samples)} {label} to {args.save_pairs}")

    # --- Tokenize / Dump ---
    dump_dir = args.dump_samples if args.dump_samples else None
    need_tokenize = args.save_tokenized

    # When --save-tokenized is set, auto-dump 50 samples alongside it
    # unless the user explicitly set --dump-samples or --dump-n 0.
    if need_tokenize and not dump_dir and args.dump_n != 0:
        dump_dir = os.path.join(args.save_tokenized, "dumped_samples")

    if not need_tokenize and not dump_dir:
        sys.exit(0)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    _patch_chat_template_for_training(tok)
    if args.dump_n is not None:
        dump_n = args.dump_n
    elif args.dump_samples:
        # Explicit --dump-samples without --dump-n: dump all
        dump_n = -1 if args.num_samples is None else args.num_samples
    elif need_tokenize:
        # Auto-dump with --save-tokenized: default 50
        dump_n = 50
    else:
        dump_n = -1

    # Dump can run independently without full tokenization.
    if dump_dir and dump_n != 0:
        dump_tools = None if args.no_tools else tools_list
        _dump_samples(
            samples,
            tok,
            dump_tools,
            dump_dir,
            dump_n,
            split_mode=split_mode,
            error_indices_list=error_indices_list,
            parse_tool_call_args=args.parse_tool_call_args,
        )

    if not need_tokenize:
        sys.exit(0)

    ds = _tokenize_samples(
        samples,
        tools_list,
        tok,
        split_mode=split_mode,
        error_indices_list=error_indices_list,
        max_length=args.max_length,
        num_proc=args.num_proc,
        no_tools=args.no_tools,
        parse_tool_call_args=args.parse_tool_call_args,
    )

    print(f"\nTokenized: {len(ds)} samples")
    if args.save_tokenized:
        ds.save_to_disk(args.save_tokenized)
        _write_swe_sft_artifact_marker(args.save_tokenized)
        print(f"Saved tokenized dataset ({len(ds)} samples) to {args.save_tokenized}")


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0

"""SWE SFT loading, processing, distributed caching, and public dataset API."""

import json
import os
import random
import shutil
import time

from datasets import Dataset

from areal.utils import logging

from .messages import (
    _balance_thinking_pairs,
    _clean_message,
    _find_segments,
    _iter_jsonl_records,
    _log_thinking_augmentation_stats,
    _msg_has_thinking,
    _prepare_trajectory,
    _split_and_filter,
    _truncate_at_task_notification,
)
from .tokenization import (
    DATASET_NUM_PROC,
    _detect_template_pattern,
    _dump_samples,
    _patch_chat_template_for_training,
    _TokenizeAndMask,
)

logger = logging.getLogger("SWESFTDataset")

_RANK0_CACHE_TIMEOUT = 36000
_RANK0_CACHE_POLL_INTERVAL = 5


def _load_trajectory_pairs(
    path: str,
    filter_errors: bool = True,
    strip_all_thinking: bool = False,
    filter_empty_tool_calls: bool = False,
    filter_bare_text_tool_calls: bool = False,
    truncate_task_notifications: bool = False,
    max_no_thinking_ratio: float | None = None,
    random_strip_thinking_prob: float = 0.0,
    random_strip_thinking_seed: int = 42,
    n_thinking_variants: int = 1,
):
    """Load trajectory JSONL and split into progressive pairs.

    When *n_thinking_variants* > 1, each trajectory is split K times:
    variant 0 preserves all thinking, variants 1~K-1 randomly strip.

    Supports nested (``conversations`` wrapper) and flat JSONL formats
    (auto-detected per record via ``_iter_jsonl_records``).

    Returns:
        Tuple of ``(all_pairs, tools)`` where *tools* is ``None`` when no
        tool definitions are found.
    """
    all_pairs = []
    all_tools = []
    records_in = 0
    total_filtered_errors = 0
    total_filtered_empty_tc = 0
    total_filtered_bare_tc = 0
    total_truncated = 0
    total_stripped_thinking = 0

    augment = n_thinking_variants > 1
    rng = (
        random.Random(random_strip_thinking_seed)
        if random_strip_thinking_prob > 0.0
        else None
    )

    if augment and random_strip_thinking_prob <= 0.0:
        logger.warning(
            "n_thinking_variants=%d but random_strip_thinking_prob=0; "
            "all variants will be identical.",
            n_thinking_variants,
        )

    # Stats collectors for augmentation logging.
    thinking_turns_per_traj = []
    total_asst_turns_per_traj = []
    patterns_per_traj = []

    for record_idx, messages, record_tools in _iter_jsonl_records(path):
        records_in = record_idx

        if truncate_task_notifications:
            truncated = _truncate_at_task_notification(messages)
            if len(truncated) < len(messages):
                total_truncated += 1
                messages = truncated

        shared_kwargs = dict(
            filter_errors=filter_errors,
            strip_all_thinking=strip_all_thinking,
            filter_empty_tool_calls=filter_empty_tool_calls,
            filter_bare_text_tool_calls=filter_bare_text_tool_calls,
        )

        if augment:
            # Variant 0: preserve all thinking.
            pairs_orig, n_err, n_empty_tc, n_bare_tc, _ = _split_and_filter(
                messages, **shared_kwargs, random_strip_thinking_prob=0.0, rng=None
            )
            total_filtered_errors += n_err
            total_filtered_empty_tc += n_empty_tc
            total_filtered_bare_tc += n_bare_tc
            all_pairs.extend(pairs_orig)
            all_tools.extend([record_tools] * len(pairs_orig))
            # Collect stats.
            segments = _find_segments(messages)
            n_think = sum(1 for s, _ in segments if _msg_has_thinking(messages[s]))
            n_asst = len(segments)
            thinking_turns_per_traj.append(n_think)
            total_asst_turns_per_traj.append(n_asst)

            # Variants 1 ~ K-1: random strip.
            variant_patterns = {frozenset()}  # original = no strip
            for _k in range(n_thinking_variants - 1):
                pairs_aug, _, _, _, n_stripped = _split_and_filter(
                    messages,
                    **shared_kwargs,
                    random_strip_thinking_prob=random_strip_thinking_prob,
                    rng=rng,
                )
                total_stripped_thinking += n_stripped
                all_pairs.extend(pairs_aug)
                all_tools.extend([record_tools] * len(pairs_aug))
                # Approximate pattern: record which pairs had their target stripped.
                # For stats, use the count as a proxy since _split_and_filter
                # doesn't return per-pair strip info.
                variant_patterns.add(frozenset([n_stripped]))
            patterns_per_traj.append(variant_patterns)
        else:
            # Single variant (original behavior).
            pairs, n_err, n_empty_tc, n_bare_tc, n_stripped = _split_and_filter(
                messages,
                **shared_kwargs,
                random_strip_thinking_prob=random_strip_thinking_prob,
                rng=rng,
            )
            total_filtered_errors += n_err
            total_filtered_empty_tc += n_empty_tc
            total_filtered_bare_tc += n_bare_tc
            total_stripped_thinking += n_stripped
            all_pairs.extend(pairs)
            all_tools.extend([record_tools] * len(pairs))

    # Log extracted tools summary.
    n_with_tools = sum(1 for t in all_tools if t is not None)
    if n_with_tools > 0:
        all_tool_names = set()
        for t_list in all_tools:
            if t_list is not None:
                for t in t_list:
                    all_tool_names.add(t.get("function", {}).get("name", "?"))
        logger.info(
            f"Extracted tools from {n_with_tools}/{len(all_tools)} pairs: "
            f"{sorted(all_tool_names)}"
        )

    filter_parts = []
    if total_truncated:
        filter_parts.append(
            f"{total_truncated} trajectories truncated at task-notification"
        )
    if total_filtered_errors:
        filter_parts.append(f"{total_filtered_errors} with tool errors")
    if total_filtered_empty_tc:
        filter_parts.append(f"{total_filtered_empty_tc} empty-content tool calls")
    if total_filtered_bare_tc:
        filter_parts.append(f"{total_filtered_bare_tc} bare-text tool calls")
    if total_stripped_thinking:
        filter_parts.append(f"{total_stripped_thinking} thinking blocks stripped")
    filter_msg = ", ".join(filter_parts) if filter_parts else "none"

    logger.info(
        f"Loaded {records_in} trajectories, "
        f"generated {len(all_pairs)} pairs "
        f"(filtered: {filter_msg})"
    )

    if augment and patterns_per_traj:
        _log_thinking_augmentation_stats(
            n_thinking_variants,
            random_strip_thinking_prob,
            records_in,
            thinking_turns_per_traj,
            total_asst_turns_per_traj,
            patterns_per_traj,
        )

    # Balance thinking / no-thinking pair ratio.
    all_pairs, all_tools = _balance_thinking_pairs(
        all_pairs, max_no_thinking_ratio, tools_list=all_tools
    )

    return all_pairs, all_tools


def _load_presplit_pairs(
    path: str,
    strip_all_thinking: bool = False,
    random_strip_thinking_prob: float = 0.0,
    random_strip_thinking_seed: int = 42,
    n_thinking_variants: int = 1,
):
    """Load pre-split pair JSONL where each line is ``{"messages": [...]}``.

    Messages are cleaned but no splitting or error-filtering is performed.
    By default, thinking is stripped from context assistant turns but
    preserved for the last assistant turn (the training target).  Set
    *strip_all_thinking* to strip from every assistant turn.

    When *n_thinking_variants* > 1, each pair is augmented: variant 0
    preserves thinking, variants 1~K-1 randomly strip the target turn.

    Also extracts per-record ``tools`` definitions so that each pair
    carries its own tools, same as ``_load_trajectory_pairs``.

    Returns:
        Tuple of ``(all_pairs, all_tools)`` where *all_tools* is a
        parallel list of per-sample tool definitions (may be ``None``).
    """
    all_pairs = []
    all_tools = []
    n_stripped = 0
    augment = n_thinking_variants > 1

    rng = (
        random.Random(random_strip_thinking_seed)
        if random_strip_thinking_prob > 0.0
        else None
    )

    def _build_pair(messages, last_asst, strip_target):
        pair = []
        for idx, m in enumerate(messages):
            is_target = m.get("role") == "assistant" and idx == last_asst
            strip = strip_all_thinking or not is_target or strip_target
            pair.append(_clean_message(m, strip_thinking=strip))
        return pair

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            if not messages:
                continue

            record_tools = record.get("tools")

            # Find the last assistant index so we can preserve its thinking.
            last_asst = None
            for i, m in enumerate(messages):
                if m.get("role") == "assistant":
                    last_asst = i

            has_thinking = (
                last_asst is not None
                and not strip_all_thinking
                and _msg_has_thinking(messages[last_asst])
            )

            if augment:
                # Variant 0: preserve all thinking.
                all_pairs.append(_build_pair(messages, last_asst, strip_target=False))
                all_tools.append(record_tools)

                # Variants 1 ~ K-1: random strip.
                for _k in range(n_thinking_variants - 1):
                    do_strip = (
                        has_thinking
                        and rng is not None
                        and rng.random() < random_strip_thinking_prob
                    )
                    if do_strip:
                        n_stripped += 1
                    all_pairs.append(
                        _build_pair(messages, last_asst, strip_target=do_strip)
                    )
                    all_tools.append(record_tools)
            else:
                # Single variant (original behavior).
                strip_target = (
                    has_thinking
                    and rng is not None
                    and rng.random() < random_strip_thinking_prob
                )
                if strip_target:
                    n_stripped += 1
                all_pairs.append(
                    _build_pair(messages, last_asst, strip_target=strip_target)
                )
                all_tools.append(record_tools)

    # Log extracted tools summary.
    n_with_tools = sum(1 for t in all_tools if t is not None)
    if n_with_tools > 0:
        all_tool_names = set()
        for t_list in all_tools:
            if t_list is not None:
                for t in t_list:
                    all_tool_names.add(t.get("function", {}).get("name", "?"))
        logger.info(
            f"Extracted tools from {n_with_tools}/{len(all_tools)} pairs: "
            f"{sorted(all_tool_names)}"
        )

    strip_msg = f", {n_stripped} thinking blocks stripped" if n_stripped else ""
    logger.info(f"Loaded {len(all_pairs)} pre-split pairs from {path}{strip_msg}")
    return all_pairs, all_tools


def _load_full_trajectories(
    path: str,
    filter_errors: bool = True,
    filter_empty_tool_calls: bool = False,
    filter_bare_text_tool_calls: bool = False,
    truncate_task_notifications: bool = False,
    random_strip_thinking_prob: float = 0.0,
    random_strip_thinking_seed: int = 42,
    n_thinking_variants: int = 1,
):
    """Load trajectory JSONL for trajectory-level training.

    Each trajectory becomes a single training sample with all assistant
    turns as targets (``loss_mask=1``).  When *filter_errors* is True,
    assistant segments with error tool responses are identified so
    tokenization can mask them (``loss_mask=0``) instead of discarding
    the entire trajectory.

    When *n_thinking_variants* > 1, each trajectory is augmented into
    K variants: the first preserves all thinking, the remaining K-1
    randomly strip thinking turns with *random_strip_thinking_prob*.

    Supports nested (``conversations`` wrapper) and flat JSONL formats
    (auto-detected per record via ``_iter_jsonl_records``).

    Returns:
        Tuple of ``(trajectories, error_indices_list, all_tools)`` where
        *trajectories* is a list of cleaned message lists,
        *error_indices_list* is a list of error segment index lists,
        and *all_tools* is a parallel list of per-sample tool definitions.
    """
    trajectories = []
    error_indices_list = []
    all_tools = []
    records_in = 0
    total_truncated = 0
    total_masked_errors = 0
    total_masked_empty_tc = 0
    total_masked_bare_tc = 0
    total_stripped_thinking = 0

    augment = n_thinking_variants > 1
    rng = (
        random.Random(random_strip_thinking_seed)
        if random_strip_thinking_prob > 0.0
        else None
    )

    if augment and random_strip_thinking_prob <= 0.0:
        logger.warning(
            "n_thinking_variants=%d but random_strip_thinking_prob=0; "
            "all variants will be identical.",
            n_thinking_variants,
        )

    # Stats collectors for augmentation logging.
    thinking_turns_per_traj = []
    total_asst_turns_per_traj = []
    patterns_per_traj = []

    for record_idx, messages, record_tools in _iter_jsonl_records(path):
        records_in = record_idx

        if truncate_task_notifications:
            truncated = _truncate_at_task_notification(messages)
            if len(truncated) < len(messages):
                total_truncated += 1
                messages = truncated

        shared_kwargs = dict(
            filter_errors=filter_errors,
            filter_empty_tool_calls=filter_empty_tool_calls,
            filter_bare_text_tool_calls=filter_bare_text_tool_calls,
        )

        if augment:
            # Variant 0: preserve all thinking (no stripping).
            result_orig = _prepare_trajectory(
                messages, **shared_kwargs, random_strip_thinking_prob=0.0, rng=None
            )
            if result_orig is None:
                continue
            cleaned_orig, masked_idxs, n_err, n_empty_tc, n_bare_tc, _ = result_orig
            trajectories.append(cleaned_orig)
            error_indices_list.append(masked_idxs)
            all_tools.append(record_tools)
            total_masked_errors += n_err
            total_masked_empty_tc += n_empty_tc
            total_masked_bare_tc += n_bare_tc

            # Collect stats: count thinking turns in this trajectory.
            segments = _find_segments(messages)
            n_think = sum(1 for s, _ in segments if _msg_has_thinking(messages[s]))
            n_asst = len(segments)
            thinking_turns_per_traj.append(n_think)
            total_asst_turns_per_traj.append(n_asst)

            # Variants 1 ~ K-1: random strip thinking.
            variant_patterns = {frozenset()}  # original = empty pattern
            for _k in range(n_thinking_variants - 1):
                result_aug = _prepare_trajectory(
                    messages,
                    **shared_kwargs,
                    random_strip_thinking_prob=random_strip_thinking_prob,
                    rng=rng,
                )
                if result_aug is None:
                    continue
                cleaned_aug, _, _, _, _, strip_pattern = result_aug
                trajectories.append(cleaned_aug)
                error_indices_list.append(masked_idxs)  # reuse
                all_tools.append(record_tools)
                total_stripped_thinking += len(strip_pattern)
                variant_patterns.add(strip_pattern)
            patterns_per_traj.append(variant_patterns)
        else:
            # Single variant (original behavior).
            result = _prepare_trajectory(
                messages,
                **shared_kwargs,
                random_strip_thinking_prob=random_strip_thinking_prob,
                rng=rng,
            )
            if result is None:
                continue
            cleaned, masked_idxs, n_err, n_empty_tc, n_bare_tc, strip_pattern = result
            trajectories.append(cleaned)
            error_indices_list.append(masked_idxs)
            all_tools.append(record_tools)
            total_masked_errors += n_err
            total_masked_empty_tc += n_empty_tc
            total_masked_bare_tc += n_bare_tc
            total_stripped_thinking += len(strip_pattern)

    # Log extracted tools summary.
    n_with_tools = sum(1 for t in all_tools if t is not None)
    if n_with_tools > 0:
        all_tool_names = set()
        for t_list in all_tools:
            if t_list is not None:
                for t in t_list:
                    all_tool_names.add(t.get("function", {}).get("name", "?"))
        logger.info(
            f"Extracted tools from {n_with_tools}/{len(all_tools)} "
            f"trajectories: {sorted(all_tool_names)}"
        )

    parts = []
    if total_truncated:
        parts.append(f"{total_truncated} trajectories truncated at task-notification")
    if total_masked_errors:
        parts.append(f"{total_masked_errors} with tool errors")
    if total_masked_empty_tc:
        parts.append(f"{total_masked_empty_tc} empty-content tool calls")
    if total_masked_bare_tc:
        parts.append(f"{total_masked_bare_tc} bare-text tool calls")
    if total_stripped_thinking:
        parts.append(f"{total_stripped_thinking} thinking blocks stripped")
    mask_msg = ", ".join(parts) if parts else "none"

    logger.info(
        f"Loaded {records_in} trajectories, "
        f"kept {len(trajectories)} for training "
        f"(masked: {mask_msg})"
    )

    if augment and patterns_per_traj:
        _log_thinking_augmentation_stats(
            n_thinking_variants,
            random_strip_thinking_prob,
            records_in,
            thinking_turns_per_traj,
            total_asst_turns_per_traj,
            patterns_per_traj,
        )

    return trajectories, error_indices_list, all_tools


def _tokenize_samples(
    messages_list,
    tools_list,
    tokenizer,
    *,
    split_mode: str = "pair",
    error_indices_list: list | None = None,
    max_length: int | None = None,
    num_proc: int | None = None,
    no_tools: bool = False,
    dump_dir: str | None = None,
    dump_n_samples: int = 0,
    parse_tool_call_args: bool = False,
):
    """Tokenize message lists into a training-ready Dataset.

    Works for both progressive pairs (``split_mode="pair"``) and
    full trajectories (``split_mode="trajectory"``).

    In pair mode, only the last assistant turn per sample gets
    ``loss_mask=1``.  In trajectory mode, all assistant turns get
    ``loss_mask=1`` except those at error segment indices.

    Args:
        tools_list: Per-sample tool definitions (parallel to
            *messages_list*).  Each element is either ``None`` or a
            list of tool dicts.
    """
    if num_proc is None:
        num_proc = max(1, min(os.cpu_count() or 1, DATASET_NUM_PROC))

    # Find representative tools for template detection.
    first_tools = None
    if tools_list:
        first_tools = next((t for t in tools_list if t is not None), None)

    if no_tools:
        tools_list = None
        first_tools = None
        logger.info("Tool definitions disabled (no_tools=True)")
    elif first_tools is not None:
        all_tool_names = set()
        for t_list in tools_list:
            if t_list is not None:
                for t in t_list:
                    all_tool_names.add(t.get("function", {}).get("name", "?"))
        logger.info(f"Using tools for chat template: {sorted(all_tool_names)}")

    if not messages_list:
        raise ValueError("No valid samples to tokenize")

    # Build dataset columns.
    data = {"messages": messages_list}
    # Serialize per-sample tools as JSON strings for the Dataset column.
    data["tools_json"] = (
        [json.dumps(t) if t else "" for t in tools_list]
        if tools_list
        else [""] * len(messages_list)
    )
    remove_cols = ["messages", "tools_json"]
    if split_mode == "trajectory":
        data["error_indices"] = error_indices_list or [[] for _ in messages_list]
        remove_cols.append("error_indices")

    dataset = Dataset.from_dict(data)
    _patch_chat_template_for_training(tokenizer)
    assistant_pattern = _detect_template_pattern(tokenizer, tools=first_tools)

    # Dump samples for inspection before the heavy map() pass.
    if dump_dir and dump_n_samples != 0:
        _dump_samples(
            messages_list,
            tokenizer,
            assistant_pattern,
            tools_list,
            dump_dir,
            dump_n_samples,
            split_mode=split_mode,
            error_indices_list=error_indices_list,
            parse_tool_call_args=parse_tool_call_args,
        )

    process_fn = _TokenizeAndMask(
        tokenizer,
        assistant_pattern,
        max_length=max_length,
        split_mode=split_mode,
        parse_tool_call_args=parse_tool_call_args,
    )

    dataset = dataset.map(process_fn, num_proc=num_proc).remove_columns(remove_cols)

    # Single filter pass: removes both apply_chat_template-failure empties and
    # overlength samples (which _TokenizeAndMask also marks as empty).
    before_filter = len(dataset)
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 0, num_proc=num_proc)
    n_filtered = before_filter - len(dataset)
    if n_filtered > 0:
        logger.info(
            f"Filtered {n_filtered} samples "
            f"(empty from template failures or exceeding max_length={max_length})"
        )

    logger.info(f"Final dataset: {len(dataset)} samples")
    return dataset


def _process_swe_sft(
    path: str,
    tokenizer,
    *,
    max_length: int | None = None,
    num_proc: int | None = None,
    pre_split: bool = False,
    filter_errors: bool = True,
    strip_all_thinking: bool = False,
    filter_empty_tool_calls: bool = False,
    filter_bare_text_tool_calls: bool = False,
    truncate_task_notifications: bool = False,
    no_tools: bool = False,
    max_no_thinking_ratio: float | None = None,
    split_mode: str = "pair",
    random_strip_thinking_prob: float = 0.0,
    random_strip_thinking_seed: int = 42,
    n_thinking_variants: int = 1,
    dump_dir: str | None = None,
    dump_n_samples: int = 0,
    parse_tool_call_args: bool = False,
):
    """Load JSONL, split into pairs, tokenize, and filter.

    Combines file loading with ``_tokenize_samples`` so that the rank-0-only
    path and the single-process path share the same logic.

    When *split_mode* is ``"trajectory"``, the full trajectory is kept as a
    single training sample with all assistant turns as targets.
    """
    error_indices_list = None

    if split_mode == "trajectory":
        messages_list, error_indices_list, tools_list = _load_full_trajectories(
            path,
            filter_errors=filter_errors,
            filter_empty_tool_calls=filter_empty_tool_calls,
            filter_bare_text_tool_calls=filter_bare_text_tool_calls,
            truncate_task_notifications=truncate_task_notifications,
            random_strip_thinking_prob=random_strip_thinking_prob,
            random_strip_thinking_seed=random_strip_thinking_seed,
            n_thinking_variants=n_thinking_variants,
        )
    elif pre_split:
        messages_list, tools_list = _load_presplit_pairs(
            path,
            strip_all_thinking=strip_all_thinking,
            random_strip_thinking_prob=random_strip_thinking_prob,
            random_strip_thinking_seed=random_strip_thinking_seed,
            n_thinking_variants=n_thinking_variants,
        )
    else:
        messages_list, tools_list = _load_trajectory_pairs(
            path,
            filter_errors=filter_errors,
            strip_all_thinking=strip_all_thinking,
            filter_empty_tool_calls=filter_empty_tool_calls,
            filter_bare_text_tool_calls=filter_bare_text_tool_calls,
            truncate_task_notifications=truncate_task_notifications,
            max_no_thinking_ratio=max_no_thinking_ratio,
            random_strip_thinking_prob=random_strip_thinking_prob,
            random_strip_thinking_seed=random_strip_thinking_seed,
            n_thinking_variants=n_thinking_variants,
        )

    return _tokenize_samples(
        messages_list,
        tools_list,
        tokenizer,
        split_mode=split_mode,
        error_indices_list=error_indices_list,
        max_length=max_length,
        num_proc=num_proc,
        no_tools=no_tools,
        dump_dir=dump_dir,
        dump_n_samples=dump_n_samples,
        parse_tool_call_args=parse_tool_call_args,
    )


def get_swe_sft_dataset(
    path: str,
    split: str | None = None,
    tokenizer=None,
    max_length: int | None = None,
    num_proc: int | None = None,
    pre_split: bool = False,
    filter_errors: bool = True,
    strip_all_thinking: bool = False,
    filter_empty_tool_calls: bool = False,
    filter_bare_text_tool_calls: bool = False,
    truncate_task_notifications: bool = False,
    no_tools: bool = False,
    skip_pretokenized_filter: bool = False,
    max_no_thinking_ratio: float | None = None,
    split_mode: str = "pair",
    random_strip_thinking_prob: float = 0.0,
    random_strip_thinking_seed: int = 42,
    n_thinking_variants: int = 1,
    cache_dir: str | None = None,
    dump_dir: str | None = None,
    dump_samples: int = 0,
    parse_tool_call_args: bool = False,
):
    """Load SWE trajectory data and convert to SFT training pairs.

    By default, tool definitions are auto-extracted from the training data's
    ``conversations[].tools`` field and passed to ``apply_chat_template``
    so that the tokenizer renders tool definitions in the system prompt
    (e.g. Qwen3 ``# Tools`` block), matching the eval-time format.
    Set *no_tools* to skip this and render without tool definitions.

    When *split_mode* is ``"trajectory"``, the full trajectory is kept as a
    single training sample with all assistant turns as targets
    (``loss_mask=1``).  Error segments are masked (``loss_mask=0``)
    when *filter_errors* is True, instead of being discarded.
    Thinking is preserved by default but can be randomly stripped
    per-turn via *random_strip_thinking_prob* (both modes).

    In distributed (SPMD) mode, only rank 0 performs the heavy processing
    (JSONL loading, pair splitting, tokenization) and saves the result as
    an Arrow dataset to *cache_dir*.  Other ranks wait for rank 0 to
    finish and then load the cached dataset directly via memory-mapped I/O.

    Args:
        path: Path to the JSONL file containing SWE trajectories, or a
            directory containing a pre-tokenized Arrow dataset (saved by
            ``python -m areal.dataset.swe_sft --save-tokenized``).
        split: Unused, kept for API compatibility.
        tokenizer: Tokenizer with ``apply_chat_template`` support.
            Not required when loading a pre-tokenized dataset.
        max_length: Max token length.  Longer sequences are filtered out.
        num_proc: Number of parallel workers for tokenization.
            Defaults to ``min(os.cpu_count(), DATASET_NUM_PROC)``.
        pre_split: If True, treat input as pre-split pairs (each line is
            ``{"messages": [...]}``) instead of full trajectories.
        filter_errors: If True (default), discard pairs whose current segment
            contains a tool result with ``is_error=True``.  In trajectory
            mode, sets ``loss_mask=0`` for error segments instead.
            Set to False to keep/train all regardless of tool errors.
        strip_all_thinking: If True, strip ``<think>...</think>`` from every
            assistant turn including the training target.
            Ignored in trajectory mode (thinking is always preserved).
        filter_empty_tool_calls: If True, discard pairs whose training-target
            assistant turn has no text content but has tool_calls.
        filter_bare_text_tool_calls: If True, discard pairs whose
            training-target assistant turn has text without ``<think>``
            tags and has tool_calls.
        truncate_task_notifications: If True, truncate trajectories at the
            first ``<task-notification>`` that follows a pure-text assistant
            turn, removing noise from background task completions.
        no_tools: If True, do not pass tool definitions to
            ``apply_chat_template`` even if the data contains them.
        skip_pretokenized_filter: If True, skip the ``max_length`` filter
            when loading a pre-tokenized dataset.  Useful when the dataset
            was already filtered during pretokenization and you want to
            avoid NFS cache conflicts from concurrent ``dataset.filter()``
            calls across ranks.
        max_no_thinking_ratio: Maximum ratio of non-thinking pairs to thinking
            pairs.  For example, ``1.0`` gives 1:1, ``2.0`` gives 1:2.
            ``None`` (default) disables balancing.
        split_mode: ``"pair"`` (default) splits trajectories into
            progressive pairs.  ``"trajectory"`` keeps the full trajectory
            as a single sample — all assistant turns are targets with
            ``loss_mask=1``, error segments are masked instead of filtered.
        random_strip_thinking_prob: Probability of stripping thinking from
            each target assistant turn.  0.0 (default) = no stripping,
            1.0 = strip all.  Works in both pair and trajectory mode.
        random_strip_thinking_seed: Random seed for reproducible thinking
            stripping decisions.
        n_thinking_variants: Number of thinking-pattern variants per
            trajectory.  ``1`` (default) = no augmentation.  ``K > 1``
            = augment each trajectory into K variants: the first
            preserves all thinking, the rest randomly strip with
            *random_strip_thinking_prob*.
        cache_dir: Directory to save/load the processed Arrow dataset.
            When set in distributed mode, rank 0 processes the data and
            saves here; other ranks load from this directory.  If the
            directory already contains a completed cache (``.done`` marker),
            all ranks load from it directly without reprocessing.
        dump_dir: Directory to write sample dump files (``.txt`` + ``.json``).
            Only rank 0 writes.  Set to None to disable.
        dump_samples: Number of random samples to dump.  ``-1`` = all,
            ``0`` = disabled.
        parse_tool_call_args: If True, convert OpenAI JSON-string
            ``tool_calls.arguments`` to dicts before ``apply_chat_template``.
            Required by GLM-4.x / GLM-5.x templates; leave at the default
            (False) for Qwen / Llama / Bailing.

    Returns:
        A HuggingFace ``Dataset`` with ``input_ids`` and ``loss_mask`` columns.
    """
    from datasets import load_from_disk

    # Pre-tokenized Arrow dataset: load directly, skip all processing.
    if os.path.isdir(path):
        logger.info(f"Loading pre-tokenized dataset from {path}")
        dataset = load_from_disk(path)

        if max_length is not None and not skip_pretokenized_filter:
            before_filter = len(dataset)
            dataset = dataset.filter(
                lambda x: len(x["input_ids"]) <= max_length, num_proc=num_proc
            )
            logger.info(
                f"Filtered {before_filter - len(dataset)} samples "
                f"exceeding max_length={max_length}"
            )

        logger.info(f"Final dataset: {len(dataset)} samples")
        return dataset

    # --- Shared kwargs for _process_swe_sft ---
    process_kwargs = dict(
        max_length=max_length,
        num_proc=num_proc,
        pre_split=pre_split,
        filter_errors=filter_errors,
        strip_all_thinking=strip_all_thinking,
        filter_empty_tool_calls=filter_empty_tool_calls,
        filter_bare_text_tool_calls=filter_bare_text_tool_calls,
        truncate_task_notifications=truncate_task_notifications,
        no_tools=no_tools,
        max_no_thinking_ratio=max_no_thinking_ratio,
        split_mode=split_mode,
        random_strip_thinking_prob=random_strip_thinking_prob,
        random_strip_thinking_seed=random_strip_thinking_seed,
        n_thinking_variants=n_thinking_variants,
        dump_dir=dump_dir,
        dump_n_samples=dump_samples,
        parse_tool_call_args=parse_tool_call_args,
    )

    # --- Distributed rank-0-only processing ---
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if cache_dir is not None and world_size > 1:
        done_marker = os.path.join(cache_dir, ".done")
        meta_path = os.path.join(cache_dir, ".meta.json")
        cache_meta = {
            "version": 1,
            "path": path,
            "tokenizer": getattr(tokenizer, "name_or_path", None),
            "process_kwargs": {
                k: v
                for k, v in process_kwargs.items()
                if k not in ("dump_dir", "dump_n_samples")
            },
        }

        def _filter_by_max_length(ds):
            if max_length is None:
                return ds
            before = len(ds)
            # Length via arrow list offsets: avoids decoding every row to
            # Python lists, which for long-context datasets costs minutes of
            # startup per rank while (on a validated cache) removing nothing —
            # build-time _TokenizeAndMask already filtered with this max_length.
            import pyarrow.compute as pc

            # ds.data is the underlying arrow table; a freshly built dataset
            # carries an indices mapping (from .filter views) whose row count
            # differs. Materialize the view first (no-op for load_from_disk).
            if getattr(ds, "_indices", None) is not None:
                ds = ds.flatten_indices()
            lengths = pc.list_value_length(ds.data.column("input_ids")).to_pylist()
            keep = [i for i, n in enumerate(lengths) if n <= max_length]
            ds = ds.select(keep)
            if len(ds) < before:
                logger.info(
                    f"Rank {rank}: filtered {before - len(ds)} samples "
                    f"exceeding max_length={max_length}"
                )
            if len(ds) == 0:
                raise ValueError(
                    f"processed dataset at {cache_dir} has 0 samples after "
                    f"max_length={max_length} filtering"
                )
            return ds

        def _load_valid_cache():
            if not os.path.exists(meta_path):
                raise ValueError(f"cached dataset metadata is missing: {meta_path}")
            with open(meta_path) as f:
                cached_meta = json.load(f)
            if cached_meta != cache_meta:
                raise ValueError(
                    f"cached dataset metadata does not match current SWE settings: "
                    f"{meta_path}"
                )
            dataset = load_from_disk(cache_dir)
            if len(dataset) == 0:
                raise ValueError(f"cached dataset is empty: {cache_dir}")
            return dataset

        def _wait_for_valid_cache():
            start = time.monotonic()
            last_error = None
            while True:
                if os.path.exists(done_marker):
                    try:
                        return _load_valid_cache()
                    except Exception as e:
                        last_error = e
                elapsed = time.monotonic() - start
                if elapsed > _RANK0_CACHE_TIMEOUT:
                    raise TimeoutError(
                        f"Waited {_RANK0_CACHE_TIMEOUT}s for rank 0 to rebuild "
                        f"a valid dataset cache at {cache_dir}. Last error: {last_error}"
                    )
                time.sleep(_RANK0_CACHE_POLL_INTERVAL)

        # Fast path: cache from a previous run (or rank 0 already finished).
        if os.path.exists(done_marker):
            if rank == 0:
                try:
                    logger.info(
                        f"Rank {rank}: loading cached processed dataset from {cache_dir}"
                    )
                    dataset = _load_valid_cache()
                    dataset = _filter_by_max_length(dataset)
                    logger.info(f"Final dataset: {len(dataset)} samples")
                    return dataset
                except Exception as e:
                    logger.warning(
                        "Rank 0: invalid processed dataset cache at %s (%s); "
                        "rebuilding it.",
                        cache_dir,
                        e,
                    )
                    shutil.rmtree(cache_dir, ignore_errors=True)
            else:
                try:
                    logger.info(
                        f"Rank {rank}: loading cached processed dataset from {cache_dir}"
                    )
                    dataset = _load_valid_cache()
                    dataset = _filter_by_max_length(dataset)
                    logger.info(f"Final dataset: {len(dataset)} samples")
                    return dataset
                except Exception as e:
                    logger.warning(
                        "Rank %d: cached processed dataset at %s is not usable "
                        "(%s); waiting for rank 0 to rebuild it.",
                        rank,
                        cache_dir,
                        e,
                    )
                    dataset = _wait_for_valid_cache()
                    dataset = _filter_by_max_length(dataset)
                    logger.info(
                        f"Rank {rank}: loaded rebuilt dataset ({len(dataset)} samples)"
                    )
                    return dataset

        if rank == 0:
            # Rank 0: do the heavy processing and save for other ranks.
            dataset = _process_swe_sft(path, tokenizer, **process_kwargs)
            if len(dataset) == 0:
                raise RuntimeError(
                    "SWE SFT preprocessing produced 0 samples; refusing to cache "
                    "an empty processed_dataset."
                )
            shutil.rmtree(cache_dir, ignore_errors=True)
            os.makedirs(cache_dir, exist_ok=True)
            dataset.save_to_disk(cache_dir)
            with open(meta_path, "w") as f:
                json.dump(cache_meta, f, sort_keys=True)
            # Write marker AFTER save completes so readers see a consistent dir.
            with open(done_marker, "w") as f:
                f.write(str(len(dataset)))
            logger.info(
                f"Rank 0: saved processed dataset "
                f"({len(dataset)} samples) to {cache_dir}"
            )
            dataset = _filter_by_max_length(dataset)
            return dataset
        else:
            # Other ranks: wait for rank 0, then load with meta validation so a
            # cache rebuilt for different settings (or mid-rmtree) is never
            # silently loaded as this rank's dataset.
            logger.info(f"Rank {rank}: waiting for rank 0 to process dataset...")
            dataset = _wait_for_valid_cache()
            dataset = _filter_by_max_length(dataset)
            logger.info(f"Rank {rank}: loaded cached dataset ({len(dataset)} samples)")
            return dataset

    # --- Non-distributed or no cache_dir: process in current process ---
    return _process_swe_sft(path, tokenizer, **process_kwargs)

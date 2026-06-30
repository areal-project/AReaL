"""SWE SFT dataset loader.

Loads SWE-bench trajectory data and converts it into progressive SFT
training pairs.  Each trajectory is split at assistant-turn boundaries
so that every pair ends with an assistant segment (assistant message +
its subsequent tool responses).

Example trajectory::

    [system, user, asst1, tool1a, tool1b, asst2, tool2, asst3]

Produces three pairs::

    Pair 1: [system, user, asst1, tool1a, tool1b]
    Pair 2: [system, user, asst1, tool1a, tool1b, asst2, tool2]
    Pair 3: [system, user, asst1, tool1a, tool1b, asst2, tool2, asst3]

In each pair, only the **last** assistant segment is trained (loss=1);
earlier assistant turns are treated as context (loss=0).

By default, pairs whose current segment contains a tool result with
``is_error=True`` are discarded.  Set ``filter_errors=False`` to keep them.

The file is organized into the following sections:

1. **Constants & Infrastructure** — shared constants, distributed sync
2. **Cleaning** — message content transforms (thinking tags, field cleanup)
3. **Filters** — keep/discard predicates (error, empty, bare-text, truncation)
4. **Splitting** — trajectory → progressive pairs (segment detection + split)
5. **Tokenization** — template detection, render→tokenize→loss_mask, dump
6. **Pipeline** — loading, processing, distributed cache, public API
7. **CLI** — ``python -m areal.dataset.swe_sft`` entry point
"""

import json
import os
import random
import re
import time

from datasets import Dataset

from areal.utils import logging

logger = logging.getLogger("SWESFTDataset")


# ============================================================
# 1. Constants & Infrastructure
# ============================================================

DATASET_NUM_PROC = 1

# Timeout (seconds) for non-rank-0 workers waiting for rank 0 to finish
# dataset processing.  Large datasets with tokenization can take minutes;
# 30 min is a generous upper bound.
_RANK0_CACHE_TIMEOUT = 36000
_RANK0_CACHE_POLL_INTERVAL = 5


def _wait_for_marker(marker_path: str):
    """Block until *marker_path* exists on disk, with timeout."""
    start = time.monotonic()
    while not os.path.exists(marker_path):
        elapsed = time.monotonic() - start
        if elapsed > _RANK0_CACHE_TIMEOUT:
            raise TimeoutError(
                f"Waited {_RANK0_CACHE_TIMEOUT}s for rank 0 to finish dataset "
                f"processing (marker: {marker_path}). Check rank 0 logs."
            )
        time.sleep(_RANK0_CACHE_POLL_INTERVAL)


def _extract_messages(record, record_idx):
    """Extract messages and tools from a parsed JSONL record.

    Handles nested (``conversations`` wrapper) and flat formats.
    Warns if multiple conversations are present.

    Returns:
        Tuple of ``(messages, record_tools)``.  *messages* may be empty.
    """
    convs = record.get("conversations", [])
    if convs:
        if len(convs) > 1:
            logger.warning(
                "Record %d has %d conversations, using only the last one.",
                record_idx,
                len(convs),
            )
        conv = convs[-1]
        return conv.get("messages", []), conv.get("tools")
    return record.get("messages", []), record.get("tools")


def _set_messages(record, messages):
    """Write *messages* back into *record* (inverse of ``_extract_messages``).

    Used by the ``--save-trajectories`` CLI path to update truncated
    messages in the original record structure before serialization.
    """
    convs = record.get("conversations", [])
    if convs:
        convs[-1]["messages"] = messages
    else:
        record["messages"] = messages


def _iter_jsonl_records(path):
    """Iterate trajectory JSONL records.

    Yields ``(record_idx, messages, record_tools)`` tuples.  Handles
    nested (``conversations`` wrapper) vs flat format auto-detection
    via ``_extract_messages``.  Records with empty messages are skipped.

    Warns about multi-user trajectories which break think-tag rendering
    in templates with ``ns.last_query_index`` logic (e.g. Bailing).
    """
    record_idx = 0
    n_multi_user = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record_idx += 1
            messages, record_tools = _extract_messages(record, record_idx)
            if not messages:
                continue
            n_user = sum(1 for m in messages if m.get("role") == "user")
            if n_user > 1:
                n_multi_user += 1
                if n_multi_user <= 3:
                    logger.warning(
                        "Record %d has %d user messages. Templates with "
                        "ns.last_query_index logic (e.g. Bailing) will NOT "
                        "render <think> for assistant turns before the last "
                        "user message. Consider filtering!",
                        record_idx,
                        n_user,
                    )
            yield record_idx, messages, record_tools
    if n_multi_user > 0:
        logger.warning(
            "Total %d/%d records have multiple user messages. "
            "These may produce no-think training signal.",
            n_multi_user,
            record_idx,
        )


# ============================================================
# 2. Cleaning — message content transforms
# ============================================================

# Match reasoning blocks with any common tag variant:
#   <think>...</think>      (Qwen standard)
#   <thinking>...</thinking> (Claude)
# The opening and closing tag names need not match exactly — mixed pairs
# like ``<think>...</thinking>`` (seen in distillation data) are handled.
_THINK_OPEN_RE = re.compile(r"<think(?:ing)?>")
_THINK_CLOSE_RE = re.compile(r"</think(?:ing)?>")
_THINK_RE = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL)


def _normalize_thinking_tags(content):
    """Normalise all thinking tag variants to ``<think>``/``</think>``.

    Distillation data from different models may use ``<thinking>`` (Claude)
    vs ``<think>`` (Qwen).  Non-standard variants are multi-token for the
    Qwen tokenizer which breaks think/tool_call boundaries.
    """
    if not content:
        return content
    content = _THINK_OPEN_RE.sub("<think>", content)
    content = _THINK_CLOSE_RE.sub("</think>", content)
    return content


def _extract_thinking(content):
    """Strip thinking blocks from *content*.

    Callers must run ``_normalize_thinking_tags`` first so that all
    tag variants have been converted to ``<think>``/``</think>``.

    Returns:
        Cleaned content with thinking blocks removed, or the original
        content unchanged if no thinking tags are found.
    """
    if not content:
        return content
    cleaned = _THINK_RE.sub("", content).strip()
    return cleaned if cleaned != content.strip() else content


def _clean_message(msg, strip_thinking=True, ensure_thinking=False):
    """Remove non-standard fields before tokenization.

    Keeps only the fields expected by tokenizer chat templates:
    role, content, reasoning_content (for assistant), tool_calls
    (for assistant), tool_call_id (for tool).

    Handles thinking content in two representations:

    - Inline ``<think>...</think>`` tags in ``content``
    - Separate ``reasoning_content`` field (DeepSeek, Qwen3 API style)

    If both are present, inline tags take priority and
    ``reasoning_content`` is dropped with a warning to avoid double
    thinking blocks in the rendered template.

    Args:
        msg: Raw message dict.
        strip_thinking: If True, remove thinking from assistant messages
            (both inline ``<think>`` tags and ``reasoning_content``).
            Used for context turns.  If False, preserve thinking as-is
            (used for the training-target assistant turn).
        ensure_thinking: If True, inject inline ``<think>\n</think>``
            on assistant turns that lack a thinking block (either
            inline or in ``reasoning_content``).  Requires the
            patched Bailing template (via
            ``_patch_chat_template_for_training``) which detects
            ``had_think_tags`` and preserves empty think blocks.
    """
    cleaned = {"role": msg["role"]}

    # Handle content — some assistant messages have content=None when
    # they only contain tool_calls.  Preserve None so chat templates
    # that distinguish None vs "" render correctly.
    content = msg.get("content")
    # Some APIs (DeepSeek, Qwen3 with enable_thinking) return thinking
    # in a separate ``reasoning_content`` field instead of inline
    # ``<think>`` tags.  Handle both representations.
    raw_reasoning = msg.get("reasoning_content") if msg["role"] == "assistant" else None
    has_thinking = False
    if content is not None:
        if msg["role"] == "assistant":
            content = _normalize_thinking_tags(content)
            has_inline_thinking = bool(_THINK_RE.search(content))
            if has_inline_thinking and raw_reasoning and raw_reasoning.strip():
                # Conflict: both reasoning_content and inline <think> tags.
                # Keep inline tags (they are already in the content the
                # tokenizer will see) and drop reasoning_content to avoid
                # double thinking blocks in the rendered template.
                if not strip_thinking:
                    logger.warning(
                        "Message has both reasoning_content and inline "
                        "<think> tags.  Keeping inline tags, dropping "
                        "reasoning_content."
                    )
                raw_reasoning = None
            elif not has_inline_thinking and raw_reasoning and raw_reasoning.strip():
                # Convert reasoning_content → inline <think> in content.
                # This ensures a single representation that templates
                # render identically to the reasoning_content path, while
                # being more transparent and debuggable.
                if not strip_thinking:
                    content = (
                        f"<think>\n{raw_reasoning.strip(chr(10))}\n</think>"
                        f"\n\n{content.lstrip(chr(10))}"
                    )
                    has_inline_thinking = True
                raw_reasoning = None
            has_thinking = has_inline_thinking or bool(
                raw_reasoning and raw_reasoning.strip()
            )
            if strip_thinking:
                content = _extract_thinking(content)
        cleaned["content"] = content
    elif msg["role"] == "assistant" and msg.get("tool_calls"):
        # Assistant with tool_calls but content=None.
        if raw_reasoning and raw_reasoning.strip():
            has_thinking = True
            if not strip_thinking:
                # Convert reasoning_content → inline <think> in content.
                cleaned["content"] = (
                    f"<think>\n{raw_reasoning.strip(chr(10))}\n</think>"
                )
            else:
                cleaned["content"] = None
            raw_reasoning = None
        else:
            cleaned["content"] = None
    else:
        # Non-assistant messages without content: default to empty string.
        cleaned["content"] = ""

    # Preserve reasoning_content for target turns only when it was NOT
    # already inlined above (i.e. only when raw_reasoning is still set).
    if not strip_thinking and raw_reasoning is not None:
        cleaned["reasoning_content"] = raw_reasoning

    # For the target assistant turn without a thinking block, inject
    # inline ``<think>\n</think>`` so that the (patched) template detects
    # think intent via ``had_think_tags`` and renders
    # ``<think>\n\n</think>\n\n`` — identical token output to the old
    # ``reasoning_content='\n'`` approach.
    #
    # Requires ``_patch_chat_template_for_training`` to have been called
    # on the tokenizer, otherwise the stock Bailing template will extract
    # and discard the empty ``<think>`` block.
    if ensure_thinking and msg["role"] == "assistant" and not has_thinking:
        cur_content = cleaned.get("content")
        if cur_content is None or cur_content == "":
            cleaned["content"] = "<think>\n</think>"
        else:
            cleaned["content"] = f"<think>\n</think>\n\n{cur_content.lstrip(chr(10))}"

    # Copy tool_calls for assistant messages
    if msg["role"] == "assistant" and msg.get("tool_calls"):
        cleaned_tool_calls = []
        for tc in msg["tool_calls"]:
            cleaned_tc = {
                "type": tc.get("type", "function"),
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": json.dumps(tc["function"]["arguments"])
                    if isinstance(tc["function"]["arguments"], dict)
                    else tc["function"]["arguments"],
                },
            }
            if "id" in tc:
                cleaned_tc["id"] = tc["id"]
            cleaned_tool_calls.append(cleaned_tc)
        cleaned["tool_calls"] = cleaned_tool_calls

    # Copy tool_call_id for tool messages
    if msg["role"] == "tool" and msg.get("tool_call_id"):
        cleaned["tool_call_id"] = msg["tool_call_id"]

    return cleaned


# ============================================================
# 3. Filters — keep/discard predicates
# ============================================================


def _segment_has_error(messages, start, end):
    """Check if any tool message in ``messages[start:end]`` has ``is_error=True``."""
    for m in messages[start:end]:
        if m.get("role") == "tool" and m.get("is_error") is True:
            return True
    return False


def _is_empty_tool_call(msg):
    """True if assistant *msg* has no text content and no reasoning but has tool_calls."""
    content = msg.get("content") or ""
    if content.strip() or not msg.get("tool_calls"):
        return False
    # If reasoning_content exists, the model did think — not a silent invocation.
    reasoning = msg.get("reasoning_content")
    if reasoning and reasoning.strip():
        return False
    return True


def _is_bare_text_tool_call(msg):
    """True if assistant *msg* has text without ``<think>`` tags and has tool_calls."""
    content = msg.get("content") or ""
    if not content.strip() or not msg.get("tool_calls"):
        return False
    # If reasoning_content exists, thinking is in a separate field — not bare text.
    reasoning = msg.get("reasoning_content")
    if reasoning and reasoning.strip():
        return False
    normalized = _THINK_OPEN_RE.sub("<think>", content)
    normalized = _THINK_CLOSE_RE.sub("</think>", normalized)
    match = _THINK_RE.search(normalized)
    return not (match and match.group(1).strip())


def _msg_has_thinking(msg):
    """True if assistant *msg* has thinking content (inline or reasoning_content)."""
    if msg.get("role") != "assistant":
        return False
    content = msg.get("content") or ""
    normalized = _THINK_OPEN_RE.sub("<think>", content)
    normalized = _THINK_CLOSE_RE.sub("</think>", normalized)
    if _THINK_RE.search(normalized):
        return True
    rc = msg.get("reasoning_content") or ""
    return bool(rc.strip())


def _truncate_at_task_notification(messages):
    """Truncate messages when a ``<task-notification>`` follows a pure-text assistant.

    Claude Code emits ``<task-notification>`` as a user message when a
    background task (e.g. ``pip install``) completes.  If the model has
    already produced a text-only summary (no tool_calls), the notification
    and all subsequent messages are noise — the model just replies
    "nothing to do".  Truncating here removes that noise.

    Only triggers when the pattern is:
        assistant (text, no tool_calls) → user (<task-notification>)

    Returns:
        Truncated message list (or the original list if no truncation needed).
    """
    for i, m in enumerate(messages):
        if m.get("role") != "user":
            continue
        if "<task-notification>" not in (m.get("content") or ""):
            continue
        # Find preceding assistant
        prev_asst = None
        for j in range(i - 1, -1, -1):
            if messages[j].get("role") == "assistant":
                prev_asst = messages[j]
                break
        if prev_asst is None:
            continue
        content = prev_asst.get("content") or ""
        if content.strip() and not prev_asst.get("tool_calls"):
            # Truncate: keep everything up to (but not including) this user msg
            return messages[:i]
    return messages


# ============================================================
# 3b. Balancing — downsample non-thinking pairs
# ============================================================


def _classify_pair(pair):
    """Classify a pair by its target assistant turn's content type.

    Returns one of:
        ``"thinking"`` — target has actual ``<think>`` content or
            non-empty ``reasoning_content``.
        ``"no_thinking_tool_call"`` — target has no thinking but has
            ``tool_calls`` (the dominant category that causes distribution
            skew).
        ``"pure_text"`` — target has no thinking and no tool_calls
            (typically the final summary turn in a trajectory).
    """
    target = pair[-1]
    if target.get("role") != "assistant":
        return "pure_text"

    content = target.get("content") or ""
    rc = target.get("reasoning_content") or ""
    has_thinking = bool(_THINK_RE.search(content)) or bool(rc.strip())

    if has_thinking:
        return "thinking"
    if target.get("tool_calls"):
        return "no_thinking_tool_call"
    return "pure_text"


def _balance_thinking_pairs(pairs, max_no_thinking_ratio, seed=42, tools_list=None):
    """Downsample non-thinking **tool-call** pairs to control balance.

    Only ``no_thinking_tool_call`` pairs (no thinking but has tool_calls)
    are subject to downsampling.  ``thinking`` pairs and ``pure_text``
    pairs (the final summary turn, no thinking and no tool_calls) are
    always kept — the latter are critical for the model to learn when
    to stop calling tools and give a final answer.

    Args:
        pairs: List of progressive SFT pairs.
        max_no_thinking_ratio: Maximum ratio of non-thinking tool-call
            pairs to thinking pairs.  For example, ``1.0`` means at most
            1:1, ``2.0`` means at most 2 non-thinking per 1 thinking pair.
            ``None`` disables downsampling.
        seed: Random seed for reproducible downsampling.

    Returns:
        Balanced list of pairs (order preserved, randomly sampled for
        the downsampled category).
    """
    if max_no_thinking_ratio is None:
        return pairs, tools_list

    thinking = []
    no_think_tc = []
    pure_text = []
    for i, pair in enumerate(pairs):
        cat = _classify_pair(pair)
        if cat == "thinking":
            thinking.append(i)
        elif cat == "no_thinking_tool_call":
            no_think_tc.append(i)
        else:
            pure_text.append(i)

    n_think = len(thinking)
    n_no_think_tc = len(no_think_tc)
    n_pure_text = len(pure_text)

    if n_think == 0:
        logger.warning(
            "No thinking pairs found; skipping balance "
            "(all %d pairs have empty thinking).",
            n_no_think_tc + n_pure_text,
        )
        return pairs, tools_list

    max_no_think_tc = int(n_think * max_no_thinking_ratio)
    if n_no_think_tc <= max_no_think_tc:
        logger.info(
            "Thinking balance OK: %d thinking + %d no-think-tc + %d pure-text "
            "(ratio %.1f <= %.1f), no downsampling needed.",
            n_think,
            n_no_think_tc,
            n_pure_text,
            n_no_think_tc / n_think,
            max_no_thinking_ratio,
        )
        return pairs, tools_list

    rng = random.Random(seed)
    sampled_tc = set(rng.sample(no_think_tc, max_no_think_tc))
    keep_indices = sorted(set(thinking) | sampled_tc | set(pure_text))
    balanced = [pairs[i] for i in keep_indices]
    balanced_tools = (
        [tools_list[i] for i in keep_indices] if tools_list is not None else None
    )

    logger.info(
        "Balanced thinking pairs: %d thinking + %d no-think-tc "
        "(downsampled from %d, ratio %.1f → %.1f) + %d pure-text (kept all).",
        n_think,
        max_no_think_tc,
        n_no_think_tc,
        n_no_think_tc / n_think,
        max_no_thinking_ratio,
        n_pure_text,
    )
    return balanced, balanced_tools


# ============================================================
# 3c. Thinking augmentation stats
# ============================================================


def _log_thinking_augmentation_stats(
    n_variants,
    prob,
    n_total_trajs,
    thinking_turns_per_traj,
    total_asst_turns_per_traj,
    patterns_per_traj,
):
    """Log adaptive-thinking augmentation quality metrics.

    Called after the augmentation loop in loaders to report how well
    the ``n_thinking_variants`` / ``random_strip_thinking_prob`` settings
    produce diverse thinking-pattern variants.

    Args:
        n_variants: ``n_thinking_variants`` setting (K).
        prob: ``random_strip_thinking_prob`` setting.
        n_total_trajs: Total number of source trajectories processed.
        thinking_turns_per_traj: List of N_thinking per source trajectory.
        total_asst_turns_per_traj: List of N_total_asst per source trajectory.
        patterns_per_traj: List of ``set[frozenset]`` — the unique strip
            patterns generated for each source trajectory (including the
            empty frozenset for the original unstripped variant).
    """
    n_eligible = sum(1 for n in thinking_turns_per_traj if n > 0)
    total_thinking = sum(thinking_turns_per_traj)
    total_asst = sum(total_asst_turns_per_traj)

    # 1. Thinking Turn Coverage
    avg_thinking = total_thinking / max(n_total_trajs, 1)
    thinking_ratio = total_thinking / max(total_asst, 1)

    # 2. Pattern Diversity
    diversity_ratios = []
    for n_think, patterns in zip(thinking_turns_per_traj, patterns_per_traj):
        if n_think == 0:
            continue
        theoretical_max = min(n_variants, 2**n_think)
        actual_unique = len(patterns)
        diversity_ratios.append(actual_unique / theoretical_max)
    avg_diversity = sum(diversity_ratios) / max(len(diversity_ratios), 1)

    # 3. Augmentation Efficiency
    n_non_trivial = 0
    for patterns in patterns_per_traj:
        # Count variants that differ from the original (non-empty strip set)
        n_non_trivial += sum(1 for p in patterns if p)
    expected_aug = (n_variants - 1) * max(n_eligible, 1)
    efficiency = n_non_trivial / max(expected_aug, 1)

    # 4. Total sample count
    n_total_samples = sum(len(p) for p in patterns_per_traj)

    logger.info(
        f"Thinking augmentation stats (K={n_variants}, p={prob:.2f}):\n"
        f"  Source trajectories: {n_total_trajs} "
        f"({n_eligible} with thinking turns)\n"
        f"  Thinking coverage: {avg_thinking:.1f} thinking turns/traj, "
        f"{thinking_ratio:.1%} of all assistant turns\n"
        f"  Pattern diversity: {avg_diversity:.2f} "
        f"(1.0 = all variants unique)\n"
        f"  Augmentation efficiency: {efficiency:.2f} "
        f"({n_non_trivial}/{expected_aug} non-trivial variants)\n"
        f"  Total samples after augmentation: {n_total_samples}"
    )


# ============================================================
# 4. Splitting — trajectory → progressive pairs
# ============================================================


def _find_segments(messages):
    """Find assistant+tools segment boundaries.

    Returns:
        List of ``(assistant_start_idx, segment_end_idx)`` tuples.
    """
    segments = []
    i = 0
    while i < len(messages):
        if messages[i].get("role") == "assistant":
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
            segments.append((i, j))
            i = j
        else:
            i += 1
    return segments


def _split_and_filter(
    messages,
    filter_errors=True,
    strip_all_thinking=False,
    filter_empty_tool_calls=False,
    filter_bare_text_tool_calls=False,
    random_strip_thinking_prob=0.0,
    rng=None,
):
    """Split trajectory into progressive pairs and optionally filter.

    By default, thinking (``<think>...</think>``) is stripped from context
    assistant turns only; the last assistant turn (training target) keeps
    its content unchanged.  Set *strip_all_thinking* to strip from every
    assistant turn including the target.

    When *random_strip_thinking_prob* > 0, each target assistant turn that
    has thinking content is independently stripped with that probability.
    Stripped turns use the context-cleaned version (thinking fully removed,
    no empty ``<think></think>`` injected).

    Args:
        messages: Raw trajectory messages.
        filter_errors: If True (default), discard pairs whose current segment
            contains a tool result with ``is_error=True``.  Set to False to
            keep all pairs regardless of tool errors.
        strip_all_thinking: If True, strip ``<think>`` blocks from every
            assistant turn including the training target.
        filter_empty_tool_calls: If True, discard pairs whose training-target
            assistant turn has no text content but has tool_calls.
        filter_bare_text_tool_calls: If True, discard pairs whose
            training-target assistant turn has text content without
            ``<think>`` tags and has tool_calls.
        random_strip_thinking_prob: Probability of stripping thinking
            from each target assistant turn.  0.0 = no stripping.
        rng: ``random.Random`` instance for reproducible sampling.

    Returns:
        Tuple of ``(pairs, n_filtered_errors, n_filtered_empty_tc,
        n_filtered_bare_tc, n_stripped)``.
    """
    segments = _find_segments(messages)
    if not segments:
        return [], 0, 0, 0, 0

    pairs = []
    n_filtered_errors = 0
    n_filtered_empty_tc = 0
    n_filtered_bare_tc = 0
    n_stripped = 0

    # Pre-clean all messages in context mode (thinking stripped).
    # This avoids re-cleaning the same message for every progressive pair
    # (O(N+K) instead of O(N*K) where K = number of segments).
    context_cleaned = [_clean_message(m, strip_thinking=True) for m in messages]

    # For target assistant turns, clean with thinking preserved (unless
    # strip_all_thinking is set, in which case context_cleaned is reusable).
    # When stripping is active (augmented variant), use ensure_thinking=False
    # so empty-thinking turns don't get <think>\n</think> injected.
    stripping_active = random_strip_thinking_prob > 0.0 and rng is not None
    target_ensure = not stripping_active
    target_cleaned = {}
    if not strip_all_thinking:
        for asst_start, _ in segments:
            target_cleaned[asst_start] = _clean_message(
                messages[asst_start],
                strip_thinking=False,
                ensure_thinking=target_ensure,
            )

    for asst_start, seg_end in segments:
        # Check if current segment has any tool errors
        if filter_errors and _segment_has_error(messages, asst_start, seg_end):
            n_filtered_errors += 1
            continue

        # Content-type filters operate on the raw assistant message.
        asst_msg = messages[asst_start]
        if filter_empty_tool_calls and _is_empty_tool_call(asst_msg):
            n_filtered_empty_tc += 1
            continue
        if filter_bare_text_tool_calls and _is_bare_text_tool_call(asst_msg):
            n_filtered_bare_tc += 1
            continue

        # Build pair: include context up to the target assistant turn,
        # truncating tool responses that follow it.  This ensures the
        # target assistant is always the *last* message so that chat
        # templates with ``loop.last``-dependent rendering (e.g. Qwen3
        # ``<think>`` injection) behave consistently.  The tool responses
        # would have loss_mask=0 anyway and only add noise.
        pair = list(context_cleaned[: asst_start + 1])
        if not strip_all_thinking:
            # Randomly strip: leave context_cleaned version (thinking
            # already removed) instead of replacing with target_cleaned.
            should_strip = (
                rng is not None
                and _msg_has_thinking(messages[asst_start])
                and rng.random() < random_strip_thinking_prob
            )
            if not should_strip:
                pair[asst_start] = target_cleaned[asst_start]
            else:
                n_stripped += 1
        pairs.append(pair)

    return pairs, n_filtered_errors, n_filtered_empty_tc, n_filtered_bare_tc, n_stripped


def _prepare_trajectory(
    messages,
    filter_errors=True,
    filter_empty_tool_calls=False,
    filter_bare_text_tool_calls=False,
    random_strip_thinking_prob=0.0,
    rng=None,
):
    """Prepare a full trajectory for trajectory-level training.

    Cleans all messages preserving thinking for every assistant turn
    (``strip_thinking=False``, ``ensure_thinking=True``).  Identifies
    which assistant segments should be masked (``loss_mask=0``) based
    on error tool responses, empty tool calls, or bare-text tool calls.

    When *random_strip_thinking_prob* > 0, each assistant turn that has
    thinking content is independently stripped with that probability.
    Stripped turns have their ``<think>`` blocks and ``reasoning_content``
    completely removed (no empty ``<think></think>`` injected).

    Args:
        messages: Raw trajectory messages.
        filter_errors: If True (default), mask segments with error tool
            responses.
        filter_empty_tool_calls: If True, mask segments whose assistant
            turn has no text content but has tool_calls.
        filter_bare_text_tool_calls: If True, mask segments whose
            assistant turn has text without ``<think>`` tags and has
            tool_calls.
        random_strip_thinking_prob: Probability of stripping thinking
            from each assistant turn that has thinking content.
            0.0 (default) = no stripping, 1.0 = strip all.
        rng: ``random.Random`` instance for reproducible sampling.

    Returns:
        Tuple of ``(cleaned_messages, masked_segment_indices,
        n_error, n_empty_tc, n_bare_tc, stripped_pattern)`` or ``None``
        if the trajectory has no assistant turns.  *stripped_pattern* is
        a ``frozenset`` of message indices whose thinking was stripped
        (empty if no stripping occurred).
    """
    segments = _find_segments(messages)
    if not segments:
        return None

    masked_indices = set()
    n_error = 0
    n_empty_tc = 0
    n_bare_tc = 0
    for idx, (asst_start, seg_end) in enumerate(segments):
        if filter_errors and _segment_has_error(messages, asst_start, seg_end):
            masked_indices.add(idx)
            n_error += 1
            continue
        asst_msg = messages[asst_start]
        if filter_empty_tool_calls and _is_empty_tool_call(asst_msg):
            masked_indices.add(idx)
            n_empty_tc += 1
            continue
        if filter_bare_text_tool_calls and _is_bare_text_tool_call(asst_msg):
            masked_indices.add(idx)
            n_bare_tc += 1

    # Determine which assistant turns to randomly strip thinking from.
    strip_thinking_indices = set()
    stripping_active = random_strip_thinking_prob > 0.0 and rng is not None
    if stripping_active:
        for asst_start, _seg_end in segments:
            if _msg_has_thinking(messages[asst_start]):
                if rng.random() < random_strip_thinking_prob:
                    strip_thinking_indices.add(asst_start)

    # When stripping is active (augmented variant), use ensure_thinking=False
    # for ALL turns so that empty-thinking turns don't get <think>\n</think>
    # injected.  Only real thinking content is preserved.
    # When stripping is inactive (variant 0 or no augmentation), keep
    # ensure_thinking=True to match the standard training format.
    default_ensure = not stripping_active

    cleaned = []
    for i, m in enumerate(messages):
        if i in strip_thinking_indices:
            cleaned.append(
                _clean_message(m, strip_thinking=True, ensure_thinking=False)
            )
        else:
            cleaned.append(
                _clean_message(m, strip_thinking=False, ensure_thinking=default_ensure)
            )

    return (
        cleaned,
        sorted(masked_indices),
        n_error,
        n_empty_tc,
        n_bare_tc,
        frozenset(strip_thinking_indices),
    )


# ============================================================
# 5. Tokenization — template detection, render, loss mask, dump
# ============================================================


# -- Chat template patch (runtime, no file modification) --------

# Both Bailing and Qwen3 templates have ``ns.last_query_index`` logic
# that prevents ``<think>`` rendering for assistant turns BEFORE the
# last user message, AND discards inline empty ``<think>\n</think>``
# extracted from content.
#
# This breaks trajectory-mode training:
# - Multi-user trajectories: turns before the last user msg lack <think>
# - Empty ensure_thinking via inline <think> gets stripped
#
# The patch below handles both Bailing (`<role>ASSISTANT</role>` style)
# and Qwen3 (`<|im_start|>assistant` style) templates:
# 1. Adds ``had_think_tags`` detection so empty ``<think>`` survives.
# 2. Removes the ``ns.last_query_index`` gate so all assistant turns
#    render ``<think>`` uniformly when think intent is detected.
#
# Applied at runtime via ``tokenizer.chat_template = patched`` — the
# original template file on disk is never modified.

_BAILING_OLD_BLOCK = (
    "{%- if loop.index0 > ns.last_query_index %}\n"
    "            {%- if reasoning_content != '' %}\n"
    "                {{- '<role>ASSISTANT</role>\\n' + '<think>\\n'"
    " + reasoning_content.strip('\\n') + '\\n</think>\\n\\n'"
    " + content.lstrip('\\n') }}\n"
    "            {%- else %}\n"
    "                {{- '<role>ASSISTANT</role>\\n' + content }}\n"
    "            {%- endif %}\n"
    "        {%- else %}\n"
    "            {{- '<role>ASSISTANT</role>\\n' + content }}\n"
    "        {%- endif %}"
)
_BAILING_NEW_BLOCK = (
    "{%- if reasoning_content != '' or had_think_tags %}\n"
    "                {{- '<role>ASSISTANT</role>\\n' + '<think>\\n'"
    " + reasoning_content.strip('\\n') + '\\n</think>\\n\\n'"
    " + content.lstrip('\\n') }}\n"
    "            {%- else %}\n"
    "                {{- '<role>ASSISTANT</role>\\n' + content }}\n"
    "            {%- endif %}"
)

# Qwen3 uses `loop.last or (not loop.last and reasoning_content)` so the
# last turn always renders <think> even with empty reasoning.  We
# preserve `loop.last` and add `had_think_tags` for inline-empty support.
_QWEN3_OLD_BLOCK = (
    "{%- if loop.index0 > ns.last_query_index %}\n"
    "            {%- if loop.last or (not loop.last and reasoning_content) %}\n"
    "                {{- '<|im_start|>' + message.role + '\\n<think>\\n'"
    " + reasoning_content.strip('\\n') + '\\n</think>\\n\\n'"
    " + content.lstrip('\\n') }}\n"
    "            {%- else %}\n"
    "                {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
    "            {%- endif %}\n"
    "        {%- else %}\n"
    "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
    "        {%- endif %}"
)
_QWEN3_NEW_BLOCK = (
    "{%- if loop.last or reasoning_content != '' or had_think_tags %}\n"
    "                {{- '<|im_start|>' + message.role + '\\n<think>\\n'"
    " + reasoning_content.strip('\\n') + '\\n</think>\\n\\n'"
    " + content.lstrip('\\n') }}\n"
    "            {%- else %}\n"
    "                {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
    "            {%- endif %}"
)

_OLD_DETECT = "{%- set reasoning_content = '' %}"
_NEW_DETECT = (
    "{%- set reasoning_content = '' %}\n"
    "        {%- set had_think_tags = ('</think>' in content) %}"
)


def _patch_chat_template_for_training(tokenizer):
    """Patch Bailing/Qwen3 chat templates to render ``<think>`` uniformly.

    Detects template family by matching known render blocks:
    - Bailing: ``<role>ASSISTANT</role>`` markers
    - Qwen3: ``<|im_start|>assistant`` markers

    Other templates (e.g. plain ChatML without ``last_query_index``)
    are left unchanged.  If the template has ``last_query_index`` but
    neither known block matches, logs a warning.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template or "last_query_index" not in template:
        return

    if _BAILING_OLD_BLOCK in template:
        family = "Bailing"
        patched = template.replace(_BAILING_OLD_BLOCK, _BAILING_NEW_BLOCK)
    elif _QWEN3_OLD_BLOCK in template:
        family = "Qwen3"
        patched = template.replace(_QWEN3_OLD_BLOCK, _QWEN3_NEW_BLOCK)
    else:
        logger.warning(
            "Chat template has last_query_index but matches neither "
            "Bailing nor Qwen3 render block. Patch skipped — empty "
            "<think> may be discarded, multi-user turns may lack <think>."
        )
        return

    patched = patched.replace(_OLD_DETECT, _NEW_DETECT)

    tokenizer.chat_template = patched
    logger.info(
        f"Patched {family} chat template for training: removed "
        "last_query_index gate, added had_think_tags detection."
    )


_TEMPLATE_PATTERNS = [
    # ChatML (Qwen, etc.):  <|im_start|>assistant\n ... <|im_end|>
    (r"<\|im_start\|>assistant\n", r"<\|im_end\|>"),
    # Llama 3:  <|start_header_id|>assistant<|end_header_id|>\n\n ... <|eot_id|>
    (r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n", r"<\|eot_id\|>"),
    # GLM:  <|assistant|> ... (ends at next <|user|>, <|observation|>, or end of string)
    (r"<\|assistant\|>", r"(?=<\|user\|>|<\|observation\|>|\Z)"),
]


def _parse_tool_call_arguments(messages):
    """Parse JSON-string arguments in tool_calls to dicts.

    OpenAI returns tool_call arguments as JSON strings, but some chat
    templates (e.g. GLM-4.x / GLM-5.x) expect parsed dicts. Most other
    templates (Qwen / ChatML, Llama 3, Bailing, ...) accept the standard
    OpenAI string form, so this conversion must be opt-in.
    """
    patched = []
    for m in messages:
        tool_calls = m.get("tool_calls")
        if not tool_calls:
            patched.append(m)
            continue
        new_tcs = []
        for tc in tool_calls:
            fn = tc.get("function", tc)
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    parsed = args
                fn = {**fn, "arguments": parsed}
                tc = {**tc, "function": fn} if "function" in tc else fn
            new_tcs.append(tc)
        patched.append({**m, "tool_calls": new_tcs})
    return patched


def _render_tokenize_mask(
    messages,
    tokenizer,
    assistant_pattern,
    tools=None,
    *,
    split_mode="pair",
    error_indices=None,
    parse_tool_call_args=False,
):
    """Render, tokenize, and build loss_mask for a message list.

    In **pair mode** (default), only the **last** assistant turn gets
    ``loss_mask=1``.  In **trajectory mode**, **all** assistant turns
    get ``loss_mask=1`` except those at indices in *error_indices*.

    When *parse_tool_call_args* is True, JSON-string ``tool_calls`` arguments
    are converted to dicts before rendering (required by GLM chat templates;
    other templates such as Qwen / Llama / Bailing must keep the OpenAI
    string form).

    Returns:
        Tuple of ``(full_text, input_ids, loss_mask, offset_mapping)``, or
        ``None`` if ``apply_chat_template`` fails.
    """
    # 1) Render the full template text.
    try:
        kwargs = {"tokenize": False}
        if tools is not None:
            kwargs["tools"] = tools
        if parse_tool_call_args:
            messages = _parse_tool_call_arguments(messages)
        full_text = tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as e:
        logger.warning(
            "apply_chat_template failed: %s. Skipping sample.",
            e,
        )
        return None

    # 2) Tokenize with offset mapping so we can map char→token.
    encoding = tokenizer(
        full_text, add_special_tokens=False, return_offsets_mapping=True
    )
    input_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]

    # 3) Build loss_mask.
    loss_mask = [0] * len(input_ids)

    if split_mode == "trajectory":
        # Trajectory mode: mask ALL assistant segments, skip error_indices.
        skip = set(error_indices) if error_indices else set()
        matches = list(assistant_pattern.finditer(full_text))

        # Verify regex matches correspond 1:1 to assistant messages.
        n_asst = sum(1 for m in messages if m.get("role") == "assistant")
        if len(matches) != n_asst:
            logger.warning(
                "Segment count mismatch: %d assistant messages but %d regex "
                "matches in rendered text. Training all matched segments "
                "(error masking disabled for this sample).",
                n_asst,
                len(matches),
            )
            skip = set()

        for seg_idx, m in enumerate(matches):
            if seg_idx in skip:
                continue
            rs, re_ = m.start(1), m.end(0)
            for tok_idx, (cs, ce) in enumerate(offset_mapping):
                if ce > rs and cs < re_:
                    loss_mask[tok_idx] = 1
    else:
        # Pair mode: mask only the LAST assistant segment.
        last_match = None
        for m in assistant_pattern.finditer(full_text):
            last_match = m
        if last_match is not None:
            rs, re_ = last_match.start(1), last_match.end(0)
            for tok_idx, (cs, ce) in enumerate(offset_mapping):
                if ce > rs and cs < re_:
                    loss_mask[tok_idx] = 1

    return full_text, input_ids, loss_mask, offset_mapping


class _TokenizeAndMask:
    """Picklable callable for ``Dataset.map(num_proc=N)``."""

    def __init__(
        self,
        tokenizer,
        assistant_pattern,
        max_length=None,
        *,
        split_mode="pair",
        parse_tool_call_args=False,
    ):
        self.tokenizer = tokenizer
        self.assistant_pattern = assistant_pattern
        self.max_length = max_length
        self.split_mode = split_mode
        self.parse_tool_call_args = parse_tool_call_args

    def __call__(self, sample):
        error_indices = (
            sample.get("error_indices", []) if self.split_mode == "trajectory" else None
        )
        tools_json = sample.get("tools_json")
        tools = json.loads(tools_json) if tools_json else None
        result = _render_tokenize_mask(
            sample["messages"],
            self.tokenizer,
            self.assistant_pattern,
            tools,
            split_mode=self.split_mode,
            error_indices=error_indices,
            parse_tool_call_args=self.parse_tool_call_args,
        )
        if result is None:
            return {"input_ids": [], "loss_mask": []}

        _full_text, input_ids, loss_mask, _offset_mapping = result

        # Early exit: overlength or empty → return empty so a single
        # filter pass removes it together with template-failure empties.
        if self.max_length is not None and len(input_ids) > self.max_length:
            return {"input_ids": [], "loss_mask": []}

        return {"input_ids": input_ids, "loss_mask": loss_mask}


def _detect_template_pattern(tokenizer, tools=None):
    """Detect the assistant role delimiter used by this tokenizer's template.

    When *tools* is provided the probe is rendered with ``tools=`` so that
    the detected delimiters match the actual training text (some templates
    alter the system block when tools are present).

    Strategy:
        1. Try known ``_TEMPLATE_PATTERNS`` (fast, battle-tested).
        2. Fall back to double-probe diff: render the template with a known
           marker and with empty content, then diff the two strings to extract
           the exact header and end-of-turn delimiters.

    Raises:
        ValueError: If both strategies fail to detect a usable pattern.
    """
    _PROBE_CONTENT = "PROBE_MARKER"

    extra_kwargs = {}
    if tools is not None:
        extra_kwargs["tools"] = tools

    probe_msgs = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": _PROBE_CONTENT},
    ]
    probe_text = tokenizer.apply_chat_template(
        probe_msgs, tokenize=False, **extra_kwargs
    )

    # --- Strategy 1: known patterns ---
    for hdr_re, eot_re in _TEMPLATE_PATTERNS:
        if re.search(hdr_re, probe_text):
            pattern = re.compile(hdr_re + r"(.*?)" + eot_re, re.DOTALL)
            logger.info(
                f"Detected template style (known pattern): "
                f"header_re={hdr_re!r}, eot_re={eot_re!r}"
            )
            return pattern

    # --- Strategy 2: double-probe diff ---
    try:
        probe_empty = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": ""},
        ]
        text_empty = tokenizer.apply_chat_template(
            probe_empty, tokenize=False, **extra_kwargs
        )

        marker_idx = probe_text.index(_PROBE_CONTENT)
        header = probe_text[:marker_idx]
        tail = probe_text[marker_idx + len(_PROBE_CONTENT) :]

        if text_empty == header + tail:
            # Extract the assistant-specific header by removing the shared
            # user-only prefix.
            user_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": "x"}],
                tokenize=False,
                **extra_kwargs,
            )
            asst_header = header[len(user_only) :]
            # end-of-turn delimiter: strip leading newlines, then take
            # up to the first newline (or the full string if none).
            eot_stripped = tail.lstrip("\n")
            eot = eot_stripped.split("\n")[0] if "\n" in eot_stripped else eot_stripped

            if asst_header and eot:
                hdr_re = re.escape(asst_header)
                eot_re = re.escape(eot)
                pattern = re.compile(hdr_re + r"(.*?)" + eot_re, re.DOTALL)
                logger.info(
                    f"Detected template style (probe diff): "
                    f"header={asst_header!r}, eot={eot!r}"
                )
                return pattern
    except (ValueError, IndexError):
        pass  # PROBE_CONTENT not found in rendered text, skip

    raise ValueError(
        "Could not detect chat template assistant delimiters. "
        "Unable to build a reliable loss mask. "
        f"Probe text: {probe_text[:200]!r}"
    )


def _dump_samples(
    samples,
    tokenizer,
    assistant_pattern,
    tools_list,
    dump_dir,
    n_samples,
    *,
    split_mode="pair",
    error_indices_list=None,
    parse_tool_call_args=False,
):
    """Dump sampled message lists as ``.txt`` + ``.json`` for inspection.

    Args:
        samples: List of message-list samples (pairs or full trajectories).
        tokenizer: Tokenizer with ``apply_chat_template`` support.
        assistant_pattern: Compiled regex from ``_detect_template_pattern``.
        tools_list: Per-sample tool definitions (parallel to *samples*),
            or ``None`` when no tools are available.
        dump_dir: Directory to write files into (created if needed).
        n_samples: Number of random samples to dump.  ``-1`` dumps all.
        split_mode: ``"trajectory"`` for trajectory-mode loss masking.
        error_indices_list: Per-sample error segment indices (trajectory mode).
    """
    import random as _random

    os.makedirs(dump_dir, exist_ok=True)

    if n_samples == -1 or n_samples >= len(samples):
        indices = list(range(len(samples)))
    else:
        indices = sorted(_random.sample(range(len(samples)), n_samples))

    n_written = 0
    for i in indices:
        sample = samples[i]
        sample_tools = tools_list[i] if tools_list else None
        err_idxs = (
            error_indices_list[i]
            if split_mode == "trajectory" and error_indices_list
            else None
        )

        result = _render_tokenize_mask(
            sample,
            tokenizer,
            assistant_pattern,
            sample_tools,
            split_mode=split_mode,
            error_indices=err_idxs,
            parse_tool_call_args=parse_tool_call_args,
        )
        if result is None:
            continue

        full_text, input_ids, loss_mask, offset_mapping = result
        n_loss = sum(loss_mask)
        base = os.path.join(dump_dir, f"sample_{i}")

        # --- .txt ---
        with open(base + ".txt", "w", encoding="utf-8") as fout:
            fout.write(
                f"Sample {i}: {len(sample)} messages, "
                f"{len(input_ids)} tokens, loss=1: {n_loss}\n"
            )
            fout.write(f"Last msg role: {sample[-1]['role']}\n")
            fout.write(f"{'=' * 72}\n\n")

            fout.write("--- Rendered Text ---\n")
            fout.write(full_text)
            fout.write("\n\n")

            fout.write("--- Token / Loss Mask ---\n")
            fout.write(f"{'Idx':>6} | {'TokenID':>8} | Loss | Token Text\n")
            fout.write(f"{'-' * 6}-+-{'-' * 8}-+------+{'-' * 40}\n")
            for t in range(len(input_ids)):
                cs, ce = offset_mapping[t]
                tok_text = repr(full_text[cs:ce])
                fout.write(
                    f"{t:>6} | {input_ids[t]:>8} | {loss_mask[t]:>4} | {tok_text}\n"
                )

        # --- .json ---
        tokens_list = []
        for t in range(len(input_ids)):
            cs, ce = offset_mapping[t]
            tokens_list.append(
                {
                    "idx": t,
                    "token_id": input_ids[t],
                    "text": full_text[cs:ce],
                    "loss": loss_mask[t],
                }
            )
        record = {
            "sample_index": i,
            "n_messages": len(sample),
            "n_tokens": len(input_ids),
            "n_loss_tokens": n_loss,
            "rendered_text": full_text,
            "tokens": tokens_list,
        }
        with open(base + ".json", "w", encoding="utf-8") as fout:
            json.dump(record, fout, ensure_ascii=False)

        n_written += 1

    logger.info(f"Dumped {n_written} samples to {dump_dir}/")


# ============================================================
# 6. Pipeline — loading, processing, distributed cache, public API
# ============================================================


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

        def _filter_by_max_length(ds):
            if max_length is None:
                return ds
            before = len(ds)
            keep = [i for i in range(len(ds)) if len(ds[i]["input_ids"]) <= max_length]
            ds = ds.select(keep)
            if len(ds) < before:
                logger.info(
                    f"Rank {rank}: filtered {before - len(ds)} samples "
                    f"exceeding max_length={max_length}"
                )
            return ds

        # Fast path: cache from a previous run (or rank 0 already finished).
        if os.path.exists(done_marker):
            logger.info(
                f"Rank {rank}: loading cached processed dataset from {cache_dir}"
            )
            dataset = load_from_disk(cache_dir)
            dataset = _filter_by_max_length(dataset)
            logger.info(f"Final dataset: {len(dataset)} samples")
            return dataset

        if rank == 0:
            # Rank 0: do the heavy processing and save for other ranks.
            dataset = _process_swe_sft(path, tokenizer, **process_kwargs)
            os.makedirs(cache_dir, exist_ok=True)
            dataset.save_to_disk(cache_dir)
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
            # Other ranks: wait for rank 0, then load.
            logger.info(f"Rank {rank}: waiting for rank 0 to process dataset...")
            _wait_for_marker(done_marker)
            dataset = load_from_disk(cache_dir)
            dataset = _filter_by_max_length(dataset)
            logger.info(f"Rank {rank}: loaded cached dataset ({len(dataset)} samples)")
            return dataset

    # --- Non-distributed or no cache_dir: process in current process ---
    return _process_swe_sft(path, tokenizer, **process_kwargs)


# ============================================================
# 7. CLI — ``python -m areal.dataset.swe_sft``
# ============================================================

if __name__ == "__main__":
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
        "before apply_chat_template. Required by GLM-4.x / GLM-5.x "
        "templates; leave off for Qwen / Llama / Bailing (which expect "
        "the standard string form).",
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
        "--truncate-task-notifications",
        action="store_true",
        help="Truncate trajectories at the first <task-notification> that "
        "follows a pure-text assistant turn. Removes noise from background "
        "task completions (e.g. pip install finishing after the model's summary).",
    )
    parser.add_argument(
        "--max-no-thinking-ratio",
        type=float,
        default=None,
        help="Maximum ratio of non-thinking pairs to thinking pairs. "
        "E.g. 1.0 = 1:1 balance, 2.0 = at most 2x non-thinking per "
        "thinking pair. Non-thinking pairs are randomly downsampled. "
        "Default: no balancing.",
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
        "--random-strip-thinking-prob",
        type=float,
        default=0.0,
        help="Probability of stripping thinking from each target assistant "
        "turn. 0.0 = no stripping (default), 1.0 = strip all. "
        "Works in both pair mode and trajectory mode.",
    )
    parser.add_argument(
        "--random-strip-thinking-seed",
        type=int,
        default=42,
        help="Random seed for reproducible thinking stripping decisions (default: 42).",
    )
    parser.add_argument(
        "--n-thinking-variants",
        type=int,
        default=1,
        help="Number of thinking-pattern variants per trajectory. "
        "1 = no augmentation (default). K > 1 = augment each trajectory "
        "into K variants: the first preserves all thinking, the rest "
        "randomly strip with --random-strip-thinking-prob.",
    )
    parser.add_argument(
        "--save-trajectories",
        default=None,
        metavar="FILE",
        help="Save preprocessed trajectories to FILE (JSONL, original format) "
        "after applying trajectory-level operations (e.g. "
        "--truncate-task-notifications) but before pair splitting. "
        "Each line preserves the original record structure with the "
        "messages field updated.",
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
    truncate_task_notifications = args.truncate_task_notifications
    max_no_thinking_ratio = args.max_no_thinking_ratio

    # --- Fast path: save preprocessed trajectories ---
    if args.save_trajectories:
        records_in = 0
        records_out = 0
        n_truncated = 0
        with (
            open(args.path, encoding="utf-8") as fin,
            open(args.save_trajectories, "w", encoding="utf-8") as fout,
        ):
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records_in += 1

                messages, _ = _extract_messages(record, records_in)

                if truncate_task_notifications and messages:
                    truncated = _truncate_at_task_notification(messages)
                    if len(truncated) < len(messages):
                        n_truncated += 1
                        _set_messages(record, truncated)

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_out += 1

        parts = []
        if n_truncated:
            parts.append(f"{n_truncated} truncated at task-notification")
        op_msg = ", ".join(parts) if parts else "no changes"
        print(
            f"Saved {records_out}/{records_in} trajectories "
            f"to {args.save_trajectories} ({op_msg})"
        )
        sys.exit(0)

    # --- Load ---
    split_mode = args.split_mode
    error_indices_list = None

    if split_mode == "trajectory":
        samples, error_indices_list, tools_list = _load_full_trajectories(
            args.path,
            filter_errors=filter_errors,
            filter_empty_tool_calls=filter_empty_tool_calls,
            filter_bare_text_tool_calls=filter_bare_text_tool_calls,
            truncate_task_notifications=truncate_task_notifications,
            random_strip_thinking_prob=args.random_strip_thinking_prob,
            random_strip_thinking_seed=args.random_strip_thinking_seed,
            n_thinking_variants=args.n_thinking_variants,
        )
        label = "trajectories"
    elif args.pre_split:
        samples, tools_list = _load_presplit_pairs(
            args.path,
            strip_all_thinking=strip_all_thinking,
            random_strip_thinking_prob=args.random_strip_thinking_prob,
            random_strip_thinking_seed=args.random_strip_thinking_seed,
            n_thinking_variants=args.n_thinking_variants,
        )
        label = "pairs"
    else:
        samples, tools_list = _load_trajectory_pairs(
            args.path,
            filter_errors=filter_errors,
            strip_all_thinking=strip_all_thinking,
            filter_empty_tool_calls=filter_empty_tool_calls,
            filter_bare_text_tool_calls=filter_bare_text_tool_calls,
            truncate_task_notifications=truncate_task_notifications,
            max_no_thinking_ratio=max_no_thinking_ratio,
            random_strip_thinking_prob=args.random_strip_thinking_prob,
            random_strip_thinking_seed=args.random_strip_thinking_seed,
            n_thinking_variants=args.n_thinking_variants,
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
        first_tools = None
        if dump_tools:
            first_tools = next((t for t in dump_tools if t is not None), None)
        assistant_pattern = _detect_template_pattern(tok, tools=first_tools)
        _dump_samples(
            samples,
            tok,
            assistant_pattern,
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
        print(f"Saved tokenized dataset ({len(ds)} samples) to {args.save_tokenized}")

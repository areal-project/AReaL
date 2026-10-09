# SPDX-License-Identifier: Apache-2.0

"""Message normalization, filtering, and trajectory splitting for SWE SFT."""

import json
import re

from areal.utils import logging

logger = logging.getLogger("SWESFTDataset")


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


def _iter_jsonl_records(path):
    """Iterate trajectory JSONL records.

    Yields ``(record_idx, messages, record_tools)`` tuples.  Handles
    nested (``conversations`` wrapper) vs flat format auto-detection
    via ``_extract_messages``.  Records with empty messages are skipped.

    Reports multi-user trajectories for visibility.  These are now
    **handled** in this pipeline: ``_patch_chat_template_for_training``
    removes the ``ns.last_query_index`` gate at tokenization time, so
    ``<think>`` renders for every assistant turn regardless of how many
    user messages precede it.  The count below only matters if the data
    is tokenized through a pipeline that does NOT apply that patch.
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
                    logger.info(
                        "Record %d has %d user messages. Handled by "
                        "_patch_chat_template_for_training (last_query_index "
                        "gate removed) so all assistant turns render <think>; "
                        "only an issue if tokenized via an unpatched template.",
                        record_idx,
                        n_user,
                    )
            yield record_idx, messages, record_tools
    if n_multi_user > 0:
        logger.info(
            "Total %d/%d records have multiple user messages "
            "(handled by the chat-template patch at tokenization time).",
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
            patched template (Bailing / Qwen3 / other ``last_query_index``
            families, via ``_patch_chat_template_for_training``) which
            detects ``had_think_tags`` and preserves empty think blocks.
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
    if raw_reasoning is not None and not raw_reasoning.strip():
        raw_reasoning = None
    has_thinking = False
    if msg["role"] == "assistant":
        if content is not None:
            content = _normalize_thinking_tags(content)
        has_inline_thinking = bool(content and _THINK_RE.search(content))
        has_separate_thinking = raw_reasoning is not None

        if has_inline_thinking and has_separate_thinking:
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
        elif not has_inline_thinking and has_separate_thinking:
            # Convert reasoning_content → inline <think> regardless of
            # whether content is text or None, and regardless of tool_calls.
            # Keeping a single representation prevents ensure_thinking from
            # adding an empty block on top of the real reasoning.
            has_thinking = True
            if not strip_thinking:
                thinking = f"<think>\n{raw_reasoning.strip(chr(10))}\n</think>"
                content = (
                    thinking
                    if content is None or content == ""
                    else f"{thinking}\n\n{content.lstrip(chr(10))}"
                )
            raw_reasoning = None

        has_thinking = has_thinking or has_inline_thinking
        if strip_thinking and content is not None:
            content = _extract_thinking(content)
        if content is None:
            # Preserve None for tool-only assistant messages; otherwise use
            # the empty string expected by ordinary assistant templates.
            cleaned["content"] = None if msg.get("tool_calls") else ""
        else:
            cleaned["content"] = content
    elif content is None:
        # Non-assistant messages without content: default to empty string.
        cleaned["content"] = ""
    else:
        cleaned["content"] = content

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
    # on the tokenizer, otherwise the stock template (Bailing / Qwen3 /
    # other ``last_query_index`` families) will extract and discard the
    # empty ``<think>`` block.
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
    """True if assistant *msg* has non-empty thinking content."""
    if msg.get("role") != "assistant":
        return False
    normalized = _normalize_thinking_tags(msg.get("content") or "")
    if any(match.group(1).strip() for match in _THINK_RE.finditer(normalized)):
        return True
    rc = msg.get("reasoning_content") or ""
    return bool(rc.strip())


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
):
    """Split trajectory into progressive pairs and optionally filter.

    By default, thinking (``<think>...</think>``) is stripped from context
    assistant turns only; the last assistant turn (training target) keeps
    its content unchanged.  Set *strip_all_thinking* to strip from every
    assistant turn including the target.

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
    Returns:
        Tuple of ``(pairs, n_filtered_errors, n_filtered_empty_tc,
        n_filtered_bare_tc)``.
    """
    segments = _find_segments(messages)
    if not segments:
        return [], 0, 0, 0

    pairs = []
    n_filtered_errors = 0
    n_filtered_empty_tc = 0
    n_filtered_bare_tc = 0

    # Pre-clean all messages in context mode (thinking stripped).
    # This avoids re-cleaning the same message for every progressive pair
    # (O(N+K) instead of O(N*K) where K = number of segments).
    context_cleaned = [_clean_message(m, strip_thinking=True) for m in messages]

    # For target assistant turns, clean with thinking preserved (unless
    # strip_all_thinking is set, in which case context_cleaned is reusable).
    target_cleaned = {}
    if not strip_all_thinking:
        for asst_start, _ in segments:
            target_cleaned[asst_start] = _clean_message(
                messages[asst_start],
                strip_thinking=False,
                ensure_thinking=True,
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
            pair[asst_start] = target_cleaned[asst_start]
        pairs.append(pair)

    return pairs, n_filtered_errors, n_filtered_empty_tc, n_filtered_bare_tc


def _prepare_trajectory(
    messages,
    filter_errors=True,
    filter_empty_tool_calls=False,
    filter_bare_text_tool_calls=False,
):
    """Prepare a full trajectory for trajectory-level training.

    Cleans all messages preserving thinking for every assistant turn
    (``strip_thinking=False``, ``ensure_thinking=True``).  Identifies
    which assistant segments should be masked (``loss_mask=0``) based
    on error tool responses, empty tool calls, or bare-text tool calls.

    Args:
        messages: Raw trajectory messages.
        filter_errors: If True (default), mask segments with error tool
            responses.
        filter_empty_tool_calls: If True, mask segments whose assistant
            turn has no text content but has tool_calls.
        filter_bare_text_tool_calls: If True, mask segments whose
            assistant turn has text without ``<think>`` tags and has
            tool_calls.
    Returns:
        Tuple of ``(cleaned_messages, masked_segment_indices,
        n_error, n_empty_tc, n_bare_tc)`` or ``None`` if the trajectory
        has no assistant turns.
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

    cleaned = [
        _clean_message(m, strip_thinking=False, ensure_thinking=True) for m in messages
    ]

    return (
        cleaned,
        sorted(masked_indices),
        n_error,
        n_empty_tc,
        n_bare_tc,
    )

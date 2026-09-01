# SPDX-License-Identifier: Apache-2.0

"""Chat-template handling, tokenization, loss masking, and sample dumps."""

import json
import os
import re

from areal.utils import logging

from .messages import _msg_has_thinking

logger = logging.getLogger("SWESFTDataset")

DATASET_NUM_PROC = 1

_TRAINING_PATCH_MARKER = "{# areal-swe-sft-training-patch-v1 #}"
_GENERATION_START_RE = re.compile(r"\{%-?\s*generation\s*-?%\}")
_GENERATION_END_RE = re.compile(r"\{%-?\s*endgeneration\s*-?%\}")


# -- Chat template patch (runtime, no file modification) --------

# Bailing / Qwen3 (and other ``ns.last_query_index`` families) prevent
# ``<think>`` rendering for assistant turns BEFORE the last user message, AND
# discard inline empty ``<think>\n</think>`` extracted from content.
#
# Without patching this breaks trajectory-mode training:
# - Multi-user trajectories: turns before the last user msg lack <think>
# - Empty ensure_thinking via inline <think> gets stripped
#
# The patch below handles Bailing (`<role>ASSISTANT</role>` style) and Qwen3
# (`<|im_start|>assistant` style) via exact block replacement:
# 1. Adds ``had_think_tags`` detection so empty ``<think>`` survives.
# 2. Removes the ``ns.last_query_index`` gate so all assistant turns render
#    ``<think>`` uniformly when think intent is detected — which is why
#    multi-user trajectories are now handled (no think loss).
# An unrecognized family (e.g. a future template revision) fails closed; add
# its exact block here rather than blindly neutralizing it.
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
_BAILING_CURRENT_OLD_BLOCK = (
    "{%- if loop.index0 > ns.last_query_index %}\n"
    "            {%- if reasoning_content %}\n"
    "                {{- '<role>ASSISTANT</role>' + '\\n<think>\\n'"
    " + reasoning_content.strip('\\n') + '\\n</think>\\n\\n'"
    " + content.lstrip('\\n') }}\n"
    "            {%- else %}\n"
    "                {{- '<role>ASSISTANT</role>' + content }}\n"
    "            {%- endif %}\n"
    "        {%- else %}\n"
    "            {{- '<role>ASSISTANT</role>' + content }}\n"
    "        {%- endif %}"
)
_BAILING_NEW_BLOCK = (
    "{{- '<role>ASSISTANT</role>' }}\n"
    "        {%- generation %}\n"
    "{%- if reasoning_content != '' or had_think_tags %}\n"
    "                {{- '\\n<think>\\n'"
    " + reasoning_content.strip('\\n') + '\\n</think>\\n\\n'"
    " + content.lstrip('\\n') }}\n"
    "            {%- else %}\n"
    "                {{- '\\n' + content }}\n"
    "            {%- endif %}"
)
_BAILING_CURRENT_NEW_BLOCK = (
    "{{- '<role>ASSISTANT</role>' }}\n"
    "        {%- generation %}\n"
    "{%- if reasoning_content or had_think_tags %}\n"
    "                {{- '\\n<think>\\n'"
    " + reasoning_content.strip('\\n') + '\\n</think>\\n\\n'"
    " + content.lstrip('\\n') }}\n"
    "            {%- else %}\n"
    "                {{- content }}\n"
    "            {%- endif %}"
)
_BAILING_ASSISTANT_END = (
    "        {{- '<|role_end|>' }}\n    {%- elif message.role == \"tool\" %}"
)
_BAILING_ASSISTANT_END_WITH_GENERATION = (
    "        {{- '<|role_end|>' }}\n"
    "        {%- endgeneration %}\n"
    '    {%- elif message.role == "tool" %}'
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
    "{{- '<|im_start|>' + message.role + '\\n' }}\n"
    "        {%- generation %}\n"
    "{%- if loop.last or reasoning_content != '' or had_think_tags %}\n"
    "                {{- '<think>\\n'"
    " + reasoning_content.strip('\\n') + '\\n</think>\\n\\n'"
    " + content.lstrip('\\n') }}\n"
    "            {%- else %}\n"
    "                {{- content }}\n"
    "            {%- endif %}"
)
_QWEN3_ASSISTANT_END = (
    "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}"
)
_QWEN3_ASSISTANT_END_WITH_GENERATION = (
    "        {{- '<|im_end|>\\n' }}\n"
    "        {%- endgeneration %}\n"
    '    {%- elif message.role == "tool" %}'
)

_OLD_DETECT = "{%- set reasoning_content = '' %}"
_NEW_DETECT = (
    "{%- set reasoning_content = '' %}\n"
    "        {%- set had_think_tags = ('</think>' in content) %}"
)

_BAILING_V3_ASSISTANT_START = (
    '{%- elif message.role == "assistant" %}\n'
    "        {%- set reasoning_content = '' %}"
)
_BAILING_V3_ASSISTANT_START_WITH_GENERATION = (
    '{%- elif message.role == "assistant" %}\n'
    "        {{- '<role>ASSISTANT</role>' }}\n"
    "        {%- generation %}\n"
    "        {%- set reasoning_content = '' %}"
)
_BAILING_V3_ASSISTANT_END = (
    "        {{- '<|role_end|>' }}\n    {%- elif message.role == \"tool\" %}"
)
_BAILING_V3_ASSISTANT_END_WITH_GENERATION = (
    "        {{- '<|role_end|>' }}\n"
    "        {%- endgeneration %}\n"
    '    {%- elif message.role == "tool" %}'
)


def _add_bailing_v3_generation_tags(template):
    """Mark Bailing V3 assistant bodies without changing rendered text."""
    if "preserved_thinking = true" not in template:
        return None
    if _BAILING_V3_ASSISTANT_START not in template:
        return None
    if template.count(_BAILING_V3_ASSISTANT_END) != 1:
        return None

    patched = template.replace(
        _BAILING_V3_ASSISTANT_START,
        _BAILING_V3_ASSISTANT_START_WITH_GENERATION,
        1,
    ).replace(
        _BAILING_V3_ASSISTANT_END,
        _BAILING_V3_ASSISTANT_END_WITH_GENERATION,
        1,
    )

    # The header is now emitted once, outside the tracked generation region.
    thinking_header = "'<role>ASSISTANT</role>' + "
    empty_thinking_header = "'<role>ASSISTANT</role>\\n<think></think>' + "
    if patched.count(thinking_header) != 1 or patched.count(empty_thinking_header) != 2:
        return None
    patched = patched.replace(thinking_header, "", 1).replace(
        empty_thinking_header,
        "'\\n<think></think>' + ",
        2,
    )
    return patched


def _has_generation_tracking(template: str) -> bool:
    """Return whether a template has balanced assistant-generation tags."""
    starts = len(_GENERATION_START_RE.findall(template))
    ends = len(_GENERATION_END_RE.findall(template))
    return starts > 0 and starts == ends


def _require_generation_tracking(tokenizer) -> None:
    template = getattr(tokenizer, "chat_template", None) or ""
    if not _has_generation_tracking(template):
        raise ValueError(
            "SWE SFT requires a chat template with Jinja generation tracking "
            "to build a structural assistant loss mask. Delimiter-based "
            "masking is unsafe when message payloads contain role or "
            "end-of-turn tokens."
        )


def _add_generation_end(template: str, family: str) -> str:
    if family == "Bailing":
        old_end = _BAILING_ASSISTANT_END
        new_end = _BAILING_ASSISTANT_END_WITH_GENERATION
    else:
        old_end = _QWEN3_ASSISTANT_END
        new_end = _QWEN3_ASSISTANT_END_WITH_GENERATION

    if template.count(old_end) != 1:
        raise ValueError(
            f"{family} assistant render block matched, but its end-of-turn "
            "boundary was not found exactly once. Refusing to create an "
            "incomplete generation mask."
        )
    return template.replace(old_end, new_end, 1)


def _patch_chat_template_for_training(tokenizer):
    """Patch Bailing/Qwen3 chat templates to render ``<think>`` uniformly.

    Detects template family by matching known render blocks:
    - Bailing V2.5: ``<role>ASSISTANT</role>`` + ``reasoning_content != ''``
    - Qwen3: ``<|im_start|>assistant`` markers

    Bailing V3 *adaptive* (config_ling_adaptive) needs no thinking patch because
    ``preserved_thinking = true`` already renders thinking uniformly. Its
    assistant body is instrumented with Jinja generation tags so Transformers
    can construct a structural loss mask without parsing role delimiters.

    Other templates without ``last_query_index`` are left unchanged and must
    already provide Jinja generation tracking. If a gated template matches no
    known block, patching fails closed; add an explicit block pattern for that
    family rather than relying on blind neutralization.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template or _TRAINING_PATCH_MARKER in template:
        return
    if "last_query_index" not in template:
        return

    # Bailing V3 adaptive already preserves thinking. Add Transformers'
    # generation tracking around assistant bodies so loss masks do not depend
    # on delimiters that may also occur literally in message payloads.
    if "preserved_thinking = true" in template and "preserved_thinking or" in template:
        patched = _add_bailing_v3_generation_tags(template)
        if patched is None:
            raise ValueError(
                "Bailing V3 adaptive template preserves thinking but its "
                "assistant block does not match the verified generation-mask "
                "layout. Refusing to trust pre-existing or partial generation "
                "tags; update the known Bailing V3 template boundaries."
            )
        tokenizer.chat_template = _TRAINING_PATCH_MARKER + patched
        logger.info(
            "Bailing V3 adaptive template detected: ensured generation "
            "tracking around assistant bodies."
        )
        return

    if _BAILING_OLD_BLOCK in template:
        family = "Bailing"
        patched = template.replace(_BAILING_OLD_BLOCK, _BAILING_NEW_BLOCK)
    elif _BAILING_CURRENT_OLD_BLOCK in template:
        family = "Bailing"
        patched = template.replace(
            _BAILING_CURRENT_OLD_BLOCK,
            _BAILING_CURRENT_NEW_BLOCK,
        )
    elif _QWEN3_OLD_BLOCK in template:
        family = "Qwen3"
        patched = template.replace(_QWEN3_OLD_BLOCK, _QWEN3_NEW_BLOCK)
    else:
        # Reaching here means the template family needs the training patch
        # (it gates rendering on last_query_index) but the verbatim block no
        # longer matches — most likely an upstream template revision. Failing
        # loudly beats silently training on data whose <think> blocks the
        # stock template strips (see _clean_message / ensure_thinking).
        #
        raise ValueError(
            "Chat template has last_query_index but matches neither the known "
            "Bailing nor Qwen3 render block; the training patch cannot be "
            "applied. Without it, empty <think> blocks are discarded and "
            "multi-turn thinking renders inconsistently. Update "
            "the known Bailing/Qwen3 block patterns for this template revision."
        )

    patched = _add_generation_end(patched, family)

    if _OLD_DETECT not in patched:
        raise ValueError(
            "Chat template render block matched but the reasoning_content "
            "detect line did not; had_think_tags would be undefined and "
            "empty <think> blocks would silently vanish. Update "
            "_OLD_DETECT/_NEW_DETECT for this template revision."
        )
    patched = patched.replace(_OLD_DETECT, _NEW_DETECT)

    tokenizer.chat_template = _TRAINING_PATCH_MARKER + patched
    logger.info(
        f"Patched {family} chat template for training: removed "
        "last_query_index gate and added structural generation tracking."
    )


def _parse_tool_call_arguments(messages):
    """Parse JSON-string arguments in tool_calls to dicts.

    OpenAI returns tool_call arguments as JSON strings, while some chat
    templates expect parsed dicts. This conversion is opt-in because other
    templates require the standard OpenAI string form.
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


def _prepare_chat_template_call(
    messages,
    tokenizer,
    tools=None,
    *,
    parse_tool_call_args=False,
):
    """Prepare messages and keyword arguments for a chat-template call."""
    _require_generation_tracking(tokenizer)
    template = getattr(tokenizer, "chat_template", None) or ""
    kwargs = {}
    if tools is not None:
        kwargs["tools"] = tools

    # Adaptive-thinking templates (Bailing V3 config_ling_adaptive) read an
    # ``enable_thinking`` flag that sets the ``detailed thinking on/off``
    # system label. Match it to whether this sample actually contains
    # thinking. In trajectory mode this naturally becomes a per-trajectory
    # decision; in pair mode it is a per-pair decision.
    if "enable_thinking" in template and "preserved_thinking" in template:
        kwargs["enable_thinking"] = any(_msg_has_thinking(m) for m in messages)
    if parse_tool_call_args:
        messages = _parse_tool_call_arguments(messages)
    return messages, kwargs


def _tokenize_and_mask(
    messages,
    tokenizer,
    tools=None,
    *,
    split_mode="pair",
    error_indices=None,
    parse_tool_call_args=False,
):
    """Tokenize once with native generation tracking and build ``loss_mask``.

    In **pair mode** (default), only the **last** assistant turn gets
    ``loss_mask=1``.  In **trajectory mode**, **all** assistant turns
    get ``loss_mask=1`` except those at indices in *error_indices*.

    When *parse_tool_call_args* is True, JSON-string ``tool_calls`` arguments
    are converted to dicts before rendering. This remains opt-in because many
    templates require the standard OpenAI string form.

    Returns:
        Tuple of ``(input_ids, loss_mask)``, or ``None`` if native assistant
        mask generation fails.
    """
    messages, kwargs = _prepare_chat_template_call(
        messages,
        tokenizer,
        tools,
        parse_tool_call_args=parse_tool_call_args,
    )
    try:
        tracked = tokenizer.apply_chat_template(
            messages,
            **kwargs,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_ids = tracked["input_ids"]
        tracked_mask = list(tracked["assistant_masks"])
        if len(tracked_mask) != len(input_ids):
            raise ValueError("assistant mask length differs from tokenized input")
    except Exception as e:
        logger.warning(
            "Native assistant-mask generation failed: %s. Skipping sample.",
            e,
        )
        return None

    loss_mask = [0] * len(input_ids)
    segments = []
    start = None
    for idx, enabled in enumerate([*tracked_mask, 0]):
        if enabled and start is None:
            start = idx
        elif not enabled and start is not None:
            segments.append((start, idx))
            start = None

    n_asst = sum(1 for m in messages if m.get("role") == "assistant")
    if len(segments) != n_asst:
        logger.warning(
            "Native assistant-mask segment mismatch: %d assistant messages "
            "but %d tracked segments. Skipping sample.",
            n_asst,
            len(segments),
        )
        return None

    if split_mode == "trajectory":
        skip = set(error_indices) if error_indices else set()
        selected = (
            segment for seg_idx, segment in enumerate(segments) if seg_idx not in skip
        )
    else:
        selected = segments[-1:] if segments else []
    for start, end in selected:
        loss_mask[start:end] = [1] * (end - start)
    return input_ids, loss_mask


def _render_tokenize_mask(
    messages,
    tokenizer,
    tools=None,
    *,
    split_mode="pair",
    error_indices=None,
    parse_tool_call_args=False,
):
    """Build a token mask and additionally render offsets for sample dumps.

    Training calls :func:`_tokenize_and_mask` directly and therefore performs
    exactly one native tracked tokenization. This helper intentionally does
    the extra render and offset tokenization needed by human-readable dumps.
    """
    result = _tokenize_and_mask(
        messages,
        tokenizer,
        tools,
        split_mode=split_mode,
        error_indices=error_indices,
        parse_tool_call_args=parse_tool_call_args,
    )
    if result is None:
        return None
    input_ids, loss_mask = result

    messages, kwargs = _prepare_chat_template_call(
        messages,
        tokenizer,
        tools,
        parse_tool_call_args=parse_tool_call_args,
    )
    try:
        full_text = tokenizer.apply_chat_template(messages, **kwargs, tokenize=False)
        encoding = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        rendered_ids = encoding["input_ids"]
        offset_mapping = encoding["offset_mapping"]
        if rendered_ids != input_ids or len(offset_mapping) != len(input_ids):
            raise ValueError("dump tokenization differs from tracked tokenization")
    except Exception as e:
        logger.warning("Sample dump rendering failed: %s. Skipping sample.", e)
        return None

    return full_text, input_ids, loss_mask, offset_mapping


class _TokenizeAndMask:
    """Picklable callable for ``Dataset.map(num_proc=N)``."""

    def __init__(
        self,
        tokenizer,
        max_length=None,
        *,
        split_mode="pair",
        parse_tool_call_args=False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split_mode = split_mode
        self.parse_tool_call_args = parse_tool_call_args

    def __call__(self, sample):
        error_indices = (
            sample.get("error_indices", []) if self.split_mode == "trajectory" else None
        )
        tools_json = sample.get("tools_json")
        tools = json.loads(tools_json) if tools_json else None
        result = _tokenize_and_mask(
            sample["messages"],
            self.tokenizer,
            tools,
            split_mode=self.split_mode,
            error_indices=error_indices,
            parse_tool_call_args=self.parse_tool_call_args,
        )
        if result is None:
            return {"input_ids": [], "loss_mask": []}

        input_ids, loss_mask = result

        # Early exit: overlength or empty → return empty so a single
        # filter pass removes it together with template-failure empties.
        if self.max_length is not None and len(input_ids) > self.max_length:
            return {"input_ids": [], "loss_mask": []}

        return {"input_ids": input_ids, "loss_mask": loss_mask}


def _dump_samples(
    samples,
    tokenizer,
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

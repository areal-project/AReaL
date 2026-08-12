# SPDX-License-Identifier: Apache-2.0

"""Chat-template handling, tokenization, loss masking, and sample dumps."""

import json
import os
import re

from areal.utils import logging

from .messages import _msg_has_thinking

logger = logging.getLogger("SWESFTDataset")

DATASET_NUM_PROC = 1


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
# An unrecognized family (e.g. a future ring3.0 revision) is skipped with a
# warning; add its exact block here rather than blindly neutralizing.
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


def _patch_chat_template_for_training(tokenizer):
    """Patch Bailing/Qwen3 chat templates to render ``<think>`` uniformly.

    Detects template family by matching known render blocks:
    - Bailing V2.5: ``<role>ASSISTANT</role>`` + ``reasoning_content != ''``
    - Qwen3: ``<|im_start|>assistant`` markers

    Bailing V3 *adaptive* (config_ling_adaptive) needs no thinking patch because
    ``preserved_thinking = true`` already renders thinking uniformly. Its
    assistant body is instrumented with Jinja generation tags so Transformers
    can construct a structural loss mask without parsing role delimiters.

    Other templates (plain ChatML without ``last_query_index``) are left
    unchanged.  If the template has ``last_query_index`` but no known block
    matches, logs a warning and skips — add an explicit block pattern for that
    family rather than relying on a blind neutralization.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template or "last_query_index" not in template:
        return

    # Bailing V3 adaptive already preserves thinking. Add Transformers'
    # generation tracking around assistant bodies so loss masks do not depend
    # on delimiters that may also occur literally in message payloads.
    if "preserved_thinking = true" in template and "preserved_thinking or" in template:
        patched = _add_bailing_v3_generation_tags(template)
        if patched is not None:
            tokenizer.chat_template = patched
            logger.info(
                "Bailing V3 adaptive template detected: added generation "
                "tracking around assistant bodies."
            )
            return
        logger.info(
            "Bailing V3 adaptive template detected (preserved_thinking=true): "
            "all assistant turns already render <think>, but its assistant "
            "block was not recognized for generation tracking."
        )
        return

    if _BAILING_OLD_BLOCK in template:
        family = "Bailing"
        patched = template.replace(_BAILING_OLD_BLOCK, _BAILING_NEW_BLOCK)
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
        # Escape hatch for uses that do not depend on think normalization
        # (e.g. precision-alignment forward dumps): set
        # AREAL_SWE_ALLOW_UNPATCHED_TEMPLATE=1 to proceed with a warning.
        if os.environ.get("AREAL_SWE_ALLOW_UNPATCHED_TEMPLATE", ""):
            logger.warning(
                "Chat template has last_query_index but matches neither known "
                "render block; proceeding UNPATCHED because "
                "AREAL_SWE_ALLOW_UNPATCHED_TEMPLATE is set. Empty <think> "
                "blocks may be discarded by the stock template."
            )
            return
        raise ValueError(
            "Chat template has last_query_index but matches neither the known "
            "Bailing nor Qwen3 render block; the training patch cannot be "
            "applied. Without it, empty <think> blocks are discarded and "
            "multi-turn thinking renders inconsistently. Update "
            "_BAILING_OLD_BLOCK/_QWEN3_OLD_BLOCK for this template revision, "
            "or set AREAL_SWE_ALLOW_UNPATCHED_TEMPLATE=1 if think "
            "normalization is irrelevant for this run."
        )

    if _OLD_DETECT not in patched:
        raise ValueError(
            "Chat template render block matched but the reasoning_content "
            "detect line did not; had_think_tags would be undefined and "
            "empty <think> blocks would silently vanish. Update "
            "_OLD_DETECT/_NEW_DETECT for this template revision."
        )
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
    # Bailing (V2.5 / V3 / V3-adaptive):  <role>ASSISTANT</role> ... <|role_end|>
    # Must be a KNOWN pattern (not probe-derived): the V3-adaptive template
    # (config_ling_adaptive) renders an empty ``<think></think>`` for
    # non-reasoning turns, which the double-probe would bake into the header and
    # then fail to match real reasoning turns (``<think>{reasoning}</think>``).
    # This header/eot captures the whole turn (think + content + tool_calls).
    (r"<role>ASSISTANT</role>", r"<\|role_end\|>"),
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
        # Adaptive-thinking templates (Bailing V3 config_ling_adaptive) read an
        # ``enable_thinking`` flag that sets the ``detailed thinking on/off``
        # system label. Match it to whether THIS rendered example actually
        # contains thinking, so the switch is trained consistently
        # (on<->think, off<->no-think); training an all-no-think example under
        # "on" (or a thinking example under "off") would corrupt the switch.
        # In trajectory mode ``messages`` is the whole trajectory (=> per-traj
        # rule: on iff >=1 assistant turn thinks); in pair mode it is the pair.
        # Gated on the preserved_thinking + enable_thinking signature so other
        # templates (e.g. Qwen3) are left untouched.
        _tmpl = getattr(tokenizer, "chat_template", None) or ""
        if "enable_thinking" in _tmpl and "preserved_thinking" in _tmpl:
            kwargs["enable_thinking"] = any(_msg_has_thinking(m) for m in messages)
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

    # Templates instrumented with Jinja's generation extension provide
    # structural assistant spans. Unlike delimiter matching, these spans cannot
    # be forged by literal role markers in user/tool payloads.
    if re.search(r"\{%-?\s*generation\s*-?%\}", _tmpl):
        try:
            tracked_kwargs = {
                **kwargs,
                "tokenize": True,
                "return_dict": True,
                "return_assistant_tokens_mask": True,
            }
            tracked = tokenizer.apply_chat_template(messages, **tracked_kwargs)
            tracked_ids = tracked["input_ids"]
            tracked_mask = list(tracked["assistant_masks"])
            if tracked_ids != input_ids or len(tracked_mask) != len(input_ids):
                raise ValueError("tracked tokenization differs from rendered text")
        except Exception as e:
            logger.warning(
                "Native assistant-mask generation failed: %s. Skipping sample.",
                e,
            )
            return None

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
                segment
                for seg_idx, segment in enumerate(segments)
                if seg_idx not in skip
            )
        else:
            selected = segments[-1:] if segments else []
        for start, end in selected:
            loss_mask[start:end] = [1] * (end - start)
        return full_text, input_ids, loss_mask, offset_mapping

    if split_mode == "trajectory":
        # Trajectory mode: mask ALL assistant segments, skip error_indices.
        skip = set(error_indices) if error_indices else set()
        matches = list(assistant_pattern.finditer(full_text))

        # Verify regex matches correspond 1:1 to assistant messages.
        n_asst = sum(1 for m in messages if m.get("role") == "assistant")
        if len(matches) != n_asst:
            # Fail closed: a spurious match (e.g. a tool output quoting the
            # chat-template header literal) would otherwise put loss on
            # user/tool tokens and silently defeat error masking.
            logger.warning(
                "Segment count mismatch: %d assistant messages but %d regex "
                "matches in rendered text. Dropping this sample.",
                n_asst,
                len(matches),
            )
            return None

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
        else:
            # Loss lands nowhere; the SFT loss path tolerates all-zero masks
            # (kept, not dropped, to preserve cache compatibility) but this
            # always signals template/pattern drift worth investigating.
            logger.warning(
                "No assistant segment matched the template pattern; sample "
                "keeps an all-zero loss_mask."
            )

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

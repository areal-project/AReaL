# SPDX-License-Identifier: Apache-2.0

"""Chat-template handling, tokenization, loss masking, and sample dumps."""

import json
import os
import random
import re

from datasets import Dataset

from areal.utils import logging

logger = logging.getLogger("SWESFTDataset")

DATASET_NUM_PROC = 1


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

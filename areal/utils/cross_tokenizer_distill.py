# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CrossTokenizerEncoding:
    """Teacher-tokenized sequence plus response span metadata."""

    teacher_input_ids: list[int]
    teacher_response_ids: list[int]
    student_response_ids: list[int]


def detect_tokenizer_family(tokenizer: Any) -> str:
    """Infer a tokenizer family from well-known chat-template tokens."""

    vocab = tokenizer.get_vocab()
    if "<|begin_of_text|>" in vocab or "<|start_header_id|>" in vocab:
        return "llama"
    if "<|im_start|>" in vocab:
        return "qwen"
    if "<｜begin▁of▁sentence｜>" in vocab or "<｜User｜>" in vocab:
        return "deepseek"
    return "unknown"


def build_chat_template_mapping(student_tokenizer: Any, teacher_tokenizer: Any):
    """Return string replacements that convert student chat markers to teacher ones."""

    student_family = detect_tokenizer_family(student_tokenizer)
    teacher_family = detect_tokenizer_family(teacher_tokenizer)
    if student_family == teacher_family:
        return []
    if student_family == "llama" and teacher_family == "qwen":
        return [
            ("<|begin_of_text|><|start_header_id|>", "<|im_start|>"),
            ("<|start_header_id|>", "<|im_start|>"),
            ("<|end_header_id|>\n\n", "\n"),
            ("<|eot_id|>", "<|im_end|>\n"),
            ("<|begin_of_text|>", ""),
            ("<|end_of_text|>", "<|im_end|>\n"),
        ]
    if student_family == "llama" and teacher_family == "deepseek":
        return [
            (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
                "<｜begin▁of▁sentence｜>",
            ),
            (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
                "<｜begin▁of▁sentence｜><｜User｜>",
            ),
            ("<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", "<｜User｜>"),
            (
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                "<｜Assistant｜>",
            ),
            ("<|eot_id|>", "<｜end▁of▁sentence｜>"),
            ("<|begin_of_text|>", "<｜begin▁of▁sentence｜>"),
            ("<|end_of_text|>", "<｜end▁of▁sentence｜>"),
        ]
    if student_family == "qwen" and teacher_family == "deepseek":
        return [
            ("<|im_start|>system\n", "<｜begin▁of▁sentence｜>"),
            ("<|im_end|>\n<|im_start|>user\n", "<｜User｜>"),
            ("<|im_start|>user\n", "<｜begin▁of▁sentence｜><｜User｜>"),
            ("<|im_end|>\n<|im_start|>assistant\n", "<｜Assistant｜>"),
            ("<|im_end|>\n", "<｜end▁of▁sentence｜>"),
            ("<|im_end|>", "<｜end▁of▁sentence｜>"),
            ("<|endoftext|>", "<｜end▁of▁sentence｜>"),
        ]
    if student_family == "qwen" and teacher_family == "llama":
        return [
            (
                "<|im_start|>system\n",
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
            ),
            (
                "<|im_start|>user\n",
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
            ),
            (
                "<|im_start|>assistant\n",
                "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n\n",
            ),
            ("<|im_end|>\n", "<|eot_id|>"),
            ("<|im_end|>", "<|eot_id|>"),
            ("<|endoftext|>", "<|end_of_text|>"),
        ]
    if student_family == "deepseek" and teacher_family == "llama":
        return [
            ("<｜begin▁of▁sentence｜>", "<|begin_of_text|>"),
            ("<｜User｜>", "<|start_header_id|>user<|end_header_id|>\n\n"),
            ("<｜Assistant｜>", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
            ("<｜end▁of▁sentence｜>", "<|eot_id|>"),
        ]
    if student_family == "deepseek" and teacher_family == "qwen":
        return [
            ("<｜begin▁of▁sentence｜><｜User｜>", "<|im_start|>user\n"),
            ("<｜User｜>", "<|im_start|>user\n"),
            ("<｜Assistant｜>", "<|im_start|>assistant\n"),
            ("<｜end▁of▁sentence｜>", "<|im_end|>\n"),
        ]
    raise NotImplementedError(
        "Unsupported cross-tokenizer distillation pair: "
        f"{student_family!r} student to {teacher_family!r} teacher."
    )


def retokenize_text(
    token_ids: list[int],
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    template_mapping: list[tuple[str, str]] | None = None,
) -> list[int]:
    """Decode student IDs with special tokens and re-encode with teacher tokenizer."""

    if student_tokenizer.get_vocab() == teacher_tokenizer.get_vocab():
        return token_ids
    if template_mapping is None:
        template_mapping = build_chat_template_mapping(
            student_tokenizer, teacher_tokenizer
        )
    text = student_tokenizer.decode(token_ids, skip_special_tokens=False)
    for old, new in template_mapping:
        text = text.replace(old, new)
    if detect_tokenizer_family(student_tokenizer) == "llama" and text.endswith("\n"):
        text = text.rstrip("\n")
    return teacher_tokenizer(text, add_special_tokens=False)["input_ids"]


def retokenize_for_distillation(
    active_student_ids: list[int],
    response_mask: list[bool],
    student_tokenizer: Any,
    teacher_tokenizer: Any,
) -> CrossTokenizerEncoding:
    """Retokenize an active sequence and its response span for teacher scoring."""

    if len(active_student_ids) != len(response_mask):
        raise ValueError(
            "active_student_ids and response_mask must have the same length, got "
            f"{len(active_student_ids)} and {len(response_mask)}."
        )
    template_mapping = build_chat_template_mapping(student_tokenizer, teacher_tokenizer)
    student_response_ids = [
        token_id
        for token_id, is_response in zip(active_student_ids, response_mask)
        if is_response
    ]
    teacher_input_ids = retokenize_text(
        active_student_ids, student_tokenizer, teacher_tokenizer, template_mapping
    )
    teacher_response_ids = retokenize_text(
        student_response_ids, student_tokenizer, teacher_tokenizer, template_mapping
    )
    return CrossTokenizerEncoding(
        teacher_input_ids=teacher_input_ids,
        teacher_response_ids=teacher_response_ids,
        student_response_ids=student_response_ids,
    )


def align_teacher_logps_to_student(
    student_ids: list[int],
    teacher_ids: list[int],
    teacher_logps: torch.Tensor,
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    student_logps: torch.Tensor | None = None,
    *,
    large_chunk_threshold: int = 6,
) -> torch.Tensor:
    """Project teacher response log-probs to student-token granularity.

    Greedily forms chunks whose decoded Unicode-normalized text is identical on
    both tokenizers. If ``student_logps`` is provided, each teacher chunk
    likelihood is distributed by the semantic-prior rule from cross-tokenizer
    OPD:

    ``log q_i = (L_T_chunk / L_S_chunk) * log p_i``.

    This preserves the student's within-chunk likelihood shape while matching
    the teacher's chunk-level likelihood. If no student prior is provided, the
    function falls back to uniform chunk assignment. Unaligned or suspiciously
    large chunks are set to ``inf`` so the actor loss can turn them into no-op
    tokens.
    """

    aligned = torch.full(
        (len(student_ids),),
        float("inf"),
        dtype=teacher_logps.dtype,
        device=teacher_logps.device,
    )
    s_ptr = 0
    t_ptr = 0
    while s_ptr < len(student_ids) and t_ptr < len(teacher_ids):
        s_end = s_ptr + 1
        t_end = t_ptr + 1
        matched = False
        while s_end <= len(student_ids) and t_end <= len(teacher_ids):
            s_text = unicodedata.normalize(
                "NFC",
                student_tokenizer.decode(
                    student_ids[s_ptr:s_end], skip_special_tokens=False
                ),
            )
            t_text = unicodedata.normalize(
                "NFC",
                teacher_tokenizer.decode(
                    teacher_ids[t_ptr:t_end], skip_special_tokens=False
                ),
            )
            if s_text == t_text and not s_text.endswith("\ufffd"):
                n_student = s_end - s_ptr
                n_teacher = t_end - t_ptr
                if (
                    n_student <= large_chunk_threshold
                    and n_teacher <= large_chunk_threshold
                ):
                    teacher_chunk_logp = teacher_logps[t_ptr:t_end].sum()
                    if student_logps is None:
                        aligned[s_ptr:s_end] = teacher_chunk_logp / n_student
                    else:
                        student_chunk_logps = student_logps[s_ptr:s_end]
                        student_chunk_logp = student_chunk_logps.sum()
                        if torch.isclose(
                            student_chunk_logp,
                            torch.zeros_like(student_chunk_logp),
                        ):
                            aligned[s_ptr:s_end] = teacher_chunk_logp / n_student
                        else:
                            aligned[s_ptr:s_end] = (
                                teacher_chunk_logp
                                / student_chunk_logp
                                * student_chunk_logps
                            )
                s_ptr = s_end
                t_ptr = t_end
                matched = True
                break
            prev = (s_end, t_end)
            if len(s_text) <= len(t_text) and s_end < len(student_ids):
                s_end += 1
            elif t_end < len(teacher_ids):
                t_end += 1
            elif s_end < len(student_ids):
                s_end += 1
            if prev == (s_end, t_end):
                break
        if not matched:
            break
    return aligned


def replace_unaligned_teacher_logps(
    teacher_logp: torch.Tensor, student_logp: torch.Tensor
) -> torch.Tensor:
    """Replace ``inf`` sentinel positions with student log-probs for zero KL."""

    return torch.where(
        torch.isposinf(teacher_logp), student_logp.detach(), teacher_logp
    )

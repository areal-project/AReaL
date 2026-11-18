# SPDX-License-Identifier: Apache-2.0

"""Rollout metadata and experience-conditioned MOPD teacher scoring hooks.

Configure ``ExperienceRolloutHook`` under ``rollout.data_hook`` and
``ExperienceTeacherHook`` under ``mopd.teacher_engine.data_hook``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch

from areal.api.data_hook import ComputeLogpHook, RolloutHook
from areal.utils.hf_utils import load_hf_tokenizer

_HOOK_DATA_KEY = "_experience_opd"
_EXPERIENCE_TEMPLATE = (
    "<experiences>\n"
    "Here are some experiences learned in previous interactions. "
    "Always check and leverage these experiences before you act.\n"
    "{}\n</experiences>\n"
)


class ExperienceStore:
    """Read-only experience text indexed by SWE instance id."""

    def __init__(self, path: str):
        with Path(path).open(encoding="utf-8") as file:
            values = json.load(file)
        if not isinstance(values, dict) or not values:
            raise ValueError("Experience file must contain a non-empty object")
        self.values: dict[str, str | list[str]] = {}
        for instance_id, value in values.items():
            texts = value if isinstance(value, list) else [value]
            if (
                not isinstance(instance_id, str)
                or not texts
                or any(not isinstance(text, str) or not text.strip() for text in texts)
            ):
                raise ValueError(f"Invalid experience for instance {instance_id!r}")
            self.values[instance_id] = value

    def choose(self, instance_id: str) -> int:
        value = self.values[instance_id]
        return random.randrange(len(value)) if isinstance(value, list) else 0

    def text(self, instance_id: str, index: int) -> str:
        value = self.values[instance_id]
        if isinstance(value, list):
            return value[index]
        if index != 0:
            raise IndexError(index)
        return value


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _user_turn_markers(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Derive user header and turn terminator from the model chat template."""
    user = "__AREAL_EXPERIENCE_USER_7F2A__"
    system = "__AREAL_EXPERIENCE_SYSTEM_9D4B__"
    try:
        user_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=False,
        )
        with_system = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as exc:
        raise ValueError("Cannot render the tokenizer chat template") from exc
    if (
        not isinstance(user_only, str)
        or not isinstance(with_system, str)
        or user_only.count(user) != 1
        or with_system.count(user) != 1
        or with_system.count(system) != 1
    ):
        raise ValueError("Chat template did not preserve the marker text")

    _, user_end = user_only.split(user)
    system_and_user_start, system_user_end = with_system.split(user)
    if user_end != system_user_end:
        raise ValueError("Chat template changed the user terminator")
    system_content_end = system_and_user_start.index(system) + len(system)
    system_turn_end = system_and_user_start.find(user_end, system_content_end)
    if system_turn_end < 0:
        raise ValueError("Cannot find the system turn terminator")
    user_start = system_and_user_start[system_turn_end + len(user_end) :].rstrip("\r\n")
    turn_end = user_end.rstrip("\r\n")
    start_ids = _encode(tokenizer, user_start)
    end_ids = _encode(tokenizer, turn_end)
    if not start_ids or not end_ids:
        raise ValueError("Chat template produced empty user-turn markers")
    return start_ids, end_ids


def _find_subsequence(sequence: torch.Tensor, pattern: torch.Tensor) -> int:
    size = pattern.numel()
    if size == 0 or sequence.numel() < size:
        return -1
    matches = sequence.unfold(0, size, 1).eq(pattern).all(dim=-1)
    indices = matches.nonzero(as_tuple=False)
    return -1 if indices.numel() == 0 else int(indices[0, 0].item())


def _teacher_view(
    trajectory: dict[str, Any],
    experience_ids: list[int],
    user_start_ids: list[int],
    turn_end_ids: list[int],
    max_context_length: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Insert experience before the first user-turn end; return original positions."""
    required = ("input_ids", "loss_mask", "attention_mask")
    if any(not isinstance(trajectory.get(key), torch.Tensor) for key in required):
        raise ValueError("Teacher input requires input_ids, loss_mask, attention_mask")
    input_ids, loss_mask, attention_mask = (trajectory[key] for key in required)
    if (
        input_ids.ndim != 2
        or loss_mask.shape != input_ids.shape
        or attention_mask.shape != input_ids.shape
    ):
        raise ValueError("Teacher input tensors must share shape [batch, seq]")

    # Pipeline padding creates an attended one-token dummy with no response.
    if input_ids.shape[-1] == 1 and not loss_mask.bool().any():
        return {key: trajectory[key] for key in required}, torch.zeros(
            (input_ids.shape[0], 1), dtype=torch.long
        )

    offset = len(experience_ids)
    if input_ids.shape[-1] + offset > max_context_length:
        raise ValueError("Experience insertion exceeds teacher context length")
    device = input_ids.device
    experience = torch.tensor(experience_ids, dtype=input_ids.dtype, device=device)
    user_start = torch.tensor(user_start_ids, dtype=input_ids.dtype, device=device)
    turn_end = torch.tensor(turn_end_ids, dtype=input_ids.dtype, device=device)
    transformed = {key: [] for key in required}
    positions = []
    for row, mask, attention in zip(input_ids, loss_mask, attention_mask, strict=True):
        start = _find_subsequence(row, user_start)
        end = _find_subsequence(row[start:], turn_end) if start >= 0 else -1
        if end < 0:
            raise ValueError("Cannot find first user turn in teacher input")
        insert_at = start + end
        generated = mask.bool().nonzero(as_tuple=False)
        if generated.numel() and insert_at > int(generated[0, 0].item()):
            raise ValueError("Experience insertion must precede generated tokens")
        transformed["input_ids"].append(
            torch.cat((row[:insert_at], experience, row[insert_at:]))
        )
        transformed["loss_mask"].append(
            torch.cat(
                (
                    mask[:insert_at],
                    torch.zeros(offset, dtype=mask.dtype, device=device),
                    mask[insert_at:],
                )
            )
        )
        transformed["attention_mask"].append(
            torch.cat(
                (
                    attention[:insert_at],
                    torch.ones(offset, dtype=attention.dtype, device=device),
                    attention[insert_at:],
                )
            )
        )
        positions.append(
            torch.cat(
                (torch.arange(insert_at), torch.arange(insert_at, row.numel()) + offset)
            )
        )
    return (
        {key: torch.stack(rows) for key, rows in transformed.items()},
        torch.stack(positions),
    )


class ExperienceRolloutHook(RolloutHook):
    """Attach the chosen experience id to an exported trajectory."""

    def __init__(self, *, experience_path: str):
        self.store = ExperienceStore(experience_path)

    def pre_rollout(
        self, data: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, tuple[str, int] | None]:
        instance_id = data.get("data_id")
        if not isinstance(instance_id, str):
            raise ValueError("Experience OPD requires a string data_id")
        if instance_id not in self.store.values:
            return None, None
        return data, (instance_id, self.store.choose(instance_id))

    def post_rollout(
        self, trajectory: dict[str, Any], state: tuple[str, int] | None
    ) -> dict[str, Any]:
        if state is None:
            raise ValueError("Missing experience selection for rollout")
        if _HOOK_DATA_KEY in trajectory:
            raise ValueError("Trajectory already contains experience metadata")
        trajectory[_HOOK_DATA_KEY] = {"instance_id": state[0], "exp_index": state[1]}
        return trajectory


class ExperienceTeacherHook(ComputeLogpHook):
    """Score a teacher-only view and align logp to the original trajectory."""

    def __init__(
        self,
        *,
        experience_path: str,
        tokenizer_path: str,
        max_context_length: int = 131072,
    ):
        if max_context_length < 1:
            raise ValueError("max_context_length must be positive")
        self.store = ExperienceStore(experience_path)
        self.tokenizer = load_hf_tokenizer(tokenizer_path)
        self.user_start_ids, self.turn_end_ids = _user_turn_markers(self.tokenizer)
        self.max_context_length = max_context_length
        self._token_cache: dict[tuple[str, int], list[int]] = {}

    def pre_compute_logp(
        self, data: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
        transformed = []
        positions = []
        for trajectory in data:
            input_ids = trajectory.get("input_ids")
            loss_mask = trajectory.get("loss_mask")
            is_dummy = (
                isinstance(input_ids, torch.Tensor)
                and isinstance(loss_mask, torch.Tensor)
                and input_ids.shape[-1] == 1
                and not loss_mask.bool().any()
            )
            if is_dummy:
                experience_ids = []
            else:
                metadata = trajectory.get(_HOOK_DATA_KEY)
                if not isinstance(metadata, dict):
                    raise ValueError("Teacher trajectory lacks experience metadata")
                instance_id = metadata.get("instance_id")
                index = metadata.get("exp_index")
                if not isinstance(instance_id, str) or not isinstance(index, int):
                    raise ValueError("Invalid experience metadata")
                cache_key = (instance_id, index)
                if cache_key not in self._token_cache:
                    text = self.store.text(instance_id, index)
                    self._token_cache[cache_key] = _encode(
                        self.tokenizer, _EXPERIENCE_TEMPLATE.format(text)
                    )
                experience_ids = self._token_cache[cache_key]
            view, row_positions = _teacher_view(
                trajectory,
                experience_ids,
                self.user_start_ids,
                self.turn_end_ids,
                self.max_context_length,
            )
            transformed.append(view)
            positions.append(row_positions)
        return transformed, positions

    def post_compute_logp(
        self, outputs: list[torch.Tensor] | None, state: list[torch.Tensor]
    ) -> list[torch.Tensor] | None:
        if outputs is None:
            return None
        if len(outputs) != len(state):
            raise ValueError("Teacher logp count does not match input count")
        aligned = []
        for output, positions in zip(outputs, state, strict=True):
            if output.ndim != 2 or output.shape[0] != positions.shape[0]:
                raise ValueError("Teacher logp shape does not match input rows")
            aligned.append(output.gather(-1, positions.to(device=output.device)))
        return aligned

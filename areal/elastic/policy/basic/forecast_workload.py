# SPDX-License-Identifier: Apache-2.0

"""Causal episode continuations shared by arrival and task-pool forecasters."""

from __future__ import annotations

from dataclasses import replace

from .observations import EpisodeView, Turn


def continue_episode(
    config,
    episode: EpisodeView,
    prototype: tuple[Turn, ...],
    rng,
    now: float,
    tool_history: tuple[Turn, ...],
) -> tuple[Turn, ...]:
    current = episode.current
    assert current is not None
    index = len(episode.completed_turns)
    tail = list(prototype[index:] or (replace(prototype[-1], tool_seconds=None),))
    first = tail[0]
    if current.completed_at is None:
        eligible = [
            t.output_tokens for t in prototype if t.output_tokens > current.generated
        ]
        output = (
            rng.choice(eligible)
            if eligible
            else current.generated + max(1, first.output_tokens)
        )
        output = min(config.max_new_tokens, max(current.generated + 1, output))
        first = replace(first, input_tokens=current.input_tokens, output_tokens=output)
    else:
        # A tool invocation is already observable, even if a sampled prototype ended here.
        tool = episode.tool
        assert tool is not None
        elapsed = (
            max(0.0, now - tool.started_at) if tool.started_at is not None else 0.0
        )
        tools = [
            t
            for t in tool_history
            if t.tool_seconds is not None and t.tool_pool == tool.pool
        ]
        survivors = [t for t in tools if t.tool_seconds > elapsed]
        if survivors:
            source = rng.choice(survivors)
            total = float(source.tool_seconds)
        else:
            positive = [t for t in tools if t.tool_seconds > 0]
            source = (
                rng.choice(positive)
                if positive
                else Turn(
                    current.input_tokens,
                    current.generated,
                    config.unseen_tool_residual_seconds,
                    0,
                    tool.pool,
                )
            )
            # Outside observed support use an explicit positive residual,
            # rather than equating survival with immediate completion.
            total = elapsed + float(source.tool_seconds)
        first = replace(
            source,
            input_tokens=current.input_tokens,
            output_tokens=current.generated,
            tool_seconds=total,
            tool_pool=tool.pool,
        )
        if len(tail) == 1:
            tail.append(replace(prototype[-1], tool_seconds=None))
    tail[0] = first
    for i in range(1, len(tail)):
        previous = tail[i - 1]
        tail[i] = replace(
            tail[i],
            input_tokens=previous.input_tokens
            + previous.output_tokens
            + previous.tool_output_tokens
            + previous.next_input_overhead,
        )
    return episode.completed_turns + tuple(tail)

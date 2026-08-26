# SPDX-License-Identifier: Apache-2.0

"""Env-driven multi-turn tool-calling agent for vision-language models.

The agent is task-agnostic: it sends the turn-0 prompt, then loops
generate -> ``env.step(assistant_text)`` -> append the env's text observation as
the next user turn, until the env signals ``done`` or ``max_turns`` is reached.
All task semantics (initial prompt incl. tool/system instructions, action
parsing, grading, feedback, termination) live in a ``MultiTurnVisionEnv``
(e.g. ``examples/multi_turn_vlm/geo3k_env.py``).

Generation goes through AReaL's OpenAI proxy as ordinary chat completions, so
the proxy owns everything this module used to assemble by hand:

- vision-expanded prompt ids and ``pixel_values`` (the processor lives in the
  proxy worker, keyed off ``rollout.tokenizer_path``)
- per-turn token / logprob bookkeeping and loss masks
- reward propagation across turns (``rollout.agent.turn_discount``)
- trajectory export shape (``rollout.agent.export_style``)

Reward semantics are unchanged: envs score only the terminal turn, so returning
that scalar and letting the proxy propagate it backwards with
``turn_discount=1.0`` reproduces the previous flat terminal reward.

NOTE: under the ``ray`` scheduler, ``cluster.fileroot`` and ``cluster.name_resolve``
must be reachable from every node Ray may schedule a worker on; ``cluster.n_nodes``
sizes the resource request but does not pin placement to the driver's node.
"""

import os
from typing import Any

from openai import AsyncOpenAI

from areal.infra import workflow_context
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.image import image2base64
from areal.workflow.vision_env import EnvStepResult, MultiTurnVisionEnv

logger = logging.getLogger("VisionMultiTurnAgent")


def inline_images(
    messages_chat: list[dict[str, Any]],
    images: list[Any],
) -> list[dict[str, Any]]:
    """Fill empty ``image_url`` slots with base64 data URIs.

    RL datasets leave ``image_url.url`` empty because the pre-proxy path injected
    images out-of-band via ``ModelRequest.image_data``. Over the proxy the image
    travels inside the request body, so the bytes must be inlined here. Remote
    URLs are not an option: the proxy needs the bytes to build ``pixel_values``.
    """
    payloads = image2base64(images)
    remaining = iter(payloads)
    filled: list[dict[str, Any]] = []
    consumed = 0

    for message in messages_chat:
        content = message.get("content")
        if not isinstance(content, list):
            filled.append(dict(message))
            continue
        parts: list[Any] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                try:
                    payload = next(remaining)
                except StopIteration:
                    raise ValueError(
                        f"messages_chat has more image_url parts than the "
                        f"{len(payloads)} images supplied by the dataset row."
                    ) from None
                consumed += 1
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{payload}"},
                    }
                )
            else:
                parts.append(part)
        filled.append({**message, "content": parts})

    if consumed != len(payloads):
        raise ValueError(
            f"Dataset row supplied {len(payloads)} images but messages_chat has "
            f"{consumed} image_url parts; they must correspond one-to-one."
        )
    return filled


class VisionMultiTurnAgent:
    """Env-driven multi-turn tool-calling agent for vision-language models.

    Parameters
    ----------
    env_factory : str
        Dotted import path to a :class:`MultiTurnVisionEnv`, resolved per episode
        in the rollout worker.
    env_args : dict
        JSON-serializable kwargs for the environment constructor.
    max_turns : int
        Hard cap on generation turns; the env also terminates on success.
    max_completion_tokens : int
        Per-turn generation budget.
    temperature, top_p : float
        Sampling parameters, mirroring ``gconfig``.
    max_tokens_per_traj : int | None
        Stop taking further turns once another one could not fit. A multi-turn
        VLM sample cannot be split across microbatches (the image binds it to
        one), so an over-long episode would otherwise break FFD packing. Set it
        to ``actor.mb_spec.max_tokens_per_mb``.
    """

    def __init__(
        self,
        env_factory: str,
        env_args: dict[str, Any] | None = None,
        max_turns: int = 2,
        max_completion_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens_per_traj: int | None = None,
        **kwargs: Any,
    ):
        self.env_factory = env_factory
        self.env_args = dict(env_args or {})
        self.max_turns = max_turns
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens_per_traj = max_tokens_per_traj
        self.extra = kwargs

    def _budget_exhausted(self, completion: Any) -> bool:
        """True when another turn could not fit in one microbatch."""
        if self.max_tokens_per_traj is None:
            return False
        usage = getattr(completion, "usage", None)
        if usage is None or usage.total_tokens is None:
            return False
        # +128 covers the appended observation and turn framing.
        projected = usage.total_tokens + self.max_completion_tokens + 128
        return projected > self.max_tokens_per_traj

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> float:
        env: MultiTurnVisionEnv = import_from_string(self.env_factory)(**self.env_args)
        reset_out = env.reset(data)
        if not reset_out.images:
            raise ValueError("env.reset must return non-empty images for a VLM agent")
        messages = inline_images(reset_out.messages_chat, reset_out.images)

        client = AsyncOpenAI(
            base_url=extra_kwargs.get("base_url") or os.environ.get("OPENAI_BASE_URL"),
            api_key=extra_kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY"),
            http_client=extra_kwargs.get("http_client"),
            max_retries=0,
        )

        # None until the first env step, so an env that only returns penalties
        # exports its negative reward instead of being clipped to zero.
        reward: float | None = None
        for turn in range(self.max_turns):
            completion = await client.chat.completions.create(
                model="default",
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            message = completion.choices[0].message

            step_out: EnvStepResult = env.step(message.content or "")
            # Envs score only the terminal turn; max mirrors the previous
            # workflow, which kept the best turn reward.
            reward = step_out.reward if reward is None else max(reward, step_out.reward)

            if (
                step_out.done
                or turn == self.max_turns - 1
                or step_out.observation is None
                or self._budget_exhausted(completion)
            ):
                break

            # Append the assistant turn exactly as returned so the proxy's cache
            # can prefix-match this completion as the next turn's parent.
            messages = messages + [
                message.model_dump(exclude_none=True),
                {"role": "user", "content": step_out.observation.strip()},
            ]

        self._log_metrics(env)
        return float(reward if reward is not None else 0.0)

    def _log_metrics(self, env: MultiTurnVisionEnv) -> None:
        """Publish the env's per-episode metrics (reward is logged by the proxy)."""
        metrics = env.get_metrics()
        if not metrics:
            return
        try:
            stats_tracker.get(workflow_context.stat_scope()).scalar(**metrics)
        except Exception as e:  # pragma: no cover - metrics must never fail a rollout
            logger.warning(f"Failed to record episode metrics: {e}")

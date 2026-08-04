# SPDX-License-Identifier: Apache-2.0
"""Rollout workflow that drives any OpenEnv-compatible environment.

OpenEnv (https://github.com/huggingface/OpenEnv) is HuggingFace's uniform
adapter layer over agentic environments -- BrowserGym, OpenSpiel, Coding,
BlackJack, Terminal-Bench, etc. Each environment exposes
``reset() / step(action) / state()`` over WebSocket.

This workflow lets AReaL train against any such environment by pointing at an
EnvClient subclass and (optionally) an Action dataclass via config. No new
Python file is needed to plug in a new environment; only a YAML change.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

from areal import workflow_context
from areal.api import RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.api.openenv_api import OpenEnvConfig
from areal.experimental.openai import ArealOpenAI
from areal.utils import logging, stats_tracker
from areal.workflow.openenv_utils import (
    _import_from_string,
    build_action,
    resolve_action_parser,
    resolve_obs_formatter,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

    from areal.api import InferenceEngine

logger = logging.getLogger("OpenEnvWorkflow")


def _instantiate_provider(cfg: OpenEnvConfig) -> Any:
    """Lazily import the OpenEnv provider requested by ``cfg``."""
    if cfg.provider == "uv":
        from openenv.core.containers.runtime.uv_provider import UVProvider

        return UVProvider(project_path=cfg.project_path)
    if cfg.provider == "docker":
        from openenv.core.containers.runtime import LocalDockerProvider

        return LocalDockerProvider(image=cfg.docker_image)
    raise ValueError(f"Unknown OpenEnv provider: {cfg.provider!r}")


def _instantiate_env_client(cfg: OpenEnvConfig) -> Any:
    """Build the concrete ``EnvClient`` from ``cfg``.

    Deferred import keeps ``openenv`` an optional dependency: users who never
    touch this workflow do not need it installed.
    """
    env_client_cls = _import_from_string(cfg.env_client_class)
    kwargs: dict[str, Any] = {
        "connect_timeout_s": cfg.connect_timeout_s,
        "message_timeout_s": cfg.message_timeout_s,
    }
    if cfg.base_url is not None:
        kwargs["base_url"] = cfg.base_url
    else:
        kwargs["provider"] = _instantiate_provider(cfg)
    return env_client_cls(**kwargs)


class OpenEnvWorkflow(RolloutWorkflow):
    """Drive one episode against an OpenEnv environment.

    The loop is: ``reset() -> [chat.completion -> parse action -> env.step()] *
    N -> aggregate reward``. Each LLM turn is cached by :class:`ArealOpenAI`
    with the environment reward for that step, so the exported trajectory
    carries per-step supervision suitable for GRPO / PPO / RLOO.

    Parameters
    ----------
    config
        :class:`OpenEnvConfig` describing which environment to launch and how
        to convert observations/actions.
    gconfig
        Standard AReaL generation hyperparameters.
    tokenizer
        Tokenizer used by the underlying inference engine.
    initial_user_prompt
        Optional prompt string prepended before the first observation. When
        ``None``, only the observation-derived user message is sent.
    reward_shaping_fn
        Optional callable ``(step_result, step, episode_data) -> float`` that
        overrides the raw environment reward. Return the value that should be
        recorded for this step. Useful for cost/length penalties.
    """

    def __init__(
        self,
        config: OpenEnvConfig,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        initial_user_prompt: str | None = None,
        reward_shaping_fn: Any = None,
    ) -> None:
        self.config = config
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        # Grouped rollout is external; each workflow instance produces one trajectory.
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer).new(
            n_samples=1
        )
        self.tokenizer = tokenizer
        self.initial_user_prompt = initial_user_prompt
        self.reward_shaping_fn = reward_shaping_fn

        self._obs_formatter = resolve_obs_formatter(config.obs_formatter)
        self._action_parser = resolve_action_parser(config.action_parser)

    def _initial_messages(self, observation: Any) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        if self.initial_user_prompt:
            messages.append({"role": "user", "content": self.initial_user_prompt})
        messages.append(self._obs_formatter(observation, step=0))
        return messages

    async def _generate_step(
        self, client: ArealOpenAI, messages: list[dict[str, str]]
    ) -> Any:
        return await client.chat.completions.create(  # type: ignore[arg-type]
            messages=messages,
            frequency_penalty=self.gconfig.frequency_penalty,
            max_completion_tokens=self.gconfig.max_new_tokens,
            stop=self.gconfig.stop,
            store=True,
            temperature=self.gconfig.temperature,
            top_p=self.gconfig.top_p,
        )

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Run one full episode against the configured OpenEnv environment.

        ``data`` is forwarded verbatim to the tokenizer; recognized keys:

        * ``seed``: passed to ``env.reset(seed=...)`` when set.
        * ``system_prompt``: overrides ``config.system_prompt`` for this episode.
        """
        openai_client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)
        env_client = _instantiate_env_client(self.config)

        reset_kwargs = dict(self.config.reset_kwargs)
        if "seed" in data and "seed" not in reset_kwargs:
            reset_kwargs["seed"] = data["seed"]

        step_rewards: list[float] = []
        completion_ids: list[str] = []
        terminal_done = False

        async with env_client as env:
            result = await env.reset(**reset_kwargs)
            messages = self._initial_messages(result.observation)
            if "system_prompt" in data:
                # Replace any existing system message with the per-episode override.
                messages = [m for m in messages if m.get("role") != "system"]
                messages.insert(0, {"role": "system", "content": data["system_prompt"]})

            for step in range(self.config.max_turns):
                completion = await self._generate_step(openai_client, messages)
                assistant_text = completion.choices[0].message.content or ""
                completion_ids.append(completion.id)

                parsed = self._action_parser(assistant_text, result.observation)
                if parsed is None:
                    logger.debug(
                        f"Action parser returned None on step {step}; "
                        "treating as no-op with zero reward."
                    )
                    step_reward = 0.0
                    step_rewards.append(step_reward)
                    openai_client.set_reward(completion.id, step_reward)
                    # No env.step: unparsed action can't produce a valid observation.
                    break

                action = build_action(parsed, self.config.action_class)
                result = await env.step(action)

                if self.reward_shaping_fn is not None:
                    step_reward = float(self.reward_shaping_fn(result, step, data))
                else:
                    step_reward = float(result.reward or 0.0)
                step_rewards.append(step_reward)
                openai_client.set_reward(completion.id, step_reward)

                # Append assistant + next observation for the following turn.
                messages = deepcopy(messages)
                messages.append({"role": "assistant", "content": assistant_text})
                if not result.done and step + 1 < self.config.max_turns:
                    messages.append(
                        self._obs_formatter(result.observation, step=step + 1)
                    )

                if result.done:
                    terminal_done = True
                    break

        # Trajectory-level bookkeeping.
        if self.config.terminal_reward_only and completion_ids:
            # Zero out intermediate rewards; keep the last one as episode reward.
            terminal_reward = step_rewards[-1] if step_rewards else 0.0
            for cid in completion_ids[:-1]:
                openai_client.set_reward(cid, 0.0)
            openai_client.set_reward(completion_ids[-1], terminal_reward)

        # Apply per-step discount by backward propagation. When step_discount
        # == 1.0 this is an identity operation (rewards already summed
        # implicitly via GRPO grouping); the discount reshapes credit for
        # earlier turns.
        if self.config.step_discount < 1.0:
            openai_client.apply_reward_discount(self.config.step_discount)

        episode_reward = float(sum(step_rewards))
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            reward=episode_reward,
            num_turns=len(step_rewards),
            terminated=float(terminal_done),
        )
        return openai_client.export_interactions("individual")


__all__ = ["OpenEnvWorkflow"]

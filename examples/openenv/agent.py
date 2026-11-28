# SPDX-License-Identifier: Apache-2.0

"""Run the OpenEnv Echo environment with an inline OpenAI proxy agent."""

import argparse
import asyncio
import os
from contextvars import ContextVar
from typing import Any

from areal.experimental.env.openenv import OpenEnvBackend
from areal.experimental.env.rollout import (
    EnvRolloutRejected,
    EnvRolloutResult,
    MissingRewardPolicy,
    RewardMode,
    TruncationPolicy,
    run_environment,
)
from areal.infra import workflow_context
from areal.utils import logging, stats_tracker

logger = logging.getLogger("OpenEnvEcho")


def echo_backend(url: str) -> OpenEnvBackend:
    from echo_env import EchoEnv

    return OpenEnvBackend(lambda _task: EchoEnv(base_url=url))


def echo_action(message: str) -> Any:
    from echo_env import CallToolAction

    return CallToolAction(tool_name="echo_message", arguments={"message": message})


class EchoAgent:
    def __init__(
        self,
        max_turns: int = 1,
        *,
        reward_mode: RewardMode = "episode",
        missing_reward_policy: MissingRewardPolicy = "reject",
        truncation_policy: TruncationPolicy = "retain_zero",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
    ):
        self.backend = echo_backend(os.environ["OPENENV_BASE_URL"])
        self.max_turns = max_turns
        self.reward_mode = reward_mode
        self.missing_reward_policy = missing_reward_policy
        self.truncation_policy = truncation_policy
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = (
            max_completion_tokens if max_completion_tokens is not None else max_tokens
        )
        self._result: ContextVar[EnvRolloutResult | None] = ContextVar(
            "openenv_rollout", default=None
        )

    @staticmethod
    def should_reject_failed_sample(error: Exception, _disposition: str) -> bool:
        return isinstance(error, EnvRolloutRejected)

    def get_episode_metadata(self) -> dict[str, Any]:
        result = self._result.get()
        if result is None:
            return {}
        return {
            "env_terminated": result.terminated,
            "env_truncated": result.truncated,
            "env_stop_reason": result.stop_reason,
        }

    async def run(
        self, data: dict[str, Any], **proxy_context: Any
    ) -> float | dict[str, float]:
        from openai import AsyncOpenAI

        self._result.set(None)
        if self.reward_mode == "dense":
            worker_runtime = proxy_context.get("worker_runtime")
            if (
                worker_runtime is None
                or worker_runtime.config.agent.export_style != "individual"
            ):
                raise ValueError(
                    "EchoAgent dense rewards require agent.export_style='individual' "
                    "on the inline worker"
                )

        client = AsyncOpenAI(
            base_url=proxy_context["base_url"],
            api_key=proxy_context["api_key"],
            http_client=proxy_context["http_client"],
            max_retries=0,
        )
        messages = [dict(message) for message in data["messages"]]

        async def decide(observation: Any) -> tuple[str, Any]:
            messages.append({"role": "user", "content": str(observation)})
            response = await client.chat.completions.create(
                model="default",
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_completion_tokens=self.max_completion_tokens,
            )
            message = response.choices[0].message
            content = message.content
            if content is None:
                raise ValueError("EchoAgent expects a text completion")
            messages.append(message.model_dump(exclude_none=True))
            return response.id, echo_action(content)

        result = await run_environment(
            self.backend,
            data,
            decide,
            max_turns=self.max_turns,
            reward_mode=self.reward_mode,
            missing_reward_policy=self.missing_reward_policy,
            truncation_policy=self.truncation_policy,
        )
        self._result.set(result)
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            env_truncated=float(result.truncated)
        )
        logger.debug(
            "OpenEnv rollout finished (turns=%d, truncated=%s, stop_reason=%r)",
            result.turns,
            result.truncated,
            result.stop_reason,
        )
        return result.proxy_reward()


async def demo(url: str, message: str) -> None:
    async def decide(_observation: Any) -> tuple[str, Any]:
        return "echo-demo", echo_action(message)

    result = await run_environment(
        echo_backend(url), {}, decide, max_turns=1, truncation_policy="retain_zero"
    )
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one OpenEnv Echo rollout")
    parser.add_argument("--env-url", required=True)
    parser.add_argument("--message", default="Hello, OpenEnv")
    args = parser.parse_args()
    asyncio.run(demo(args.env_url, args.message))

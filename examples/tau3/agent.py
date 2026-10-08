"""Adapter from AReaL's proxy workflow to τ³-Bench's synchronous simulator."""

import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from litellm import register_model
from tau2.agent.llm_agent import LLMAgent, LLMSoloAgent
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.user.user_simulator import DummyUser, UserSimulator

from areal.utils import logging

logger = logging.getLogger("Tau3Agent")

register_model(
    {
        "dummy": {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    }
)


@dataclass
class Tau3EnvConfig:
    """Text τ³-Bench simulator options; the user API key comes from the environment."""

    domain: str = "airline"
    max_steps: int = 50
    episode_timeout_seconds: float = 600.0
    solo_mode: bool = False
    user_llm_base_url: str | None = None
    user_llm: str | None = None
    user_llm_args: dict[str, Any] = field(default_factory=dict)


class Tau3AgentWorkflow:
    """Run one τ³-Bench task through the AReaL proxy and return its reward."""

    def __init__(
        self,
        econfig: Tau3EnvConfig | dict[str, Any] | None = None,
        gen_args: dict[str, Any] | None = None,
        timeout: float = 600.0,
    ):
        self.econfig = (
            Tau3EnvConfig(**econfig) if isinstance(econfig, dict) else econfig
        ) or Tau3EnvConfig()
        self.gen_args = gen_args or {}
        self.timeout = timeout

    def _run_sync(self, data: dict[str, Any], base_url: str, api_key: str) -> float:
        config = self.econfig
        task_id = data["task_id"]
        split = data.get("split", "train")
        tasks = registry.get_tasks_loader(config.domain)(split)
        task = next((item for item in tasks if item.id == task_id), None)
        if task is None:
            raise ValueError(
                f"τ³-Bench task {task_id!r} missing from {config.domain}/{split}"
            )

        environment = registry.get_env_constructor(config.domain)(
            solo_mode=config.solo_mode
        )
        tools = environment.get_tools()
        try:
            user_tools = environment.get_user_tools()
        except ValueError:
            user_tools = []
        agent_args = {
            "api_base": base_url,
            "api_key": api_key,
            "temperature": self.gen_args.get("temperature", 1.0),
            "max_tokens": self.gen_args.get("max_completion_tokens", 1024),
        }

        if config.solo_mode:
            agent = LLMSoloAgent(
                tools=tools + user_tools,
                domain_policy=environment.get_policy(),
                llm="openai/dummy",
                llm_args=agent_args,
                task=task,
            )
            user = DummyUser()
        else:
            if not config.user_llm_base_url or not config.user_llm:
                raise ValueError("τ³-Bench user LLM URL and model are required")
            key_file = os.environ.get("TAU3_USER_API_KEY_FILE")
            if not key_file:
                raise ValueError(
                    "TAU3_USER_API_KEY_FILE is required for simulated users"
                )
            user_api_key = Path(key_file).read_text().strip()
            if not user_api_key:
                raise ValueError("The τ³-Bench user API key file is empty")
            agent = LLMAgent(
                tools=tools,
                domain_policy=environment.get_policy(),
                llm="openai/dummy",
                llm_args=agent_args,
            )
            user = UserSimulator(
                tools=user_tools or None,
                instructions=str(task.user_scenario),
                llm=f"openai/{config.user_llm}",
                llm_args={
                    "api_base": config.user_llm_base_url,
                    "api_key": user_api_key,
                    **config.user_llm_args,
                },
            )

        simulator = Orchestrator(
            domain=config.domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=config.max_steps,
            solo_mode=config.solo_mode,
            timeout=self.timeout,
        )
        simulation = simulator.run()
        reward = evaluate_simulation(
            simulation=simulation,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=config.solo_mode,
            domain=config.domain,
        ).reward
        logger.info(f"τ³-Bench {config.domain}/{task_id}: reward={reward}")
        return reward

    async def run(self, data: dict[str, Any], **extra_kwargs: Any) -> float:
        """Keep the rollout controller's event loop free during synchronous simulation."""
        base_url = extra_kwargs.get("base_url")
        api_key = extra_kwargs.get("api_key")
        if not base_url or not api_key:
            raise ValueError("AReaL proxy base_url and api_key are required")
        return await asyncio.wait_for(
            asyncio.to_thread(self._run_sync, data, base_url, api_key), self.timeout
        )

from __future__ import annotations

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from agent.camel_terminal_agent import CamelTerminalAgent
from agent_rl_config import TaskTimeouts
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker
from areal.utils.perf_tracer import atrace_scope

from .pre_build_tasks_utils import build_docker_image


class CamelRLVRWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        n_trajs: int = 1,
        max_tokens: int = 32768,
        max_iteration: int = 50,
        max_workers: int = 25,
        non_think_mode: bool = True,
        task_timeouts: TaskTimeouts | None = None,
        filter_uniform_reward: bool = False,
        encourage_completion_reward: bool = False,
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.max_tokens = max_tokens
        self.max_iteration = max_iteration
        self.rollout_stat_scope = rollout_stat_scope
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        self.n_trajs = n_trajs
        self.non_think_mode = non_think_mode
        self.task_timeouts = task_timeouts or TaskTimeouts()
        self.filter_uniform_reward = filter_uniform_reward
        self.encourage_completion_reward = encourage_completion_reward
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def arun_episode(self, engine, data):
        clients = [
            ArealOpenAI(
                engine=engine,
                tokenizer=self.tokenizer,
                tool_call_parser="qwen25",
            )
            for _ in range(self.n_trajs)
        ]
        uids = [uuid.uuid4().hex[:8] for _ in range(self.n_trajs)]

        loop = asyncio.get_running_loop()
        try:
            async with atrace_scope(
                f"build_docker_image:{data.get('task_name')}",
                args={"timeout": self.task_timeouts._reset_env},
            ):
                await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        partial(
                            build_docker_image,
                            task=data,
                            timeout=self.task_timeouts._reset_env,
                        ),
                    ),
                    timeout=self.task_timeouts._reset_env + 60.0,
                )
        except TimeoutError:
            print(
                f"Timeout while building docker image for task {data.get('task_name')}"
            )
            return None

        print(f"\n{'=' * 70}")
        print(f"[EPISODE START] Task {data.get('task_name')}")
        print(f"{'=' * 70}\n")

        rewards = await asyncio.gather(
            *[
                CamelTerminalAgent(
                    max_tokens_per_turn=self.gconfig.max_new_tokens,
                    max_total_tokens=self.max_tokens,
                    max_iteration=self.max_iteration,
                    output_path=f"{self.dump_dir}/CamelTerminalAgent_Output",
                    executor=self.executor,
                    non_think_mode=self.non_think_mode,
                    task_timeouts=self.task_timeouts,
                    encourage_completion_reward=self.encourage_completion_reward,
                ).run_agent(
                    data=data,
                    client=clients[i],
                    uid=uids[i],
                    traj_i=i,
                )
                for i in range(self.n_trajs)
            ]
        )

        print(f"\n{'=' * 70}")
        print(f"[EPISODE END] Task {data.get('task_name')}")
        print(f"{'=' * 70}\n")

        completions_with_reward = {}
        if self.filter_uniform_reward:
            valid_rewards = [reward for reward in rewards if reward is not None]
            if valid_rewards and all(
                reward == valid_rewards[0] for reward in valid_rewards
            ):
                print(
                    f"Rank {os.getenv('RANK', '0')} - Task {data.get('task_name')} "
                    "has uniform reward across trajectories. Discarding all."
                )
                return completions_with_reward
            if not valid_rewards:
                print(
                    f"Rank {os.getenv('RANK', '0')} - Task {data.get('task_name')} "
                    "all trajectories failed."
                )
                return completions_with_reward

        for i, (reward, client) in enumerate(zip(rewards, clients)):
            if reward is None:
                print(
                    f"Rank {os.getenv('RANK', '0')} - Task {data.get('task_name')}, "
                    f"Trajectory {i} failed."
                )
                os.makedirs(f"{self.dump_dir}/failed_tasks", exist_ok=True)
                with open(
                    f"{self.dump_dir}/failed_tasks/{data.get('task_name')}_traj_{i}.txt",
                    "w",
                ) as f:
                    f.write(f"Task {data.get('task_name')} trajectory {i} failed.\n")
                continue

            print(
                f"Rank {os.getenv('RANK', '0')} - Task {data.get('task_name')}, "
                f"Trajectory {i} reward: {reward}"
            )
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)
            client.apply_reward_discount(turn_discount=0.9)
            completions = client.export_interactions(style="individual")
            completions_with_reward.update(completions)

        if len(completions_with_reward) == 0:
            print(f"All trajectories failed for task {data.get('task_name')}.")
            completions_with_reward = None

        stats_tracker.get(self.rollout_stat_scope).scalar(
            num_full_passes=sum(1 for reward in rewards if reward == 1.0)
        )
        stats_tracker.get(self.rollout_stat_scope).scalar(
            num_trajectories_failed=sum(1 for reward in rewards if reward is None)
        )

        print(
            f"Rank {os.getenv('RANK', '0')} - Task {data.get('task_name')} completed."
        )
        return completions_with_reward

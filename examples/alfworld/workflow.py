"""ALFWorld GRPO Workflow for AReaL.

Uses env_client to communicate with a separate env_server process
that handles textworld interactions.
"""
import re
import logging
from typing import Any

from openai.types.chat import ChatCompletion

from areal.api import RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker
from areal import workflow_context

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\nAction: your next action".

The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. put {obj} in/on {recep}
4. open {recep}
5. close {recep}
6. use {obj}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:
Thought: <your thoughts>
Action: <your next action>"""


def parse_action(llm_response: str) -> str:
    if not llm_response:
        return "look"

    text = re.sub(r"<think>.*?</think>", "", llm_response, flags=re.DOTALL).strip()
    if not text:
        text = re.sub(r"</?think>", "", llm_response).strip()
        if not text:
            return "look"

    action_match = re.search(
        r"(?i)^[\s\-\d.]*\*{0,2}action\*{0,2}[\s]*:[\s]*(.+)", text, re.MULTILINE
    )
    if action_match:
        action = action_match.group(1).strip().split("\n")[0].strip().rstrip(".")
        if not action:
            return "look"
        return action.lower()

    thought_action = re.search(
        r"(?i)thought\s*:.*?\n\s*\*{0,2}action\*{0,2}\s*:\s*(.+)", text, re.DOTALL
    )
    if thought_action:
        action = thought_action.group(1).strip().split("\n")[0].strip().rstrip(".")
        if not action:
            return "look"
        return action.lower()

    alf_commands = (
        "go to ", "take ", "put ", "open ", "close ", "use ",
        "clean ", "heat ", "cool ", "examine ", "look", "inventory",
    )
    for line in text.split("\n"):
        line = re.sub(r"^[\d.\-\*\s]+", "", line.strip()).strip()
        line_lower = line.lower()
        for cmd in alf_commands:
            if line_lower.startswith(cmd):
                return line_lower.rstrip(".")

    return "look"


class ALFWorldAgent:
    def __init__(self, gconfig: GenerationHyperparameters, max_steps: int = 30):
        self.gconfig = gconfig
        self.max_steps = max_steps

    async def run_agent(self, data: dict, client: ArealOpenAI) -> float:
        from examples.alfworld.env_client import get_client
        env = get_client()

        game_file = data["game_file"]
        task_desc = data["task_desc"]

        try:
            env_id = await env.create(game_file)
        except Exception as e:
            logger.warning("Env create failed for %s: %s", game_file, e)
            return 0.0

        try:
            try:
                obs, info = await env.reset(env_id)
            except Exception as e:
                logger.warning("Env reset failed for %s: %s", game_file, e)
                return 0.0

            task_description = "\n".join(obs.split("\n\n")[1:]) if "\n\n" in obs else obs
            if task_desc:
                task_description = task_desc

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]

            failure_summary = data.get("failure_summary", "")
            if failure_summary:
                messages.append({"role": "user", "content": (
                    "Before you start, here are common failure patterns from past attempts on similar tasks. "
                    "Use them to avoid repeating mistakes:\n\n"
                    "--- FAILURE PATTERNS TO AVOID ---\n" + failure_summary
                )})

            messages.append({"role": "user", "content": f"Now, it's your turn to solve a new task.\n{task_description}"})

            reward = 0.0
            last_response_id = None

            for step in range(self.max_steps):
                response: ChatCompletion = await client.chat.completions.create(
                    messages=messages,
                    model="default",
                    **self.gconfig.to_openai_args_dict(),
                )
                if not response.choices:
                    action = "look"
                    content = ""
                else:
                    message = response.choices[0].message
                    content = message.content or ""
                    action = parse_action(content)

                messages.append({"role": "assistant", "content": content})
                last_response_id = response.id

                try:
                    obs, step_reward, done, won = await env.step(env_id, action)
                except Exception as e:
                    logger.warning("Env step failed for %s (step %d): %s", game_file, step, e)
                    client.set_reward(response.id, 0.0)
                    break

                if done:
                    reward = 1.0 if (won or step_reward > 0) else 0.0
                    client.set_reward(response.id, reward)
                    break
                else:
                    client.set_reward(response.id, 0.0)

                messages.append({"role": "user", "content": f"Observation: {obs.strip()}"})
            else:
                if last_response_id:
                    client.set_reward(last_response_id, 0.0)

            return reward
        finally:
            await env.close(env_id)


class ALFWorldWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: Any,
        export_style: str = "concat",
        max_steps: int = 30,
        turn_discount: float = 0.9,
    ):
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer
            tokenizer = load_hf_tokenizer(tokenizer)

        self.tokenizer = tokenizer
        self.export_style = export_style
        self.chat_template_type = "concat" if export_style == "concat" else "hf"
        self.turn_discount = turn_discount

        self.agent = ALFWorldAgent(
            gconfig=gconfig.new(n_samples=1),
            max_steps=max_steps,
        )

    async def arun_episode(self, engine, data):
        client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            chat_template_type=self.chat_template_type,
        )

        reward = await self.agent.run_agent(data=data, client=client)
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        client.apply_reward_discount(turn_discount=self.turn_discount)
        completions_with_reward = client.export_interactions(style=self.export_style)
        return completions_with_reward

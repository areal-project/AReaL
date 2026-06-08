# Agent Workflow Training

Train custom agents using the OpenAI Python SDK or high-level agent frameworks.

## Quick Start

```bash
python3 examples/agent_workflow/train.py \
    --config examples/agent_workflow/config.yaml \
    scheduler.type=local
```

This script will run the example agent in
[`areal/workflow/openai/math_agent.py`](../../areal/workflow/openai/math_agent.py). You
can also modify the `workflow` parameter in `trainer.train` to run other example agents
defined in the same file.

## Write Your Own Agent

1. Write a Python **`async` function** that takes a dict as agent input. The function
   must directly use the OpenAI Python SDK or use it indirectly through a high-level
   agent framework (e.g.,
   [OpenAI Agent](https://openai.github.io/openai-agents-python/),
   [CAMEL-AI](https://www.camel-ai.org/)). The function can either return a float as the
   final reward, or a dict where the reward of each interaction is keyed by the
   completion or response ID.

1. Wrap the function within AReaL's [`AgentWorkflow`](../../areal/api/workflow_api.py).
   This class is for pure typing and API regulation usage.

```python
async def my_agent(data: dict) -> float:
    ...

from areal.api.workflow_api import AgentWorkflow

class MyAgentWorkflow(AgentWorkflow):
    async def run(self, data: dict):
        return await my_agent(data)
```

3. Place your agent code in a path that can be imported in Python (e.g.,
   `areal/workflow/openai/my_agent.py`), and update the `workflow` parameter in the
   training script to reference that path (e.g.,
   `areal.workflow.openai.my_agent.MyAgent`).

See `areal/workflow/openai/` for concrete examples.

## Implementation Notes

1. Ensure that your agent workflow uses async functions and `await` to maximize
   concurrency, especially for overlapping file I/O, HTTP requests, or multiple LLM
   generations.
1. We recommend using the AReaL-provided `base_url`, `api_key`, and `http_client` from
   `extra_kwargs` to construct the `AsyncOpenAI` object for sending completion requests.
   Constructing a `httpx.AsyncClient` in each workflow will increase latency by ~50ms,
   resulting in a ~10s overhead for a batch of 256.
1. It is also valid to ignore the provided argument. You can instead start a subprocess
   (i.e., setting `rollout.agent.mode=subproc`) to run the agent. While this provides
   more flexibility for writing the agent, using a subprocess will introduce even larger
   overhead.

# Agent Workflow

This document describes AReaL's agent workflow system, which enables training language
models using agent frameworks while capturing token-level data for reinforcement
learning.

**Notes**:

1. This page targets developers seeking a deep understanding of the codebase. For a
   practical guide, see the [Agentic RL Guide](../tutorial/agentic_rl.md).

1. Read the [`RolloutWorkflow` reference](../reference/rollout_workflow.md) first, as
   agent workflows are built on top of `RolloutWorkflow`.

1. **Scheduler compatibility**: Agent workflows are supported on `local` and `slurm`
   schedulers only. The `ray` scheduler is incompatible with the HTTP proxy
   architecture.

## Overview

Agent workflows allow training models using popular agent frameworks (OpenAI Agents SDK,
CAMEL-AI, LangChain, etc.) without modifying their core logic. AReaL automatically
captures token-level information needed for RL training while preserving the agent's
original behavior.

Key benefits:

- **Flexibility**: Supports any framework using OpenAI/Anthropic messaging protocols
- **Unified development**: Same code for benchmarking, evaluation, and RL training
- **Algorithmic correctness**: Token-level tracking avoids training-inference mismatch

The challenge is that agent frameworks interact with LLMs through high-level APIs that
don't expose token IDs and log probabilities. AReaL solves this by:

1. **Intercepting LLM calls** via a proxy server or direct client
1. **Tracking token-level data** in an `InteractionCache`
1. **Building conversation trees** for multi-turn reward propagation
1. **Exporting training-ready tensors** with proper reward attribution

## Relationship with RolloutWorkflow

Agent workflows are not a separate abstraction—they are automatically wrapped into
`RolloutWorkflow` through `OpenAIProxyWorkflow`:

```
User's Agent Code (async def run())
           ↓
   OpenAIProxyWorkflow (wrapper)
           ↓
   RolloutWorkflow.arun_episode()
           ↓
   dict[str, InteractionWithTokenLogpReward]
           ↓
   Tensor dictionary for training
```

Any class with an `async def run(data, **extra_kwargs)` method is recognized as an agent
workflow and wrapped automatically when passed to the trainer.

## Two Integration Paradigms

AReaL offers two approaches for integrating agent frameworks:

| Aspect                  | Proxy Approach                              | Direct Approach                            |
| ----------------------- | ------------------------------------------- | ------------------------------------------ |
| **Code modification**   | None (just change `base_url` and `api_key`) | Must accept `ArealOpenAI` client           |
| **Communication**       | HTTP via proxy server                       | Direct engine calls                        |
| **Framework support**   | Any OpenAI-compatible framework             | Frameworks accepting custom clients        |
| **Performance**         | HTTP overhead (minimal)                     | No HTTP overhead                           |
| **Engine state access** | Limited                                     | Full access                                |
| **Recommended for**     | Existing agents, third-party frameworks     | Legacy code. **Don't use it proactively.** |

See the [Agentic RL Guide](../tutorial/agentic_rl.md) for concrete examples.

### Proxy Approach

The proxy approach keeps agent code independent from AReaL. Your agent uses the standard
OpenAI/Anthropic messaging protocol with a customized `base_url` pointing to AReaL's
proxy server.

AReaL's trainer automatically provides `base_url` and `http_client` during RL training.

```python
class MyAgent:
    async def run(self, data, **extra_kwargs):
        # AReaL injects these kwargs
        http_client = extra_kwargs.get("http_client")
        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")

        # Standard OpenAI SDK usage
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )

        response = await client.chat.completions.create(
            model="default",
            messages=data["messages"],
        )

        # Return reward (float) or reward dict
        return compute_reward(response, data["answer"])
```

### Direct Approach

> **Legacy Pattern**: The direct approach using `ArealOpenAI` with `RolloutWorkflow` is
> considered legacy and should not be used for new projects. Prefer the proxy approach
> above, which keeps agent code independent from AReaL internals.

The direct approach uses `ArealOpenAI`, which extends `AsyncOpenAI` and binds directly
to the inference engine. This approach requires the workflow to inherit
`RolloutWorkflow` and use the engine from `arun_episode`.

```python
from areal.experimental.openai import ArealOpenAI

class MyWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine, data):
        # Create client bound to engine
        client = ArealOpenAI(engine=engine, tokenizer=self.tokenizer)

        # Use like standard OpenAI client
        response = await client.chat.completions.create(
            model="default",
            messages=data["messages"],
        )

        # Set reward and export
        reward = compute_reward(response, data["answer"])
        client.set_last_reward(reward)
        client.apply_reward_discount(turn_discount=0.9)

        return client.export_interactions(style="individual")
```

## Execution Modes

The proxy approach supports two execution modes, configured via `rollout.agent.mode`:

### Inline Mode (Default)

The agent runs in the same process as the rollout worker. AReaL calls the agent's `run`
method directly as an async coroutine, passing `base_url`, `api_key`, and `http_client`
via `extra_kwargs`.

```yaml
rollout:
  agent:
    mode: inline
```

**Characteristics:**

- No serialization overhead
- Direct access to shared HTTP client
- Lower latency
- Requires async code

### Subprocess Mode

The agent runs in a separate process pool (`ProcessPoolExecutor`). AReaL serializes the
agent and data, executes in a subprocess, and deserializes the result.

```yaml
rollout:
  agent:
    mode: subproc
    subproc_max_workers: 4  # Process pool size
```

**Characteristics:**

- Agent must be picklable (serializable)
- `OPENAI_BASE_URL` and `OPENAI_API_KEY` are set as environment variables
- Agent reads `base_url` and `api_key` from environs instead of `extra_kwargs`
- Synchronous code allowed inside `run()` (AReaL wraps with `asyncio.run()`)
- Pickling overhead for agent and data
- Useful for non-async libraries or process isolation

**Subprocess example:**

```python
import os
from openai import OpenAI  # Sync client is OK

class MySyncAgent:
    async def run(self, data, **extra_kwargs):
        # In subproc mode, base_url and api_key come from environment
        client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            api_key="DUMMY",
        )

        response = client.chat.completions.create(
            model="default",
            messages=data["messages"],
        )

        return compute_reward(response, data["answer"])
```

## Architecture

### Proxy Server

When an agent workflow is detected, AReaL spawns proxy workers running FastAPI servers
that implement OpenAI-compatible endpoints.

```
┌─────────────────────────────────────────────────────────────────┐
│                         PPOTrainer                              │
│         (Detects agent workflow, initializes proxies)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RolloutController                            │
│                                                                 │
│  ┌──────────────┐     ┌──────────────┐                          │
│  │   Rollout    │     │    Proxy     │  FastAPI server          │
│  │   Worker     │◄────│    Worker    │  /v1/chat/completions    │
│  │              │     │              │  /v1/responses           │
│  │ SGLang/vLLM  │     │              │  /v1/messages            │
│  └──────────────┘     └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

**Key file:** `areal/experimental/openai/proxy/proxy_rollout_server.py`

### Four-Process Architecture (Proxy)

The proxy mode introduces a proxy server between the agent and the inference engine:

```
│ Controller Process │  │ Rollout Worker (RPC) │  │ Proxy Worker │  │ GPU Process │
│                    │  │                      │  │              │  │             │
│ RolloutController  │  │  Flask HTTP Server   │  │ FastAPI HTTP │  │ SGLang/vLLM │
│        │           │  │        │             │  │    Server    │  │      │      │
│        ▼           │  │   /call endpoint     │  │ OpenAI API   │  │ Inference   │
│ BatchTaskDispatcher│  │        │             │  │ compatible   │  │   Engine    │
│   (bg thread)      │  │        ▼             │  │      │       │  │      │      │
│        │           │  │   Engine Thread      │  │      │       │  │      │      │
│        │           │  │        │             │  │      │       │  │      │      │
│        │    HTTP   │  │        ▼             │  │      │       │  │      │      │
│ submit ├────POST───┼─>│   RemoteInfEngine    │  │      │       │  │      │      │
│ task 1 │           │  │        │             │  │      │       │  │      │      │
│        │           │  │        ▼             │  │      │       │  │      │      │
│ submit │           │  │ OpenAIProxyWorkflow  │  │      │       │  │      │      │
│ task 2 │           │  │        │             │  │      │       │  │      │      │
│        │           │  │  OpenAIProxyClient ──┼──┼──────┤       │  │      │      │
│ submit │           │  │        │             │  │      │       │  │      │      │
│ task 3 │           │  │   agent.run()        │  │      │       │  │      │      │
│        │           │  │        │             │  │      │       │  │      │      │
│        │           │  │        ▼             │  │      │       │  │      │      │
│        │           │  │   OpenAI API call ───┼──┼─>  /chat/ ───┼──┼─> generate  │
│        │           │  │        │             │  │ completions  │  │    tokens   │
│        │           │  │        │             │  │      │       │  │      │      │
│        │           │  │  ChatCompletion <────┼──┼──────<───────┼──┼──────┘      │
│        │           │  │        │             │  │   (cached)   │  │             │
│        │           │  │        │             │  │      │       │  │             │
│        │           │  │        ▼             │  │      │       │  │             │
│        │           │  │     reward           │  │      │       │  │             │
│        │           │  │        │             │  │      │       │  │             │
│        │           │  │   set_reward() ──────┼──┼─>  /rl/      │  │             │
│        │           │  │        │             │  │ set_reward   │  │             │
│        │           │  │        ▼             │  │      │       │  │             │
│        │           │  │     ...              │  │      │       │  │             │
│        │           │  │        │             │  │      │       │  │             │
│        │           │  │        ▼             │  │      │       │  │             │
│        │           │  │    trajectory        │  │      │       │  │             │
│        │           │  │        │             │  │      │       │  │             │
│    collect<────────┼──┼────────┘             │  │      │       │  │             │
│                    │  │                      │  │              │  │             │
└────────────────────┴──┴──────────────────────┴──┴──────────────┴──┴─────────────┘
```

The `OpenAIProxyWorkflow` contains an `OpenAIProxyClient` that manages the session
lifecycle with the proxy server. Key interactions include:

- **chat/completions**: Routes agent's OpenAI API calls to inference engine, caches
  token-level data
- **set_reward**: Assigns rewards to completions for RL training

### Data Flow Detail

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                               Rollout Worker + Proxy Worker                                │
│                                                                                            │
│  ┌─────────────────────┐      ┌──────────────────────────────────────────────────────────┐ │
│  │ OpenAIProxyWorkflow │      │               ProxyRolloutServer (FastAPI)               │ │
│  │                     │      │                                                          │ │
│  │ 1. grant_capacity()─┼─────>│                                                          │ │
│  │                     │      │                                                          │ │
│  │ 2. start_session() ─┼─────>│ → SessionData created                                    │ │
│  │    ← session_id, session_api_key │ │
│  │                     │      │                                                          │ │
│  │ 3. agent.run()      │      │   ┌──────────────────────────────────────────────────┐   │ │
│  │    │                │      │   │                   ArealOpenAI                    │   │ │
│  │    └─> OpenAI call ─┼─────>│   │                                                  │   │ │
│  │                     │      │   │  /chat/completions                               │   │ │
│  │                     │      │   │    → tokenize, engine.agenerate() ───────────────┼───┼─┼──┐
│  │                     │      │   │    → cache in InteractionCache    <──────────────┼───┼─┼──┤
│  │    ChatCompletion  <┼──────┤   │    → return ChatCompletion                       │   │ │  │
│  │                     │      │   │                                                  │   │ │  │
│  │                     │      │   └──────────────────────────────────────────────────┘   │ │  │
│  │                     │      │                                                          │ │  │
│  │ 4. set_reward()    ─┼─────>│ → reward stored in InteractionCache                      │ │  │
│  │                     │      │                                                          │ │  │
│  │ 5. end_session()   ─┼─────>│ → session marked complete                                │ │  │
│  │                     │      │                                                          │ │  │
│  │ 6. export_          │      │                                                          │ │  │
│  │    trajectories()  ─┼─────>│ → apply discount, to_tensor_dict()                       │ │  │
│  │    → tensors       <┼──────┤                                                          │ │  │
│  └─────────────────────┘      └──────────────────────────────────────────────────────────┘ │  │
│                                                                                            │  │
└────────────────────────────────────────────────────────────────────────────────────────────┘  │
                                                                                                │
                                             ┌──────────────────────────────────────────────────┘
                                             │
                                             ▼
                           ┌─────────────────────────────────────────────────────────┐
                           │                  GPU Process (SGLang/vLLM)              │
                           │                                                         │
                           │   Continuous batching, KV cache, tensor parallelism     │
                           └─────────────────────────────────────────────────────────┘
```

### Proxy Endpoints

| Endpoint                    | Auth                             | Purpose                          |
| --------------------------- | -------------------------------- | -------------------------------- |
| `POST /grant_capacity`      | Admin key                        | Reserve slot (staleness control) |
| `POST /rl/start_session`    | Admin key                        | Create unique session ID         |
| `POST /v1/chat/completions` | Session key                      | OpenAI chat completions API      |
| `POST /v1/responses`        | Session key                      | OpenAI responses API             |
| `POST /v1/messages`         | Session key                      | Anthropic Messages API           |
| `POST /rl/set_reward`       | Session key                      | Assign reward to interaction     |
| `POST /rl/end_session`      | Session key                      | Mark session complete            |
| `POST /export_trajectories` | Admin key + `session_id` in body | Export with reward discounting   |

## Session Lifecycle

Each agent execution follows this lifecycle:

```
1. Reserve capacity
   POST /grant_capacity → Staleness control

2. Start session
   POST /rl/start_session → Returns session_id and unique API key

3. Agent execution (multiple LLM calls)
   POST /v1/chat/completions (with session API key in Authorization header)
     → Proxy tokenizes messages
     → Engine generates tokens with logprobs
     → Response stored in InteractionCache
     → ChatCompletion returned to agent

4. Assign rewards
   POST /rl/set_reward (with session API key)
     Body: {"reward": 1.0}                           → Last completion
     Body: {"interaction_id": "...", "reward": 0.5}  → Specific completion

5. End session
   POST /rl/end_session (with session API key)

6. Export trajectories
   POST /export_trajectories (with admin API key, body: {session_id: ..., discount: 0.9})
     → Apply reward backpropagation
     → Return InteractionWithTokenLogpReward objects
     → Clean up session and API key mappings
```

## Token-Level Tracking

### InteractionCache

The `InteractionCache` (extends `OrderedDict`) stores `InteractionWithTokenLogpReward`
objects keyed by completion ID.

**Key file:** `areal/experimental/openai/cache.py`

**Parent-child resolution**: When a new interaction is added, the cache finds its parent
by checking if any existing interaction's messages are a prefix of the new one:

```python
# Parent: [system, user]
# Child:  [system, user, assistant, user]
# → Child's parent is set to Parent
```

### InteractionWithTokenLogpReward

This dataclass stores completion data with token-level information:

```python
@dataclass
class InteractionWithTokenLogpReward:
    model_response: ModelResponse | None  # Token IDs, logprobs from engine
    reward: float | None
    parent: InteractionWithTokenLogpReward | None
    messages: list[dict]                  # Input messages
    output_message_list: list[dict] | None
    completion: ChatCompletion | None     # OpenAI response object
```

**Key file:** `areal/experimental/openai/types.py`

The `to_tensor_dict()` method converts to training format:

```python
{
    "input_ids": torch.tensor([...], dtype=torch.int32),
    "loss_mask": torch.tensor([0]*input_len + [1]*output_len, dtype=torch.int32),
    "logprobs": torch.tensor([0]*input_len + output_logprobs, dtype=torch.float32),
    "versions": torch.tensor([...], dtype=torch.int32),
    "attention_mask": torch.ones(..., dtype=torch.bool),
    "rewards": torch.tensor([reward], dtype=torch.float32),
}
```

## Reward System

### Assignment

Rewards can be assigned in two ways:

1. **Return from `run()` method**:

   - `float`: Applied to last completion
   - `dict[str, float]`: Maps completion IDs to rewards

1. **Explicit API calls** (direct approach):

   ```python
   client.set_last_reward(1.0)
   client.set_reward(completion_id, 0.5)
   ```

### Backpropagation

For multi-turn conversations, rewards propagate backward through the conversation tree
with geometric discounting:

```
# Conversation tree:
A → B → C (leaf, reward=1.0)

# With discount=0.9:
C.reward = 1.0
B.reward = 0 + 1.0 × 0.9 = 0.9
A.reward = 0 + 0.9 × 0.9 = 0.81
```

Processing occurs in reverse topological order (leaves first), ensuring children's
rewards are finalized before propagating to parents.

### Configuration

```python
# Direct approach
client.apply_reward_discount(turn_discount=0.9)
interactions = client.export_interactions(style="individual")

# Proxy approach (via export endpoint)
POST /export_trajectories
Body: {"discount": 0.9, "style": "individual"}
```

## Workflow Resolution

When a workflow is passed to the trainer, AReaL resolves it as follows:

**Key file:** `areal/infra/remote_inf_engine.py` (`_resolve_workflow` method)

```python
def _resolve_workflow(workflow, workflow_kwargs, group_size, proxy_addr):
    # 1. RolloutWorkflow instance → use directly
    # 2. RolloutWorkflow class → instantiate with kwargs
    # 3. String path → import and resolve recursively
    # 4. Has run() method → wrap with OpenAIProxyWorkflow

    if not isinstance(resolved, RolloutWorkflow):
        resolved = OpenAIProxyWorkflow(
            agent=resolved,
            proxy_addr=proxy_addr,
            ...
        )

    # Apply grouping if needed
    if group_size > 1:
        resolved = GroupedRolloutWorkflow(resolved, group_size)

    return resolved
```

## OpenAIProxyWorkflow

The `OpenAIProxyWorkflow` wraps user agents into `RolloutWorkflow`:

**Key file:** `areal/experimental/openai/proxy/workflow.py`

```python
class OpenAIProxyWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine, data):
        # 1. Grant capacity
        await self._grant_capacity(http_session)

        # 2. Create proxy client (manages session)
        proxy_client = OpenAIProxyClient(...)

        async with proxy_client:
            # 3. Run agent with session API key
            rewards = await self._run_agent(proxy_client.session_api_key, data)

            # 4. Assign rewards
            if isinstance(rewards, float):
                await proxy_client.set_last_reward(rewards)
            elif isinstance(rewards, dict):
                for id, reward in rewards.items():
                    await proxy_client.set_reward(id, reward)

        # 5. Export interactions
        return await proxy_client.export_interactions(
            discount=self.discount,
            style=self.export_style,
        )
```

The `_run_agent` method handles both execution modes:

- **Inline**: Calls `agent.run()` directly as a coroutine
- **Subprocess**: Submits to `ProcessPoolExecutor`, sets `OPENAI_BASE_URL` environment
  variable, wraps with `asyncio.run()`

## ArealOpenAI Client

The `ArealOpenAI` class extends `AsyncOpenAI` for direct engine integration:

**Key file:** `areal/experimental/openai/client.py`

### Key Methods

| Method                                 | Description                         |
| -------------------------------------- | ----------------------------------- |
| `chat.completions.create(...)`         | OpenAI-compatible chat API          |
| `responses.create(...)`                | OpenAI responses API                |
| `set_reward(id, reward)`               | Set reward for specific interaction |
| `set_last_reward(reward)`              | Set reward for last interaction     |
| `apply_reward_discount(turn_discount)` | Apply backward reward discounting   |
| `export_interactions(style)`           | Export for training                 |

### Export Styles

| Style        | Description                                                                                                          |
| ------------ | -------------------------------------------------------------------------------------------------------------------- |
| `individual` | Returns all interactions as separate entries. Trajectories may share prefixes.                                       |
| `concat`     | Builds conversation tree, returns only leaf nodes. Only valid for linear conversations with matched token sequences. |

## Public API

```python
from areal.experimental.openai import (
    ArealOpenAI,                     # Direct approach client
    InteractionWithTokenLogpReward,  # Token-level data structure
    OpenAIProxyClient,               # HTTP client for proxy sessions
    OpenAIProxyWorkflow,             # Workflow wrapper
)
```

## Training with Agent Trajectories

A complete agentic episode may contain multiple LLM interactions (turns). For training,
these are treated as independent input-output-reward tuples:

```
Turn 1: [system, user]                         → output_1 → reward_1 (discounted)
Turn 2: [system, user, asst, user]             → output_2 → reward_2 (discounted)
Turn 3: [system, user, asst, user, asst, user] → output_3 → reward_3 (final)
```

Each tuple includes full token-level data for policy gradient computation: input token
IDs, output token IDs, and log probabilities. The discounted rewards ensure the RL
objective correctly credits earlier actions for final outcomes.

### Token Consistency Guarantee

Because AReaL stores the actual tokens used during inference (not re-tokenized text),
there is no risk of tokenization mismatch between rollout and training. The tokens sent
to the inference engine are exactly the tokens used for gradient computation.

### Efficient Training with Tree Attention

Multi-turn trajectories often share long token prefixes, which can slow down training
due to redundant computation. AReaL addresses this with prefix-shared tree attention,
which computes attention over shared prefixes only once.

## See Also

- [RolloutWorkflow Reference](./rollout_workflow.md) - Core workflow abstraction
- [Agentic RL Guide](../tutorial/agentic_rl.md) - Practical training guide
- [Workflow Best Practices](../best_practices/workflow.md) - Implementation tips

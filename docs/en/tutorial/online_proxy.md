# Online RL Training

This guide explains how to train language models using the online mode, where the user
first launches an AReaL RL service that exposes a proxy gateway, and external
applications (agent runtimes, human evaluators, or any OpenAI-compatible client)
interact with the model through this gateway. Each interaction is automatically
collected as RL training data.

**Disclaimer:** This API is experimental and subject to change.

## Overview

AReaL supports three execution modes for agent workflows:

| Mode         | Description                                        | Use Case                          |
| ------------ | -------------------------------------------------- | --------------------------------- |
| `inline`     | Agent runs in-process with the rollout worker      | Most agent frameworks             |
| `subproc`    | Agent runs in a subprocess pool                    | Non-async or isolation-heavy code |
| **`online`** | External users drive the interaction via HTTP APIs | Human feedback, external runtimes |

This guide focuses on **online mode**, which is unique because the agent code lives
_outside_ of AReaL. AReaL exposes an OpenAI-compatible HTTP API, and any application
that speaks the chat completions protocol can connect to it.

For the offline training guide, see [agentic RL guide](./agentic_rl.md).

## Architecture

```
                          External Application
                         (ZeroClaw, scripts, etc.)
                                  |
                      POST /chat/completions
                      POST /rl/set_reward
                                  |
                                  v
                      +-------------------+
                      |  Proxy Gateway    |  (FastAPI, stateless router)
                      |  - Session mgmt   |
                      |  - Key auth       |
                      |  - Load balancing |
                      +-------------------+
                         /        |        \
                        v         v         v
                  +---------+ +---------+ +---------+
                  | Proxy   | | Proxy   | | Proxy   |
                  | Worker  | | Worker  | | Worker  |  (one per rollout worker)
                  +---------+ +---------+ +---------+
                      |           |           |
                      v           v           v
                  +---------+ +---------+ +---------+
                  | SGLang/ | | SGLang/ | | SGLang/ |
                  | vLLM    | | vLLM    | | vLLM    |  (inference servers)
                  +---------+ +---------+ +---------+
                                  |
                      Token-level data collected
                                  |
                                  v
                      +-------------------+
                      |   RL Trainer      |
                      |   (PPOTrainer)    |
                      +-------------------+
```

**Key components:**

- **Proxy Gateway**: A lightweight FastAPI server that routes requests from external
  applications to backend proxy workers. It manages session lifecycle, authentication,
  and load balancing.
- **Proxy Workers**: Backend servers colocated with rollout workers. Each worker manages
  sessions, records token-level data (token IDs, log probabilities), and exports
  trajectories for training.
- **Inference Servers**: SGLang or vLLM servers that perform the actual LLM inference.

## Quick Start

### Step 1: Configure Online Mode

Set `rollout.agent.mode` to `online` in your config YAML:

```yaml
# config.yaml
rollout:
  agent:
    mode: online
    admin_api_key: "my-secret-admin-key"  # Protect management endpoints
    session_timeout_seconds: 3600          # Session timeout (default: 1 hour)
```

### Step 2: Start the RL Service

```bash
python3 examples/openclaw/train.py --config examples/openclaw/config.yaml \
    experiment_name=my-exp trial_name=trial-0 \
    rollout.backend=sglang:d1 actor.backend=fsdp:d1 \
    actor.path=Qwen/Qwen3-0.6B \
    scheduler.type=local \
    rollout.agent.admin_api_key=my-secret-admin-key
```

After initialization, AReaL prints the gateway address:

```
(AReaL) RLTrainer INFO: Proxy gateway available at http://x.x.x.x:8090
```

### Step 3: Start a Session

Use the provided helper script or any HTTP client:

```bash
curl -X POST http://<gateway>/rl/start_session \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret-admin-key" \
  -d '{"task_id": "demo-task-0"}'
```

You should see the current session ID and the API key for this agent session in the
output.

**Why a unique API key for each agent session?** Since there may be many concurrent
agent applications running, and they invoke the same endpoint (e.g.,
"/chat/completions") in the URL, we need a mechanism to differentiate the trajectories
from different agents. Therefore, we allocate unique API keys for each agent session or
trajectory, and they have one-to-one relationship. In this way, we can track the
interactions within the same trajectory and set rewards as well.

### Step 4: Interact with the Model

Use any OpenAI-compatible client. For example, with `curl`:

```bash
curl http://<gateway>/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-sess-xxxxxxxxxxxx" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 12 * 15 + 3?"}],
    "temperature": 0.7
  }'
```

Or any evaluation scripts with the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<gateway>",
    api_key="sk-sess-xxxxxxxxxxxx",
)

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 12 * 15 + 3?"}],
)
print(response.choices[0].message.content)
```

### Step 5: Assign a Reward and End the Session

After the interaction, assign a reward to provide the RL training signal:

```bash
curl http://<gateway>/rl/set_reward \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-sess-xxxxxxxxxxxx" \
  -d '{"reward": 1.0}'
```

You can also use the completion ID during agent rollout to set rewards for intermediate
steps.

Then, finish the session with:

```bash
curl http://<gateway>/rl/end_session \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-sess-xxxxxxxxxxxx" \
  -d '{}'
```

### Step 6: Batched Sampling

Integrate Steps 3 through 5 into a single bash script, and then run it concurrently with
tools like `sbatch`. **You must call `/rl/start_session` again to obtain a new API key
for each agent session.**

After enough data has been accumulated in AReaL's buffer, AReaL will automatically enter
the training stage.

## FAQ

> Q: When will the updated model be loaded for inference?

The model will be loaded after every training step. In other words, the model used for
inference is always the latest. For model saving and checkpointing, see
[CLI reference](../cli_reference.md)

> Q: How to control the submission rate of the agent script? Will the RL server be
> overloaded?

AReaL has its internal rate limit, referred to as **staleness control**. If too many
concurrent requests have been submitted, the gateway will return 429 to the client. See
[async RL guide](../algorithms/async.md) for details about staleness control.

> Q: Can I use this approach to train OpenClaw?

The approach in this documentation is different from training a personalized agent,
because:

- OpenClaw assumes single-threaded interaction with the user, meaning that the user
  cannot open many concurrent sessions that may mutually interfere
- OpenClaw requires one-time setup with a fixed URL and API key

The core usage difference is that the OpenClaw example uses a **fixed** API key all over
the interaction. By calling `start_session` multiple times, the old session is
automatically ended, its trajectory exported for training, and a new session starts with
the same API key. No reconfiguration of your application is needed between episodes.

For details of training the OpenClaw agent, see
[OpenClaw example](../../../examples/openclaw/README.md).

## Authentication

Online mode uses a two-tier authentication system:

| Auth Type           | Token                         | Used For                                        |
| ------------------- | ----------------------------- | ----------------------------------------------- |
| **Admin API key**   | `rollout.agent.admin_api_key` | `start_session`, `export_trajectories`          |
| **Session API key** | Issued by `start_session`     | `chat/completions`, `set_reward`, `end_session` |

- The **admin API key** is configured in the YAML and protects management endpoints.
- The **session API key** is unique per session and scoped to that session's
  interactions.

## API Reference

All endpoints are served by the proxy gateway.

### Management Endpoints (Admin Auth)

#### `POST /rl/start_session`

Start a new session or refresh an existing one.

**Request body:**

```json
{
  "task_id": "my-task-0",
  "api_key": null
}
```

Pass `api_key` from a previous session to refresh. Omit or set `null` for a new session.

**Response:**

```json
{
  "session_id": "my-task-0",
  "api_key": "sk-sess-xxxxxxxxxxxx"
}
```

#### `GET /health`

Health check. Returns the number of backend workers.

### Session Endpoints (Session Auth)

#### `POST /chat/completions`

OpenAI-compatible chat completions endpoint. Tokens and log probabilities are
automatically recorded.

#### `POST /responses`

OpenAI Responses API endpoint (alternative to chat completions).

#### `POST /v1/messages`

Anthropic Messages API endpoint for Claude-compatible clients.

#### `POST /rl/set_reward`

Assign a reward to an interaction.

**Request body:**

```json
{
  "reward": 1.0,
  "interaction_id": null
}
```

If `interaction_id` is null, the reward is assigned to the last interaction.

#### `POST /rl/end_session`

Explicitly end a session and export its trajectory. Used in the **batched sampling**
pattern where each sample has its own API key. Not needed when using session refresh.

## Error Handling

| HTTP Code | Meaning                            | Action                                     |
| --------- | ---------------------------------- | ------------------------------------------ |
| 200       | Success                            | -                                          |
| 401       | Missing or invalid authentication  | Check your API key                         |
| 409       | API key already bound to a session | End existing session first, or use refresh |
| 429       | No capacity available              | Retry after a short delay                  |
| 502       | Backend worker unreachable         | Check that the RL service is running       |

For HTTP 429 during refresh, the training pipeline may not have cycled yet. Retry after
a few seconds (default timeout is 120 seconds).

## How Training Works

Training runs **asynchronously** under the hood:

1. External applications interact with the model through the gateway
1. Each session's interactions are recorded with token-level data
1. When a session ends (via refresh or explicit end), its trajectory is exported
1. Once enough trajectories are collected (controlled by `train_dataset.batch_size`),
   AReaL performs a training step
1. Updated model weights are transparently served to subsequent sessions

The model improves silently as you collect more episodes. For details on asynchronous
training and staleness control, see the [Asynchronous RL Guide](../algorithms/async.md).

## Configuration Reference

All online mode settings live under `rollout.agent`:

```yaml
rollout:
  agent:
    mode: online                    # Required: set to "online"
    admin_api_key: "areal-admin-key"  # Admin key for management endpoints
    session_timeout_seconds: 3600   # Session timeout in seconds
    turn_discount: 1.0              # Reward discount for multi-turn conversations
    export_style: individual        # "individual" or "concat"
```

| Field                     | Default           | Description                               |
| ------------------------- | ----------------- | ----------------------------------------- |
| `mode`                    | `inline`          | Must be `online` for external access      |
| `admin_api_key`           | `areal-admin-key` | Admin API key (change in production!)     |
| `session_timeout_seconds` | `3600`            | Auto-cleanup stale sessions after this    |
| `turn_discount`           | `1.0`             | Geometric discount for multi-turn rewards |
| `export_style`            | `individual`      | How to export interactions for training   |

## Limitations

- **Scheduler compatibility**: Online mode requires `local` or `slurm` schedulers. The
  `ray` scheduler is not supported.
- **Single-controller mode**: Online mode only works in single-controller mode
  (`scheduler.type=local` or `scheduler.type=slurm`).

## See Also

- [OpenClaw Example](https://github.com/areal-project/AReaL/tree/main/examples/openclaw)
  \- Complete end-to-end example with ZeroClaw
- [Agentic RL Tutorial](agentic_rl.md) - Agent framework integration (inline/subproc
  modes)
- [Custom Agent Workflows](../customization/agent.md) - Creating custom agent workflows
- [Agent Workflow Reference](../reference/agent_workflow.md) - Internal architecture
  details

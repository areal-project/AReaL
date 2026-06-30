# Agent Service — Hermes

## Overview

This example runs the **Hermes agent**
([Nous Research `hermes-agent`](https://github.com/nousresearch/hermes-agent)) inside
AReaL's Agent Service. Hermes ships as a Python library, so the Worker instantiates
**one in-process Hermes `AIAgent` per session** directly inside the Worker process —
**you never launch Hermes yourself**. The Agent Service is started with the
`areal agent run` CLI; you then interact and (optionally) train through a small set of
single-purpose scripts.

```
Client → Gateway (HTTP) → Router → DataProxy (session state) → Worker (Hermes AIAgent)
```

Each user message becomes one turn of the Hermes conversation; the per-session `AIAgent`
drives its configured OpenAI-compatible upstream LLM internally. AReaL's DataProxy owns
the conversation history and replays it into every turn, so the `AIAgent` is kept
**stateless across turns** (`skip_memory=True`, `skip_context_files=True`,
`session_db=None`). A consequence of this design: Hermes' own `memory` tool reports
"Memory is not available" — that is expected, since cross-turn context comes from the
DataProxy's replayed history, not from Hermes' persistence.

This directory also contains the **RL training flow** (`train.py` + `config.yaml`). For
training, run `train.py` to bring up AReaL's trainer plus an inference gateway, then
point the Hermes turns at that gateway (via `hermes_loop.py`'s inference-routing flags)
so every LLM call is captured as a training trajectory.

> **See also**
>
> - [Agentic RL tutorial](../../docs/en/tutorial/agentic_rl.md) — background on how
>   AReaL trains agents
> - [Custom agent workflows](../../docs/en/customization/agent.md) — how to integrate
>   your own agent framework
> - [Agent workflow reference](../../docs/en/reference/agent_workflow.md) — internal
>   architecture details

**Disclaimer**: RL-finetuned models may exhibit unexpected behaviors. Please ensure
strict permission rules and an isolated execution environment for your agent runtime.

## How it fits together

```
┌──────────────────────────────────────┐   LLM calls (self-evolution)   ┌────────────────────────┐
│  Agent Service                        │ ─────────────────────────────▶ │  AReaL inference gateway│
│  (areal agent run)                    │   inf_base_url = http://<gw>   │  (started by train.py) │
│  Gateway/Router/DataProxy/Worker      │   session key  = sk-sess-*     │                        │
│  + in-process Hermes AIAgent          │ ◀───────────────────────────── │  records tokens +      │
└──────────────────────────────────────┘   model output                 │  logprobs → RL         │
        ▲                                                                └────────────────────────┘
        │ hermes_loop.py                                                           │
        │ (the interactive "You:" prompt)                       set_reward.py (score the trajectory)
```

One **episode** = the turns collected under a single per-session `sk-sess-*` key (minted
by `start_session.py`). You score it with `set_reward.py`, then start the next episode.

## Prerequisites

### 1. GPUs (for RL training only)

A GPU machine with at least **2 NVIDIA GPUs** (compute capability 8.0 or higher, i.e.
Ampere / Hopper). Not required if you only run the agent against an env upstream LLM
(plain chat, no training).

### 2. Install Hermes into AReaL's venv

Hermes' top-level module is `run_agent`. The Worker process is forked with
`sys.executable` (the interpreter you launch the controller with), so **`areal` and
`run_agent` must be importable from the same venv**:

```bash
uv pip install hermes-agent
python -c "import areal; from run_agent import AIAgent; print('co-import OK')"
```

A bare `hermes-agent` install is moderate, **not** heavy: every large optional
integration (`anthropic`, `slack`, `matrix`, `modal`, browser/messaging, …) sits behind
a `pip install hermes-agent[extra]` marker and is not pulled in. No torch/CUDA/ML
packages are added.

> **Gotchas**
>
> - **Run with the project `.venv` python directly** (`python` / `.venv/bin/python`),
>   **not `uv run`** — `uv run` re-syncs the env to `uv.lock` and resets the shared
>   packages (`openai`, `pydantic`, `rich`, …) to AReaL's pinned versions.
> - **Do not run `uv sync`** while you need Hermes — it removes `hermes-agent` (it is
>   not in `uv.lock`). Re-add with `uv pip install hermes-agent`.

## Quick start — plain chat (no GPU, no training)

This is the fastest way to see Hermes working: start the service against an
OpenAI-compatible upstream and chat. No GPU and no training pipeline involved.

### 1. Configure the upstream LLM

The upstream the Hermes agent routes to is set via environment variables:

| Variable                   | Default    | Description                           |
| -------------------------- | ---------- | ------------------------------------- |
| `HERMES_UPSTREAM_BASE_URL` | (required) | OpenAI-compatible base URL (`.../v1`) |
| `HERMES_UPSTREAM_API_KEY`  | (required) | API key for the upstream              |
| `HERMES_UPSTREAM_MODEL`    | `default`  | Upstream model id                     |
| `HERMES_MAX_TURNS`         | `10`       | Max agentic iterations per turn       |
| `HERMES_ENABLED_TOOLSETS`  | (all)      | Comma-separated toolset allowlist     |
| `HERMES_DISABLED_TOOLSETS` | (none)     | Comma-separated toolset denylist      |

```bash
export HERMES_UPSTREAM_BASE_URL="https://your-llm/v1"
export HERMES_UPSTREAM_API_KEY="sk-..."
export HERMES_UPSTREAM_MODEL="your-model"
```

### 2. Start the Agent Service

```bash
areal agent run \
    --service default \
    --agent examples.hermes.hermes.HermesAgent \
    --num-pairs 1 \
    --admin-api-key sk-xxx
```

> If the `areal` console script is not on your `PATH`, use
> `python -m areal.v2.cli.main agent run ...` instead.

The CLI boots one Worker+DataProxy pair behind a Router and Gateway, then prints the
gateway address and returns (the stack runs in the background). Look for:

```
service=default gateway=http://127.0.0.1:PORT
```

That `http://127.0.0.1:PORT` is your `<agent-gateway>`. You can re-query it any time:

```bash
areal agent status --service default
```

### 3. Chat

```bash
python examples/hermes/hermes_loop.py http://<agent-gateway> --admin-api-key sk-xxx
```

Type at the `You:` prompt; `quit` exits. To verify multi-turn history replay, send a
fact then a recall — a correct second answer proves the DataProxy replayed turn-1 into
turn-2 (the `AIAgent` is stateless across turns):

```
You: Remember my favorite number is 4242. Acknowledge in one sentence.
You: What is my favorite number? Answer with just the number.
```

Run it non-interactively by piping the turns in:

```bash
printf 'Remember my favorite number is 4242. Acknowledge in one sentence.\nWhat is my favorite number? Answer with just the number.\nquit\n' \
  | python examples/hermes/hermes_loop.py http://<agent-gateway> --admin-api-key sk-xxx
```

### 4. Stop the service

```bash
areal agent stop --service default
```

## RL training (self-evolution)

RL training requires *(input, output, reward)* tuples. In self-evolution mode every LLM
call the in-process Hermes agent makes flows through AReaL's inference gateway, which
records tokens and log-probabilities; you then assign a scalar reward per episode.

The five steps below are run from the repo root.

### Step 1 — Start the training service (embeds the inference gateway)

The training defaults already live in `config.yaml` (v2 controllers, 1 node × 2 GPUs,
`batch_size=1`, admin keys). The command line still wins over the file, so you can
override any of them; the explicit form below documents what the defaults are:

```bash
uv run python3 examples/hermes/train.py \
    --config examples/hermes/config.yaml \
    actor.backend=fsdp:d1 \
    rollout.backend=sglang:d1 \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node=2 \
    actor.admin_api_key=gsm8k-123 \
    rollout.admin_api_key=sk-xxx \
    actor._version=v2 \
    rollout._version=v2
```

> Because these are now defaults in `config.yaml`, the minimal command is just:
>
> ```bash
> uv run python3 examples/hermes/train.py --config examples/hermes/config.yaml
> ```

Find this line in the logs and note the address — it is your `<inf-gateway>` below:

```
Proxy gateway available at http://X.X.X.X:PORT
```

> **Key wiring**
>
> `rollout.admin_api_key` (`sk-xxx`) is the **inference gateway** admin key — pass the
> same value to `start_session.py --admin-key` and `set_reward.py` (Steps 3 and 5).
> `actor.admin_api_key` (`gsm8k-123`) is the trainer/actor admin key and is not used by
> the interaction scripts. See the [CLI reference](../../docs/en/cli_reference.md) for
> every field and the [allocation mode reference](../../docs/en/reference/alloc_mode.md)
> for GPU layout.

### Step 2 — Start the Hermes Agent Service

Same command as the quick start. The agent's env upstream is optional here — when
self-evolution fields are supplied per turn (Step 4), the inference gateway upstream
takes over.

```bash
areal agent run \
    --service default \
    --agent examples.hermes.hermes.HermesAgent \
    --num-pairs 1 \
    --admin-api-key sk-xxx
```

Note the printed `<agent-gateway>` address.

### Step 3 — Start a session on the inference gateway

```bash
python examples/hermes/start_session.py http://<inf-gateway> --admin-key sk-xxx
```

Copy the printed `sk-sess-*` key — you forward it to the agent (Step 4) and use it to
score the episode (Step 5). To reuse the same key for the next episode (auto-ends the
previous one, exports its trajectory):

```bash
python examples/hermes/start_session.py http://<inf-gateway> --admin-key sk-xxx --api-key sk-sess-xxxx
```

### Step 4 — Interact (produces a trajectory)

Chat as in the quick start, but forward the inference-routing flags so the agent's LLM
calls flow through the inference gateway under your session key and get captured. You
**must** actually interact, otherwise the episode has no data.

```bash
python examples/hermes/hermes_loop.py http://<agent-gateway> \
    --admin-api-key sk-xxx \
    --inf-base-url http://<inf-gateway> \
    --session-api-key sk-sess-xxxx
```

### Step 5 — Score the episode

```bash
python examples/hermes/set_reward.py http://<inf-gateway> \
    --api-key sk-sess-xxxx --reward 1.0
```

Keep the reward in **\[-1, 1\]** for training stability.

### How training works

Training runs **asynchronously** under the hood. Once enough trajectories have been
collected (controlled by `train_dataset.batch_size` in the config — `1` by default here,
so every scored episode triggers a step), AReaL automatically performs a training step
and updates the model weights. The updated weights are transparently served to
subsequent sessions — the agent does not need to restart or reload. For details on
asynchronous training and staleness control, see our
[code walkthrough](../../docs/en/tutorial/gsm8k_grpo.md) and
[paper](https://arxiv.org/abs/2505.24298).

## Send requests directly

The interactive loop drives the **Responses** API (`/v1/responses`):

```bash
curl -X POST http://<agent-gateway>/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xxx" \
  -d '{
    "input": [{"type": "message", "content": "Explain RLHF in simple terms"}],
    "model": "hermes-agent",
    "user": "my-session"
  }'
```

The admin key is the `--admin-api-key` you passed to `areal agent run`.

## Files

| File               | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| `hermes.py`        | `HermesAgent` — in-process per-session Hermes `AIAgent` runnable |
| `hermes_loop.py`   | Standalone interactive `You:` prompt against the agent gateway   |
| `start_session.py` | Mint a per-session `sk-sess-*` key on the inference gateway      |
| `set_reward.py`    | Assign a scalar reward to a session's trajectory                 |
| `train.py`         | RL trainer entry point (embeds the inference gateway)            |
| `config.yaml`      | Training configuration (v2 controllers, 2-GPU defaults)          |
| `_fmt.py`          | Shared CLI formatting helpers                                    |

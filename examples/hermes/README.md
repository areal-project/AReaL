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

## Quick start — plain chat

The five steps below are run from the repo root.

### Step 1 — Start the training service (embeds the inference gateway)

`config.yaml` holds the defaults (v2 controllers, 1 node × 2 GPUs, `batch_size=1`, admin
keys); CLI flags override it. The explicit form below just documents those defaults:

> **Why reward/advantage normalization is disabled here**
>
> Online mode currently trains on one independently rewarded trajectory per group.
> Centering that singleton group reward subtracts the reward from itself, while
> centering the flat token advantages from a one-trajectory batch likewise makes every
> advantage zero. Either operation erases the task-conditioned learning signal. Using
> GRPO-style group centering instead requires at least two trajectories from the same
> task and a workflow that supports grouped rollouts/session grouping and preserves
> their shared group identity through training. Changing `n_samples` alone is
> insufficient; the current online one-sample path does not support this directly.

```bash
uv run python3 examples/hermes/train.py \
    --config examples/hermes/config.yaml \
    actor.path=/path/to/your_model \
    actor.admin_api_key=sk-123456 \
    rollout.admin_api_key=sk-123456
```

Note this address from the logs — it is your `<inf-gateway>` below:

```
Proxy gateway available at http://X.X.X.X:PORT
```

> **Key wiring**
>
> `rollout.admin_api_key` is the **inference gateway** admin key — reuse it for
> `start_session.py --admin-key` and `set_reward.py` (Steps 3 and 5).
> `actor.admin_api_key` is the trainer/actor key, unused by the interaction scripts. See
> the [CLI reference](../../docs/en/cli_reference.md) and
> [allocation mode reference](../../docs/en/reference/alloc_mode.md) for fields and GPU
> layout.

### Step 2 — Start the Hermes Agent Service

Same command as the quick start. The agent's env upstream is optional here — once
self-evolution fields are supplied per turn (Step 4), the inference gateway upstream
takes over.

```bash
areal agent run \
    --service default \
    --agent examples.hermes.hermes.HermesAgent \
    --num-pairs 1 \
    --admin-api-key sk-123456
```

Note the printed `<agent-gateway>` address.

### Step 3 — Start a session on the inference gateway

Copy the printed `sk-sess-*` key — forward it to the agent (Step 4) and score the
episode with it (Step 5). To reuse the key for the next episode (auto-ends and exports
the previous one):

```bash
python examples/hermes/start_session.py http://<inf-gateway> --admin-key sk-123456
```

These are **your own upstream LLM credentials** (the agent's fallback chat backend),
**not** the `sk-sess-*` key returned above — fill in your provider's values:

```bash
export HERMES_UPSTREAM_BASE_URL="https://your-llm/v1"
export HERMES_UPSTREAM_API_KEY="your-upstream-api-key"
export HERMES_UPSTREAM_MODEL="your-model"
```

### Step 4 — Interact (produces a trajectory)

Forward the inference-routing flags so the agent's LLM calls flow through the inference
gateway under your session key and get captured. You **must** actually interact, or the
episode has no data. `<your session-api-key>` is the `sk-sess-*` key returned by
`start_session.py` in Step 3.

```bash
python examples/hermes/hermes_loop.py http://<agent-gateway> \
    --admin-api-key sk-123456 \
    --inf-base-url http://<inf-gateway> \
    --session-api-key <your session-api-key>
```

### Step 5 — Score the episode

Use the same `sk-sess-*` key from Step 3 as `--api-key`:

```bash
python examples/hermes/set_reward.py http://<inf-gateway> \
    --api-key <your session-api-key> --reward 1.0
```

Keep the reward in **\[-1, 1\]** for training stability.

## Files

| File               | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| `hermes.py`        | `HermesAgent` — in-process per-session Hermes `AIAgent` runnable |
| `hermes_loop.py`   | Standalone interactive `You:` prompt against the agent gateway   |
| `start_session.py` | Mint a per-session `sk-sess-*` key on the inference gateway      |
| `set_reward.py`    | Assign a scalar reward to a session's trajectory                 |
| `train.py`         | RL trainer entry point (embeds the inference gateway)            |
| `config.yaml`      | Training configuration (v2 controllers, 2-GPU defaults)          |

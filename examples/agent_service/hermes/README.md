# Agent Service — Hermes

## Overview

This example runs the **Hermes agent**
([Nous Research `hermes-agent`](https://github.com/nousresearch/hermes-agent)) inside
AReaL's Agent Service. Hermes ships as a Python library, so the Worker instantiates
**one in-process Hermes `AIAgent` per session** directly inside the Worker process —
**you never launch Hermes yourself**. The single entry point is `run_agent_service.py`,
which boots the whole service (Gateway → Router → DataProxy → Worker) and drops into an
interactive prompt.

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
training, run the Agent Service in **self-evolution** mode: AReaL's trainer exposes an
inference gateway, and `run_agent_service.py --use-areal-inference` routes the
in-process Hermes agent's LLM calls through that gateway so every call is captured as a
training trajectory.

> **See also**
>
> - [Agentic RL tutorial](../../../docs/en/tutorial/agentic_rl.md) — background on how
>   AReaL trains agents
> - [Custom agent workflows](../../../docs/en/customization/agent.md) — how to integrate
>   your own agent framework
> - [Agent workflow reference](../../../docs/en/reference/agent_workflow.md) — internal
>   architecture details

**Disclaimer**: RL-finetuned models may exhibit unexpected behaviors. Please ensure
strict permission rules and an isolated execution environment for your agent runtime.

## How it fits together

```
┌──────────────────────────────────────┐   LLM calls (self-evolution)   ┌────────────────────────┐
│  run_agent_service.py                 │ ─────────────────────────────▶ │  AReaL inference gateway│
│  Agent Service (Gateway/Router/       │   inf_base_url = http://<gw>   │  (started by train.py) │
│  DataProxy/Worker) + in-process       │   session key  = sk-sess-*     │                        │
│  Hermes AIAgent (per session)         │ ◀───────────────────────────── │  records tokens +      │
└──────────────────────────────────────┘   model output                 │  logprobs → RL         │
        ▲                                                                └────────────────────────┘
        │ you chat in the                                                          │
        │ interactive "You:" prompt                              set_reward (score the trajectory)
```

One **episode** = the turns collected under a single per-session `sk-sess-*` key. You
score it with `set_reward.py`, then start the next episode.

## Prerequisites

### 1. GPUs (for RL training only)

A GPU machine with at least **2 NVIDIA GPUs** (compute capability 8.0 or higher, i.e.
Ampere / Hopper). Not required if you only run the agent against an env upstream LLM.

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

## Launch the Hermes Agent Service

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

The env upstream is the fallback when **not** self-evolving; with
`--use-areal-inference` the per-session inference upstream takes over and the env
upstream becomes optional (see [RL training](#rl-training-self-evolution) below).

### 2. Start the service

```bash
python examples/agent_service/hermes/run_agent_service.py
```

The launcher boots one Worker+DataProxy pair behind a Gateway, prints the Router/Gateway
addresses and a random admin key, waits for `All services ready.`, then drops into a
`You:` prompt. Type `quit` to exit; all child processes are cleaned up automatically.

## RL training (self-evolution)

RL training requires *(input, output, reward)* tuples. In self-evolution mode every LLM
call the in-process Hermes agent makes flows through AReaL's inference gateway, which
records tokens and log-probabilities; you then assign a scalar reward per episode.

### Step 1 — Start the training service (embeds the inference gateway)

```bash
uv run python3 examples/agent_service/hermes/train.py \
    --config examples/agent_service/hermes/config.yaml \
    experiment_name=my-exp trial_name=trial-0 \
    rollout.backend=sglang:d1 actor.backend=fsdp:d1 \
    actor.path=Qwen/Qwen3-0.6B scheduler.type=local \
    rollout.agent.admin_api_key=sk-test123456
```

Find this line in the logs and note the address — it is your `<gateway>` below:

```
Proxy gateway available at http://X.X.X.X:PORT
```

> **Configuration**
>
> You can modify `examples/agent_service/hermes/config.yaml` to suit your setup.
> Command-line arguments override values in the YAML file, and all options are parsed
> into the dataclasses defined in `areal/api/cli_args.py`. See the
> [CLI reference](../../../docs/en/cli_reference.md) for a full description of each
> field and the [allocation mode reference](../../../docs/en/reference/alloc_mode.md)
> for GPU layout options.

### Step 2 — Launch the Hermes Agent Service against the gateway

```bash
python examples/agent_service/hermes/run_agent_service.py \
    --use-areal-inference --inf-base-url http://<gateway> \
    --inf-admin-key sk-test123456
```

The launcher mints a per-session `sk-sess-*` key on the inference gateway and prints it:

```
Session API key (use with set_reward.py): sk-sess-xxxxxxxxxxxx
```

Copy that key — you need it to score the episode in Step 4. Each turn forwards the
`inf_base_url` / `inf_model` / `session_api_key` fields to the agent, so its LLM calls
flow through the inference service under that key and the trajectory is captured.

### Step 3 — Interact (produces a trajectory)

Chat in the `You:` prompt. You **must** actually interact, otherwise the episode has no
data.

### Step 4 — Score the episode

```bash
python examples/agent_service/hermes/set_reward.py http://<gateway> \
    --api-key sk-sess-xxxxxxxxxxxx --reward 1.0
```

Keep the reward in **\[-1, 1\]** for training stability.

### How training works

Training runs **asynchronously** under the hood. Once enough trajectories have been
collected (controlled by `train_dataset.batch_size` in the config), AReaL automatically
performs a training step and updates the model weights. The updated weights are
transparently served to subsequent sessions — the agent does not need to restart or
reload. For details on asynchronous training and staleness control, see our
[code walkthrough](../../../docs/en/tutorial/gsm8k_grpo.md) and
[paper](https://arxiv.org/abs/2505.24298).

## Verify multi-turn history replay

Send two messages in a row — a fact, then a recall:

```
You: Remember my favorite number is 4242. Acknowledge in one sentence.
You: What is my favorite number? Answer with just the number.
```

A correct second answer (`4242`) proves the DataProxy replayed turn-1 history into
turn-2 (the `AIAgent` itself is stateless across turns). To run it non-interactively,
pipe the turns in:

```bash
printf 'Remember my favorite number is 4242. Acknowledge in one sentence.\nWhat is my favorite number? Answer with just the number.\nquit\n' \
  | python examples/agent_service/hermes/run_agent_service.py
```

## Send requests directly

The example drives the **Responses** API (`/v1/responses`):

```bash
curl -X POST http://localhost:<gateway-port>/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin-key>" \
  -d '{
    "input": [{"type": "message", "content": "Explain RLHF in simple terms"}],
    "model": "hermes-agent",
    "user": "my-session"
  }'
```

The admin key is printed on startup (or pass `--admin-api-key`).

## Connectivity self-checks (no agent)

Verify the inference gateway and the session lifecycle **without** the agent. These
scripts talk to the gateway's control plane directly:

```bash
# Full lifecycle: start_session → chat/completions → set_reward → refresh
python examples/agent_service/hermes/demo_lifecycle.py http://<gateway> --admin-key sk-test123456

# Just mint / refresh a session key
python examples/agent_service/hermes/start_session.py http://<gateway> --admin-key sk-test123456
```

`start_session.py` mints a session key (or, with `--api-key <key>`, refreshes an
existing one: end old session → export trajectory → start a fresh session bound to the
same key). `demo_lifecycle.py` exercises the whole control plane with a built-in prompt.

## Files

| File                   | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| `hermes.py`            | `HermesAgent` — in-process per-session Hermes `AIAgent` runnable |
| `run_agent_service.py` | Controller-based launcher + interactive conversation             |
| `train.py`             | RL trainer entry point (embeds the inference gateway)            |
| `config.yaml`          | Training configuration                                           |
| `set_reward.py`        | Assign a scalar reward to a session's trajectory                 |
| `start_session.py`     | Mint / refresh a session key on the inference gateway            |
| `demo_lifecycle.py`    | Control-plane connectivity self-check (no agent)                 |
| `_fmt.py`              | Shared CLI formatting helpers                                    |

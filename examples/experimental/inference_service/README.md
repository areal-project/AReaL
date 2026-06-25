# AReaL Inference Service Examples

This directory contains examples and benchmarks for the AReaL Inference Service
(`RolloutControllerV2`) — an experimental rollout backend that exposes an
OpenAI-compatible proxy gateway so any external agent runtime can submit chat requests
and receive RL training data.

## Directory Structure

```
inference_service/
├── openclaw_tau2/           OpenClaw + TAU²-bench agent environment
│   ├── openclaw/            OpenClaw agent adapter for TAU²-bench
│   ├── tau2_env/            Socket-based environment server for cross-process tool execution
│   ├── task_runner.py       TAU²-bench task runner (standard)
│   └── task_runner_socket.py TAU²-bench task runner (socket variant)
├── benchmark/               Performance benchmarking scripts
│   ├── benchmark.py         Sweep entry point (uses RolloutControllerV2)
│   ├── benchmark.yaml       Configuration for the benchmark
│   ├── start_servers.py     Helper to launch SGLang servers
│   ├── collect_metrics.py   SGLang Prometheus metrics collector
│   ├── collect_trajectories.py  Trajectory collection (standalone mode)
│   └── worker.py            Per-task worker subprocess
├── online_rollout.py        Online RL demo (human-in-the-loop)
├── online_rollout.yaml      Config for online rollout
├── tau2_rollout.py          Offline TAU²-bench rollout demo
├── tau2_rollout.yaml        Config for tau2 rollout
├── human_in_the_loop_demo.py  Automated HITL demo script
└── README.md                This file
```

______________________________________________________________________

## Example 1: Offline τ²-Bench Rollout

This example runs rollout-only data generation on the
[$\\tau^2$-Bench](https://github.com/sierra-research/tau2-bench) using the AReaL
Inference Service. Unlike the full training pipeline in `examples/tau2/`, this script
performs rollouts without a training step — useful for evaluation, data collection, or
debugging agent behaviour.

### Installation

#### AReaL

Follow the
[AReaL installation guide](https://areal-project.github.io/AReaL/en/tutorial/installation.html).

#### Tau2

Install the (forked) tau2-bench package:

```bash
pip install git+https://github.com/dhh1995/tau2-bench.git@dhh/async-and-custom-completion
```

Set the `TAU2_DATA_DIR` environment variable:

```bash
export TAU2_DATA_DIR=/path/to/tau2-bench/data
```

### Running

All commands should be executed from the **repository root**.

```bash
python3 examples/experimental/inference_service/tau2_rollout.py \
    --config examples/experimental/inference_service/tau2_rollout.yaml \
    econfig.user_llm_base_url=<USER_LLM_BASE_URL> \
    cluster.fileroot=<EXPERIMENT_ROOT> \
    cluster.name_resolve.nfs_record_root=<NAME_RESOLVE_ROOT>
```

| Placeholder           | Description                                             | Example                     |
| --------------------- | ------------------------------------------------------- | --------------------------- |
| `<USER_LLM_BASE_URL>` | OpenAI-compatible base URL of the user simulator LLM    | `http://localhost:8000/v1/` |
| `<EXPERIMENT_ROOT>`   | Directory for experiment artifacts (logs, trajectories) | `/tmp/areal/experiments`    |
| `<NAME_RESOLVE_ROOT>` | Shared path for name-resolve records                    | `/tmp/areal/name_resolve`   |

### Result

A successful rollout prints per-batch statistics after every batch:

```
(AReaL) 20260319-14:18:25.768 Tau2GatewayRollout INFO: Batch 2: n_trajs=16, rewards=tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.]), avg_reward=0.1250
```

Each line reports the batch index, number of trajectories, individual rewards, and the
batch-level average reward.

______________________________________________________________________

## Example 2: Human-in-the-Loop Online RL Demo

This example demonstrates **human-in-the-loop (HITL) online RL**: a human (or an
automated script acting as one) chats with the model through any OpenAI-compatible
client, provides feedback after each conversation, and the gateway accumulates
trajectories into a training batch. It is the simplest end-to-end illustration of how
AReaL's inference service enables closed-loop RL without modifying the training code.

The automated demo script is `human_in_the_loop_demo.py`. It uses
[zeroclaw](https://github.com/dhh1995/zeroclaw) as the chat client and exercises the
following procedure:

1. **Launch `online_rollout.py`** — starts the SGLang inference engine and the proxy
   gateway, then waits until the gateway address is printed to the log.
1. **Patch `~/.zeroclaw/config.toml`** — redirects zeroclaw's default provider to the
   local gateway and injects the admin API key so all requests are attributed to a
   single RL session. The original config is restored on exit.
1. **Run four HITL rounds** — for each round the script:
   - Asks the model *"how many r's are in the word strawberry?"*.
   - If the answer is wrong, provides a corrective turn and asks once more.
   - Calls `POST /rl/set_reward` on the gateway to push a scalar reward (`1.0` for
     correct, `0.0` for wrong after two attempts).
1. **Verify the batch** — waits for `online_rollout.py` to emit a `Rollout complete` log
   line confirming that all four trajectories (= `batch_size`) were collected and
   processed.

### Prerequisites

- **AReaL installed** — follow the
  [installation guide](https://areal-project.github.io/AReaL/en/tutorial/installation.html).
- **zeroclaw installed** — any OpenAI-compatible CLI that supports
  `--session-state-file` can be substituted; the demo uses zeroclaw for convenience.
- **A zeroclaw config at `~/.zeroclaw/config.toml`** with at least a `default_provider`
  key — the script will patch it temporarily.
- **One GPU** — the default YAML (`online_rollout.yaml`) requests 1 GPU with SGLang.

### Running the Automated Demo

All commands should be executed from the **repository root**.

```bash
python3 examples/experimental/inference_service/human_in_the_loop_demo.py
```

Key CLI arguments:

| Argument             | Default               | Description                                                          |
| -------------------- | --------------------- | -------------------------------------------------------------------- |
| `--actor-path`       | `Qwen/Qwen3-0.6B`     | Path to the HuggingFace model weights                                |
| `--admin-key`        | `sk-test123456`       | Admin API key (must match `rollout.agent.admin_api_key` in the YAML) |
| `--request-timeout`  | `3600`                | Per-request timeout in seconds                                       |
| `--gateway-wait`     | `600`                 | Seconds to wait for the gateway to become ready                      |
| `--question`         | *strawberry question* | Question posed in every HITL round                                   |
| `--external-url`     | `None`                | External API URL (enables external model mode)                       |
| `--external-api-key` | `None`                | API key for the external provider                                    |
| `--external-model`   | `None`                | Model name sent to the external API                                  |

You can override the model path without editing the script:

```bash
python3 examples/experimental/inference_service/human_in_the_loop_demo.py \
    --actor-path /path/to/your/model
```

### External Model Mode (optional)

Example 2 can also run HITL with an external OpenAI-compatible provider instead of the
local rollout model. Pass the external flags through `human_in_the_loop_demo.py`; they
are forwarded to `online_rollout.py`:

```bash
python3 examples/experimental/inference_service/human_in_the_loop_demo.py \
    --external-url https://api.openai.com/v1 \
    --external-api-key sk-... \
    --external-model gpt-4o
```

When `--external-url` is set, the controller enables external model mode and routes chat
traffic through the unified `/chat/completions` + `/export_trajectories` external flow.

### Running a Manual HITL Session

To drive the rollout interactively instead of using the automated script:

**Step 1 — Start the online rollout server:**

```bash
python3 examples/experimental/inference_service/online_rollout.py \
    --config examples/experimental/inference_service/online_rollout.yaml \
    actor.path=<MODEL_PATH>
```

Wait until the log prints:

```
Proxy gateway available at http://127.0.0.1:<PORT>
```

**Step 2 — Chat with the model** using any OpenAI-compatible client, pointing it at
`http://127.0.0.1:<PORT>/v1` with `Authorization: Bearer sk-test123456`.

**Step 3 — Submit a reward** after each conversation turn via HTTP:

```bash
curl -X POST http://127.0.0.1:<PORT>/rl/set_reward \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-test123456" \
    -d '{"reward": 1.0}'
```

Repeat Steps 2–3 until `batch_size` (default: 4) trajectories are complete. The server
will log `Rollout complete` and exit.

### Expected Output

When the demo finishes successfully you should see:

```
════════════════════════════════════════════════════════════════
  Step 5: Check online_rollout output for databatch
════════════════════════════════════════════════════════════════
  ── Rollout log (last 40 lines) ──
  ...
  ✔ Databatch detected:
  (AReaL) ... InferenceServiceOnlineTrain INFO: Rollout complete (4 trajectories), avg_reward=X.XXXX
```

Each of the four HITL rounds also prints whether the model answered correctly on the
first or second try, for example:

```
  ── Trajectory 0 ──
  Q: how many r's are in the word strawberry?
  A: There are 3 r's in the word "strawberry".
  ✔ Correct on first try.
```

______________________________________________________________________

## OpenClaw + TAU²-Bench Agent Environment

The `openclaw_tau2/` directory provides an OpenClaw agent adapter for
[TAU²-bench](https://github.com/sierra-research/tau2-bench), enabling multi-turn
customer service agent tasks (airline, retail, telecom) to run through the AReaL
inference service stack.

Key components:

- **OpenClaw agent** (`openclaw/agent.py`) — wraps TAU²-bench agent interface for
  OpenClaw CLI compatibility
- **Socket environment server** (`tau2_env/environment_socket.py`) — enables
  cross-process tool execution between the OpenClaw CLI and the TAU²-bench environment
- **Task runners** (`task_runner.py`, `task_runner_socket.py`) — orchestrate single-task
  execution with TAU²-bench's simulation loop

### Installation

```bash
pip install git+https://github.com/dhh1995/tau2-bench.git@dhh/async-and-custom-completion
export TAU2_DATA_DIR=/path/to/tau2-bench/data
```

______________________________________________________________________

## Inference Service Benchmark

The `benchmark/` directory measures AReaL inference service full-stack overhead on
TAU²-bench agent tasks using `RolloutControllerV2` + `Tau2AgentWorkflow`.

The benchmark launches the IS services (Router, DataProxy, Gateway) through the
controller. The Agent and User SGLang servers are started separately so you can control
GPU allocation for your hardware.

### How It Works

```
benchmark.py
  └─ RolloutControllerV2
       ├─ Router + DataProxy + Gateway  (launched by controller, CPU-only)
       └─ connects to Agent SGLang      (pre-existing, via --agent-endpoint)

Tau2AgentWorkflow
  └─ connects to User SGLang            (pre-existing, via --user-endpoint)
```

### Option A: Single Node, 8 GPUs, Docker (Small Model)

For models that fit in 4 GPUs (e.g., Qwen3-30B-A3B or smaller). Agent and User each get
4 GPUs on the same node.

**Step 1** — Start both SGLang servers (Agent on GPUs 0-3, User on GPUs 4-7):

```bash
python3 examples/experimental/inference_service/benchmark/start_servers.py \
    --model-path /models/Qwen3-30B-A3B-Instruct \
    --tp 4
```

**Step 2** — Run the benchmark (controller launches IS services only, no extra GPUs
needed):

```bash
python3 examples/experimental/inference_service/benchmark/benchmark.py \
    --config examples/experimental/inference_service/benchmark/benchmark.yaml \
    --model-path /models/Qwen3-30B-A3B-Instruct \
    --agent-endpoint http://127.0.0.1:30000 \
    --user-endpoint http://127.0.0.1:30001/v1 \
    --concurrencies "2,5,10" \
    --num-tasks 20 --num-trials 2
```

### Option B: Multi-Node Cluster (Large Model, e.g., Qwen3-235B)

For models requiring TP=8, run Agent and User on separate 8-GPU nodes.

**Step 1** — Start Agent SGLang on node A (all 8 GPUs):

```bash
python3 examples/experimental/inference_service/benchmark/start_servers.py \
    --model-path /models/Qwen3-235B-A22B-Instruct-2507 \
    --tp 8 --agent-only
```

**Step 2** — Start User SGLang on node B (all 8 GPUs):

```bash
python3 examples/experimental/inference_service/benchmark/start_servers.py \
    --model-path /models/Qwen3-235B-A22B-Instruct-2507 \
    --tp 8 --agent-only --agent-port 30001
```

**Step 3** — Run the benchmark from either node:

```bash
python3 examples/experimental/inference_service/benchmark/benchmark.py \
    --config examples/experimental/inference_service/benchmark/benchmark.yaml \
    --model-path /models/Qwen3-235B-A22B-Instruct-2507 \
    --agent-endpoint http://<node-A>:30000 \
    --user-endpoint http://<node-B>:30001/v1 \
    --concurrencies "5,10,15,20,25,30" \
    --num-tasks 50 --num-trials 4
```

### Configuration

See [`benchmark/benchmark.yaml`](benchmark/benchmark.yaml) for full configuration. Key
overrides can be passed as CLI arguments:

| Argument           | Default            | Description                            |
| ------------------ | ------------------ | -------------------------------------- |
| `--model-path`     | (from YAML)        | Path to model weights                  |
| `--agent-endpoint` | (none)             | Pre-existing Agent SGLang URL          |
| `--user-endpoint`  | (from YAML)        | User simulator SGLang URL              |
| `--concurrencies`  | `5,10,15,20,25,30` | Comma-separated concurrency levels     |
| `--num-tasks`      | `50`               | Number of TAU²-bench tasks per trial   |
| `--num-trials`     | `4`                | Number of trials per concurrency level |
| `--output-dir`     | `./trajectories`   | Output directory for results           |

### Reference Results: Qwen3-235B-A22B-Instruct-2507

Tested on TAU²-bench airline domain, 50 tasks × 4 trials per concurrency, 2 nodes ×
8×H200 GPUs (TP=8 per instance).

#### Baseline (OpenClaw → SGLang direct) vs Target (OpenClaw → IS → SGLang)

Each cell: `Baseline / Target (Δ)`.

| Metric      | c=5                   | c=10                  | c=15                  | c=20                  | c=25                  | c=30                  |
| ----------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| Pass@1      | 38% / 30% (-8pp)      | 38% / 38% (+1pp)      | 34% / 32% (-2pp)      | 36% / 34% (-2pp)      | 34% / 34% (0pp)       | 36% / 38% (+3pp)      |
| Avg E2E (s) | 4.82 / 4.57 (-5%)     | 8.99 / 8.57 (-5%)     | 13.05 / 12.58 (-4%)   | 17.03 / 16.33 (-4%)   | 20.73 / 20.16 (-3%)   | 24.72 / 23.41 (-5%)   |
| Input Tok/s | 15,207 / 16,017 (+5%) | 18,204 / 19,204 (+5%) | 18,820 / 20,138 (+7%) | 19,281 / 19,388 (+1%) | 19,433 / 20,474 (+5%) | 19,480 / 20,780 (+7%) |
| Tasks/min   | 2.5 / 2.6 (+6%)       | 3.0 / 3.2 (+5%)       | 3.1 / 3.1 (+1%)       | 3.0 / 3.0 (0%)        | 3.0 / 3.4 (+13%)      | 3.4 / 3.7 (+8%)       |

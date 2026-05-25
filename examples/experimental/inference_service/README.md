# AReaL Inference Service Examples

This directory contains two examples that use the AReaL Inference Service
(`RolloutControllerV2`) — an experimental rollout backend that exposes an
OpenAI-compatible proxy gateway so any external agent runtime can submit chat requests
and receive RL training data.

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

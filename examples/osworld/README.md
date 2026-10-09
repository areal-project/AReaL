# OSWorld GRPO Example

## 1. Overview

This example runs GRPO/PPO training on [OSWorld](https://github.com/xlang-ai/OSWorld)
desktop-control tasks with a Vision-Language base model (Qwen3-VL-4B-Instruct by
default). The agent observes screenshots, emits keyboard/mouse actions through a
`pyautogui`-style action space, and receives a scalar reward from each task's
`evaluate()` function.

Three components cooperate:

- **AReaL** — the training framework (this repo): FSDP actor, SGLang rollout engine,
  GRPO loss.
- **OSWorld** — the environment code (a sibling checkout, defaults to `../OSWorld`):
  task definitions, the Ubuntu VM image, the in-VM controller HTTP server.
- **Remote sandbox provider** — runs the OSWorld VM behind an HTTPS gateway. Training
  containers usually lack Docker/KVM, so OSWorld's bundled `docker` provider can't run
  in-process; an external host actually starts the VM and forwards controller calls.

Two transports ship in this example:

- `workflow/gateway_sandbox.py` — a pluggable HTTPS-gateway client (recommended).
  Requires a vendor SDK exposed as a Python module named `pssdk` that exports
  `BaseSandboxClusterTool` and `with_retry`. A thin adapter is fine if your provider
  uses different names.
- `workflow/remote_desktop_env.py` — a self-hosted alternative. Pair it with
  `remote_server.py` running on a docker-capable machine.

## 2. Layout

```
__init__.py
README.md                         this file
apply_env_patches.sh              SGLang/pydrive patches needed for the conda env
config_osworld_sglang.yaml        training config (Qwen3-VL-4B-Instruct, 2x GPU)
osworld_config.py                 OSWorldAgentConfig (extends GRPOConfig)
osworld_requirements.txt          OSWorld host-side deps with conflicting versions filtered out
remote_server.py                  optional self-hosted docker bridge (paired with remote_desktop_env.py)
run_train.sh                      launcher with smoke / smoke-text / full stages
smoke.py                          end-to-end sandbox smoke (skips trainer)
train.py                          PPO entry point
workflow/
  __init__.py
  osworld_workflow.py             multi-turn VLM rollout workflow + Plan B VL bridge
  gateway_sandbox.py              DesktopEnv subclass that proxies controller calls through HTTPS gateway
  remote_desktop_env.py           alternative DesktopEnv replacement that talks to remote_server.py
```

## 3. Prerequisites

- Python 3.12+ in an AReaL-compatible CUDA env: `torch 2.9.x+cu129`, `sglang >= 0.5.10`,
  `transformers 5.x`, `areal 1.x`.
- 2x GPU with at least 80 GB each — one for the FSDP actor, one for the SGLang rollout
  engine.
- An OSWorld checkout sibling to AReaL (`../OSWorld`). Override the location via the
  `osworld_root` config field if it lives elsewhere.
- Either an HTTPS-gateway-based sandbox provider with a vendor SDK (gateway path), or a
  separate machine with Docker access (self-hosted path).

## 4. Setup

1. Create or reuse a Python 3.12 conda env. Run `uv sync --extra cuda` per AReaL's main
   README. Then install OSWorld's filtered host deps:

   ```bash
   pip install -r examples/osworld/osworld_requirements.txt
   pip install "nvidia-cudnn-cu12==9.16.0.29" "protobuf>=6.31.1,<7" "grpcio-status==1.80.0"
   ```

   The second `pip install` un-downgrades packages that `easyocr` and
   `google-ai-generativelanguage` drag in.

1. Apply environment patches (idempotent):

   ```bash
   bash examples/osworld/apply_env_patches.sh
   ```

   This script patches:

   - SGLang JIT kernel flag — auto-detects whether the local `nvcc` supports C++20.
   - `pydrive` to `pydrive2` shim — OSWorld imports the unmaintained `pydrive` package.

1. **Gateway path only** — install your sandbox provider's SDK. The expected protocol is
   a Python module named `pssdk` exporting:

   - `BaseSandboxClusterTool(cluster_endpoint, application_secret_token, session_id, global_call_timeout)`
     constructor.
   - `.session_id` (read property).
   - `.sandbox_start(body=None, call_timeout=...) -> dict`.
   - `.sandbox_stop(call_timeout=...) -> dict`.
   - `with_retry(max_attempts, retry_interval, infinite_retry_on_resource_limit, exclude_methods)`
     class decorator.

   If your provider exports a different module name, write a thin adapter module called
   `pssdk` that re-exports these symbols.

## 5. Configure

Defaults live in `config_osworld_sglang.yaml`. Notable fields:

- `actor.path` — HuggingFace model directory for the VL base. Qwen3-VL-4B-Instruct is
  recommended.
- `gateway_endpoint`, `gateway_token` — empty by default; must be set for the gateway
  path.
- `gateway_timeout_secs` — per-call timeout to the gateway (default 1800).
- `remote_server_url` — empty by default; set non-empty to use the self-hosted path
  instead.
- `text_only` — smoke-only ablation. Strips screenshots and lets you point `actor.path`
  at a text-only model to verify the PPO loop without VL.
- `osworld_root`, `evaluation_examples_dir`, `test_meta_path` — auto-discovered from the
  sibling `OSWorld/` checkout when left empty.

Two ways to provide credentials:

```bash
# via env vars (recommended; secrets stay out of source control)
export OSWORLD_SANDBOX_ENDPOINT="https://your-gateway/..."
export OSWORLD_SANDBOX_TOKEN="sk-..."

# via CLI override
python -m examples.osworld.train --config examples/osworld/config_osworld_sglang.yaml \
    gateway_endpoint=$OSWORLD_SANDBOX_ENDPOINT \
    gateway_token=$OSWORLD_SANDBOX_TOKEN
```

## 6. Run flow

### 6a. Sandbox smoke (skip trainer)

This verifies the gateway/SDK end-to-end without touching the trainer:

```bash
export OSWORLD_SANDBOX_ENDPOINT="https://your-gateway/..."
export OSWORLD_SANDBOX_TOKEN="sk-..."
python examples/osworld/smoke.py
```

Expected last lines:

```
sandbox started: <uuid>
reset ok; screenshot bytes=NNNN
step ok; reward=0 done=False ...
evaluate result: 0.0
closed
```

`evaluate result: 0.0` means the agent didn't solve the task but the evaluator returned
a real reward — this is the success signal for the smoke test.

### 6b. Training smoke, Plan A (text-only ablation, fastest end-to-end)

```bash
export OSWORLD_SANDBOX_TOKEN="sk-..."
export AREAL_TEXT_ONLY_MODEL=/path/to/Qwen3-4B-Instruct   # any model_type=qwen3 base
bash examples/osworld/run_train.sh smoke-text
```

This routes through the same workflow but `text_only=true` strips screenshots and uses a
text-only base, so you can verify the PPO loop end-to-end without exercising the VL
training path.

### 6c. Training smoke, Plan B (full VL pipeline)

```bash
export OSWORLD_SANDBOX_TOKEN="sk-..."
bash examples/osworld/run_train.sh smoke
```

The `OSWorldWorkflow._attach_vl_tensor_dicts` bridge re-runs the HF processor on each
turn's prefix and writes `mm_token_type_ids` plus `multi_modal_input` (`pixel_values`,
`image_grid_thw`) into the cached training tensor dict, which is what
`FSDPEngine._prepare_mb_list`'s VL path needs.

### 6d. Full training

```bash
bash examples/osworld/run_train.sh full
```

## 7. Self-hosted alternative

Skip this section if you have a gateway. Otherwise, run OSWorld on a separate
Docker-capable machine and let the trainer talk to it over HTTP.

On the docker machine:

```bash
docker pull xlang/osworld-docker:latest
pip install -r OSWorld/requirements.txt flask pydrive2 "oauth2client<4.1.4"
python remote_server.py --osworld-root /path/to/OSWorld --host 0.0.0.0 --port 8000 --max-envs 2
```

On the train side:

```bash
python -m examples.osworld.train --config examples/osworld/config_osworld_sglang.yaml \
    remote_server_url=http://<remote-host>:8000 \
    rollout.max_concurrent_rollouts=1 \
    n_trajs=1
```

Provider precedence inside `osworld_workflow._build_env`:
`gateway_endpoint + gateway_token` > `remote_server_url` > in-process `DesktopEnv`.

## 8. Configuration knobs (top-level YAML fields)

| Field                                                         | Meaning                                                                                           |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `n_trajs`                                                     | Trajectories per task per episode (GRPO group size).                                              |
| `max_steps`                                                   | Max agent / env turns before forcing `env.evaluate()`.                                            |
| `max_workers`                                                 | ThreadPool size for blocking `DesktopEnv` calls.                                                  |
| `provider_name`                                               | OSWorld provider (`docker` default; only used if neither gateway nor `remote_server_url` is set). |
| `env_reset_wait_secs`                                         | Sleep after `env.reset` to let the VM settle.                                                     |
| `test_meta_path`                                              | Which OSWorld task meta file to train on (default `test_small.json`).                             |
| `text_only`                                                   | Smoke-only ablation; strips screenshots from messages.                                            |
| `gateway_endpoint` / `gateway_token` / `gateway_timeout_secs` | Gateway transport.                                                                                |
| `remote_server_url` / `remote_request_timeout_secs`           | Self-hosted transport.                                                                            |

## 9. Reward semantics

Each trajectory is attributed the float returned by `DesktopEnv.evaluate()` (typically
`0.0` or `1.0`). The reward is applied to the last assistant turn and discounted
backwards per turn by `turn_discount` (default `0.9`).

## 10. Concurrency notes

A sandbox session is one OSWorld VM container (1 vCPU, 4 GB RAM, idle-reaped after
roughly 50 minutes). Your provider's quota controls how many concurrent sessions you can
hold. Start with `rollout.max_concurrent_rollouts=1` and ramp up while watching for HTTP
429s. The bundled retry decorator (`_RetryingClusterTool` in `gateway_sandbox.py`) parks
on 429s rather than killing trajectories.

## 11. Known limitations of the gateway path

| Limitation                                                                                                                                                                       | Impact                                                                                                      | Mitigation                                                                   |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Subset of OSWorld setup verbs supported: `launch`, `download`, `execute`, `open`, `chrome_open_tabs`, `activate_window`, `close_window`, `command`, `sleep`, `change_wallpaper`. | Tasks using `googledrive`, `login`, or `replay` skip-with-warning and may have an inaccurate initial state. | Pick task subsets without those verbs, or extend the controller.             |
| `controller.get_file()` routes through `/execute` plus base64 (slow for large files).                                                                                            | OSWorld's `/file` endpoint requires form-urlencoded, but typical gateways only allow JSON.                  | Ask your provider to allow form encoding, or live with slower file transfer. |
| `/terminal` returns 500 when no active terminal exists.                                                                                                                          | One warning line, non-fatal.                                                                                | OSWorld behavior; ignore.                                                    |

## 12. Troubleshooting

| Symptom                                                 | Likely cause / fix                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: pssdk` at startup                 | Gateway path is selected but the vendor SDK isn't installed. Install it or fall back to `remote_server_url`.                                                                                                                                                                                 |
| 401 on `sandbox_start`                                  | Wrong or expired `OSWORLD_SANDBOX_TOKEN`.                                                                                                                                                                                                                                                    |
| 429 on `sandbox_start` (`ClusterQuotaExceededErr`)      | Global quota exhausted; the retry decorator parks — wait or scale provider capacity.                                                                                                                                                                                                         |
| SGLang JIT compile fails with `std::integral` not found | Run `apply_env_patches.sh` first; the script auto-detects nvcc C++20 support.                                                                                                                                                                                                                |
| `requests` health check timeouts to local SGLang        | Corporate `HTTP_PROXY` in the environment routes localhost requests to a proxy that can't reach internal IPs. `run_train.sh` already appends a generic `NO_PROXY` allowlist for `localhost,127.0.0.1,10.0.0.0/8`; append your internal domains via the `NO_PROXY` env var before invocation. |
| `eval-rollout/0 readiness timeout`                      | The forked Python subprocess re-imports torch + sglang + megatron, which takes 3+ minutes on slow filesystems. `_wait_for_fork_ready` in `areal/infra/scheduler/local.py` should be at least 600s; a small core patch is recommended for slow-disk users.                                    |
| `KeyError: 'mm_token_type_ids'` in `_prepare_mb_list`   | The VL bridge isn't running. Verify `text_only=false` is in effect and that `processor_path=actor.path` was passed through.                                                                                                                                                                  |

## 13. What's not yet covered

- `setup/upload` and `setup/execute_with_verification` verbs are not wired through
  `gateway_sandbox.py`; tasks needing them will skip-with-warning.
- WandB is disabled in the default config — flip `stats_logger.wandb.mode` to enable.

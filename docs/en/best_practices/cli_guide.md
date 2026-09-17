# AReaL CLI Guide

The `areal` CLI ships three top-level subcommand groups, one per service in the AReaL
2.0 microservice architecture:

- **`areal train`** — resolve an experiment driver and config, forward hydra overrides,
  and hand off to the training script.
- **`areal inf`** — launch and manage a local inference service (gateway, router, model
  workers, data proxies).
- **`areal agent`** — launch and manage a local agent service (gateway, router, N
  worker/data-proxy pairs).

`areal inf` and `areal agent` share the same lifecycle model — `run`, `ps`, `status`,
`logs`, `stop` — with per-service state stored under `~/.areal/` (override with the
`AREAL_HOME` env var). `areal train` is deliberately stateless: it only wires argv
through to a Python `main(args)` driver.

## Training CLI (`areal train`)

`areal train` wires an AReaL training driver function and an experiment config file into
a single command. It does not manage the training process lifecycle (unlike `areal inf`,
which maintains service state); it only "finds the driver, loads the config, and passes
hydra-style overrides through."

### Basic concepts

The minimum execution unit of a training job is a **driver function** — typically a
`main(args: list[str])` in some script under `examples/`. The CLI does exactly three
things:

1. Resolve the driver from `module.path:func`
1. Resolve `--config <path>` to an absolute path and prepend it to argv
1. Append every trailing argument unchanged to argv (typically hydra overrides)

The driver function's return value is used as the exit code if it returns `int`;
anything else (including `None`) is treated as 0.

### Usage

```bash
areal train run \
  --config <path/to/experiment.yaml> \
  --driver <module.path>:<func> \
  [<hydra-override-1> <hydra-override-2> ...]
```

| flag / arg               | required | description                                                                   |
| ------------------------ | -------- | ----------------------------------------------------------------------------- |
| `--config`               | yes      | Experiment YAML path; the file must exist (the CLI runs an `exists` check)    |
| `--driver`               | yes      | Driver entry point in `module.path:func` form (colon-separated)               |
| trailing positional args | no       | Forwarded to the driver verbatim; typically hydra-style `key=value` overrides |

`run_cmd` uses `context_settings={"ignore_unknown_options": True}` — trailing positional
args **can include `--xxx` flags**; the CLI does not try to parse them and forwards them
as-is to the driver.

### Examples

Run GSM8K GRPO (the most common baseline):

```bash
areal train run \
  --config examples/math/gsm8k_grpo.yaml \
  --driver examples.math.gsm8k_rl:main \
  experiment_name=gsm8k_grpo_test \
  trial_name=t1
```

Run SFT:

```bash
areal train run \
  --config examples/math/gsm8k_sft.yaml \
  --driver examples.math.gsm8k_sft:main
```

### Driver function conventions

The CLI calls the driver with a single argument `argv: list[str]`, so the driver must
look like:

```python
def main(args: list[str]) -> int | None:
    config, _ = load_expr_config(args, GRPOConfig)   # or any other *Config dataclass
    ...
    return 0
```

`load_expr_config` lives in `areal.api.cli_args` and consumes `args` itself: it
recognises the YAML path after `--config`, and treats every remaining `key=value` as a
hydra override merged into the config dataclass. Hydra parsing is **done by the
driver**, not the CLI.

Minimum template for writing a new driver:

```python
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal import PPOTrainer

def main(args):
    config, _ = load_expr_config(args, GRPOConfig)
    with PPOTrainer(config, train_dataset=..., valid_dataset=...) as trainer:
        trainer.train(workflow="...", workflow_kwargs={...})
    return 0
```

### Hydra overrides

Any driver that uses `load_expr_config` to parse args supports hydra-style overrides.
Common override targets:

```bash
# experiment / trial naming
experiment_name=my_run trial_name=t1

# cluster size
cluster.n_nodes=4 cluster.n_gpus_per_node=8

# training hyperparameters
actor.optimizer.lr=5e-6
total_train_epochs=20

# rollout backend
rollout.backend=sglang:d2p1t2
rollout.max_concurrent_rollouts=128

# datasets
train_dataset.batch_size=256
```

The CLI does not validate whether these keys are legal; unknown fields will be reported
by hydra when the driver loads the config.

### Exit codes

| Scenario                                               | exit code                                 |
| ------------------------------------------------------ | ----------------------------------------- |
| Driver returns `int`                                   | Returned value used directly as exit code |
| Driver returns `None` / other                          | 0                                         |
| `--driver` does not contain `:`                        | UsageError (click default 2)              |
| Module referenced by `--driver` cannot be imported     | ClickException (1)                        |
| Function referenced by `--driver` is not on the module | ClickException (1)                        |
| `--config` path does not exist                         | Caught by click `exists=True` (2)         |

Exceptions raised inside the driver **are not caught by the CLI** — the default Python
behaviour applies (traceback printed, process exits).

### Not implemented yet

`areal train` currently only implements `run`. The following are reasonable future
extensions but are **not** in this version:

- `areal train ps` / `status` / `stop` — lifecycle management for training jobs
  (requires a training service state concept first)

## Inference Service CLI (`areal inf`)

`areal inf` launches and manages an AReaL inference service on the local machine. It
starts the gateway/router, registers models, inspects service state, and manages logs.

### Basic concepts

An inference service typically contains the following components:

- `gateway`: exposes OpenAI-compatible API and RL API to the outside.
- `router`: maintains the model → worker/data-proxy routing.
- `model worker`: the actual inference backend, e.g. SGLang.
- `data proxy`: records interactions and rewards, and supports trajectory export.

The CLI's local state is stored under `~/.areal/inf/` by default. The root directory can
be overridden via `AREAL_HOME`:

```bash
export AREAL_HOME=/path/to/areal-home
```

### Launching the service

Launch an empty inference service:

```bash
areal inf run \
  --service default \
  --host 127.0.0.1 \
  --port 8080 \
  --admin-api-key areal-admin-key \
  --scheduler local \
  --detach
```

`--scheduler` selects the scheduling backend for workers / data-proxies. Only `local` is
supported today (and is the default). Once the service starts, this value is pinned into
the service state, so subsequent `register` / `stop` / `status` calls read it from state
and do not need `--scheduler` again.

List services known to the local machine:

```bash
areal inf ps
areal inf status --service default
```

`ps` shows the service list; `status` drills into the state of the gateway, router,
data-proxy, workers, etc.

List registered models:

```bash
areal inf models --service default
```

### Registering a model

`register` makes the CLI launch a local inference backend together with a data-proxy:

```bash
areal inf register \
  --service default \
  --model-name qwen-local \
  --backend sglang:d1 \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --tokenizer-path Qwen/Qwen2.5-7B-Instruct \
  --engine-args "--mem-fraction-static 0.8" \
  --proxy-args "--request-timeout 120 --chat-template-type hf"
```

`--engine-args` is a shell-style string forwarded verbatim to the sglang / vllm worker
process; `--proxy-args` is the analogous flag for the data-proxy process. Available
data-proxy flags include `--request-timeout`, `--set-reward-finish-timeout`,
`--tool-call-parser`, `--reasoning-parser`, `--engine-max-tokens`, and
`--chat-template-type {hf|concat}`.

A model can also be registered directly at `run` time:

```bash
areal inf run \
  --service default \
  --port 8080 \
  --admin-api-key areal-admin-key \
  --model qwen-local \
  --backend sglang:d1 \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --engine-args "--mem-fraction-static 0.8" \
  --proxy-args "--request-timeout 120 --chat-template-type hf" \
  --detach
```

### Plain inference requests

Once a model is registered, the gateway's OpenAI-compatible endpoint can be called
directly:

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer areal-admin-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-local",
    "messages": [
      {"role": "user", "content": "Hi, give me a quick intro to AReaL."}
    ],
    "max_tokens": 128
  }'
```

### Logs and cleanup

View logs:

```bash
areal inf logs --service default --component gateway -f
areal inf logs --service default --component router -f
areal inf logs --service default --component qwen-local-worker-0 -f
areal inf logs --service default --component qwen-local-data-proxy-0 -f
```

Each model's worker / data-proxy log file is named `<model-name>-worker-<rank>` and
`<model-name>-data-proxy-<rank>`. If `--component` is wrong or the file does not exist,
the CLI prints the available names.

Deregister a model:

```bash
areal inf deregister --service default --model-name qwen-local
```

Stop the service:

```bash
areal inf stop --service default
```

Force stop:

```bash
areal inf stop --service default --force
```

### Configuration file

`areal inf` reads a default config from:

```bash
~/.areal/inf/config.toml
```

Additional config files can be passed in:

```bash
areal inf --config ./inf.toml run --service default --detach
```

Example:

```toml
[default]
service = "default"

[launch]
gateway_host = "127.0.0.1"
gateway_port = 8080
routing_strategy = "round_robin"

[scheduler]
type = "local"

[register.internal]
backend = "sglang:d1"
model_health_timeout = 600
engine_args = "--mem-fraction-static 0.8"
proxy_args = "--request-timeout 120 --chat-template-type hf"
```

## Agent Service CLI (`areal agent`)

`areal agent` launches a set of agent service processes on the local machine (gateway /
router + N worker/data-proxy pairs) so that an upstream application can interact with
the agent over HTTP. Its shape is very similar to `areal inf`, but it serves an agent
(multi-turn interaction with session state) rather than stateless inference.

### Basic concepts

A running agent service has the following components:

- `gateway`: exposes session and agent APIs to the outside. One per service.
- `router`: routes requests to a data-proxy based on load. One per service.
- `worker[i]`: the process that actually runs user agent code. The agent class is
  imported from the `module.path` form passed via `--agent`.
- `data-proxy[i]`: the session-management / accounting layer in front of `worker[i]`.
  Each worker is paired with one proxy to form a **pair**. `--num-pairs` controls the
  number of replicas.

Request flow:

```
client → gateway → router → data-proxy[i] → worker[i] → agent code
                            └ session lifecycle / record ┘
```

The CLI's local state is stored under `~/.areal/agent/` by default. The root directory
can be overridden via `AREAL_HOME`:

```bash
export AREAL_HOME=/path/to/areal-home
```

### Launching the service

Minimum launch — a single (worker, proxy) pair:

```bash
areal agent run \
  --service default \
  --agent my_package.my_agent.MyAgent \
  --num-pairs 1 \
  --admin-api-key areal-agent-admin
```

`--agent` is required; it is the import path the worker process uses to load the agent
class.

Force-start by clearing stale state:

```bash
areal agent run --service default --agent ... --force
```

### Inspecting service state

List all agent services on the local machine:

```bash
areal agent ps
areal agent ps --all          # include stale rows
areal agent ps --json
```

Output columns: `SERVICE / STATUS / GATEWAY / AGENT`.

Drill into a single service for per-component health:

```bash
areal agent status --service default
```

The output includes the gateway, router, and each pair's worker + proxy. `--watch` mode
refreshes on an interval (default 2 seconds):

```bash
areal agent status --service default --watch --interval 1
```

JSON mode plays well with jq:

```bash
areal agent status --service default --json | jq '.pairs[].worker'
```

### Talking to the service

The CLI **does not** manage how an application talks to the service — applications hit
the gateway HTTP endpoints directly. The status command tells you the gateway URL:

```bash
GATEWAY_URL=$(areal agent status --service default --json | jq -r '.gateway.url')
echo "gateway at $GATEWAY_URL"
```

The application then sends requests against that URL with `--admin-api-key` (or a
session key obtained from the gateway).

### Logs

Each component writes a separate log file:

```bash
areal agent logs --service default --component gateway -f
areal agent logs --service default --component router -f
areal agent logs --service default --component worker-0 -f
areal agent logs --service default --component proxy-0 -f
```

Naming convention:

- `gateway` / `router`: service-level singletons
- `worker-<i>` / `proxy-<i>`: the worker / data-proxy of the i-th pair (i starts from 0)

If `--component` is wrong, the CLI prints the available names. `-f` uses `tail -F`
semantics.

### Stopping

```bash
areal agent stop --service default
```

The default is a two-phase shutdown: SIGTERM, wait `--grace-period` (10s), then SIGKILL.
Immediate SIGKILL:

```bash
areal agent stop --service default --force
```

`--keep-state` preserves the state file (kills processes but leaves the on-disk
`<svc>.json` alone):

```bash
areal agent stop --service default --keep-state
```

### Configuration file

`areal agent` reads `~/.areal/agent/config.toml` as defaults on startup; additional
config files can be passed in:

```bash
areal agent --config ./my-agent.toml run --service default --agent ...
```

Example:

```toml
[default]
service = "default"
admin_api_key = "areal-agent-admin"
log_level = "info"

[run]
agent = "my_package.my_agent.MyAgent"
num_pairs = 2
setup_timeout = 120
health_poll_interval = 5
drain_timeout = 30
session_timeout = 1800
```

Precedence: **CLI flag > TOML passed via `--config` > `~/.areal/agent/config.toml` >
hardcoded defaults**.

### Not implemented yet

The current `areal agent` does **not** include:

- Session-level CLI operations (start session / set reward / export trajectory) — these
  are tightly coupled with the application and are handled by the application talking
  directly to the gateway HTTP.
- Automatic failure recovery / heartbeat monitoring — `status` is on-demand; it does not
  continuously observe component health. If a worker dies, users have to discover it by
  running `status` or checking logs.
- Distributed scheduling — only local processes on the local machine; k8s / slurm and
  friends are out of scope for the current CLI.

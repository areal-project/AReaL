# AReaL Agent Service CLI

`areal agent` is the agent subcommand group under the top-level `areal` CLI. It launches
a set of agent service processes on the local machine (gateway / router + N
worker/data-proxy pairs) so that an upstream application can interact with the agent
over HTTP. Its shape is very similar to `areal inf`, but it serves an agent (multi-turn
interaction with session state) rather than stateless inference.

## Basic concepts

A running agent service has the following components:

- `gateway`: exposes session and agent APIs to the outside. One per service.
- `router`: routes requests to a data-proxy based on load. One per service.
- `worker[i]`: the process that actually runs user agent code. The agent class is
  imported from the `module.path` form passed via `--agent`.
- `data-proxy[i]`: the session-management / accounting layer in front of `worker[i]`.
  Each worker is paired with one proxy to form a **pair**.`--num-pairs` controls the
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

## Launching the service

Minimum launch — a single (worker, proxy) pair:

```bash
areal agent run \
  --service default \
  --agent my_package.my_agent:MyAgent \
  --num-pairs 1 \
  --admin-api-key areal-agent-admin
```

`--agent` is required; it is the import path the worker process uses to load the agent
class.

Multiple pairs (scale throughput horizontally):

```bash
areal agent run \
  --service default \
  --agent my_package.my_agent:MyAgent \
  --num-pairs 4
```

Wire the agent to an inference service so that the agent's internal LLM calls go to a
gateway launched by `areal inf`:

```bash
areal agent run \
  --service default \
  --agent my_package.my_agent:MyAgent \
  --num-pairs 2 \
  --inf-addr http://127.0.0.1:8080 \
  --inf-api-key areal-admin-key \
  --inf-model qwen-local
```

The `--inf-*` group is **optional**: if the agent uses a different LLM interface
internally (direct OpenAI, local vLLM, etc.), these flags are not needed.

Force-start by clearing stale state:

```bash
areal agent run --service default --agent ... --force
```

## Inspecting service state

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

## Talking to the service

The CLI **does not** manage how an application talks to the service —applications hit
the gateway HTTP endpoints directly. The status command tells you the gateway URL:

```bash
GATEWAY_URL=$(areal agent status --service default --json | jq -r '.gateway.url')
echo "gateway at $GATEWAY_URL"
```

The application then sends requests against that URL with `--admin-api-key` (or a
session key obtained from the gateway).

## Logs

Each component writes a separate log file:

```bash
areal agent logs --service default --component gateway -f
areal agent logs --service default --component router -f
areal agent logs --service default --component worker-0 -f
areal agent logs --service default --component proxy-0 -f
```

Naming convention:

- `gateway` / `router`: service-level singletons
- `worker-<i>` / `proxy-<i>`: the worker / data-proxy of the i-th pair (istarts from 0)

If `--component` is wrong, the CLI prints the available names. `-f` uses `tail -F`
semantics.

## Stopping

```bash
areal agent stop --service default
```

The default is a two-phase shutdown: SIGTERM, wait `--grace-period`(10s), then SIGKILL.
Immediate SIGKILL:

```bash
areal agent stop --service default --force
```

`--keep-state` preserves the state file (kills processes but leaves the on-disk
`<svc>.json` alone):

```bash
areal agent stop --service default --keep-state
```

## Configuration file

`areal agent` reads `~/.areal/agent/config.toml` as defaults on startup;additional
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
agent = "my_package.my_agent:MyAgent"
num_pairs = 2
setup_timeout = 120
health_poll_interval = 5
drain_timeout = 30
session_timeout = 1800

[inference]
addr = "http://127.0.0.1:8080"
api_key = "areal-admin-key"
model = "qwen-local"
```

Precedence: \*\*CLI flag > TOML passed via `--config` > `~/.areal/agent/config.toml`

> hardcoded defaults\*\*.

## Relationship with `areal inf`

- `areal inf` launches a **stateless inference service**: given a prompt,it returns a
  completion.
- `areal agent` launches an **agent service with session state**: the agent class may
  need multi-turn interaction, state, and context memory.
- They are **independent**: you can run only `areal inf`, only `areal agent`, or both —
  in the latter case, use `--inf-*` to route the agent's LLM calls to the inf gateway.

## Not implemented yet

The current `areal agent` does **not** include:

- Session-level CLI operations (start session / set reward / export trajectory) — these
  are tightly coupled with the application and are handled by the application talking
  directly to the gateway HTTP
- Automatic failure recovery / heartbeat monitoring — `status` is on-demand; it does not
  continuously observe component health. If a worker dies, users have to discover it by
  running `status` or checking logs.
- Distributed scheduling — only local processes on the local machine; k8s / slurm and
  friends are out of scope for the current CLI.

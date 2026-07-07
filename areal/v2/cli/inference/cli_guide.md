# AReaL Inference Service CLI

`areal inf` launches and manages an AReaL inference service on the local machine. It
starts the gateway/router, registers models, inspects service state, and manages logs.

## Basic concepts

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

## Launching the service

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

## Registering a model

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

## Plain inference requests

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

## Logs and cleanup

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

## Configuration file

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

# SWE-bench RL training with AReaL-SWEAgent

This example runs SWE-bench coding-agent RL (GRPO) in AReaL. The actual agent loop,
sandboxing and reward computation live in a **separate repository**,
[AReaL-SWEAgent](https://github.com/areal-project/AReaL-SWEAgent): for each SWE-bench
instance it edits code inside an isolated sandbox, grades the patch, and returns a
reward for the RL loop. AReaL serves the policy being trained and drives rollouts
through its OpenAI-compatible proxy.

```
AReaL (this repo)                         AReaL-SWEAgent (separate checkout)
  PPOTrainer / rollout proxy  ──▶  run_agent_with_reward()  ──▶  sandbox + grade
        ▲                                                              │
        └──────────────── reward, stats ◀──────────────────────────────┘
```

## 1. Get AReaL-SWEAgent

Clone it next to AReaL (the default lookup path is `../AReaL-SWEAgent`) and install its
dependencies into the same environment AReaL workers run in:

```bash
git clone https://github.com/areal-project/AReaL-SWEAgent.git
cd AReaL-SWEAgent
pip install -r requirements.txt
```

AReaL imports it as the Python package `aweagent`; the repository directory name
(`AReaL-SWEAgent`) and the package name (`aweagent`) intentionally differ.

## 2. Point AReaL at the checkout

AReaL discovers the checkout in this order (first match wins):

1. `econfig.agent_root` in your YAML (legacy alias: `econfig.swe_agent_root`).
1. `AWEAGENT_ROOT` / `SWE_AGENT_ROOT` environment variable.
1. `../AReaL-SWEAgent` relative to the AReaL repo root.

In a multi-node / containerized setup the checkout must be importable on every worker,
so put it on shared storage and set both the env var and `PYTHONPATH`, e.g. in the
worker `scheduling_spec.env_vars`:

```yaml
env_vars:
  AWEAGENT_ROOT: "/path/to/AReaL-SWEAgent"
  SWE_AGENT_ROOT: "/path/to/AReaL-SWEAgent"
  PYTHONPATH: "/path/to/AReaL-SWEAgent:/path/to/AReaL"
  # URL of the AEnvironment sandbox service used by AReaL-SWEAgent.
  AENV_SYSTEM_URL: "http://your-aenv-service:8080"
```

## 3. Configure the agent (econfig)

The SWE example reads an `econfig` block (see `examples/swe/utils.py`):

| Field                   | Meaning                                                                                                     |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- |
| `agent_type`            | Agent to run: `swe` (built-in tool-use) or `cc` (Claude Code).                                              |
| `agent_config`          | Generic config name under `AReaL-SWEAgent/aweagent/configs/`. Overrides the per-type fields below when set. |
| `swe_agent_config`      | Config used when `agent_type=swe` (default `1_0_0/min-swe-agent-train-top1`).                               |
| `cc_agent_config`       | Config used when `agent_type=cc`.                                                                           |
| `agent_root`            | Path to the AReaL-SWEAgent checkout (see section 2).                                                        |
| `step_limit`            | Max agent interaction steps per episode.                                                                    |
| `max_completion_tokens` | Max completion tokens per agent LLM call.                                                                   |
| `timeout`               | Max wall-clock time per episode (seconds).                                                                  |

## 4. Dataset format

`train_swe_rl.py` loads a JSONL file (`train_dataset.path`) where each line is a
SWE-bench instance with at least:

- `instance_id` — e.g. `django__django-10097`
- `problem_statement` — the GitHub issue text
- `eval_script` — shell script used by AReaL-SWEAgent to grade the patch

## 5. Run

Edit the placeholder paths in `qwen3_30b_a3b_grpo.yaml` (model, dataset,
`AWEAGENT_ROOT`, caches, sandbox/W&B endpoints), then launch:

```bash
python -m examples.swe.train_swe_rl --config examples/swe/qwen3_30b_a3b_grpo.yaml
```

With `scheduler.type=slurm`, AReaL launches the rollout / actor / proxy workers; each
rollout calls into AReaL-SWEAgent, which runs the agent in a sandbox and returns the
reward.

### Bailing V3 with AWEX colocation

Bailing V3 uses a Theta-derived SGLang runtime and an AWEX build with the
`BailingMoeV3ForCausalLM` converter. Pin both source revisions before starting a run:

- [SGLang compatibility source](https://github.com/dingzhiqiang/sglang/commit/1743a69fbeb48f1e2d1daa2f7963a82074fce0b9):
  `1743a69fbeb48f1e2d1daa2f7963a82074fce0b9`, based on the public Theta-derived
  [Ling V3 branch](https://github.com/inclusionAI/sglang/commit/3127285c7de5b5d3f5379146c3abe17101a57ac7).
- [AWEX V3 source](https://github.com/dingzhiqiang/asystem-awex/commit/a3c26c7d5a8e5c33b09eeaf46d18c5aabebd04fd):
  `a3c26c7d5a8e5c33b09eeaf46d18c5aabebd04fd`, submitted in
  [AWEX #120](https://github.com/inclusionAI/Awex/pull/120).

The compatibility changes are covered by CPU tests; the combined public runtime still
requires the GPU checks below before production use.

Use separate actor and rollout images/environments: AReaL's default dependency set pins
SGLang 0.5.10.post1 and Transformers \<=5.3.0, while the Theta-derived runtime has its
own dependency requirements. Installing its dependencies into the actor's environment is
not supported by this recipe. Both images need the same AReaL and AWEX sources, and the
rollout image also needs the pinned SGLang runtime and its matching kernels. Source
paths alone do not resolve kernel or Python dependency incompatibilities.

Starting from `qwen3_30b_a3b_grpo.yaml`, merge the following settings into a copy. This
example uses eight 8-GPU nodes: the actor and rollout each use the same 64 physical
GPUs. Set `V3_MODEL`, `AWEX_ROOT`, `AREAL_ROOT`, `SWE_AGENT_ROOT`, `THETA_SGLANG_ROOT`
(the fork's `python` directory), and `V3_ROLLOUT_IMAGE` in your launch environment. Keep
the existing actor image, sandbox settings and other site-specific fields. Add the
pinned AWEX source to the actor's existing `scheduling_spec[0].env_vars.PYTHONPATH` as
well.

```yaml
cluster:
  n_nodes: 8
  n_gpus_per_node: 8

train_dataset:
  batch_size: 8

rollout:
  backend: "sglang:d8t8p1"
  request_timeout: 20000.0
  pause_grace_period: 30
  scheduling_strategy:
    type: colocation
    target: actor
  # A separate rollout image keeps the fork dependencies and allocator settings
  # independent from the actor. Configure reservation/nodelist for your cluster.
  scheduling_spec:
    - task_type: worker
      port_count: 2
      gpu: 1
      cpu: 4
      mem: 32
      image: ${oc.env:V3_ROLLOUT_IMAGE}
      cmd: python3 -m areal.infra.rpc.rpc_server
      env_vars:
        AREAL_SGLANG_FORK: theta
        AREAL_SGLANG_CONTRACT: strict
        AREAL_AWEX_SENTINEL: "1"
        PYTORCH_CUDA_ALLOC_CONF: ""
        PYTHONPATH: ${oc.env:THETA_SGLANG_ROOT}:${oc.env:AWEX_ROOT}:${oc.env:SWE_AGENT_ROOT}:${oc.env:AREAL_ROOT}
        AENV_SYSTEM_URL: ${oc.env:AENV_SYSTEM_URL}
        SWE_AGENT_ROOT: ${oc.env:SWE_AGENT_ROOT}
        AWEAGENT_ROOT: ${oc.env:SWE_AGENT_ROOT}
  agent:
    tool_call_parser: ling3
    reasoning_parser: ling3

actor:
  backend: "megatron:(attn:d4p2t4c2|ffn:d4p2e8)"
  path: ${oc.env:V3_MODEL}
  weight_update_mode: awex
  megatron:
    use_deterministic_algorithms: false
sglang:
  enable_memory_saver: true
  mem_fraction_static: 0.45
  disable_cuda_graph: true
  attention_backend: fa3
  disable_radix_cache: false
```

`AREAL_SGLANG_FORK=theta` selects `abort` followed by `retract`: the first request
returns partial generations to AReaL; the second pauses scheduler execution before
weight transfer. Any remaining retracted requests recompute their KV cache after resume.
Cache release still requires every native idle condition except an empty waiting queue,
and this exemption is limited to a successful retract pause.

For Slurm, propagate `SBATCH_DISTRIBUTION=block:block` and
`SLURM_DISTRIBUTION=block:block` through the launch environment/container so rank
placement stays consistent with AWEX colocation. Retain shared cache paths from your
cluster setup. Resource and memory limits must be checked on that cluster.

Before a long run, validate that the checkpoint's EOS token matches the assistant turn
terminator in its chat template. Check the strict SGLang contract at startup, then run
at least two RL steps with weight updates. After each update, compare AWEX sentinel
values with the expected shard of a saved checkpoint (different TP ranks alone are not
proof of correctness), and check temperature-zero generation. Disable
`AREAL_AWEX_SENTINEL` after bring-up because it adds GPU to CPU synchronization. Full
TP8/NCCL, request retraction and model-quality validation cannot be replaced by the CPU
unit tests.

## 6. Set up the AEnvironment backend

`AENV_SYSTEM_URL` (section 2) must point at a running
[AEnvironment](https://github.com/inclusionAI/AEnvironment) platform. AReaL never talks
to a sandbox directly — AReaL-SWEAgent asks the AEnvironment API service to spin up two
kinds of environment per rollout:

| Role                                                                    | Environment AReaL-SWEAgent requests                          | Ships in the AEnvironment repo?               |
| ----------------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| **Task sandbox** — where the agent runs shell commands and edits code   | `persistent-bash-env` (swe agent) / `cc-bash-env` (cc agent) | No — provided by your AEnvironment deployment |
| **Grading sandbox** — where the patch is scored resolved / not-resolved | `swebench` (overridable via `SWEBENCH_EVAL_ENV`)             | Yes — `aenv/builtin-envs/swebench`            |

### 6.1 Deploy the platform

AEnvironment deploys on Kubernetes via Helm (see its
[deployment guide](https://github.com/inclusionAI/AEnvironment/blob/main/docs/getting_started/deployment.md)):

```bash
git clone https://github.com/inclusionAI/AEnvironment.git
cd AEnvironment
helm install aenv-platform ./deploy \
  --namespace aenv --create-namespace --wait --timeout 10m
```

The API service is then reachable in-cluster at port `8080` — this is the value for
`AENV_SYSTEM_URL`:

```
AENV_SYSTEM_URL = http://api-service.aenv.svc.cluster.local:8080
```

### 6.2 Register the `swebench` grading environment (guaranteed example)

The grading environment is the reward source and is the one piece shipped in the
AEnvironment repo, so it always works out of the box. Install the CLI, point it at your
deployment, then build and publish the bundled `swebench` environment:

```bash
pip install aenvironment
aenv config init
aenv config set hub_backend http://envhub.aenv.svc.cluster.local:8083/
export AENV_SYSTEM_URL=http://api-service.aenv.svc.cluster.local:8080

cd AEnvironment/aenv/builtin-envs/swebench   # config.json: name=swebench, version=1.0.0
aenv build --push                            # requires a configured image registry
aenv push                                    # publish metadata to EnvHub
aenv list                                    # `swebench` should now appear
```

Its reward entrypoint is `swebench_reward(instance, model_patch, timeout)`, which is
exactly the `call_reward({"instance", "model_patch", "timeout"})` contract
AReaL-SWEAgent uses to grade patches.

> **Version note:** the repo ships `swebench@1.0.0`, while AReaL-SWEAgent defaults to
> `swebench@1.0.4`. Either export `SWEBENCH_EVAL_ENV=swebench@1.0.0` in the worker
> `env_vars`, or bump the `version` in `config.json` before `aenv build`.

### 6.3 Provide the task sandbox

The `persistent-bash-env` / `cc-bash-env` task sandboxes are **not** shipped in the
AEnvironment repo. Use the image already registered in your AEnvironment cluster, or
build your own environment that exposes a shell (the repo's
`aenv/builtin-envs/mini-terminal` is the closest reference) and register it the same way
as section 6.2. AReaL-SWEAgent selects it through the agent config's `aenv_version`
field.

## 7. Train the Claude Code (cc) agent

Section 5 runs the built-in `swe` agent. To train the **Claude Code** agent instead —
Claude Code solves the task in a `cc-bash-env` sandbox and the same `swebench`
environment grades the resulting patch — switch `econfig.agent_type` to `cc` and select
a cc config:

```yaml
econfig:
  agent_type: cc
  cc_agent_config: train_cc_time3600   # a config under AReaL-SWEAgent/aweagent/configs/
  agent_root: /path/to/AReaL-SWEAgent
  timeout: 3600.0
```

The cc agent's LLM calls are still routed through AReaL's OpenAI-compatible proxy (AReaL
injects the per-rollout `base_url`/`api_key`), so training stays on-policy exactly like
the swe path.

### 7.1 Environment variables for cc

In addition to the section 2 variables (`AWEAGENT_ROOT`, `PYTHONPATH`,
`AENV_SYSTEM_URL`), the cc agent reads these from the worker `env_vars` — they are
consumed by the cc config in AReaL-SWEAgent (`aweagent/configs/1_0_0/cc.yaml`), not by
AReaL itself:

| Variable                                 | Purpose                                                    |
| ---------------------------------------- | ---------------------------------------------------------- |
| `SWEBENCH_EVAL_ENV`                      | Grading environment name, e.g. `swebench@1.0.0` (see 6.2). |
| `REMOTE_PROXY_SERVICE_URL`               | Gateway that fronts the Claude CLI traffic (proxy mode).   |
| `REMOTE_PROXY_API_KEY`                   | API key for that proxy gateway.                            |
| `ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL` | Used instead when running the CLI in direct mode.          |

The concrete agent behaviour — system prompt, `cli_flags`, tool allow-list,
thinking-token budgets — lives entirely in the AReaL-SWEAgent cc config; see that
repository for the available `cc_agent_config` values and their fields.

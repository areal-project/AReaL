# SWE-bench RL training with AReaL-SWEAgent

This example runs SWE-bench coding-agent RL (GRPO) in AReaL. The actual agent
loop, sandboxing and reward computation live in a **separate repository**,
[AReaL-SWEAgent](https://github.com/areal-project/AReaL-SWEAgent): for each
SWE-bench instance it edits code inside an isolated sandbox, grades the patch,
and returns a reward for the RL loop. AReaL serves the policy being trained and
drives rollouts through its OpenAI-compatible proxy.

```
AReaL (this repo)                         AReaL-SWEAgent (separate checkout)
  PPOTrainer / rollout proxy  ──▶  run_agent_with_reward()  ──▶  sandbox + grade
        ▲                                                              │
        └──────────────── reward, stats ◀──────────────────────────────┘
```

## 1. Get AReaL-SWEAgent

Clone it next to AReaL (the default lookup path is `../AReaL-SWEAgent`) and
install its dependencies into the same environment AReaL workers run in:

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
2. `AWEAGENT_ROOT` / `SWE_AGENT_ROOT` environment variable.
3. `../AReaL-SWEAgent` relative to the AReaL repo root.

In a multi-node / containerized setup the checkout must be importable on every
worker, so put it on shared storage and set both the env var and `PYTHONPATH`,
e.g. in the worker `scheduling_spec.env_vars`:

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

| Field | Meaning |
| --- | --- |
| `agent_type` | Agent to run: `swe` (built-in tool-use) or `cc` (Claude Code). |
| `agent_config` | Generic config name under `AReaL-SWEAgent/aweagent/configs/`. Overrides the per-type fields below when set. |
| `swe_agent_config` | Config used when `agent_type=swe` (default `1_0_0/min-swe-agent-train-top1`). |
| `cc_agent_config` | Config used when `agent_type=cc`. |
| `agent_root` | Path to the AReaL-SWEAgent checkout (see section 2). |
| `step_limit` | Max agent interaction steps per episode. |
| `max_completion_tokens` | Max completion tokens per agent LLM call. |
| `timeout` | Max wall-clock time per episode (seconds). |

## 4. Dataset format

`train_swe_rl.py` loads a JSONL file (`train_dataset.path`) where each line is a
SWE-bench instance with at least:

- `instance_id` — e.g. `django__django-10097`
- `problem_statement` — the GitHub issue text
- `eval_script` — shell script used by AReaL-SWEAgent to grade the patch

## 5. Run

Edit the placeholder paths in `qwen3_30b_a3b_grpo.yaml` (model,
dataset, `AWEAGENT_ROOT`, caches, sandbox/W&B endpoints), then launch:

```bash
python -m examples.swe.train_swe_rl --config examples/swe/qwen3_30b_a3b_grpo.yaml
```

With `scheduler.type=slurm`, AReaL launches the rollout / actor / proxy workers;
each rollout calls into AReaL-SWEAgent, which runs the agent in a sandbox and
returns the reward.

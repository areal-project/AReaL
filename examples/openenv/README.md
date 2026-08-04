# OpenEnv RL Training

Train against any [OpenEnv](https://github.com/huggingface/OpenEnv) environment
(BrowserGym, OpenSpiel, coding sandboxes, chess, echo, ...) with GRPO. Switch
environments by editing YAML -- no new Python required.

## Layout

```
examples/openenv/
├── configs.py           # OpenEnvExperimentConfig(GRPOConfig)
├── train.py             # Entrypoint; instantiates OpenEnvWorkflow
├── echo_smoke.yaml      # 1-step sanity check against a public HF Space
├── blackjack_grpo.yaml  # Real GRPO on OpenSpiel BlackJack via UVProvider
└── README.md
```

## Prerequisites

Install the optional `openenv` extra plus your usual CUDA/inference extras:

```bash
uv sync --extra cuda --extra openenv
```

The Echo smoke config targets a public HuggingFace Space, so it needs no local env
install. The BlackJack config uses `UVProvider` (no Docker), which launches the
OpenSpiel env project via `uv run`.

## Run

**Smoke test (Echo, no Docker, no local env install):**

```bash
uv run python examples/openenv/train.py --config examples/openenv/echo_smoke.yaml
```

Expected: training starts, `train_metrics.jsonl` shows non-zero rewards within a few
steps.

**Learning curve (BlackJack):**

```bash
uv run python examples/openenv/train.py --config examples/openenv/blackjack_grpo.yaml
```

Expected on a single GPU: reward mean rises from ~0.5 (random policy) toward ~0.6 as the
policy learns "stand on 17+".

## Wire in a new environment

1. Find the environment's `EnvClient` subclass and Action dataclass (see the
   [OpenEnv envs directory](https://github.com/huggingface/OpenEnv/tree/main/envs)).
1. Duplicate one of the YAML files and edit the `openenv:` block:
   ```yaml
   openenv:
     env_client_class: coding_env.PythonCodeActEnv
     provider: uv
     project_path: git+https://huggingface.co/spaces/openenv/coding_env
     action_class: coding_env.CodeAction
     action_parser: tag          # json | tag | passthrough | dotted.path
     obs_formatter: auto         # auto | dotted.path
     system_prompt: |
       ...
     max_turns: 10
   ```
1. Adjust `system_prompt` / `action_parser` so your model's outputs match the Action
   schema.

If the built-in parsers/formatters don't cover your env, drop a Python file under this
directory and pass its dotted path in `action_parser` / `obs_formatter`.

## Design notes

`OpenEnvWorkflow.arun_episode`:

1. `env.reset(seed=...)` → initial `observation`.
1. Format `observation` into a user chat message; call LLM.
1. Parse the LLM output into an Action; call `env.step(action)`.
1. Record the step reward on the completion via `ArealOpenAI.set_reward`.
1. Repeat until `done` or `max_turns` is hit.
1. Optional: `apply_reward_discount(step_discount)` to bias credit backward.
1. Return `client.export_interactions("individual")` for GRPO grouping.

Each turn is its own trajectory row in the exported dict, so `GroupedRolloutWorkflow`
handles the group-relative advantage normalization identically to any other multi-turn
workflow.

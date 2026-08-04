# Training against OpenEnv environments

AReaL ships an `OpenEnvWorkflow` that speaks the
[HuggingFace OpenEnv](https://github.com/huggingface/OpenEnv) protocol. This lets you
train against any OpenEnv-compatible environment (BrowserGym, OpenSpiel, coding
sandboxes, chess, echo, ...) without writing a new workflow class per environment --
only YAML.

## Install

```bash
uv sync --extra cuda --extra openenv
```

Environments come from their own packages. Two common paths:

- **Remote HuggingFace Space** (zero local install): point `openenv.base_url` at the
  Space's public URL.
- **Local UV process** (no Docker): set `openenv.provider: uv` and
  `openenv.project_path` to a `git+https://...` spec or a local path.
- **Local Docker**: set `openenv.provider: docker` and `openenv.docker_image`.

## Minimum config

```yaml
openenv:
  # GenericEnvClient speaks the OpenEnv protocol by URL alone, so no
  # env-specific Python package is required. Point it at any running
  # OpenEnv-compatible server (a public HF Space here).
  env_client_class: openenv.core.generic_client.GenericEnvClient
  base_url: https://openenv-echo-env.hf.space  # xor: use provider/project_path below
  action_parser: json                          # json | tag | passthrough | dotted.path
  obs_formatter: auto                          # auto | dotted.path
  system_prompt: |
    Return a JSON object with keys "tool_name" and "arguments".
  max_turns: 4
  step_discount: 1.0                           # discount for step rewards; 1.0 disables
  terminal_reward_only: false                  # true = only keep last step's reward
```

Point `env_client_class` at an env-specific subclass (e.g. `echo_env.EchoEnv`)
only when you need the extra typing / action-class helpers it ships; that
subclass must be `pip install`-able in the workflow's environment.

A ready-to-run example lives at `examples/openenv/echo_smoke.yaml`.

## Action parsers

The workflow ships three parsers out of the box:

| shorthand     | expects                                     | returns                  |
| ------------- | ------------------------------------------- | ------------------------ |
| `json`        | last JSON object anywhere in the completion | `dict`                   |
| `tag`         | `<action>...</action>` block                | `dict` (if JSON) / `str` |
| `passthrough` | the raw completion string                   | `str`                    |

For custom logic, implement a callable with the signature

```python
def __call__(self, completion: str, observation: Any) -> Any | None: ...
```

and pass its dotted import path as `action_parser`.

`action_class` (optional) tells the workflow to expand a parser's `dict` output into
`ActionClass(**parsed)`. If the coercion fails (bad kwargs), the dict is passed through
unchanged.

## Observation formatters

`auto` (the default) JSON-encodes dataclasses / dicts / objects with a `__dict__`, and
passes strings through verbatim. For richer prompt shaping (e.g. rendering a chess board
as ASCII), implement a callable

```python
def __call__(self, observation: Any, step: int) -> dict[str, str]: ...
```

returning a `{"role": "user", "content": "..."}` message.

## Reward propagation

- By default each `env.step` reward is set on the corresponding LLM completion, so a
  GRPO group sees per-step advantages.
- `step_discount < 1.0` runs `ArealOpenAI.apply_reward_discount` after the episode ends,
  propagating credit backward geometrically.
- `terminal_reward_only: true` zeroes intermediate rewards and keeps only the final one
  -- match this to the environment's evaluation semantics.

## Add a new environment

1. Find the environment's client class (e.g. `browsergym_env.BrowserGymEnv`) and its
   Action dataclass.
1. Duplicate `examples/openenv/echo_smoke.yaml`, edit the `openenv:` block, and adjust
   `system_prompt` so your model's outputs are parseable.
1. Run:
   ```bash
   uv run python examples/openenv/train.py --config path/to/your_config.yaml
   ```

That's it -- no new Python required for the common case.

# AReaL Training Service CLI

`areal train` is the training subcommand group under the top-level `areal` CLI. It wires
up an AReaL training driver function and an experiment config file into a single
command. It does not manage the training process lifecycle (unlike `areal inf`, which
maintains service state); it only "finds the driver, loads the config, and passes
hydra-style overrides through."

## Basic concepts

The minimum execution unit of a training job is a **driver function** —typically a
`main(args: list[str])` in some script under `examples/`. The CLI does exactly three
things:

1. Resolve the driver from `module.path:func`
1. Resolve `--config <path>` to an absolute path and prepend it to argv
1. Append every trailing argument unchanged to argv (typically hydra overrides)

The driver function's return value is used as the exit code if it returns `int`;
anything else (including `None`) is treated as 0.

## Usage

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

Note that `run_cmd` is configured with
`context_settings={"ignore_unknown_options": True}` — this means trailing positional
args **can include `--xxx` flags**; the CLI does not try to parse them and forwards them
as-is to the driver.

## Examples

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

## Driver function conventions

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
hydra override merged into the config dataclass. In other words, hydra parsing is **done
by the driver**, not the CLI.

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

## Hydra overrides

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

## Exit codes

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

## Not implemented yet

`areal train` currently only implements `run`. The following are reasonable future
extensions but are **not** in this version:

- `areal train ps` / `status` / `stop` — lifecycle management for training jobs
  (requires a training service state concept first)

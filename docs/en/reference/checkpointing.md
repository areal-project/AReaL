# Checkpointing

This document describes AReaL's checkpointing system, which handles model saving for
evaluation and fault-tolerant recovery during distributed RL training.

## Overview

AReaL provides two complementary checkpointing mechanisms:

| Mechanism          | Purpose                                    | Format                       | Includes Optimizer/DataLoader State |
| ------------------ | ------------------------------------------ | ---------------------------- | ----------------------------------- |
| **Saver**          | Export models for evaluation or publishing | HuggingFace                  | No                                  |
| **RecoverHandler** | Resume training after failures             | DCP (Distributed Checkpoint) | Yes                                 |

Both mechanisms are invoked automatically during training and can be configured via
`config.saver` and `config.recover` respectively.

## Checkpoint Formats

### HuggingFace Format

Used by `Saver` for model export:

- Standard HuggingFace model format (safetensors + config.json)
- Compatible with `transformers.AutoModel.from_pretrained()`
- Can be uploaded to HuggingFace Hub
- Does not include optimizer state

### DCP Format (Distributed Checkpoint)

Used by `RecoverHandler` for fault tolerance:

- Backend's native distributed checkpoint format (`torch.distributed.checkpoint` or
  Megatron distributed checkpoint)
- Sharded across all ranks for efficient parallel I/O
- Includes model weights, optimizer state, RNG state, etc
- Backend-specific: checkpoints are only compatible with the same parallelism
  configuration
- Overwrites previous checkpoint to save disk space

## Architecture

```
PPOTrainer.train()
│
├── Training loop
│   ├── Rollout, compute values, PPO update...
│   │
│   ├── _save_hf()                          # HuggingFace export
│   │   └── Saver.save()
│   │       └── engine.save(weight_format="hf")
│   │
│   └── _save_recover_checkpoint()          # Fault tolerance
│       └── RecoverHandler.dump()
│           └── engine.save(weight_format="dcp", with_optim=True)
│
└── On restart
    └── RecoverHandler.load()
        ├── Restore dataloader, saver, evaluator states
        └── engine.load(weight_format="dcp", with_optim=True)
```

## Saver: HuggingFace Model Export

The [`Saver`](https://github.com/areal-project/AReaL/blob/main/areal/utils/saver.py)
periodically exports model weights in HuggingFace format for evaluation or deployment.

### Save Mode

The `mode` parameter controls how checkpoints are written:

| Mode    | Behavior                                                                                                                                                                                                                              |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `auto`  | Use async for Archon engine, sync for others (default). Zero-config optimal for all engines.                                                                                                                                          |
| `sync`  | Always synchronous `dcp.save()`.                                                                                                                                                                                                      |
| `async` | Always process-based async with pinned-memory staging. Archon engine only; other engines fall back to sync with a warning. Extra CPU pinned memory proportional to per-rank model shard size (e.g., ~17.5 GB/rank for 70B on 8 GPUs). |

With the default `auto` mode, Archon engine users get async checkpoint saving
automatically - the training loop blocks only while the checkpoint is staged to pinned
CPU memory, and the actual disk I/O happens in a background process.

### Configuration

Configure via `config.saver`:

| Parameter     | Type        | Default  | Description                          |
| ------------- | ----------- | -------- | ------------------------------------ |
| `mode`        | str         | `"auto"` | Save mode (see above).               |
| `freq_epochs` | int \| None | None     | Save every N epochs. None disables.  |
| `freq_steps`  | int \| None | None     | Save every N steps. None disables.   |
| `freq_secs`   | int \| None | None     | Save every N seconds. None disables. |

Example configuration:

```yaml
saver:
  freq_epochs: 1      # Save at end of each epoch
  freq_steps: null    # Disabled
  freq_secs: null     # Disabled
  # mode defaults to "auto" - Archon users get async automatically
```

Saving is triggered when any of epoch/step/time condition is met.

### Output Location

Checkpoints are saved to:

```
{fileroot}/checkpoints/{user}/{experiment_name}/{trial_name}/default/
└── epoch{E}epochstep{S}globalstep{G}/
    ├── config.json
    ├── model.safetensors (or model-00001-of-00002.safetensors, etc.)
    ├── tokenizer.json
    └── ...
```

### Usage

Load saved checkpoints with standard HuggingFace APIs:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/checkpoint/epoch0epochstep99globalstep99"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/path/to/checkpoint/epoch0epochstep99globalstep99"
)
```

## RecoverHandler: Fault Tolerance

The
[`RecoverHandler`](https://github.com/areal-project/AReaL/blob/main/areal/utils/recover.py)
enables resuming training after failures by saving complete training state.

### Configuration

Configure via `config.recover`:

| Parameter     | Type        | Default    | Description                                      |
| ------------- | ----------- | ---------- | ------------------------------------------------ |
| `mode`        | str         | "disabled" | Recovery mode: "on"/"auto" or "off"/"disabled"   |
| `freq_epochs` | int \| None | None       | Checkpoint every N epochs                        |
| `freq_steps`  | int \| None | None       | Checkpoint every N steps                         |
| `freq_secs`   | int \| None | None       | Checkpoint every N seconds                       |
| `retries`     | int         | 3          | Number of recovery retries when recovery enabled |

#### Recovery Modes

| Mode                | Behavior                                        |
| ------------------- | ----------------------------------------------- |
| `on` or `auto`      | Automatically resume if valid checkpoint exists |
| `off` or `disabled` | No checkpointing or recovery                    |

When recovery is enabled (`on`/`auto`), the system will:

1. Periodically save recovery checkpoints (model weights, optimizer state, dataloader
   position)
1. Automatically resume from the last valid checkpoint on restart
1. Retry up to `retries` times on failure

Example configuration:

```yaml
recover:
  mode: on            # or "auto" for backward compatibility
  freq_steps: 100     # Checkpoint every 100 steps
  retries: 3
```

### What Gets Saved

RecoverHandler saves complete training state:

| Component         | Contents                                           |
| ----------------- | -------------------------------------------------- |
| Model weights     | DCP format, sharded across ranks                   |
| Optimizer state   | Momentum, variance (Adam), learning rate scheduler |
| RNG state         | Python, NumPy, PyTorch, CUDA random states         |
| Dataloader state  | Current position in dataset                        |
| Training progress | Epoch, step, global_step counters                  |
| Auxiliary states  | Saver, Evaluator, StatsLogger states               |

### Output Location

Recovery checkpoints are saved to:

```
{fileroot}/checkpoints/{user}/{experiment_name}/{trial_name}/
├── default/
│   └── recover_checkpoint/     # Model + optimizer (DCP format)
│       ├── __0_0.distcp
│       ├── __1_0.distcp
│       └── ...
├── critic/                     # If using critic
│   └── recover_checkpoint/
└── recover_info/               # Metadata
    ├── step_info.json
    ├── saver_info.json
    ├── evaluator_info.json
    ├── stats_logger_info.json
    ├── checkpoint_info.json
    └── dataloader_info.pkl
```

### Recovery Process

When training resumes:

1. `RecoverHandler.load()` restores all saved state (if any)
1. Training continues from `last_step_info.next().global_step`
1. Inference engine weights are synchronized to match recovered state

## Best Practices

### Frequency Guidelines

| Scenario           | Recommended Setting                       |
| ------------------ | ----------------------------------------- |
| Long training runs | `freq_epochs: 1` or `freq_steps: 1000`    |
| Unpredictable time | `freq_secs: 7200`                         |
| Unstable clusters  | `freq_steps: 100` with `recover.mode: on` |
| Limited disk space | Lower frequency, rely on final checkpoint |
| Debugging          | `freq_steps: 1` for quick iteration       |

### Disk Space Considerations

- **Saver**: Each save creates a new directory. High frequency consumes significant
  space.
- **RecoverHandler**: Overwrites previous checkpoint. Only one copy exists at a time.

### Recovery Tips

1. **Verify checkpoint validity**: Check `recover_info/step_info.json` for the last
   saved step
1. **Same config required**: DCP checkpoints require identical parallelism
   configuration, experiment name, and trial name
1. **Clean restart**: Delete `recover_info/` directory to start fresh

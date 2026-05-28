# Code Style Rules

Rules beyond pre-commit (ruff format/lint).

## Design Patterns

- **Prefer composition over inheritance**: Avoid deep class hierarchies
  - Good: `Engine` holds a `Checkpointer` instance
  - Avoid: `CheckpointableEngine(Engine)` →
    `FSDPCheckpointableEngine(CheckpointableEngine)`
- Keep inheritance shallow (≤2 levels when possible)
- Use mixins sparingly; prefer explicit delegation

## Logging

- Use `areal.utils.logging.getLogger(name)` with PascalCase descriptive name, NOT
  `print` or stdlib `logging`
  - Good: `getLogger("RLVRWorkflow")`, `getLogger("ArchonEngine")`,
    `getLogger("GSM8KReward")`
  - Avoid: `getLogger(__name__)` or dotted paths like `getLogger("areal.engine.fsdp")`
- For per-rank loggers: `[{Component} Rank {N}]` (e.g., `[FSDPEngine Rank 0]`)
- Log levels:
  - DEBUG: Detailed tracing (avoid in hot paths)
  - INFO: Milestones (training start, checkpoint saved)
  - WARNING: Recoverable issues
  - ERROR: Failures requiring attention
- Register new loggers in `areal/utils/logging.py` (`LOGGER_COLORS_EXACT` or
  `LOGGER_PATTERNS`)
- Test colors: `python -m areal.utils.logging`

**Color Scheme:**

| Color      | Category                               | Examples                           |
| ---------- | -------------------------------------- | ---------------------------------- |
| blue       | Infrastructure (Schedulers, Launchers) | `LocalScheduler`, `RayLauncher`    |
| white      | Orchestration (Controllers, RPC)       | `TrainController`, `SGLangWrapper` |
| purple     | RL-specific (Workflows, Rewards)       | `RLVRWorkflow`, `GSM8KReward`      |
| green      | Data/Metrics (Stats, Dataset)          | `StatsLogger`, `RLTrainer`         |
| cyan       | Compute backends (Engines, Platforms)  | `FSDPEngine`, `CUDAPlatform`       |
| yellow/red | Warning/Error levels (override)        | Any logger at WARNING+             |

## Performance Patterns

- **Avoid GPU-CPU sync**: `.item()`, `.tolist()`, `print(tensor)` cause sync
- **Prefer batch operations**: Avoid Python loops over tensor elements
- **In-place ops**: Use when safe, but careful with autograd (`.add_()` vs `+`)

## Naming Conventions

| Type             | Pattern       | Example                             |
| ---------------- | ------------- | ----------------------------------- |
| Config dataclass | `XxxConfig`   | `GRPOConfig`, `FSDPConfig`          |
| Engine class     | `XxxEngine`   | `FSDPEngine`, `ArchonEngine`        |
| Workflow class   | `XxxWorkflow` | `RLVRWorkflow`, `MultiTurnWorkflow` |
| Reward function  | `xxx_reward`  | `math_reward`, `code_reward`        |

## Tensor Conventions

- Shape convention: `[batch, seq_len, hidden]` or document clearly
- Use `torch.Size` assertions for shape validation in debug
- Prefer explicit dtype/device over implicit conversion

## Import Style

- Group: stdlib, third-party, areal (ruff handles order)
- Avoid `from x import *` (CLAUDE.md rule)
- Prefer explicit imports over module-level imports for large modules

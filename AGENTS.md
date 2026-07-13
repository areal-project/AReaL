<!-- Go-to brief for AI coding agents working on AReaL. -->

# AGENTS.md -- AReaL Agent Operations Guide

## Quick reference

**Tech stack**: Python 3.12+ | PyTorch | FSDP2 / Megatron / Archon | SGLang / vLLM

```bash
# Environment
uv sync --extra cuda            # CUDA + SGLang inference (default); for vLLM: cp pyproject.vllm.toml pyproject.toml && cp uv.vllm.lock uv.lock && uv sync --extra cuda
uv sync --extra sandbox         # Daytona cloud sandbox backend (optional)
source .venv/bin/activate        # activate venv BEFORE pre-commit or git commit if venv exists
pre-commit install --install-hooks  # hooks: Ruff, clang-format, mdformat, nbstripout, conventional-commits
pre-commit run --all-files       # lint + format everything

# Tests
uv run pytest tests/test_<topic>.py

# CLI docs
uv run python docs/generate_cli_docs.py

# Docs build (canonical, release-aligned)
./docs/build_all.sh
# Do NOT use `jupyter-book build docs/en|docs/zh` directly for final preview/release,
# because it skips AReaL-specific static setup and output packaging.
```

**Hard rules** -- never violate:

- No wildcard imports (`from x import *`).
- No hardcoded secrets, paths, or endpoints.
- No skipping pre-commit hooks.
- No guessing cluster configs or rebuilding CUDA/driver stacks.
- Integration tests require multi-node hardware -- explain skips explicitly.

**Always do**:

- Read relevant files before modifying code.
- Run `pre-commit run --all-files` before committing.
- Follow existing code patterns in the same module.
- Add tests for new functionality.
- Ask for decisions and clarifications with short, structured options instead of broad
  open-ended questions. Use the platform's native question/clarification tool if
  available.

**Ask first** before:

- Modifying config structures in `areal/api/cli_args.py`.
- Adding new dependencies.
- Changing launcher or scheduler logic.
- Deleting or renaming public APIs.

When unsure, leave a `TODO(agent)` comment and note the constraint in your response.

______________________________________________________________________

## Repository map

```
areal/                     Core Python package
|-- api/                   Config dataclasses, contracts, IO structs
|-- dataset/               Stateful dataset loaders (GSM8K, Geometry3K, CLEVR, ...)
|-- engine/                Training backends (FSDP2, Megatron) + inference adapters
|-- experimental/          Prototype engines/workflows (Archon MoE engine)
|-- infra/                 Launchers (Local/Ray/Slurm), schedulers, utilities
|-- models/                Model adapters (Megatron-Core, Transformers, custom heads)
|-- reward/                Built-in reward functions + math parsers
|-- tests/                 Unit/integration test suites
|-- trainer/               High-level orchestrators (PPOTrainer, SFTTrainer)
|-- utils/                 Cross-cutting helpers (logging, data, checkpoints, RL ops)
+-- workflow/              RolloutWorkflow implementations (RLVR, multi-turn, vision)

docs/                      Jupyter Book docs (https://areal-project.github.io/AReaL/)
examples/                  Training scripts and launcher recipes
```

______________________________________________________________________

## Code style & patterns

- **Composition over inheritance** -- keep hierarchies \<= 2 levels; prefer delegation.

| Type             | Pattern         | Example                                   |
| ---------------- | --------------- | ----------------------------------------- |
| Config dataclass | `XxxConfig`     | `GRPOConfig`, `FSDPEngineConfig`          |
| Engine class     | `XxxEngine`     | `FSDPEngine`, `ArchonEngine`              |
| Workflow class   | `XxxWorkflow`   | `RLVRWorkflow`, `MultiTurnWorkflow`       |
| Reward function  | `xxx_reward_fn` | `gsm8k_reward_fn`, `geometry3k_reward_fn` |

**Logging**: `areal.utils.logging.getLogger(name)` with **PascalCase** names -- never
`print` or `logging.__name__`. Per-rank format: `[{Component} Rank {N}]`. Register new
loggers with color in `areal/utils/logging.py`.

**Performance**:

- No GPU-CPU sync in hot paths (`.item()`, `.tolist()`, `print(tensor)`).
- Batch ops over Python loops on tensor elements.
- Explicit `dtype`/`device`; `torch.Size` assertions for shape validation.

**Typing & imports**: explicit type hints; reuse `areal/api/cli_args.py` dataclasses; no
wildcard imports; heavy optional deps inside functions.

**Async**: rollout workflows must stay non-blocking (`await` + `aiofiles`); no sync I/O
in `arun_episode`.

______________________________________________________________________

## Domain experts & skills

Fire the appropriate **expert subagent** or **load a skill** based on what you're
working on. Experts are read-only consultants with deep domain knowledge; skills are
step-by-step implementation guides.

| Working on...                | Fire subagent      | Load skill          |
| ---------------------------- | ------------------ | ------------------- |
| FSDP engine code             | `fsdp-expert`      | --                  |
| Archon engine / new model    | `archon-expert`    | `add-archon-model`  |
| Megatron engine code         | `megatron-expert`  | --                  |
| RL algorithms / PPO / GRPO   | `algorithm-expert` | --                  |
| Launcher / scheduler / infra | `launcher-expert`  | `debug-distributed` |
| New reward function          | --                 | `add-reward`        |
| New dataset loader           | --                 | `add-dataset`       |
| New rollout workflow         | --                 | `add-workflow`      |
| Unit tests                   | --                 | `add-unit-tests`    |
| Distributed debugging        | --                 | `debug-distributed` |

**How to invoke experts and skills** (platform-specific):

| Platform | Fire expert subagent                                                               | Load skill                                         |
| -------- | ---------------------------------------------------------------------------------- | -------------------------------------------------- |
| OpenCode | `task(subagent_type="<name>", load_skills=[], run_in_background=true, prompt="…")` | `skill(name="<name>")` or `load_skills=["<name>"]` |
| Codex    | Invoke registered subagent by canonical name (see `.codex/config.toml`)            | Reference `.agents/skills/<name>/SKILL.md`         |

**Harness layout**:

| Component         | OpenCode                                | Codex                                                  |
| ----------------- | --------------------------------------- | ------------------------------------------------------ |
| Root instructions | `AGENTS.md`                             | `AGENTS.md`                                            |
| Agent configs     | `.opencode/agents/*.md` (frontmatter)   | `.codex/config.toml` + `.codex/agents/*.toml` + `*.md` |
| Skills            | `.opencode/skills/` + `.agents/skills/` | `.agents/skills/<name>/SKILL.md`                       |

Directly executable workflows (both platforms): `add-workflow`, `review-pr`,
`create-pr`, `translate-doc-zh`.

______________________________________________________________________

## Core concepts

**Trainer** orchestrator (`areal/trainer/`, `PPOTrainer`, `SFTTrainer`): manages the
training loop, dataset loading, and workflow execution. Entry point:
`examples/math/gsm8k_rl.py`.

**Rollout workflows** (`areal/workflow/`, `RolloutWorkflow.arun_episode`): define how
episodes are generated. Use `add-workflow` skill for step-by-step guide.

**Engines**: *Inference engines* handle async generation via `engine.agenerate()` and
manage weight updates. *Training engines* consume rollout tensors, compute PPO/GRPO
updates, and broadcast weight versions (FSDP2, Megatron, or Archon).

**Weight versioning**: async workflows require version alignment via `WeightUpdateMeta`
(`areal/api/engine_api.py`). Critical for correctness across distributed training.

**Observability**: emit metrics via `stats_tracker.get()`, persist artifacts under
`dump_dir`, checkpoint via `areal/utils/saver.py` / `recover.py`.

**Launcher / scheduler**: training requires cluster setup (local / Ray / Slurm) via
configs in `areal/infra/launcher/`. See `launcher-expert` for deployment guidance.

______________________________________________________________________

## API & config rules

*Applies to: `areal/api/**`*

- **Field ordering**: required -> common optional -> rare optional -> internal (`_`
  prefix).
- **Validation**: `__post_init__` with `ValueError` and clear message.
- **Backward compat**: add fields with defaults; deprecate before removing; avoid type
  changes.
- **CLI**: use `Literal` for enum choices; all public configs need docstrings with
  constraints.

______________________________________________________________________

## Distributed code rules

*Applies to: `areal/engine/**`, `areal/experimental/**`*

- Never create global process groups at module level; always pass `process_group`
  explicitly.
- `dist.get_rank(group)` not `dist.get_rank()` when group matters.
- DeviceMesh dimensions must match `ArchonParallelDims`: `dp_shard`, `tp`, `cp`, `ep`,
  `etp`.
- All-reduce: all ranks must call. Broadcast: explicit `src`. Barrier: debugging only.

| Issue         | Cause                            | Fix                       |
| ------------- | -------------------------------- | ------------------------- |
| Hang          | Mismatched collective calls      | All ranks call same op    |
| Wrong results | Incorrect `ReduceOp`             | Check SUM vs MEAN         |
| OOM           | Unsharded tensor on wrong device | Verify DTensor placements |

Debug env vars: `TORCH_DISTRIBUTED_DEBUG=DETAIL`, `NCCL_DEBUG=INFO`,
`CUDA_LAUNCH_BLOCKING=1`. See the `debug-distributed` skill for the full workflow.

______________________________________________________________________

## Testing rules

*Applies to: `**/tests/**`, `test_*.py`*

| Marker                                  | When                             |
| --------------------------------------- | -------------------------------- |
| `@pytest.mark.slow`                     | > 10s (excluded from default CI) |
| `@pytest.mark.slow` + `@pytest.mark.ci` | Slow but must run in CI          |
| `@pytest.mark.asyncio`                  | Async tests                      |

- Naming: `test_<what>_<condition>_<expected>()` with Arrange/Act/Assert.
- GPU: skip gracefully (`@pytest.mark.skipif(not CUDA_AVAILABLE, reason="...")`).
- Distributed mocking: `torch.distributed.fake_pg`; don't mock FSDP/DTensor internals.
- Assertions: `torch.testing.assert_close()` with explicit `rtol`/`atol`; prefer
  `tmp_path`, `monkeypatch`.

| Suite       | Command                       | GPU       |
| ----------- | ----------------------------- | --------- |
| Unit        | `pytest tests/test_*.py`      | No        |
| GRPO        | `pytest tests/grpo/`          | Yes       |
| FSDP        | `pytest tests/test_fsdp_*.py` | Yes       |
| Distributed | `pytest tests/torchrun/`      | Multi-GPU |

______________________________________________________________________

## Collaboration & review

- **Branches**: kebab-case (`feature/multi-turn-metrics`, `bugfix/fsdp-weight-sync`).
- **Commits**: Conventional Commits (e.g., `feat:`, `fix:`, `docs:`, `gov:`), ~72 char
  subject, imperative voice. Squash WIP before PR.
- **Pre-merge**: full pre-commit stack; doc-only edits need at least `mdformat --check`.
- **PRs**: tie to issue, highlight risk areas, list test commands executed, note skipped
  suites with reasons.

| Skill                | Purpose                                                |
| -------------------- | ------------------------------------------------------ |
| `create-pr`          | Rebase, squash, and create or update a PR              |
| `commit-conventions` | Commit message conventions to load before `git commit` |
| `review-pr`          | Dynamic PR review with targeted expert consultation    |
| `translate-doc-zh`   | Translate English docs to Chinese                      |

______________________________________________________________________

## Reference material

- **Docs portal**: <https://areal-project.github.io/AReaL/>
- **Quickstart**: `docs/tutorial/quickstart.md`
- **Architecture**: `docs/tutorial/gsm8k_grpo.md`
- **Customization**: `docs/customization/*.md`
- **Algorithms**: `docs/algorithms/*.md`
- **Best practices**: `docs/best_practices/*.md`
- **CLI reference**: `docs/cli_reference.md`
- **Agent workflow**: `docs/customization/agent.md`

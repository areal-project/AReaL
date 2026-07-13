# AI-Assisted Development

AReaL ships with first-class configurations for
[Codex](https://developers.openai.com/codex), [OpenCode](https://opencode.ai/), and
[Claude Code](https://docs.anthropic.com/en/docs/claude-code). The repository includes
shared project instructions, domain expert agents, and reusable implementation skills so
the three harnesses stay aligned.

This page treats Codex as the canonical harness for repo-local workflows. OpenCode and
Claude Code remain supported compatibility layers.

## Codex

### Start a Codex session

```bash
cd /path/to/AReaL
codex
```

Codex reads project context from `AGENTS.md`. In this repository, the Codex harness is
organized around:

- **Project instructions**: `AGENTS.md`
- **Repo-local skills**: `.agents/skills/<name>/SKILL.md`
- **Custom agents**: `.codex/config.toml` plus `.codex/agents/<name>.toml` and
  `.codex/agents/<name>.md`

Custom agents are registered in `.codex/config.toml` and point at per-agent TOML files:

```toml
[agents.fsdp-expert]
description = "FSDP2 expert for configuration, parallel strategy, weight sync, and integration guidance."
config_file = "./agents/fsdp-expert.toml"
```

Each agent TOML then points at its markdown instructions file:

```toml
name = "fsdp-expert"
model_instructions_file = "./fsdp-expert.md"
```

### Directly executable Codex workflows

In AReaL, the directly executable Codex workflows are **repo-local skills** under
`.agents/skills/`. Codex does not use `/` commands here as its primary workflow surface.
Instead, you ask Codex to use the matching skill and carry out the workflow end to end.

| Skill                   | Purpose                                                            |
| ----------------------- | ------------------------------------------------------------------ |
| `add-dataset`           | Add a new dataset loader under `areal/dataset/`                    |
| `add-workflow`          | Create a new `RolloutWorkflow` implementation                      |
| `add-reward`            | Implement a new reward function                                    |
| `add-archon-model`      | Add a new model architecture to the Archon engine                  |
| `add-unit-tests`        | Add or extend unit tests                                           |
| `debug-distributed`     | Troubleshoot hangs, OOMs, NCCL issues, and launcher misconfig      |
| `commit-conventions`    | Prepare a Conventional Commit message before committing            |
| `review-pr`             | Run the read-only PR review workflow with risk analysis            |
| `create-pr`             | Rebase, squash, prepare metadata, and create or update a GitHub PR |
| `translate-doc-zh`      | Translate `docs/en/` content into `docs/zh/`                       |
| `update-docker-image`   | Update runtime image dependencies and drive the Docker PR flow     |
| `upgrade-megatron-core` | Audit and upgrade Megatron-Core compatibility                      |
| `upgrade-vllm`          | Audit and upgrade vLLM compatibility                               |

These skills are the answer to "what is Codex's directly executable workflow in this
repository?".

### Custom Codex agents

Codex custom agents are specialized consultants registered in `.codex/config.toml`.
Unlike OpenCode's automatic expert routing, Codex agents are an explicit tool surface:
ask Codex to consult the relevant agent, or rely on repository instructions that tell it
when to do so.

| Agent                  | Purpose                                                          |
| ---------------------- | ---------------------------------------------------------------- |
| `planner`              | Plan multi-file work and architectural changes                   |
| `simple-code-reviewer` | Perform quick post-change risk checks                            |
| `code-verifier`        | Run targeted formatting, lint, and verification commands         |
| `fsdp-expert`          | FSDP2 configuration, parallel strategy, and weight sync guidance |
| `archon-expert`        | Archon and MoE integration guidance                              |
| `megatron-expert`      | Megatron pipeline parallel training guidance                     |
| `algorithm-expert`     | PPO, GRPO, DAPO, reward shaping, and RL workflow guidance        |
| `launcher-expert`      | Local, Ray, Slurm, and inference launcher guidance               |

### Typical Codex sessions

```text
> Add a new rollout workflow for multimodal evaluation

Codex: [loads add-workflow, inspects areal/workflow/, implements changes]

> Review this PR

Codex: [loads review-pr, analyzes the diff, consults the matching expert agents]

> Create or update the PR

Codex: [loads create-pr, checks branch state, prepares the PR workflow]
```

## OpenCode Compatibility

### Install OpenCode and oh-my-opencode

```bash
curl -fsSL https://opencode.ai/install | bash
```

Alternative methods: `brew install anomalyco/tap/opencode`,
`npm install -g opencode-ai`, or a release binary from GitHub. See the OpenCode docs for
current install details.

[oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode) remains optional, but
useful if you want richer OpenCode orchestration and tool integrations.

### Start OpenCode

```bash
cd /path/to/AReaL
opencode
```

OpenCode reads `AGENTS.md` and discovers agents, commands, skills, and plugins from
`.opencode/`. It can also read `.claude/skills/` where relevant.

### OpenCode workflow surface

OpenCode keeps its own command and agent system:

| Command             | Purpose                                    |
| ------------------- | ------------------------------------------ |
| `/create-pr`        | Rebase, squash commits, and create PR      |
| `/review-pr`        | Intelligent code review with risk analysis |
| `/translate-doc-zh` | Translate English documentation to Chinese |

OpenCode-specific assets live in `.opencode/agents/`, `.opencode/command/`,
`.opencode/skills/`, and `.opencode/package.json`.

## Claude Code Compatibility

Claude Code remains supported through `CLAUDE.md` plus the `.claude/` directory for
agents, commands, hooks, and rules.

## Configuration Files

```
AReaL/
|-- AGENTS.md                # Project context (loaded automatically)
|-- .agents/
|   +-- skills/              # Codex repo-local skills
|-- .codex/
|   |-- config.toml          # Registers custom Codex subagents
|   +-- agents/              # Codex agent config files and instruction markdown
|-- .opencode/
|   |-- agents/              # OpenCode expert agents
|   |-- command/             # OpenCode slash commands
|   |-- skills/              # OpenCode skills
|   +-- package.json         # OpenCode plugin dependencies
+-- .claude/
    |-- agents/              # Claude Code agents
    |-- commands/            # Claude Code commands
    |-- hooks/               # Claude Code hooks
    +-- rules/               # Claude Code rules
```

## Harness Comparison

AReaL supports all three harnesses, but each one exposes reusable workflows differently:

| Concept                          | Codex                                   | OpenCode                                  | Claude Code                                       |
| -------------------------------- | --------------------------------------- | ----------------------------------------- | ------------------------------------------------- |
| Project context                  | `AGENTS.md`                             | `AGENTS.md`                               | `CLAUDE.md`                                       |
| Repo workflows                   | `.agents/skills/`                       | `.opencode/skills/`, `.opencode/command/` | `.claude/skills/`, `.claude/commands/`            |
| Custom subagents                 | `.codex/config.toml` + `.codex/agents/` | `.opencode/agents/`                       | `.claude/agents/`                                 |
| Primary executable workflow form | Repo-local skill                        | Slash command or skill                    | Slash command or skill                            |
| Agent dispatch                   | Explicit custom agent invocation        | `task(subagent_type="...", ...)`          | Automatic routing                                 |
| Expert names                     | `fsdp-expert`, `archon-expert`, ...     | `fsdp-expert`, `archon-expert`, ...       | `fsdp-engine-expert`, `archon-engine-expert`, ... |
| Commit helper                    | `commit-conventions` skill              | `commit-conventions` skill                | `commit-conventions` skill + `/gen-commit-msg`    |

Claude Code also ships with additional general-purpose agents:

| Agent                  | Purpose                                                        |
| ---------------------- | -------------------------------------------------------------- |
| `planner`              | Creates implementation plans before complex multi-file changes |
| `code-verifier`        | Runs pre-commit hooks and tests after code changes             |
| `simple-code-reviewer` | Performs quick code quality checks before commits              |

Claude Code configuration lives in:

```
AReaL/
|-- CLAUDE.md              # Project context and constraints
+-- .claude/
    |-- agents/            # 5 domain experts + 3 general-purpose (8 total)
    |-- skills/            # Guided workflows (shared with OpenCode)
    |-- commands/          # Automated actions (create-pr, gen-commit-msg, review-pr)
    |-- hooks/             # Pre/post action hooks
    +-- rules/             # Code quality standards
```

## Contributing

We welcome contributions to both the codebase and AI development configurations:

- **Code contributions**: New features, bug fixes, documentation improvements
- **AI config contributions**: New Codex skills, custom agents, or compatibility-layer
  improvements
- **Codex configs**: Edit `.codex/`, `.agents/`, and `AGENTS.md`
- **OpenCode configs**: Edit files in `.opencode/`
- **Claude Code configs**: Edit files in `.claude/`

See [CONTRIBUTING.md](https://github.com/areal-project/AReaL/blob/main/CONTRIBUTING.md)
for guidelines.

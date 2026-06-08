---
name: commit-conventions
description: AReaL commit message conventions. MUST load on every git commit -- provides Conventional Commits format with scope inference from file paths.
---

# Commit Conventions

Commit message conventions and scope inference rules for the AReaL repository.

## When to Use

**ALWAYS load this skill when making any git commit in AReaL.** This includes:

- Direct commits (`git commit`)
- Commits during PR creation (`/create-pr`)
- Commits delegated via `task(load_skills=["commit-conventions"], ...)`
- Any agent workflow that produces a commit

## Commit Message Format

```
<type>(<scope>): <subject>

<body>

[Optional sections:]
Key changes:
- change 1
- change 2

Refs: #123, #456
```

## Type Selection

| Type       | When to Use                      |
| ---------- | -------------------------------- |
| `feat`     | New feature or capability        |
| `fix`      | Bug fix                          |
| `docs`     | Documentation only               |
| `gov`      | Governance or maintainer changes |
| `style`    | Formatting/style-only changes    |
| `refactor` | Code change without feature/fix  |
| `perf`     | Performance improvement          |
| `test`     | Adding or fixing tests           |
| `build`    | Build system or dependencies     |
| `ci`       | CI pipeline or workflow changes  |
| `chore`    | Build, deps, config changes      |
| `revert`   | Revert a previous commit         |

## Scope Inference

Infer scope from the **primary** changed file paths:

| File Path Pattern                                            | Scope                          |
| ------------------------------------------------------------ | ------------------------------ |
| `areal/workflow/`                                            | `workflow`                     |
| `areal/engine/`                                              | `engine`                       |
| `areal/reward/`                                              | `reward`                       |
| `areal/dataset/`                                             | `dataset`                      |
| `areal/api/`                                                 | `api`                          |
| `areal/utils/`                                               | `utils`                        |
| `areal/infra/`                                               | `infra`                        |
| `areal/trainer/`                                             | `trainer`                      |
| `areal/models/`                                              | `models`                       |
| `areal/experimental/`                                        | `archon`                       |
| `docs/`                                                      | `docs`                         |
| `examples/`                                                  | `examples`                     |
| `AGENTS.md`, `.agents/`, `.claude/`, `.codex/`, `.opencode/` | `agents`                       |
| Multiple areas                                               | Omit scope or use broader term |

## Rules

- **Subject**: imperative mood, ~50-72 chars, no trailing period
- **Body**: explain "why" not "what", wrap at 72 chars
- **Key changes**: bullet list of main modifications (for complex commits with 3+ files)
- **Refs**: reference issues/PRs if applicable

## Examples

**Single file fix:**

```
fix(reward): handle empty completion in gsm8k

Return 0 reward instead of raising exception when
completion string is empty after extraction.
```

**Multi-file feature:**

```
feat(engine): add CPU offload support to ArchonEngine

Enable torch_memory_saver for model offloading during
rollout phase to reduce GPU memory pressure.

Key changes:
- Add offload/onload methods to ArchonEngine
- Integrate with weight update flow
- Handle ROCm compatibility
```

**Docs only:**

```
docs: update algorithm comparison table

Add SAPO and GSPO to the algorithm family documentation
with configuration examples.
```

**Agent/tooling changes:**

```
chore(agents): port review-pr command to OpenCode

Add OpenCode-native commands with task() category
delegation instead of hardcoded model names.

Key changes:
- Create .opencode/command/ with review-pr, create-pr
- Replace hardcoded model routing with platform-native review routing
- Add expert subagent consultation patterns
```

**Governance/maintainer changes:**

```
gov(agents): add maintainer ownership for service modules

Update CODEOWNERS and maintainer references to reflect
current governance responsibilities.

Key changes:
- Add maintainers for agent_service and infra ownership
- Align governance docs with updated reviewer responsibilities
```

______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .opencode/skills/commit-conventions/SKILL.md
Invocation: Automatically loaded on every git commit via load_skills

## Purpose

Provides Conventional Commits format with AReaL-specific scope inference
from file paths. Unlike other skills, this one is NOT user-triggered --
it is loaded by the system/agent on every commit operation.

## How to Update

### When New Modules Are Added
1. Add the file path pattern and scope to the "Scope Inference" table
2. Keep table sorted by areal/ subpackages first, then top-level dirs

### When Commit Types Change
1. Update the "Type Selection" table
2. Add/update examples to illustrate the new type

### When Adding Examples
1. Each example should demonstrate a distinct commit pattern
2. Keep examples realistic -- use actual AReaL module names
3. Show both subject-only and subject+body+key-changes variants

### Important Design Decisions
- This skill is ALWAYS loaded (not optional) -- keep it concise to
  minimize token overhead on every commit
- Scope inference is path-based, not content-based -- simpler and
  more deterministic
- "Multiple areas" -> omit scope rather than invent a new one

================================================================================
-->

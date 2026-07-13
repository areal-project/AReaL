---
name: gen-commit-msg
description: Generate intelligent commit messages based on staged changes. Invoke with /gen-commit-msg.
---

# Generate Commit Message

Generate a well-formatted commit message based on staged changes.

## Usage

```
/gen-commit-msg [--amend] [--scope <scope>]
```

**Arguments:**

- `--amend`: Amend the previous commit instead of creating new
- `--scope <scope>`: Force a specific scope (e.g., `workflow`, `engine`)

## Workflow

### Step 1: Analyze Changes

```bash
# Check staged files
git diff --cached --name-only

# Check staged content
git diff --cached

# Check recent commit style
git log --oneline -5
```

### Step 2: Categorize Changes

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

### Step 3: Determine Scope

Infer scope from changed files:

- `areal/workflow/` → `workflow`
- `areal/engine/` → `engine`
- `areal/reward/` → `reward`
- `areal/dataset/` → `dataset`
- `areal/api/` → `api`
- `docs/` → `docs`
- Multiple areas → omit scope or use broader term

### Step 4: Generate Message

**Format:**

```
<type>(<scope>): <subject>

<body>

[Optional sections:]
Key changes:
- change 1
- change 2

Refs: #123, #456
```

**Rules:**

- Subject: imperative mood, ~50-72 chars, no period
- Body: explain "why" not "what", wrap at 72 chars
- Key changes: bullet list of main modifications (for complex commits)
- Refs: reference issues/PRs if applicable

### Step 5: Confirm and Commit

Show preview:

```
─────────────────────────────────────
feat(workflow): add vision support to RLVR

Add VisionRLVRWorkflow for vision-language RL training.
Supports image inputs alongside text prompts.
─────────────────────────────────────
```

Ask user to confirm, then execute:

```bash
git commit -m "$(cat <<'EOF'
<message>
EOF
)"
```

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

______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .claude/commands/gen-commit-msg.md
Invocation: /gen-commit-msg

## Design Philosophy

- Automates commit message generation following Conventional Commits format
- Matches repository's existing style
- Requires user confirmation before commit

## How to Update

### Adding New Scopes
Update "Determine Scope" section with new file path mappings.

### Changing Format
Update "Generate Message" format template and rules.

================================================================================
-->

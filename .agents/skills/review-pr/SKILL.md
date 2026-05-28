---
name: review-pr
description: Read-only pull request review workflow with risk analysis, targeted checklists, and Codex subagent consultation.
---

# Review Pull Request

Use this skill when the user asks for a PR review of the current branch or a specific
PR.

## Inputs

- Optional PR number
- Optional `--quick` to stop after the change analysis phase

## Hard Rules

- Stay read-only.
- Do not edit files, commit, push, rebase, or change GitHub state.
- Do not run build, install, or test commands that mutate the environment.
- Use `gh` for PR metadata and git diff retrieval.

## Reference Files

- `references/review-pr-domains-and-signals.md`
- `references/review-pr-templates.md`

## Workflow

### Phase 1: Resolve PR context

1. Use `gh pr view` to fetch PR title, body, state, draft status, and changed files.
1. If no PR exists, stop and report that clearly.
1. If the PR is closed, stop.
1. Record the branch name and changed file list.

### Phase 2: Change analysis

1. Classify changed files using `references/review-pr-domains-and-signals.md`.
1. Determine the highest overall risk level: `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW`.
1. Build a `CHANGE_ANALYSIS_REPORT` that lists:
   - detected domains/signals
   - risk level
   - affected files
   - related frameworks
   - likely failure modes

If `--quick` is set, return the change analysis report and stop here.

### Phase 3: Review planning

1. Select the smallest useful set of review passes from
   `references/review-pr-templates.md`.
1. Split by risk area, not by file count.
1. Always include at least one general logic pass.

### Phase 4: Expert consultation

Consult the matching Codex subagents registered in `.codex/config.toml` when relevant:

- `archon-expert`
- `fsdp-expert`
- `megatron-expert`
- `algorithm-expert`
- `launcher-expert`

If the Codex runtime supports parallel subagent execution, run independent review passes
in parallel. Otherwise, execute them serially.

### Phase 5: Final review

Produce findings first, ordered by severity:

1. `CRITICAL`
1. `HIGH`
1. `MEDIUM`
1. `LOW`

For every finding, include:

- file path
- line number when available
- why it is a bug, regression, or risk
- concrete fix direction

## What to Ignore

- Pure style nits with no correctness impact
- Issues outside the changed scope unless the PR makes them worse
- Failures that standard linters or formatters would already catch
- Speculative concerns with no concrete trigger in the diff

## Output Shape

Use this structure:

```markdown
CHANGE_ANALYSIS_REPORT:
- detected_domains: [...]
- detected_signals: [...]
- risk_level: ...
- affected_files: [...]
- related_frameworks: [...]
- identified_risks: [...]

Findings
1. [severity] Title — path:line
   - Problem: ...
   - Fix: ...

Open Questions
- ...

Residual Risk
- ...
```

---
name: review-pr
description: Intelligent PR code review with dynamic agent allocation based on domains and signals
allowed-tools: Read, Grep, Glob, Bash, Task
---

<!-- Reference data (auto-loaded via @import) -->

@.claude/data/review-pr-domains-and-signals.md @.claude/data/review-pr-templates.md

# PR Code Review (Dynamic Agent Allocation)

Intelligent code review for the current branch's Pull Request. Dynamically generates
targeted review tasks based on PR changes.

## Arguments

`$ARGUMENTS`

- No arguments: Review PR for current branch
- PR number: Review specific PR (e.g., `/review-pr 123`)
- `--quick`: Quick mode, only run Phase 1 analysis

## Quick Start

1. Get current branch PR: `gh pr view --json number,title,state,isDraft`
1. If PR doesn't exist or is closed, stop and explain
1. Execute Phases 1-4 in order

## Workflow Overview

```
Phase 1: Deep PR Analysis [Haiku + Sonnet]
    |- 1.0 PR Status Check [Haiku]
    |- 1.1 Get PR Summary [Haiku]
+- 1.2-1.4 Domain/Signal Detection [Sonnet]
    |
Phase 2: Dynamic Agent Planning [Sonnet]
    |
Phase 3: Execute Review Tasks [Parallel, Dynamic Model Selection]
    |
Phase 4: Confidence Scoring & Summary [Haiku]
```

## Model Configuration

| Mode                      | CRITICAL/HIGH | MEDIUM | LOW    |
| ------------------------- | ------------- | ------ | ------ |
| **Default**               | Opus          | Sonnet | Haiku  |
| **Quick** (`--quick`)     | Sonnet        | Sonnet | Sonnet |
| **Economy** (`--economy`) | Sonnet        | Haiku  | Haiku  |

______________________________________________________________________

## Phase 1: Deep PR Analysis

### 1.0 PR Status Check \[Haiku\]

Check if PR should be reviewed:

- Is it closed? -> Stop
- Is it a draft? -> Note but continue
- Is it bot-generated? -> Skip

### 1.1 Get PR Summary \[Haiku\]

Get basic PR info: title, description, modified files, change summary.

### 1.2 Domain & Signal Detection \[Sonnet\]

Analyze each file change, detecting L1 domains and L2 signals by risk level.

**Reference**: See `review-pr-domains-and-signals.md` for complete domain tables:

- L1 domains (Distributed Runtime, Model Compute & Attention, Inference Backend &
  Serving, etc.)
- L2 signals per domain
- cross-domain linkage rules

### 1.3 Domain-Specific Risk Identification

Based on detected domains/signals, identify corresponding risks and linked checks.

**Reference**: See `review-pr-domains-and-signals.md` for risk lists per domain.

### 1.4 Output Change Analysis Report

```
CHANGE_ANALYSIS_REPORT:
- detected_domains: [Distributed Runtime, Model Compute & Attention, ...]
- detected_signals: [weight_sync, tree_attn, ...]
- risk_level: CRITICAL | HIGH | MEDIUM | LOW
- affected_files: [file1.py, file2.py, ...]
- identified_risks: [risk1, risk2, ...]
- related_frameworks: [archon, fsdp, megatron, vllm, service-stack, ...]
```

______________________________________________________________________

## Phase 2: Dynamic Agent Planning \[Sonnet\]

### 2.1 Planning Principles

1. **Generate tasks by risk area**: Each high-risk area gets a dedicated task
1. **Merge related changes**: Interdependent changes can be merged
1. **Review depth selection**: CRITICAL/HIGH -> `comprehensive`, MEDIUM -> `targeted`,
   LOW -> `basic`
1. **Model routing**: `comprehensive` -> Opus, `targeted` -> Sonnet, `basic` -> Haiku
1. **Minimum coverage**: Even simple changes get at least 1 basic review task

### 2.2 Task Template Selection

Based on detected domains/signals, select appropriate review task templates.

**Reference**: See `review-pr-templates.md` for complete task templates:

- Domain templates (Distributed Runtime, Model Compute & Attention, Inference Backend &
  Serving, etc.)
- Universal + signal-specific add-on templates

### 2.3 Output Review Task List

```
GENERATED_REVIEW_TASKS:
1. [comprehensive -> Opus] Task Name
   - Reason: XXX domain/signal detected
   - Checklist: [...]
   - Focus files: [...]

2. [targeted -> Sonnet] Task Name
   - Reason: ...
   ...
```

______________________________________________________________________

## Phase 3: Execute Review Tasks \[Parallel\]

### 3.1 Execution Rules

- Use Phase 2 specified model for each task
- Execute all agents **in parallel**
- Each agent reviews independently

### 3.2 Agent Output Format

```
REVIEW_RESULT:
task_name: "Task Name"
model: Opus | Sonnet | Haiku
findings:
  - issue: "Issue description"
    severity: CRITICAL | HIGH | MEDIUM | LOW
    file: "path/to/file.py"
    line: 123
    code_snippet: |
      Relevant code snippet
    reason: "Why this is an issue"
    suggestion: "Fix suggestion"
```

### 3.3 Review Depth Mapping

| Review Depth      | Model  | Requirements                                                               |
| ----------------- | ------ | -------------------------------------------------------------------------- |
| **comprehensive** | Opus   | Complete context, cross-file traces, verify parallel strategy interactions |
| **targeted**      | Sonnet | Changed code + direct callers/callees, type signature consistency          |
| **basic**         | Haiku  | Format and basic correctness only                                          |

______________________________________________________________________

## Phase 4: Confidence Scoring & Summary \[Haiku\]

### 4.1 Confidence Scoring (0-100)

| Score   | Meaning                               |
| ------- | ------------------------------------- |
| **0**   | False positive or pre-existing issue  |
| **25**  | May be real, cannot verify            |
| **50**  | Real but minor or rare                |
| **75**  | Very likely real, important           |
| **100** | Confirmed real, will frequently occur |

### 4.2 Summary Report Format

```markdown
# PR Review Summary

## PR Overview
- **Title**: PR title
- **Detected Domains**: [...]
- **Detected Signals**: [...]
- **Risk Level**: CRITICAL | HIGH | MEDIUM | LOW
- **Generated Review Tasks**: N

## Executed Review Tasks
| # | Model | Task Name | Reason |
|---|-------|-----------|--------|

## Findings

### CRITICAL Severity (Confidence >= 75)
#### Issue 1: [Title]
- **File**: `path/to/file.py:123`
- **Confidence**: 85
- **Description**: ...
- **Fix Suggestion**: ...

### HIGH Severity (Confidence >= 50)
...

## Review Statistics
- Total issues: X (CRITICAL: X, HIGH: X, MEDIUM: X, LOW: X)
- Filtered false positives: X
```

### 4.3 Output Integrity (CRITICAL)

The Phase 4 summary report is the **FINAL DELIVERABLE** of this command.

- Output the COMPLETE report exactly as specified in Section 4.2 -- every section, every
  finding, every field.
- Do NOT abbreviate, summarize, or compress any part of the report.
- Do NOT omit findings, code snippets, fix suggestions, or statistics.
- If the report is long, that is expected and correct -- **completeness > brevity**.
- The orchestrating agent receiving this output MUST present it **VERBATIM** to the
  user. No re-summarization, no condensing, no "brief version".

______________________________________________________________________

## Dynamic Generation Examples

| PR Type        | Detected Domains/Signals                            | Generated Tasks                 |
| -------------- | --------------------------------------------------- | ------------------------------- |
| Docs only      | \[Low-Risk Hygiene / tests_docs_config\]            | 1 basic -> Haiku                |
| Config only    | \[API & Config Compatibility / dataclass_schema\]   | 1-2 basic/targeted              |
| Single bug fix | \[Numerics & Tensor Semantics / shape_dtype\]       | 2-4 targeted -> Sonnet          |
| Archon core    | \[Distributed Runtime / mesh_dtensor, weight_sync\] | 4-8 comprehensive -> Opus       |
| Cross-domain   | \[Workflow & Trainer + Distributed + Hygiene\]      | 5-10 mixed review depths/models |

______________________________________________________________________

## False Positive Guide (Rate Confidence 0)

- Pre-existing issues (not introduced by this PR)
- Intentionally designed code that looks like a bug
- Issues linter/compiler would catch
- Issues on lines user didn't modify
- Explicitly disabled issues (lint ignore comments)

______________________________________________________________________

## Important Notes

- **Do NOT** check build signals or try to build/type-check
- Use `gh` to interact with GitHub, not web fetch
- **Do NOT** automatically post comments to PR
- Must provide file path and line number when referencing issues

______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .claude/commands/review-pr.md
Invocation: /review-pr
Related files:
- .claude/data/review-pr-domains-and-signals.md: Domain and signal detection tables
  - .claude/data/review-pr-templates.md: Review task templates

## Structure

- Main file (this): workflow and phases, @imports data files
- data/review-pr-domains-and-signals.md: domain and signal detection tables
- data/review-pr-templates.md: task templates

## How to Update

### Adding New Domains or Signals
Edit `.agents/skills/review-pr/references/review-pr-domains-and-signals.md`, then regenerate
the derived data files with `python3 .agents/skills/review-pr/sync_review_pr_refs.py --write`.

### Adding New Task Templates
Edit `.agents/skills/review-pr/references/review-pr-templates.md`, then regenerate the
derived data files with `python3 .agents/skills/review-pr/sync_review_pr_refs.py --write`.

### Adjusting Model Selection
Modify "Model Configuration" table in this file.

================================================================================
-->

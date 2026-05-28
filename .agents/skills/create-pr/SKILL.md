---
name: create-pr
description: Rebase the current branch onto the latest base branch, squash local commits, generate a Conventional Commit message, and create or update the GitHub pull request.
---

# Create Pull Request

Use this skill when the user asks to create or update a PR for the current branch.

## Inputs

- Optional `--draft`
- Optional `--base <branch>` (default: `main`)

## Preconditions

1. Verify the current branch is not `main` or `master`.
1. Check for uncommitted changes with `git status --short`.
1. Ensure `gh` is available.
1. If there are uncommitted changes, stop and ask the user whether to commit or stash
   them first.

## Workflow

### Step 1: Check for an existing PR

- Run `gh pr view --json number,title,url,state,isDraft`.
- If a PR already exists, tell the user before rewriting history or force-pushing.

### Step 2: Fetch and rebase

```bash
git fetch origin <base>
git rebase origin/<base>
```

- If rebase conflicts occur, abort the rebase and stop.
- Tell the user which files conflicted and ask them to resolve manually.

### Step 3: Squash into one commit

```bash
git reset --soft origin/<base>
```

- Load the `commit-conventions` skill before generating the commit message.
- Infer `type` and `scope` from the staged diff.
- Keep the commit subject imperative and under about 72 characters.

### Step 4: Generate PR title and body

- Use the squashed commit message style for the PR title.
- Follow the repository PR template at `.github/PULL_REQUEST_TEMPLATE.md`.
- Summarize user-facing changes, risk areas, test commands run, and skipped suites with
  reasons.

### Step 5: Push and create or update the PR

- Push the branch.
- If history was rewritten, confirm before force-pushing.
- Create or update the PR with `gh pr create` or `gh pr edit`.
- Respect `--draft` when requested.

## Guardrails

- Never create a PR from `main` or `master`.
- Never silently force-push over an existing PR branch.
- Never bypass `commit-conventions` for the squashed commit.
- If `gh` authentication or remote permissions fail, stop and report the exact blocker.

## Output

Report:

- Base branch used
- Final commit message
- PR title
- PR URL, if creation or update succeeded
- Any steps that were skipped or require user follow-up

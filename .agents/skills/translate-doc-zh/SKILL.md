---
name: translate-doc-zh
description: Translate an English document under `docs/en/` into the matching Chinese document under `docs/zh/`.
---

# Translate Docs EN to ZH

Use this skill when the user asks to translate a document from `docs/en/` to `docs/zh/`.

## Input

- A markdown file path under `docs/en/`

## Workflow

### Step 1: Validate the source path

1. Confirm the file exists.
1. Confirm it is under `docs/en/`.
1. Confirm it ends with `.md`.
1. If any check fails, stop and ask the user for a valid `docs/en/...md` path.

### Step 2: Resolve the target path

- Source: `docs/en/<path>.md`
- Target: `docs/zh/<path>.md`

### Step 3: Choose translation mode

- If the Chinese file already exists, update only the changed parts while preserving the
  rest.
- If the Chinese file does not exist, translate the full document.

## Translation Rules

- Preserve technical terms such as FSDP, FSDP2, GRPO, PPO, DAPO, MoE, LLM, RL, RLVR,
  Codex, Claude Code, OpenCode, Megatron, Archon, SGLang, vLLM, PyTorch, HuggingFace,
  and Transformers.
- Do not translate file paths, code blocks, CLI flags, or literal configuration keys.
- Preserve Markdown structure, tables, and fenced code blocks.
- Use concise, professional Chinese terminology.

## Error Handling

- If the target directory does not exist, create it before writing the translated file.
- If the source document is partially translated already, preserve sections that do not
  need updates.

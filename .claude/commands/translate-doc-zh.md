---
name: translate-doc-zh
description: 'Translate English documentation to Chinese. Usage: /translate-doc-zh docs/en/path/to/file.md'
argument-hint: <path-to-en-doc>
---

# Document Translation (EN → ZH)

Translate English documentation to Chinese for the AReaL project.

## Usage

```
/translate-doc-zh $ARGUMENTS
```

**Arguments (`$ARGUMENTS`):**

- Path to English document (e.g., `docs/en/tutorial/quickstart.md`)

## Document to Translate

!`if [ -f "$ARGUMENTS" ]; then echo "File: $ARGUMENTS"; echo ""; head -50 "$ARGUMENTS"; else echo "ERROR: File not found: $ARGUMENTS"; echo "Please provide a valid path to an English document in docs/en/"; fi`

## Workflow

### Step 1: Validate the Path

1. Check if `$ARGUMENTS` exists
1. Verify it is inside `docs/en/` directory and ends with `.md`
1. If invalid, inform user and stop

### Step 2: Determine Output Path

- Input: `$ARGUMENTS` (e.g., `docs/en/tutorial/quickstart.md`)
- Output: Replace `docs/en/` with `docs/zh/` (e.g., `docs/zh/tutorial/quickstart.md`)

### Step 3: Check if Chinese Document Exists

- If `docs/zh/<path>` exists → **Modification Scenario**
- If `docs/zh/<path>` does NOT exist → **Full Translation Scenario**

______________________________________________________________________

## Scenario 1: Modification (Chinese Document Exists)

Use when Chinese document already exists.

### Workflow

1. Read both English and Chinese documents
1. Compare differences to identify:
   - New sections in English
   - Modified sections in English
   - Deleted sections
1. Translate only changed parts
1. Preserve all other Chinese content unchanged

**Translation Rules:**

- Preserve English technical terms: FSDP, FSDP2, GRPO, PPO, DAPO, MoE, LLM, RL, RLVR,
  Claude Code, OpenCode, Megatron, Archon, SGLang, vLLM, PyTorch, HuggingFace,
  Transformers, etc.
- File paths and code examples remain unchanged
- Professional and rigorous terminology
- Preserve Markdown format, code blocks, tables

______________________________________________________________________

## Scenario 2: Full Translation (Chinese Document Does NOT Exist)

Use when Chinese document does not exist.

### Workflow

1. Read the English source document
1. Translate entire document to Chinese

**Translation Rules:**

- Preserve English technical terms: FSDP, FSDP2, GRPO, PPO, DAPO, MoE, LLM, RL, RLVR,
  Claude Code, OpenCode, Megatron, Archon, SGLang, vLLM, PyTorch, HuggingFace,
  Transformers, etc.
- File paths and code examples remain unchanged
- Professional and rigorous terminology
- Preserve Markdown format, code blocks, tables

3. Create new Chinese document at `docs/zh/<path>`

______________________________________________________________________

## Error Handling

### Invalid Path

If user provides an invalid path:

1. Tell user the file does not exist
1. Ask user to provide a valid path starting with `docs/en/` and ending with `.md`

### Write Failure

If target directory does not exist:

1. Create the directory first using Bash mkdir -p
1. Then write the file

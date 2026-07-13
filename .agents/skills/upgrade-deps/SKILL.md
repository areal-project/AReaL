---
name: upgrade-deps
description: Upgrade focused runtime dependencies in AReaL. First validates and updates per-package API checklists for structural completeness, then updates pyproject files, resolves conflicts, locks, updates the Dockerfile, and audits API compatibility against the checklists.
---

## Usage

```text
/upgrade-deps <package==version> [<package==version> ...]
```

**Arguments:** One or more pinned package versions, e.g.,
`/upgrade-deps megatron-core==0.17.0 sglang==0.5.10 vllm==0.18.0 transformers==4.58.0`.

If a package is omitted, its current version can be preserved or upgraded, depending on
the resolution of `uv lock`.

______________________________________________________________________

## Architecture

### Dual-Manifest Model

AReaL maintains **two pyproject files** because SGLang and vLLM pin
mutually-incompatible `torch` / `torchao` versions:

| File                  | Inference backend | Lock file      |
| --------------------- | ----------------- | -------------- |
| `pyproject.toml`      | SGLang (default)  | `uv.lock`      |
| `pyproject.vllm.toml` | vLLM              | `uv.vllm.lock` |

Both share the same core dependencies, megatron extras, and dev group. They diverge only
in inference backend extras and `torch`/`torchao` version constraints.

The `Dockerfile` builds both variants from a single file using `ARG VARIANT` (`sglang`
or `vllm`). The base image, torch install, and flash-attn wheels are all
variant-specific.

### Focused Packages

The following packages are **focused** — their API usage in AReaL is cataloged, and any
version change triggers the API compatibility audit (Step 6):

| Package           | Import path       |
| ----------------- | ----------------- |
| `megatron-core`   | `megatron.core`   |
| `megatron-bridge` | `megatron.bridge` |
| `mbridge`         | `mbridge`         |
| `transformers`    | `transformers`    |
| `sglang`          | `sglang`          |
| `vllm`            | `vllm`            |
| `peft`            | `peft`            |
| `torchao`         | `torchao`         |

`torch` is tracked in the Package Impact Matrix below for scope and Docker awareness,
but is **not** a focused package — it does not receive API auditing.

### Package Impact Matrix

Every entry below has a **variant scope** that determines which files to edit, which
lock files to regenerate, and whether the Dockerfile needs review. All focused packages
plus `torch` (tracked for Docker impact only, not API-audited) are listed.

| Package           | Scope            | pyproject.toml locations                                       | pyproject.vllm.toml locations                                  | Docker impact |
| ----------------- | ---------------- | -------------------------------------------------------------- | -------------------------------------------------------------- | ------------- |
| `sglang`          | sglang-only      | `[optional-deps].sglang`                                       | —                                                              | Base image    |
| `vllm`            | vllm-only        | —                                                              | `[optional-deps].vllm`                                         | No            |
| `megatron-core`   | shared           | `[optional-deps].megatron` + `[tool.uv].override-dependencies` | `[optional-deps].megatron` + `[tool.uv].override-dependencies` | No            |
| `megatron-bridge` | shared           | `[optional-deps].megatron`                                     | `[optional-deps].megatron`                                     | No            |
| `mbridge`         | shared           | `[optional-deps].megatron` (git pin)                           | `[optional-deps].megatron` (git pin)                           | No            |
| `transformers`    | shared           | `[project].dependencies`                                       | `[project].dependencies`                                       | No            |
| `peft`            | shared           | `[project].dependencies`                                       | `[project].dependencies`                                       | No            |
| `torch`           | shared-divergent | `[project].dependencies` (≥2.9.1)                              | `[project].dependencies` (≥2.10.0)                             | Stage 1 torch |
| `torchao`         | shared-divergent | `[tool.uv].override-dependencies` (==0.15.0)                   | `[project].dependencies` (==0.16.0)                            | No            |

**Scope rules:**

- **sglang-only / vllm-only**: Edit and lock only the affected variant.
- **shared**: Edit both files; lock both variants.
- **shared-divergent**: The two variants intentionally use different versions of the
  same package. If the user supplies a single target version, **ask which variant(s) to
  apply it to** rather than assuming convergence. If the user supplies two targets
  (e.g., `torch==2.9.1@sglang torch==2.10.0@vllm`), apply each accordingly.

### Upgrade Families

Some packages are tightly coupled and should be upgraded together:

| Family    | Members                                  | Reason                                                    |
| --------- | ---------------------------------------- | --------------------------------------------------------- |
| megatron  | `megatron-core`, `megatron-bridge`       | megatron-bridge wraps megatron-core; tightly coupled APIs |
| inference | `sglang` or `vllm` + `torch` + `torchao` | Inference backends pin specific torch/torchao versions    |

When the user upgrades one member, **check the checklist of all family members** for
required co-upgrades and warn if a co-upgrade is needed but not requested.

### Per-Package API Checklists

Each focused package has a dedicated markdown file under `checklists/` that documents
how AReaL uses that package's APIs:

```
.agents/skills/upgrade-deps/
├── SKILL.md
└── checklists/
    ├── megatron-core.md
    ├── megatron-bridge.md
    ├── vllm.md
    ├── sglang.md
    ├── transformers.md
    ├── peft.md
    └── torchao.md
```

Each checklist file MUST contain the following sections:

````markdown
---
package: <pip-package-name>
github: <org/repo>                    # e.g., NVIDIA/Megatron-LM
branch_template: "v${VERSION}"        # how to construct the git branch/tag from version
upstream_paths:                        # source paths to cross-reference
  - path/to/relevant/file.py
  - path/to/relevant/module/
---

## Affected Files

### Primary (most likely to break)
| File | Imports / Usage |
| ---- | --------------- |
| ...  | ...             |

### Secondary
...

### Tertiary (tests, infra)
...

## API Usage Catalog

For each function/class, verify the call signature against the upstream source.
Focus on: removed params, renamed params, new required params, changed return types,
moved/renamed modules.

### 1. `module.submodule.function_or_class`

**Source:** `upstream_path/file.py`

Called in `areal/path/to/file.py:LINE`:
\```python
actual_call_site_code()
\```

**Check:** [what to verify]

### 2. ...

## Version-Guarded Code (if any)
- [file:line] description of version guard and when it can be removed
````

To populate a checklist, catalog AReaL's usage of the package by grepping for imports
and call sites, then fill in the template sections. Each checklist should be
self-contained — all API signatures, file paths, and upstream references must be
recorded directly in the checklist file.

______________________________________________________________________

## Workflow

### Step 0: Parse Input and Record Baseline

1. Parse arguments. Each argument is `<package==version>` or, for shared-divergent
   packages, `<package==version@variant>` (e.g., `torchao==0.16.0@vllm`).
1. Validate that each package name is a recognized focused package (see Focused Packages
   table). If unrecognized, warn but proceed — it may be a transitive dependency the
   user wants to pin.
1. For **shared-divergent** packages (`torch`, `torchao`): if the user supplies a single
   version without a `@variant` qualifier, **ask which variant(s) to apply it to**
   before proceeding.
1. Record the current **resolved** versions of ALL focused packages from the applicable
   lock file(s) as the baseline snapshot. Use `uv.lock` for sglang-scoped and shared
   packages; `uv.vllm.lock` for vllm-scoped and shared packages. Packages scoped to a
   single variant appear only in that variant's lock file. This baseline will be used
   later to detect which packages actually changed — including transitive bumps caused
   by `uv lock` resolution, not just explicit pin changes.

### Step 0.5: Validate and Update Checklists

Before modifying any dependencies, verify that the API checklists for the packages being
upgraded are structurally complete — i.e., they document all current AReaL import sites
and call patterns. Stale checklists lead to missed breaking changes in Step 6.

For each **focused package explicitly requested** in the command that has a checklist
file under `checklists/`:

1. **Discover all usages.** Grep the AReaL codebase (`areal/`, `tests/`, `examples/`)
   for all imports matching the package's import path(s). See the Import Patterns table
   in `CHECKLIST_MAINTENANCE.md` § 2 for package-specific grep patterns. For HTTP-based
   integrations (sglang, vllm), also scan request-building code for endpoint paths and
   JSON field names.

1. **Compare against the checklist.** Diff the discovered files against the Affected
   Files tables (Primary / Secondary / Tertiary) in the checklist. Identify:

   - **Missing** — files that import the package but are not listed in any tier.
   - **Stale** — files listed in the checklist that no longer import the package.
   - **Changed** — files whose actual imports differ from what's documented.

1. **Classify and update.** For each discrepancy, follow the Structural Validation
   Procedure in `CHECKLIST_MAINTENANCE.md` § 3:

   - Add missing files to the appropriate tier (Primary / Secondary / Tertiary).
   - Add new API Usage Catalog entries for genuinely new call patterns.
   - Remove stale entries.
   - Renumber catalog entries if needed.

1. **Update the Checklist File Status table** at the bottom of this file if entry counts
   changed.

1. **Report** changes before proceeding:

   ```text
   Checklist validation for <package>:
   - Added N files to Affected Files (P primary, S secondary, T tertiary)
   - Added M new API catalog entries: [brief list]
   - Removed K stale entries: [brief list]
   - No changes needed (if clean)
   ```

If a requested focused package does **not** have a checklist file, create one from
`checklists/_TEMPLATE.md` using the full procedure in `CHECKLIST_MAINTENANCE.md` § 5.

**Scope note:** This step validates only the packages explicitly named in the command.
Packages that are transitively bumped are identified later in Step 5; their checklists
are validated at that point using the same procedure before the API audit in Step 6.

### Step 1: Update Pyproject Files

For each requested package, using the Package Impact Matrix:

1. Determine the **variant scope** (sglang-only, vllm-only, shared, shared-divergent).
1. Edit the version pin in **all declaration locations** for each affected pyproject
   file. A single package may appear in multiple locations:
   - `[project].dependencies`
   - `[project.optional-dependencies].<extra>`
   - `[tool.uv].override-dependencies`
1. For **upgrade families**: if `megatron-core` is being upgraded, also check whether
   `megatron-bridge` needs a corresponding version bump. Warn if it does but was not
   included in the command.

### Step 2: Lock Dependencies (Variant-Aware)

Regenerate lock files for each **affected variant**. A variant is affected if any of its
pyproject file's dependencies were modified. If conflicts arise during locking, do NOT
attempt to resolve them in this step — just report them and defer resolution to Step 3.

**SGLang variant** (if `pyproject.toml` was modified):

```bash
bash scripts/uv_lock.sh
```

**vLLM variant** (if `pyproject.vllm.toml` was modified):

```bash
bash scripts/uv_lock.sh
```

If `uv lock` fails for either variant:

1. Read the error message carefully.
1. If it's a version conflict, proceed to Step 3 to resolve, then **return here to
   re-lock**.
1. Do NOT modify dependencies without user approval to make the lock succeed.

**Validation:** After locking, verify both lock files exist and are non-empty.

### Step 3: Resolve Dependency Conflicts

Resolve any conflicts from Step 2. After all conflicts are resolved, **return to Step 2
and re-lock** the affected variant(s). Classify each conflict:

**Auto-resolvable** — only AReaL's pin conflicts with an upstream package, and the
upstream's required version is acceptable. Update AReaL's pin automatically.

**Needs user input** — two upstream packages have mutual conflicts (e.g., sglang
requires `torch==2.9.1` but vllm requires `torch==2.10.0`). Summarize and ask the user.

Output format:

```
Summary

---
Auto-resolved (no action required):
- <name>: <packageA> requires <versionA>, <packageB> requires <versionB>,
  AReaL specified <oldVersion>, updated to <newVersion>
- ...

---
Conflicts (need user resolution):
- <name>: <packageA> requires <versionA>, <packageB> requires <versionB>
- ...
```

You may use `override-dependencies` in `[tool.uv]` to force-pin versions where needed.
Remember that `sglang` and `vllm` are **separate variants** maintained in different
pyproject files — they are never installed together in the same environment.

### Step 4: Update Dockerfile (If Required)

Review the Package Impact Matrix "Docker impact" column. Only proceed with Dockerfile
changes if an upgraded package has Docker impact. The most common triggers are:

**Base image change** (triggered by `sglang` upgrade):

The Dockerfile base image is `lmsysorg/sglang:v{SGLANG_VERSION}-cu129-amd64-runtime`. If
`sglang` was upgraded, update line 9 of the `Dockerfile`:

```dockerfile
FROM lmsysorg/sglang:v{NEW_SGLANG_VERSION}-cu129-amd64-runtime
```

Verify that the new base image tag exists on Docker Hub / GHCR before committing.

**Torch version change** (triggered by `torch` upgrade):

The Dockerfile Stage 1 installs torch with variant-specific versions. Update the version
mapping in the `RUN uv venv` command:

```dockerfile
&& if [ "$VARIANT" = "vllm" ]; then TORCH_VER="{VLLM_TORCH}"; else TORCH_VER="{SGLANG_TORCH}"; fi \
```

Also check flash-attn wheel compatibility — **both** flash-attn-2 and flash-attn-3
installs use a `TORCH_TAG` that must match the torch major.minor version. Update both
occurrences in the Dockerfile:

```dockerfile
# flash-attn-2 (first flash-attn RUN block)
&& if [ "$VARIANT" = "vllm" ]; then TORCH_TAG="torch{VLLM_TORCH_MAJOR_MINOR}"; else TORCH_TAG="torch{SGLANG_TORCH_MAJOR_MINOR}"; fi \

# flash-attn-3 (second flash-attn RUN block — same TORCH_TAG pattern)
&& if [ "$VARIANT" = "vllm" ]; then TORCH_TAG="torch{VLLM_TORCH_MAJOR_MINOR}"; else TORCH_TAG="torch{SGLANG_TORCH_MAJOR_MINOR}"; fi \
```

**No Dockerfile changes needed** for: `megatron-core`, `megatron-bridge`,
`transformers`, `peft`, `vllm`, `torchao`. These are installed via
`uv pip install -r pyproject.toml` in Stage 3, which reads the updated pyproject
automatically.

### Step 5: Identify Updated Focused Packages

Compare the baseline snapshot (Step 0) against the **resolved** versions in the lock
files (`uv.lock` and/or `uv.vllm.lock`) produced by Step 2 (or re-locked after Step 3
conflict resolution). This catches not only explicitly requested upgrades but also
**transitive version bumps** — e.g., upgrading `sglang` may pull in a newer
`transformers` through dependency resolution.

Build a list of focused packages whose resolved version actually changed. This list
determines which API checklists to audit in Step 6.

The focused packages to check are listed in the Focused Packages table (Architecture
section):

- `megatron-core` (imports as `megatron.core`)
- `megatron-bridge` (imports as `megatron.bridge`)
- `transformers`
- `sglang`
- `vllm`
- `peft`
- `torchao`

If a package version did NOT change (even if it was in the user's input but resolved to
the same version), skip its API audit.

For any newly-identified package whose checklist was NOT already validated in Step 0.5,
run the same structural validation procedure (see `CHECKLIST_MAINTENANCE.md` § 3) on its
checklist before proceeding to Step 6.

### Step 6: API Compatibility Audit

For each updated focused package that has a checklist file under `checklists/`:

#### 6a. Clone upstream source

Read the checklist frontmatter to get `github` and `branch_template`. Clone or checkout
the target version:

```bash
REPO_ROOT=$(pwd)
PKG_DIR="${REPO_ROOT}/<package>-src"
VERSION="<target_version>"
# Validate VERSION to prevent command injection
if [[ ! "$VERSION" =~ ^[a-zA-Z0-9._/-]+$ ]]; then
  echo "Error: Invalid version format: $VERSION"; exit 1
fi
BRANCH=$(echo "<branch_template>" | sed "s/\${VERSION}/$VERSION/")
if [ ! -d "$PKG_DIR" ]; then
  git clone --depth 1 --branch "$BRANCH" "https://github.com/<github>.git" "$PKG_DIR"
else
  (cd "$PKG_DIR" && git fetch origin && git checkout "$BRANCH")
fi
```

If cloning fails (tag doesn't exist, etc.), report to the user immediately.

#### 6b. Audit API signatures

For EACH entry in the checklist's API Usage Catalog:

1. Open the upstream source file at the target version (paths listed in the checklist's
   `upstream_paths` frontmatter).
1. Compare the function/class signature against the current AReaL invocation.
1. Flag any of:
   - **Removed parameters** still passed by AReaL → must remove from call site
   - **Renamed parameters** → must rename in call site
   - **New required parameters** (no default) → must add to call site
   - **New optional parameters** with useful defaults → document but skip
   - **Changed return types** → must update consumers
   - **Removed functions/classes** → must find replacement
   - **Moved modules** → must update import paths
   - **Changed method signatures** on returned objects → must update call sites
1. Record findings per-file.

#### 6c. Check version-guarded code

If the checklist has a "Version-Guarded Code" section, check whether any guards
reference versions at or below the new target. If so, verify the upstream fix is present
and note the dead code for cleanup.

#### 6d. Apply code changes (if any)

For each flagged incompatibility:

1. Update the call site in the affected AReaL file.
1. **Preserve existing behavior** — do NOT refactor beyond what's required.
1. If a function was removed, check the upstream migration guide or changelog.
1. Priority order: engine layer → model layer → infra layer → test files.

If there are unresolvable breaking changes, **STOP and ask the user** before proceeding.

#### 6e. Update checklist file

Update the checklist file to reflect the post-upgrade state. Follow the Content Update
Procedure in `CHECKLIST_MAINTENANCE.md` § 4:

1. Update API signatures in catalog entries where upstream changed.
1. Update call-site code snippets where Step 6d modified AReaL code.
1. Update version-guarded code entries (remove cleaned-up guards, add new ones).
1. Update frontmatter `upstream_paths` if source files moved.
1. Update the entry count in the Checklist File Status table (bottom of this file).

This ensures the checklist remains an accurate reference for future upgrades.

#### 6f. Clean up cloned repositories

Remove the cloned upstream source directories to avoid cluttering the workspace:

```bash
rm -rf "${REPO_ROOT}/<package>-src"
```

### Step 7: Run Pre-Commit

```bash
pre-commit run --all-files
```

### Step 8: Generate Upgrade Summary

Dump a formatted markdown summary to `upgrade-summary.md` in the repository root (this
file is gitignored and ephemeral). The summary MUST include:

```markdown
## Dependency Upgrade Summary

**Date:** YYYY-MM-DD
**Requested:** <original command>

### Version Changes

| Package | Old Version | New Version | Variant(s) |
| ------- | ----------- | ----------- | ---------- |
| ...     | ...         | ...         | ...        |

### Dependency Resolution

- <auto-resolved change description>
- ...

### Dockerfile Changes

- <change description, or "No changes required">

### API Compatibility Audit

#### <package-name> (old → new)

**Breaking changes found:**
- [file:line] description of change

**Module moves / renames:**
- [old_path] → [new_path]

**Version-guarded code:**
- [file:line] status (still needed / can be removed)

**No breaking changes found** _(if clean)_

#### ...

### Unresolved Issues (if any)

- <description of issue and why it could not be auto-resolved>
```

If the upgrade failed at any step, the summary should still be generated with the
failure reason clearly documented in the "Unresolved Issues" section.

### Step 9: Create PR and Trigger CI (Optional)

Ask the user if they want to create a PR.

If the user agrees:

1. Load the `create-pr` skill to create the PR.
1. Trigger the CI workflow manually via `.github/workflows/build-docker-image.yml` (only
   if the Dockerfile was modified or inference backend versions changed).
1. The Docker build CI builds both sglang and vllm images, then automatically triggers
   testing on each. Debug until the overall workflow succeeds.
1. If you encounter issues that cannot be resolved, ask the user for help.

______________________________________________________________________

## Checklist File Status

| Package           | Checklist file                  | Status                                                                                                                                             |
| ----------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `megatron-core`   | `checklists/megatron-core.md`   | ✅ 18 API entries (parallel_state, DDP, optimizer, pipeline, checkpointing, transformer config, FP8, GPTModel, tensor_parallel, layer specs, RoPE) |
| `megatron-bridge` | `checklists/megatron-bridge.md` | ✅ 7 API entries (AutoBridge, LoRA, save/load HF, monkey-patch guard)                                                                              |
| `mbridge`         | `checklists/mbridge.md`         | ✅ 14 API entries (AutoBridge, Bridge properties, weight mappings, LLMBridge subclassing, register_model, monkey-patch target)                     |
| `vllm`            | `checklists/vllm.md`            | ✅ 14 API entries (entrypoints, LoRA manager, worker V0/V1, tool parsers, CLI)                                                                     |
| `sglang`          | `checklists/sglang.md`          | ✅ 14 API entries (HTTP endpoints, tool/reasoning parsers, CLI flags, version guards)                                                              |
| `transformers`    | `checklists/transformers.md`    | ✅ 12 API entries (Auto\* classes, tokenizer, flash attention monkey-patches, Qwen VL internals, LR schedulers)                                    |
| `peft`            | `checklists/peft.md`            | ✅ 4 API entries (LoraConfig, TaskType, get_peft_model, weight key format)                                                                         |
| `torchao`         | `checklists/torchao.md`         | ✅ 5 API entries (fp8_blockwise_mm, enable_fp8_linear/experts, shard validation, Triton kernels)                                                   |

# Checklist Maintenance Guide

This document is the single reference for maintaining the per-package API checklists
under `checklists/`. It covers two complementary maintenance activities:

- **Structural validation** — ensuring every AReaL import site is documented in the
  checklist. Performed in SKILL.md → **Step 0.5** before any dependency changes.
- **Content update** — ensuring documented API signatures and code snippets match the
  post-upgrade state. Performed in SKILL.md → **Step 6e** after code changes are
  applied.

Both activities share the same checklist format. This guide also covers creating new
checklists and writing good API catalog entries.

______________________________________________________________________

## 1. Checklist Structure Reference

Each checklist file lives at `checklists/<package-name>.md` and follows the format
defined in `checklists/_TEMPLATE.md`. The key sections are:

| Section                  | Purpose                                                                                                                                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **YAML frontmatter**     | `package`, `github` (org/repo), `branch_template`, `upstream_paths` — metadata used by the API audit (Step 6a) to clone the correct upstream tag and locate source files.                                          |
| **Affected Files**       | Three-tier table (Primary / Secondary / Tertiary) listing every AReaL file that imports or uses the package, with a summary of what it imports.                                                                    |
| **API Usage Catalog**    | Numbered entries documenting each distinct API surface (function, class, HTTP endpoint, schema contract). Each entry includes the upstream source path, the AReaL call-site code, and specific Check instructions. |
| **Version-Guarded Code** | List of AReaL code locations that have version-specific conditionals for this package (e.g., `if version < "0.4.10": ...`).                                                                                        |

______________________________________________________________________

## 2. Import Patterns by Package

Use these patterns to discover all AReaL usages of each focused package. Grep across
`areal/`, `tests/`, and `examples/`.

| Package           | Import path(s)    | Grep patterns                                    | Notes                                                                                 |
| ----------------- | ----------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `megatron-core`   | `megatron.core`   | `from megatron.core`, `import megatron.core`     |                                                                                       |
| `megatron-bridge` | `megatron.bridge` | `from megatron.bridge`, `import megatron.bridge` |                                                                                       |
| `mbridge`         | `mbridge`         | `from mbridge`, `import mbridge`                 | Pinned to git commit; also check `mbridge.core.*` submodule imports                   |
| `transformers`    | `transformers`    | `from transformers`, `import transformers`       | Exclude `flash_attn.bert_padding` false positives                                     |
| `sglang`          | `sglang`          | `from sglang`, `import sglang`                   | Also scan `sglang_remote.py` for HTTP endpoint paths and JSON field names             |
| `vllm`            | `vllm`            | `from vllm`, `import vllm`                       | Also scan `vllm_remote.py` and `vllm_ext/` for HTTP endpoints and JSON fields         |
| `peft`            | `peft`            | `from peft`, `import peft`                       | Also check for indirect PEFT schema dicts (`"r"`, `"lora_alpha"`, `"target_modules"`) |
| `torchao`         | `torchao`         | `from torchao`, `import torchao`                 | All usage goes through `prototype.*` path                                             |

### Edge cases to watch for

- **HTTP-only integrations** (sglang, vllm): The primary integration is REST-based, not
  Python imports. Grep for endpoint path strings (e.g., `"/generate"`,
  `"/update_weights_from_distributed"`) and JSON field names in request/response parsing
  code.
- **Indirect / schema-only usage**: Some files reference a package's conventions without
  importing it directly (e.g., PEFT config dicts in `areal/api/io_struct.py`). These
  should be listed in Affected Files and may warrant an API catalog entry if the schema
  contract is non-trivial.
- **Conditional / lazy imports**: Imports inside `if TYPE_CHECKING:` blocks or inside
  function bodies. These are still real dependencies — include them.
- **String references**: Config enums or registry keys that name a package (e.g.,
  `"sglang"` in `SGLangConfig`). These rarely need catalog entries but should appear in
  Affected Files if the file also has substantive usage.
- **Test stubs**: Files like `tests/experimental/archon/conftest.py` that conditionally
  stub out a package. List in Tertiary Affected Files.

______________________________________________________________________

## 3. Structural Validation Procedure

> **Referenced by:** SKILL.md → Step 0.5

This procedure checks whether a checklist documents ALL current AReaL import sites and
call patterns. Run it before upgrading a package to ensure the API audit (Step 6) has
complete coverage.

### 3.1 Discover all usages

For each focused package being validated:

1. Grep the AReaL codebase using the patterns from § 2. Search across `areal/`,
   `tests/`, and `examples/`.
1. For HTTP-based packages, also scan the relevant remote engine files for endpoint
   paths and JSON field names.
1. Exclude false positives: comments, docstrings, unrelated substrings (e.g.,
   `flash_attn.bert_padding` is not a `transformers` import).
1. Collect results as `{file_path, imported_names[]}` for each hit.

### 3.2 Compare against Affected Files tables

Parse the checklist's Primary / Secondary / Tertiary tables and compare against the
discovered files. Build three lists:

- **MISSING** — files found in the codebase that are not listed in any tier.
- **STALE** — files listed in the checklist that no longer import the package (file
  deleted, import removed, or moved to a different file).
- **CHANGED** — files listed in the checklist whose actual imports differ from what the
  "Imports / Usage" column documents.

### 3.3 Classify new files into tiers

Use these rules (consistent with existing checklists):

| Tier          | Directories                                                                                                                                                               | Rationale                                                                |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Primary**   | `areal/engine/`, `areal/experimental/engine/`, `areal/experimental/models/`                                                                                               | Engine layer — most likely to break on API changes, highest blast radius |
| **Secondary** | `areal/models/`, `areal/workflow/`, `areal/infra/`, `areal/api/`, `areal/utils/`, `areal/trainer/`, `areal/dataset/`, `areal/reward/`, `areal/experimental/` (non-engine) | Model / infra / workflow layer — important but lower blast radius        |
| **Tertiary**  | `tests/`, `examples/`, `docs/`, tool scripts                                                                                                                              | Tests and examples — breakage is contained, lower priority               |

When a file spans concerns (e.g., a test that exercises engine internals), classify by
the file's location, not its content.

### 3.4 Determine new API catalog entries

For each MISSING or CHANGED file, examine the specific imports and call sites:

- If the usage pattern is **already covered** by an existing catalog entry (same
  function/class, same call pattern) → just add the file to the Affected Files table. No
  new catalog entry needed.
- If the usage involves a function/class **NOT in the catalog**, or uses it with
  **significantly different parameters** → create a new API catalog entry (see § 6 for
  format).

"Significantly different" means: different keyword arguments, different return value
consumption, or a different code path that could break independently.

### 3.5 Update the checklist

1. Add missing files to the appropriate tier table with their imports summary.
1. Remove stale file entries. If uncertain whether a removal is correct, add a
   `<!-- TODO: verify removal -->` comment instead of deleting.
1. Update the "Imports / Usage" column for CHANGED files.
1. Add new API catalog entries following the format in § 6.
1. Renumber catalog entries sequentially (1, 2, 3, ...).
1. Update the entry count in SKILL.md's **Checklist File Status** table at the bottom.

### 3.6 Report format

After completing validation, report changes before proceeding:

```text
Checklist validation for <package>:
- Added N files to Affected Files (P primary, S secondary, T tertiary)
- Added M new API catalog entries: [brief list]
- Removed K stale file entries: [brief list]
- Updated J existing entries with changed imports
- No changes needed (if clean)
```

______________________________________________________________________

## 4. Content Update Procedure

> **Referenced by:** SKILL.md → Step 6e

This procedure updates a checklist's API signatures and code snippets AFTER an upgrade
has been applied and code changes (Step 6d) are complete. The goal is to keep the
checklist accurate for the next upgrade cycle.

### 4.1 Update API signatures

For each catalog entry where the upstream API signature changed at the new version:

1. Update the **Source** path if the upstream file was moved or renamed.
1. Note any new parameters, removed parameters, or type changes in the **Check**
   instructions — these become the baseline for the next upgrade.
1. If a function was removed and replaced, update the entry to reference the
   replacement.

### 4.2 Update call-site code snippets

If Step 6d modified an AReaL call site to accommodate the new version:

1. Replace the code snippet in the corresponding catalog entry with the updated code.
1. Update the line number references if they shifted.
1. Ensure the snippet includes enough context to understand the parameters used and the
   return value consumed.

### 4.3 Update version-guarded code

- **Remove** entries for version guards that were cleaned up (the guarded version is now
  below the minimum supported version).
- **Add** entries for any new version guards introduced during the upgrade.
- **Update** threshold versions in existing entries.

### 4.4 Update frontmatter

- Add or remove entries in `upstream_paths` if upstream source files were moved,
  renamed, or deleted.
- Update `branch_template` if the upstream project changed its tagging convention (e.g.,
  from `v${VERSION}` to `release/${VERSION}`).

### 4.5 Update SKILL.md status table

Update the entry count and description in the **Checklist File Status** table at the
bottom of SKILL.md to reflect the current state of the checklist.

______________________________________________________________________

## 5. Creating a New Checklist

When a new package becomes "focused" (added to SKILL.md's Focused Packages table):

1. **Copy the template.** Copy `checklists/_TEMPLATE.md` to
   `checklists/<package-name>.md`.

1. **Fill in YAML frontmatter:**

   - `package`: the pip package name (e.g., `sglang`)
   - `github`: the GitHub org/repo (e.g., `sgl-project/sglang`)
   - `branch_template`: how to construct the git tag from a version string (e.g.,
     `v${VERSION}`)
   - `upstream_paths`: list of source paths in the upstream repo that are most relevant
     to AReaL's usage

1. **Run structural validation.** Follow the full procedure in § 3 to populate the
   Affected Files tables and API Usage Catalog.

1. **Update SKILL.md.** Add the package to:

   - The **Focused Packages** table (Architecture section)
   - The **Package Impact Matrix** (with correct scope and pyproject locations)
   - The **Checklist File Status** table (bottom of file)

1. **Commit** the new checklist alongside the SKILL.md update.

______________________________________________________________________

## 6. API Catalog Entry Format

Each entry documents one distinct API surface — a function, class, HTTP endpoint, or
schema contract. Use this format:

````markdown
### N. `fully.qualified.FunctionOrClass`

**Source:** `upstream-repo/path/to/file.py`

Called in `areal/path/to/file.py` (description, lines N–M):

```python
# Paste the ACTUAL call site code — enough context to understand
# the parameters used and the return value consumed.
actual_call(param1=..., param2=...)
```

**Check:** [What to verify — be specific.]
````

### Guidelines

- **One entry per distinct API surface.** A function, class, HTTP endpoint, or schema
  contract gets one entry. If the same function is called in multiple files with the
  same pattern, list all call sites in a single entry.
- **Multiple call sites.** When the same API is used in several files, list the primary
  call site with a full code snippet, then note additional call sites with a brief
  reference: "Also called in `areal/path/to/other.py`."
- **HTTP endpoints.** Document the endpoint path, HTTP method, request payload fields,
  and response fields that AReaL parses. These are API surfaces even though they are not
  Python imports.
- **Indirect / schema usage.** If AReaL constructs a dict that must conform to a
  package's schema (e.g., PEFT config dicts), document the schema contract and note that
  it is indirect.
- **Check instructions.** Be specific and actionable:
  - Good: "Confirm `bias` still accepts the string `"none"` (not renamed to an enum)."
  - Bad: "Check if anything changed."
- **Code snippets.** Include enough surrounding code to show parameter names, types, and
  how the return value is consumed. Strip unrelated code. Include line numbers for
  reference.

______________________________________________________________________

## 7. When NOT to Add a Catalog Entry

Not every import needs its own API catalog entry. Skip a dedicated entry when:

- **Type-only imports.** A class imported solely for type annotations or `isinstance`
  checks (e.g., `from transformers import PretrainedConfig` used only as
  `config: PretrainedConfig`). List the file in Affected Files but skip the catalog
  unless the type annotation relies on specific attributes that could change.
- **Re-exports / pass-throughs.** A file that re-exports a symbol without calling it.
- **Standard patterns unlikely to break.** Accessing `__version__`, basic enum values,
  or other stable public API that has never changed across versions.
- **Test files duplicating Primary/Secondary coverage.** If a test file uses the exact
  same API pattern already documented in a Primary or Secondary entry, list it in
  Tertiary Affected Files but skip a separate catalog entry. Exception: if the test
  exercises a unique code path or parameter combination, add an entry.

When in doubt, **add the entry**. A slightly verbose checklist is better than a gap that
causes a missed breaking change.

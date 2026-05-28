# `/review-pr` Refactor & Update Plan (Codex / OpenCode / Claude)

Last updated: 2026-03-31 Owner: AI workflow maintainers Scope:
`.agents/skills/review-pr/*`, `.opencode/{command,data}/review-pr*`,
`.claude/{commands,data}/review-pr*`

______________________________________________________________________

## 1. Goals and Non-Goals

### Goals

1. Rebuild `/review-pr` taxonomy and templates for better coverage of new AReaL modules.
1. Keep template count manageable while preserving depth for high-risk PRs.
1. Eliminate cross-platform drift through a single semantic source of truth.
1. Reduce false positives by prioritizing path-based triggers.

### Non-Goals

1. Changing `/review-pr` into a file-mutating workflow (must remain read-only).
1. Adding one-off niche templates for every micro-feature.
1. Rewriting all platform command orchestration logic in one iteration.

______________________________________________________________________

## 2. Current State Summary (Problem Statement)

### What works

1. Existing workflow phases are clear (context -> analysis -> planning -> delegated
   review -> summary).
1. Platform-specific command wrappers already exist and are functional.
1. Risk-first review behavior (CRITICAL/HIGH/MEDIUM/LOW) is already enforced.

### What is broken or stale

1. Domain/signal matrix lags behind repository evolution (`tree_attn`, `vllm_ext`,
   service stack, `archon_weight_sync`).
1. Taxonomy is overly fragmented in some areas and missing in others.
1. Three-platform data files are near-duplicates and easy to drift.
1. Trigger rules rely heavily on generic keywords in places where path triggers are
   safer.

______________________________________________________________________

## 3. Target Information Architecture

Use a **two-layer taxonomy**:

- **L1 Domain (12 domains)**: template-level planning unit.
- **L2 Signals (2-9 per domain)**: detection and checklist specialization unit.

This avoids both extremes: too many tiny templates or too coarse single-pass review.

### 3.1 L1 Domains and L2 Signals

1. **Distributed Runtime**
   - L2: `archon_core`, `archon_parallel`, `process_group`, `fsdp_core`,
     `megatron_core`, `collectives`, `mesh_dtensor`, `activation_checkpointing`,
     `weight_sync`
1. **Model Compute & Attention**
   - L2: `tree_attn`, `sdpa_varlen`, `sp_cp_attention_mask`, `triton_kernel`,
     `archon_model_family`, `archon_attention_stack`, `archon_moe_modeling`
1. **Inference Backend & Serving**
   - L2: `vllm_ext`, `remote_inference_backend`, `request_lifecycle`
1. **Service Orchestration**
   - L2: `service_routing_dataflow`, `session_consistency`
1. **Workflow & Trainer Contract**
   - L2: `workflow_engine_boundary`, `dataset_surface`, `async_contract`,
     `weight_version_contract`
1. **API & Config Compatibility**
   - L2: `dataclass_schema`, `cli_compat`, `backward_compat`,
     `project_dependency_config`
1. **Numerics & Tensor Semantics**
   - L2: `shape_dtype`, `numerical_stability`, `reward_surface`, `compile_dynamo`,
     `mixed_precision_fp8`
1. **Checkpoint & Recovery**
   - L2: `dcp_consistency`, `optimizer_state`, `resume_compat`
1. **Launcher & Infrastructure**
   - L2: `launcher_resource_match`, `scheduler_contract`, `rpc_transport`,
     `runtime_image_config`
1. **Low-Risk Hygiene**
   - L2: `tests_docs_config`, `logging_import_security`, `project_docs_metadata`
1. **Harness & Agent Infrastructure**
   - L2: `skill_definition`, `platform_command_data`, `agent_registry_config`
1. **CI/CD & Release Automation**
   - L2: `github_workflow_jobs`, `runner_provisioning`, `release_delivery`

______________________________________________________________________

## 4. Detection Strategy (Path-first)

### 4.1 Trigger priority

1. **Path trigger (primary)**
1. **Keyword trigger (secondary, only to refine existing path hits)**
1. **Linkage trigger (auto-add dependent checks)**

### 4.2 Canonical path mapping (v1)

1. `areal/models/tree_attn/**` -> Model Compute & Attention (HIGH/MEDIUM by touched
   files)
1. `areal/engine/vllm_ext/**` -> Inference Backend & Serving (HIGH)
1. `areal/experimental/models/archon/**` + `areal/experimental/engine/archon_engine.py`
   \+ `areal/experimental/engine/archon_checkpoint.py` -> Distributed Runtime + Model
   Compute & Attention (CRITICAL/HIGH)
1. `areal/experimental/agent_service/**` -> Service Orchestration (HIGH)
1. `areal/experimental/inference_service/**` -> Service Orchestration (HIGH)
1. `areal/experimental/engine/archon_weight_sync.py` -> Distributed Runtime (CRITICAL)
1. `areal/infra/rpc/**` -> Launcher & Infrastructure + Distributed Runtime (HIGH)
1. `areal/workflow/**`, `areal/trainer/**` -> Workflow & Trainer Contract (HIGH/MEDIUM)
1. `areal/api/**` -> API & Config Compatibility (MEDIUM)
1. `areal/utils/{saver.py,recover.py,async_checkpoint.py}` + `*checkpoint*.py` ->
   Checkpoint & Recovery (CRITICAL/HIGH)
1. `areal/experimental/models/archon/activation_checkpoint.py` -> Distributed Runtime
   (MEDIUM/HIGH)
1. `areal/experimental/models/archon/compile.py` -> Numerics & Tensor Semantics +
   Distributed Runtime (MEDIUM)
1. `pyproject.toml`, `uv.lock` -> API & Config Compatibility (MEDIUM)
1. `Dockerfile`, `.dockerignore` -> Launcher & Infrastructure (HIGH)
1. `.agents/**`, `.claude/**`, `.opencode/**`, `.codex/**`, `AGENTS.md`, `CLAUDE.md` ->
   Harness & Agent Infrastructure (MEDIUM/HIGH)
1. `.github/workflows/**` -> CI/CD & Release Automation (HIGH/CRITICAL)
1. `docs/build_all.sh`, `docs/generate_cli_docs.py`, `.github/PULL_REQUEST_TEMPLATE.md`,
   `README.md`, `CONTRIBUTING.md` -> Low-Risk Hygiene (LOW/MEDIUM)

### 4.3 Must-not-regress coverage (from current review-pr)

The migration must preserve the existing high-risk framework coverage already present
today:

1. `areal/experimental/models/archon/**` + `areal/experimental/engine/archon_engine.py`
   - `areal/experimental/engine/archon_checkpoint.py` -> Distributed Runtime + Model
     Compute & Attention
1. `areal/engine/fsdp_utils/**` + `areal/engine/fsdp_engine.py` -> Distributed Runtime
1. `areal/engine/megatron_engine.py` + `areal/engine/megatron_utils/**` -> Distributed
   Runtime
1. `areal/trainer/**` -> Workflow & Trainer Contract
1. `areal/reward/**` -> Numerics & Tensor Semantics + Workflow & Trainer Contract
1. `areal/dataset/**` -> Workflow & Trainer Contract + API & Config Compatibility

### 4.4 Noise control rules

1. No repo-wide standalone triggering for `current_platform`, `RTensor`, or `fp8`.
1. These keywords only refine severity/checklists after domain has been path-selected.
1. Cap task fan-out: max one primary template per domain, plus one general logic pass.

______________________________________________________________________

## 5. Template Strategy

Maintain **12 domain templates** (one per L1), each with signal-specific checklists,
plus **1 universal logic pass**.

### Template inventory (v1)

1. Distributed Runtime Review
1. Model Compute & Attention Review
1. Inference Backend & Serving Review
1. Service Orchestration Review
1. Workflow & Trainer Contract Review
1. API & Config Compatibility Review
1. Numerics & Tensor Semantics Review
1. Checkpoint & Recovery Review
1. Launcher & Infrastructure Review
1. Low-Risk Hygiene Review
1. Harness & Agent Infrastructure Review
1. CI/CD & Release Automation Review

### Mandatory universal pass

- Always run one lightweight **General Logic & Boundary** pass for non-doc PRs.

______________________________________________________________________

## 6. Severity Mapping

1. **CRITICAL**: distributed invariants, checkpoint correctness, core weight sync
   safety.
1. **HIGH**: service orchestration, inference backend runtime, trainer/workflow
   contracts.
1. **MEDIUM**: API compatibility, numerics/tensor semantics in bounded scope.
1. **LOW**: docs/tests/config-only hygiene.

Rule: domain default severity can be escalated by L2 signal combinations (e.g.,
`mesh_dtensor` + `weight_sync`).

______________________________________________________________________

## 7. Cross-Domain Linkage Rules (Auto-appended checks)

1. `tree_attn` -> also append Numerics & Tensor Semantics checks.
1. `archon_core` or `archon_parallel` -> also append Model Compute & Attention checks.
1. `archon_model_family` or `archon_moe_modeling` -> also append Numerics & Tensor
   Semantics checks.
1. `reward_surface` -> also append Workflow & Trainer Contract checks.
1. `compile_dynamo` -> also append Distributed Runtime checks.
1. `vllm_ext` -> also append Launcher & Infrastructure checks.
1. Service Orchestration changes -> also append Workflow & Trainer async-contract
   checks.
1. `archon_weight_sync` -> also append DTensor + process-group + checkpoint interaction
   checks.
1. RPC transport changes -> also append Distributed Runtime synchronization checks.
1. `mixed_precision_fp8` + Distributed Runtime -> also append mesh + weight-sync
   compatibility checks.
1. `runtime_image_config` -> also append Inference Backend & Serving checks.
1. `project_dependency_config` -> also append API & Config Compatibility checks.
1. `github_workflow_jobs` or `release_delivery` -> also append Launcher & Infrastructure
   checks.
1. `skill_definition` or `platform_command_data` -> also append Low-Risk Hygiene checks.

______________________________________________________________________

## 8. Three-Platform Synchronization Model

## 8.1 Source-of-truth

Semantic content is authored only in:

1. `.agents/skills/review-pr/references/review-pr-domains-and-signals.md`
1. `.agents/skills/review-pr/references/review-pr-templates.md`

Canonical semantic scope includes:

1. taxonomy (L1/L2), path/linkage rules, and severity rules
1. checklist bodies and change-analysis vocabulary

Wrapper-specific scope (non-canonical) includes:

1. command syntax and frontmatter fields
1. shell snippets, import/include syntax, and runtime-specific routing policies
1. OpenCode-only and Claude-only execution options

## 8.2 Derived outputs

Generated/derived files:

1. `.opencode/data/review-pr-domains-and-signals.md`
1. `.opencode/data/review-pr-templates.md`
1. `.claude/data/review-pr-domains-and-signals.md`
1. `.claude/data/review-pr-templates.md`

## 8.3 Mechanical sync (definition)

"Mechanical sync" means deterministic conversion, not manual copy/paste:

1. Read canonical `.agents` files.
1. Emit OpenCode and Claude data copies with the same generic review-depth vocabulary
   (`comprehensive`, `targeted`, `basic`).
1. Keep all runtime routing and platform execution choices in the wrapper command files
   only.
1. Preserve section order and checklist content exactly.

______________________________________________________________________

## 9. Implementation Plan

### Phase A (Foundation)

1. Replace the old review-pr taxonomy with L1/L2 domain/signal references in canonical
   `.agents` references.
1. Rebuild template file into 12 domain templates + universal logic pass.
1. Add linkage rules and fan-out caps.

### Phase B (Platform sync)

1. Add sync script at `.agents/skills/review-pr/sync_review_pr_refs.py`.
1. Regenerate `.opencode/data/*` and `.claude/data/*`.
1. Add CI check: fail if derived files differ from regeneration output.

### Phase C (Command layer alignment)

1. Update `.opencode/command/review-pr.md` wording to reflect new domains/signals.
1. Update `.claude/commands/review-pr.md` wording similarly.
1. Keep orchestration differences platform-specific (task categories vs model routing).

### Phase D (Validation)

1. Run **classification lane** with `/review-pr --quick` against representative PRs
   from:
   - `tree_attn`
   - `vllm_ext`
   - `agent_service/inference_service`
   - `archon_weight_sync`
1. Measure in classification lane:
   - expected detected domains/signals
   - expected severity
   - false positive rate
   - missing high-risk findings
1. Run **delegation lane** with full `/review-pr` (non-quick) on the same fixtures.
1. Measure in delegation lane:
   - false positive rate
   - missing high-risk findings
   - number of spawned review tasks per PR

______________________________________________________________________

## 10. Plan Review (Critical self-review)

### Strengths

1. Improves coverage for newly introduced high-risk modules.
1. Shrinks maintenance overhead by moving to domain templates.
1. Prevents cross-platform drift via deterministic derivation.

### Risks

1. Over-broad domains can dilute checklist quality.
1. First migration may temporarily shift severity distributions.
1. Command docs can lag behind taxonomy if sync discipline is not enforced.

### Mitigations

1. Keep L2 signals explicit and path-anchored.
1. Use golden PR cases for pre/post comparison.
1. Add CI consistency gate for derived platform data files.
1. Require wrapper wording updates in the same migration PR as taxonomy changes.

______________________________________________________________________

## 11. Acceptance Criteria

1. Canonical references express all 12 domains + L2 signals and linkage rules.
1. Derived OpenCode/Claude data files regenerate with zero manual edits.
1. No mixed old/new label vocabulary remains after regeneration.
1. Representative PRs trigger expected domains:
   - `tree_attn` PR -> Model Compute & Attention (+ Numerics linkage)
   - `vllm_ext` PR -> Inference Backend & Serving (+ Launcher linkage)
   - service-stack PR -> Service Orchestration (+ Workflow linkage)
   - `archon_weight_sync` PR -> Distributed Runtime (CRITICAL)
1. Legacy high-risk coverage remains intact (Archon/FSDP/Megatron/Reward/Dataset).
1. Task fan-out remains bounded (no uncontrolled template explosion).

______________________________________________________________________

## 12. Out of Scope for v1 (defer)

1. Fully unifying command orchestration syntax across OpenCode and Claude.
1. Creating standalone domain types for every backend keyword.
1. Automatically posting review comments to GitHub.

______________________________________________________________________

## 13. Recommended Next Action

Land **Phase A + B + C** in one migration PR (taxonomy + sync tooling + wrapper
alignment), then run **Phase D** validation immediately using fixed fixtures.

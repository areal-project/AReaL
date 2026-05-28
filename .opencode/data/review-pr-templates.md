# PR Review: Domain Templates Reference

This file contains canonical domain templates for PR review. Referenced by:
`.opencode/command/review-pr.md`

______________________________________________________________________

## Template Selection Rules

1. Select templates by detected L1 domains and L2 signals.
1. Use at most one primary template per domain.
1. Always include **General Logic & Boundary** for non-doc/config-only PRs.
1. Apply cross-domain linkage checks from `review-pr-domains-and-signals.md`.

______________________________________________________________________

## Universal Template

### General Logic & Boundary

```
Applicable: Any non-doc/config-only change
Checklist:
- Boundary condition correctness (empty inputs, singleton, max-size)
- Conditional logic correctness (branch inversion, short-circuit mistakes)
- Error-path behavior (exceptions propagated with actionable context)
- Return-value consistency across code paths
- No newly introduced hidden behavior changes
```

______________________________________________________________________

## Domain 1 Template: Distributed Runtime Review \[comprehensive\]

```
Applicable signals: archon_core, archon_parallel, process_group, fsdp_core, megatron_core, collectives, mesh_dtensor, activation_checkpointing, weight_sync
Checklist:
- Archon engine and checkpoint lifecycle remain aligned with distributed runtime assumptions
- FSDP and Megatron engine invariants still match process-group, sharding, and pipeline assumptions
- Archon parallel-dims and mesh construction still match downstream placement logic
- Process-group creation/usage/cleanup is rank-consistent
- Collective operations are called by all required ranks in consistent order
- DeviceMesh dimensions and DTensor placements are correct for each path
- Activation checkpoint placement remains compatible with parallel and sharding order requirements
- Local/global tensor conversion boundaries are explicit and correct
- Weight version propagation and update ordering are deterministic
- No debug-only barriers left in hot path
```

## Domain 2 Template: Model Compute & Attention Review \[comprehensive\]

```
Applicable signals: tree_attn, sdpa_varlen, sp_cp_attention_mask, triton_kernel, archon_model_family, archon_attention_stack, archon_moe_modeling
Checklist:
- Attention mask semantics preserved under TP/SP/CP
- Archon model-family registration and per-family wiring remain internally consistent
- Archon attention/Ulysses slicing and gather paths preserve layout assumptions
- Archon MoE router, grouped experts, and weight-conversion interfaces remain aligned
- Tree attention index/order invariants are maintained
- Kernel assumptions on dtype/shape/contiguity are satisfied
- No silent behavior change in sequence packing/unpacking
- Tensor layouts remain compatible with downstream modules
```

## Domain 3 Template: Inference Backend & Serving Review \[comprehensive\]

```
Applicable signals: vllm_ext, remote_inference_backend, request_lifecycle
Checklist:
- Request lifecycle (enqueue, execution, cancellation, timeout) is coherent
- Worker state transitions are safe under concurrency
- Backend-specific extension points stay API-compatible
- Error handling does not strand in-flight requests
- Versioning/weight-update interactions are explicit and safe
```

## Domain 4 Template: Service Orchestration Review \[comprehensive\]

```
Applicable signals: service_routing_dataflow, session_consistency
Checklist:
- Gateway/router/data-proxy routing rules are deterministic
- Session affinity and history consistency are preserved
- Controller/worker coordination has no lost-update window
- Async boundaries avoid blocking operations in critical paths
- Failure/retry behavior does not duplicate or drop work
```

## Domain 5 Template: Workflow & Trainer Contract Review \[comprehensive\]

```
Applicable signals: workflow_engine_boundary, dataset_surface, async_contract, weight_version_contract
Checklist:
- RolloutWorkflow and Engine interfaces remain contract-compatible
- Dataset/output structure still matches workflow and trainer consumption expectations
- Async flow uses await consistently and avoids sync I/O in async paths
- Weight update/version handshake is preserved end-to-end
- Trainer lifecycle transitions are valid for all execution branches
- Call ordering assumptions across trainer/workflow/engine are unchanged or justified
```

## Domain 6 Template: API & Config Compatibility Review \[targeted\]

```
Applicable signals: dataclass_schema, cli_compat, backward_compat, project_dependency_config
Checklist:
- Public API signature and default value changes are intentional and compatible
- Dataclass validation remains complete and informative
- CLI options preserve expected compatibility semantics
- New fields include safe defaults or explicit migration handling
- Breaking changes are documented and scoped
- Dependency and build-system changes remain compatible with supported environments
```

## Domain 7 Template: Numerics & Tensor Semantics Review \[targeted\]

```
Applicable signals: shape_dtype, numerical_stability, reward_surface, compile_dynamo, mixed_precision_fp8
Checklist:
- Tensor shape/dtype transitions are explicit and internally consistent
- Numerical stability is protected (log/division/softmax/clamp paths)
- Reward-side numerical behavior remains consistent with workflow consumption expectations
- torch.compile / dynamo assumptions still hold for dynamic shapes and distributed execution
- Mixed-precision behavior is correct for forward + backward + reduce paths
- In-place and view/reshape operations do not corrupt gradient flow
- Device placement and dtype combinations remain legal across code paths
```

## Domain 8 Template: Checkpoint & Recovery Review \[comprehensive\]

```
Applicable signals: dcp_consistency, optimizer_state, resume_compat
Checklist:
- Save/load requires and enforces all-rank participation where needed
- State dict naming/structure is stable or migration-safe
- Optimizer state sharding/gather behavior is consistent
- Resume path restores model + optimizer + version state coherently
- Async checkpoint behavior preserves ordering and durability assumptions
```

## Domain 9 Template: Launcher & Infrastructure Review \[targeted\]

```
Applicable signals: launcher_resource_match, scheduler_contract, rpc_transport, runtime_image_config
Checklist:
- Resource assignment matches declared parallel strategy assumptions
- Scheduler decisions preserve required placement/affinity constraints
- RPC serialization/deserialization keeps shape/dtype/device semantics
- Transport retries/timeouts do not violate idempotency expectations
- Cross-process startup/shutdown ordering is robust
- Runtime image and build configuration remain aligned with supported variants
```

## Domain 10 Template: Low-Risk Hygiene Review \[basic\]

```
Applicable signals: tests_docs_config, logging_import_security, project_docs_metadata
Checklist:
- Tests/docs/config edits are internally consistent and non-misleading
- Logging follows project conventions and avoids sensitive leakage
- No wildcard imports or obvious dependency hygiene regressions
- No accidental secrets/keys/tokens introduced
- Docs build scripts and project templates stay aligned with real contributor workflow
```

## Domain 11 Template: Harness & Agent Infrastructure Review \[targeted\]

```
Applicable signals: skill_definition, platform_command_data, agent_registry_config
Checklist:
- Canonical skills and derived platform data remain structurally aligned
- Command docs still point to the correct data files and execution model
- Agent registry/config changes preserve command discovery and expert routing
- Cross-platform mirrors are regenerated after canonical changes
```

## Domain 12 Template: CI/CD & Release Automation Review \[comprehensive\]

```
Applicable signals: github_workflow_jobs, runner_provisioning, release_delivery
Checklist:
- Workflow triggers, job dependencies, and permissions still enforce required validation
- Runner/image provisioning remains reproducible and compatible with job expectations
- Release, docker, and docs deployment jobs publish the intended artifacts only
- CI changes do not silently skip tests, formatting, or release gates
```

______________________________________________________________________

## Signal-Specific Add-On Checklists

Use these only when corresponding L2 signals are detected.

### `tree_attn` Add-On \[comprehensive\]

```
- Node/edge indexing is deterministic and shape-safe
- Tree traversal order matches attention mask semantics
- FSDP/Megatron/Archon variant modules remain behaviorally aligned
```

### `vllm_ext` Add-On \[comprehensive\]

```
- Server and worker extension hooks still match upstream expectations
- Request pause/resume/cancel semantics remain coherent
- Integration-specific monkey-patches are scoped and guarded
```

### `archon_model_family` Add-On \[comprehensive\]

```
- ModelSpec registration stays unique and complete for supported model types
- Per-family model/args/spec/state-adapter wiring remains consistent
- Pipelining hooks and model-part boundaries stay compatible with runtime assumptions
```

### `archon_moe_modeling` Add-On \[comprehensive\]

```
- Router top-k, gating dtype, and expert grouping semantics remain coherent
- GroupedExperts layout and token reordering assumptions still match expert execution
- MoE weight conversion paths stay consistent with runtime sharding expectations
```

### `service_routing_dataflow` Add-On \[comprehensive\]

```
- Route selection and fallback ordering are deterministic
- Data proxy transformations preserve payload integrity
- Session-key partitioning logic is collision-safe
```

### `remote_inference_backend` Add-On \[targeted\]

```
- Remote backend request/response semantics remain consistent across supported engines
- Backend-specific transport options do not change lifecycle expectations silently
- Shared request payload assumptions remain compatible across remote backends
```

### `weight_sync` Add-On \[comprehensive\]

```
- Versioned updates are monotonic and race-safe
- Broadcast/all-gather points are aligned with consumer expectations
- Local caching behavior cannot serve stale weights indefinitely
```

### `activation_checkpointing` Add-On \[targeted\]

```
- Checkpoint wrappers are applied in a parallelism-safe order
- Selective checkpoint policies still cover the intended modules only
- Activation recompute paths do not break sharding or sequence-parallel assumptions
```

### `reward_surface` Add-On \[targeted\]

```
- AsyncRewardWrapper-facing reward interfaces remain contract-compatible
- Reward outputs keep expected shape, dtype, and per-sample semantics
- Workflow assumptions about reward timing and batching remain valid
```

### `compile_dynamo` Add-On \[targeted\]

```
- torch.compile and dynamo guards still tolerate expected dynamic-shape inputs
- fullgraph and mark_dynamic choices remain compatible with distributed execution paths
- Compile-specific changes do not silently alter runtime fallback behavior
```

### `rpc_transport` Add-On \[targeted\]

```
- RTensor conversion is reversible and metadata-complete
- Batch fetch/request framing preserves ordering and boundaries
- Retry logic does not replay non-idempotent actions incorrectly
```

### `runtime_image_config` Add-On \[targeted\]

```
- Docker base image and build args still match supported backend variants
- Layer ordering preserves expected cache and dependency behavior
- Image contents remain aligned with runtime assumptions documented in the repo
```

### `project_dependency_config` Add-On \[targeted\]

```
- Python/version constraints and extras remain internally consistent
- Lockfile changes match the intended dependency update scope
- Build backend/tooling changes do not break install or publish workflows
```

### `github_workflow_jobs` Add-On \[comprehensive\]

```
- Workflow triggers and job graph still run required validation paths
- Required secrets/permissions are scoped correctly
- Matrix or conditional changes do not silently skip critical jobs
```

### `project_docs_metadata` Add-On \[basic\]

```
- Docs build entrypoints and contributor-facing metadata remain mutually consistent
- Public templates and contributor instructions still match the actual workflow
- Build/preview guidance still points to the supported commands
```

### `skill_definition` / `platform_command_data` Add-On \[targeted\]

```
- Canonical and derived review-pr data files stay in sync after edits
- Command/import paths remain correct after file moves or renames
- Wrapper-specific routing stays out of canonical reference files
```

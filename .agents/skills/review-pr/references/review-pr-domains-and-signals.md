# PR Review: Domain & Signal Detection Reference

This file contains the canonical change-domain and signal detection tables for PR
review. Referenced by: `.agents/skills/review-pr/SKILL.md`

______________________________________________________________________

## Severity-to-Review-Depth Mapping

- **CRITICAL**: use `comprehensive` review depth
- **HIGH**: use `comprehensive` review depth
- **MEDIUM**: use `targeted` review depth
- **LOW**: use `basic` review depth

______________________________________________________________________

## L1 Domains and L2 Signals

## Domain 1: Distributed Runtime (CRITICAL/HIGH)

| L2 Signal                  | File Path Pattern                                                                              | Code Pattern                                                                                                                                                              |
| -------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `archon_core`              | `areal/experimental/engine/archon_engine.py`, `areal/experimental/engine/archon_checkpoint.py` | `ArchonEngine`, `ArchonCheckpointManager`, `archon`                                                                                                                       |
| `archon_parallel`          | `areal/experimental/models/archon/`, `parallel_dims.py`, `parallelize.py`                      | `ArchonParallelDims`, `_build_mesh`, `apply_moe_ep_tp`, `apply_tp`, `apply_cp`, `ExpertTensorParallel`, `etp`, `parallelize_module`, `ColwiseParallel`, `RowwiseParallel` |
| `process_group`            | `areal/engine/fsdp_utils/`, `areal/engine/megatron_utils/`, `areal/experimental/engine/`       | `new_group`, `ProcessGroup`, `dist.get_rank(`                                                                                                                             |
| `fsdp_core`                | `areal/engine/fsdp_engine.py`, `areal/engine/fsdp_utils/`                                      | `FSDP`, `fully_shard`, `FullyShardedDataParallel`                                                                                                                         |
| `megatron_core`            | `areal/engine/megatron_engine.py`, `areal/engine/megatron_utils/`                              | `MegatronEngine`, `pipeline`, `micro-batch`                                                                                                                               |
| `collectives`              | `areal/engine/`, `areal/infra/rpc/`                                                            | `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `broadcast`, `barrier`                                                                                        |
| `mesh_dtensor`             | `areal/experimental/models/archon/`, `areal/engine/fsdp_utils/`                                | `DeviceMesh`, `DTensor`, `Shard(`, `Replicate(`, `distribute_tensor`                                                                                                      |
| `activation_checkpointing` | `areal/experimental/models/archon/activation_checkpoint.py`, `areal/models/`, `areal/engine/`  | `activation_checkpoint`, `checkpoint_wrapper`, `selective_checkpoint`                                                                                                     |
| `weight_sync`              | `areal/experimental/engine/archon_weight_sync.py`, `areal/api/engine_api.py`, `areal/engine/`  | `WeightUpdateMeta`, `set_version`, `update_weights`                                                                                                                       |

## Domain 2: Model Compute & Attention (HIGH/MEDIUM)

| L2 Signal                | File Path Pattern                                                                                                                                          | Code Pattern                                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `tree_attn`              | `areal/models/tree_attn/`                                                                                                                                  | `TreeAttention`, `tree_attn`, `TreeNode`, `tree`                                                       |
| `sdpa_varlen`            | `attention/sdpa.py`, `attention/varlen.py`, `areal/models/tree_attn/`                                                                                      | `sdpa`, `flash_attn`, `varlen`, `causal_mask`                                                          |
| `sp_cp_attention_mask`   | `areal/models/tree_attn/`, `areal/experimental/models/archon/attention/`                                                                                   | `SequenceParallel`, `context_parallel`, `mask`                                                         |
| `triton_kernel`          | `areal/models/tree_attn/triton_kernel.py`                                                                                                                  | `triton`, `kernel`, `autotune`                                                                         |
| `archon_model_family`    | `areal/experimental/models/archon/model_spec.py`, `areal/experimental/models/archon/qwen*/`                                                                | `ModelSpec`, `register_model_spec`, `supported_model_types`, `state_dict_adapter`, `rope`              |
| `archon_attention_stack` | `areal/experimental/models/archon/attention/`, `areal/experimental/models/archon/ulysses.py`                                                               | `ulysses_slice_inputs`, `ulysses_gather_output`, `gather_seq_scatter_heads`, `sdpa`, `varlen`          |
| `archon_moe_modeling`    | `areal/experimental/models/archon/moe/`, `areal/experimental/models/archon/expert_parallel.py`, `areal/experimental/models/archon/moe_weight_converter.py` | `TokenChoiceTopKRouter`, `RouterGateLinear`, `GroupedExperts`, `MoEWeightConverter`, `expert_parallel` |

## Domain 3: Inference Backend & Serving (HIGH)

| L2 Signal                  | File Path Pattern                                              | Code Pattern                                                     |
| -------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------- |
| `vllm_ext`                 | `areal/engine/vllm_ext/`                                       | `areal_vllm_server`, `vllm_worker_extension`, `pause_generation` |
| `remote_inference_backend` | `areal/engine/vllm_remote.py`, `areal/engine/sglang_remote.py` | `vllm`, `sglang`, `OpenAI`, `request`, `response`                |
| `request_lifecycle`        | `areal/engine/`, `areal/infra/launcher/`                       | `enqueue`, `dequeue`, `cancel`, `timeout`                        |

## Domain 4: Service Orchestration (HIGH)

| L2 Signal                  | File Path Pattern                                                                                                                                                                               | Code Pattern                                                     |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `service_routing_dataflow` | `areal/experimental/agent_service/gateway/`, `areal/experimental/agent_service/router/`, `areal/experimental/inference_service/data_proxy/`, `areal/experimental/inference_service/controller/` | `route`, `gateway`, `router`, `DataProxy`, `controller`, `batch` |
| `session_consistency`      | `areal/experimental/agent_service/`, `areal/experimental/inference_service/`                                                                                                                    | `session`, `affinity`, `history`, `state`                        |

## Domain 5: Workflow & Trainer Contract (HIGH/MEDIUM)

| L2 Signal                  | File Path Pattern                                                                               | Code Pattern                                        |
| -------------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `workflow_engine_boundary` | `areal/workflow/`, `areal/trainer/`, `areal/engine/`                                            | `RolloutWorkflow`, `arun_episode`, `agenerate`      |
| `dataset_surface`          | `areal/dataset/`                                                                                | `DataLoader`, `IterableDataset`, `get_*_dataset`    |
| `async_contract`           | `areal/workflow/`, `areal/experimental/agent_service/`, `areal/experimental/inference_service/` | `async def`, `await`, `aiofiles`, `asyncio`         |
| `weight_version_contract`  | `areal/api/engine_api.py`, `areal/workflow/`, `areal/trainer/`                                  | `WeightUpdateMeta`, `set_version`, `weight version` |

## Domain 6: API & Config Compatibility (MEDIUM)

| L2 Signal                   | File Path Pattern                     | Code Pattern                                                                     |
| --------------------------- | ------------------------------------- | -------------------------------------------------------------------------------- |
| `dataclass_schema`          | `areal/api/`                          | `@dataclass`, `field(`, `__post_init__`                                          |
| `cli_compat`                | `areal/api/cli_args.py`               | `Literal`, `help`, `default`                                                     |
| `backward_compat`           | `areal/api/`, `areal/infra/launcher/` | `deprecated`, `compat`, `version`                                                |
| `project_dependency_config` | `pyproject.toml`, `uv.lock`           | `requires-python`, `dependencies`, `optional-dependencies`, `build-system`, `uv` |

## Domain 7: Numerics & Tensor Semantics (MEDIUM)

| L2 Signal             | File Path Pattern                                                               | Code Pattern                                            |
| --------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `shape_dtype`         | `areal/engine/`, `areal/models/`, `areal/trainer/`                              | `.view(`, `.reshape(`, `dtype=`, `.contiguous(`         |
| `numerical_stability` | `areal/engine/`, `areal/reward/`, `areal/utils/functional/`                     | `log(`, `softmax`, `eps=`, `.clamp(`, `nan`, `inf`      |
| `reward_surface`      | `areal/reward/`                                                                 | `reward_fn`, `AsyncRewardWrapper`, `MathVerifyWorker`   |
| `compile_dynamo`      | `areal/experimental/models/archon/compile.py`, `areal/models/`, `areal/engine/` | `torch.compile`, `_dynamo`, `mark_dynamic`, `fullgraph` |
| `mixed_precision_fp8` | `areal/engine/megatron_utils/fp8/`, `areal/experimental/models/archon/`         | `fp8`, `bf16`, `fp16`, `mixed precision`                |

## Domain 8: Checkpoint & Recovery (CRITICAL/HIGH)

| L2 Signal         | File Path Pattern                                                   | Code Pattern                                    |
| ----------------- | ------------------------------------------------------------------- | ----------------------------------------------- |
| `dcp_consistency` | `areal/utils/async_checkpoint.py`, `areal/engine/**/checkpoint*.py` | `dcp.save`, `dcp.load`, `DistributedCheckpoint` |
| `optimizer_state` | `areal/engine/fsdp_utils/checkpoint.py`, `areal/utils/saver.py`     | `optimizer state`, `state_dict`                 |
| `resume_compat`   | `areal/utils/recover.py`, `areal/utils/saver.py`                    | `resume`, `load_state_dict`, `migration`        |

## Domain 9: Launcher & Infrastructure (HIGH/MEDIUM)

| L2 Signal                 | File Path Pattern                                                      | Code Pattern                                              |
| ------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------- |
| `launcher_resource_match` | `areal/infra/launcher/`                                                | `LaunchConfig`, `RayLauncher`, `SlurmLauncher`            |
| `scheduler_contract`      | `areal/infra/scheduler/`, `areal/scheduler/`                           | `Scheduler`, `placement`, `resource`                      |
| `rpc_transport`           | `areal/infra/rpc/`, `areal/experimental/inference_service/data_proxy/` | `RTensor`, `serialize`, `rpc`, `fetch`                    |
| `runtime_image_config`    | `Dockerfile`, `.dockerignore`                                          | `FROM`, `ARG`, `RUN`, `ENV`, `COPY`, `uv sync`, `VARIANT` |

## Domain 10: Low-Risk Hygiene (LOW)

| L2 Signal                 | File Path Pattern                                                                                                                                                       | Code Pattern                                                                                                 |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `tests_docs_config`       | `tests/`, `docs/`, `*.md`, `*.yaml`, `*.json`, `*.toml`                                                                                                                 | -                                                                                                            |
| `logging_import_security` | `areal/`, `examples/`                                                                                                                                                   | `getLogger`, `print(`, `import *`, `api_key`, `token`, `password`                                            |
| `project_docs_metadata`   | `docs/build_all.sh`, `docs/generate_cli_docs.py`, `docs/en/`, `docs/zh/`, `README.md`, `CONTRIBUTING.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/` | `jupyter-book`, `generate_cli_docs`, `build_all`, `_build`, `checklist`, `template`, `contributing`, `usage` |

## Domain 11: Harness & Agent Infrastructure (MEDIUM/HIGH)

| L2 Signal               | File Path Pattern                                                             | Code Pattern                                                 |
| ----------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `skill_definition`      | `.agents/skills/**/SKILL.md`, `.agents/skills/**/references/`                 | `description:`, `## Workflow`, `## Reference Files`, `skill` |
| `platform_command_data` | `.claude/commands/`, `.claude/data/`, `.opencode/command/`, `.opencode/data/` | `@.`, `/review-pr`, `/create-pr`, `data/`, `task(`           |
| `agent_registry_config` | `.codex/config.toml`, `.codex/agents/`, `AGENTS.md`, `CLAUDE.md`              | `agents`, `skills`, `registry`, `subagent`, `config.toml`    |

## Domain 12: CI/CD & Release Automation (HIGH/CRITICAL)

| L2 Signal              | File Path Pattern                                                                                                          | Code Pattern                                              |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `github_workflow_jobs` | `.github/workflows/*.yml`                                                                                                  | `jobs:`, `runs-on:`, `needs:`, `if:`, `workflow_dispatch` |
| `runner_provisioning`  | `.github/workflows/bake-gcp-image.yml`, `.github/workflows/runner-heartbeat.yml`                                           | `gcp`, `runner`, `image`, `heartbeat`                     |
| `release_delivery`     | `.github/workflows/build-docker-image.yml`, `.github/workflows/tag-release-image.yml`, `.github/workflows/deploy-docs.yml` | `docker`, `tag`, `release`, `pages`, `publish`            |

______________________________________________________________________

## Must-Not-Regress Core Coverage

The refactor must preserve these existing review surfaces:

- Archon core: `areal/experimental/models/archon/`,
  `areal/experimental/engine/archon_engine.py`
- FSDP core: `areal/engine/fsdp_utils/`, `areal/engine/fsdp_engine.py`
- Megatron core: `areal/engine/megatron_engine.py`, `areal/engine/megatron_utils/`
- Reward: `areal/reward/`
- Dataset: `areal/dataset/`
- Trainer: `areal/trainer/`
- Harness: `.agents/`, `.claude/`, `.opencode/`, `.codex/`
- CI/CD and release: `.github/workflows/`, `Dockerfile`, `pyproject.toml`

______________________________________________________________________

## Cross-Domain Linkage Rules

| Detected Signal                                | Auto-Linked Review                                  |
| ---------------------------------------------- | --------------------------------------------------- |
| `archon_core` or `archon_parallel`             | Model Compute & Attention checks                    |
| `archon_model_family` or `archon_moe_modeling` | Numerics & Tensor Semantics checks                  |
| `tree_attn`                                    | Numerics & Tensor Semantics checks                  |
| `reward_surface`                               | Workflow & Trainer Contract checks                  |
| `compile_dynamo`                               | Distributed Runtime checks                          |
| `vllm_ext`                                     | Launcher & Infrastructure checks                    |
| `service_routing_dataflow`                     | Workflow & Trainer async-contract checks            |
| `weight_sync`                                  | DTensor/process-group/checkpoint interaction checks |
| `rpc_transport`                                | Distributed Runtime synchronization checks          |
| `mixed_precision_fp8` + Distributed Runtime    | mesh + weight-sync compatibility checks             |
| `runtime_image_config`                         | Inference Backend & Serving checks                  |
| `project_dependency_config`                    | API & Config Compatibility checks                   |
| `github_workflow_jobs` or `release_delivery`   | Launcher & Infrastructure checks                    |
| `skill_definition` or `platform_command_data`  | Low-Risk Hygiene checks                             |

______________________________________________________________________

## Risk Identification Guidance

### Distributed Runtime Risks

- Archon mesh construction or parallel-dims mismatch
- EP/TP/CP application order errors in Archon parallelization
- Activation checkpoint placement violating TP/CP/FSDP ordering assumptions
- Archon engine lifecycle drift around distributed setup and checkpoint boundaries
- Collective call order mismatch across ranks
- Wrong process-group scope in rank-sensitive logic
- Mesh dimension mismatch and invalid DTensor placement
- Weight version drift between rollout and training workers

### Model Compute & Attention Risks

- Attention mask inconsistency under TP/SP/CP paths
- Tree attention index/routing mismatch
- Archon model-family registration or per-family wiring drift
- Archon MoE router/expert behavior diverging from weight-conversion expectations
- Archon Ulysses slicing/gather semantics mismatching attention layout assumptions
- Kernel assumptions violating dtype/shape invariants
- Sequence packing alignment errors

### Service Orchestration Risks

- Session affinity or history drift across gateway/router/data proxy
- Async message handling holes and dropped tasks
- Controller/worker lifecycle desynchronization

### Inference Backend & Serving Risks

- Request lifecycle inconsistencies (enqueue/cancel/timeout)
- Worker state transitions leaving requests stranded
- Backend extension hooks drifting from runtime expectations

### Workflow & Trainer Contract Risks

- Workflow-engine contract drift across async boundaries
- Weight version handshake mismatch between rollout and train
- Trainer lifecycle transition inconsistencies

### API & Config Compatibility Risks

- Breaking config/schema changes without migration path
- Dataclass or CLI default changes altering behavior silently
- Missing validation for newly introduced fields
- Dependency or build-system pin changes breaking supported environments

### Numerics & Tensor Semantics Risks

- Silent shape/dtype mismatch under distributed paths
- Unstable numerical operations in loss/reward logic
- torch.compile or dynamo guard changes breaking graph assumptions
- Mixed-precision interaction regressions

### Checkpoint & Recovery Risks

- Partial-rank checkpoint participation
- Incompatible state key evolution
- Resume path breaking optimizer/model synchronization

### Launcher & Infrastructure Risks

- Resource assignment mismatching parallel strategy assumptions
- RPC transport metadata loss (shape/dtype/device)
- Startup/shutdown ordering races across processes
- Runtime image or build-arg drift from supported inference/training variants

### Low-Risk Hygiene Risks

- Docs/config drift from actual runtime behavior
- Logging or import hygiene regressions
- Sensitive data exposure in logs or config
- Documentation/build scripts or project templates drifting from actual workflow

### Harness & Agent Infrastructure Risks

- Skill and command docs drifting from actual platform behavior
- Cross-platform data files falling out of sync with canonical references
- Agent registry/config changes breaking expert routing or command discovery

### CI/CD & Release Automation Risks

- Workflow trigger or job dependency changes skipping required validation
- Runner provisioning drift causing flaky or non-reproducible CI
- Release or docs deployment jobs publishing the wrong artifacts

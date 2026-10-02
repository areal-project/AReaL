# Native Megatron Muon implementation (runtime and quality checks passed)

## Design

The first implementation uses native `TensorParallelMuon`, chained with Adam for
embeddings, output weights and non-matrix parameters. It does not use
`LayerWiseDistributedOptimizer`. Optimizer states remain replicated across DP. Set
`actor.megatron.ddp.use_distributed_optimizer: false` explicitly. Megatron DDP then
all-reduces full gradients; tensor-parallel behavior is delegated to MCore's
`muon_tp_mode`. The existing scheduler and gradient clipping operate on the native
chained optimizer.

The checkpoint manager now has an explicit replicated-optimizer opt-in, used only by the
Muon engine path. It calls native `sharded_state_dict` without the byte-sharded
distributed-Adam metadata. The ordinary Adam/SGD path retains `dp_reshardable` metadata.
Passing the new opt-in requires sharded checkpoints. GPU job
`job-018c661c528fa9e5e7ca586f` validated this save/load path on both DP2/TP1 and
DP2/TP2, with a newly constructed optimizer before restore.

BF16 and FP32 are accepted for this path. FP16, LoRA, FP8, precision-aware optimizer
states, Megatron FSDP, unwrapped models, parameter-gather overlap, AWEX flat-buffer
residency, and AdamW DTE delta transfer are rejected. The provided experiment disables
offload. No offload compatibility is claimed.

## Files

- `areal/api/cli_args.py`: Muon choice, nine native options and validation.
- `areal/engine/megatron_utils/muon.py`: option mapping and early compatibility checks.
- `areal/engine/megatron_engine.py`: native factory integration and checkpoint opt-in.
- `areal/engine/megatron_utils/checkpointer.py`: replicated state template handling.
- `examples/math/gsm8k_grpo_megatron_muon.yaml`: three-update DP2 GRPO recipe.
- `examples/math/gsm8k_rl.py`: honor configured dataset splits, including slices.
- `tests/test_megatron_muon_config.py`: structured config and unsupported combinations.
- `tests/test_megatron_optimizer_config.py`: native mapping and Adam/SGD regression.
- `tests/test_megatron_async_save.py`: replicated optimizer template metadata.
- `tests/test_megatron_muon_distributed.py` and corresponding torchrun script: actual
  Qwen engine DP2/TP1 and DP2/TP2 update/restore checks.

## Execution scripts

RSI command `areal_validation` accepts `script`, which receives workspace and output
arguments. Use `research/muon/setup_environment.sh` first, then
`research/muon/check_environment.sh`, `research/muon/run_distributed.sh`, and
`research/muon/run_grpo.sh`. The scripts use allocation 972320, `--mpi=none`, and the
existing `areal-dev-sglang-0821.sif` image. Local environment coordinates are recorded
separately in `local_environment.json`. They are experiment coordinates, not application
defaults. GPU availability is rechecked per job.

The distributed test executes three optimizer updates with rank-distinct DP inputs,
verifies finite loss/gradient norm/LR and changing weights, checks DP parameter hashes
after every update, and saves through the actual checkpoint manager. It runs a fourth
update, restores step three, compares all optimizer state tensors and scheduler state,
then repeats the fourth update and compares against the uninterrupted result. Every rank
writes its own JSON report.

## Current evidence and limitations

Model header and tokenizer-config inspection passed. Targeted Ruff checks passed at an
intermediate revision. Job `job-0975a4e51e6e3e67e1a9b7e3` completed engine import and
targeted unit tests: **64 passed, 2 skipped**. The skips are pre-existing async-save
fixture isolation cases. Distributed and checkpoint tests passed (details below). GRPO
passed in job `job-a517f7dcead05c0b7e3c7ca3`; see `RESULTS.md`. Full pre-commit and
lock-file generation passed in `job-2dae2c6db8b5b747fea8228e`; final artifact checks
also passed in `job-99bd98b07e4ceed4192ef0af`.

Environment setup failures are recorded by RSI jobs. The existing container's Git proxy
referred to an unavailable loopback proxy on the GPU node. After clearing it only in the
installation subprocess, pip reported an invalid inherited CA bundle path. Setup now
selects the container's certifi bundle; TLS verification remains enabled. Neither CUDA
nor driver packages are changed.

The PyPI index exposes emerging-optimizers 0.3.0, not 0.2.0. The dependency is now
pinned to 0.3.0; its native `_init_group(skip_non_grad_params=False)` interface remains
compatible with the MCore 0.19 factory. Actual import succeeded in the isolated
container environment. BF16 update and recovery tests passed on the two tested DP/TP
layouts.

## Distributed and recovery evidence

`job-018c661c528fa9e5e7ca586f`: **2 passed in 311.03s**, no skipped GPU case. The
explicit model was `/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct`. The command
was `python -m pytest -q -s tests/test_megatron_muon_distributed.py`, with
`AREAL_MUON_TEST_MODEL` set to that path. The execution wrapper is
`research/muon/run_distributed.sh`.

All six rank reports are preserved under `research/muon/evidence/distributed/`. Each
rank has 112 Muon matrix parameters and 86 Adam parameters; embedding, bias and
normalization parameters are asserted to use Adam. The Muon matrix hash changes after
three updates, and 112 finite momentum tensors are present. Different DP inputs produce
identical post-update DP parameter hashes.

| Layout  | Rank 0 losses, first three updates | Learning rates        | Restore                       |
| ------- | ---------------------------------- | --------------------- | ----------------------------- |
| DP2/TP1 | 12.83257, 9.72126, 3.49840         | 0.001, 0.0009, 0.0008 | Exact state and fourth update |
| DP2/TP2 | 12.82713, 9.69527, 4.15968         | 0.001, 0.0009, 0.0008 | Exact state and fourth update |

The test asserts finite model parameters, loss, gradient norm and learning rate, and
`update_successful == 1` at every update. The configured gradient clipping threshold is
1; reported gradient norms are the native pre-clipping norms. Optimizer state checks
include Muon momentum, scalar Adam moments and BF16 FP32 master parameters. The
scheduler restores step 3 and advances to step 4; the resumed fourth update uses LR
0.0007 and matches uninterrupted training.

Checkpoint directories remain in this run's execution output at
`execution-outputs/job-018c661c528fa9e5e7ca586f/distributed/`, under each
`test_muon_three_updates_and_ch*/checkpoint-step3` directory. Large checkpoint payloads
are not duplicated into the source worktree.

Runtime verification covers BF16, DP2, TP1/TP2, blockwise Muon TP mode, and synchronous
saving on one node. FP32 and other native TP modes are accepted by configuration but
have not been exercised. Async checkpoint saving, optimizer CPU offload, and multi-node
training are not claimed as tested. Unsupported FP16, LoRA, FP8, state sharding and
parameter-gather overlap fail at startup.

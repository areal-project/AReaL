# Megatron Muon investigation

This document preserves the investigation history, including failures. Current
validation results are in `RESULTS.md`.

## Workspace and boundaries

- Worktree:
  `/storage/openpsi/users/fenghui/projects/AReaL-RSI/.rsi/research-lab/worktrees/research_58654ea421ca4619/research_ce92f6a2a491b52e7fd5b227`
- Branch: `rsi/research_58654ea421ca4619/research_ce92f6a2a491b52e7fd5b227`
- Git common directory: `/storage/openpsi/users/fenghui/projects/AReaL/.git`
- Original checkout is not updated. No other run artifacts were read.
- Task-specific implementations on other branches are excluded by the research boundary.

## Initial findings

`MegatronEngine._create_optimizer` accepts Adam and SGD. The checkpoint manager asserts
`use_distributed_optimizer` and asks for a sharded optimizer state with `dp_reshardable`
metadata. Merely accepting the string `muon` is insufficient. Model DDP construction
occurs earlier in `make_mcore_model`, including delegated bridge paths. The DDP gradient
layout must agree with the optimizer.

The existing LR scheduler consumes optimizer parameter groups. Validation needs to
include both Muon and Adam groups, restore their moments and master weights, and compare
a resumed update against an uninterrupted fourth step.

The local interpreter is Python 3.10.15 with Megatron 0.6.0 and Torch
2.4.0+ppu1.4.0.dev.oe; emerging-optimizers, SGLang, and pre-commit are missing. This is
not the requested training environment. Allocation 972320 was running at inspection. A
read-only probe of the existing SGLang image is pending.

## Official component sources

- [MCore 0.19 optimizer API](https://docs.nvidia.com/megatron-core/developer-guide/0.19.0/apidocs/core/core.optimizer.html)
- [Native optimizer factory](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.0/megatron/core/optimizer/__init__.py)
- [Native config](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.0/megatron/core/optimizer/optimizer_config.py)
- [Native layer-wise optimizer](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.0/megatron/core/optimizer/layer_wise_optimizer.py)
- [Official Emerging Optimizers project](https://github.com/NVIDIA-NeMo/Emerging-Optimizers)

The installed Emerging Optimizers 0.3.0 wheel metadata declares `torch` and `absl-py`
dependencies; both are supplied by the recorded environment.

The factory requires emerging-optimizers >=0.2 and rejects FP16. Layer-wise mode also
rejects optimizer-step parameter-gather overlap. Muon has a scalar Adam fallback.
Layer-wise state serialization documents a fixed DP topology. These source findings
still need confirmation against the installed 0.19 wheel and actual distributed
execution.

## Approval history

Initially requested approval for OptimizerConfig Muon fields and an optional dependency.
`get_state` subsequently returned user guidance `message_cf44e249878e112f79abd7a3`:
proceed directly without further confirmation. Implementation began after that
authorization. No launcher/scheduler logic has been changed. The initial
emerging-optimizers range was not installable from PyPI and has been replaced with the
available 0.3.0 release.

## Initial acceptance checklist

- Config mapping and invalid-combination tests; Adam/SGD regression.
- Matrix vs embedding/bias/norm routing.
- At least three actual finite optimizer updates with changing weights.
- Multi-GPU DP/TP completion and gradient synchronization.
- Checkpoint save/load, state equality, scheduler equality, and fourth update.
- GSM8K GRPO rollout/reward/evaluation and at least three training updates.
- Full pre-commit checks.

## Model file inspection

The explicitly requested model directory
`/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct` contains a Qwen2 BF16
configuration (28 layers, hidden size 1536, vocabulary 151936). The safetensors header
contains 338 tensors, and every tensor offset is within the 3,087,467,144 byte file. The
embedding shape is `[151936, 1536]`. Tokenizer configuration specifies Qwen2Tokenizer
and includes a chat template. These are file integrity checks, not proof of successful
runtime loading.

The RSI `run_job` submission and subsequent state/report calls have not returned at the
time of this note; no job identifier or experiment result is available.

## Experiment history (not acceptance)

- `job-50eaf8b9578d268a7dde199d`: setup failed; initial redirected logs were not
  published for a failed job. Subsequent scripts stream their logs.
- `job-42dc3553a2e35d44ab8d4e08`: mbridge clone failed because a GPU-node Git proxy
  pointed to unavailable loopback port 13661.
- `job-ef04d0b80c3f2c9d592e8663`: invalid inherited CA bundle path; switched to the
  existing container certifi bundle without disabling TLS verification.
- `job-be00ac520795cec4e64955ab`: PyPI has no emerging-optimizers 0.2.0 wheel; changed
  the dependency pin to 0.3.0 after checking its official API.
- `job-bab567a01ab20fc98add0f24`: deliberately terminated the obsolete install
  subprocess (exit 143), which was slowly downloading a Transformers version outside the
  repository lock. The Slurm allocation was retained.

Container mbridge is already the exact locked commit
`310e8fb35ccf4fcd4419d32973e563a6d43ee5fb`. Transformers 5.3.0 matches the actual lock
and override. Setup now preserves both, installing only MCore 0.19, Emerging Optimizers
0.3.0 and the configured Bridge 0.6 wheel. The early unvalidated candidate
`candidate_be57f161499d616916b272c1` has been withdrawn because its dependency range was
not installable.

GPU job `job-cf6b7366e7083f54966d4ecb` failed at checkpoint save: the direct-manager
test omitted destination directory creation required by MCore 0.19. Fixed the test to
create its checkpoint directory before calling the manager. Both DP2/TP1 and DP2/TP2
remain pending final recovery validation.

GRPO job `job-035e3317b83ce56a74e49b5a` loaded local GSM8K slices, initialized both
actor and rollout engines, then failed during proxy initialization because the baseline
example used the default admin key on a non-loopback host. The Muon recipe now accepts a
per-run random admin secret from the environment. The run script generates it without
printing it and uses umask 077. Authentication is not bypassed. No GRPO updates occurred
in that failed job.

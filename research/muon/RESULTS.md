# Historical Muon validation results

This file records the earlier one-epoch Muon validation in a separate checkout. For the
three-epoch comparison against the unchanged Megatron GSM8K baseline in the current
repository, use `BASELINE_COMPARISON.md` and its 2026-10-03 evidence directory. The
paths and job identifiers below refer to the earlier run only.

Historical status: **the one-epoch runtime and quality checks passed**.

## Code location

- Worktree:
  `/storage/openpsi/users/fenghui/projects/AReaL-RSI/.rsi/research-lab/worktrees/research_58654ea421ca4619/research_ce92f6a2a491b52e7fd5b227`
- Branch: `rsi/research_58654ea421ca4619/research_ce92f6a2a491b52e7fd5b227`
- Shared Git repository: `/storage/openpsi/users/fenghui/projects/AReaL/.git`
- The original checkout is not updated. Its user changes are preserved.

## Environment

The scripts use existing Slurm allocation `972320` with `srun --mpi=none`, the existing
`/storage/openpsi/images/areal-dev-sglang-0821.sif` image and the isolated Python
environment recorded in `local_environment.json`. CUDA/driver packages were not rebuilt
or replaced.

| Component           | Actual version                                 |
| ------------------- | ---------------------------------------------- |
| Python              | 3.12.3                                         |
| PyTorch             | 2.9.1+cu129                                    |
| Megatron Core       | 0.19.0                                         |
| Emerging Optimizers | 0.3.0                                          |
| Megatron Bridge     | 0.6.0 (repository source wheel)                |
| Transformers        | 5.3.0 (repository override)                    |
| SGLang              | 0.5.10.post1                                   |
| GPU                 | NVIDIA L20X; 8 available on the allocated node |

Model: `/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct`. Data:
`/storage/openpsi/data/gsm8k`, main split, local Parquet files.

## Checks

| Check                                                                     | Result                          | Evidence                                                  |
| ------------------------------------------------------------------------- | ------------------------------- | --------------------------------------------------------- |
| Config mapping, invalid options, Adam/SGD regression, checkpoint metadata | 66 passed, 2 pre-existing skips | `evidence/quality-final/pytest-final.log`                 |
| BF16 DP2/TP1 and DP2/TP2 updates and checkpoint restore                   | 2 passed, 6 ranks completed     | `evidence/distributed/distributed.log` and per-rank JSON  |
| GSM8K GRPO, three updates, rollout/reward/evaluation                      | Passed, exit code 0             | `evidence/grpo/grpo.log` and `evidence/grpo/summary.json` |
| Full pre-commit, both lockfiles, generated CLI docs                       | Passed                          | `evidence/quality-first/pre-commit-final.log`             |

The distributed reports include parameter groups/shapes, finite loss and pre-clipping
gradient norm, learning rates, nonzero Muon momentum, changing Muon matrix hashes,
scheduler state, and optimizer state hashes. The test asserts DP parameter equality
after every update with different DP inputs. It saves model/optimizer/scheduler/RNG,
constructs a fresh optimizer, loads step 3 and verifies that the resumed fourth step is
bitwise identical to an uninterrupted fourth step. There are no skipped distributed
cases.

The two existing unit skips concern async-save fixture isolation when the Megatron
engine is imported first. No missing hardware was counted as a pass.

## Commands

Experiments were submitted with RSI's `areal_validation` command, passing one of the
scripts below as `script`. Each script receives the frozen workspace and output
directory as arguments, and uses the existing Slurm allocation. `wait_for_job` was used
to retrieve the submitted job's result.

```bash
# Within the recorded Python/container environment:
python -m pytest -q tests/test_megatron_muon_config.py \
  tests/test_megatron_optimizer_config.py tests/test_megatron_async_save.py

export AREAL_MUON_TEST_MODEL=/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct
python -m pytest -q -s tests/test_megatron_muon_distributed.py

# The wrapper sets explicit model/data/output paths and generates a random
# proxy admin secret without printing it. Create the output directory first.
bash research/muon/run_grpo.sh "$PWD" /absolute/path/to/output
```

Actual wrappers: `setup_environment.sh`, `run_distributed.sh`, `run_grpo.sh`,
`run_quality_gate.sh` in this directory. The GRPO configuration is
`examples/math/gsm8k_grpo_megatron_muon.yaml`, invoked by
`examples/math/gsm8k_rl.py --config`.

## Supported path and limits

This is ordinary native `TensorParallelMuon` chained with scalar Adam. Optimizer states
are replicated across DP; ordinary Megatron DDP all-reduces gradients. This is **not**
layer-wise distributed Muon or sharded optimizer state. Set
`actor.megatron.ddp.use_distributed_optimizer: false` explicitly.

The runtime tests cover BF16, DP2, TP1/TP2, blockwise TP mode, synchronous
checkpointing, clipping and scheduler restore on one node. FP32 and other native TP
modes have config coverage only. Multi-node, offload and async-save runtime support have
not been verified. FP16, LoRA, FP8, precision-aware optimizer states, FSDP,
parameter-gather overlap, AWEX residency and AdamW DTE are rejected for this
integration.

Failed experiments remain recorded in `INVESTIGATION.md`. Neither startup success nor
unit tests are substituted for runtime acceptance.

## GSM8K GRPO result

Job `job-a517f7dcead05c0b7e3c7ca3` succeeded with exit code 0. The frozen experiment
used the Muon YAML and the explicit model/data paths above. `summarize_grpo.py`
extracted all logged scalar metrics and checked finiteness, three successful updates,
evaluation records, weight-sync timing and the published recovery pointer. Its output is
`evidence/grpo/summary.json`.

| Update | Actor loss (mean) | Pre-clipping gradient norm | LR     | Mean task reward |
| ------ | ----------------- | -------------------------- | ------ | ---------------- |
| 1      | -0.00056158       | 3.9404                     | 0.0001 | 0.625            |
| 2      | 0                 | 0                          | 0.0001 | 0.75             |
| 3      | 0.00030430        | 3.6962                     | 0.0001 | 0.375            |

The log denominator is six dataset batches per epoch; `total_train_steps: 3` stops this
run after three updates. All update-success flags equal 1. Update 2 has zero normalized
advantages (`ppo_actor/advantage_zero_fraction = 1`) and zero gradient; this is
recorded, not a skipped optimizer step. The independent distributed test additionally
proves nonzero Muon momentum and changing matrix weights.

There are 46 saved rollout samples, including prefetched training samples and 8
evaluation samples from four questions with two samples each. Evaluation reward is
**0**; no accuracy gain is claimed from this short test. Evaluation records use weight
version 3; rollout records span versions 0 through 3, and all three updates report
positive NCCL weight synchronization time. No logged scalar is NaN or Inf.

The GRPO recovery payload includes optimizer state and was published after the third
update. Its generation name contains `step00000002` because the trainer uses zero-based
global steps. The output directory is:

`/storage/openpsi/users/fenghui/projects/AReaL-RSI/.rsi/research-lab/runs/research_58654ea421ca4619/execution-outputs/job-a517f7dcead05c0b7e3c7ca3`

The run log has no Python traceback. SGLang prints SIGTERM/SIGQUIT child-exit messages
during the scheduler's explicit process-tree shutdown after training has completed; the
overall job exits successfully. These teardown messages are retained in the original
log.

## Quality gate and lockfiles

`job-2dae2c6db8b5b747fea8228e` completed `pre-commit run --all-files` successfully after
hooks formatted the new research notes and generated both lockfiles and CLI references.
No hook was bypassed. clang-format reports "no files to check" because this Python
change has no matching C/C++ source. The same job reran the related unit suites: **66
passed, 2 pre-existing skips**.

The generated lockfiles add Emerging Optimizers 0.3.0. uv 0.11.8 also refreshed Darwin
x86_64 selections to Torch/Torchaudio 2.2.2, Torchvision 0.17.2 and Ray 2.49.2, and
removed wheel entries outside the configured platforms. The Linux training selections,
including MCore 0.19 and Torch 2.9.1+cu129 in the SGLang lock, remain unchanged. These
are generated resolver changes; Darwin runtime support was not tested. No CUDA/driver
installation was done.

The full first-pass resolver log remains in the quality job output as
`pre-commit-first.log` (about 3.9 MB), rather than being added as a large source file.
The concise final-pass log and unit output are copied into the worktree evidence
directory.

Final artifact check: `job-99bd98b07e4ceed4192ef0af` passed the full pre-commit suite
again. Logs and hardware details are preserved under `evidence/quality-final/`. The
measured driver is 570.148.08, CUDA is 12.9, cuDNN is 9.16.0.29, NCCL is 2.27.5, and
Transformer Engine is 2.14.1+27de67e1. This environment uses the existing container with
isolated MCore/Bridge/Emerging Optimizers packages; it is not a fresh full `uv sync`.

# Partial rollout validation

This evidence belongs only to run `research_eb6f8db22a9f4019`. Code branch:
`rsi/research_eb6f8db22a9f4019/research_ce92f6a2a491b52e7fd5b227`.

Code worktree (the original checkout is not updated):

```text
/storage/openpsi/users/fenghui/projects/AReaL-RSI/.rsi/research-lab/worktrees/research_eb6f8db22a9f4019/research_ce92f6a2a491b52e7fd5b227
```

## Verified training

Job `job-1cf3dfe8cddb19337aed582a` succeeded with exit code 0. Both the disabled control
and enabled trial completed eight real GSM8K training iterations from the same original
Qwen2.5-1.5B-Instruct model. The control was newly run here. This is a short functional
training validation, not a convergence or accuracy comparison. No formal
scoring/verification command is configured for this run.

| Enabled step | Generated tokens | Masked tokens | Partially masked trajectories | Optimizer skipped |
| ------------ | ---------------: | ------------: | ----------------------------: | ----------------- |
| 1            |             2528 |             0 |                             0 | No                |
| 2            |             3856 |             0 |                             0 | No                |
| 3            |             4036 |          4036 |                             0 | Yes               |
| 4            |             4183 |          4183 |                             0 | Yes               |
| 5            |             3865 |          3865 |                             0 | Yes               |
| 6            |             3691 |          3664 |                             1 | No                |
| 7            |             3121 |             0 |                             0 | No                |
| 8            |             4343 |          2336 |                            14 | No                |

Total masked: **18,084 / 29,623 = 61.0471593%**. Fifteen training trajectories retained
fresh actions while their stale actions were masked. The enabled trial ran five
optimizer calls; iterations 3–5 skipped globally empty minibatches. Iteration 6's valid
actions had zero gradient norm; iteration 8 had gradient norm 2.7023 while retaining 14
partially masked trajectories. Disabled training had nonzero gradient norms on all eight
iterations and emitted no new masking metrics. Do not interpret the eight iterations as
eight nonzero optimizer updates.

There are 63 mixed-version records among 188 enabled rollout dumps. These dumps include
unconsumed generation and are **not** the retained-training count above. For example,
`enabled/.../rollout/6/61.jsonl` records version runs `[[5, 145], [6, 41]]`. Explicit
half-open segment boundaries are also returned as `ModelResponse.generation_segments`;
transport behavior tests check them.

Evidence in this directory:

- `evidence/training_summary.json`: per-step metrics and dump examples.
- `evidence/enabled_metrics.txt`, `evidence/disabled_metrics.txt`: exact selected log
  lines, with terminal color escapes removed.
- `evidence/enabled.yaml`, `evidence/disabled.yaml`: actual saved configurations.
- `evidence/pytest_56.txt`: 56 tests passed in 41.49 seconds, including a real
  two-process Gloo test with one empty rank and one rank retaining valid actions.

Full logs, configs, command lines, rollout dumps and feedback:

```text
/storage/openpsi/users/fenghui/projects/AReaL-RSI/.rsi/research-lab/runs/research_eb6f8db22a9f4019/execution-outputs/job-1cf3dfe8cddb19337aed582a/
```

The runner's feedback deliberately leaves `acceptance_verified=false`: the job exit
alone is insufficient. Subsequent log analysis with `scripts/rsi_gsm8k_analyze.py`
verified all eight steps per trial, nonzero masking, retained mixed training rows and
absent masking statistics in disabled mode.

## Configuration and reproduction

Slurm allocation 972320, node slurmd-3, eight GPUs; original allocation environment:
`/storage/openpsi/images/areal-dev-20260508.sif`, `/opt/.venv/bin/python`, PyTorch
2.9.1+cu129 and Transformers 5.3.0. `environment.json` records the original local model
and GSM8K dataset locations. No past training checkpoint or output is used.

The experiment uses FSDP with four actor GPUs and SGLang with four rollout GPUs, seed 1,
prompt batch size 8, two samples per prompt, maximum 1024 generated tokens, 64
concurrent rollouts and maximum head offpolicyness 4. The enabled token-gap threshold is
the default 1. The large queue staleness deliberately exercises the masking boundary,
including entirely stale batches; it is not a recommended throughput/learning
configuration.

The authorized job command is `gsm8k_training`, executing:

```bash
scripts/rsi_gsm8k_validate.sh OUTPUT_DIRECTORY
```

The runner executes
`srun --mpi=none --jobid=972320 --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 singularity exec --nv ...`
with explicit `/storage`, workspace and output binds. Inside the configured container,
it runs:

```bash
python -m pytest -q tests/test_partial_rollout_versions.py tests/test_ppo_actor_truncation.py tests/infra/test_remote_inf_engine.py tests/v2/inference_service/test_inf_bridge.py
python examples/math/gsm8k_rl.py --config OUTPUT_DIRECTORY/disabled.yaml
python examples/math/gsm8k_rl.py --config OUTPUT_DIRECTORY/enabled.yaml
```

Use `RSI_VALIDATION_ENV` to supply a different environment JSON. For this research run,
experiments and checks were submitted via `rsi-tools call run_job` and observed with
`wait_for_job`. The runner removes inherited unreachable HTTP proxies only from
subprocess environments. It creates a random ephemeral admin key; saved YAML contains an
environment interpolation, never the key itself.

Analyze completed output with:

```bash
python scripts/rsi_gsm8k_analyze.py OUTPUT_DIRECTORY --output summary.json
```

## Debugging history and limitations

All failed attempts below are from this run, and are not successful validation:

1. `job-6935a6aa04ea99b37e976cb1`: remote tokenizer download failed before training.
   Fixed by using original shared local model/data resources.
1. `job-057c5f000111ca8aa57d4f6f`: data-service health checks timed out because of
   inherited unreachable proxies. Removed these for training child processes.
1. `job-b92bc53a06ab6dca79d83785`: proxy server refused the default admin key on a
   non-loopback address. Fixed with an ephemeral per-job key.
1. `job-1dd2bc9ce67c7f74532e7e59`: disabled eight steps passed; enabled training failed
   on a globally empty loss minibatch. Actor and critic now coordinate and skip only
   globally empty updates. Candidate `candidate_d2db2e5db980f630dd33b509` was withdrawn
   for this concrete defect.
1. `job-16d0aaec2d81126df2bdc7d6`: 55 tests and disabled eight steps passed; enabled
   training stalled at iteration 6 because the locally empty rank skipped
   advantage-normalization collectives. Captured stacks in that job's
   `stack-107372.txt`, `stack-107379.txt`, `stack-107386.txt`, and `stack-107393.txt`;
   terminated only that validation process tree. The fix enables empty-rank
   normalization participation only with stale masking. The added real Gloo regression
   and final training passed.

The final real GPU training covers v1 remote SGLang and FSDP on one node. V2 bridge
continuation has behavior-test coverage, not a full v2 GPU training run. Multi-node,
Megatron, Archon, real vLLM training, and a separate GPU critic were not exercised.
Critic empty-minibatch handling has unit coverage. No long-run GSM8K accuracy claim is
made.

## Final checks

`job-174bcf080d0f0bdde81a1939` succeeded with **270 tests passed in 37.51 seconds**,
full `pre-commit run --all-files` passed, and changed-file hooks passed. The full check
includes `uv-lock`, generated English/Chinese CLI docs, Ruff and mdformat. The C/C++
formatter had no applicable files. No hook was disabled. Both lockfiles and dependency
manifests remain unchanged.

See `evidence/regression_270_final.txt`, `evidence/pre_commit_all_passed.txt`, and
`evidence/pre_commit_changed_passed.txt`. Full output is in this run's
`execution-outputs/job-174bcf080d0f0bdde81a1939/` directory.

Earlier checks: `job-66ae674fdb51f997aac57525` passed 214 normalization tests, fixed
formatting and generated CLI docs, but hit a download timeout in the existing Megatron
Bridge dependency. `job-b35e14a5dc6985e7e4ac973d` passed 270 tests and changed-file
hooks, but full hooks detected container-default mirror rewrites of lockfiles. Those
unrelated changes were reverted. Raising the HTTP timeout and selecting the repository
lockfile's original default registry resolved the environment issues; the final full
check passed without lock changes.

General methodological reference: [AReaL paper](https://arxiv.org/abs/2505.24298). The
strict token version-gap rule is from this task; no external task solution was imported.

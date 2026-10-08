# τ³-Bench training example

This example runs text τ³-Bench tasks through the AReaL proxy. It uses the standard
simulated user (`solo_mode: false`) and returns τ³-Bench's task reward to GRPO. Install
τ³-Bench v1.0.1 separately so that `tau2` can be imported, and point `TAU2_DATA_DIR` to
that checkout's `data` directory.

The user simulator calls an OpenAI-compatible service. Put its API key in a readable
file with restricted permissions, then set these variables before launching:

```bash
export TAU3_USER_BASE_URL="https://YOUR-HOST/v1"
export TAU3_USER_MODEL="YOUR-MODEL"
export TAU3_USER_API_KEY_FILE="/path/to/user-api-key-file"
export QWEN3_5_35B_A3B_BASE_MODEL_PATH="/path/to/model"
export AREAL_PROXY_ADMIN_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

`dapo-math_grpo_qwen3_5_tau3.yaml` is a one-step smoke configuration. It uses four
airline tasks, two samples per task, and four concurrent rollouts. The parent
configuration enables chunked logits and a 1024-token LM-head loss chunk. The τ³
configuration disables CPU-staged AdamW because the current Megatron-Core installation
is incompatible with that optimizer wrapper. Run the baseline and HARTS with the same
configuration, changing only the tree training flags:

```bash
python examples/tau3/train.py \
  --config examples/cpu_staged_offload/dapo-math_grpo_qwen3_5_tau3.yaml \
  trial_name=tau3-baseline

python examples/tau3/train.py \
  --config examples/cpu_staged_offload/dapo-math_grpo_qwen3_5_tau3.yaml \
  trial_name=tau3-harts \
  actor.enable_tree_training=true actor.pad_to_maximum=true
```

The supplied model path, available GPU memory, and inference topology must match the
parent Qwen3.5 configuration. The training and simulated-user models are independent;
the user service is needed only when `solo_mode: false`.

For a two-step performance comparison, use `dapo-math_grpo_qwen3_5_tau3_perf.yaml`. It
retains the four-task, two-sample workload and raises the generation and request limits
to 20,480 and 32,767 tokens respectively, with a 2,048-token dataset input limit. Run
both modes with the same model, service, and hardware:

```bash
python examples/tau3/train.py \
  --config examples/cpu_staged_offload/dapo-math_grpo_qwen3_5_tau3_perf.yaml \
  trial_name=tau3-perf-baseline

python examples/tau3/train.py \
  --config examples/cpu_staged_offload/dapo-math_grpo_qwen3_5_tau3_perf.yaml \
  trial_name=tau3-perf-harts \
  actor.enable_tree_training=true actor.pad_to_maximum=true
```

### Eight-GPU airline run on 2026-10-02

Qwen3.5-35B-A3B-Base used the allocation in the parent configuration on eight L20X GPUs.
The external simulated user used `gpt-6-luna`; airline ran in standard mode
(`solo_mode: false`). Both runs requested three updates with seed 1. The baseline
completed two; its third update stopped before model execution because five real
microbatches on one DP rank required two transport dummies while other ranks had seven.
MoE rejects that padding to protect auxiliary gradients. HARTS completed all three
updates. The published performance config above now limits the run to the two-update
comparison window.

| Mode     | Step | Mean prompt / generated tokens | Original actor tokens | Tree ratio | Rollout | Actor update |    AWEX |
| -------- | ---: | -----------------------------: | --------------------: | ---------: | ------: | -----------: | ------: |
| Baseline |    1 |                  6,446 / 3,712 |               274,260 |          — |  80.4 s |       29.8 s | 242.8 s |
| Baseline |    2 |                  6,130 / 3,914 |               210,930 |          — | 116.1 s |       12.7 s | 237.4 s |
| HARTS    |    1 |                  7,494 / 2,127 |               394,450 |      0.469 | 140.9 s |       64.5 s | 263.7 s |
| HARTS    |    2 |                 11,665 / 3,252 |               686,160 |      0.546 | 125.1 s |       31.2 s | 249.5 s |
| HARTS    |    3 |                  7,934 / 3,264 |               369,540 |      0.536 |  87.6 s |       15.8 s | 257.0 s |

The actor token count is `ppo_actor/update/n_tokens` before tree compaction. Generated
length is mean sequence length minus mean prompt length. Every completed step had a zero
no-EOS ratio with the 20,480-token generation cap. Step 1 includes compilation and other
first-update costs. For the completed steady-state steps, baseline step 2 processed
16.6k original actor tokens/s; HARTS steps 2–3 processed 22.5k/s combined, about 1.35×
as many. AWEX took roughly four minutes per update in both modes, so actor gains
contribute less to total step time.

These are short online runs, not paired replay: the runs produced different tasks and
dialogue lengths despite the shared seed. For example, HARTS step 2 had 3.25× as many
actor tokens as baseline step 2. The times describe observed work and do not establish
an end-to-end speedup on identical trajectories.

### Ten-step pressure configuration

`dapo-math_grpo_qwen3_5_tau3_stress.yaml` runs 10 updates with 32 task prompts and 16
samples per prompt (512 trajectories per update), 32 concurrent rollouts, and a queue
capacity of 512. It keeps the performance configuration's 2,048-token input limit,
20,480-token generation cap, and 32,767-token request cap. The airline training split
has 30 task IDs; the dataset loader repeats the first two IDs to fill each 32-prompt
batch. To reproduce the reference run, set `actor.enable_tree_training=true`,
`actor.pad_to_maximum=true`, and `actor.mb_spec.max_tokens_per_mb=65536` for HARTS; the
baseline uses the default 32,768-token microbatch cap. The pressure configuration allows
up to 3,600 seconds for each multistep simulation; the smaller runs retain the
600-second default.

### Ten-step airline reference run

On one eight-L20X node, the baseline (`tau3-stress-baseline-20261003-r3`) and HARTS
(`tau3-stress-harts-20261006-r8`) each completed 10 updates with the pressure
configuration. Both kept 1,024-token chunked LM-head loss. Excluding step 1 compilation,
the native StatsLogger tables report:

| Steps 2–10                            |   Baseline |      HARTS |
| ------------------------------------- | ---------: | ---------: |
| Original actor tokens                 |   268.072M |   268.139M |
| Accumulated actor time                | 8,127.52 s | 6,869.15 s |
| Original actor tokens per second      |     32,983 |     39,035 |
| Weighted compact/original token ratio |          — |     0.4351 |

HARTS actor throughput was 1.1835× baseline throughput. The native logs recorded 14
baseline and 15 HARTS trajectory timeouts at the 3,600-second limit; these trajectories
were rejected and resampled. The two runs sampled separate live conversations and used
different actor microbatch token caps (32,768 baseline, 65,536 compact HARTS tokens).
This comparison measures actor throughput rather than paired-trajectory or end-to-end
speedup.

The W&B runs `tau3base20261003r3` and `tau3harts20261006r8` in project
`sct-test/wht_tree_training` replay numeric metrics from the historical native logs;
they were not logged live during training.

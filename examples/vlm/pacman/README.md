# Pacman RL: SGLang + Megatron

This example ports the released two-stage Pacman recipe to current AReaL. Game rules,
reward shaping, prompts, Edward options and trajectory auditing come from a pinned
external recipe checkout. The AReaL core only provides explicit plugin interfaces and
token metadata/reduction support. Importing the example does not patch an engine.

**Status: implementation and static checks only. No local unit tests or GPU training
have been run.** The original release itself describes its two-stage training as an
unverified recipe. This example does not claim a reproduced score or win rate.

## Sources and boundaries

| Source                                                                                                | Pinned revision                            | Use                                             |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------ | ----------------------------------------------- |
| [Original AReaL fork](https://github.com/luzai/AReaL/tree/ee872bae29152f4b553349385aced59abd1651ba)   | `ee872bae29152f4b553349385aced59abd1651ba` | Algorithm reference; do not install this fork   |
| [areal-pacman](https://github.com/luzai/areal-pacman/tree/219579e5323a0b24efb531b02f54ecc8848dd968)   | `219579e5323a0b24efb531b02f54ecc8848dd968` | Environment bridge, prompts, rewards and Edward |
| [pacman-python](https://github.com/luzai/pacman-python/tree/cbb97115e407abc86a44adc82a1b8f360b3e8da0) | `cbb97115e407abc86a44adc82a1b8f360b3e8da0` | Original Pygame game                            |

Initial upstream integration base: `b5f0820c307e9a02056131a54c6f7f92fa03ec55`. The
example requires the source checkouts, because the released environment records and
validates Git revisions and game source hashes. It reuses selected helpers from that
exact version; upgrading either external checkout requires a compatibility audit. Their
code and assets retain their original authorship and licensing.

## Recipe

| Setting        | C1                               | C2                                                       |
| -------------- | -------------------------------- | -------------------------------------------------------- |
| Initial model  | Qwen3.5-9B                       | Complete C1 model checkpoint; new optimizer              |
| Ghosts         | Disabled                         | Normal                                                   |
| Model decision | One currently open U/D/L/R token | One advertised Edward option token                       |
| Critic         | None                             | None                                                     |
| Task signal    | Raw reward for this decision     | Complete episode return                                  |
| Normalization  | None                             | Sample mean/std over 12 unique episodes, then clip at 20 |
| PPO reduction  | Equal decision/token weight      | Equal complete-episode weight                            |

Both stages use 12 sampled episodes per initial state, temperature 0.7, top-p 1, one
output token, no thinking, PPO clip 0.05, KL coefficient 0.01, learning rate 5e-7, Adam
with BF16 optimizer state, one PPO minibatch and recomputed proximal log-probabilities.
Rejection uses the original sequence-level ratio bounds \[0.8, 1.25\]; a sequence here
is one decision completion.

C1 is **critic-free PPO**. Its single-token advantage is the local shaped reward plus
proximal/reference KL; no value loss or cross-environment-step GAE is introduced. C2
normalizes unique episode returns before broadcasting them to decisions. Each accepted
token contributes with weight `1 / episode_decision_count`. Rejection changes the
numerator only; the denominator retains the original episode mass. Longer games
therefore do not acquire more weight merely by generating more decisions.

Both YAMLs preserve training seeds 28–107, validation seeds 108–111, training RNG seed
1, 512 environment steps, batch size 4 and 5 epochs (100 updates per stage). Reward
coefficients remain explicit in the YAMLs: pellets +1, vulnerable ghost +5, completion
+50, death -100, executed step -0.05, wall -0.5 and the original nearest-pellet shaping.

## Setup on the GPU host

Use the CUDA/SGLang and Megatron environment matching this AReaL checkout, including its
pinned Megatron Bridge and SGLang versions. No project dependency manifest is changed by
this example. The integration was inspected against SGLang 0.5.10.post1 and Megatron
Bridge 0.4.0; dependency upgrades need a new runtime check.

From the activated AReaL environment, place the two source dependencies outside the
AReaL worktree. Set paths for your machine:

```bash
export AREAL_ROOT=/absolute/path/to/current/AReaL
export PACMAN_SOURCE_ROOT=/absolute/path/to/pacman-sources
export MAAPACMAN_PACMAN_ROOT="$PACMAN_SOURCE_ROOT/pacman-python"
mkdir -p "$PACMAN_SOURCE_ROOT"
git clone --branch release/maapacman-v0.1.0 https://github.com/luzai/areal-pacman.git "$PACMAN_SOURCE_ROOT/areal-pacman"
git -C "$PACMAN_SOURCE_ROOT/areal-pacman" checkout --detach 219579e5323a0b24efb531b02f54ecc8848dd968
git clone --branch release/maapacman-v0.1.0 https://github.com/luzai/pacman-python.git "$MAAPACMAN_PACMAN_ROOT"
git -C "$MAAPACMAN_PACMAN_ROOT" checkout --detach cbb97115e407abc86a44adc82a1b8f360b3e8da0
uv pip install --no-deps -e "$PACMAN_SOURCE_ROOT/areal-pacman"
uv pip install 'pygame>=2.5'
export SGLANG_RETURN_ORIGINAL_LOGPROB=0
cd "$AREAL_ROOT"
```

Keep both dependency checkouts clean. Install the package using its editable package
mapping; do not prepend the external recipe root to `PYTHONPATH` or start Python from
that root. Its top-level `sitecustomize.py` belongs to the old deployment and must not
activate. Every rollout worker and SGLang server needs the current AReaL example module
and the same installed dependencies. The templates propagate the two required source
paths through worker environment settings.

The 8-GPU template allocates four one-GPU SGLang replicas and one four-GPU Megatron
actor (TP=4). Reference shares actor GPUs with phase offload. PP=CP=EP=1 is enforced; DP
and TP are configurable. To use DP=4, TP=1 on the same four actor GPUs, append
`actor.backend=megatron:d4p1t1`; reference inherits this topology. Full prompt groups
stay intact, and the group count in each batch must be divisible by DP. The current
batch size of 4 satisfies DP=4. The synchronized microbatch splitter can still fail when
a short group has too few decision rows to match the other ranks' microbatch count. This
change only permits DP in the recipe configuration; it does not implement dummy
microbatches or establish DP1/DP4 numerical equivalence. TP=1 also requires each GPU to
hold the full model parameters; recheck peak memory for actor and reference.

The XCCL template is a placement template, not a hardware qualification. Verify
model/vision initialization, TMS, available GPU memory, host RAM for retained episode
images, shared filesystem paths and checkpoint disk space on the target machine. Actor
and reference have an initialization overlap before phase offload. Do not copy
`TMS_INIT_ENABLE=0` from an AWEX example into this XCCL recipe.

## AWEX colocation

`curriculum1_awex_colocate.yaml` and `curriculum2_awex_colocate.yaml` compose the
original curricula with `awex_colocate.yaml`. They retain the rewards, KL=0.01,
sampled12 decoding, batch size 4, 512-step horizon and optimizer settings. They use
single-controller v1 with a local scheduler, an eight-GPU Megatron actor (DP=4, TP=2), a
reference with the same topology, and eight colocated SGLang TP=1 instances. The
original XCCL recipes remain available.

AWEX 0.8.1 with its Qwen3.5 converter is required (already in AReaL's CUDA dependency
set). Training workers disable automatic TMS regions; actor and reference use explicit
Megatron residency, while SGLang enables its memory saver. Keep the overlay's worker
environment, `actor.offload=false`, and `ref.offload=true`. AWEX manages the actor's
residency transitions explicitly. The generic trainer releases the reference before
starting SGLang and orders each update as:

```text
rollout -> release SGLang -> reference onload/score/offload
        -> actor onload/recompute/advantages/PPO/save
        -> AWEX weight publication -> restore SGLang -> next rollout
```

On the same eight GPUs this doubles training DP relative to a separated DP=2/TP=2 actor.
It does **not** double rollout concurrency: local colocation requires eight rollout
workers to match the eight actor workers, but the current batch of four initial-state
groups and zero staleness allow only four groups at once. Each group runs its twelve
episodes on one SGLang instance. The remaining instances are idle in that rollout batch;
increasing global batch size or staleness would change the recipe. Variable-length
episodes can still expose the DP microbatch splitting limitation.

Launch C1 with a fresh artifact root and the setup variables above:

```bash
bash examples/vlm/pacman/scripts/train_c1_awex.sh \
  actor.path=/absolute/path/to/Qwen3.5-9B \
  environment.max_steps=2 planner_audit.max_steps=2 total_train_steps=2
```

The short horizon above is a startup check. Omit those overrides for the release
horizon. To validate on the target GPU environment:

```bash
# Small-model check: two GPUs for existing numerical tests, one for colocated RL.
bash examples/vlm/pacman/scripts/smoke_awex.sh
# Full eight-GPU topology: two updates per curriculum and C1 -> C2 checkpoints.
bash examples/vlm/pacman/scripts/e2e_awex.sh
```

Use different fresh artifact roots for those commands. Expected results are successful
reference residency transitions, two finite PPO updates, AWEX publication followed by
rollout with the new weight version, and complete checkpoints. These scripts and runtime
behavior have **not been GPU-validated**; static checks alone do not establish
weight-transfer correctness or throughput.

## Run

Run the training module directly in **single-controller mode**. Both recipes use
`scheduler.type: local`; the trainer starts the required workers and inference servers.
The scripts set `AREAL_SPMD_MODE=0` explicitly; use the same setting for manual runs. Do
not wrap this entry point with the legacy SPMD launcher or add `allocation_mode`.

Use a fresh absolute artifact root for each invocation to keep manifests, checkpoints
and stage transitions unambiguous.

Standalone C1 (eight GPUs, Qwen3.5-9B, 100 updates; no C2 transition):

```bash
export PACMAN_ARTIFACT_ROOT=/absolute/path/to/new/pacman-c1
bash examples/vlm/pacman/scripts/train_c1.sh
# Or limit the same recipe to two updates:
# bash examples/vlm/pacman/scripts/train_c1.sh total_train_steps=2
# To use a local model, append actor.path=/absolute/path/to/Qwen3.5-9B.
```

This startup script requires the setup above and has **not been run on GPUs**.

Minimal GPU check (two GPUs, Qwen3.5-0.8B, one update, one initial state, 32 steps):

```bash
export PACMAN_ARTIFACT_ROOT=/absolute/path/to/new/pacman-smoke
bash examples/vlm/pacman/scripts/smoke.sh
```

The smoke script first runs the prepared tests, including a two-GPU numerical TP check,
then trains/evaluates C1 and validates the resulting model/processor checkpoint. Its
reduced seeds, horizon and model are explicitly smoke settings, not reproduction.

Both real curricula, two updates each, using the eight-GPU template:

```bash
export PACMAN_ARTIFACT_ROOT=/absolute/path/to/new/pacman-e2e
bash examples/vlm/pacman/scripts/e2e.sh
```

The script validates C1's latest HF checkpoint and initializes C2 from those weights
with a new optimizer. For the full 100 + 100 updates, use a fresh artifact root and set
`PACMAN_TRAIN_STEPS=null`. Both scripts are **unrun locally**.

To run a stage separately:

```bash
# For C2, also export CURRICULUM1_CHECKPOINT to a complete C1 HF checkpoint.
AREAL_SPMD_MODE=0 python -m examples.vlm.pacman.train \
  --config examples/vlm/pacman/curriculum1.yaml total_train_steps=2
```

Expected validation: all prepared numerical checks pass; rollout tokens belong to the
recorded support; actor/reference forward passes and PPO backward complete with finite
metrics; XCCL updates permit the next rollout; per-episode trajectory artifacts,
validation outputs and complete HF checkpoints appear under the selected root. Inspect
behavior/proximal log-probability differences and rejection rates before a full run;
passing static checks does not establish SGLang/Megatron numerical agreement.

## Extension points and known differences

- The workflow automatically uses AReaL's group-scoped processor cache for both
  curricula. Identical full messages (including inline PNG contents) and the same
  processor share one native processing call across the 12 candidate episodes. Each
  decision keeps private containers and immutable shared processor tensors, so the
  existing rollout RTensor transport can export repeated image tensors as one shard.
  Image decoding and re-encoding are also reused on cache hits. Changed prompts or
  images are processed separately; different rollout groups never share cache entries.
  Group finalization closes the cache, including on cancellation.

- `rollout/pacman_processor_cache_hit` (and its `eval-rollout` counterpart) records
  whether each decision reused processing; its mean is the processor cache hit rate. It
  is not a measurement of bytes saved or end-to-end speedup. JSON/Base64 transport and
  full-batch reference input loading remain unchanged. Sampling, rewards, action
  constraints and PPO settings are unchanged.

- The existing `scripts/smoke.sh` includes the cache regression checks: concurrent
  reuse, exact native tensor equivalence, input/group isolation, and alias preservation
  through trajectory construction and RTensor export. To run just these checks in the
  configured environment, use
  `python -m pytest examples/vlm/pacman/tests/test_workflow.py`. Use `scripts/e2e.sh`
  above for the real training path and inspect the cache metric alongside reference
  forward timing. These tests and GPU scripts have **not been run locally**.

- `GenerationHyperparameters.request_plugin` constructs a worker-local request plugin.
  This example uses SGLang's public
  [CustomLogitProcessor API](https://github.com/sgl-project/sglang/blob/v0.5.10.post1/python/sglang/srt/sampling/custom_logit_processor.py)
  to mask tokens before temperature/softmax. It is an exact allowed-token constraint;
  JSON-schema validation alone would not align the training distribution.

- `MegatronEngineConfig.policy_distribution` replaces default logprob/entropy gathering
  in training and proximal/reference forward passes. This example gathers only allowed
  logits across TP and uses identity backward for replicated loss gradients. It requires
  CP=1, ordinary full-logit output and no tree training or fused chunked LM-head loss.

- `PPOActorConfig.objective_plugin` prepares advantages and optionally wraps the
  existing PPO loss and its microbatch normalization mass. Implementations must preserve
  row order and count. `loss_reduction_weights` is a generic per-token reduction
  primitive.

- `[B,S,K]` token metadata is split/padded/packed with the token axis preserved. The
  prediction-aligned support ledger uses token ID + 1, with zero reserved for padding.
  Infinite configuration bounds round-trip through strict JSON RPC; actual nonfinite
  task rewards are still rejected.

These hooks default to disabled. All game-specific checks and formulas are in this
example. The async workflow moves the release's blocking environment/file I/O to
per-episode worker threads and submits generation back to the inference event loop;
errors remain errors, and cancellation cancels pending generation.

The entry point regenerates the release's initial-state seeds with **current AReaL
provenance**, and records the resolved config and rows in a content-addressed input
manifest. It does not pretend these are the original release's immutable audited data
bundle. Original episode reward/prompt/Edward audit logic remains active during rollout.
Incomplete episode groups are discarded as a whole, so C2 never normalizes a partial
12-episode group.

The fork's `keep_last`/`keep_best` saver policy is not ported into core. Current AReaL
saves each update and retains all checkpoints; budget disk space accordingly. The
training logger's task-reward mean remains decision-weighted, so use complete episode
returns and outcomes in trajectory artifacts to compare stage performance. Changing
backends and TP placement preserves the intended objective, not bitwise trajectories.

Prepared tests can be run on CI/the target host independently:

```bash
python -m pytest tests/test_rl_plugins.py examples/vlm/pacman/tests -m 'not slow'
python -m pytest examples/vlm/pacman/tests/test_policy_tp.py -m slow
```

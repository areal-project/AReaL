# Multi-Teacher On-Policy Distillation (MOPD)

MOPD combines on-policy reinforcement learning with token-level targets from one or
more teacher checkpoints. Each dataset source selects one required teacher group,
which assigns non-negative weights to any number of teachers. The weights are applied
directly and are not normalized.

MOPD currently runs in single-controller mode with Megatron actor and teacher engines,
an SGLang rollout engine, and AWEX colocated weight transfer. Actor, rollout, and a
persistent teacher companion share the same GPUs but never own the large model weights
at the same time.

## Runtime lifecycle

Each training step follows three exclusive phases:

1. **Rollout:** SGLang generates trajectories while AReaL propagates the source's
   teacher group as internal task metadata. Dataset samples and workflow inputs need
   no routing field.
2. **Teacher:** rollout weights and KV cache are offloaded. A forked Megatron teacher
   process onloads, loads each required checkpoint, and scores its routed samples. The
   actor materializes and clears all teacher RTensors before the teacher weights are
   offloaded again. The companion process stays alive for reuse by the next step.
3. **Train:** the actor computes the configured RL and distillation loss, updates its
   weights, and publishes the next version to SGLang through AWEX. SGLang drops stale
   KV cache entries and allocates a new empty cache before generation continues.

The actor and `mopd.teacher_engine` must use identical parallel strategies and world
sizes, including the pipeline parallel size. The current implementation requires
teacher and actor controllers to use v1. PP, TP, CP, DP, and EP values are validated by
the selected Megatron model and allocation.

## Configuration

Add `mopd` to a PPO configuration:

```yaml
actor:
  backend: "megatron:(attn:d1p1t4c2|ffn:d1p1e8)"
  weight_update_mode: awex

rollout:
  backend: sglang:d8t1
  scheduling_strategy: {type: colocation, target: actor, fork: true}

train_dataset:
  mixture_sampling_policy: proportional
  sources:
    - {path: /data/code, type: rl, teacher_group: coding}
    - {path: /data/mixed, type: rl, teacher_group: mixed}

mopd:
  teachers:
    coder: {path: /models/teacher-coder}
    reasoning: {path: /models/teacher-reasoning}
  teacher_groups:
    coding: {coder: 1.0}
    mixed: {coder: 0.3, reasoning: 0.7}
  teacher_engine:
    backend: ${actor.backend}
    optimizer: null
    disable_dropout: true
    scheduling_strategy: {type: colocation, target: actor, fork: true}
    scheduling_spec: ${actor.scheduling_spec}
  manager:
    type: disk
    staging_root: /dev/shm/areal-mopd
  loss:
    rl_coefficient: 0.0
    distillation_coefficient: 1.0
```

When MOPD is enabled, every entry in `train_dataset.sources` must declare a
`teacher_group` that matches a key in `mopd.teacher_groups`; validation datasets follow
the same rule when configured. Outside MOPD, `teacher_group` defaults to `null`.
Samples cannot override their source's teacher group and need no `task_type` field. A
teacher group can reference any number of known teacher IDs and must contain at least
one positive weight.
`mixture_sampling_policy: proportional` preserves source-size proportions.
`uniform` gives every source the same number of samples per epoch by cycling shorter
sources deterministically before the distributed sampler shuffles global indices.

`manager.type: disk` loads checkpoints from shared storage and supports multi-node
runs. `local_memory` asynchronously stages one upcoming checkpoint below
`staging_root`, atomically publishes it to the persistent teacher, and removes it
after loading. Because this path is visible only on the controller host,
`local_memory` requires `scheduler.type: local` and a single-node actor/teacher
topology. `min_free_bytes` can reserve free space below the staging root.

For teacher weights $w_j$, define $S_T(a)=\sum_j w_j\log\pi_{T_j}(a)$ and
$W=\sum_j w_j$. MOPD minimizes the raw weighted reverse KL
$\sum_j w_j D_{KL}(\pi_\theta \parallel \pi_{T_j})$ with the on-policy
score-function surrogate:

```text
rho(a) = min(exp(log pi_theta(a) - log pi_old(a)), importance_ratio_cap)
reward(a) = S_T(a) - W * stop_gradient(log pi_theta(a))
mopd_loss = -mean(rho(a) * reward(a))
loss = rl_coefficient * rl_loss + distillation_coefficient * mopd_loss
```

`importance_ratio_cap` defaults to `5.0` and bounds the importance-sampling
multiplier to prevent exponential overflow.

This is a weighted sum of reverse-KL objectives, equivalently a geometric teacher
ensemble up to an additive constant. It is not teacher cross-entropy or an arithmetic
mixture of teacher probabilities. Route weights are applied directly and are not
normalized.

Set `rl_coefficient: 0.0` for pure distillation. Set both coefficients to positive
values for joint RL and distillation.

## Examples

- `examples/mopd/gsm8k_qwen3_14b_to_0_6b.py` provides the local GSM8K entry point and
  dry-run validator.
- `examples/mopd/gsm8k_qwen3_14b_to_0_6b_local.yaml` configures a Qwen3-14B teacher
  and Qwen3-0.6B actor on one eight-GPU node.

Validate a local configuration without starting workers:

```bash
MOPD_STUDENT_MODEL_PATH=/models/Qwen3-0.6B \
MOPD_TEACHER_MODEL_PATH=/models/Qwen3-14B \
MOPD_GSM8K_PATH=/data/gsm8k \
AREAL_ADMIN_API_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')" \
python -m examples.mopd.gsm8k_qwen3_14b_to_0_6b \
  --config examples/mopd/gsm8k_qwen3_14b_to_0_6b_local.yaml \
  --dry-run
```

## Operational notes

- Actor and teacher checkpoints must share the same token-ID mapping, and every model
  architecture must be supported by its selected Megatron adapter.
- The teacher companion process is persistent, but its model weights must be offloaded
  before actor ownership resumes. `DrainReceipt` is the phase reclamation boundary for
  teacher RTensors.
- Keep W&B credentials and service endpoints in environment variables; do not add
  them to YAML or shell files.
- A cancelled actor or rollout Slurm child job can be normal when the driver has
  already completed and is cleaning up its persistent workers. Use the driver exit
  code and final training-step log to determine success.

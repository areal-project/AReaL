# DTE separation example

This directory contains an opt-in example for transferring AdamW weight updates from a
Megatron actor to a separately scheduled SGLang rollout worker with
[AReaL-DTE](https://github.com/areal-project/AReaL-DTE).

Install AReaL-DTE in the controller and worker environments before enabling the example.
For local development, use an editable checkout:

```bash
pip install -e /path/to/AReaL-DTE
```

The Qwen3-30B-A3B GSM8K example uses two 8-GPU allocations: one for the actor and one
for rollout. Launch it with the scheduler used by your cluster, for example:

```bash
python3 examples/math/gsm8k_rl.py \
  --config examples/dte/gsm8k_grpo_qwen3_30b_a3b.yaml \
  scheduler.type=slurm \
  actor.scheduling_spec.0.image=/path/to/areal.sif
```

The first weight update is a full synchronization. Later contiguous versions use sparse
AdamW deltas, with a periodic full-weight anchor after every 20 successfully committed
deltas. Set `actor.enable_delta_weight_update=false` to return to the existing AWEX
full-weight behavior.

## Optimizer-step boundary

The current separation AdamW detector reconstructs exactly one optimizer step between
weight updates. Therefore, enabling DTE requires:

```yaml
actor:
  ppo_n_minibatches: 1
```

Each PPO minibatch performs an optimizer step, while weight synchronization happens only
after all PPO minibatches finish. Values greater than one would make the optimizer step
advance by more than the detector can invert and would force a full-weight fallback.
AReaL rejects that configuration at startup instead of silently running full
synchronization while DTE appears enabled.

## Effective performance defaults

The example spells out the portable performance settings in the worker environment; none
of them depends on a site-specific path or cluster:

| Optimization                    | Effective setting                       | Source                                    |
| ------------------------------- | --------------------------------------- | ----------------------------------------- |
| Streaming AdamW reconstruction  | `DTE_STREAMING_RECONSTRUCT=1`           | Explicit; also enforced by delta transfer |
| Coalesced two-round sparse P2P  | `DTE_DELTA_P2P_COALESCE=1`              | Explicit; also the AReaL-DTE default      |
| Pipelined inversion collectives | 512 MiB in-flight window                | Explicit AReaL setting                    |
| Inversion compute device        | Payload device (GPU for this example)   | Explicit AReaL setting                    |
| Compact change indices          | `int32` when the parameter size permits | Standard AReaL detector path              |
| Batched operation remapping     | Once per parameter                      | Standard AReaL-DTE payload path           |

Snapshot verification, weight digests, phase timing, recovery, and deterministic rollout
diagnostics are intentionally not enabled here because they are validation or experiment
controls rather than requirements of the separation delta path.

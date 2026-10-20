# V2 AWEX colocation

V2 colocation lets Megatron training workers and SGLang inference workers share GPUs. It
uses the shared AWEX adapters and converters; placement selects the colocated memory and
transfer lifecycle.

## Supported configurations

| Component           | Supported configuration                                                                                  |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| Training            | Megatron, full weights                                                                                   |
| Inference           | SGLang with `enable_memory_saver: true`                                                                  |
| Placement           | Local scheduler, or full-node Slurm allocations                                                          |
| Slurm GPU namespace | Contiguous numeric device IDs matching the guard placement                                               |
| Checkpoints         | HF export and synchronous DCP recovery                                                                   |
| Unsupported         | LoRA colocation, FSDP colocation, vLLM colocation, Ray v2 guard colocation, asynchronous v2 DCP recovery |

UUID, reordered, or partial-node Slurm GPU allocations are rejected before training
guards are forked. Existing v1 colocation and v2 separated weight updates keep their own
placement and transfer paths.

Use `actor._version: v2` and `rollout._version: v2`, select Megatron and SGLang
backends, and configure actor colocation with the rollout guards. The actor and rollout
must cover the same physical GPUs. Their parallel layouts may differ, but every SGLang
TP/PP group must fit the supported node-local placement. Set
`actor.megatron.async_save: false` when enabling recovery.

## Memory ownership

Initialization creates rollout guards, initializes training workers, and releases
training memory before loading SGLang. Auxiliary models initialize after that release.
Offloaded v2 auxiliary models require Megatron and use its explicit weight residency
manager rather than the inherited inference TMS hook.

Each training phase pauses rollout submission, drains generation, and releases rollout
weights, KV cache, and CUDA graphs. Reference, critic, and teacher scoring run before
actor memory is restored. The actor then trains, saves checkpoints, publishes weights,
and commits the policy version. Training memory is released before inference memory is
restored.

An incomplete weight transfer or version commit leaves generation paused. Exported CUDA
IPC storage stays alive until the reader acknowledges consumption; an unacknowledged
export also blocks another publication or adapter teardown. After the final training
step, inference may serve evaluation, but the rollout dispatcher does not start another
training batch.

## Recovery

Synchronous DCP recovery uses the same immutable checkpoint generations and `LATEST`
publication protocol as other supported training paths. Restore loads the selected
generation inside the colocated training phase, reuses the existing gateway, publishes
the recovered weights, and commits the recovered version before resuming generation.
Checkpoint corruption or a failed restore does not silently resume training.

Asynchronous v2 DCP recovery is rejected because worker shutdown does not yet provide a
reliable drain for arbitrarily long background saves. Supporting it requires validating
both asynchronous publication and final worker teardown.

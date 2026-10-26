# V2 AWEX colocation

V2 colocation lets Megatron training workers and SGLang inference workers share GPUs. It
uses the shared AWEX adapters and converters; placement selects the colocated memory and
transfer lifecycle.

## Supported configurations

| Component           | Supported configuration                                                    |
| ------------------- | -------------------------------------------------------------------------- |
| Training            | Megatron, full weights                                                     |
| Inference           | SGLang with `enable_memory_saver: true`                                    |
| Placement           | Local scheduler, or full-node Slurm allocations                            |
| Slurm GPU namespace | Contiguous numeric device IDs matching the guard placement                 |
| Checkpoints         | HF export and synchronous/asynchronous DCP recovery                        |
| Unsupported         | LoRA colocation, FSDP colocation, vLLM colocation, Ray v2 guard colocation |

UUID, reordered, or partial-node Slurm GPU allocations are rejected before training
guards are forked. Existing v1 colocation and v2 separated weight updates keep their own
placement and transfer paths.

Use `actor._version: v2` and `rollout._version: v2`, select Megatron and SGLang
backends, and configure actor colocation with the rollout guards. The actor and rollout
must cover the same physical GPUs. Their parallel layouts may differ, but every SGLang
TP/PP group must fit the supported node-local placement. Set
`actor.megatron.async_save: true` to overlap DCP disk writes with training.

## Allocator configuration

TMS 0.0.9 and 0.0.10 reject `expandable_segments:True`. Both v1 and v2 AWEX rollout
startup validate allocator settings without rewriting the environment. Configure the
rollout role independently from the actor, for example in each `rollout.scheduling_spec`
entry's `env_vars`:

```yaml
PYTORCH_ALLOC_CONF: expandable_segments:False
PYTORCH_CUDA_ALLOC_CONF: expandable_segments:False
```

PyTorch 2.9 prefers `PYTORCH_ALLOC_CONF`, but TMS also checks the legacy
`PYTORCH_CUDA_ALLOC_CONF` directly. Neither variable may enable expandable segments in
the rollout process. Omitting that option is also valid. Actor allocator settings can
remain independent. GPU masks must retain ascending, contiguous device order; reordered
masks are rejected rather than sorted.

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

DCP recovery uses the same immutable checkpoint generations and `LATEST` publication
protocol as other supported training paths. Restore loads the selected generation inside
the colocated training phase, reuses the existing gateway, publishes the recovered
weights, and commits the recovered version before resuming generation. Checkpoint
corruption or a failed restore does not silently resume training.

Asynchronous saves stage their payload before returning, so subsequent weight updates
and offload do not modify the saved snapshot. `LATEST` is published only after the
background save finalizes. Normal shutdown drains every training rank before destroying
workers or their parent rollout guards. The drain uses `actor.request_timeout` (or the
corresponding critic setting), which must cover the longest expected checkpoint write. A
failed or timed-out drain raises and preserves worker resources for diagnosis; it does
not report a successful shutdown. Abrupt process termination may leave an unpublished
generation; recovery continues to select the last successfully published checkpoint.

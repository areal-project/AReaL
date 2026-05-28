---
name: debug-distributed
description: Guide for debugging distributed training issues in AReaL. Use when user encounters hangs, wrong results, OOM, or communication errors.
---

# Debug Distributed Training

Debugging guide for distributed training issues in AReaL (FSDP2, TP, CP, EP).

## When to Use

This skill is triggered when:

- Training hangs or deadlocks
- Results differ across ranks or are numerically wrong
- OOM errors in distributed settings
- NCCL/communication errors or device mesh issues

## Debugging Principles

### Minimal Reproduction

**Always follow the minimal demo principle**: Reproduce with the least amount of code to
narrow down the issue faster.

```python
# Bad: Debug in full training loop
# Good: Create minimal script
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()

# Reproduce the exact operation that fails
tensor = torch.ones(10).cuda()
dist.all_reduce(tensor)  # <-- Isolate the failing op
print(f"Rank {rank}: {tensor}")
```

**Reduction strategy:**

1. Remove unrelated model components
1. Use small tensor sizes
1. Reduce world_size to minimum (e.g., 2 GPUs)
1. Remove torch.compile if possible
1. Disable activation checkpointing

## Step-by-Step Debugging Guide

### 1. Hang Debugging (Deadlocks, Synchronization)

**Environment Variables for Debugging**:

```bash
# Full debug logging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# torch.compile debugging
export TORCH_LOGS="+dynamo,recompiles"
export TORCHDYNAMO_VERBOSE=1
```

**Dump Call Stack with py-spy** (for hung processes):

```bash
# Find process IDs
ps aux | grep python

# Dump call stack of specific rank
py-spy dump --pid <PID>

# Record flame graph for performance analysis
py-spy record -o profile.svg --pid <PID> --duration 30
```

**Common Causes**:

1. **Mismatched Collectives**: One rank calls `all_reduce`, another doesn't.
1. **Wrong Process Group**: Using wrong group for collective.
1. **Tensor Shape Mismatch**: Different shapes across ranks.

**Debug Steps**:

```python
# Verify group membership
mesh = parallel_dims.get_mesh("dp_shard_cp")
group = mesh.get_group()
print(f"Rank {dist.get_rank()}: group size = {dist.get_world_size(group)}")

# Print shapes on all ranks
print(f"Rank {dist.get_rank()}: tensor.shape = {tensor.shape}")
dist.barrier()
```

**Timeout Adjustment** (for debugging only):

```python
from areal.engine.core.distributed import patch_dist_group_timeout
from datetime import timedelta
patch_dist_group_timeout(timedelta(minutes=30))
```

### 2. Wrong Results (Gradient, Reduction Issues)

**Check DTensor Placements**:

```python
from torch.distributed.tensor import DTensor
if isinstance(param, DTensor):
    print(f"Param {name}: placements={param.placements}, mesh={param.device_mesh}")
```

**Verify Gradient Reduction**:

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"Rank {dist.get_rank()}: {name} grad_sum = {param.grad.sum().item()}")
```

### 3. OOM Issues (Memory, Sharding)

**Check Memory Usage**:

```python
print(f"Rank {dist.get_rank()}: "
      f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB, "
      f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")
```

**Check FSDP Coverage**:

```python
for name, param in model.named_parameters():
    is_dtensor = isinstance(param, DTensor)
    print(f"{name}: is_dtensor={is_dtensor}, shape={param.shape}")
```

### 4. Communication Errors

| Error                     | Cause                | Solution                           |
| ------------------------- | -------------------- | ---------------------------------- |
| `NCCL WARN Cuda failure`  | GPU communication    | Check NCCL version, GPU topology   |
| `RuntimeError: Timed out` | Rank synchronization | Increase timeout, check code paths |
| `Invalid device mesh`     | Mesh configuration   | Verify world_size = dp * tp * cp   |

## Debugging Tools

### Environment Variables Reference

| Variable                          | Purpose                                |
| --------------------------------- | -------------------------------------- |
| `TORCH_DISTRIBUTED_DEBUG=DETAIL`  | Detailed distributed logging           |
| `NCCL_DEBUG=INFO`                 | NCCL communication logging             |
| `NCCL_DEBUG_SUBSYS=ALL`           | All NCCL subsystems                    |
| `TORCH_LOGS="+dynamo,recompiles"` | torch.compile logging                  |
| `TORCHDYNAMO_VERBOSE=1`           | Dynamo verbose output                  |
| `CUDA_LAUNCH_BLOCKING=1`          | Synchronous CUDA (slow, for debugging) |

### py-spy for Call Stack Analysis

```bash
# Install
pip install py-spy

# Dump call stack of hung process
py-spy dump --pid <PID>

# Dump all Python processes
pgrep -f python | xargs -I {} py-spy dump --pid {}

# Record flame graph
py-spy record -o profile.svg --pid <PID> --duration 30
```

### Rank-Conditional Printing

```python
def print_all_ranks(msg):
    for r in range(dist.get_world_size()):
        if dist.get_rank() == r:
            print(f"[Rank {r}] {msg}")
        dist.barrier()
```

### Check Device Mesh

```python
def debug_mesh(parallel_dims):
    mesh = parallel_dims.world_mesh
    for dim_name in mesh.mesh_dim_names:
        submesh = parallel_dims.get_mesh(dim_name)
        if submesh:
            print(f"Rank {dist.get_rank()}: {dim_name} size={submesh.size()}")
```

### Validate Tensor Consistency

```python
def check_tensor_consistency(tensor, name, group=None):
    local_sum = tensor.sum().item()
    tensor_sums = [None] * dist.get_world_size(group)
    dist.all_gather_object(tensor_sums, local_sum, group=group)
    if dist.get_rank() == 0 and len(set(tensor_sums)) > 1:
        print(f"WARNING: {name} inconsistent: {tensor_sums}")
```

## Key Files Reference

| Component       | File                                                          |
| --------------- | ------------------------------------------------------------- |
| Parallel Dims   | `areal/experimental/models/archon/parallel_dims.py`           |
| Expert Parallel | `areal/experimental/models/archon/expert_parallel.py`         |
| Ulysses (CP)    | `areal/experimental/models/archon/ulysses.py`                 |
| FSDP/TP Apply   | `areal/experimental/models/archon/qwen2/infra/parallelize.py` |

______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .claude/skills/debug-distributed/SKILL.md
Invocation: /debug-distributed

## Purpose

Debugging guide for distributed training issues.
Covers FSDP2, Tensor Parallelism, Context Parallelism, and Expert Parallelism.

## How to Update

### When Adding New Parallelism Features
1. Add section for the parallelism type
2. Document common error patterns and debugging snippets

### When PyTorch Distributed APIs Change
1. Update DTensor/DeviceMesh examples
2. Update environment variable references

### When New Error Patterns Emerge
1. Add to "Common Errors and Solutions" table
2. Reference relevant source files

================================================================================
-->

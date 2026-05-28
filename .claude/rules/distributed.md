---
paths:
  - areal/engine/**
  - areal/experimental/**
  - areal/engine/fsdp_utils/**
---

# Distributed Code Rules

## Process Group Management

- **Never create global process group** in module-level code
- Always pass `process_group` explicitly, don't rely on default
- Use `dist.get_rank(group)` not `dist.get_rank()` when group matters
- Clean up process groups in `__del__` or context manager

## DeviceMesh & DTensor

- Mesh dimension names must match `ArchonParallelDims`: `dp_shard`, `tp`, `cp`, `ep`,
  `etp`
- DTensor requires consistent mesh across all ranks
- Use `DTensor.from_local()` with correct placements

## Communication Patterns

- **All-reduce**: Must be called by all ranks in the group
- **Broadcast**: Specify `src` rank explicitly
- **Barrier**: Avoid unless necessary (debugging only)
- Check `NCCL_ASYNC_ERROR_HANDLING` for deadlock debugging

## Common Pitfalls

| Issue         | Cause                            | Fix                            |
| ------------- | -------------------------------- | ------------------------------ |
| Hang          | Mismatched collective calls      | Ensure all ranks call same op  |
| Wrong results | Incorrect reduction op           | Check `ReduceOp` (SUM vs MEAN) |
| OOM           | Unsharded tensor on wrong device | Verify DTensor placements      |

## Debugging

- Set `TORCH_DISTRIBUTED_DEBUG=DETAIL` for verbose logging
- Use `NCCL_DEBUG=INFO` for NCCL-level issues
- See `/debug-distributed` skill for comprehensive guide

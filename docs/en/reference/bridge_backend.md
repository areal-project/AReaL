# Megatron-HF Bridge Backend

AReaL currently supports two bridge backends for `MegatronEngine`:

- `mbridge` (default)
- `megatron-bridge`

Set the backend with:

```yaml
actor:
  megatron:
    bridge_type: mbridge
```

- Use `bridge_type=megatron-bridge` to enable the new path.
- `mbridge` is the default choice if this argument is not present

## Why this feature exists

- `mbridge` is being deprecated and does not provide PEFT/LoRA support.
- `megatron-bridge` supports more/ newer model architectures.
- `megatron-bridge` provides built-in PEFT/LoRA implementations.

## Recommendation

- For new GPU training workflows, prefer `megatron-bridge`.
- Keep `mbridge` for backward compatibility and environments that still depend on it.
- Prefer `mbridge` when using disk-based weight broadcast as it has optimized HF
  load/save path.
- If you use XCCL for weight broadcast, load/save time is less important.

## Current limitation

- Tree-attention training in `MegatronEngine` currently supports only `mbridge`.
- The `megatron-bridge` backend is not supported in the tree-attention path yet.
- `megatron-bridge` does support faster/optimized HF model load/save implementations.

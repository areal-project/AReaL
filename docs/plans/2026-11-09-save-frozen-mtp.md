# Preserve frozen Qwen MTP in HF checkpoints

## Behavior

For Qwen3.5, Qwen3.5-MoE and Qwen3-Next exports using `megatron-bridge`,
`megatron.enable_mtp=false` leaves the training-side MTP head unbuilt. After exporting
trained backbone weights, rank zero copies the original `mtp.*` tensors into a separate
HF safetensors shard using bounded CPU buffers. It restores the source MTP configuration
and rebuilds the index from actual tensors, including the correct byte total.

The source is the bridge's resolved local initialization checkpoint. An export-time
`base_model_path` override only selects assets and must not change MTP weight
provenance. The original source checkpoint must remain accessible until export
completes.

Only MTP-exclusive tensors are copied. Shared embeddings and `lm_head` retain their
trained values, even when they originally shared a source shard with MTP. Existing
complete MTP tensors are retained after shape/dtype validation; their values are not
compared with the source. Partial MTP exports fail explicitly. Single-file exports
become indexed exports so HF loaders can discover the additional shard. Repeated exports
replace the backbone with the latest trained values. When resuming from a completed HF
export, Bridge may emit an empty MTP-only shard; the exporter replaces that empty shell
with the frozen source tensors while rejecting unrelated nonempty shards.

## Switches and scope

| Setting                                         | Training-side behavior                                                                       |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `enable_mtp=false`, `enable_mtp_training=false` | No MTP model parameters on training GPUs; preserve frozen source MTP during HF export.       |
| `enable_mtp=true`, `enable_mtp_training=false`  | Construct the MTP head; auxiliary MTP training is disabled. Use the existing bridge export.  |
| `enable_mtp=true`, `enable_mtp_training=true`   | Construct and train the MTP head with its auxiliary objective; export current model weights. |

This change does not alter inference-engine speculative decoding, NCCL/AWEX transfer
protocols, DCP recovery serialization, LoRA adapter exports, or other bridge
implementations. Disk-based weight updates reuse the completed HF export. DCP records
the constructed training model; with MTP disabled, the original frozen MTP source is
still needed for subsequent HF export. A standalone inference checkpoint is the complete
HF export, not the DCP directory.

No configuration field or dependency is added. Copying uses 8 MiB CPU buffers and does
not construct a model or allocate CUDA tensors. The verified source layout is BF16
Qwen3.8-27B (`model_type=qwen3_5`); quantized export compatibility is not established.

## Verification

CPU safetensors round trips cover mixed MTP/shared source shards, exact tensor values,
configuration and index consistency, single/sharded output, repeated export, missing or
partial weights, truncated files and source-directory protection. Tests also execute the
actual engine save method with bridge doubles to check enabled-head and LoRA bypasses.
The CPU suite also covers HF resume and re-export with an empty MTP-only shard. A
separate copy check against the actual Qwen3.8-27B source verified all 15 MTP tensors
(849,398,784 payload bytes) with per-tensor SHA-256 and safetensors header loading.
Temporary copied weights were removed after verification. Full distributed Megatron save
and SGLang reload require GPU integration validation.

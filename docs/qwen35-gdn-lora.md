# Qwen3.5 Gated DeltaNet LoRA

This note covers `patch_qwen35_hybrid_lora_specs` in
`areal/engine/megatron_utils/megatron_bridge_patches.py`: why Gated DeltaNet `in_proj`
is unfused, and why that unfuse still leaves one adapter on the concatenated mixer
instead of four independent ones.

## Why `in_proj` is unfused

Stock AReaL Qwen3.5 LoRA already unfuses two projections: full-attention `linear_qkv`
and MLP `linear_fc1`. The default target list is `linear_qkv`, `linear_proj`,
`linear_fc1`, and `linear_fc2`. Gated DeltaNet layers do not have those names. They have
one concatenated `in_proj` and an `out_proj`, and the original patch left `in_proj`
fused with the input RMSNorm. A stock Qwen3.5 LoRA SFT run therefore freezes every
DeltaNet mixer. That is most of a Qwen3.5 trunk.

AReaL's `lora_targets_preset: all` (and `attn`) adds the mixer linears on a hybrid
backbone: `in_proj` and `out_proj`. The `dense` preset is the same attention and MLP
matrices, with the DeltaNet projections left out. `qv` does not touch them either.
`PointerHead` is unrelated. It stays a dense linear and is never a LoRA target. Full
fine-tuning does not need the unfuse either; the fused norm-plus-linear still trains.

Hugging Face stores the mixer as five plain `nn.Linear`s (`in_proj_qkv`, `in_proj_z`,
`in_proj_a`, `in_proj_b`, `out_proj`), and PEFT can wrap those names separately. A
Hugging Face training forward still calls the four input projections separately.

Megatron-Bridge builds the opposite graph. `get_gated_delta_net_module_spec` makes
`in_proj` one fused LayerNorm-plus-linear (`fuse_input_layernorm: True`) whose weight is
those four Hugging Face matrices concatenated. `GDNLinearMappingSeparate` in
Megatron-Bridge is the load/save map between that one tensor and `in_proj_qkv` /
`in_proj_z` / `in_proj_b` / `in_proj_a`. Megatron-Bridge LoRA adds the adapter to the
normalized activation, and it reads that activation back from a fused linear that
supports `return_layernorm_output`. MindSpeed's fused LayerNorm-plus-linear does not
return it. That is the same reason `linear_qkv` and `linear_fc1` are already split into
`PTNorm` plus `ColumnParallelLinear`. With `unfuse_gdn=True` the patch does that split
for GDN `in_proj`, and it replaces `out_proj` with a plain `RowParallelLinear` so the
LoRA wrapper will wrap it.

## One `in_proj` adapter versus four mixer adapters

The pointer-head plan (`qwen3.5_pointerhead_decision_training`) and
`docs/qwen35-decision-head-plan.md` both stop at this unfuse. Their LoRA target list
names the fused modules: `in_proj` and `out_proj`, one adapter each. Hugging Face PEFT
can instead wrap `in_proj_qkv`, `in_proj_z`, `in_proj_b`, and `in_proj_a` as four
independent rank-`r` adapters.

Those four Hugging Face names are independent rank-`r` adapters. Each has its own
down-projection. `out_proj` is one more, and that one already matches Megatron: both
sides have a single output projection.

After the unfuse, Megatron still has one `in_proj` adapter. Its down-projection is
`r × hidden`, and its up-projection covers the concatenated rows `[q, k, v, z, b, a]`.
The four slices share one rank-`r` bottleneck. Four independent adapters spend three
extra down-projections (`3 × r × hidden`) so a qkv update cannot use up the rank that
`z`, `b`, and `a` need. `b` and `a` are only `num_v_heads` rows, so a shared
up-projection puts almost all of the rank into qkv. Those two small outputs are the
mixer's beta and decay, so the missing rank is not a small slice of a big matrix.

Megatron-Bridge's PEFT export does not remove the gap. `peft_bridge` slices the single
up-projection into the four Hugging Face names and writes the same down-projection onto
all four (`_split_gdn_in_proj_linear_out_weight` in
`megatron/bridge/models/conversion/peft_bridge.py`). The saved file looks like four PEFT
modules. The four `lora_A` tensors are copies of one matrix. A checkpoint trained with
four independent down-projections is not the same object. Loading one layout into the
other drops three of them. The split has to exist while the adapter is trained.

`in_proj_qkv` itself stays one adapter. Hugging Face does not give q, k, and v separate
mixer adapters. Splitting those three would be a second mismatch.

The same pattern already exists for full attention and the MLP, and AReaL does not use
the fix Megatron-Bridge ships for it. Plain `LoRA` puts one adapter on fused
`linear_qkv` and one on fused `linear_fc1`. Hugging Face PEFT targets `q_proj`,
`k_proj`, `v_proj` and `gate_proj`, `up_proj` separately. `CanonicalLoRA`
(`megatron/bridge/peft/canonical_lora.py`) keeps those fused base GEMMs and adds
separate adapters, then scatters the adapter outputs into Megatron's packed row order.
AReaL's Megatron hook constructs plain `LoRA`, so Qwen3.5 attention and MLP LoRA have
this same shared-rank gap. Matching the Hugging Face target set means using
`CanonicalLoRA` for those projections and extending it for the mixer.

## How to match four mixer adapters

Keep the unfused `ColumnParallelLinear` as the frozen base GEMM. Leave
`GDNLinearMappingSeparate` on that one weight. Add a wrapper in the style of
`LoRALinearSplitQKV` / `LoRALinearSplitFC1UpGate`, used when `lora_targets_preset` is
`all` or `attn`:

- `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`: four `ParallelLinearAdapter`s on
  the post-norm activation. `in_proj_qkv` is one adapter, with one down-projection,
  covering q, k, and v together.
- Pack the four outputs into the row order `GatedDeltaNet.forward` splits: local
  `[q, k, v, z, b, a]`, then add that vector to the base `in_proj` output. The section
  sizes are the ones in `merge_gdn_linear_weights`: `qk_dim`, `qk_dim`, `v_dim`,
  `v_dim`, `num_v_heads`, `num_v_heads`, divided across tensor parallel.
- `out_proj` stays a single `RowParallelLinear` adapter.
- `dense` and `qv` do not build these four adapters.

`z`, `b`, and `a` are each one contiguous block in both layouts, so a normal
column-parallel adapter of those output sizes lands on the right rows. `in_proj_qkv`
does not. Hugging Face stores it as flat `[Q; K; V]`. Megatron stores each
tensor-parallel rank's rows as local q, then local k, then local v
(`merge_gdn_linear_weights`). At tensor parallel 1 those orders are the same and a stock
column-parallel adapter is enough. Above that, a contiguous shard of the flat matrix is
a prefix of Q (or a span across Q and K), while the base linear owns a slice of every
group. The qkv adapter's up-projection has to be row-sharded with that blocked layout,
the same job `LoRALinearSplitQKV._interleave_qkv` does for attention. Sharing one
down-projection across those q, k, and v blocks keeps it a single `in_proj_qkv` adapter.

Splitting the base module into four `ColumnParallelLinear`s would also give four
adapters, and it would change the GEMM, the tensor-parallel layout, and the weight
bridge. The wrapper leaves the base forward and the checkpoint map alone.

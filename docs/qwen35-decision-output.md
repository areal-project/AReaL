# Qwen3.5 PointerHead: last-stage output layout

This note is only for the Qwen3.5 **decision** path (`actor.is_decision_model`). It does
not describe SFT, GRPO, critic training, or how AReaL packs sequences in general. Those
pipelines already have their own last-stage reading of the language-model head (token
logprobs or a scalar value).

The decision model is a small add-on on top of that: `PointerHead` replaces
`output_layer` on the last pipeline stage and emits per-token `q`/`k` projections
`[tokens, 2 * dp]`. `_reassemble_and_unpad_output` exists so that tensor lines up with
the packed question rows already in the micro-batch. It is not a new packing feature.

Related notes: [Decision Dataset](en/customization/decision_dataset.md) (JSONL schema
and encoded columns), [decision-head plan](qwen35-decision-head-plan.md),
[GDN LoRA](qwen35-gdn-lora.md).

## What the decision loss needs to see

A decision micro-batch is one causal **row per question** (`state + branch_k`). Each row
already carries the same 1D per-token fields as `input_ids`:

- `token_roles`: `1` at `</opt>`, `2` at `<decide>`, else `0`
- `choice_label` / `question_type` / `soft_target` at those positions

AReaL's existing packer (`pack_tensor_dict`) concatenates those rows into one stream.
The decision loss (still owned by the trainer, not the engine) walks that stream, finds
`<decide>` and `</opt>` in each row, and scores them. For that walk it uses
`cu_seqlens`, which AReaL already attaches to every packed micro-batch.

## `cu_seqlens` (existing AReaL field, reused here)

`cu_seqlens` is a length-`B+1` integer tensor. Entry `i` is where row `i` starts in the
packed stream; the last entry is the total token count.

```
two question rows, lengths 5 and 6
cu_seqlens = [0, 5, 11]
row 0 = packed[0:5]
row 1 = packed[5:11]
```

The decision loss does:

```python
for i in range(len(cu_seqlens) - 1):
    sl = slice(int(cu_seqlens[i]), int(cu_seqlens[i + 1]))
    roles = token_roles[sl]
    # </opt> at roles == 1, <decide> at roles == 2
```

Dummy rows added so every pipeline rank has the same number of micro-batches have no
`<decide>` token and are skipped. Nothing here changes how SFT or GRPO use `cu_seqlens`.

Megatron may then pad each row so its length is a multiple of `tp` (and `tp * cp * 2` if
context parallel is on). That alignment is also existing engine code (`pad_mb_list`):

| Field                                | Meaning on a decision micro-batch                   |
| ------------------------------------ | --------------------------------------------------- |
| `cu_seqlens` on the **forward** pack | Boundaries after alignment pads                     |
| `old_cu_seqlens`                     | Boundaries of the real tokens (what the loss wants) |

Example, same two rows aligned to 8:

```
forward pack:     A B C D E pad pad pad | a b c d e f pad pad
cu_seqlens:       [0, 8, 16]
old_cu_seqlens:   [0, 5, 11]
```

When `CP == 1` (the Qwen3.5 case today — the padded BSHD path rejects `CP > 1`),
`forward_backward_batch` already unpads last-stage output before the loss callback. The
decision branch then just calls `loss_fn(output, inputs)`.

## `_reassemble_and_unpad_output`

`MegatronEngine._reassemble_and_unpad_output` is a last-stage helper used **only** when
`is_decision_model` is set, from `_compute_logprobs_and_loss` and
`_compute_forward_result`. Actor logprobs and critic values do not go through it.

```python
# areal/engine/megatron_engine.py
def _reassemble_and_unpad_output(self, output, inputs):
    cp_padded_cu_seqlens = inputs.get("_cp_padded_cu_seqlens")
    if cp_padded_cu_seqlens is None:
        return output, inputs   # Qwen3.5 today: already unpadded
    output = reassemble_cp_packed_logprobs(output, cp_padded_cu_seqlens)
    output = unpad_logits(
        output,
        inputs.get("_cp_padding_length", 0),
        cp_padded_cu_seqlens,
        inputs.get("_cp_old_cu_seqlens"),
    )
    inputs = {k: v for k, v in inputs.items() if not k.startswith("_cp_")}
    return output, inputs
```

Why it exists: if this decision model is ever run with `CP > 1`, each rank only holds a
slice of every question row. `PointerHead` scores `<decide>` against every `</opt>` **in
that same row**. Those tokens can land on different CP ranks, so the helper all-gathers
the local `[tokens, 2 * dp]` slices, restores original order, then drops alignment pads.

It reuses `reassemble_cp_packed_logprobs` / `unpad_logits` that the actor path already
uses for 1D logprobs. The only decision-specific part is applying them to the 2D pointer
projections instead of calling `gather_logprobs_entropy`.

### Example (`CP = 2`, two question rows)

This path is **not** used for current Qwen3.5 decision runs (`CP` must be 1). It is here
so the helper is not a black box if someone later enables CP.

Real lengths 5 and 6, aligned to 8 + 8. Packed length 16; each CP rank sees 8 tokens.
Megatron's causal split gives each rank the first half-chunk and the mirrored last
half-chunk of every row.

One aligned row `[A B C D E F G H]`:

| Rank | Tokens          |
| ---- | --------------- |
| 0    | `A B` and `G H` |
| 1    | `C D` and `E F` |

`<decide>` might be `H` on rank 0 while `</opt>` tokens are on rank 1. Rank 0 cannot
score the question from its slice.

After all-gather + permute:

```
[A B C D E F G H | a b c d e f g h]     # length 16
```

After `unpad_logits` with `old_cu_seqlens = [0, 5, 11]`:

```
[A B C D E | a b c d e f]               # length 11
```

That is the `[11, 2 * dp]` tensor the decision `loss_fn` should receive, next to
`inputs` whose `cu_seqlens` is `[0, 5, 11]` and whose `token_roles` already match those
11 tokens.

## What this helper does not do

- It does not pack the batch. Packing is existing `MegatronEngine._prepare_mb_list`.
- It does not implement `question_loss`. The trainer still has to pass
  `loss_fn(output, inputs)` into `train_batch`.
- It does not change the actor or critic last-stage paths.
- On today's Qwen3.5 decision config it returns `(output, inputs)` unchanged.

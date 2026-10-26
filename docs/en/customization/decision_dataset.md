# Decision Dataset

A decision model does not write an answer token by token. It reads a **state** (the
document, ticket, or passage) and one or more **questions**, then points at one of the
question's **options**.

You write labelled questions in JSONL. AReaL turns each question into one training row
the PointerHead loss can score. You do not build `choice_label` or `soft_target`
yourself; those tensors are produced when the dataset is loaded.

This page is the user guide for that JSONL format and for the
`train_dataset.dataset_kwargs` that control optional rewriting of choice questions.

## What you write vs. what training sees

| You write (JSONL)                                            | Training sees (one row per question)                                       |
| ------------------------------------------------------------ | -------------------------------------------------------------------------- |
| `state` + `questions` with a `label` (and optional `target`) | `input_ids`, `token_roles`, `choice_label`, `question_type`, `soft_target` |

`augment`, `p_none`, and `p_none_pair` only change the JSONL **before** that encoding.
They are training-time rewrites of choice questions. They are not columns in the batch.

## Question types and labels

Every record is one JSON object with a `state` and a non-empty `questions` object. Each
question has a `type`, optional `instructions` / `criteria`, and a required `label`.

```json
{"state": "Two charges posted for order 4411.",
 "questions": {
   "team": {
     "type": "choice",
     "instructions": "Which team owns this?",
     "criteria": {"billing": "Payments and invoices", "shipping": "Delivery"},
     "label": "billing"
   },
   "angry": {
     "type": "noul",
     "instructions": "Is the customer angry?",
     "label": false
   },
   "priority": {
     "type": "score",
     "instructions": "How urgent is this?",
     "criteria": ["low", "normal", "high"],
     "label": 1
   }
 }}
```

`state`, `instructions`, and each option description may be a string, a number, a list,
or a nested object. Nested values are flattened to text (field names are kept as
labels).

### `choice`

Pick exactly one named option. `criteria` is a dict of option key → description (`null`
is allowed). **`label` is the option key**, not an index: `"billing"`, not `0`.

### `noul`

A yes / no question. **`label` is a boolean.** Options are always `false` then `true`
(shown to the model as `no` / `yes`). You may add `criteria.false` / `criteria.true` to
describe those two sides.

### `score`

An ordered scale. `criteria` is a list of level descriptions from low to high. **`label`
is the 0-based level index** (`1` means the second level).

### Soft targets (`target`)

When the answer is a distribution rather than a single option (for example uniform over
every choice because the case is unknowable), add `target` next to `label`. Keys match
the option names (`choice`), `"false"` / `"true"` (`noul`), or `"0"`, `"1"`, …
(`score`). Missing keys are treated as 0, then the vector is normalized.

```json
"target": {"billing": 0.5, "shipping": 0.5}
```

A hard `label` is still required. The loss uses `target` when it is present.

## Files and config

Point `train_dataset.path` at either:

- a `.jsonl` file (one record per line), or
- a suite directory with `manifest.json` plus `train.jsonl` / `development.jsonl` (or
  `dev.jsonl`). If the manifest lists a sha256 or record count, both are checked.

```yaml
train_dataset:
  type: decision
  path: /data/tickets.jsonl
  split: train
  dataset_kwargs:
    seed: 0
    max_state: 384
    max_branch: 1024
    # leave these out to use the defaults below
    # augment: true
    # p_none: 0.1
    # p_none_distract: 0.12
    # p_distract: 0.15
    # p_none_pair: 0.0
```

`type: sft` also selects this loader when `path` is a decision JSONL, a suite directory,
or contains `decision`.

| `dataset_kwargs`  | Default                                         | What it does                                                         |
| ----------------- | ----------------------------------------------- | -------------------------------------------------------------------- |
| `augment`         | `true` on `train`, `false` on every other split | Run the choice-question rewrites below                               |
| `p_none`          | `0.1`                                           | Probability of the “none is correct” rewrite                         |
| `p_none_distract` | `0.12`                                          | Probability of adding a wrong “none of the above”                    |
| `p_distract`      | `0.15`                                          | Probability of adding an irrelevant extra option                     |
| `p_none_pair`     | `0.0`                                           | Probability of also emitting a present/absent pair                   |
| `seed`            | `0`                                             | Per-record RNG seed for those rewrites                               |
| `max_state`       | `384`                                           | Token budget for `<state>` + the state text                          |
| `max_branch`      | `1024`                                          | Token budget for one full row (`state` + one question)               |
| `max_length`      | from `train_dataset.max_length`                 | Drop rows longer than this after encoding                            |
| `verify_suite`    | `true`                                          | Check suite checksums / record counts                                |
| `n_per_source`    | off                                             | Keep the first N records of each `_meta.source` (PoC / smoke slices) |
| `max_records`     | off                                             | Keep at most this many records after `n_per_source`                  |

To encode the file **exactly as written**, set `augment: false` (or use a non-train
split). Validation and test already do that by default.

`p_none + p_none_distract + p_distract` must be `<= 1`. They are three non-overlapping
slices of a single random draw, not independent coin flips.

## `augment`: rewriting choice questions

Only **`choice`** questions are rewritten. `noul` and `score` are left alone. Questions
that carry a soft `target` only have their option **order** shuffled; no option is
inserted or removed, because that would change what `target` means.

When `augment` is on, every choice question is processed as follows.

### Option order is always shuffled

The model must not learn “the first option is the answer”. After any of the insertions
below, the remaining keys are shuffled.

### `p_none` — make “none of the above” the answer

With probability `p_none`, and only when the question already has **more than two**
options:

1. The labelled option is **removed**.
1. A “none of the above” option is inserted (wording is picked from a small pool so the
   model cannot latch onto one phrase).
1. That new option becomes the label.

Use this so the model actually picks “none” when the true reason is missing, instead of
forcing a wrong leftover option.

### `p_none_distract` — add “none” as a wrong extra option

With probability `p_none_distract`, the true option **stays**, and a “none of the above”
option is added as a distractor. The label does not change.

Use this so “none of the above” is not a shortcut for “pick this wording”.

### `p_distract` — add an irrelevant option

With probability `p_distract`, an unrelated option is added (for example weather or tax
filing) and the label is unchanged.

Use this so the model ignores options that do not appear in the state.

The three events are exclusive. For the default `0.1 / 0.12 / 0.15`, a given choice
question is:

- rewritten as “none is correct” 10% of the time
- given a wrong “none” 12% of the time
- given an irrelevant extra option 15% of the time
- only shuffled 63% of the time

### `p_none_pair` — extra present / absent siblings

This is **in addition to** the rewrite above, not one of those three slices. Default is
`0` (off).

With probability `p_none_pair`, AReaL also appends two extra records for one eligible
choice question (at least three options, no soft `target`):

| Sibling     | Options                                       | Label                       |
| ----------- | --------------------------------------------- | --------------------------- |
| **present** | original options + a “none” option            | the original correct option |
| **absent**  | the same list with the correct option removed | that same “none” option     |

Option order is shared. The only difference is whether the evidence still matches a
listed option. That pair teaches the model that “none” is about coverage, not about the
phrase itself.

A train record with three questions and `p_none_pair: 1.0` therefore yields the three
original question rows **plus** two extra rows (the present and absent siblings), not
three extra copies of the whole record.

## Encoded row fields

After loading, each dataset item is **one question**. A record with three questions
becomes three rows. Each row is the state tokens followed by that question's
instruction, options, and a final decide token.

Schematic tokens for a two-option choice:

```
<state> Two charges posted for order 4411.
<q> Which team owns this?
<opt> billing: Payments and invoices </opt>
<opt> shipping: Delivery </opt>
<decide>
```

Those delimiters are existing special tokens in Qwen tokenizers (`<|fim_prefix|>`,
`<|fim_middle|>`, `<|box_start|>`, `<|box_end|>`, `<|fim_suffix|>`). User text cannot
forge them: a literal `<|box_end|>` in the state is rewritten before tokenization.

`attention_mask` is added later when rows are collated. The columns you get from the
dataset are:

### `input_ids`

Token ids of that schematic sequence.

### `token_roles`

Same length as `input_ids`. Marks the two positions the loss reads:

| Value | Position                                                   |
| ----- | ---------------------------------------------------------- |
| `0`   | everything else (state, instruction, option text, padding) |
| `1`   | each `</opt>` token, in option order                       |
| `2`   | the `<decide>` token                                       |

The loss scores the `<decide>` hidden state against every `</opt>` hidden state in the
same row.

### `choice_label`

Same length as `input_ids`. `-1` everywhere except at `<decide>`, where it holds the
integer option index `y` (0-based, in the order options appear after any shuffle /
insertion).

You never write this index in JSONL. `"label": "billing"` becomes `0` or `1` depending
on the encoded option order.

### `question_type`

Same length as `input_ids`. `-1` everywhere except at `<decide>`:

| Value | JSONL `type` |
| ----- | ------------ |
| `0`   | `choice`     |
| `1`   | `noul`       |
| `2`   | `score`      |

The loss uses this to decide whether to add the ordinal (ranked-probability) term on
`score` questions.

### `soft_target`

Same length as `input_ids`, float.

- **Hard label only:** every position is `-1.0`. The loss uses `choice_label`.
- **Soft `target` present:** each `</opt>` position holds that option's probability
  (`t_0`, `t_1`, …). Other positions stay `-1.0`. The loss uses this distribution.

`-1.0` means “no soft label here”, not a probability.

### Worked numbers

For the `team` question above, if options encode as `billing` then `shipping` and the
hard label is `billing`:

| Token                             | `token_roles` | `choice_label` | `question_type` | `soft_target` |
| --------------------------------- | ------------- | -------------- | --------------- | ------------- |
| state / instruction / option text | `0`           | `-1`           | `-1`            | `-1.0`        |
| `</opt>` of billing               | `1`           | `-1`           | `-1`            | `-1.0`        |
| `</opt>` of shipping              | `1`           | `-1`           | `-1`            | `-1.0`        |
| `<decide>`                        | `2`           | `0`            | `0`             | `-1.0`        |

If the same question also had `"target": {"billing": 0.7, "shipping": 0.3}`, the two
`</opt>` `soft_target` values would be `0.7` and `0.3`. `<decide>` still stores
`choice_label = 0`.

These 1D fields pack with `input_ids` through AReaL's usual micro-batch splitter and
sequence packer. Dummy pad tokens used for alignment are role `0` and are ignored.

## Length limits

A row that does not fit is skipped (a warning is logged):

- `max_state`: tokens in `<state>` plus the state text.
- `max_branch`: tokens in the whole row (state + that question).
- `train_dataset.max_length`: optional extra cap after encoding.

Raise `max_state` / `max_branch` together when your states are long. The defaults (`384`
/ `1024`) match a short-document recipe.

## Recommended starting points

**Encode the file as-is** (debugging, or you already shuffled options offline):

```yaml
dataset_kwargs:
  augment: false
```

**Default train-time choice robustness** (shuffle always; occasional none / distractor
insertions; no extra sibling rows):

```yaml
dataset_kwargs:
  # augment defaults to true on split=train
  p_none: 0.1
  p_none_distract: 0.12
  p_distract: 0.15
  p_none_pair: 0.0
```

**Also train explicit none-present / none-absent pairs** (only on choice questions with
at least three options):

```yaml
dataset_kwargs:
  p_none_pair: 0.25
```

Keep `augment: false` on the validation set so reported accuracy matches the published
labels.

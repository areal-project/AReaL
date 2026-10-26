# Qwen3.5 PointerHead decision SFT

Train a Qwen3.5 decision model (PointerHead readout + Gated DeltaNet LoRA) with AReaL on
Ascend NPUs.

## Single-NPU PoC

Two epochs, LoRA rank 16, lr `5e-5`, `head_dim=256`. Give the script a local Qwen3.5
checkpoint and your decision data. If you pass a suite directory it samples
`n_per_source` records so you do not encode the full train split.

Put this checkout first on `PYTHONPATH` if the image also has an `/AReaL` install. The
launch script appends the node IP to `NO_PROXY` so local worker health checks are not
sent through `HTTP_PROXY`.

### Suite directory (sampled)

A suite is a folder with `manifest.json` plus `train.jsonl` / `development.jsonl`.

```bash
cd /path/to/AReaL   # repo root

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0

MODEL_PATH=/path/to/Qwen3.5-4B-Base \
SUITE_PATH=/path/to/decision-suite \
  bash examples/decision/run_npu_poc.sh
```

### Already-sliced JSONL

```bash
MODEL_PATH=/path/to/Qwen3.5-4B-Base \
TRAIN_PATH=/path/to/train.jsonl \
VALID_PATH=/path/to/development.jsonl \
  bash examples/decision/run_npu_poc.sh
```

### Unpacked command

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0

python3 examples/decision/prepare_v7_sample.py \
  --suite /path/to/decision-suite \
  --out-dir examples/decision/data \
  --n-per-source 32 --dev-per-source 4

python3 examples/decision/train_decision.py \
  --config examples/decision/qwen3_5_decision_sft_npu_poc.yaml \
  scheduler.type=local \
  cluster.n_gpus_per_node=1 \
  actor.path=/path/to/Qwen3.5-4B-Base \
  train_dataset.path=$PWD/examples/decision/data/poc-train.jsonl \
  valid_dataset.path=$PWD/examples/decision/data/poc-dev.jsonl
```

A successful run finishes two epochs, logs `sft/decision_loss` trending down, writes
LoRA adapters plus `head.pt`, and exits 0. Override `EPOCHS`, `STEPS`, `N_PER_SOURCE`,
or `FILEROOT` on the shell script.

## Pytest e2e

The two-step NPU test skips unless you point it at local assets:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DECISION_MODEL=/path/to/Qwen3.5-4B-Base
export DECISION_SUITE=/path/to/decision-suite
# or: export DECISION_TRAIN=/path/to/train.jsonl DECISION_VALID=/path/to/development.jsonl

python3 -m pytest tests/test_qwen35_decision.py tests/test_qwen35_decision_e2e.py \
  -q --tb=short
```

`test_qwen35_decision_e2e.py::test_qwen35_decision_sft_single_npu_poc` is marked `npu`
and `slow`. It trains two steps, checks `decision_loss` is logged, and expects a
`head.pt` checkpoint.

## Parallelism

`actor.backend` is a Megatron layout. The world size is `DP * TP * PP * CP * EP`, and on
one node that equals `cluster.n_gpus_per_node`.

| Layout                       | Backend                          | On Qwen3.5-4B-Base                                            |
| ---------------------------- | -------------------------------- | ------------------------------------------------------------- |
| Data parallel (DP)           | `megatron:d2`, `megatron:d4`     | Replicates the model and splits the batch                     |
| Tensor parallel (TP)         | `megatron:d1t2`, `megatron:d1t4` | Splits layers. Sizes 2 and 4 divide the 4 KV heads            |
| Sequence parallel (SP)       | same backend as TP               | Turns on automatically when TP is greater than 1              |
| Context parallel (CP)        | `megatron:d1c2`, `megatron:d1c4` | Splits each packed sequence across NPUs                       |
| Pipeline parallel (PP) / VPP | `p` in the backend string        | Unsupported for this head. See below                          |
| Expert parallel (EP)         | `e` in the backend string        | For MoE models. This 4B checkpoint is dense, so leave EP at 1 |
| DP + TP                      | `megatron:d4p1t2`                | 8 NPUs, pipeline size 1. This is `qwen3_5_decision_sft.yaml`  |

`examples/decision/run_npu_parallel.sh` maps the supported layouts to
`PARALLEL=single|dp2|tp2|cp2|d4t2`. `tp2` is also the sequence-parallel example. A step
cap writes a checkpoint and runs eval at the last step. Run one layout at a time on a
single node so each local scheduler can bind its own ports.

Single-NPU convergence on a small suite slice (about a few minutes after load):

```bash
MODEL_PATH=/path/to/Qwen3.5-4B-Base \
SUITE_PATH=/path/to/decision-suite \
N_PER_SOURCE=8 DEV_PER_SOURCE=2 EPOCHS=1 STEPS=24 \
PARALLEL=single DEVICES=0 \
  bash examples/decision/run_npu_parallel.sh
```

Two-step checks, one layout at a time:

```bash
MODEL_PATH=/path/to/Qwen3.5-4B-Base \
SUITE_PATH=/path/to/decision-suite \
N_PER_SOURCE=2 DEV_PER_SOURCE=1 STEPS=2 EPOCHS=1 \
PARALLEL=dp2 DEVICES=0,1 \
  bash examples/decision/run_npu_parallel.sh

PARALLEL=tp2 DEVICES=0,1 STEPS=2 \
  bash examples/decision/run_npu_parallel.sh

PARALLEL=cp2 DEVICES=0,1 STEPS=2 \
  bash examples/decision/run_npu_parallel.sh

PARALLEL=d4t2 DEVICES=0,1,2,3,4,5,6,7 STEPS=2 \
  bash examples/decision/run_npu_parallel.sh
```

`test_qwen35_decision_e2e.py::test_qwen35_decision_sft_multi_npu` runs the two-step
`dp2`, `tp2`, and `cp2` layouts the same way as the single-NPU test: `decision_loss` is
logged and `head.pt` is saved. It is marked `multi_npu` and `slow`.

### Pipeline parallel

Pipeline parallel does not train this head. Qwen3.5 ties the input embedding to the
language-model head. The key issue is that Megatron's tied-embedding machinery assumes
the output layer is a vocabulary projection, but PointerHead is a different kind of
head. With pipeline size 1 the tie stays on one rank, which is why DP, TP, and CP work.
With pipeline size greater than 1, rank 0 owns the embedding and the last rank owns
`PointerHead`. Megatron still treats that head as the tied vocabulary matrix and reads
`output_layer.weight` from it.

Two of those reads showed up in the NPU runs:

1. **Load.** `MegatronModelBridge._broadcast_shared_embeddings` sends the embedding to
   the last stage. That stage has no embedding, so it builds the receive buffer with
   `empty_like(output_layer.weight)`. `PointerHead` has `q` and `k` only, so initialize
   raises `AttributeError`.
1. **Forward and gradient sync.** `GPTModel.shared_embedding_or_output_weight` returns
   `self.output_layer.weight` on the post-process stage, and `finalize_model_grads`
   all-reduces that tensor with the stage-0 embedding gradient. The first training step
   hits the same missing attribute.

The load crash can be patched on its own. Before `_broadcast_shared_embeddings`, attach
a temporary `vocab × hidden` parameter named `weight` on the `PointerHead` (dtype
`params_dtype`, not the head's fp32 `q`/`k`). Let the broadcast copy the embedding into
it, then delete the parameter before the optimizer is built. That gets past
`initialize`. It does not make a step succeed: the temporary parameter is gone by the
time the forward asks for `output_layer.weight` again. Leaving the matrix in place would
also register a tied vocabulary parameter the decision loss never uses, and the gradient
all-reduce would still expect it to be the shared embedding.

Untying the embedding is a model-construction change. Stage 0 allocates that matrix as a
shared weight (`skip_weight_param_allocation` on the output layer). A head-only patch
cannot turn the tie off after the pipeline modules exist. `run_npu_parallel.sh` exits if
the parsed pipeline size is greater than 1.

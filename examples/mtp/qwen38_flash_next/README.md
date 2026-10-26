# Flash-Next native MTP-only SFT

Train the text component of `Qwen4ExpForConditionalGeneration` through Megatron-Bridge.
The native MTP subtree is the only trainable subtree. Text backbone, embeddings, output
head, PLE and vision weights retain their source values. Vision tensors and the frozen
PLE table are streamed from the source when exporting HF weights. HF export requires
`mtp_only=True`: the other Bridge training modes do not yet apply `freeze_ple_table`, so
trusting that flag could silently discard PLE updates. This recipe uses CP1 and padded
BSHD; it does not implement vision training.

## Runtime

Use a compatible AReaL CUDA environment. The validated runtime uses Python 3.12, PyTorch
2.9.1+cu129, Transformer Engine 2.14.1, cuDNN 9.19.0.56 and these pins:

| Dependency                    | Version/source                                                                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Megatron-Core                 | 0.19.0, `5be9626709af2722333bf54797c954c09edeada3`                                                                                                     |
| Megatron-Bridge               | 0.6.0, `5d3268b972d55fe1d08b98b04e8c453467d4e51d`                                                                                                      |
| Qwen4Exp components           | [mcore-bridge](https://github.com/dingzhiqiang/mcore-bridge/tree/7ad1b2d0e1ebc1b11b87c469719f6225e645d2fe), `7ad1b2d0e1ebc1b11b87c469719f6225e645d2fe` |
| Transformers / tokenizers     | 5.16.1 / 0.23.1                                                                                                                                        |
| safetensors / huggingface-hub | 0.8.0 / 1.5.0                                                                                                                                          |

The Bridge wheel is the
[same pinned release used by AReaL](https://github.com/sitabulaixizawaluduo/Megatron-Bridge/releases/download/areal-v0.6.0-exaone45-fix.2/megatron_bridge-0.6.0-py3-none-any.whl).
Its SHA256 is `d50f79060836d39958ef80ca9e7e8b6766b3da24d1d96b42e22e3ac91be2251c`. The
model component library supplies GDN, QSA, hyper-connections, PLE and MTP;
Megatron-Bridge handles conversion. Keep this environment isolated: the model's
Transformers version differs from AReaL's default dependency range. Verify the imported
packages and the label-free `process_mtp_loss` API before a large run; the latter must
accept `input_ids` and `tp_group`.

## Two-node run

Set `MODEL_PATH` to the original HF checkpoint. Prepare a fixed, tokenized dataset:

```bash
export OUTPUT_ROOT=<new-data-directory>
python examples/mtp/qwen38_flash_next/prepare_fixed_dataset.py
export SFT_DATASET=$OUTPUT_ROOT/fixed_sft_16384
```

The eight deterministic natural-language samples contain 16,384 actual tokens and 16,256
supervised tokens apiece; prompt tokens are masked, with no padding counted as
supervision. This fixed-length example uses four genuine microbatches per update.
Acceptance requires near-16K effective samples, rather than a rule that every production
microbatch must have that length.

Inside an existing two-node, eight-GPU-per-node allocation, start one worker on each
node with the same runtime and source. Set `NODE_RANK` to 0 or 1:

```bash
export NNODES=2 NPROC_PER_NODE=8 NODE_RANK=<node-rank>
export MASTER_ADDR=<rank-zero-host> MASTER_PORT=<free-port>
export FILERoot=<new-output-directory> TRIAL_NAME=continuous-16384
export TOTAL_TRAIN_STEPS=5 SEQUENCE_LENGTH=16384
export VERIFY_PARAMETER_STEPS=5 PARAMETER_VERIFICATION_DIR=$FILERoot/parameter-checks
export ACCEPTANCE_METRICS_FILE=$FILERoot/acceptance_metrics.jsonl
bash examples/mtp/qwen38_flash_next/run.sh train
```

The config selects TP8/PP2/EP8, BF16, full recomputation, and all four switches:
`bridge_type: megatron-bridge`, `enable_mtp: true`, `enable_mtp_training: true`,
`mtp_only: true`. Allocate resources through your site's scheduler; `run.sh` only starts
processes inside that allocation.

`instrumented_sft.py` records finite losses/gradient norms, per-step throughput and
maximum rank memory. Optional parameter checks hash all rank-local frozen parameters
before and after the run and require the MTP state to change on its owning ranks.
Hashing and snapshots run outside the measured training step.

## Recovery and export

`acceptance_tolerances.yaml` declares exact conversion and combined BF16 bounds before
comparison. Use the same seed, dataset, topology and optimizer for recovery. For a 16K
continuous five-update baseline, set `RECOVER_MODE=auto`, `RECOVER_FREQ_STEPS=4`,
`MTP_SNAPSHOT_STEP=5`, `MTP_SNAPSHOT_LABEL=continuous-step5`, and shared new
`MTP_SNAPSHOT_DIR` and `RECOVERY_STATE_DIR` directories. The checkpoint saved after
update 4 has `global_step=3`; its restored next update has zero-based index 4 (update
5).

Start a fresh `recovered-export` process with the original recovery trial config,
`TOTAL_TRAIN_STEPS=5`, `MTP_SNAPSHOT_STEP=1`, `MTP_SNAPSHOT_LABEL=recovered-step5`,
`VERIFY_PARAMETER_STEPS=1`, and a distinct `EXPORT_TRIAL_NAME` for its logs.
`export_recovered.py` preserves the original recovery config while changing the output
trial. Compare with `compare_recovery.py`, setting `SNAPSHOT_ROOT`,
`CONTINUOUS_METRICS_FILE`, `RECOVERED_METRICS_FILE`, and `RECOVERY_STATE_DIR`. It
requires identical next-input batch hashes, exact restored model/FP32 optimizer
master/moment/scheduler/RNG boundary state, and numerically matching next updates and
losses. Runtime snapshots are large evidence artifacts; keep them outside the source
checkout.

`run.sh full-parity` compares short fixed-input logits/loss for the full loaded
checkpoint against the mcore-bridge model path. It moves the candidate to CPU before
building the reference and exits afterward; it never trains or exports using those moved
DDP buffers. Reference weights are copied from the converted candidate, so this proves
equal-weight forward equivalence. The separate complete HF roundtrip establishes source
conversion provenance.

For complete untrained HF roundtrip, set `HF_EXPORT_ROOT` to a new directory and
`SOURCE_ROUNDTRIP_TRIAL_NAME`, plus expected source config/index SHA256 variables, then
run `run.sh source-export`. Verify all payload bytes with `verify_source_roundtrip.py`.
For trained export use `run.sh recovered-export` with `EXPORT_TRIAL_NAME`, then
`verify_recovery_export.py` to check all keys, shards, changed MTP and exact
preservation of every frozen non-MTP tensor, including the backbone, embeddings, output
head, PLE and vision. These validators refuse incomplete or mismatched inputs; their
path inputs are environment variables.

Tiny graph checks (`conversion_roundtrip.py`, `no_mtp_parity.py`,
`mtp_only_two_step.py`) complement full-model experiments. Their results alone do not
establish full-model training or recovery acceptance.

Recovery snapshots include exact fingerprints of every frozen backbone parameter (with
all registered aliases, shape and dtype), plus persistent PLE hash buffers. Compare
these before the first update, at save/restore, and after the next update. The trained
HF export must preserve every non-MTP source tensor byte exactly, including embeddings,
the LM head, experts, PLE and vision; only native MTP keys are excluded from that
frozen-payload comparison.

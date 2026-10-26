# GLM-5.3-Flash bring-up

This is an experimental text actor integration for `Glm5NextForConditionalGeneration`
(`glm5_next`, nested `glm5_next_text`). CPU architecture tests, focused CUDA kernel
tests, a reduced four-layer distributed SFT smoke, and three optimizer updates of the
original 45-layer model with FP32 master weights and moments have passed. The reduced
smoke covered TP1/EP8 and three-node TP4/EP12, HF export, DCP save, and a follow-up
optimizer/model/LR/RNG reload from global step 3. The original model's profile512 NLL
was 1.87--1.89 across three data-parallel replicas on TP4/EP12; this is the first
full-model loss evidence. HF logps and per-layer routing have been captured for the same
input, but HF/SGLang numerical parity, routing replay, and live weight transfer still
require validation before production RL. The
[full-model FP32 SFT baseline](#full-model-fp32-sft-baseline) below records the
completed training run separately from those RL gates.

## Architecture and runtime

The source model has 45 layers: 34 KDA and 11 NoPE DSA layers. DSA appears at zero-based
indices 3, 7, ..., 43; layer 44 is KDA. The first three MLPs are dense and the remaining
42 are MoE. Each layer uses four mHC residual streams, 20 Sinkhorn iterations,
`hc_eps=1e-6`, and `rms_norm_eps=1e-5`. SwiGLU clamps at 10. The frozen DSA indexer uses
four-token key pools and always includes the causal unpooled tail.

Use `bridge_type: mbridge`, BF16 actor compute, CP=1, packed THD inputs, and
`use_mbridge_save: false`. KDA is replicated across TP; sequence-parallel gather/scatter
does not shard its weights. DSA uses native TP projections. Full activation recompute is
supported by the integration; the specialized mHC recompute arena, MTP, VPP, custom
pipeline layouts, critic training, and LoRA are outside this initial scope.

The repository's pinned stock Megatron-Core 0.19 lacks the required mHC interfaces. The
implementation targets the interfaces in
[`radixark/Megatron-LM@e8f57451`](https://github.com/radixark/Megatron-LM/tree/e8f57451):
native mHC layer slots, configurable block head contraction, FP32 parameter retention,
and SwiGLU clamping. The layer builder rejects runtimes missing these capabilities.
Dependency pins are unchanged. An explicitly selected runtime also needs Transformer
Engine, FLA's KDA/short-convolution/gated RMSNorm APIs, Triton, and TileLang. FlashMLA
is optional. Kernel sources are adapted from Apache-2.0
[`radixark/miles@ee36bb2`](https://github.com/radixark/miles/tree/ee36bb2), with
provenance in each file. The selected MCore fork imports Miles by default; the
library-mode patch below makes that integration optional for this SFT recipe.

## CPU checks

```bash
python examples/glm5_next/validate.py audit "$GLM53_MODEL"
pytest -q tests/test_glm5_next.py tests/test_glm5_next_dsa.py \
  tests/test_glm5_next_checkpoint.py tests/test_glm5_next_bridge.py
```

The audit reads raw JSON without loading model weights or importing AReaL/CUDA. Add
`--check-runtime` inside the intended training environment to check interfaces, import
the complete Megatron engine, and construct the bridge configuration. This does not
allocate model weights or execute kernels. Actual mbridge conversion tests skip when
mbridge or a compatible mHC runtime is unavailable; the CPU DSA test uses real Torch
projections and compares absorption against explicit dense MLA, including gradients.

The isolated runtime uses the MCore revision linked above and mbridge
`310e8fb35ccf4fcd4419d32973e563a6d43ee5fb`, with Python 3.12, Torch 2.9.1+cu129,
Transformer Engine 2.14.1, FLA 0.4.2, Triton 3.5.1, TileLang 0.1.9, and Transformers
5.3.0. The distributed smoke used the same versions and an external MCore compatibility
patch: absent top-level Miles is optional, while incomplete installations still fail;
checkpoint tensor concatenation also works when Megatron CLI global arguments are not
initialized. Apply the included [library-mode patch](patches/mcore-library-mode.patch)
to the exact MCore revision above before training (dependency pins remain unchanged):

```bash
git -C "$MCORE_SOURCE" apply --check "$AREAL_SOURCE/examples/glm5_next/patches/mcore-library-mode.patch"
git -C "$MCORE_SOURCE" apply "$AREAL_SOURCE/examples/glm5_next/patches/mcore-library-mode.patch"
```

The runtime probe checked tensor merging with unset CLI args and both low-memory modes,
missing versus incomplete Miles imports, and Miles/native replay precedence. R3 itself
has not been validated. Any custom training entry point must protect trainer creation
with `if __name__ == "__main__"` because DCP save spawns subprocesses.

The checkpoint's FP8 storage is distinct from BF16 actor compute. Its root
`quantization_config` is retained; legacy `model.layers.*` exclusions are expanded to
`model.language_model.layers.*`. Both fast save and online conversion honor these
exclusions, including BF16 `kv_b_proj`. KDA Q/K/V convolutions merge into a replicated
actor parameter and split on export. mHC scale remains one FP32 tensor with three
entries, avoiding a lossy streaming reconstruction.

The actor builds only the text tower. HF export copies the source's frozen
`model.visual.*` tensors into separate shards, retains the multimodal config, zeros MTP
layer counts, and repairs the index. The original local source checkpoint must remain
accessible; export refuses to overwrite it. The native mbridge save option is rejected
because it would retain FP8 metadata while writing dequantized text weights.

## Four-layer SFT smoke

Prepare a smaller checkpoint on CPU, including KDA, DSA, one MoE layer, and the original
vision tower. This copies selected tensors and consumes disk space; the destination must
be new. It does not alter the source checkpoint or reduce hidden/expert sizes.

```bash
python examples/glm5_next/validate.py prepare-smoke "$GLM53_MODEL" "$GLM53_SMOKE"
python examples/glm5_next/validate.py audit "$GLM53_SMOKE" --check-runtime
```

`sft_smoke.yaml` uses the existing deterministic profile dataset for three optimizer
steps and an HF save. Fill its mandatory paths, node/GPU counts, and allocation from the
actual execution environment; keep CP=1 and do not enable VPP. The topology and private
environment must be checked before allocating hardware. On an already allocated local
execution environment, the entry point is:

```bash
python -m areal.infra.launcher.local examples/profile/train_sft_profile.py \
  --config examples/glm5_next/sft_smoke.yaml \
  actor.path="$GLM53_SMOKE" \
  allocation_mode="$GLM53_ALLOCATION" \
  cluster.n_nodes="$GLM53_NODES" \
  cluster.n_gpus_per_node="$GLM53_GPUS_PER_NODE" \
  cluster.fileroot="$GLM53_OUTPUT" \
  cluster.name_resolve.nfs_record_root="$GLM53_NAME_ROOT"
```

Check finite loss/grad norms, step completion, checkpoint index/config consistency, and
reload. The synthetic repeated sample is a plumbing test; its loss is not a quality
metric. Validate TP=1 first, then TP>1/SP and EP, checking replicated KDA gradients and
FP32 mHC optimizer state across ranks.

The earlier loss around 11.76 was measured with only the first four decoder layers
connected to the original final norm and language-model head. It is not a valid
full-model loss: the original checkpoint has 45 layers. A full-model no-update forward
on the same 512-token profile produced NLL 1.8667, 1.8936, and 1.8820 on three DP
replicas. A follow-up check found identical non-expert weight hashes across matching TP
shards but different logps on repeated forwards of the same replica. A controlled probe
changing only unfused MoE unpermute accumulation from BF16 to FP32 eliminated
repeat-to-repeat variation. Cross-replica differences remained, starting with small KDA
output differences before the first MoE layer. This diagnostic does not change the
production accumulator or establish HF parity.

## Full-model FP32 SFT baseline

[`sft_full_fp32.yaml`](sft_full_fp32.yaml) preserves the successful 45-layer run from
2026-10-26 (Slurm job 985824, code `fd3b290d2cfc8984abe11f868737479763ea80b8`). It uses
four nodes with eight GPUs each, TP4/DP8/EP32, PP1/CP1, full activation recompute, batch
size 24, and the deterministic 512-token profile. Model compute is BF16; gradients,
optimizer master weights, and both Adam moments are FP32. The optional BF16 optimizer
moments mode is explicitly disabled by this recipe's dtype settings.

| Step | Loss   | Grad norm | Successful update | Train step (s) |
| ---- | ------ | --------- | ----------------- | -------------- |
| 1    | 1.8964 | 21.450    | 1                 | 76.123         |
| 2    | 1.8147 | 20.211    | 1                 | 35.820         |
| 3    | 1.6599 | 21.599    | 1                 | 37.558         |

The job completed with exit code 0 in 379 seconds. Runtime assertions inspected both
optimizer leaves on all 32 ranks: all 64 records had CPU FP32 master weights, `exp_avg`,
and `exp_avg_sq`. Those three arrays occupied about 915.94 GiB per node, before
pinned-allocation rounding and other host overhead. The tested nodes had 1.5 TiB host
memory and GPUs reporting 143,771 MiB each. The launch gate required at least 1,100 GiB
available host memory per node. Budget for initialization as well as training: sampled
driver memory reached about 134 GiB during initialization, versus about 99 GiB after
training steps.

Use the runtime described above and provide paths for the original full checkpoint,
outputs, and a unique NFS name-resolution directory. On each of the four allocated
nodes, with its corresponding `NODE_RANK` and a shared rendezvous address/port, run:

```bash
export AREAL_SPMD_MODE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nnodes=4 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  examples/profile/train_sft_profile.py \
  --config examples/glm5_next/sft_full_fp32.yaml \
  actor.path="$GLM53_MODEL" \
  cluster.fileroot="$GLM53_OUTPUT" \
  cluster.name_resolve.nfs_record_root="$GLM53_NAME_ROOT"
```

Select network interfaces and cache directories for the actual cluster. The measured run
warmed KDA/DSA kernel caches before training; these times are not a cold-start
benchmark. Its external resource monitors timed out in `nvidia-smi` at shutdown; the
training steps and Slurm exit status were successful. A subsequent node check found no
remaining GPU processes and host memory returned to the prior baseline.

This is a three-step synthetic SFT validation, not a convergence or long-context quality
result. Checkpoint saving is disabled here; the save/reload evidence above comes from
the reduced model. It does not validate RL rollout, reward, SGLang logp parity,
multimodal training, or live weight transfer.

## Numerical and RL gates

1. Run `pytest -q -m slow tests/test_glm5_next_kernels.py` on CUDA. This checks packed
   kpool causality and sparse MLA forward/backward against a dense oracle.

   The gradient oracle uses FP32 leaves. An independent dense BF16/autograd baseline
   bounds maximum and RMS error to twice the baseline error, with an additional 1%
   relative RMS limit. This accounts for cancellation in BF16 probability/value-gradient
   products while bounding both the worst element and aggregate numerical error. Two
   random seeds exercise the packed NoPE case.

1. Perform a no-update HF → Megatron → HF roundtrip. Check tensor names, shapes,
   quantization exclusions/scales, mHC scale precision, and untouched vision tensors.
   FP8 requantization is not byte-exact; compare dequantized tensors and logps.

1. Capture HF, Megatron, and SGLang logps for identical token IDs and scoring masks,
   with sampling disabled and equal model weights. Save each as NPZ arrays `token_ids`,
   `loss_mask`, and `logprobs`. Compare each pair with an explicitly chosen tolerance:

   ```bash
   python examples/glm5_next/validate.py compare-logps reference.npz candidate.npz \
     --atol 0.05
   ```

   The tool rejects mismatched token IDs/masks, non-finite scored logps, and empty
   comparisons. It reports mean, p99, and maximum absolute differences and exits nonzero
   on failure. The example tolerance must be calibrated to the reference runtime and
   quantization mode; it is not evidence of numerical agreement. The current native HF
   5.16.1 comparison is a diagnostic, not a pass: AReaL and HF profile512 NLLs are
   1.8667 and 1.8871, while 57.0% of scored logps are within 0.05. Hidden-state error
   starts near 3e-5 at layer 0 and accumulates; this is being investigated before
   choosing a production tolerance.

1. Validate native MoE routing replay on the 42 sparse layers. DSA indexer replay is not
   implemented. Compare routes and logps separately before enabling R3 in RL.

1. Validate an actual distributed weight update and receiver cache refresh. The sender
   uses the GLM bridge for names, convolution splits, and mHC tensors, then applies the
   checkpoint's FP8 policy. Receiver loading is still a GPU validation gate, especially
   for absorbed MLA weights.

1. Complete these checks before production RL. The full-model SFT baseline above has
   already passed its separate three-step training check. No training job is submitted
   by these preparation or comparison commands.

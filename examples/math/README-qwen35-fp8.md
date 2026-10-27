# Qwen3.5 FP8 RL with Megatron-Bridge, SGLang, and AWEX

This opt-in path uses TransformerEngine FP8 GEMMs for Qwen3.5 Dense training, with BF16
model parameters and the configured optimizer state. SGLang serves a separate block-FP8
checkpoint. AWEX converts each updated training shard into the inference layout, then
transmits E4M3 weights and FP32 scales together. The vision encoder, embedding, output
head, norms, convolution and GDN B/A projections retain their original precision during
rollout. During training, MCore fuses GDN Q/K/V/Z/B/A into one input projection; this
projection uses BF16 through MCore's native per-module recipe. The other language
projections use the configured FP8 compute recipe. This is a mixed-precision path.

The implementation follows slime's separation of training precision and rollout
quantization, and its blockwise weight-update approach. References are pinned to
[slime's low-precision guide](https://github.com/THUDM/slime/blob/18868a79b0b083a093798a94ed0f8cb4f71b9c14/docs/en/advanced/low-precision.md)
and
[FP8 quantizer](https://github.com/THUDM/slime/blob/18868a79b0b083a093798a94ed0f8cb4f71b9c14/slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_fp8.py).

## Requirements and supported configuration

- A CUDA environment with TransformerEngine blockwise FP8 support, such as H100, H200,
  or Blackwell. Use the repository's Megatron-Bridge and AWEX dependencies.
- A BF16 Qwen3.5 checkpoint for `actor.path`, and a separately converted FP8 checkpoint
  for `sglang.model_path`.
- `actor.megatron.bridge_type=megatron-bridge` and `actor.weight_update_mode=awex`.
- `actor.megatron.fp8_config.recipe=blockwise`, `param=false`, `direct_convert=false`,
  and `actor.megatron.ddp.fp8_param_gather=false`.
- `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1` on training workers; the recipes set it.
- `sglang.fp8_gemm_runner_backend=triton` preserves the writable FP32 scale grids used
  by AWEX. Packed UE8M0 and swizzled scale/weight layouts are rejected.
- Quantized matrix shards and logical fused-projection boundaries must align to 128.
  Unsupported TP layouts fail during initialization. Training and inference TP sizes may
  differ when their block boundaries align.
- PP>1 requires uniform projection precision across layers. Layer-specific projection
  exclusions are rejected before AWEX metadata discovery.
- Full AWEX updates, including colocated and separated adapters. DTE delta transfer is
  rejected for FP8 payloads. The shipped recipes exercise colocation.
- MTP is disabled in the recipes. KV cache and recurrent-state precision remain separate
  settings; enabling FP8 weights does not enable FP8 recurrent state.

The existing BF16 defaults are unchanged. Checkpoint saving and recovery use the BF16
training model; rollout-only FP8 weights and scales are derived again from the current
training weights at each AWEX synchronization.

## Prepare the rollout checkpoint

Use local model directories. The destination must not already exist. Conversion reads
one source safetensors shard at a time and quantizes one projection at a time on the
GPU. The source checkpoint remains unchanged. MoE models are outside this initial scope.

```bash
export MODEL_PATH=/path/to/Qwen3.5-0.8B
export FP8_MODEL_PATH=/path/to/Qwen3.5-0.8B-AReaL-FP8
export FILEROOT=/path/to/experiments
python -m areal.tools.prepare_qwen35_fp8 "$MODEL_PATH" "$FP8_MODEL_PATH"
```

Do not use an arbitrary third-party FP8 checkpoint without checking its exclusion list
and scale format. AReaL validates the runtime weight/scale shapes and expected precision
before publishing AWEX metadata.

## Minimal GPU run

One GPU, Qwen3.5-0.8B, two GRPO steps:

```bash
bash examples/math/run_qwen35_fp8_smoke.sh
```

The script prepares the FP8 checkpoint if `config.json` is absent. Expected behavior is
an FP8 training initialization message, finite loss/gradient metrics, and completed AWEX
updates followed by successful rollout on the updated model. Two steps exercise a
repeated update, rather than only initial model loading.

## Dense end-to-end run

Set `MODEL_PATH` to a BF16 Qwen3.5 Dense directory and `FP8_MODEL_PATH` to its converted
rollout directory. The recipe uses one node with four GPUs, training DP=2 / TP=2, and
one inference instance with TP=4. A small Dense checkpoint can exercise resharding
before moving to a larger Dense model.

```bash
bash examples/math/run_qwen35_fp8_e2e.sh
```

This runs four GRPO steps. Extra positional arguments are forwarded as AReaL
configuration overrides. Set `TRIAL_NAME` to select a trial name; otherwise the scripts
generate a timestamped trial. Adjust scheduling specifications for the actual GPU
environment before submitting cluster jobs.

## Validation status

The unit tests and both GPU scripts have **not been run locally**. They require
execution in the target environment before treating this path as validated. No
convergence, throughput, or reward-parity claim is made.

```bash
pytest tests/test_qwen35_fp8.py tests/test_megatron_bridge_deterministic.py
```

Coverage includes provider propagation, the GDN BF16 recipe, exclusions, replicated KV
heads, write-through inference scale views, block alignment, scale metadata offsets, and
a CUDA quantization round trip with zero blocks and changed weights/scales. For
deployment acceptance, run Dense RL, repeat weight updates, and compare generated log
probabilities and reward trends against a BF16 baseline.

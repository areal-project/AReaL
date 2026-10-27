# Qwen3.5 Dense FP8 VLM RL

These examples train Qwen3.5-2B on Geometry3K images and questions with Megatron-Bridge,
SGLang, AWEX weight updates, and KPop filtering. The main recipe is adapted from
[the Qwen3.5-2B VLM recipe](../vlm/qwen3_5_2b_megatron_geometry3k_grpo.yaml). All launch
scripts reuse [the Geometry3K entrypoint](../vlm/geometry3k_grpo.py), which loads the
model processor and runs `VisionRLVRWorkflow` for both training and evaluation. The
workflow sends images to SGLang and retains pixel values, image grids, and multimodal
token types for training.

## Files and GPU layouts

| File                                             | Purpose                                                                                 |
| ------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `qwen3_5_2b_megatron_geometry3k_grpo.yaml`       | Main VLM recipe: 8 GPUs, training DP=2 / TP=2 on 4 GPUs, SGLang TP=4 on 4 separate GPUs |
| `qwen3_5_2b_megatron_geometry3k_grpo_smoke.yaml` | Single-GPU colocated variant with offload, a small batch, and two training steps        |
| `prepare_qwen35_fp8.py`                          | Convert a BF16 checkpoint into a separate FP8 rollout checkpoint                        |
| `run_qwen35_fp8.sh`                              | Prepare the rollout checkpoint if needed, then launch the main recipe                   |
| `run_qwen35_fp8_smoke.sh`                        | Launch the single-GPU VLM smoke recipe                                                  |
| `run_qwen35_fp8_e2e.sh`                          | Run four steps with the main 8-GPU layout and smaller batches                           |

The main recipe retains the reference's optimizer settings, including `lr=5e-7`, and
Geometry3K train/test splits. Prompts, including image tokens, are limited to 2048
tokens; generation is limited to 2048 more, fitting the 4096-token training microbatch
budget. It replaces vLLM with SGLang, uses FP8 rollout weights, and enables AWEX and
KPop. The single-GPU recipe inherits these settings and reduces the generation budget to
128 tokens. Short validation runs check the training path, not convergence.

## Precision and requirements

- Use a CUDA environment with TransformerEngine blockwise FP8 support, such as H100,
  H200, or Blackwell, and the repository's Megatron-Bridge, SGLang, and AWEX
  dependencies.
- `MODEL_PATH` must contain the **complete BF16 Qwen3.5-2B VLM checkpoint**, including
  vision weights and processor/tokenizer files. Do not use a text-only export.
- Training uses BF16 model parameters and TransformerEngine FP8 GEMMs in supported
  language projections. MCore's fused GDN Q/K/V/Z/B/A input projection uses BF16 through
  its native per-module recipe. The vision tower retains BF16 precision and is not
  frozen by this recipe. This is mixed-precision VLM training.
- `actor.megatron.bridge_type=megatron-bridge`, `actor.weight_update_mode=awex`,
  `fp8_config.recipe=blockwise`, `param=false`, `direct_convert=false`, and
  `ddp.fp8_param_gather=false` select the supported path.
- Training workers set `NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`.
- SGLang loads the separate FP8 checkpoint, with `enable_multimodal=true`,
  `skip_tokenizer_init=true`, and `fp8_gemm_runner_backend=triton`.
- AWEX converts each updated shard into the inference layout and sends E4M3 weights with
  FP32 block scales. Vision weights, embedding, output head, norms, convolution, and GDN
  B/A projections retain their original precision during rollout.
- Quantized shards and fused-projection boundaries must align to 128. Training and
  inference TP sizes may differ when their block boundaries align; unsupported layouts
  fail during initialization. Packed UE8M0 and swizzled scale/weight layouts are
  rejected.
- Full AWEX updates are used. DTE delta transfer is unsupported for FP8 payloads.
- MTP is disabled. The recipes retain the default unchunked LM-head path for image
  batches. KV cache and recurrent-state precision are separate settings; FP8 weights do
  not enable FP8 recurrent state.

Checkpoint saving and recovery use the BF16 training model and its configured optimizer
state. Rollout-only FP8 tensors are derived again from the current training weights on
each AWEX synchronization. MoE models are outside this example's scope.

The FP8 implementation follows slime's separation of training precision and rollout
quantization. See the pinned
[low-precision guide](https://github.com/THUDM/slime/blob/18868a79b0b083a093798a94ed0f8cb4f71b9c14/docs/en/advanced/low-precision.md)
and
[FP8 quantizer](https://github.com/THUDM/slime/blob/18868a79b0b083a093798a94ed0f8cb4f71b9c14/slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_fp8.py).

## Prepare and run

Run from the repository root in the target CUDA environment:

```bash
export MODEL_PATH=/path/to/Qwen3.5-2B
export FP8_MODEL_PATH=/path/to/Qwen3.5-2B-AReaL-FP8
export FILEROOT=/path/to/experiments

# Optional: prepare explicitly; the launch scripts also do this if config.json is absent.
python -m examples.fp8.prepare_qwen35_fp8 "$MODEL_PATH" "$FP8_MODEL_PATH"

# Main training recipe: one node, eight GPUs.
bash examples/fp8/run_qwen35_fp8.sh
```

The conversion destination must not already exist. Conversion reads one source
safetensors shard at a time and quantizes one projection at a time on the GPU. It copies
the processor/tokenizer assets and preserves the source checkpoint. If conversion was
interrupted, select a fresh destination before retrying. Do not substitute an arbitrary
third-party FP8 checkpoint without checking its exclusion list and scale format.

The launch scripts generate a timestamped trial name unless `TRIAL_NAME` is set. Extra
arguments are forwarded as configuration overrides. The recipes explicitly set
`scheduler.type=local` for the direct Python entrypoint. Adapt the scheduler and worker
specifications to the actual environment before submitting a cluster job. For example:

```bash
bash examples/fp8/run_qwen35_fp8.sh actor.rejection_sampling.upper=1.0
```

## GPU validation

Minimal test, one GPU with sufficient memory for Qwen3.5-2B and its vision tower:

```bash
bash examples/fp8/run_qwen35_fp8_smoke.sh
```

End-to-end test, eight GPUs with different training and inference TP sizes:

```bash
bash examples/fp8/run_qwen35_fp8_e2e.sh
```

Both commands use real Geometry3K images. Expected behavior is successful multimodal
rollout, finite loss and gradient metrics, completed optimizer steps, and successful
rollout after AWEX updates. The two-step smoke run exercises a subsequent weight update;
the four-step end-to-end run also exercises TP resharding. These short runs disable
periodic evaluation and saving. The main recipe retains both.

The GPU scripts have **not been run locally**. Run them in the target environment before
treating VLM forward/backward, FP8 weight synchronization, or the TP layouts as
validated. No convergence, throughput, or reward-parity claim is made. Existing focused
tests are:

```bash
pytest tests/test_qwen35_fp8.py tests/test_megatron_bridge_deterministic.py \
  tests/test_rejection_sampling.py tests/test_sglang_multimodal_config.py
```

## KPop settings

Both recipes enable token-level KPop:

```yaml
actor:
  ppo_n_minibatches: 1
  recompute_logprob: true
  use_decoupled_loss: true
  prox_logp_method: recompute
  rejection_sampling:
    level: token
    action: mask
    metric: binary_kl
    upper: 2.0
    lower: null
```

`upper` bounds the maximum of forward and reverse Bernoulli KL for each generated token.
It is not a probability-ratio cap or a reference-model KL coefficient. `2.0` follows the
existing KPop example and has not been tuned for Qwen3.5 FP8 VLM training. Smaller
values reject more tokens; keep `lower: null`.

The recipes retain `recompute`: KPop compares a cached training forward before
optimization with rollout logprobs. To compare the current training forward with rollout
instead, use `actor.prox_logp_method=reuse_train_logp` and
`actor.recompute_logprob=false`, keeping `actor.ppo_n_minibatches=1`. This changes PPO
ratio semantics and skips the extra proximal forward. AReaL still normalizes policy loss
by the original valid token count.

For threshold comparisons, keep other settings fixed and try `1.0` and `0.5` as stricter
candidates. Track `behave_filtered_ratio`, `rs_filtered_fraction`, the unfiltered
`logp_abs_diff`, reward, and gradient norms. Do not target a fixed rejection percentage
or use filtered-only KL statistics to conclude that the original mismatch disappeared.
Check weight versions, synchronization, and logprob settings if rejection suddenly
increases. KPop does not change the FP8 quantizer or AWEX synchronization.

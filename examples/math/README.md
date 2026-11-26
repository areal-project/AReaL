## Hyper-parameters for GSM8K Finetuning on Qwen2.5-1.5b-Instruct

`gsm8k_grpo.yaml` uses `6e-6`, retuned for the current FSDP FP32-master optimizer path.

### Current FP32-Master Recipe

A seed-1 run on 8 NVIDIA A800 GPUs completed the official 10-epoch schedule with the
following held-out evaluation results:

| lr     | weight decay | group size | best eval reward | final eval reward |
| ------ | ------------ | ---------- | ---------------- | ----------------- |
| 6.0E-6 | 0.017        | 4          | **0.78412**      | 0.77767           |

### Historical Pre-FP32-Master Sweep

The results below were collected with BF16 parameter and optimizer-state storage. They
are retained for reference and are not directly comparable with the current recipe.

| lr       | weight decay | group size | max task_reward |
| -------- | ------------ | ---------- | --------------- |
| 1.70E-05 | 0.017        | 4          | **0.79570**     |
| 1.30E-05 | 0.015        | 8          | 0.79355         |
| 1.50E-05 | 0.01         | 4          | 0.79043         |
| 1.50E-05 | 0.02         | 4          | 0.78984         |
| 1.00E-05 | 0.02         | 4          | 0.78311         |
| 1.00E-05 | 0.01         | 8          | 0.78066         |

#### Training Details

- Devices: 8 Nvidia H800 GPUs
- Optimizer: Adam
- LR Scheduler: Constant
- Gradient Clipping: 1.0
- Max_new_tokens: 1024
- Max_head_offpolicyness: 2
- Training Time: ~35 minutes (batchsize 4), ~65 minutes (batchsize 8)

## Score centering with SGLang or vLLM

`gsm8k_score_centering.py` uses the native single-turn `RLVRWorkflow` to carry the
sampler head IDs and probabilities into FSDP or Megatron training. Unlike `gsm8k_rl.py`,
it does not use the OpenAI agent transport, which does not carry these heads.

Start from the existing recipe with these overrides, and set model paths, resource
allocation and output directories for your environment:

```bash
python3 examples/math/gsm8k_score_centering.py \
  --config examples/math/gsm8k_grpo.yaml scheduler.type=local \
  actor.use_score_centering=true actor.recompute_logprob=false \
  actor.use_decoupled_loss=false actor.rejection_sampling=null \
  gconfig.score_centering_top_k=128 gconfig.top_k=-1 gconfig.top_p=1.0 \
  gconfig.temperature=1.0 actor.temperature=1.0
```

Enabling score centering selects REINFORCE with a detached head/tail correction,
replacing the clipped PPO surrogate. With `actor.rejection_sampling=null`, scores remain
unweighted. To enable token-level truncated importance sampling (TIS), use AReaL's
existing configuration:

```yaml
actor:
  use_score_centering: true
  recompute_logprob: false
  use_decoupled_loss: false
  rejection_sampling:
    level: token
    metric: ratio
    action: clamp
    upper: 2.0
    lower: null
```

TIS uses detached current-trainer/sampler probability ratios, capped at `upper`. The
correction centers the weighted scores; it is not multiplied by the sampled weight
again. Sequence-level rejection, masking, KL metrics and positive lower bounds are
unsupported. `upper` must be finite, greater than one and representable in FP32. The
existing default cap remains unchanged.

Ordinary FSDP training supports TP and Ulysses CP, including TP=2/CP=2. Configure the
actor allocation with the existing parallelism syntax, for example `fsdp:d1t2c2`, and
allocate four trainer GPUs plus the rollout resources. Vocabulary size must divide
evenly across TP ranks; model attention-head divisibility constraints also apply. FSDP
does not provide pipeline parallelism.

Megatron also supports TP, packed CP and PP with the same SC/TIS settings. Use the
existing actor allocation syntax, for example `megatron:d1t2c2p2` for eight trainer
GPUs, plus separate rollout resources. The correction is computed on the last pipeline
stage. See the [Megatron validation notes](score_centering_megatron.md) for tested
layouts and the corrected-verifier Countdown experiment.

The calculation uses PyTorch operations and checkpointed chunks. TP normalizes across
vocabulary shards without gathering full-vocabulary logits. CP partitions sampler heads
with the input tokens and differentiably gathers token outputs. Full local logits remain
live. No new dependency or inference runtime patch is required.

Support requires native SGLang or vLLM text rollout, temperature 1, and no sampling
truncation or penalties. SAPO/CISPO, decoupled PPO, tree training, OpenAI agent
workflows and distillation remain unsupported. Megatron additionally requires ordinary
packed logits: chunked LM-head loss, MTP training and differentiable entropy are
unsupported. Evaluation disables head collection. Other training paths retain their
existing behavior when score centering is disabled.

For vLLM, use AReaL's vLLM environment and set `rollout.backend=vllm:d1`,
`rollout._version=v1`, and `vllm.model` to the actor checkpoint in the command above.
Keep the same SC and TIS settings. Managed servers automatically raise `max_logprobs` to
fit the requested head width and use `raw_logprobs` with `generation_config=vllm`.
Requests disable inherited truncation and penalties so the returned probabilities
describe the sampling distribution. Manually started servers must use
`--max-logprobs 128 --logprobs-mode raw_logprobs --generation-config vllm` (increase 128
if requesting a wider head). SC is not supported by the vLLM chat/vision path or AReaL's
v2 inference bridge yet.

vLLM returns the sampled token together with its top-K head. When the sample is outside
that head, the adapter removes the extra sampled entry, including at tied probabilities.
It retains the original probabilities without renormalization and validates every
returned entry before training.

The vLLM live checks cover single-rank FSDP with SC and SC + TIS. Megatron with vLLM,
trainer TP/CP with vLLM, and vLLM server TP/PP have not been validated together here.

Tests cover dense-reference gradients and optimizer updates, vocabulary shards, CP
padding and inactive partitions, actor alignment and sampler metadata. Run:

```bash
python -m pytest tests/test_score_centering.py tests/test_score_centering_tis.py \
  tests/test_score_centering_vocab_parallel.py tests/test_score_centering_parallel.py \
  tests/test_score_centering_megatron.py tests/test_vllm_score_centering.py -q
```

The opt-in real-model tests require a local Qwen3 checkpoint and four CUDA devices:

```bash
AREAL_TEST_MODEL=/path/to/local/qwen3-checkpoint python -m pytest \
  tests/test_score_centering_parallel.py -k local_checkpoint -q
```

The Megatron checkpoint tests additionally cover PP and CP=4; the largest layout needs
eight CUDA devices. Set the same `AREAL_TEST_MODEL` and select
`tests/test_score_centering_megatron.py -k local_checkpoint` to run them.

The live vLLM check needs an SC-configured FP32 server (`--dtype float32`), the same
local checkpoint, and a separate CUDA device for training. Set `AREAL_TEST_MODEL` and
`AREAL_TEST_VLLM_URL`, and start the server with
`TRITON_F32_DEFAULT=ieee NVIDIA_TF32_OVERRIDE=0` to disable reduced-precision FP32 dot
products for this numerical check; then run
`tests/test_vllm_score_centering.py -k local_checkpoint` with pytest. It checks head
widths 1, 8 and 128 through native rollout and `RLVRWorkflow`, compares returned
probabilities and top-K membership with a teacher-forced FP32 dense forward, checks
SC/TIS gradients against a full-vocabulary surrogate, and performs two FSDP updates both
with and without TIS. The server stays fixed during these updates. The smoke uses
synthetic positive and negative rewards to exercise backward; it is not a learning-curve
experiment or a weight-synchronization test. The trainer uses BF16; the FP32 sampler is
used only for a strict cross-engine numerical oracle. Production sampling can use BF16,
where the trainer and sampler kernels can have different probabilities.

The earlier [Countdown reproduction notes](score_centering_countdown.md) describe plain
SC. Multi-node and NPU execution have not been validated. This feature makes no
performance claim.

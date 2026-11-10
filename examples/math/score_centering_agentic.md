# Score centering for multi-turn and text-only agents

Score centering (SC) can collect SGLang sampler heads through `MultiTurnWorkflow`,
`ArealOpenAI` Chat Completions and Responses, and AReaL's OpenAI proxy. Text tool calls
use the existing tool parser and agent loop. Both individual-turn and concatenated
training exports carry the original sampler probabilities. The FSDP and Megatron
training paths use the same SC calculation and optional token-level TIS.

## Configuration

Use the existing settings; no extra dependency or new YAML field is needed:

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

gconfig:
  score_centering_top_k: 128
  temperature: 1.0
  greedy: false
  top_p: 1.0
  top_k: -1
  frequency_penalty: 0.0
```

Set `actor.rejection_sampling=null` for SC without TIS. The `top_k` sampling option
remains untruncated; `score_centering_top_k` controls only how many sampler
probabilities are collected. Use the same head width for every training request.

The math entry point forwards `gconfig.score_centering_top_k` to its agent. The
multi-turn math example forwards it through `gconfig.to_openai_args_dict()`. Both
examples disable head collection in their evaluation configurations, which use a
different temperature. Apply the settings above to the existing math or multi-turn math
YAML and use its normal launcher and resource allocation.

For a custom text agent, forward the extension on **every** model request:

```python
response = await client.chat.completions.create(
    model="default",
    messages=messages,
    temperature=1.0,
    top_p=1.0,
    tools=tools,
    extra_body={"score_centering_top_k": 128},
)
```

Here `client` is `ArealOpenAI`, or the standard OpenAI client connected to AReaL's proxy
backed by native SGLang. The same extension works with `responses.create`.
`GenerationHyperparameters.to_openai_args_dict()` and the Responses/Agents conversion
helpers add it automatically. The standard OpenAI response schema does not change; AReaL
retains sampler heads internally for training.

## Turn and tool alignment

Each generated token retains its raw token ID, sampled log-probability, sampler head and
rollout weight version. Parsing a tool call or removing stop tokens from displayed text
does not change these training records. Tool results, feedback, prompt tokens and
inserted separators have zero loss mask and no sampler probability mass.

Individual export trains each response with its whole prompt masked. Concat export
preserves the parent's sampled token prefix and historical heads, then appends only the
new prompt gap and new sampled response. Retokenized or truncated parent prefixes,
missing heads and inconsistent head widths are rejected. Sequence padding cannot pad the
head-candidate dimension, including when combining separate rollout batches.

This support covers text inputs and ordinary unconstrained stochastic sampling at
temperature 1. Multimodal SC and grammar-constrained decoding are not supported. Tool
definitions may appear in prompts, and generated calls may be parsed and executed by the
agent; constraints that modify the sampling distribution are not supported. Existing
trainer restrictions, including tree-attention training and distillation, continue to
apply. This change does not add support to AReaL's v2 data proxy or to an external
inference provider that does not return native sampler heads.

## Regression checks

```bash
python -m pytest tests/test_score_centering_agentic.py \
  tests/experimental/openai/test_score_centering_proxy.py \
  tests/test_score_centering.py tests/test_score_centering_tis.py
```

The agentic tests cover both OpenAI APIs, individual/concat export, real text tool-call
parsing and tool replies, native retry turns, empty responses, repeated stop tokens,
streamed presentation, retry cleanup, proxy serialization and sampling validation. A
full-vocabulary oracle checks the SC/TIS gradient using exported historical heads.

## Multi-turn validation experiment

The following experiment exercises the math example's retry workflow with Qwen3-0.6B on
GSM8K. Each episode allows up to two attempts. The reported training accuracy is the
final episode's binary correctness, including successful retries.

Both arms use the same REINFORCE objective with token-level TIS capped at 2. The control
sets only the SC correction to zero; it is not the default clipped PPO objective. Both
collect the same head width, 128. Binary rewards are mapped to -1/+1 before
within-prompt mean subtraction.

A fixed Gaussian perturbation is created once from the initial weights:
`delta = 0.02 * initial_weights * Normal(0, 1)`. After each trainer update, the sampler
loads the current weights plus that same perturbation. Noise is seeded per parameter,
shared by tied embeddings/output weights and stored in BF16. The trainer and held-out
evaluation use unperturbed current weights. The arms use the same seed and perturbation,
but asynchronous sampling does not guarantee identical trajectories.

The configuration uses seed 0, 64 questions x 8 responses per update, at most 1024 new
tokens per turn, thinking disabled, and Megatron TP=CP=PP=1 with SGLang. Adam uses
learning rate `1.7e-5` and weight decay `0.017`. Each run targets 300 updates; clean
held-out evaluation uses 64 questions every 20 updates. These settings adapt the fixed
sampler-mismatch stress test to the math example.

![Multi-turn GSM8K training accuracy under fixed sampler weight noise](score_centering_agentic_multiturn_noise002.png)

The orange curve uses SC and TIS; the green curve uses TIS without the SC correction
(`uncentered` in the legend). SC completed 300 training updates. The control was still
running when the accompanying data snapshot was collected, with 260 updates recorded.
Its recorded training accuracy is zero from step 125 through step 259. Over the shared
window of steps 240-259, mean training accuracy is **34.3% with SC and 0% for the
control**.

SC also degrades late in this run: its final 20 training updates average **19.1%**, and
its final clean held-out accuracy is **21.9%**, below its initial **59.4%**. This
single-seed experiment supports retaining training signal longer under this specific
mismatch. It does not establish general superiority, prevent all degradation, or measure
tool-task learning. Text-tool support has separate unit and live-generation checks. No
throughput claim is made.

The [sanitized metric snapshot](score_centering_agentic_multiturn_results.json) contains
the setup and per-step values without account links or deployment paths. The learning
runs used the implementation before rebasing onto current main; regression checks on the
rebased PR are reported separately.

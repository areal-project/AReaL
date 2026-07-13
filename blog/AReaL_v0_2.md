<h1 align="center">
<em>AReaL</em> v0.2: Training a SOTA 7B LRM with 1.5x Throughput Improvement
</h1>

<p align="center" style="font-size: 0.8em; color: #666;">
Release Date: 2025-03-31
</p>

## Introduction

We are excited to release AReaL v0.2 (boba), featuring three major milestones:

- **SGLang Support**: With the addition of SGlang support and a series of engineering
  optimizations, AReaL v0.2 achieves a speed improvement of 1.5x over AReaL v0.1 on 7B
  models.
- **SOTA 7B Model**: AReaL's RL training becomes more stable and sample-efficient. We
  obtain a SOTA 7B model in mathematical reasoning, which achieves pass@1 score of 61.9
  on aime24 and 48.3 on aime25 respectively.
- **Competitive 32B Model**: We train a highly competitive 32B model at an extremely low
  cost, achieving results comparable to QwQ-32B using only 200 data samples.

|                                 **Model (7B)**                                 | **AIME 2024** | **AIME 2025** | **GPQA-Diamond** |
| :----------------------------------------------------------------------------: | :-----------: | :-----------: | :--------------: |
|                               R1-Distill-Qwen-7B                               |     55.0      |     39.7      |       47.1       |
|                                 Light-R1-7B-DS                                 |     56.7      |     44.9      |       40.9       |
|   [AReaL-boba-RL-7B 🤗](https://huggingface.co/inclusionAI/AReaL-boba-RL-7B)   |   **61.9**    |   **48.3**    |     **47.6**     |
|                                **Model (32B)**                                 | **AIME 2024** | **AIME 2025** | **GPQA-Diamond** |
|                              R1-Distill-Qwen-32B                               |     72.6      |     54.9      |       63.2       |
|                                    QwQ-32B                                     |     78.9      |     70.2      |       64.6       |
|                                Light-R1-32B-DS                                 |     76.2      |     67.8      |       63.5       |
| [AReaL-boba-SFT-32B 🤗](https://huggingface.co/inclusionAI/AReaL-boba-SFT-32B) |     78.8      |     62.1      |       60.1       |

*Table 1: The performance of AReaL-boba-RL-7B and AReaL-boba-SFT-32B. We obtain SOTA 7B
model using RL on math reasoning. Although our dataset primarily consists of math and
logic problems, we observed that RL training led to measurable improvements on the
challenging STEM benchmark GPQA. Additionally, We train a highly competitive 32B model
using only 200 data samples, replicating **QwQ-32B's** inference performance on AIME
2024.*

<span id="eval_detail"></span> We declare that the reported numbers are based on our
re-testing, with each number representing the average results of 32 sampling responses.
The evaluation code was available in the `evaluation/` folder. Please roll back to
[commit f55fe68](https://github.com/areal-project/AReaL/commit/f55fe685c661b0117d888fcae47d6f46e9a85087)
(Feb 4, 2026) or earlier.

For the baselines and the SFT model AReaL-boba-SFT-32B, we follow the recommended
configuration (temperature = 0.6, top_p = 0.95, suggested by DeepSeek) and use the
default R1-Distill-Qwen template for prompts. For the RL-trained models, we retaine the
same temperature as during RL rollout (1.0).

Notably, during GPQA testing, we found that all answers are "A." To address this bias,
we randomized the answer options.

### Training Speed Comparison

![throughput_comparison_with_v0.1.0.png](/assets/figures/thpt_comparison.png)

AReaL v0.2.0 features the following system optimizations:

- **Upgraded Generation Backend: vLLM 0.6.3 → SGLang v0.4.0**

The generation backend has been upgraded from vLLM 0.6.3 to SGLang v0.4.0, leveraging
its radix attention mechanism to significantly improve throughput in scenarios where
multiple responses are sampled from the same prompt. Additionally, SGLang automatically
flushes radix caches upon weight updates, ensuring correctness in on-policy
reinforcement learning (RL). We will continue tracking advancements in the community to
integrate further optimizations.

- **Optimized Training for Variable-Length Sequences & Large Batches**

To handle variable sequence lengths efficiently, we eliminate padding and instead pack
sequences into 1D tensors. A dynamic allocation algorithm (approximately) optimally
distributes sequences under a maximum token budget, balancing micro-batch sizes while
minimizing the number of micro-batches. This approach maximizes GPU memory utilization,
enabling efficient computation across a large batch of variable-length inputs.

- **High-Performance Data Transfer for 1K-GPU Scaling**

AReaL employs NCCL with GPU-Direct RDMA (GDRDMA) over InfiniBand/RoCE, enabling direct
GPU-to-GPU communication that bypasses costly CPU-mediated transfers and PCIe
bottlenecks. Compared to traditional Ethernet-based approaches, this reduces latency and
increases throughput, keeping generation-to-training data transfer overhead below 3
seconds even in a 1,000-GPU cluster.

## Training Recipe

### SOTA 7B model using RL on math reasoning

#### Base Model

We use
[R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) as
our foundation model.

#### Dataset Curation

Our training dataset
([AReaL-boba-106k](https://huggingface.co/datasets/inclusionAI/AReaL-RL-Data/blob/main/data/boba_106k_0319.jsonl))
combines resources from multiple open-source projects:

- [DeepScaleR](https://github.com/agentica-project/deepscaler)
- [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/)
- [Light-R1](https://github.com/Qihoo360/Light-R1)
- [DAPO](https://github.com/BytedTsinghua-SIA/DAPO)

We enhanced this with challenging problems from:

- [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) (AoPS/Olympiad
  subsets)
- [ZebraLogic](https://hf-mirror.com/datasets/WildEval/ZebraLogic)

To maintain an appropriate difficulty level, we filtered out overly simple questions.
Specifically, We generate 8 solutions per question using DeepSeek-R1-Distill-Qwen-7B and
filter out questions where all solutions were correct.

#### Reward Function

We adopt a sparse sequence-level reward mechanism. The model is instructed to enclose
the final answer within \\boxed{}, and the boxed answer is then verified. Correct
responses receive a reward of +5, while incorrect ones are penalized with -5.

Notably, we observe that the KL reward can impair performance, particularly in long
chain-of-thought training, so we set it to zero.

#### RL Algorithm

We employ Proximal Policy Optimization (PPO) as our training algorithm. We remove the
critic model to save compute. We set both the discount factor γ and the GAE parameter λ
to 1. Such practices are also adopted by the
[Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) project.

#### Token-Level Loss Normalization

Averaging the loss in the sequence level can underweight the overall contribution of
longer texts. To address this, we normalize the loss at the token level, this practice
is also highlighted in [DAPO](https://github.com/BytedTsinghua-SIA/DAPO).

#### Rollout Strategy

During the rollout phase, we sample 512 questions per batch, and the LLM generates 16
responses per question—resulting in a total batch size of 8,192. To minimize output
truncation, we set the maximum generation length to 27K tokens. In our experiment, the
truncation rate remained below 5%.

#### Advantage Normalization

During the training phase, we compute advantages using GAE and normalize them across all
generated tokens.

#### Key Hyperparameters

|  **Parameter**  | **Value** |
| :-------------: | :-------: |
| PPO Minibatches |     4     |
|  Learning Rate  |   2e-5    |
|     Adam ε      |   1e-5    |

This configuration balances convergence speed with training stability, avoiding collapse
risks from higher learning rates or smaller ε values.

### Approaching QwQ-32B's performance using only 200 data samples

At the 32B model size, we further refine the training data and released
[AReaL-boba-SFT-200](https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/blob/main/AReaL-boba-SFT-200.jsonl),
a high-quality dataset with only **200 data points**. Accompanied by relevant training
scripts, we replicated **QwQ-32B's** inference performance on **AIME 2024** via
Supervised Fine-Tuning (SFT).

### Evaluation Best Practices

During evaluation, we use vLLM v0.6.3 as the generation framework. We identified several
settings that influence evaluation performance, particularly for long-context
generation. We recommend manually configuring the following options:

```bash
enforce_eager=True
no_enable_chunked_prefill=True
disable_custom_all_reduce=True
disable_sliding_window=True
```

Following the practice of DeepSeek models, we incorporate a directive in the prompt:
"Please reason step by step, and enclose your final answer in \\boxed{}." To encourage
long context reasoning, we also enforce that the model begins each response with "\\n"
at the start of its output.

To ensure reliable pass@1 estimation, we:

- Sample 32 answers per problem
- Use temperature=0.6 and top_p=0.95 for SFT models
- Maintain training temperature (1.0) for RL models

## Conclusion & Future Work

Our results demonstrate that **high-quality data is equally critical as algorithmic
innovations**. When conducting RL training on a powerful base model, we require more
challenging problems to facilitate learning. Therefore, we integrate resources from
multiple recent open-source projects and filter problems by difficulty. A
straightforward strategy for data filtering involves removing problems that the base
model consistently solves correctly across multiple sampling attempts, as they no longer
contribute to improving the model's performance.

AReaL delivers stable and fast training with cutting-edge model performances. Since
initial release, we've continuously improved system efficiency, training stability, and
accessibility. All aforementioned techniques have been implemented in AReaL, with
[**reproducible configurations**](/examples/configs/) for various model sizes and
different hardware setups.

Looking ahead, the AReaL team will:

- Further optimize the RL training throughput
- Introduce new algorithmic features
- Continue open-sourcing training data
- Expand to broader reasoning tasks

We believe these contributions lower the barrier for high-quality RL training while
pushing the boundaries of reasoning capabilities. The project will continue evolving -
we welcome community feedback and collaboration to drive further progress in this
exciting field.

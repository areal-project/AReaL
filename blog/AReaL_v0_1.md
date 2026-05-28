<h1 align="center">
<em>AReaL</em> v0.1: Reliable RL Training of 1.5B and 7B LLMs on Math Reasoning Tasks
</h1>

<p align="center" style="font-size: 0.8em; color: #666;">
Release Date: 2025-02-24
</p>

Our initial release AReaL v0.1 includes reproducible experiments based on our AReaL
system for 1.5B and 7B LRMs, validated across diverse computational budgets. We
demonstrate that with AReaL, users can:

1. **Reliably train a 1.5B distilled model** with RL to **surpass o1-Preview on math
   reasoning within 40 hours** 🚀
1. **Reliably perform an R1-Zero-style experiment with a 7B model** by running RL
   training on Qwen2.5-7B, and **observe emergent thinking tokens and continuous model
   improvement on math reasoning**

______________________________________________________________________

## Training a 1.5B LRM from the Distilled Model

Starting with the
[`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
model, we follow DeepScaleR to iteratively increase the output context length from 8K to
16K and eventually 24K. The training reward continuously increases during RL training.
We observe that the response length first **shrinks in the 8K training stage**, then
**increases in the 16K and 24K training stages**.

Our experiments are conducted on 16 nodes, each equipped with 8 H800 GPUs. The results,
along with the associated training curves, are presented below.

![16nodes_reward_length.png](/assets/figures/distill_1.5b_24k_curve.png)

*Figure 1. Training rewards and response lengths during RL training. The base model is
DeepSeek-R1-Distill-Qwen-1.5B. Curves are averaged with a window size of 25.*

These three stages progressively enhanced the performance of the base model, as
demonstrated below:

| Model                         | MATH500 | AIME 2024 | AMC 2023 | Download                                                                        |
| ----------------------------- | ------- | --------- | -------- | ------------------------------------------------------------------------------- |
| o1-Preview                    | 81.4    | 40.0      | -        | -                                                                               |
| DeepSeek-R1-Distill-Qwen-1.5B | 82.8    | 28.8      | 62.9     | -                                                                               |
| DeepScaleR (Official)         | 87.8    | 43.1      | 73.6     | -                                                                               |
| AReaL Stage 1: 8K (Ours)      | 85.7    | 33.2      | 74.7     | [🤗 HuggingFace](https://huggingface.co/inclusionAI/AReaL-1.5B-Preview-Stage-1) |
| AReaL Stage 2: 16K (Ours)     | 87.4    | 34.2      | 79.6     | [🤗 HuggingFace](https://huggingface.co/inclusionAI/AReaL-1.5B-Preview-Stage-2) |
| AReaL Stage 3: 24K (Ours)     | 88.0    | 40.2      | 81.2     | [🤗 HuggingFace](https://huggingface.co/inclusionAI/AReaL-1.5B-Preview-Stage-3) |

*Table 1. Evaluation on competition-level mathematics benchmarks including AIME 2024,
AMC 2023, and MATH-500. Results report Pass@1 accuracy averaged over 32 samples per
problem at temperature 0.6.*

### RL Training Details

Our RL training process incorporates insights from
[DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1),
[DeepScaleR](https://github.com/agentica-project/deepscaler), and
[ReaLHF](https://github.com/openpsi-project/ReaLHF). Key components include:

- **Base Model**:
  [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- **Dataset**: 40k high-quality mathematical reasoning tasks from
  [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset)
- **Reward Design**: +5 for correct responses, -5 for incorrect responses (evaluated
  using Sympy following
  [Qwen's protocol](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation))
- **RL Algorithm**: Simplified PPO variant without the critic network
- **Key Hyperparameters**:
  - Discount factor (γ) and GAE parameter (λ) both set to 1 (following
    [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero))
  - Small KL reward coefficient
- **Iterative Context Lengthening**: Three-stage training (8K→16K→24K) following
  [DeepScaleR](https://github.com/agentica-project/deepscaler)
- **Batch Size**: 1024 questions per step with 8 rollouts per question, which is
  **larger** than prior works

### AReaL with Various Computational Resources

AReaL adapts efficiently to different computational budgets. Increased resources
accelerate RL training, significantly boosting research progress. We provide detailed
hardware requirements and environment setup guides for different configurations in
[our tutorials](/examples/README.md).

![hours.png](/assets/figures/1.5b_time_n1n4n16.png)

*Figure 2. Total RL training time for 10 epochs across different resource
configurations.*

## Qwen2.5-7B-Zero RL Training

We conduct RL training starting from Qwen2.5-7B using DeepSeek-R1-Zero style training.
Initial results using data from
[Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) show both
rewards and response lengths gradually increasing during training. This **simultaneous
growth of response lengths and rewards** suggests **emergent deep thinking
capabilities** in solving complex reasoning problems.

![7b_zero_training_curve.png](/assets/figures/7b_zero_training_curve.png) *Figure 3.
Qwen2.5-7B-Zero RL training curve*

Evaluation of intermediate checkpoints on MATH500 and AIME24 datasets shows continuous
improvement in both accuracy and response length:

![7b_zero_eval_acc.png](/assets/figures/7b_zero_eval_acc.png) *Figure 4. Test accuracy
and response length on MATH500 and AIME24 datasets*

Additional experiments on the
[DeepScaleR](https://github.com/agentica-project/deepscaler) dataset show similar
trends:

| Model                                  | MATH-500 | AIME 2024 | AMC 2023 |
| -------------------------------------- | -------- | --------- | -------- |
| Qwen2.5-7B                             | 34.0     | 2.0       | 17.8     |
| Open-Reasoner-Zero-7B (Step 150)       | ~77      | ~15       | -        |
| Qwen2.5-7B-Zero (Step 150, DeepScaleR) | 75.9     | 13.9      | 56.1     |
| Qwen2.5-7B-Zero (Step 150, ORZ)        | 77.3     | 13.9      | 57.1     |
| Qwen2.5-7B-Zero (Step 200, ORZ)        | 78.0     | 14.5      | 58.7     |

*Table 2. Evaluation on AIME 2024, AMC 2023, and MATH-500 (Pass@1 accuracy averaged over
32 samples at temperature 1.0).*

## Open Research Questions on Algorithm and Model

Based on AReaL, our team is also actively exploring the research frontier of LRMs. We
highlight some of the major research topics we are working on.

- **Effective Training Datasets for LRMs** How do we design training tasks that enable
  LRMs to progressively evolve sophisticated reasoning skills via RL?
  - For more advanced distilled models, such as DeepSeek-R1-Distill-Qwen-7B and
    DeepSeek-R1-Distill-Qwen-32B, what kind of task distribution could drive further
    enhancements?
  - How to properly curate the distillation dataset for cold-start to achieve the best
    RL results?
- **Boosting Reasoning with Auxiliary Training Signals.** Can auxiliary signals - such
  as Process Reward Models (PRMs) and expert-curated labels - be adopted in RL training
  to refine the reasoning processes of LRMs?
- **Adaptive Response Lengths: Quality Over Quantity.** How can RL training schemes
  teach LRMs to dynamically adjust response complexity based on task difficulty? The
  goal is to eliminate redundant "over-reasoning" for simpler inputs while preserving
  rigor for challenging problems.
- **Sample Efficient RL for LRMs** In addition to system-level acceleration, can we
  further speed up the LRM training process with a more sample-efficient advanced RL
  algorithm? Can we get the critic back to reduce gradient variance without compromising
  the computation efficiency? How to better encourage exploration?

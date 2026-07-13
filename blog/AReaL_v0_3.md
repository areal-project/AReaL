<h1 align="center">
<em>AReaL</em> v0.3: SOTA Coding Models with 2.77x Faster Asynchronous RL Training
</h1>

## Introduction

We now release AReaL v0.3, featuring three major milestones:

- **A fully asynchronous RL training pipeline with system and RL algorithm co-design**,
  achieving over 2.77x speedup without any performance drop
- **SOTA coding models**, i.e., a 14B model with a **69.1 score on LCB-v5**.
  Reproducible results with fully open-sourced datasets are also provided
- Experimental support for **multi-turn agentic RL training**

### Performance Results

|                                   **Model (8B)**                                    | **LiveCodeBench v5** (2024.10-2025.2) | **Codeforce**  | **CodeContests** |
| :---------------------------------------------------------------------------------: | :-----------------------------------: | :------------: | :--------------: |
|                                      Qwen3-8B                                       |                 58.8                  |   1879/96.7%   |       31.4       |
|                              DeepSeek-R1-0528-Qwen3-8B                              |                 58.4                  |   1945/97.3%   |       31.0       |
| [🤗 AReaL-boba²-8B-Open](https://huggingface.co/inclusionAI/AReaL-boba-2-8B-subset) |                 62.0                  |   1933/97.2%   |     **41.4**     |
|       [🤗 AReaL-boba²-8B](https://huggingface.co/inclusionAI/AReaL-boba-2-8B)       |               **63.0**                | **1962/97.5%** |       40.8       |

|                                    **Model (14B)**                                    | **LiveCodeBench v5** (2024.10-2025.2) | **Codeforce**  | **CodeContests** |
| :-----------------------------------------------------------------------------------: | :-----------------------------------: | :------------: | :--------------: |
|                                       Qwen3-14B                                       |                 65.4                  |   1978/97.7%   |       38.3       |
|                                 DeepCoder-14B-Preview                                 |                 60.6                  |   1936/95.3%   |       40.1       |
| [🤗 AReaL-boba²-14B-Open](https://huggingface.co/inclusionAI/AReaL-boba-2-14B-subset) |                 67.3                  |   1990/97.8%   |     **46.2**     |
|       [🤗 AReal-boba²-14B](https://huggingface.co/inclusionAI/AReaL-boba-2-14B)       |               **69.1**                | **2044/98.2%** |       46.1       |

|    **Larger Models**    | **LiveCodeBench v5** (2024.10-2025.2) | **Codeforce** | **CodeContests** |
| :---------------------: | :-----------------------------------: | :-----------: | :--------------: |
|       Qwen3-235B        |                 70.7                  |     2056      |        -         |
|       DeepSeek-R1       |                 64.3                  |     2029      |        -         |
| OpenAI-o3-mini (Medium) |                 66.3                  |     2036      |        -         |

*Table 1: Coding Task Performance Comparison. AReaL-boba²-8B/14B-Open denotes training
results on open-sourced data. AReaL-boba²-8B/14B models are trained with an additional
small amount of internal data and achieve SOTA performance on LiveCodeBench, Codeforce &
CodeContests.*

To access our latest models and training data, please visit this
[Huggingface Link](https://huggingface.co/collections/inclusionAI/areal-boba-2-683f0e819ccb7bb2e1b2f2d5).

## Motivation for Asynchronous RL System

### Inference devices are underutilized

During the synchronous RL training process, the generation step must wait until the
finish of the longest output within a batch. Due to the varying output lengths for LRM,
a synchronous RL system suffers from severe training inefficiency.

Some works
([DeepCoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51),
[Intellect](https://www.primeintellect.ai/blog/intellect-2)) use outputs generated from
a previous model version to update the current model. For the best performances, the
model version used for rollout generation is limited to only one or two steps older.
However, all these systems still follow a batched generation setting, the issue of
system inefficiency during the generation phase still remains unaddressed.

![](/assets/figures/sync_one_step_gen.png)

*Fig.1. Left: Execution timeline of a synchronous RL training. Right: Execution timeline
of one-step overlap RL system.*

### Scalability is poor in synchronous RL systems

Synchronous systems distribute generation across all devices, reducing the per-GPU
decoding batch size. This pushes the decoding process into a memory-IO-bound regime
where additional devices fail to improve throughput.

![](/assets/figures/gen_scaling_trend.png)

*Fig2. Left: Strong scaling of batched generation throughput for a 1.5B LRM. Right:
Generation becomes memory-IO bound as GPU count increases.*

## Asynchronous AReaL Overview

We design a system that fully decouples generation and training across separate GPU
clusters. This system should be hardware-efficient, scalable, and equipped with the
flexibility for a customized RL workflow. We implement these principles in AReaL. Fig.3
presents the architecture and data flow of AREAL. The system comprises 4 core
components:

![](/assets/figures/arch.png)

*Fig.3 The architecture featuring asynchronous generation and training components.*

### Core Components

- **Interruptible Rollout Worker** handles two types of requests: (1) The *generate*
  request generates responses given prompts. (2) The *update_weights* request interrupts
  all ongoing generations and loads parameters of new versions. Upon the interruption,
  the rollout workers discard KV caches computed by old weights, and re-compute them
  using the new weights. Afterwards, the rollout workers continue to decode the
  unfinished sequences until the next interruption or termination. We emphasize that
  such interruptions and in-flight weight updates would result in trajectories composed
  of segments produced by different model versions. This introduces a novel algorithmic
  challenge, which will be addressed in the next section.

- **Reward Service** evaluates the accuracy of the responses generated by the model. For
  example, in the coding task, this service extracts the code and executes unit tests to
  verify its accuracy.

- **Trainer Workers** continuously sample from the replay buffer, accumulating data
  until reaching the configured training batch size. They then perform PPO updates and
  store the resulting parameters in distributed storage. To ensure data freshness, data
  from the replay buffer is used only once.

- **Rollout Controller** serves as a critical bridge between the rollout workers, reward
  service, and the model workers. During the training process, it reads data from the
  dataset and invokes the rollout worker's *generate* request. The received response is
  then sent to the reward service to obtain the reward. The trajectory, along with the
  reward, is stored in the replay buffer, waiting to be trained by the model worker.
  After the model worker updates the parameters, the controller calls the rollout
  worker's *update_weight*. We illustrate the generation and training management in
  Fig.4. This asynchronous pipeline ensures continuous full utilization of both
  generation and training resources.

![](/assets/figures/async_timeline.png)

*Fig 4. Execution timeline of our fully asynchronous RL system.*

## Algorithmic Challenges and Solutions for Asynchronous RL

### Challenges

While the asynchronous system design offers significant acceleration through improved
device utilization, it introduces several technical challenges that require algorithmic
considerations.

- **Challenge 1: Data Staleness** - Due to the asynchronous nature of AREAL, each
  training batch contains data from multiple prior policy versions. Data staleness would
  lead to a distribution gap between the training data and the latest model. In
  asynchronous RL training for LRMs, this issue could be even more severe for long
  trajectories due to extended decoding time.

- **Challenge 2: Inconsistent Policy Versions** - As shown in Fig.4, a single generated
  trajectory may involve segments produced by different policy versions. This
  inconsistency fundamentally violates the formulation of standard PPO that assumes all
  actions being generated by a single policy.

### Solutions

To overcome these two challenges, we propose two solutions:

- **Solution 1: Staleness-Aware Training** - To avoid negative impact of the data
  staleness issue, we constrain the version discrepancy between the policy version of
  generated trajectories and the training policy. We introduce a hyperparameter η
  representing the maximum permitted staleness. When η = 0, the system degenerates to
  the synchronous RL setting where generation exactly matches training batches. When η =
  1, the system recovers to the previous one-step overlap methods.

- **Solution 2: Decoupled PPO Objective** - We apply a decoupled PPO objective that
  disentangles the behavior policy and the proximal policy. The behavior policy
  represents the policy used for sampling trajectories and the proxy policy is a
  proximal policy serving as a recent target to regularize the update of online policy.

![](/assets/figures/decoupled_ppo_obj.png)

## Validating Asynchronous AReaL

We first evaluate Asynchronous AReaL on a math task to validate our system and
algorithmic choices. Building on our previous release, AReaL-boba, we adopt
R1-Distill-Qwen as the base model and
[AReaL-boba-106k](https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/blob/main/AReaL-boba-106k.jsonl)
as the training dataset.

### End-to-End Performance Comparison

We compared synchronous and asynchronous training on 1.5B and 7B models. Under equal
resource constraints and training steps, the asynchronous system was over twice as fast
as the synchronous one. Evaluations on AIME24 confirmed that this acceleration does not
compromise performance.

|  **Model (1.5B)**  | **AIME24** | **# Nodes** | **PPO Steps** | **Training Hours** |
| :----------------: | :--------: | :---------: | :-----------: | :----------------: |
|     basemodel      |    29.3    |      -      |       -       |         -          |
| Synchronous AReaL  |    42.0    |     16      |     1000      |        41.0        |
| Asynchronous AReaL |    42.2    |     16      |     1000      |      **14.8**      |

|   **Model (7B)**   | **AIME24** | **# Nodes** | **PPO Steps** | **Training Hours** |
| :----------------: | :--------: | :---------: | :-----------: | :----------------: |
|     basemodel      |    54.3    |      -      |       -       |         -          |
| Synchronous AReaL  |    63.0    |     24      |     1000      |        57.7        |
| Asynchronous AReaL |    63.1    |     24      |     1000      |      **25.4**      |

*Table 2: End-to-end comparison between asynchronous AReaL and synchronous AReaL.
Following last release AReaL-boba, we adopt R1-Distill-Qwen as the basemodel and
[AReaL-boba-106k](https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/blob/main/AReaL-boba-106k.jsonl)
as the training data. Each performance number represents the average results of 32
sampling responses.*

### Ablation Study

#### System Ablations

We ablate interruptible generation and present the resulting generation throughput in
Fig.5. Without interruptible generation, the controller must wait for the longest
response. In particular, interruptible generation leads to a 12% and 17% throughput
increase for 1.5B and 7B models respectively on 4 nodes.

![](/assets/figures/interrupt_gen_ablation.png)

*Fig.5 Ablation study of interruptible generation.*

#### Algorithm Ablations

We vary the maximum allowed staleness η and compare configurations with and without the
decoupled PPO objective. Fig.6 shows the learning curves after 1000 PPO updates,
demonstrating that naive PPO fails to match the performance of the synchronous RL oracle
(i.e., the performance when η = 0). Even slight staleness can significantly degrade
final performance due to the improper clipping center and policy changes during
interruptible generation. Furthermore, increasing data staleness consistently degrades
learning performance.

![](/assets/figures/algo_ablation.png)

*Fig.6 Ablation Study on Decoupled PPO Objective with DeepSeek-R1-Distill-Qwen-1.5B.
Left: Learning curves with naive PPO. Right: Learning curves with decoupled PPO
objective.*

|  η  | **AIME 24** |                       | **AIME 25** |                       |
| :-: | :---------: | :-------------------: | :---------: | :-------------------: |
|     |  naive PPO  | + Decoupled Objective |  naive PPO  | + Decoupled Objective |
|  0  |    42.0     |                       |    32.9     |                       |
|  1  |    41.8     |         42.1          |    30.7     |         31.9          |
|  2  |    40.0     |         41.8          |    32.1     |         32.5          |
|  4  |    23.3     |         42.2          |    23.1     |         32.0          |
|  8  |    35.7     |         41.0          |    27.8     |         31.1          |
| 16  |    35.8     |         38.7          |    26.2     |         32.5          |
|  ∞  |    34.0     |         36.9          |    26.9     |         29.9          |

*Table 3: Evaluation scores when varying data staleness, comparing performance with and
without the decoupled objective.*

```{note}
In practice, the effect of staleness could depend on various factors, such as learning rate, batch size, the number of mini batches, generation length, and also the RL algorithm. Here we aim to illustrate the impact of staleness on the training performance of a synchronous RL system, and show that stalenss control and decoupled PPO could well mitigate the negative impact of staleness. Here we follow a large batch size setting with `batch_size=512`, `n_rollouts=16`, `n_mini_batch=4`, `max_new_tokens=8192`, and `lr=2e-5`. The staleness is defined as the gap of model versions, where the model version increases by 1 when a batch of `batch_size` X `n_rollouts` samples are trained on.
```

However, the decoupled PPO objective substantially improves training stability when
handling stale data. Notably, even with the decoupled objective, unbounded staleness
(maximum staleness → ∞) still results in inferior performance compared to the
zero-staleness oracle. When properly constrained, moderate staleness (e.g., η ≤ 4) has
minimal impact on final performance while significantly accelerating training through
the asynchronous pipeline, as demonstrated in Tab.3 and Fig.7.

![](/assets/figures/staleness_throughput.png)

*Fig.7 The relationship between η and training throughput. Larger η leads to higher
throughput.*

## Training Recipe with Asynchronous AReaL

### Dataset Curation

We train the 7B and 14B model using open-source data from
[DeepCoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51).
After deduplication and quality filtering, approximately 7,600 coding problems remained.
Additionally, we have around 10,000 internal data entries that can further improve the
model's performance. We are actively working on open-sourcing this portion of the data.

### Rollout Strategy

During the rollout phase, the LLM generates 16 responses per question. To minimize
output truncation, we set the maximum generation length to 27K tokens.

### Filtering Strategy

In the design of our asynchronous system, we can naturally incorporate strategies to
filter some rollouts. Specifically, we exclude the questions where all 16 answers are
either entirely correct or entirely incorrect.

### Key Hyperparameters

|      **Parameter**       | **Value** |
| :----------------------: | :-------: |
|     PPO Minibatches      |     1     |
|      Learning Rate       |   2e-5    |
|          Adam ε          |   1e-5    |
|        Batch Size        |   2,048   |
| max_head_offpolicyness η |    16     |
|     success_rate_ub      |   0.95    |
|     success_rate_lb      |   0.05    |

## Resources

- [boba² Code dataset](https://huggingface.co/datasets/inclusionAI/AReaL-boba-2-RL-Code)
- **Reproduce boba² Code Models**
  - 🤗 **Model weights**: [8B-code](https://huggingface.co/inclusionAI/AReaL-boba-2-8B),
    [14B-code](https://huggingface.co/inclusionAI/AReaL-boba-2-14B),
    [8B-code-open](https://huggingface.co/inclusionAI/AReaL-boba-2-8B-subset),
    [14B-code-open](https://huggingface.co/inclusionAI/AReaL-boba-2-14B-subset)
  - [Evaluation Guide](https://areal-project.github.io/AReaL/en/tutorial/eval.html)
  - [Training configs](https://github.com/areal-project/AReaL/tree/main/examples/configs/v0.3-qwen3-code)
    and instructions (`reference/reproduce.html`, revert to a version before v0.5.1 to
    see the reproduction guide)

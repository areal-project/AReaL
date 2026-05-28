# Key Milestones

This document summarizes the major achievements and milestones across AReaL's
development history, highlighting the key contributions of each version release.

## AReaL-lite (July 2025): Simplifying RL for Everyone

*📖
[Step-by-step Tutorial](https://areal-project.github.io/AReaL/en/tutorial/gsm8k_grpo.html)*

AReaL-lite represents a fundamental rethinking of how researchers interact with
reinforcement learning systems. Born from the recognition that AReaL's system-first
architecture created barriers for AI researchers, this lightweight version prioritizes
algorithm development over infrastructure complexity.

AReaL-lite achieves 90% of the original system's functionality while reducing the
codebase by 80%. Researchers can now implement complete RL workflows within a single
file, leveraging intuitive PyTorch-centric APIs that feel natural to the ML community.
The architecture follows familiar SPMD patterns while providing native support for
asynchronous training and multi-turn agentic workflows.

Perhaps most importantly, AReaL-lite bridges the gap between research experimentation
and production deployment. Its clean abstractions integrate seamlessly with existing ML
tools while maintaining the performance characteristics that made AReaL successful in
the first place.

> **Architecture Evolution**: The releases below (v0.1-v0.3) were built on the legacy
> `realhf` codebase. AReaL has now fully transitioned to the new `areal` architecture,
> with AReaL-lite serving as both a standalone system and the foundation for future
> development.

## AReaL v0.3 (August 2025): Breaking the Speed Barrier

*📖
[Full Blog Post](https://github.com/areal-project/AReaL/blob/main/blog/AReaL_v0_3.md)*

Version 0.3 marked AReaL's boldest architectural leap: completely decoupling generation
from training across separate GPU clusters. This wasn't just an incremental
improvement—it fundamentally reimagined how RL systems could scale, achieving a
remarkable **2.77x speedup** while maintaining training stability.

The breakthrough came from solving two critical challenges that had plagued asynchronous
RL: data staleness and inconsistent policy versions during generation. Through
innovative staleness-aware training controls and a novel decoupled PPO objective, v0.3
turned these theoretical obstacles into practical advantages. The system now handles
weight updates mid-generation without losing progress, enabling true continuous training
at scale.

The results spoke for themselves: AReaL's 14B model achieved a **69.1 score on
LiveCodeBench v5**, establishing new state-of-the-art performance in coding tasks.
Beyond raw performance, v0.3 introduced experimental support for multi-turn agentic
workflows, opening new possibilities for complex reasoning applications.

| Model           | LiveCodeBench v5 | Codeforce  | CodeContests |
| --------------- | ---------------- | ---------- | ------------ |
| AReaL-boba²-8B  | **63.0**         | 1962/97.5% | 40.8         |
| AReaL-boba²-14B | **69.1**         | 2044/98.2% | 46.1         |

## AReaL v0.2 (March 2025): Engineering Excellence Meets SOTA Performance

*📖
[Full Blog Post](https://github.com/areal-project/AReaL/blob/main/blog/AReaL_v0_2.md)*

The second major release focused on transforming AReaL from a promising research
prototype into a production-ready system. At the heart of v0.2 was the transition from
vLLM to SGLang v0.4.0, bringing sophisticated radix attention mechanisms that
dramatically improved throughput for multi-response sampling scenarios—exactly what RL
training demands.

But the real breakthrough came from recognizing that **data quality trumps algorithmic
complexity**. Our team curated a focused dataset of 106k high-quality problems,
strategically filtering out cases where base models already achieved perfect accuracy.
This surgical approach to data curation, combined with systematic engineering
optimizations, delivered remarkable results.

AReaL v0.2 produced the **best-performing 7B mathematical reasoning model** of its time,
achieving 61.9 pass@1 on AIME 2024 and 48.3 on AIME 2025. Perhaps even more
impressively, the team demonstrated that careful data curation could achieve competitive
32B model performance using just **200 high-quality training samples**—a striking
validation of the "quality over quantity" philosophy.

The system improvements were equally significant: dynamic sequence packing eliminated
memory waste from padding, while data transfer with GPU-Direct RDMA enabled efficient
scaling to 1000+ GPU clusters. These optimizations delivered a **1.5x throughput
improvement** that made large-scale experiments accessible to more research teams.

| Model              | AIME 2024 | AIME 2025 | GPQA-Diamond |
| ------------------ | --------- | --------- | ------------ |
| AReaL-boba-RL-7B   | **61.9**  | **48.3**  | **47.6**     |
| AReaL-boba-SFT-32B | 78.8      | 62.1      | 60.1         |

## AReaL v0.1 (February 2025): The Foundation That Started It All

*📖
[Full Blog Post](https://github.com/areal-project/AReaL/blob/main/blog/AReaL_v0_1.md)*

At its initial release, AReaL's 1.5B model surpassed o1-Preview on mathematical
reasoning in just 40 hours of training. v0.1 also demonstrated emergent thinking
behaviors in Qwen2.5-7B through R1-Zero-style training. As training progressed, models
simultaneously developed longer reasoning chains and higher accuracy—a strong signal
that the system was learning to "think" rather than merely pattern-match.

The technical foundation was elegantly simple: a critic-free PPO variant that traded for
computational efficiency, paired with sparse binary rewards (+5/-5) that let
mathematical correctness drive learning.

| Model Stage   | MATH500  | AIME 2024 | AMC 2023 |
| ------------- | -------- | --------- | -------- |
| Stage 1 (8K)  | 85.7     | 33.2      | 74.7     |
| Stage 2 (16K) | 87.4     | 34.2      | 79.6     |
| Stage 3 (24K) | **88.0** | **40.2**  | **81.2** |

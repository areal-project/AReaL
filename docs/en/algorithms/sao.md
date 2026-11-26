# Single-Rollout Asynchronous Optimization (SAO)

Last updated: Nov 11, 2026

SAO ([arXiv:2607.07508](https://arxiv.org/abs/2607.07508); *Single-Rollout
Asynchronous Optimization for Agentic Reinforcement Learning*) is a **recipe** on
top of stock AReaL PPO: one sample per prompt (G=1), a value critic, and
token-level rejection sampling that masks importance ratios outside a band
(DIS-style). It is **not** a new rollout / training engine.

## What changes vs GRPO

| Knob | Typical GRPO example | SAO example |
|------|----------------------|-------------|
| `gconfig.n_samples` | ≥4 / 8 | **1** |
| Critic | off | **on** (value model colocated with actor) |
| Advantage baseline | group reward norm | batch `adv_norm` (no group) |
| `actor.rejection_sampling` | often upper-only / looser | **mask** band `[0.3, 5.0]` |

This example uses **only stock AReaL APIs** (critic block + `rejection_sampling`).
No engine code changes in this PR.

**Not in this PR:** paper multiple critic passes per actor step (K=2) and
skip-observation GAE. Stock critic is one pass; treat K=2 / skip-obs as follow-ups.

## Reproduce

Use the shared math entrypoint:

| Backend | CMD |
|---------|-----|
| **local** | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_sao.yaml scheduler.type=local` |
| **ray** | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_sao.yaml scheduler.type=ray` |
| **slurm** | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_sao.yaml scheduler.type=slurm` |

Config: [`examples/math/gsm8k_sao.yaml`](https://github.com/areal-project/AReaL/blob/main/examples/math/gsm8k_sao.yaml).

## Reported numbers (honest scope)

Community MATH-hard online runs (not AReaL CI; not paper table scores):

- Model: `Qwen3-4B-Instruct-2507`
- Data: MATH train (~7.5k) mapped to GSM8K-style schema
- Scale: 24 GPUs across three nodes (eight GPUs per node), `total_train_steps=1000`
- Metric: online `acc = correct / (correct + incorrect)` on the training stream
- **Out of scope there:** held-out MATH-500 / AIME

| Setting | G | Critic | DIS band | Steps | Online acc (start → end window) |
|---------|---|--------|----------|-------|----------------------------------|
| SAO (report) | 1 | yes (K=2)† | mask [0.3, 5] | 1000 | ~40% → **~49%** (steps 951–1000) |
| GRPO | 8 | no | no | 268‡ | ~39% → **~15%** |
| GRPO+DIS | 8 | no | yes | 1000 | ~40% → **~57%** |

† Report used K=2; this checked-in yaml is stock one-pass critic until a separate PR.
‡ Bare GRPO early-stopped after collapse. Absolute scores are **not** comparable
across G=1 vs G=8 (8× trajectories per step). Under the same G=1 budget, SAO beat
“DIS only” (~43%) and bare running-mean (~10%) in that report.

The checked-in `gsm8k_sao.yaml` keeps the shared **1.5B / 8-GPU / GSM8K** template so
you can smoke-test the recipe. Matching the table above needs scaling the model,
data, and cluster toward that MATH-hard setup.

## References

- SAO paper: [arXiv:2607.07508](https://arxiv.org/abs/2607.07508)
- AReaL: [arXiv:2505.24298](https://arxiv.org/abs/2505.24298)

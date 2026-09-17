# Second-Moment Trust Policy Optimization (M2PO)

Last updated: Oct 23, 2025

Author: [Jingyuan Ma](https://github.com/tsjyma)

![m2po figure](../figures/m2po.png)

Second-Moment Trust Policy Optimization (M2PO) (Zheng et al., 2025) is an RL method
that achieves stable off-policy training even when data is stale by at least 256 model
updates and matches on-policy performance by constraining the second moment of
importance weights to suppress only extreme outliers while preserving informative
updates.

The first step of M2PO is to compute the second moment: $$
\hat{M_2}=\frac{1}{N}\sum_{i=1}^NM_{2,i}=\frac{1}{N}\sum_{i=1}^N(\log{r_i})^2=\frac{1}{N}\sum_{i=1}^N\left(\log\frac{\pi_\theta
(a_i|s_i)}{\pi_{behav}(a_i|s_i)}\right)^2 $$

The second step is to compute the second-moment mask:

<center>
<img src="../figures/m2po_masking.png" width = "298" height = "217" alt="m2po masking"/>
</center>

The final step is to optimize the objective:

$$ J_{\text{M2PO}}(\theta) =
\frac{1}{\sum_{i=1}^G|o_i|}\sum_{i=1}^G\sum_{t=1}^{|o_i|}M_{i,t}\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}A_{i,t}.
$$

Where $M$ is computed in the second step and

$$ A_{i,t}=\frac{r_i-mean({R_i}_{i=1}^G)}{std({R_i}_{i=1}^G)}. $$

For more details:

- AReaL details: [AReaL paper](https://arxiv.org/abs/2505.24298)

- M2PO details: [M2PO paper](https://arxiv.org/abs/2510.01161)

## Core Parameters

- `actor.m2_threshold`: The threshold for the mean of the second moment, used in
  computing the M2PO mask as $\tau_{M_2}$

## Example Usage

We recommend changing the parameters in the configuration file
(`examples/math/gsm8k_m2po.yaml`).

| Backend   | CMD                                                                                                                         |
| --------- | --------------------------------------------------------------------------------------------------------------------------- |
| **local** | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_m2po.yaml scheduler.type=local --<other_args_to_overwrite>` |
| **ray**   | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_m2po.yaml scheduler.type=ray --<other_args_to_overwrite>`   |
| **slurm** | `python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_m2po.yaml scheduler.type=slurm --<other_args_to_overwrite>` |

## Test Result

![m2po test figure](../figures/m2po_test.png)

In this test, trial names follow these conventions:

- **stale:** the value of `max_head_offpolicyness`
- **dx+dy**: `x` is the number of rollout workers and `y` is the number of training
  workers
- **rollout**: the value of `max_concurrent_rollout`

The GRPO setting is `stale 256 d2+d1 rollout 96`.

The key findings across the trials are as follows:

- The `grad_norm` of GRPO is higher than M2PO, which may cause training instability.
- The evaluation reward of M2PO is higher than GRPO.

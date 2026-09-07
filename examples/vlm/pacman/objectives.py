# SPDX-License-Identifier: Apache-2.0

"""The release's critic-free PPO objectives, independent of training backend."""

from typing import Any

import torch

from areal.api.cli_args import PPOActorConfig
from areal.api.rl_plugins import PPOObjective
from areal.utils.data import KLEstimator, Normalization, TrajBatchMeta


class PacmanObjective(PPOObjective):
    """C1: raw step reward. C2: normalize complete episodes, then weight each equally.

    Every training row contains exactly one generated action token. The
    advantage is its task signal plus the release's proximal-reference KL
    penalty; no value model or cross-game-step GAE is introduced.
    """

    def __init__(self, curriculum: int):
        if curriculum not in (1, 2):
            raise ValueError("curriculum must be 1 or 2")
        self.curriculum = curriculum

    def compute_advantages(
        self, data: dict[str, Any], meta: TrajBatchMeta | None, config: PPOActorConfig
    ) -> dict[str, Any]:
        if (
            "values" in data
            or "teacher_logp" in data
            or "mopd_teacher_logp_sum" in data
        ):
            raise ValueError("Pacman release objectives do not use a critic or teacher")
        if config.adv_norm is not None or config.ppo_n_minibatches != 1:
            raise ValueError(
                "Release objectives require adv_norm=null and ppo_n_minibatches=1"
            )
        if (
            config.use_sapo_loss
            or config.use_cispo_loss
            or config.m2_threshold is not None
        ):
            raise ValueError("Release objectives require the original PPO surrogate")
        if config.mask_no_eos_with_zero or config.overlong_reward_penalty:
            raise ValueError(
                "Single-token actions must retain their reward without EOS"
            )
        if not config.use_decoupled_loss or not config.recompute_logprob:
            raise ValueError(
                "Release PPO requires decoupled loss and proximal recomputation"
            )
        mask = torch.roll(data["loss_mask"].float(), -1, -1)
        mask[:, -1] = 0
        if mask.ndim != 2 or not torch.all(mask.sum(-1) == 1):
            raise ValueError(
                "Every real decision must contain exactly one output token"
            )
        rewards = data["rewards"].float()
        if rewards.shape != (mask.shape[0],) or not torch.isfinite(rewards).all():
            raise ValueError("Every decision must carry one finite task reward")
        score = (rewards + config.reward_bias) * config.reward_scaling
        if self.curriculum == 1:
            if config.reward_norm is not None:
                raise ValueError("C1 is critic-free PPO without group normalization")
            if any(key in data for key in ("episode_ids", "episode_returns")):
                raise ValueError("C1 must use step-local rewards")
        else:
            score, weights = self._normalize_episodes(data, mask, score, meta, config)
            data["loss_reduction_weights"] = weights
        score = score.clamp(-config.reward_clip, config.reward_clip)
        prox = data.get("prox_logp")
        ref = data.get("ref_logp")
        if prox is None or (config.kl_ctl != 0 and ref is None):
            raise ValueError(
                "Release KL requires proximal and reference log-probabilities"
            )
        if ref is None:
            ref = torch.zeros_like(prox)
        for value in (prox, ref):
            if (
                value.shape != mask.shape
                or not torch.isfinite(value[mask.bool()]).all()
            ):
                raise ValueError("Policy log-probabilities are missing or non-finite")
        kl_reward = -config.kl_ctl * KLEstimator(config.kl_estimator)(
            prox * mask, ref * mask
        )
        kl_reward *= mask
        advantages = score.unsqueeze(-1) * mask + kl_reward
        behavior = torch.roll(data["logprobs"], -1, -1) * mask
        if not torch.isfinite(behavior).all():
            raise ValueError("Behavior log-probabilities must be finite")
        data.update(
            original_rewards=rewards.clone(),
            rewards=score,
            advantages=advantages,
            returns=advantages.clone(),
            kl_rewards=kl_reward,
            tot_rewards=advantages.clone(),
            logprobs=behavior,
            loss_mask=mask,
            is_truncated=torch.ones(
                mask.shape[0], dtype=torch.bool, device=mask.device
            ),
        )
        return data

    @staticmethod
    def _normalize_episodes(
        data: dict[str, Any],
        mask: torch.Tensor,
        score: torch.Tensor,
        meta: TrajBatchMeta | None,
        config: PPOActorConfig,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm = config.reward_norm
        if (
            norm is None
            or norm.mean_level != "group"
            or norm.std_level != "group"
            or norm.group_size != 12
        ):
            raise ValueError(
                "C2 requires group normalization over 12 complete episodes"
            )
        if meta is None or sum(meta.traj_group_sizes) != mask.shape[0]:
            raise ValueError("C2 requires intact initial-state trajectory groups")
        ids = data.get("episode_ids")
        returns = data.get("episode_returns")
        sizes = data.get("episode_group_sizes")
        if any(
            value is None or value.shape != score.shape
            for value in (ids, returns, sizes)
        ):
            raise ValueError(
                "C2 requires episode IDs, returns and group sizes for each decision"
            )
        if not torch.equal(data["rewards"].float(), returns.float()):
            raise ValueError("C2 decisions must carry their complete episode return")
        output, weights = [], []
        offset = 0
        for count in meta.traj_group_sizes:
            sl = slice(offset, offset + count)
            group_ids = ids[sl]
            if not torch.all(sizes[sl] == 12):
                raise ValueError("C2 group-size metadata must equal 12")
            starts = torch.ones_like(group_ids, dtype=torch.bool)
            starts[1:] = group_ids[1:] != group_ids[:-1]
            first = starts.nonzero().flatten()
            if first.numel() != 12 or group_ids[first].unique().numel() != 12:
                raise ValueError(
                    "C2 requires 12 distinct, contiguous, complete episodes per initial state"
                )
            index = starts.long().cumsum(0) - 1
            episode_scores = score[sl][first]
            if not torch.equal(score[sl], episode_scores[index]):
                raise ValueError("Episode return changed between its decisions")
            normalized = Normalization(norm)(episode_scores, group_sizes=[12])
            output.append(normalized[index])
            counts = torch.zeros(12, device=mask.device, dtype=torch.float32)
            counts.scatter_add_(0, index, mask[sl].sum(-1))
            if not torch.all(counts > 0):
                raise ValueError("Every episode must contain a policy decision")
            weights.append(mask[sl] / counts[index].unsqueeze(-1))
            offset += count
        return torch.cat(output), torch.cat(weights)

    def loss_weight(self, data: dict[str, Any]) -> torch.Tensor:
        if self.curriculum == 1:
            return super().loss_weight(data)
        weights = data.get("loss_reduction_weights")
        if weights is None or weights.shape != data["loss_mask"].shape:
            raise ValueError("C2 episode weights were lost during batch preparation")
        return (weights * data["loss_mask"]).sum()

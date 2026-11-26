# SPDX-License-Identifier: Apache-2.0

"""Estimate the cost of publishing Actor weights to Rollout."""

from __future__ import annotations

from ..rollout.cost_model import memory_per_rank


def publication_seconds(
    model, hardware, parallel, replicas, *, weight_bytes, hbm_efficiency, bandwidth
):
    source = weight_bytes / parallel.pp * (1 - 1 / parallel.tp)
    if model.mlp_type == "moe":
        source += (
            model.expert_parameters_per_layer
            * model.layers
            * 2
            / parallel.pp
            * (1 - 1 / (parallel.ep * parallel.etp))
        )
    sent = max_endpoint = 0.0
    for config in replicas:
        weights, _ = memory_per_rank(model, config)
        sent += weights * config.world_size
        max_endpoint = max(max_endpoint, weights)
    copy_seconds = (source + max_endpoint) / (hardware.hbm_gbps * 1e9 * hbm_efficiency)
    return copy_seconds + sent / (bandwidth * 1e9)

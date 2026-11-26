# SPDX-License-Identifier: Apache-2.0

"""Saturated Actor throughput for the predicted rollout mix."""

from dataclasses import replace
from math import isfinite

from .cost_model import actor_config


def estimate_actor_capacity(models, template, role, pattern, publication):
    """Return the capacity of a legal representative Actor batch in episodes/s."""
    execution = replace(template, actor_backend=role.backend)
    config = actor_config(models.actor_template, execution)
    model = models.actor_model
    batch = tuple(episode.completed_turns for episode in pattern.templates)
    try:
        estimate = model.estimate_batch(config, batch)
        duration = estimate.duration_seconds()
        if (
            not isfinite(duration)
            or duration <= 0
            or estimate.peak_gib > model.hardware.memory_gb * 1e9 / 2**30
        ):
            return None
        return len(batch) / (duration + publication)
    except ValueError:
        return None

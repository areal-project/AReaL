# SPDX-License-Identifier: Apache-2.0

"""Saturated Actor throughput for the predicted rollout mix."""


def actor_capacity(templates, batch_size, batch_seconds, publication_seconds):
    """Estimate episodes per second, including one publication per PPO update."""
    if not templates:
        return None
    count = len(templates)
    weighted = 0.0
    for offset in range(0, count, batch_size):
        batch = tuple(templates[(offset + i) % count] for i in range(batch_size))
        weighted += min(batch_size, count - offset) * batch_seconds(batch)
    return batch_size / (weighted / count + publication_seconds)

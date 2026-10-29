# SPDX-License-Identifier: Apache-2.0

"""Staleness-aware capacity manager for rollout generation.

This module provides the StalenessManager class which manages capacity
and staleness constraints for asynchronous rollout generation in RL training.
"""

from threading import Lock
from typing import Protocol

from areal.api import RolloutStat


class VersionProvider(Protocol):
    def get_version(self) -> int:
        raise NotImplementedError()


class StalenessManager:
    """Manages rollout capacity based on staleness and concurrency constraints.

    The manager ensures that:
    1. The number of concurrent rollouts doesn't exceed the configured maximum
    2. Rollouts don't become too stale (off-policy) by limiting acceptance based on
       the current model version and maximum allowed offpolicyness

    Parameters
    ----------
    version_provider : VersionProvider
        Provider for current model version (e.g., InferenceEngine)
    max_concurrent_rollouts : int
        Maximum number of concurrent rollouts allowed
    consumer_batch_size : int
        Expected batch size for consuming rollouts during training
    max_staleness : int
        Maximum allowed offpolicyness (version difference) for rollouts
    """

    def __init__(
        self,
        version_provider: VersionProvider,
        max_concurrent_rollouts: int,
        consumer_batch_size: int,
        max_staleness: int,
    ):
        """Initialize the staleness manager.

        Parameters
        ----------
        version_provider : VersionProvider
            Provider for current model version (e.g., InferenceEngine)
        max_concurrent_rollouts : int
            Maximum number of concurrent rollouts allowed
        consumer_batch_size : int
            Expected batch size for consuming rollouts during training
        max_staleness : int
            Maximum allowed offpolicyness (version difference) for rollouts
        """
        self.version_provider = version_provider
        self.max_concurrent_rollouts = max_concurrent_rollouts
        self.consumer_batch_size = consumer_batch_size
        self.max_staleness = max_staleness

        # Thread-safe access to rollout statistics
        self.lock = Lock()
        self.rollout_stat = RolloutStat()

    def get_pending_limit(self) -> int:
        """Get the maximum number of pending rollouts allowed.

        Returns
        -------
        int
            Maximum number of pending rollouts (enqueued)
        """
        return (self.max_staleness + 1) * self.consumer_batch_size

    def get_capacity(self) -> int:
        """Calculate available capacity for new rollouts.

        Considers both concurrency limits and staleness constraints.
        Obtains current model version from version_provider.

        Returns
        -------
        int
            Number of new rollout slots available. Can be negative if over capacity.

        Notes
        -----
        The staleness control formula is:
        max_samples = (max_staleness + current_version + 1) * consumer_batch_size
        capacity = min(concurrency_limit, max_samples - current_samples)

        This ensures that by the time samples are consumed, they won't exceed
        the maximum allowed staleness.
        """
        with self.lock:
            current_version = self.version_provider.get_version()
            # Calculate concurrency-based capacity
            max_concurrent_rollouts = max(1, self.max_concurrent_rollouts)
            concurrency_capacity = max_concurrent_rollouts - self.rollout_stat.running

            # Calculate staleness-based capacity
            ofp = self.max_staleness
            sample_cnt = self.rollout_stat.accepted + self.rollout_stat.running
            consumer_bs = max(1, self.consumer_batch_size)
            staleness_capacity = (ofp + current_version + 1) * consumer_bs - sample_cnt

            # Return the minimum of both constraints
            capacity = min(concurrency_capacity, staleness_capacity)
            return capacity

    def on_version_recovered(self, version: int) -> None:
        """Adjust accepted count after checkpoint recovery.

        When a checkpoint is recovered, the version jumps from 0 to the
        recovered value. Without adjusting accepted, the capacity formula
        yields (max_staleness + version + 1) * batch_size instead of the
        intended (max_staleness + 1) * batch_size, causing a burst of
        submissions and unbounded staleness growth.

        Expected to be called during trainer init, before any rollouts are
        submitted, so running == 0. If running > 0 (unlikely in practice),
        accepted is still set correctly and the capacity formula remains
        bounded — (max_staleness + 1) * consumer_bs - running.
        """
        with self.lock:
            consumer_bs = max(1, self.consumer_batch_size)
            self.rollout_stat.accepted = version * consumer_bs

    def on_rollout_enqueued(self) -> None:
        """Callback when a rollout is enqueued as a pending input task.

        Thread-safe method to increment the enqueued counters.
        """
        with self.lock:
            self.rollout_stat.enqueued += 1

    def on_rollout_submitted(self) -> None:
        """Callback when a rollout is submitted for execution.

        Thread-safe method to decrement enqueued counter and increment running counters.
        """
        with self.lock:
            self.rollout_stat.enqueued -= 1
            self.rollout_stat.running += 1

    def on_rollout_accepted(self) -> None:
        """Callback when a rollout completes successfully and is accepted.

        Thread-safe method to decrement running counter and increment accepted counter.
        """
        with self.lock:
            self.rollout_stat.running -= 1
            self.rollout_stat.accepted += 1

    def on_rollout_rejected(self) -> None:
        """Callback when a rollout completes but is rejected.

        Thread-safe method to decrement running counter and increment rejected counter.
        """
        with self.lock:
            self.rollout_stat.running -= 1
            self.rollout_stat.rejected += 1

    def get_stats(self) -> RolloutStat:
        """Get a snapshot of current rollout statistics.

        Returns
        -------
        RolloutStat
            Current rollout statistics (enqueued, accepted, running)
        """
        with self.lock:
            return RolloutStat(
                accepted=self.rollout_stat.accepted,
                enqueued=self.rollout_stat.enqueued,
                rejected=self.rollout_stat.rejected,
                running=self.rollout_stat.running,
            )

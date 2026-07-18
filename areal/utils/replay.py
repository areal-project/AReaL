# SPDX-License-Identifier: Apache-2.0

"""No-op rollout stub used by the trainer in trajectory replay mode.

When ``trajectory_debug.replay_rollout_data`` is enabled, the trainer loads
previously dumped trajectories from disk instead of running inference. In this
mode there is no real inference engine, so :class:`NoOpRollout` stands in for it
and exposes the same interface as a real rollout engine. This lets
``PPOTrainer.train()`` call ``pause``/``resume``/``set_version``/``offload`` and
friends without branching on replay mode at every call site.

Note that proxy-related methods (``start_proxy``/``start_proxy_gateway``) are
intentionally NOT provided: replay mode never initializes proxy workers (the
trainer guards those calls behind ``if not self._replay_mode``), so adding them
here would only create dead code that must be kept in sync with the real engine.
"""


class NoOpRollout:
    """Stub replacing the inference engine in replay mode.

    Provides the same interface as a real rollout engine so that
    ``PPOTrainer.train()`` can call pause/resume/set_version without
    branching on replay mode at every call site.
    """

    staleness_manager = None
    workflow_executor = None

    def pause(self):
        pass

    def resume(self):
        pass

    def destroy(self):
        pass

    def set_version(self, version):  # noqa: ARG002
        pass

    def offload(self):
        pass

    def onload(self):
        pass

    def config_perf_tracer(self, *args, **kwargs):  # noqa: ARG002
        pass

    def save_perf_tracer(self, *args, **kwargs):  # noqa: ARG002
        pass

    def export_stats(self):
        return {}

    async def pause_generation(self):
        pass

    async def continue_generation(self):
        pass

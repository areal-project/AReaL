# SPDX-License-Identifier: Apache-2.0

"""Agent Service Guard — process supervisor backed by the shared guard.

Pure pass-through to ``areal.infra.rpc.guard``.  All orchestration logic
(launching Router, Gateway, Worker+DataProxy pairs) lives in the
:mod:`~areal.experimental.agent_service.controller` module.

Quick start::

    python -m areal.experimental.agent_service.guard \\
        --experiment-name demo --trial-name run0 \\
        --role agent-guard --worker-index 0
"""

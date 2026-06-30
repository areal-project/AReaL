# SPDX-License-Identifier: Apache-2.0

"""Hermes example: an in-process per-session :class:`AgentRunnable`.

Hosts the Hermes agent runtime that the Worker loads via ``agent_cls_path``
(``examples.hermes.hermes.HermesAgent``).  Unlike the OpenClaw
example (one Gateway subprocess per session), Hermes is a Python library, so
this runtime instantiates one Hermes ``AIAgent`` per RL session directly inside
the Worker process.
"""

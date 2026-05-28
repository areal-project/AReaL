# SPDX-License-Identifier: Apache-2.0

"""AReaL operator CLI — companion to the v2 microservice control plane.

This package exposes a single ``areal`` console-script that drives the v2
service gateways (inference / agent / training / weight-update) from a
shell, rather than from a Python script that has to instantiate the
matching controller. It is intentionally light at import time so that
adding a verb in a follow-up PR does not pull torch / ray / megatron /
sglang / vllm into the parser-construction path.

The full per-verb design surface is tracked in the upstream design
discussion issue.
"""
